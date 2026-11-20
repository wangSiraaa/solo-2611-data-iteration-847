"""Stateful ingestion library under a fixed reader contract.

Responsibilities are deliberately separated from the compatibility service:

  * ``check_compatibility`` (row-free) decides whether the *contract pair* is
    safe.  A SAFE verdict is **not** acceptance and never touches row state.
  * This library owns the target row state, applies revision rules, and makes
    one atomic decision per file.

State model
-----------
For each ``reader_version`` there is an isolated row store::

    rows[row_id] = RowState(revision, canonical_values, encoded_values,
                            sources, writer_values)

The *business content* of a row is its canonical reader value: a dict of
``int`` / ``Fraction`` / ``str`` / ``None``.  Encodings and display names are
not part of identity, so ``1`` and ``"2/2"`` (both reader DECIMAL ``1``)
compare equal, and a writer renaming ``old_amount`` to ``amount`` while keeping
``field_id`` changes nothing.

Revision rules (per file, for each ``row_id``)
---------------------------------------------
  * missing ``revision`` means ``1``; revisions are positive integers.
  * the highest revision in the file wins;
  * every occurrence is first validated for writer-domain legality — an
    illegal low-revision row rejects the file, it is not silently ignored;
  * a lower revision than the stored one is validated and then ignored;
  * same revision + same canonical value is a replay (no change);
  * same revision + different canonical value is a conflict that rejects the
    **entire file**, even if a higher revision appears later in the same file.

Atomicity
---------
All validation and conflict detection happen on a staged copy.  The staged
state and receipt are published in a single assignment guarded by a lock, so a
failed file leaves no new rows and no receipt.

Idempotency
-----------
Receipts are keyed by ``(reader_version, file_id)``.  Re-submitting a file whose
``writer_version`` and canonical ``(row_id, revision, values)`` content is
identical returns the original receipt (``"replay": true``).  Same ``file_id``
with different content is rejected.

Costs (documented, not hidden)
------------------------------
* Ingest of a file with ``m`` rows and ``k`` columns is O(m*k) plus a
  fingerprint over the file; conflict checks are dictionary lookups.
* Publish builds a fresh snapshot copy O(n*k) and sorts by ``row_id``
  O(n log n); no versioned-view recomputation is needed because each accepted
  row already stores its adapted values and provenance.
* Reader versions are isolated maps, so distinct reader contracts never
  rescan each other's rows.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .adapter import (
    AdaptationResult,
    CanonicalRow,
    adapt_rows,
)
from .compat import check_compatibility
from .contract import Contract
from .errors import ContractError
from .types import rational_string


@dataclass
class RowState:
    row_id: str
    revision: int
    canonical: dict
    encoded: dict
    sources: dict
    writer_values: dict

    def clone(self) -> "RowState":
        return RowState(
            row_id=self.row_id,
            revision=self.revision,
            canonical=dict(self.canonical),
            encoded=dict(self.encoded),
            sources=_deep_copy_json(self.sources),
            writer_values={k: _copy_value(v) for k, v in self.writer_values.items()},
        )


@dataclass
class Receipt:
    file_id: str
    writer_version: str
    reader_version: str
    fingerprint: str
    sequence: int
    applied: list[dict] = field(default_factory=list)
    ignored: list[dict] = field(default_factory=list)
    replayed_rows: list[str] = field(default_factory=list)
    row_ids: list[str] = field(default_factory=list)

    def to_json(self, *, replay: bool = False) -> dict:
        return {
            "status": "ACCEPTED",
            "file_id": self.file_id,
            "writer_version": self.writer_version,
            "reader_version": self.reader_version,
            "fingerprint": self.fingerprint,
            "sequence": self.sequence,
            "row_ids": list(self.row_ids),
            "applied": _deep_copy_json(self.applied),
            "ignored": _deep_copy_json(self.ignored),
            "replayed_rows": list(self.replayed_rows),
            "replay": replay,
        }


@dataclass
class IngestResult:
    status: str  # "ACCEPTED" | "REJECTED"
    file_id: str | None
    writer_version: str
    reader_version: str
    receipt: dict | None = None
    errors: list[str] = field(default_factory=list)
    adaptation: AdaptationResult | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "ACCEPTED"

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "file_id": self.file_id,
            "writer_version": self.writer_version,
            "reader_version": self.reader_version,
            "receipt": self.receipt,
            "errors": list(self.errors),
        }


class IngestionLibrary:
    """Fixed-reader-contract target state with atomic file ingestion."""

    def __init__(self, reader: Contract):
        if reader.role != "reader":
            raise ContractError("ingestion requires a reader contract")
        self.reader = reader
        self._lock = threading.RLock()
        # reader_version -> {row_id: RowState}
        self._rows: dict[str, dict[str, RowState]] = {reader.version: {}}
        # (reader_version, file_id) -> Receipt
        self._receipts: dict[tuple[str, str], Receipt] = {}
        self._sequence = 0

    # -- read-only views ---------------------------------------------------

    @property
    def reader_version(self) -> str:
        return self.reader.version

    def row_count(self, reader_version: str | None = None) -> int:
        rv = reader_version or self.reader.version
        with self._lock:
            return len(self._rows.get(rv, {}))

    def get_row(self, row_id: str, reader_version: str | None = None) -> RowState | None:
        rv = reader_version or self.reader.version
        with self._lock:
            row = self._rows.get(rv, {}).get(row_id)
            return row.clone() if row else None

    def snapshot(self, reader_version: str | None = None) -> list[RowState]:
        rv = reader_version or self.reader.version
        with self._lock:
            return [
                self._rows[rv][rid].clone()
                for rid in sorted(self._rows.get(rv, {}))
            ]

    # -- write path --------------------------------------------------------

    def ingest(
        self,
        writer: Contract,
        document: Any,
        *,
        reader_version: str | None = None,
    ) -> IngestResult:
        """Ingest one file atomically.  Returns a receipt or a rejection."""
        rv = reader_version or self.reader.version
        if rv != self.reader.version:
            return IngestResult(
                status="REJECTED",
                file_id=document.get("file_id") if isinstance(document, dict) else None,
                writer_version=writer.version,
                reader_version=rv,
                errors=[
                    f"this library is fixed to reader {self.reader.version!r}; "
                    f"reader {rv!r} requires a separate library instance"
                ],
            )
        with self._lock:
            return self._ingest_locked(writer, self.reader, document, rv)

    def _ingest_locked(
        self, writer: Contract, reader: Contract, document: Any, rv: str
    ) -> IngestResult:
        def reject(file_id, errors, adaptation=None):
            return IngestResult(
                status="REJECTED",
                file_id=file_id,
                writer_version=writer.version,
                reader_version=rv,
                errors=list(errors),
                adaptation=adaptation,
            )

        if not isinstance(document, dict):
            return reject(None, ["file document must be a JSON object"])
        unknown = set(document) - {
            "file_id",
            "writer_version",
            "reader_version",
            "rows",
        }
        if unknown:
            return reject(None, [f"unexpected file keys {sorted(unknown)}"])

        file_id = document.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            return reject(None, ["file_id must be a non-empty string"])
        if document.get("writer_version") != writer.version:
            return reject(
                file_id,
                [
                    f"file declares writer {document.get('writer_version')!r}, "
                    f"fixed writer is {writer.version!r}"
                ],
            )
        if document.get("reader_version") != rv:
            return reject(
                file_id,
                [
                    f"file declares reader {document.get('reader_version')!r}, "
                    f"fixed reader is {rv!r}"
                ],
            )

        # 1. Contract safety is checked first, but a SAFE result is only a
        #    precondition - it is not acceptance.
        report = check_compatibility(writer, reader)
        if not report.safe:
            adaptation = AdaptationResult(
                status="INCOMPATIBLE",
                writer_version=writer.version,
                reader_version=rv,
                records=None,
                errors=[],
                report=report,
            )
            errors = [f"column {c.field_id}: {c.reasons[0]}" for c in report.columns if not c.safe]
            errors += list(report.errors)
            return reject(file_id, errors, adaptation)

        rows_raw = document.get("rows")
        if not isinstance(rows_raw, list):
            return reject(file_id, ["rows must be a list"])
        if not rows_raw:
            return reject(file_id, ["file contains no rows"])

        # 2. Validate every row against the writer domain and adapt it.
        #    Duplicates are allowed here because revision handling needs them.
        adaptation = adapt_rows(
            writer, reader, rows_raw, allow_duplicate_row_ids=True
        )
        if adaptation.status != "ADAPTED":
            return reject(
                file_id,
                [f"{e.message}" for e in adaptation.errors] or ["adaptation failed"],
                adaptation,
            )
        canonical_rows = adaptation.canonical_rows

        # 3. Structural check: duplicate row_id with no revision field is a
        #    hard input error (two default-1 copies can only be a replay or a
        #    conflict, and both are handled below by revision logic; but a
        #    file may not carry two different rows with the same identity and
        #    no revision marker - treat as revision 1 and let conflict logic
        #    decide, which is exactly what the rules require).

        # 4. Group by row_id and reject any within-file same-revision conflict
        #    outright.  The winner per row is the highest revision present.
        grouped: dict[str, list[CanonicalRow]] = {}
        for row in canonical_rows:
            grouped.setdefault(row.row_id, []).append(row)

        conflicts: list[str] = []
        for row_id, occurrences in grouped.items():
            _check_within_file_conflicts(row_id, occurrences, conflicts)
        if conflicts:
            return reject(file_id, conflicts, adaptation)

        # 5. File fingerprint over the canonical, representation-independent
        #    content (deduped per (row_id, revision)).
        fingerprint = _file_fingerprint(writer.version, rv, grouped)

        # 6. Exact replay: same file_id, same contract/content -> old receipt.
        existing_receipt = self._receipts.get((rv, file_id))
        if existing_receipt is not None:
            if existing_receipt.fingerprint == fingerprint:
                return IngestResult(
                    status="ACCEPTED",
                    file_id=file_id,
                    writer_version=writer.version,
                    reader_version=rv,
                    receipt=existing_receipt.to_json(replay=True),
                    errors=[],
                    adaptation=adaptation,
                )
            return reject(
                file_id,
                [
                    f"file_id {file_id!r} was already accepted with different "
                    f"canonical content or contract"
                ],
                adaptation,
            )

        # 7. Revision resolution against the stored target snapshot.  Stage a
        #    clone; commit only at the very end.
        #
        #    Every occurrence in the file is examined, not only the winner:
        #    a same-revision value that differs from what is already stored is
        #    a conflict that rejects the whole file, even when a higher
        #    revision for the same row appears later in the file.  Lower
        #    revisions were already type-validated above and are then ignored.
        current = self._rows.setdefault(rv, {})
        staged: dict[str, RowState] = {}
        applied: list[dict] = []
        ignored: list[dict] = []
        replayed: list[str] = []

        for row_id in sorted(grouped):
            occurrences = grouped[row_id]
            stored = current.get(row_id)
            winner = max(occurrences, key=lambda r: r.revision)

            if stored is not None:
                stored_revision = stored.revision
                row_conflicted = False
                for occ in occurrences:
                    if occ.revision != stored_revision:
                        continue
                    if not _canonical_equal(occ.domain_values, stored.canonical):
                        conflicts.append(
                            f"row {row_id!r} revision {occ.revision}: incoming "
                            f"canonical value differs from the accepted value"
                        )
                        row_conflicted = True
                if row_conflicted:
                    continue

            if stored is None:
                staged[row_id] = _rowstate_from(winner)
                applied.append(
                    {"row_id": row_id, "revision": winner.revision, "action": "insert"}
                )
            elif winner.revision > stored.revision:
                staged[row_id] = _rowstate_from(winner)
                applied.append(
                    {
                        "row_id": row_id,
                        "revision": winner.revision,
                        "previous_revision": stored.revision,
                        "action": "supersede",
                    }
                )
            elif winner.revision == stored.revision:
                # Equal canonical value (a difference would be a conflict above)
                # and any lower occurrences are implied by the winner.
                replayed.append(row_id)
            else:  # winner.revision < stored.revision
                ignored.append(
                    {
                        "row_id": row_id,
                        "revision": winner.revision,
                        "stored_revision": stored.revision,
                        "action": "ignored_lower_revision",
                    }
                )

        # A same-revision conflict against the target rejects the file, even
        # if a higher revision for the same row was also present.
        if conflicts:
            return reject(file_id, conflicts, adaptation)

        # 8. Commit atomically: staged rows plus the new receipt.
        self._sequence += 1
        new_rows = dict(current)
        new_rows.update(staged)
        self._rows[rv] = new_rows
        receipt = Receipt(
            file_id=file_id,
            writer_version=writer.version,
            reader_version=rv,
            fingerprint=fingerprint,
            sequence=self._sequence,
            applied=applied,
            ignored=ignored,
            replayed_rows=replayed,
            row_ids=sorted(grouped),
        )
        self._receipts[(rv, file_id)] = receipt
        return IngestResult(
            status="ACCEPTED",
            file_id=file_id,
            writer_version=writer.version,
            reader_version=rv,
            receipt=receipt.to_json(replay=False),
            errors=[],
            adaptation=adaptation,
        )

    # -- publication -------------------------------------------------------

    def publish(self, reader_version: str | None = None) -> dict:
        """Return ``reader-records/v1`` from the real target row snapshot.

        Each row keeps its revision, adapted values, and the original input /
        adaptation-plan provenance in ``sources``.
        """
        rv = reader_version or self.reader.version
        with self._lock:
            store = self._rows.get(rv, {})
            rows = [
                {
                    "row_id": state.row_id,
                    "revision": state.revision,
                    "values": dict(state.encoded),
                    "sources": _deep_copy_json(state.sources),
                }
                for state in (store[rid] for rid in sorted(store))
            ]
        return {
            "format": "reader-records/v1",
            "reader_version": rv,
            "rows": rows,
        }

    def numeric_totals(self, reader_version: str | None = None) -> dict[str, str]:
        rv = reader_version or self.reader.version
        totals: dict[str, Fraction] = {}
        for reader_field in self.reader.fields:
            if reader_field.type.kind != "TEXT":
                totals[reader_field.field_id] = Fraction(0)
        for state in self.snapshot(rv):
            for field_id, value in state.canonical.items():
                if field_id in totals and value is not None:
                    totals[field_id] += value
        return {k: rational_string(v) for k, v in totals.items()}

    def receipt_for(self, file_id: str, reader_version: str | None = None) -> dict | None:
        rv = reader_version or self.reader.version
        with self._lock:
            receipt = self._receipts.get((rv, file_id))
            return receipt.to_json() if receipt else None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reader_for(version: str) -> Contract:  # pragma: no cover - unreachable
    raise ContractError(
        f"reader version {version!r} is not available in this library; create a "
        f"separate IngestionLibrary for each fixed reader contract"
    )


def _check_within_file_conflicts(
    row_id: str, occurrences: list[CanonicalRow], conflicts: list[str]
) -> None:
    """A file may repeat a (row_id, revision) only with the same canonical value."""
    by_revision: dict[int, CanonicalRow] = {}
    for occ in occurrences:
        prior = by_revision.get(occ.revision)
        if prior is None:
            by_revision[occ.revision] = occ
        elif not _canonical_equal(prior.domain_values, occ.domain_values):
            conflicts.append(
                f"row {row_id!r} revision {occ.revision}: file contains two "
                f"different canonical values at the same revision"
            )


def _rowstate_from(row: CanonicalRow) -> RowState:
    return RowState(
        row_id=row.row_id,
        revision=row.revision,
        canonical={k: _copy_value(v) for k, v in row.domain_values.items()},
        encoded=dict(row.values),
        sources=_deep_copy_json(row.sources),
        writer_values={k: _copy_value(v) for k, v in row.writer_values.items()},
    )


def _canonical_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for key in a:
        left, right = a[key], b[key]
        left_num = isinstance(left, (int, Fraction)) and not isinstance(left, bool)
        right_num = isinstance(right, (int, Fraction)) and not isinstance(right, bool)
        if left_num or right_num:
            if not (left_num and right_num) or Fraction(left) != Fraction(right):
                return False
        elif left != right:
            return False
    return True


def _copy_value(value: Any) -> Any:
    return Fraction(value) if isinstance(value, Fraction) else value


def _tagged_value(value: Any) -> Any:
    if value is None:
        return ["NULL"]
    if isinstance(value, Fraction):
        return ["NUM", rational_string(value)]
    if isinstance(value, bool):  # pragma: no cover - never stored
        return ["BOOL", value]
    if isinstance(value, int):
        return ["NUM", str(value)]
    if isinstance(value, str):
        return ["TEXT", value]
    raise ContractError(f"cannot fingerprint value of type {type(value).__name__}")


def _file_fingerprint(
    writer_version: str, reader_version: str, grouped: dict[str, list[CanonicalRow]]
) -> str:
    """Canonical content hash: representation-independent, revision-aware."""
    entries = []
    for row_id in sorted(grouped):
        seen_revisions: dict[int, Any] = {}
        for occ in grouped[row_id]:
            tagged = {k: _tagged_value(v) for k, v in occ.domain_values.items()}
            prior = seen_revisions.get(occ.revision)
            if prior is None:
                seen_revisions[occ.revision] = tagged
            elif prior != tagged:
                # Within-file conflict is detected elsewhere; include a marker
                # so two different conflicting files do not share a receipt.
                seen_revisions[occ.revision] = ["CONFLICT"] + sorted(
                    [prior, tagged], key=repr
                )
        entries.append(
            [
                row_id,
                [[rev, seen_revisions[rev]] for rev in sorted(seen_revisions)],
            ]
        )
    payload = json.dumps(
        {
            "writer_version": writer_version,
            "reader_version": reader_version,
            "rows": entries,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deep_copy_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy_json(v) for v in value]
    return value
