"""File adaptation: validate every row against the writer domain, then apply
the *actual* compatibility plan to produce ``reader-records/v1``.

Design points, all deliberate:

  * The writer declaration and reader contract are fixed for a run; a file
    that declares other versions is rejected.
  * Every row is checked to lie inside the declared writer domains *before*
    conversion.  A value that merely parses is not enough; it must be legal
    for the declared type (range, scale, nullability, JSON encoding).
  * Conversion errors are never silently dropped.  A malformed row rejects
    the whole file and the error list names every offending row/field.
  * The emitted numeric values are exact minimal rational strings, so the
    supplied ``reader_client.py`` can re-validate them against the contract.
  * Original row ids, source field ids and the conversion basis are kept in
    ``sources`` for provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .compat import ColumnPlan, CompatibilityReport, check_compatibility
from .contract import Contract
from .errors import ContractError
from .types import (
    IntType,
    TextType,
    decode_writer_value,
    encode_reader_value,
    is_legal_value,
    rational_string,
    type_to_json,
)

RECORDS_FORMAT = "reader-records/v1"


@dataclass
class RowError:
    row_id: Any
    index: int | None
    message: str

    def to_json(self) -> dict:
        return {"row_id": self.row_id, "index": self.index, "message": self.message}


@dataclass
class CanonicalRow:
    """A successfully adapted row with both encoded and canonical forms.

    ``values`` are the ``reader-records/v1`` encodings; ``domain_values`` are
    representation-independent Python values (int / Fraction / str / None)
    used for revision equality.  ``writer_values`` are the decoded writer
    values used for the file fingerprint.
    """

    row_id: str
    revision: int
    values: dict
    domain_values: dict
    sources: dict
    writer_values: dict


@dataclass
class AdaptationResult:
    status: str  # "ADAPTED" | "INCOMPATIBLE" | "INVALID"
    writer_version: str
    reader_version: str
    records: dict | None
    errors: list[RowError]
    report: CompatibilityReport
    canonical_rows: list[CanonicalRow] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "writer_version": self.writer_version,
            "reader_version": self.reader_version,
            "errors": [e.to_json() for e in self.errors],
            "records": self.records,
            "compatibility": self.report.to_json(),
        }


def parse_row_entry(raw: Any, index: int) -> tuple[str, int, dict]:
    """Validate the envelope of one input row.

    A row is ``{"row_id": str, "revision": int>=1 (optional, default 1),
    "values": {field_id: raw}}``.
    """
    if not isinstance(raw, dict):
        raise ContractError(f"row #{index}: must be an object")
    row_id = raw.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise ContractError(f"row #{index}: row_id must be a non-empty string")
    revision = raw.get("revision", 1)
    if type(revision) is not int or revision < 1:
        raise ContractError(f"row {row_id!r}: revision must be a positive integer")
    values = raw.get("values")
    if not isinstance(values, dict):
        raise ContractError(f"row {row_id!r}: values must be an object")
    unknown = set(raw) - {"row_id", "revision", "values"}
    if unknown:
        raise ContractError(f"row {row_id!r}: unexpected keys {sorted(unknown)}")
    return row_id, revision, values


def validate_and_decode_row(
    writer: Contract, row_id: str, values: dict
) -> tuple[dict, list[str]]:
    """Decode one row into abstract writer values; collect all field errors."""
    errors: list[str] = []
    declared = writer.field_by_id()
    unknown = set(values) - set(declared)
    for key in sorted(unknown):
        errors.append(f"row {row_id!r}: value for undeclared writer field {key!r}")
    decoded: dict[str, Any] = {}
    for field_id, wf in declared.items():
        if field_id not in values:
            if wf.type.nullable:
                decoded[field_id] = None
                continue
            errors.append(
                f"row {row_id!r}: missing non-nullable writer field {field_id!r}"
            )
            continue
        try:
            decoded[field_id] = decode_writer_value(
                wf.type, values[field_id], where=f"row {row_id!r}.{field_id}"
            )
        except Exception as exc:  # DomainError / ContractError
            errors.append(str(exc))
    return decoded, errors


def canonical_writer_value(value: Any) -> Any:
    """Representation-independent form used for equality and fingerprints."""
    if value is None:
        return None
    if isinstance(value, Fraction):
        return rational_string(value)
    if isinstance(value, bool):  # pragma: no cover - rejected earlier
        return value
    return value


def apply_column(plan: ColumnPlan, decoded: dict[str, Any]) -> tuple[Any, dict]:
    """Apply one planned column to a decoded row, returning (encoded, source)."""
    if plan.source == "default":
        return plan.default_encoded, {
            "default": plan.default_encoded,
            "conversion": "fill_default",
            "reader_type": type_to_json(plan.reader_type),
        }

    raw_value = decoded[plan.writer_field_id]
    source = {
        "field_id": plan.writer_field_id,
        "writer_field_id": plan.writer_field_id,
        "writer_type": type_to_json(plan.writer_type),
        "reader_type": type_to_json(plan.reader_type),
        "conversion": plan.conversions[0] if plan.conversions else "identity",
        "input": canonical_writer_value(raw_value),
    }
    if raw_value is None:
        source["conversion"] = "null_passthrough"
        return None, source

    reader_type = plan.reader_type
    if isinstance(reader_type, TextType):
        encoded = raw_value
    elif isinstance(reader_type, IntType):
        number = raw_value if isinstance(raw_value, int) else Fraction(raw_value)
        if isinstance(number, Fraction):
            if number.denominator != 1:
                raise ContractError(
                    f"internal: {plan.field_id} value {rational_string(number)} "
                    f"is not an integer for {reader_type.describe()}"
                )
            number = number.numerator
        if not is_legal_value(reader_type, number):
            raise ContractError(
                f"internal: {plan.field_id} value {number} escaped "
                f"{reader_type.describe()} despite a SAFE plan"
            )
        encoded = encode_reader_value(reader_type, number)
    else:
        number = raw_value if isinstance(raw_value, Fraction) else Fraction(raw_value)
        # Exactness guard: the plan is proven safe, but never emit a value the
        # reader contract would reject.
        if not is_legal_value(reader_type, number):
            raise ContractError(
                f"internal: {plan.field_id} value {rational_string(number)} "
                f"escaped {reader_type.describe()} despite a SAFE plan"
            )
        encoded = encode_reader_value(reader_type, number)
    return encoded, source


def adapt_rows(
    writer: Contract,
    reader: Contract,
    rows_raw: list,
    *,
    allow_duplicate_row_ids: bool = False,
) -> AdaptationResult:
    """Adapt a list of raw rows; whole-file rejection on any row error.

    ``allow_duplicate_row_ids`` is used by the ingestion library, which must
    see every occurrence of a row (to detect same-revision conflicts and
    highest-revision wins).  The standalone adapter keeps the stricter
    default: duplicate row ids reject the file.
    """
    report = check_compatibility(writer, reader)
    if not report.safe:
        return AdaptationResult(
            status="INCOMPATIBLE",
            writer_version=writer.version,
            reader_version=reader.version,
            records=None,
            errors=[RowError(None, None, r) for r in report.errors]
            or [
                RowError(None, None, f"column {c.field_id}: {c.reasons[0]}")
                for c in report.columns
                if not c.safe
            ],
            report=report,
        )

    if not isinstance(rows_raw, list):
        return AdaptationResult(
            status="INVALID",
            writer_version=writer.version,
            reader_version=reader.version,
            records=None,
            errors=[RowError(None, None, "rows must be a list")],
            report=report,
        )

    errors: list[RowError] = []
    prepared: list[CanonicalRow] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows_raw):
        try:
            row_id, revision, values = parse_row_entry(raw, index)
        except ContractError as exc:
            errors.append(RowError(None, index, str(exc)))
            continue
        if row_id in seen and not allow_duplicate_row_ids:
            errors.append(
                RowError(row_id, index, f"row {row_id!r}: duplicate row_id in file")
            )
            continue
        seen.add(row_id)
        decoded, row_errors = validate_and_decode_row(writer, row_id, values)
        if row_errors:
            errors.extend(RowError(row_id, index, msg) for msg in row_errors)
            continue
        encoded_values: dict[str, Any] = {}
        sources: dict[str, Any] = {}
        canonical: dict[str, Any] = {}
        try:
            for plan in report.columns:
                encoded, source = apply_column(plan, decoded)
                encoded_values[plan.field_id] = encoded
                sources[plan.field_id] = source
                canonical[plan.field_id] = _canonical_reader_value(
                    plan, encoded
                )
        except Exception as exc:
            errors.append(RowError(row_id, index, str(exc)))
            continue
        prepared.append(
            CanonicalRow(
                row_id=row_id,
                revision=revision,
                values=encoded_values,
                domain_values=canonical,
                sources=sources,
                writer_values=decoded,
            )
        )

    if errors:
        return AdaptationResult(
            status="INVALID",
            writer_version=writer.version,
            reader_version=reader.version,
            records=None,
            errors=errors,
            report=report,
        )

    records = {
        "format": RECORDS_FORMAT,
        "reader_version": reader.version,
        "rows": [
            {
                "row_id": row.row_id,
                "revision": row.revision,
                "values": row.values,
                "sources": row.sources,
            }
            for row in prepared
        ],
    }
    return AdaptationResult(
        status="ADAPTED",
        writer_version=writer.version,
        reader_version=reader.version,
        records=records,
        errors=[],
        report=report,
        canonical_rows=prepared,
    )


def _canonical_reader_value(plan: ColumnPlan, encoded: Any) -> Any:
    """Representation-independent reader value (int/Fraction/str/None).

    For a reader that is ``DECIMAL`` we keep an exact ``Fraction``; for a
    reader that is ``INT`` we keep a Python ``int``.  Equality across these
    two is handled in :func:`_canonical_equal`, which compares numerically.
    """
    if encoded is None:
        return None
    if isinstance(plan.reader_type, TextType):
        return encoded
    if isinstance(plan.reader_type, IntType):
        return int(encoded)
    return Fraction(encoded)


def adapt_file_document(writer: Contract, reader: Contract, doc: Any) -> AdaptationResult:
    """Adapt a ``{"file_id":..., "writer_version":..., "reader_version":...,
    "rows":[...]}`` document against fixed contracts."""
    if not isinstance(doc, dict):
        return AdaptationResult(
            status="INVALID",
            writer_version=writer.version,
            reader_version=reader.version,
            records=None,
            errors=[RowError(None, None, "file document must be an object")],
            report=check_compatibility(writer, reader),
        )
    errors: list[RowError] = []
    if "writer_version" in doc and doc["writer_version"] != writer.version:
        errors.append(
            RowError(
                None,
                None,
                f"file declares writer {doc['writer_version']!r}, fixed writer is "
                f"{writer.version!r}",
            )
        )
    if "reader_version" in doc and doc["reader_version"] != reader.version:
        errors.append(
            RowError(
                None,
                None,
                f"file declares reader {doc['reader_version']!r}, fixed reader is "
                f"{reader.version!r}",
            )
        )
    unknown = set(doc) - {"file_id", "writer_version", "reader_version", "rows"}
    if unknown:
        errors.append(RowError(None, None, f"unexpected file keys {sorted(unknown)}"))
    if errors:
        return AdaptationResult(
            status="INVALID",
            writer_version=writer.version,
            reader_version=reader.version,
            records=None,
            errors=errors,
            report=check_compatibility(writer, reader),
        )
    return adapt_rows(writer, reader, doc.get("rows", []))


def write_records(records: dict, path: str) -> None:
    import json
    from pathlib import Path

    Path(path).write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
