"""Command-line entry points for the external contract toolkit.

    python -m extcontract compat WRITER.json READER.json [--json]
    python -m extcontract adapt  WRITER.json READER.json INPUT.json OUT.json
    python -m extcontract ingest WRITER.json READER.json FILE.json \
                                   [--state-dir DIR] [--publish OUT.json]

``compat`` prints a human-readable per-column verdict plus the public plan.
``adapt`` writes ``reader-records/v1`` (consumable by reader_client.py).
``ingest`` uses a small on-disk JSON state file so repeated invocations keep
the atomic, revision-aware target state; ``--publish`` writes the current
reader-records/v1 snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapter import adapt_file_document, write_records
from .compat import check_compatibility
from .contract import load_contract
from .ingest import IngestionLibrary


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_compat(args: argparse.Namespace) -> int:
    writer = load_contract(args.writer, role="writer")
    reader = load_contract(args.reader, role="reader")
    report = check_compatibility(writer, reader)
    if args.json:
        print(json.dumps(report.to_json(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.safe else 1
    print(report.summary())
    for column in report.columns:
        mark = "SAFE" if column.safe else "UNSAFE"
        source = column.source
        detail = f"[{mark}] {column.field_id} ({source})"
        print(detail)
        for reason in column.reasons:
            print(f"    - {reason}")
        if column.witness is not None:
            print(f"    witness: {json.dumps(column.witness, ensure_ascii=False)}")
    for ignored in report.ignored_writer_fields:
        print(f"[IGNORED] writer field {ignored['field_id']}")
    if report.errors:
        for error in report.errors:
            print(f"[ERROR] {error}")
    if report.safe:
        print("PLAN:")
        print(json.dumps(report.adaptation_plan(), ensure_ascii=False, indent=2))
    return 0 if report.safe else 1


def cmd_adapt(args: argparse.Namespace) -> int:
    writer = load_contract(args.writer, role="writer")
    reader = load_contract(args.reader, role="reader")
    doc = _load_json(args.input)
    result = adapt_file_document(writer, reader, doc)
    if result.status != "ADAPTED":
        print(json.dumps(result.to_json(), ensure_ascii=False, indent=2, sort_keys=True))
        print(f"adaptation rejected: {result.status}", file=sys.stderr)
        return 1
    write_records(result.records, args.output)
    print(f"wrote {len(result.records['rows'])} rows to {args.output}")
    return 0


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"sequence": 0, "rows": {}, "receipts": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _json_writer_value(value):
    """JSON-safe writer value for the on-disk state file.

    Fractions are written as fraction strings, ints and strings as themselves.
    This is used only for the CLI's informational ``writer_values`` metadata;
    canonical reader values drive all equality decisions.
    """
    if value is None or isinstance(value, (int, str)):
        return value
    return str(value)


def _dump_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_ingest(args: argparse.Namespace) -> int:
    writer = load_contract(args.writer, role="writer")
    reader = load_contract(args.reader, role="reader")
    doc = _load_json(args.file)
    state_path = Path(args.state_dir) / f"state-{reader.version}.json"

    # Rehydrate a library from the on-disk state, ingest, then persist only on
    # acceptance (atomic at the file level: a rejected file writes nothing).
    library = _library_from_state(reader, _load_state(state_path))
    result = library.ingest(writer, doc)
    if result.accepted:
        _dump_state(_state_from_library(library), state_path)
        print(json.dumps(result.to_json(), ensure_ascii=False, indent=2, sort_keys=True))
        if args.publish:
            write_records(library.publish(), args.publish)
        return 0
    print(json.dumps(result.to_json(), ensure_ascii=False, indent=2, sort_keys=True))
    print("file rejected; target state unchanged", file=sys.stderr)
    return 1


def _library_from_state(reader, state: dict) -> IngestionLibrary:
    from .ingest import Receipt, RowState
    from .types import IntType, TextType
    from fractions import Fraction

    field_types = {f.field_id: f.type for f in reader.fields}

    def decode_canonical(field_id, raw):
        if raw is None:
            return None
        t = field_types.get(field_id)
        if isinstance(t, TextType):
            return raw
        if isinstance(t, IntType):
            return int(raw)
        # DECIMAL (and anything numeric) round-trips as a minimal fraction
        return Fraction(raw)

    library = IngestionLibrary(reader)
    library._sequence = state.get("sequence", 0)
    rows = {}
    for row_id, raw in state.get("rows", {}).items():
        canonical = {
            k: decode_canonical(k, v) for k, v in raw["canonical"].items()
        }
        rows[row_id] = RowState(
            row_id=row_id,
            revision=raw["revision"],
            canonical=canonical,
            encoded=raw["encoded"],
            sources=raw["sources"],
            writer_values=raw.get("writer_values", {}),
        )
    library._rows[reader.version] = rows
    for key, raw in state.get("receipts", {}).items():
        file_id = raw["file_id"]
        library._receipts[(reader.version, file_id)] = Receipt(
            file_id=file_id,
            writer_version=raw["writer_version"],
            reader_version=reader.version,
            fingerprint=raw["fingerprint"],
            sequence=raw["sequence"],
            applied=raw.get("applied", []),
            ignored=raw.get("ignored", []),
            replayed_rows=raw.get("replayed_rows", []),
            row_ids=raw.get("row_ids", []),
        )
    return library


def _state_from_library(library: IngestionLibrary) -> dict:
    rows = {}
    for row_id, row in library._rows[library.reader_version].items():
        rows[row_id] = {
            "revision": row.revision,
            "canonical": {
                k: (None if v is None else str(v)) for k, v in row.canonical.items()
            },
            "encoded": row.encoded,
            "sources": row.sources,
            "writer_values": {
                k: _json_writer_value(v) for k, v in row.writer_values.items()
            },
        }
    receipts = {}
    for (_, file_id), receipt in library._receipts.items():
        receipts[file_id] = {
            "file_id": receipt.file_id,
            "writer_version": receipt.writer_version,
            "fingerprint": receipt.fingerprint,
            "sequence": receipt.sequence,
            "applied": receipt.applied,
            "ignored": receipt.ignored,
            "replayed_rows": receipt.replayed_rows,
            "row_ids": receipt.row_ids,
        }
    return {"sequence": library._sequence, "rows": rows, "receipts": receipts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="extcontract")
    sub = parser.add_subparsers(dest="command", required=True)

    p_compat = sub.add_parser("compat", help="row-free compatibility report")
    p_compat.add_argument("writer")
    p_compat.add_argument("reader")
    p_compat.add_argument("--json", action="store_true")
    p_compat.set_defaults(func=cmd_compat)

    p_adapt = sub.add_parser("adapt", help="adapt a file to reader-records/v1")
    p_adapt.add_argument("writer")
    p_adapt.add_argument("reader")
    p_adapt.add_argument("input")
    p_adapt.add_argument("output")
    p_adapt.set_defaults(func=cmd_adapt)

    p_ingest = sub.add_parser("ingest", help="atomically ingest a file")
    p_ingest.add_argument("writer")
    p_ingest.add_argument("reader")
    p_ingest.add_argument("file")
    p_ingest.add_argument("--state-dir", default=".extcontract-state")
    p_ingest.add_argument("--publish", default=None)
    p_ingest.set_defaults(func=cmd_ingest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
