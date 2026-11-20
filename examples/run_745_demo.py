"""End-to-end demonstration of the 745 scenario.

Run with:

    PYTHONPATH=. python3.11 examples/run_745_demo.py

It uses the fixed writer/reader contracts in ``materials/745`` and the supplied
(never imported) ``reader_client.py`` to prove:

  1. initial INT8 rows 1 / -2 -> DECIMAL(3,0) with status default "ok",
     client amount total -1;
  2. narrowing the reader to DECIMAL(2,0) is rejected over the whole domain
     even though both sample rows still fit;
  3. an INT16 writer against DECIMAL(3,0) is likewise rejected;
  4. DECIMAL(3,1) 1/10 -> DECIMAL(4,2) converts exactly; nullable writer to
     non-nullable reader is rejected;
  5. F-BAD (one:=2 @r1 conflicts with accepted 1; three:=1 @r1) rejects the
     whole file, three stays invisible, target total stays -1;
  6. F-NEW (one:=3 @r2, two:=4 @r2) is accepted atomically, the published
     slice totals 7 and the supplied client reads it, while the old slice is
     still -1.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "materials" / "745"
sys.path.insert(0, str(ROOT))

from extcontract import (  # noqa: E402
    IngestionLibrary,
    adapt_rows,
    check_compatibility,
    load_contract,
    parse_contract,
    replace_field_type,
    write_records,
)

OUT = ROOT / "examples" / "out"


def client(contract, records):
    proc = subprocess.run(
        [sys.executable, str(MATERIALS / "reader_client.py"), str(contract), str(records)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"reader_client failed: {proc.stderr}")
    return json.loads(proc.stdout)


def banner(n, title):
    print(f"\n=== {n}. {title} ===")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    material = MATERIALS

    writer = load_contract(material / "writer-contract.json", role="writer")
    reader = load_contract(material / "reader-contract.json", role="reader")
    input_doc = json.loads((material / "input.json").read_text(encoding="utf-8"))

    # 1. Baseline -----------------------------------------------------------
    banner(1, "INT8 1/-2 -> DECIMAL(3,0), status default ok")
    report = check_compatibility(writer, reader)
    print("compat:", report.summary())
    adapted = adapt_rows(writer, reader, input_doc["rows"])
    assert adapted.status == "ADAPTED", adapted.errors
    base_path = OUT / "base-records.json"
    write_records(adapted.records, base_path)
    totals = client(material / "reader-contract.json", base_path)
    print("reader_client:", totals)
    assert totals == {"numeric_totals": {"amount": "-1"}, "row_count": 2}

    # 2. Narrow reader amount to DECIMAL(2,0) -------------------------------
    banner(2, "reader_revision amount -> DECIMAL(2,0): whole domain unsafe")
    narrowed = replace_field_type(
        reader,
        field_id="amount",
        new_type_json={"kind": "DECIMAL", "precision": 2, "scale": 0},
    )
    narrowed_result = adapt_rows(writer, narrowed, input_doc["rows"])
    print("compat:", narrowed_result.report.summary())
    print("status:", narrowed_result.status, "(samples 1/-2 still fit)")
    assert narrowed_result.status == "INCOMPATIBLE"

    # 3. INT16 writer against DECIMAL(3,0) ----------------------------------
    banner(3, "writer_revision amount -> INT16 vs DECIMAL(3,0): unsafe")
    w2 = parse_contract(
        {
            "version": "w2",
            "fields": [
                {
                    "field_id": "amount",
                    "name": "amount",
                    "type": {"kind": "INT", "bits": 16},
                    "nullable": False,
                }
            ],
        },
        role="writer",
    )
    r_int16 = adapt_rows(w2, reader, [{"row_id": "one", "values": {"amount": 1}}])
    print("compat:", r_int16.report.summary())
    print("status:", r_int16.status)
    assert r_int16.status == "INCOMPATIBLE"

    # 4. DECIMAL(3,1) -> (4,2) exact; nullable conflict ---------------------
    banner(4, "DECIMAL(3,1) 1/10 -> DECIMAL(4,2) exact; nullable -> required")
    wd = parse_contract(
        {"version": "wd", "fields": [
            {"field_id": "d", "name": "d",
             "type": {"kind": "DECIMAL", "precision": 3, "scale": 1},
             "nullable": False}]},
        role="writer",
    )
    rd = parse_contract(
        {"version": "rd", "fields": [
            {"field_id": "d", "name": "d",
             "type": {"kind": "DECIMAL", "precision": 4, "scale": 2},
             "nullable": False}]},
        role="reader",
    )
    dec_result = adapt_rows(wd, rd, [{"row_id": "r", "values": {"d": "1/10"}}])
    assert dec_result.status == "ADAPTED"
    dec_path = OUT / "decimal-records.json"
    rd_path = OUT / "reader-rd.json"
    write_records(dec_result.records, dec_path)
    rd_path.write_text(json.dumps(rd.canonical_json()), encoding="utf-8")
    print("reader_client:", client(rd_path, dec_path))
    assert dec_result.records["rows"][0]["values"]["d"] == "1/10"

    wnull = parse_contract(
        {"version": "wn", "fields": [
            {"field_id": "d", "name": "d",
             "type": {"kind": "INT", "bits": 8}, "nullable": True}]},
        role="writer",
    )
    rnull = parse_contract(
        {"version": "rn", "fields": [
            {"field_id": "d", "name": "d",
             "type": {"kind": "DECIMAL", "precision": 3, "scale": 0},
             "nullable": False}]},
        role="reader",
    )
    nullable_report = check_compatibility(wnull, rnull)
    print("nullable writer -> non-nullable reader:", nullable_report.summary())
    assert not nullable_report.safe

    # 5-6. Stateful ingestion: bootstrap, F-BAD, F-NEW ---------------------
    banner(5, "bootstrap, then F-BAD rejection")
    library = IngestionLibrary(reader)
    boot = {
        "file_id": "bootstrap",
        "writer_version": "w1",
        "reader_version": "r1",
        "rows": [
            {"row_id": "one", "revision": 1, "values": {"amount": 1}},
            {"row_id": "two", "revision": 1, "values": {"amount": -2}},
        ],
    }
    assert library.ingest(writer, boot).accepted
    print("bootstrap totals:", library.numeric_totals())

    f_bad = {
        "file_id": "F-BAD",
        "writer_version": "w1",
        "reader_version": "r1",
        "rows": [
            {"row_id": "one", "revision": 1, "values": {"amount": 2}},
            {"row_id": "three", "revision": 1, "values": {"amount": 1}},
        ],
    }
    bad = library.ingest(writer, f_bad)
    print("F-BAD accepted:", bad.accepted)
    print("F-BAD errors:", bad.errors)
    assert not bad.accepted
    assert library.numeric_totals() == {"amount": "-1"}
    assert library.get_row("three") is None
    print("target totals after F-BAD:", library.numeric_totals())

    banner(6, "F-NEW acceptance, publish, client total 7")
    f_new = {
        "file_id": "F-NEW",
        "writer_version": "w1",
        "reader_version": "r1",
        "rows": [
            {"row_id": "one", "revision": 2, "values": {"amount": 3}},
            {"row_id": "two", "revision": 2, "values": {"amount": 4}},
        ],
    }
    new = library.ingest(writer, f_new)
    print("F-NEW accepted:", new.accepted)
    assert new.accepted, new.errors
    print("target totals after F-NEW:", library.numeric_totals())
    assert library.numeric_totals() == {"amount": "7"}
    published = library.publish()
    published_path = OUT / "published-records.json"
    write_records(published, published_path)
    print("published reader_client:", client(material / "reader-contract.json", published_path))
    assert client(material / "reader-contract.json", published_path)["numeric_totals"] == {
        "amount": "7"
    }

    # The old slice from the bootstrap is still -1 when adapted directly.
    old = adapt_rows(writer, reader, boot["rows"])
    old_path = OUT / "old-slice-records.json"
    write_records(old.records, old_path)
    old_totals = client(material / "reader-contract.json", old_path)
    print("old slice reader_client:", old_totals)
    assert old_totals["numeric_totals"] == {"amount": "-1"}

    print("\nAll demo assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
