"""End-to-end polyline catalog scenario against the real ingestion entry.

The script drives the mandated screen story entirely through the public API:

  1. ingest channels A=(0,4,0) and B=(0,3,0) and publish a catalog from the
     library snapshot (revisions + sources embedded);
  2. budget 5, equal weights   -> extra point to A, optimum 3;
  3. budget 5, B weight 2      -> extra point to B, optimum 4;
  4. correct A's midpoint 4->2 at revision 2 through the original ingestion
     entry, republish           -> extra point to B, optimum 2; the old
     catalog still explains the pre-correction polylines;
  5. a null-broken channel proves lines never span gaps and reports the exact
     minimum point count when the budget cannot cover every segment;
  6. the new catalog is saved and reopened as plain JSON, and the reopened
     object (no library attached) produces the identical screen plan.

All numbers are exact rationals; run with ``PYTHONPATH=. python3 examples/run_polyline_demo.py``.
"""

from __future__ import annotations

import json
import tempfile
from fractions import Fraction
from pathlib import Path

from extcontract import (
    ChannelSpec,
    IngestionLibrary,
    load_catalog,
    parse_contract,
)

SPEC = ChannelSpec(name_field="ch", time_field="t", value_field="v")


def contracts():
    fields = [
        {"field_id": "ch", "name": "channel", "type": {"kind": "TEXT"}},
        {
            "field_id": "t",
            "name": "time",
            "type": {"kind": "DECIMAL", "precision": 18, "scale": 6},
            "nullable": True,
        },
        {
            "field_id": "v",
            "name": "value",
            "type": {"kind": "DECIMAL", "precision": 18, "scale": 6},
            "nullable": True,
        },
    ]
    # The same field layout is a legal writer and reader pair under two
    # distinct contract versions.
    writer = parse_contract({"version": "w1", "fields": fields}, role="writer")
    reader = parse_contract({"version": "r1", "fields": fields}, role="reader")
    return writer, reader


def doc(file_id, rows):
    return {
        "file_id": file_id,
        "writer_version": "w1",
        "reader_version": "r1",
        "rows": rows,
    }


def row(row_id, ch, t, v, revision=1):
    return {"row_id": row_id, "revision": revision, "values": {"ch": ch, "t": t, "v": v}}


def show(title, plan):
    print(f"\n=== {title} ===")
    print(json.dumps(plan.to_json(), ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    writer, reader = contracts()
    library = IngestionLibrary(reader)

    # 1. Bootstrap A and B through the original atomic ingestion entry point.
    rows = [row(f"a{i}", "A", str(i), str(v)) for i, v in enumerate((0, 4, 0))]
    rows += [row(f"b{i}", "B", str(i), str(v)) for i, v in enumerate((0, 3, 0))]
    accepted = library.ingest(writer, doc("boot", rows))
    assert accepted.accepted, accepted.errors

    old_build = library.publish_polyline_catalog(SPEC, tolerance=Fraction(2))
    assert old_build.built, old_build.errors
    old_catalog = old_build.catalog

    # Stored pair-error answer: A's head-to-tail chord, earliest worst time.
    pair = old_catalog.endpoint_error("A", 0, 0, 2)
    print(
        f"A chord 0->2: max_error={pair.max_error} at earliest time "
        f"{old_catalog.earliest_error_time('A', 0, 0, 2)}"
    )

    # 2. Equal weights, budget 5.
    equal = old_catalog.allocate_display(5, {"A": 1, "B": 1})
    show("budget 5, weights A=1 B=1 (expect optimum 3, extra point -> A)", equal)
    assert equal.optimal == 3
    assert {c.name: c.point_count for c in equal.channels} == {"A": 3, "B": 2}

    # 3. B carries double weight.
    weighted = old_catalog.allocate_display(5, {"A": 1, "B": 2})
    show("budget 5, weights A=1 B=2 (expect optimum 4, extra point -> B)", weighted)
    assert weighted.optimal == 4
    assert {c.name: c.point_count for c in weighted.channels} == {"A": 2, "B": 3}

    # 4. Correct A's midpoint at a higher revision, then republish.
    correction = library.ingest(writer, doc("fix-a", [row("a1", "A", "1", "2", revision=2)]))
    assert correction.accepted, correction.errors
    new_catalog = library.publish_polyline_catalog(SPEC).catalog
    corrected = new_catalog.allocate_display(5, {"A": 1, "B": 1})
    show("after A midpoint 4->2 (expect optimum 2, extra point -> B)", corrected)
    assert corrected.optimal == 2
    assert {c.name: c.point_count for c in corrected.channels} == {"A": 2, "B": 3}

    # The old catalog is immutable and still describes the old chord error 4.
    assert old_catalog.endpoint_error("A", 0, 0, 2).max_error == 4
    assert new_catalog.endpoint_error("A", 0, 0, 2).max_error == 2
    print("\nold catalog still reports A chord error 4; new catalog reports 2")

    # 5. Null-broken channels: gaps cannot be bridged; short budget is named.
    gap_library = IngestionLibrary(reader)
    gap_rows = []
    for channel, peak in (("A", 4), ("B", 3)):
        gap_rows += [
            row(f"{channel}0", channel, "0", "0"),
            row(f"{channel}1", channel, "1", str(peak)),
            row(f"{channel}2", channel, "2", "0"),
            row(f"{channel}3", channel, "3", None),
            row(f"{channel}4", channel, "4", "0"),
            row(f"{channel}5", channel, "5", str(peak)),
            row(f"{channel}6", channel, "6", "0"),
        ]
    assert gap_library.ingest(writer, doc("gaps", gap_rows)).accepted
    gap_catalog = gap_library.publish_polyline_catalog(SPEC).catalog
    short = gap_catalog.allocate_display(5, {"A": 1, "B": 1})
    show("null gaps, budget 5 (expect INFEASIBLE, minimum 8)", short)
    assert short.status == "INFEASIBLE" and short.minimum_required == 8
    floor_plan = gap_catalog.allocate_display(8, {"A": 1, "B": 1})
    assert floor_plan.status == "FEASIBLE"
    assert all(len(c.polylines) == 2 for c in floor_plan.channels)
    print("at 8 points each channel keeps two separate polylines; no line crosses null")

    # 6. Save, reopen from raw JSON, and consume without a library.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "catalog.json"
        new_catalog.save(path)
        reopened = load_catalog(json.loads(path.read_text(encoding="utf-8")))
        screen_plan = reopened.allocate_display(5, {"A": 1, "B": 1})
        assert screen_plan.optimal == corrected.optimal
        print(f"\nreopened catalog reproduces optimum {screen_plan.optimal} from {path.name}")

    print("\nALL SCENARIO CHECKS PASSED")


if __name__ == "__main__":
    main()
