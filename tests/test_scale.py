"""Scale checks: 20 columns, 5000 rows, 50 contract versions.

The point is not raw speed but that the documented envelope is exercised and
that costs are linear in rows*columns (planning) and rows*columns plus one
sort (publish), with no per-version rescanning.
"""

from __future__ import annotations

import time

from extcontract import IngestionLibrary, check_compatibility, parse_contract

COLUMNS = 20
ROWS = 5000
VERSIONS = 50


def build_contracts(ncols=COLUMNS):
    writer_fields = []
    reader_fields = []
    for i in range(ncols):
        writer_fields.append(
            {
                "field_id": f"c{i}",
                "name": f"writer_{i}",
                "type": {"kind": "INT", "bits": 16},
                "nullable": False,
            }
        )
        reader_fields.append(
            {
                "field_id": f"c{i}",
                "name": f"reader_{i}",
                "type": {"kind": "DECIMAL", "precision": 6, "scale": 0},
                "nullable": False,
            }
        )
    writer = parse_contract({"version": "w1", "fields": writer_fields}, role="writer")
    reader = parse_contract({"version": "r1", "fields": reader_fields}, role="reader")
    return writer, reader


def make_doc(file_id, count, offset=0):
    rows = []
    for r in range(count):
        values = {f"c{i}": (r + offset) % 30000 - 15000 for i in range(COLUMNS)}
        rows.append({"row_id": f"row{r + offset}", "revision": 1, "values": values})
    return {
        "file_id": file_id,
        "writer_version": "w1",
        "reader_version": "r1",
        "rows": rows,
    }


def test_full_envelope():
    writer, reader = build_contracts()

    start = time.perf_counter()
    report = check_compatibility(writer, reader)
    plan_ms = (time.perf_counter() - start) * 1000
    assert report.safe
    assert len(report.columns) == COLUMNS

    library = IngestionLibrary(reader)
    start = time.perf_counter()
    result = library.ingest(writer, make_doc("big", ROWS))
    ingest_ms = (time.perf_counter() - start) * 1000
    assert result.accepted, result.errors
    assert library.row_count() == ROWS

    start = time.perf_counter()
    published = library.publish()
    publish_ms = (time.perf_counter() - start) * 1000
    assert len(published["rows"]) == ROWS
    assert len(published["rows"][0]["values"]) == COLUMNS

    # 50 contract versions are isolated maps; ingesting under each is cheap and
    # does not rescan other versions' rows.
    start = time.perf_counter()
    for v in range(VERSIONS):
        rv = _versioned_reader(reader, v)
        lib = IngestionLibrary(rv)
        assert lib.ingest(
            writer,
            {
                "file_id": "f",
                "writer_version": "w1",
                "reader_version": rv.version,
                "rows": [
                    {
                        "row_id": "only",
                        "revision": 1,
                        "values": {f"c{i}": i for i in range(COLUMNS)},
                    }
                ],
            },
        ).accepted
    versions_ms = (time.perf_counter() - start) * 1000

    # Loose ceilings: generous enough to avoid flakiness, tight enough to catch
    # accidental quadratic rescanning.
    assert plan_ms < 2000
    assert ingest_ms < 30000
    assert publish_ms < 5000
    assert versions_ms < 10000
    print(
        f"plan={plan_ms:.1f}ms ingest={ingest_ms:.1f}ms "
        f"publish={publish_ms:.1f}ms versions={versions_ms:.1f}ms"
    )


def _versioned_reader(base, v):
    fields = []
    for f in base.fields:
        fields.append(
            {
                "field_id": f.field_id,
                "name": f.name,
                "type": {"kind": "DECIMAL", "precision": 6, "scale": 0},
                "nullable": False,
            }
        )
    return parse_contract({"version": f"r{v}", "fields": fields}, role="reader")
