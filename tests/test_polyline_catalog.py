"""Tests for exact polyline compression and revision-aware catalogs.

Covers the mandated screen scenario:

  * channels A=(0,4,0), B=(0,3,0), budget 5, equal weights -> extra point to
    A, optimum 3;
  * B weight 2 -> extra point to B, optimum 4;
  * correcting A's midpoint to 2 through the original ingestion entry ->
    extra point to B, optimum 2;
  * null-broken channels never bridge the gap, and a short budget is reported
    with the exact minimum point count.

All assertions use exact rationals; nothing here uses floats.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from extcontract import (
    CatalogBuildResult,
    ChannelSpec,
    IngestionLibrary,
    load_catalog,
    parse_contract,
)
from extcontract.polyline import (
    build_segments,
    channel_curve,
    channel_error_for_budget,
    channel_min_points,
    channel_plan,
    interpolate,
    min_polyline,
    pair_error,
    pair_error_table,
    polyline_for_count,
    segment_curve,
)

SPEC = ChannelSpec(name_field="ch", time_field="t", value_field="v")


def reader_contract():
    return parse_contract(
        {
            "version": "r1",
            "fields": [
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
            ],
        },
        role="reader",
    )


def writer_contract():
    return parse_contract(
        {
            "version": "w1",
            "fields": [
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
            ],
        },
        role="writer",
    )


def row(row_id, ch, t, v, revision=1):
    return {"row_id": row_id, "revision": revision, "values": {"ch": ch, "t": t, "v": v}}


def file_doc(file_id, rows, writer_version="w1", reader_version="r1"):
    return {
        "file_id": file_id,
        "writer_version": writer_version,
        "reader_version": reader_version,
        "rows": rows,
    }


def ab_library(a=(0, 4, 0), b=(0, 3, 0)):
    library = IngestionLibrary(reader_contract())
    rows = [row(f"a{i}", "A", str(i), str(a[i])) for i in range(3)]
    rows += [row(f"b{i}", "B", str(i), str(b[i])) for i in range(3)]
    result = library.ingest(writer_contract(), file_doc("boot", rows))
    assert result.accepted, result.errors
    return library


# ---------------------------------------------------------------------------
# pure geometry
# ---------------------------------------------------------------------------

def test_interpolation_and_pair_error_are_exact():
    segments = build_segments([(0, 0), (1, 4), (2, 0)])
    samples = segments[0].samples
    assert interpolate(samples, 0, 2, 1) == Fraction(0)
    pair = pair_error(samples, 0, 2)
    assert pair.max_error == Fraction(4)
    assert pair.earliest_index == 1
    # Adjacent endpoints always have zero error at the first index.
    assert pair_error(samples, 0, 1).max_error == 0
    assert pair_error(samples, 0, 1).earliest_index == 0


def test_pair_error_uses_rational_interpolation_no_float():
    # samples (0,0),(1,1),(2,0): chord 0->2 is the zero line, so the middle
    # sample deviates by exactly 1 (no division artefacts, all Fraction).
    segments = build_segments([(0, 0), (1, 1), (2, 0)])
    pair = pair_error(segments[0].samples, 0, 2)
    assert pair.max_error == Fraction(1)
    assert isinstance(pair.max_error, Fraction)


def test_earliest_index_on_tied_maxima():
    # (0,0),(1,2),(2,0),(3,2): chord 0->3 gives line values 2/3 and 4/3 at
    # t=1,2; deviations 4/3 at both interior samples -> earliest index is 1.
    segments = build_segments([(0, 0), (1, 2), (2, 0), (3, 2)])
    pair = pair_error(segments[0].samples, 0, 3)
    assert pair.max_error == Fraction(4, 3)
    assert pair.earliest_index == 1


def test_min_polyline_dp_beats_naive_greedy_expectation():
    # (0,0),(1,2),(2,2),(3,0): the single head-to-tail chord reaches error 2,
    # so cap 1 needs a third vertex; cap 0 keeps every original point.
    segments = build_segments([(0, 0), (1, 2), (2, 2), (3, 0)])
    segment = segments[0]
    table = pair_error_table(segment.samples)
    assert table[(0, 3)].max_error == 2
    polyline = min_polyline(segment, Fraction(1))
    assert polyline.point_count == 3
    assert polyline.vertex_indices[0] == 0
    assert polyline.vertex_indices[-1] == 3
    assert polyline.max_error <= 1
    assert min_polyline(segment, 0).vertex_indices == (0, 1, 2, 3)


def test_segment_curve_is_pareto_ordered_and_full_count_zero():
    segments = build_segments([(i, v) for i, v in enumerate((0, 4, 0))])
    curve = segment_curve(segments[0])
    counts = [c for c, _e, _v in curve]
    errors = [e for _c, e, _v in curve]
    assert counts == sorted(counts)
    assert errors == sorted(errors, reverse=True)
    assert curve[-1][1] == 0 and curve[-1][2] == (0, 1, 2)
    assert curve[0][0] == 2 and curve[0][1] == 4


def test_null_splits_segments_and_gap_is_never_bridged():
    segments = build_segments(
        [(0, 1), (1, 2), (2, 3), (3, None), (4, 5), (5, 6)]
    )
    assert [s.size for s in segments] == [3, 2]
    # Floor: head+tail of the 3-point segment (2) plus both points of the
    # 2-point segment (2).  No plan merges the segments across the null gap.
    assert channel_min_points(segments) == 4
    curve = channel_curve(segments)
    counts = [count for count, _error in curve]
    assert counts[0] == 4 and counts[-1] == 5
    plan = channel_plan(segments, 4)
    assert {p.segment_index for p in plan} == {0, 1}
    with pytest.raises(Exception):
        channel_plan(segments, 3)  # below the mandatory floor


def test_single_point_segment_is_kept_and_empty_run_omitted():
    segments = build_segments([(0, 5), (1, None), (2, 7)])
    assert [s.size for s in segments] == [1, 1]
    assert channel_min_points(segments) == 2
    plan = channel_plan(segments, 2)
    assert all(p.point_count == 1 and p.max_error == 0 for p in plan)
    # Leading/trailing/duplicate nulls create no empty segments.
    assert build_segments([(0, None), (1, 1), (2, None)])[0].size == 1


def test_polyline_for_count_deterministic_tie_break():
    segments = build_segments([(0, 0), (1, 2), (2, 0), (3, 2)])
    first = polyline_for_count(segments[0], 3)
    second = polyline_for_count(segments[0], 3)
    assert first.vertex_indices == second.vertex_indices


# ---------------------------------------------------------------------------
# catalog construction, conflicts and provenance
# ---------------------------------------------------------------------------

def test_build_catalog_stores_snapshot_revisions_and_sources():
    library = ab_library()
    result = library.publish_polyline_catalog(SPEC, tolerance=Fraction(2))
    assert result.built and result.errors == []
    catalog = result.catalog
    assert catalog.reader_version == "r1"
    assert catalog.tolerance == 2
    snapshot = catalog.snapshot
    assert snapshot["format"] == "reader-records/v1"
    assert len(snapshot["rows"]) == 6
    assert all("revision" in row and "sources" in row for row in snapshot["rows"])
    channel_a = catalog.channel("A")
    assert [r.value for r in channel_a.readings] == [Fraction(0), 4, 0]
    pair = catalog.endpoint_error("A", 0, 0, 2)
    assert pair.max_error == 4 and pair.earliest_index == 1
    assert catalog.earliest_error_time("A", 0, 0, 2) == 1
    # Build cap 2 on A's (0,4,0) requires all three points.
    assert catalog.minimum_points("A", 0) == 3
    polylines = catalog.build_tolerance_polylines("A")
    assert polylines[0].vertex_indices == (0, 1, 2)


def test_duplicate_time_within_channel_is_rejected_entirely():
    library = IngestionLibrary(reader_contract())
    result = library.ingest(
        writer_contract(),
        file_doc(
            "dup",
            [
                row("x1", "A", "0", "1"),
                row("x2", "A", "0", "2"),
            ],
        ),
    )
    assert result.accepted
    build = library.publish_polyline_catalog(SPEC)
    assert not build.built
    assert any("duplicate time" in e and "A" in e for e in build.errors)


def test_same_time_in_different_channels_is_not_a_conflict():
    library = IngestionLibrary(reader_contract())
    result = library.ingest(
        writer_contract(),
        file_doc(
            "ok",
            [row("x1", "A", "0", "1"), row("x2", "B", "0", "9")],
        ),
    )
    assert result.accepted
    build = library.publish_polyline_catalog(SPEC)
    assert build.built, build.errors
    assert {c.name for c in build.catalog.channels} == {"A", "B"}


def test_null_value_creates_break_in_catalog():
    library = IngestionLibrary(reader_contract())
    result = library.ingest(
        writer_contract(),
        file_doc(
            "gap",
            [
                row("g1", "A", "0", "1"),
                row("g2", "A", "1", None),
                row("g3", "A", "2", "4"),
                row("g4", "A", "3", "5"),
            ],
        ),
    )
    assert result.accepted
    catalog = library.publish_polyline_catalog(SPEC).catalog
    channel = catalog.channel("A")
    assert [s.size for s in channel.segments] == [1, 2]
    assert channel.readings[1].value is None
    # One point for the single-sample segment, two for the longer one.
    assert catalog.required_points() == {"A": 3}


def test_unknown_or_ill_typed_spec_fields_are_rejected():
    library = ab_library()
    bad = ChannelSpec(name_field="missing", time_field="t", value_field="v")
    assert not library.publish_polyline_catalog(bad).built
    text_as_time = ChannelSpec(name_field="ch", time_field="ch", value_field="v")
    assert not library.publish_polyline_catalog(text_as_time).built


def test_negative_tolerance_rejected():
    library = ab_library()
    with pytest.raises(Exception):
        library.publish_polyline_catalog(SPEC, tolerance=Fraction(-1))


# ---------------------------------------------------------------------------
# mandated screen allocation scenarios
# ---------------------------------------------------------------------------

def _allocate(library, budget, weights):
    catalog = library.publish_polyline_catalog(SPEC).catalog
    return catalog, catalog.allocate_display(budget, weights)


def test_budget_five_equal_weights_extra_point_goes_to_a():
    library = ab_library()
    catalog, plan = _allocate(library, 5, {"A": 1, "B": 1})
    assert plan.status == "FEASIBLE"
    assert plan.optimal == Fraction(3)
    points = {c.name: c.point_count for c in plan.channels}
    assert points == {"A": 3, "B": 2}
    by_name = {c.name: c for c in plan.channels}
    assert by_name["A"].max_error == 0
    assert by_name["B"].max_error == 3
    assert all(c.weighted_error <= plan.optimal for c in plan.channels)
    # Every chosen polyline keeps head and tail original samples.
    for channel in plan.channels:
        for polyline in channel.polylines:
            segment = catalog.channel(channel.name).segments[polyline.segment_index]
            assert polyline.vertex_indices[0] == 0
            assert polyline.vertex_indices[-1] == segment.size - 1


def test_b_weight_two_extra_point_goes_to_b_optimum_four():
    library = ab_library()
    _catalog, plan = _allocate(library, 5, {"A": 1, "B": 2})
    assert plan.status == "FEASIBLE"
    assert plan.optimal == Fraction(4)
    points = {c.name: c.point_count for c in plan.channels}
    assert points == {"A": 2, "B": 3}
    by_name = {c.name: c for c in plan.channels}
    assert by_name["A"].weighted_error == 4
    assert by_name["B"].weighted_error == 0


def test_correction_through_ingest_republishes_new_errors_keeps_old_catalog():
    library = ab_library()
    old_catalog = library.publish_polyline_catalog(SPEC).catalog
    old_plan = old_catalog.allocate_display(5, {"A": 1, "B": 1})
    assert old_plan.optimal == 3

    # Correct A's midpoint 4 -> 2 at a higher revision through the original
    # ingestion entry point.
    correction = file_doc(
        "fix-a",
        [row("a1", "A", "1", "2", revision=2)],
    )
    accepted = library.ingest(writer_contract(), correction)
    assert accepted.accepted, accepted.errors
    new_catalog = library.publish_polyline_catalog(SPEC).catalog
    new_plan = new_catalog.allocate_display(5, {"A": 1, "B": 1})
    assert new_plan.optimal == 2
    points = {c.name: c.point_count for c in new_plan.channels}
    assert points == {"A": 2, "B": 3}
    by_name = {c.name: c for c in new_plan.channels}
    assert by_name["A"].max_error == 2 and by_name["B"].max_error == 0

    # The old, already built catalog still explains the pre-correction lines.
    assert old_catalog.endpoint_error("A", 0, 0, 2).max_error == 4
    assert old_catalog.snapshot["rows"] != new_catalog.snapshot["rows"]
    corrected_row = next(
        r for r in new_catalog.snapshot["rows"] if r["row_id"] == "a1"
    )
    assert corrected_row["revision"] == 2
    assert corrected_row["values"]["v"] == "2"


def test_infeasible_budget_reports_minimum_required_and_no_gap_bridging():
    library = IngestionLibrary(reader_contract())
    result = library.ingest(
        writer_contract(),
        file_doc(
            "gaps",
            [
                row("a0", "A", "0", "0"),
                row("a1", "A", "1", "4"),
                row("a2", "A", "2", "0"),
                row("a3", "A", "3", None),
                row("a4", "A", "4", "0"),
                row("a5", "A", "5", "4"),
                row("a6", "A", "6", "0"),
                row("b0", "B", "0", "0"),
                row("b1", "B", "1", "3"),
                row("b2", "B", "2", "0"),
                row("b3", "B", "3", None),
                row("b4", "B", "4", "0"),
                row("b5", "B", "5", "3"),
                row("b6", "B", "6", "0"),
            ],
        ),
    )
    assert result.accepted
    catalog = library.publish_polyline_catalog(SPEC).catalog
    # Each channel has two non-empty, three-point segments -> 2 mandatory
    # points per segment, 4 per channel.
    assert catalog.required_points() == {"A": 4, "B": 4}
    plan = catalog.allocate_display(5, {"A": 1, "B": 1})
    assert plan.status == "INFEASIBLE"
    assert plan.minimum_required == 8
    assert plan.channels == []
    assert "at least 8 points" in plan.errors[0]

    # At the floor budget every segment keeps only its two endpoints, so the
    # head-to-tail chord error is paid and the null gap is never bridged.
    feasible = catalog.allocate_display(8, {"A": 1, "B": 1})
    assert feasible.status == "FEASIBLE"
    for channel in feasible.channels:
        assert len(channel.polylines) == 2
        assert channel.point_count == 4
        assert all(polyline.point_count == 2 for polyline in channel.polylines)
        # Two separate polylines per channel: no polyline spans the null gap.
        segment_indices = {polyline.segment_index for polyline in channel.polylines}
        assert segment_indices == {0, 1}
    # The floor plan genuinely carries the chord errors (A=4, B=3).
    by_name = {c.name: c for c in feasible.channels}
    assert by_name["A"].max_error == 4
    assert by_name["B"].max_error == 3


def test_weights_require_every_channel_and_must_be_positive():
    library = ab_library()
    catalog = library.publish_polyline_catalog(SPEC).catalog
    with pytest.raises(Exception):
        catalog.allocate_display(5, {"A": 1})
    with pytest.raises(Exception):
        catalog.allocate_display(5, {"A": 0, "B": 1})
    with pytest.raises(Exception):
        catalog.allocate_display(5, {"A": 1.0, "B": 1})  # no floats


# ---------------------------------------------------------------------------
# save / reopen round trip: the screen consumes the reopened catalog
# ---------------------------------------------------------------------------

def test_saved_catalog_reopens_and_reproduces_plan(tmp_path: Path):
    library = ab_library()
    catalog = library.publish_polyline_catalog(SPEC).catalog
    path = tmp_path / "catalog.json"
    catalog.save(path)

    reopened = load_catalog(json.loads(path.read_text(encoding="utf-8")))
    assert reopened.fingerprint == catalog.fingerprint
    assert reopened.endpoint_error("B", 0, 0, 2).max_error == 3
    plan = reopened.allocate_display(5, {"A": 1, "B": 2})
    assert plan.status == "FEASIBLE" and plan.optimal == 4
    # Serialised plan is itself plain JSON with rational strings only.
    encoded = json.dumps(plan.to_json(), ensure_ascii=False)
    assert json.loads(encoded)["optimal"] == "4"


def test_tampered_catalog_is_refused(tmp_path: Path):
    from extcontract import load_catalog_file

    library = ab_library()
    path = tmp_path / "catalog.json"
    library.publish_polyline_catalog(SPEC).catalog.save(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["channels"][0]["readings"][0]["value"] = "999"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Exception):
        load_catalog_file(path)


# ---------------------------------------------------------------------------
# scale limits
# ---------------------------------------------------------------------------

def test_more_than_twelve_channels_rejected():
    library = IngestionLibrary(reader_contract())
    rows = [
        row(f"r{i}", f"C{i}", "0", "1")
        for i in range(13)
    ]
    assert library.ingest(writer_contract(), file_doc("big", rows)).accepted
    result = library.publish_polyline_catalog(SPEC)
    assert not result.built
    assert any("12 channel" in e for e in result.errors)


def test_more_than_seventy_points_rejected():
    library = IngestionLibrary(reader_contract())
    rows = [row(f"r{i}", "A", str(i), str(i)) for i in range(71)]
    assert library.ingest(writer_contract(), file_doc("pts", rows)).accepted
    result = library.publish_polyline_catalog(SPEC)
    assert not result.built
    assert any("70" in e and "point" in e for e in result.errors)


def test_earliest_error_time_and_rational_weights_are_exact():
    # (0,0),(1,1),(2,0): chord error 1 at the middle time t=1.
    library = IngestionLibrary(reader_contract())
    assert library.ingest(
        writer_contract(),
        file_doc("rat", [row("r0", "A", "0", "0"), row("r1", "A", "1", "1"), row("r2", "A", "2", "0")]),
    ).accepted
    catalog = library.publish_polyline_catalog(SPEC).catalog
    assert catalog.earliest_error_time("A", 0, 0, 2) == 1
    # Fraction-string weights stay exact: weight 1/2 turns error 1 into 1/2.
    plan = catalog.allocate_display(2, {"A": "1/2"})
    assert plan.status == "FEASIBLE"
    assert plan.optimal == Fraction(1, 2)
    assert plan.channels[0].weight == Fraction(1, 2)


def test_float_tolerance_and_float_weight_are_refused():
    library = ab_library()
    with pytest.raises(Exception):
        library.publish_polyline_catalog(SPEC, tolerance=0.5)
    catalog = library.publish_polyline_catalog(SPEC).catalog
    with pytest.raises(Exception):
        catalog.allocate_display(5, {"A": 1.0, "B": 1})


def test_channel_with_only_null_values_has_no_segments_and_no_floor():
    library = IngestionLibrary(reader_contract())
    assert library.ingest(
        writer_contract(),
        file_doc(
            "allnull",
            [row("n0", "A", "0", None), row("n1", "A", "1", None)],
        ),
    ).accepted
    catalog = library.publish_polyline_catalog(SPEC).catalog
    channel = catalog.channel("A")
    assert channel.segments == []
    assert catalog.required_points() == {"A": 0}
    # A null-only channel still requires a weight entry and needs zero points;
    # the plan stays feasible with an empty polyline list for it.
    plan = catalog.allocate_display(0, {"A": 1})
    assert plan.status == "FEASIBLE"
    assert plan.optimal == 0
    assert plan.channels[0].point_count == 0
    assert plan.channels[0].polylines == []


def test_build_cap_drives_minimum_points_query_and_polyline():
    library = ab_library()
    catalog = library.publish_polyline_catalog(SPEC, tolerance=Fraction(3)).catalog
    # B's (0,3,0) head-tail chord has error 3: cap 3 allows the 2-point floor.
    assert catalog.minimum_points("B", 0) == 2
    polyline = catalog.minimum_polyline("B", 0)
    assert polyline.vertex_indices == (0, 2)
    assert polyline.max_error == 3
    # An explicit smaller cap still forces all three original points.
    assert catalog.minimum_points("B", 0, cap=Fraction(2)) == 3


def test_unknown_channel_name_in_query_raises():
    catalog = ab_library().publish_polyline_catalog(SPEC).catalog
    with pytest.raises(Exception):
        catalog.endpoint_error("ZZZ", 0, 0, 1)


def test_unsorted_ingestion_is_ordered_and_null_still_splits():
    library = IngestionLibrary(reader_contract())
    result = library.ingest(
        writer_contract(),
        file_doc(
            "unordered",
            [
                row("r2", "A", "2", "0"),
                row("r0", "A", "0", "0"),
                row("r1", "A", "1", None),
                row("r3", "A", "3", "5"),
            ],
        ),
    )
    assert result.accepted
    catalog = library.publish_polyline_catalog(SPEC).catalog
    channel = catalog.channel("A")
    assert [reading.time for reading in channel.readings] == [0, 1, 2, 3]
    assert [segment.size for segment in channel.segments] == [1, 2]


def test_multi_segment_budget_spends_extra_points_on_worst_segment():
    # Two independent 3-point segments split by a null, with chord errors 4
    # and 8. The exact DP must send the first extra point to the error-8
    # segment rather than splitting points evenly.
    segments = build_segments(
        [(0, 0), (1, 4), (2, 0), (3, None), (4, 0), (5, 8), (6, 0)]
    )
    assert channel_min_points(segments) == 4
    assert channel_curve(segments) == [
        (4, Fraction(8)),
        (5, Fraction(4)),
        (6, Fraction(0)),
    ]
    floor_plan = channel_plan(segments, 4)
    assert [p.max_error for p in floor_plan] == [Fraction(4), Fraction(8)]
    better_plan = channel_plan(segments, 5)
    assert sum(p.point_count for p in better_plan) == 5
    assert max(p.max_error for p in better_plan) == 4
    # The budget helper reports the same optimum without building polylines.
    assert channel_error_for_budget(segments, 5) == 4
    with pytest.raises(Exception):
        channel_error_for_budget(segments, 3)  # below the 4-point floor
    # Both segments remain separate polylines at every budget.
    assert {p.segment_index for p in better_plan} == {0, 1}
