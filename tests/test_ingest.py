"""Tests for the revision-aware, atomic ingestion library."""

from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from extcontract import (
    IngestionLibrary,
    adapt_rows,
    load_contract,
    parse_contract,
    write_records,
)

MATERIALS = Path("materials/745")
CLIENT = MATERIALS / "reader_client.py"


@pytest.fixture()
def writer():
    return load_contract(MATERIALS / "writer-contract.json", role="writer")


@pytest.fixture()
def reader():
    return load_contract(MATERIALS / "reader-contract.json", role="reader")


@pytest.fixture()
def library(reader):
    return IngestionLibrary(reader)


def bootstrap(library, writer):
    boot = {
        "file_id": "boot",
        "writer_version": "w1",
        "reader_version": "r1",
        "rows": [
            {"row_id": "one", "revision": 1, "values": {"amount": 1}},
            {"row_id": "two", "revision": 1, "values": {"amount": -2}},
        ],
    }
    result = library.ingest(writer, boot)
    assert result.accepted, result.errors
    return result


def file_doc(file_id, rows, writer_version="w1", reader_version="r1"):
    return {
        "file_id": file_id,
        "writer_version": writer_version,
        "reader_version": reader_version,
        "rows": rows,
    }


def run_client(contract_path, records_path):
    return subprocess.run(
        [sys.executable, str(CLIENT), str(contract_path), str(records_path)],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Bootstrap + F-BAD + F-NEW (the mandated scenario)
# ---------------------------------------------------------------------------

def test_missing_revision_defaults_to_1(library, writer):
    result = library.ingest(
        writer,
        file_doc("r", [{"row_id": "one", "values": {"amount": 1}}]),
    )
    assert result.accepted
    assert library.get_row("one").revision == 1


def test_fbad_rejected_whole_file_three_invisible(library, writer, reader):
    bootstrap(library, writer)
    f_bad = file_doc(
        "F-BAD",
        [
            {"row_id": "one", "revision": 1, "values": {"amount": 2}},
            {"row_id": "three", "revision": 1, "values": {"amount": 1}},
        ],
    )
    result = library.ingest(writer, f_bad)
    assert not result.accepted
    assert any("one" in e and "differs" in e for e in result.errors)
    # Nothing applied: total unchanged at -1, and "three" is invisible.
    assert library.numeric_totals() == {"amount": "-1"}
    assert library.get_row("three") is None
    assert library.row_count() == 2
    assert library.get_row("one").canonical["amount"] == Fraction(1)


def test_fnew_accepted_totals_seven_and_old_slice_minus_one(library, writer):
    bootstrap(library, writer)
    f_new = file_doc(
        "F-NEW",
        [
            {"row_id": "one", "revision": 2, "values": {"amount": 3}},
            {"row_id": "two", "revision": 2, "values": {"amount": 4}},
        ],
    )
    result = library.ingest(writer, f_new)
    assert result.accepted, result.errors
    assert library.numeric_totals() == {"amount": "7"}
    one = library.get_row("one")
    two = library.get_row("two")
    assert one.revision == 2 and one.canonical["amount"] == Fraction(3)
    assert two.revision == 2 and two.canonical["amount"] == Fraction(4)
    # Status still filled from the reader default on the new slice.
    assert one.encoded["status"] == "ok"
    assert one.sources["status"]["default"] == "ok"

    # The published new slice is readable by the real client at total 7.
    new_records = library.publish()
    path = Path("/tmp/published-new.json")
    write_records(new_records, path)
    proc = run_client(MATERIALS / "reader-contract.json", path)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"numeric_totals": {"amount": "7"}, "row_count": 2}

    # Reconstruct the pre-F-NEW slice to show it is still -1: adapt bootstrap
    # rows directly (the library itself has moved forward).
    boot = {"rows": [
        {"row_id": "one", "values": {"amount": 1}},
        {"row_id": "two", "values": {"amount": -2}},
    ]}
    old = adapt_rows(writer, library.reader, boot["rows"])
    old_path = Path("/tmp/old-slice.json")
    write_records(old.records, old_path)
    proc = run_client(MATERIALS / "reader-contract.json", old_path)
    assert json.loads(proc.stdout)["numeric_totals"] == {"amount": "-1"}


def test_three_conflict_is_never_visible_even_via_publish(library, writer):
    bootstrap(library, writer)
    library.ingest(
        writer,
        file_doc("F-BAD", [
            {"row_id": "one", "revision": 1, "values": {"amount": 2}},
            {"row_id": "three", "revision": 1, "values": {"amount": 1}},
        ]),
    )
    rows = library.publish()["rows"]
    assert {r["row_id"] for r in rows} == {"one", "two"}


# ---------------------------------------------------------------------------
# Revision semantics
# ---------------------------------------------------------------------------

def test_lower_revision_validated_then_ignored(library, writer):
    bootstrap(library, writer)
    library.ingest(
        writer,
        file_doc("high", [{"row_id": "one", "revision": 3, "values": {"amount": 5}}]),
    )
    result = library.ingest(
        writer,
        file_doc("low", [{"row_id": "one", "revision": 2, "values": {"amount": 9}}]),
    )
    assert result.accepted
    assert result.receipt["ignored"][0]["action"] == "ignored_lower_revision"
    assert library.get_row("one").canonical["amount"] == Fraction(5)


def test_lower_revision_with_illegal_input_rejects_whole_file(library, writer):
    bootstrap(library, writer)
    library.ingest(
        writer,
        file_doc("high", [{"row_id": "one", "revision": 3, "values": {"amount": 5}}]),
    )
    before = library.numeric_totals()
    result = library.ingest(
        writer,
        file_doc("badlow", [
            {"row_id": "one", "revision": 2, "values": {"amount": 200}},  # outside INT8
        ]),
    )
    assert not result.accepted
    assert any("outside INT(8)" in e for e in result.errors)
    assert library.numeric_totals() == before


def test_same_revision_same_value_is_replay(library, writer):
    bootstrap(library, writer)
    result = library.ingest(
        writer,
        file_doc("replay", [{"row_id": "one", "revision": 1, "values": {"amount": 1}}]),
    )
    assert result.accepted
    assert result.receipt["replayed_rows"] == ["one"]
    assert result.receipt["applied"] == []


def test_same_revision_different_value_rejects_even_with_higher_later(library, writer):
    bootstrap(library, writer)
    result = library.ingest(
        writer,
        file_doc("mixed", [
            {"row_id": "one", "revision": 1, "values": {"amount": 2}},
            {"row_id": "one", "revision": 5, "values": {"amount": 9}},
        ]),
    )
    assert not result.accepted
    assert library.get_row("one").canonical["amount"] == Fraction(1)


def test_within_file_same_revision_conflict(library, writer):
    # two different values for the same (row_id, revision) inside one file
    result = library.ingest(
        writer,
        file_doc("dupconflict", [
            {"row_id": "x", "revision": 1, "values": {"amount": 1}},
            {"row_id": "x", "revision": 1, "values": {"amount": 2}},
        ]),
    )
    assert not result.accepted
    assert library.get_row("x") is None


def test_within_file_equivalent_values_not_conflict(library, writer):
    # Same canonical value at same revision -> replay, accepted.
    result = library.ingest(
        writer,
        file_doc("dupsame", [
            {"row_id": "x", "revision": 1, "values": {"amount": 1}},
            {"row_id": "x", "revision": 1, "values": {"amount": 1}},
        ]),
    )
    assert result.accepted
    assert library.get_row("x").canonical["amount"] == Fraction(1)


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------

def test_failed_file_leaves_no_new_rows(library, writer):
    bootstrap(library, writer)
    before = library.numeric_totals()
    before_ids = set(library.publish() and {r["row_id"] for r in library.publish()["rows"]})
    result = library.ingest(
        writer,
        file_doc("partial", [
            {"row_id": "new_a", "revision": 1, "values": {"amount": 10}},
            {"row_id": "new_b", "revision": 1, "values": {"amount": 999999}},  # invalid
        ]),
    )
    assert not result.accepted
    assert library.numeric_totals() == before
    assert {r["row_id"] for r in library.publish()["rows"]} == before_ids
    assert library.get_row("new_a") is None


def test_safe_contract_is_not_acceptance(library, writer):
    """A SAFE type plan must not by itself change target state."""
    # Contract pair is SAFE, but the file has a bad value -> rejected.
    result = library.ingest(
        writer,
        file_doc("badvalue", [
            {"row_id": "one", "revision": 1, "values": {"amount": 1.5}},
        ]),
    )
    assert not result.accepted
    assert library.row_count() == 0


# ---------------------------------------------------------------------------
# File identity / idempotency
# ---------------------------------------------------------------------------

def test_file_replay_returns_original_receipt(library, writer):
    boot = bootstrap(library, writer)
    same = file_doc("boot", [
        {"row_id": "one", "revision": 1, "values": {"amount": 1}},
        {"row_id": "two", "revision": 1, "values": {"amount": -2}},
    ])
    again = library.ingest(writer, same)
    assert again.accepted
    assert again.receipt["replay"] is True
    assert again.receipt["sequence"] == boot.receipt["sequence"]
    assert library.numeric_totals() == {"amount": "-1"}


def test_file_replay_equivalent_representation(library, writer):
    # Fingerprint is representation-independent: a DECIMAL writer with "2/2"
    # must match "1" for the same canonical value.
    w = parse_contract(
        {
            "version": "wq",
            "fields": [
                {
                    "field_id": "amount",
                    "name": "amount",
                    "type": {"kind": "DECIMAL", "precision": 3, "scale": 0},
                    "nullable": False,
                }
            ],
        },
        role="writer",
    )
    boot = library.ingest(
        w,
        file_doc("f", [{"row_id": "one", "revision": 1, "values": {"amount": "1"}}],
                 writer_version="wq"),
    )
    assert boot.accepted
    again = library.ingest(
        w,
        file_doc("f", [{"row_id": "one", "revision": 1, "values": {"amount": "2/2"}}],
                 writer_version="wq"),
    )
    assert again.accepted and again.receipt["replay"] is True


def test_file_id_reuse_different_content_rejected(library, writer):
    bootstrap(library, writer)
    result = library.ingest(
        writer,
        file_doc("boot", [
            {"row_id": "one", "revision": 1, "values": {"amount": 5}},
        ]),
    )
    assert not result.accepted
    assert any("already accepted with different" in e for e in result.errors)


def test_rejected_file_does_not_leave_receipt(library, writer):
    library.ingest(
        writer,
        file_doc("bad", [{"row_id": "one", "values": {"amount": 9999}}]),
    )
    assert library.receipt_for("bad") is None


# ---------------------------------------------------------------------------
# Reader-version isolation
# ---------------------------------------------------------------------------

def test_reader_version_isolation(reader):
    lib_r1 = IngestionLibrary(reader)
    w = parse_contract(
        {
            "version": "w1",
            "fields": [
                {
                    "field_id": "amount",
                    "name": "amount",
                    "type": {"kind": "INT", "bits": 8},
                    "nullable": False,
                }
            ],
        },
        role="writer",
    )
    assert lib_r1.ingest(
        w, file_doc("f1", [{"row_id": "one", "values": {"amount": 1}}])
    ).accepted
    # A different library with a different reader contract starts empty.
    r2 = parse_contract(
        {
            "version": "r2",
            "fields": [
                {
                    "field_id": "amount",
                    "name": "amount",
                    "type": {"kind": "DECIMAL", "precision": 3, "scale": 0},
                    "nullable": False,
                }
            ],
        },
        role="reader",
    )
    lib_r2 = IngestionLibrary(r2)
    assert lib_r2.row_count() == 0
    assert lib_r2.publish()["rows"] == []
    # Cross-version ingest into r1's library is refused.
    rejected = lib_r1.ingest(w, file_doc("f2", [
        {"row_id": "one", "values": {"amount": 1}}], reader_version="r2"))
    assert not rejected.accepted


# ---------------------------------------------------------------------------
# Display-name change keeps identity; provenance retained
# ---------------------------------------------------------------------------

def test_display_name_change_preserves_row_identity(library):
    w_old = parse_contract(
        {
            "version": "w1",
            "fields": [
                {
                    "field_id": "amount",
                    "name": "old_amount",
                    "type": {"kind": "INT", "bits": 8},
                    "nullable": False,
                }
            ],
        },
        role="writer",
    )
    r = parse_contract(
        {
            "version": "r1",
            "fields": [
                {
                    "field_id": "amount",
                    "name": "amount",
                    "type": {"kind": "DECIMAL", "precision": 3, "scale": 0},
                    "nullable": False,
                }
            ],
        },
        role="reader",
    )
    lib = IngestionLibrary(r)
    assert lib.ingest(
        w_old, file_doc("f", [{"row_id": "one", "revision": 1, "values": {"amount": 1}}])
    ).accepted
    # New writer swaps display name, same field_id; same canonical value at
    # same revision is a replay, not a conflict.
    w_new = parse_contract(
        {
            "version": "w1",
            "fields": [
                {
                    "field_id": "amount",
                    "name": "amount",
                    "type": {"kind": "INT", "bits": 8},
                    "nullable": False,
                }
            ],
        },
        role="writer",
    )
    result = lib.ingest(
        w_new, file_doc("f2", [{"row_id": "one", "revision": 1, "values": {"amount": 1}}])
    )
    assert result.accepted
    assert "one" in result.receipt["replayed_rows"]
    row = lib.get_row("one")
    assert row.sources["amount"]["writer_field_id"] == "amount"
    assert row.encoded["amount"] == "1"
