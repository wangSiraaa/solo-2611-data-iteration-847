"""Tests for file adaptation -> reader-records/v1 and the supplied reader_client."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from extcontract import (
    adapt_file_document,
    adapt_rows,
    load_contract,
    parse_contract,
    write_records,
)

MATERIALS = Path("materials/745")
CLIENT = MATERIALS / "reader_client.py"


@pytest.fixture(scope="module")
def writer():
    return load_contract(MATERIALS / "writer-contract.json", role="writer")


@pytest.fixture(scope="module")
def reader():
    return load_contract(MATERIALS / "reader-contract.json", role="reader")


def load_input():
    return json.loads((MATERIALS / "input.json").read_text(encoding="utf-8"))


def run_client(contract_path, records_path):
    proc = subprocess.run(
        [sys.executable, str(CLIENT), str(contract_path), str(records_path)],
        capture_output=True,
        text=True,
    )
    return proc


# ---------------------------------------------------------------------------
# Baseline bootstrap from provided input
# ---------------------------------------------------------------------------

def test_input_rows_adapt_and_client_totals_minus_one(writer, reader, tmp_path):
    rows = load_input()["rows"]
    result = adapt_rows(writer, reader, rows)
    assert result.status == "ADAPTED"
    records = result.records
    assert records["format"] == "reader-records/v1"
    assert records["reader_version"] == "r1"
    assert [r["row_id"] for r in records["rows"]] == ["one", "two"]
    assert [r["values"]["amount"] for r in records["rows"]] == ["1", "-2"]
    assert all(r["values"]["status"] == "ok" for r in records["rows"])

    out = tmp_path / "records.json"
    write_records(records, out)
    proc = run_client(MATERIALS / "reader-contract.json", out)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload == {"numeric_totals": {"amount": "-1"}, "row_count": 2}


def test_missing_status_filled_with_reader_default(writer, reader):
    result = adapt_rows(writer, reader, [{"row_id": "x", "values": {"amount": 5}}])
    assert result.status == "ADAPTED"
    row = result.records["rows"][0]
    assert row["values"]["status"] == "ok"
    assert row["sources"]["status"] == {
        "default": "ok",
        "conversion": "fill_default",
        "reader_type": {"kind": "TEXT"},
    }


# ---------------------------------------------------------------------------
# Whole-domain rejection is independent of the sample values
# ---------------------------------------------------------------------------

def test_contract_narrowing_rejected_even_when_rows_fit(writer):
    narrowed = parse_contract(
        {
            "version": "r1",
            "fields": [
                {
                    "field_id": "amount",
                    "name": "amount",
                    "type": {"kind": "DECIMAL", "precision": 2, "scale": 0},
                    "nullable": False,
                },
                {
                    "field_id": "status",
                    "name": "status",
                    "type": {"kind": "TEXT"},
                    "nullable": False,
                    "default": "ok",
                },
            ],
        },
        role="reader",
    )
    result = adapt_rows(writer, narrowed, [{"row_id": "one", "values": {"amount": 1}}])
    assert result.status == "INCOMPATIBLE"
    assert result.records is None
    # The single row 1 fits DECIMAL(2,0); rejection is over the full INT8 domain.
    assert any("capacity" in r for c in result.report.columns for r in c.reasons)


def test_int16_writer_rejected_by_decimal_3_0_reader(reader):
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
    result = adapt_rows(w2, reader, [{"row_id": "one", "values": {"amount": 1}}])
    assert result.status == "INCOMPATIBLE"


# ---------------------------------------------------------------------------
# Writer-domain validation occurs before conversion; no silent drops
# ---------------------------------------------------------------------------

def test_out_of_domain_integer_rejects_file(reader):
    w = parse_contract(
        {
            "version": "w9",
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
    result = adapt_rows(
        w, reader, [{"row_id": "ok", "values": {"amount": 1}},
                    {"row_id": "bad", "values": {"amount": 200}}]
    )
    assert result.status == "INVALID"
    assert result.records is None
    assert any("bad" in e.message and "outside INT(8)" in e.message for e in result.errors)
    # The valid row is not silently emitted on its own.
    assert sum(1 for e in result.errors if "ok" in (e.row_id or "")) == 0


def test_float_value_rejected_for_integer(reader):
    w = parse_contract(
        {
            "version": "w9",
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
    result = adapt_rows(w, reader, [{"row_id": "r", "values": {"amount": 1.0}}])
    assert result.status == "INVALID"
    assert any("JSON integer" in e.message for e in result.errors)


def test_decimal_inexact_input_rejected():
    w = parse_contract(
        {
            "version": "wd",
            "fields": [
                {
                    "field_id": "d",
                    "name": "d",
                    "type": {"kind": "DECIMAL", "precision": 3, "scale": 1},
                    "nullable": False,
                }
            ],
        },
        role="writer",
    )
    r = parse_contract(
        {
            "version": "rd",
            "fields": [
                {
                    "field_id": "d",
                    "name": "d",
                    "type": {"kind": "DECIMAL", "precision": 4, "scale": 2},
                    "nullable": False,
                }
            ],
        },
        role="reader",
    )
    result = adapt_rows(w, r, [{"row_id": "r", "values": {"d": "1/100"}}])
    assert result.status == "INVALID"  # scale 2 cannot be held by writer scale 1
    assert any("not representable at scale" in e.message for e in result.errors)


def test_distinct_representations_are_same_reader_value():
    w = parse_contract(
        {
            "version": "wq",
            "fields": [
                {
                    "field_id": "d",
                    "name": "d",
                    "type": {"kind": "DECIMAL", "precision": 4, "scale": 2},
                    "nullable": False,
                }
            ],
        },
        role="writer",
    )
    r = parse_contract(
        {
            "version": "rq",
            "fields": [
                {
                    "field_id": "d",
                    "name": "d",
                    "type": {"kind": "DECIMAL", "precision": 4, "scale": 2},
                    "nullable": False,
                }
            ],
        },
        role="reader",
    )
    result = adapt_rows(
        w, r, [{"row_id": "r", "values": {"d": "2/4"}}]
    )
    assert result.status == "ADAPTED"
    assert result.records["rows"][0]["values"]["d"] == "1/2"


def test_precise_decimal_rescale_1_10_to_1_100():
    w = parse_contract(
        {
            "version": "wd2",
            "fields": [
                {
                    "field_id": "d",
                    "name": "d",
                    "type": {"kind": "DECIMAL", "precision": 3, "scale": 1},
                    "nullable": False,
                }
            ],
        },
        role="writer",
    )
    r = parse_contract(
        {
            "version": "rd2",
            "fields": [
                {
                    "field_id": "d",
                    "name": "d",
                    "type": {"kind": "DECIMAL", "precision": 4, "scale": 2},
                    "nullable": False,
                }
            ],
        },
        role="reader",
    )
    result = adapt_rows(w, r, [{"row_id": "r", "values": {"d": "1/10"}}])
    assert result.status == "ADAPTED"
    assert result.records["rows"][0]["values"]["d"] == "1/10"
    assert result.records["rows"][0]["sources"]["d"]["conversion"] == "decimal_exact_rescale"


def test_null_passes_only_when_both_nullable():
    w = parse_contract(
        {
            "version": "wn",
            "fields": [
                {
                    "field_id": "d",
                    "name": "d",
                    "type": {"kind": "INT", "bits": 8},
                    "nullable": True,
                }
            ],
        },
        role="writer",
    )
    r = parse_contract(
        {
            "version": "rn",
            "fields": [
                {
                    "field_id": "d",
                    "name": "d",
                    "type": {"kind": "DECIMAL", "precision": 3, "scale": 0},
                    "nullable": True,
                }
            ],
        },
        role="reader",
    )
    result = adapt_rows(w, r, [{"row_id": "r", "values": {"d": None}}])
    assert result.status == "ADAPTED"
    assert result.records["rows"][0]["values"]["d"] is None
    assert result.records["rows"][0]["sources"]["d"]["conversion"] == "null_passthrough"


def test_duplicate_field_id_in_values_rejected(writer, reader):
    # JSON objects collapse duplicates, so a genuine duplicate field_id must be
    # rejected at the contract level instead (covered in test_compat).
    result = adapt_rows(
        writer, reader, [{"row_id": "r", "values": {"amount": 1, "bogus": 2}}]
    )
    assert result.status == "INVALID"
    assert any("undeclared writer field" in e.message for e in result.errors)


def test_file_document_version_pinning(writer, reader):
    doc = {
        "file_id": "f",
        "writer_version": "w2",
        "reader_version": "r1",
        "rows": [{"row_id": "one", "values": {"amount": 1}}],
    }
    result = adapt_file_document(writer, reader, doc)
    assert result.status == "INVALID"
    assert any("fixed writer is 'w1'" in e.message for e in result.errors)


def test_file_document_accepted(writer, reader):
    doc = {
        "file_id": "f",
        "writer_version": "w1",
        "reader_version": "r1",
        "rows": load_input()["rows"],
    }
    result = adapt_file_document(writer, reader, doc)
    assert result.status == "ADAPTED"


def test_int_to_int_widening_adapts():
    w = parse_contract(
        {"version": "wi", "fields": [
            {"field_id": "a", "name": "a", "type": {"kind": "INT", "bits": 8}, "nullable": False}]},
        role="writer",
    )
    r = parse_contract(
        {"version": "ri", "fields": [
            {"field_id": "a", "name": "a", "type": {"kind": "INT", "bits": 16}, "nullable": False}]},
        role="reader",
    )
    result = adapt_rows(w, r, [{"row_id": "x", "values": {"a": -5}},
                               {"row_id": "y", "values": {"a": 127}}])
    assert result.status == "ADAPTED"
    assert result.records["rows"][0]["values"]["a"] == "-5"
    assert result.records["rows"][1]["values"]["a"] == "127"


def test_decimal_scale0_to_int_adapts():
    w = parse_contract(
        {"version": "wd3", "fields": [
            {"field_id": "a", "name": "a", "type": {"kind": "DECIMAL", "precision": 2, "scale": 0}, "nullable": False}]},
        role="writer",
    )
    r = parse_contract(
        {"version": "rd3", "fields": [
            {"field_id": "a", "name": "a", "type": {"kind": "INT", "bits": 8}, "nullable": False}]},
        role="reader",
    )
    result = adapt_rows(w, r, [{"row_id": "x", "values": {"a": "42"}}])
    assert result.status == "ADAPTED"
    assert result.records["rows"][0]["values"]["a"] == "42"


def test_integer_rejected_for_decimal_writer_value():
    w = parse_contract(
        {"version": "wd4", "fields": [
            {"field_id": "d", "name": "d",
             "type": {"kind": "DECIMAL", "precision": 3, "scale": 1},
             "nullable": False}]},
        role="writer",
    )
    r = parse_contract(
        {"version": "rd4", "fields": [
            {"field_id": "d", "name": "d",
             "type": {"kind": "DECIMAL", "precision": 4, "scale": 2},
             "nullable": False}]},
        role="reader",
    )
    result = adapt_rows(w, r, [{"row_id": "r", "values": {"d": 1}}])
    assert result.status == "INVALID"
    assert any("fraction string" in e.message for e in result.errors)
