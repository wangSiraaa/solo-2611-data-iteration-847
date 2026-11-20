"""Tests for exact finite-type compatibility and adaptation planning."""

from __future__ import annotations

from fractions import Fraction

import pytest

from extcontract import (
    ContractError,
    DecimalType,
    IntType,
    TextType,
    check_compatibility,
    parse_contract,
    replace_field_type,
)


def _writer(fields, version="w1"):
    return parse_contract(
        {"version": version, "fields": fields}, role="writer"
    )


def _reader(fields, version="r1"):
    return parse_contract(
        {"version": version, "fields": fields}, role="reader"
    )


def _field(field_id, type_json, nullable=False, name=None, default=None):
    out = {
        "field_id": field_id,
        "name": name or field_id,
        "type": type_json,
        "nullable": nullable,
    }
    if default is not None:
        out["default"] = default
    return out


def compat_single(wtype, rtype, wnull=False, rnull=False, rdefault=None):
    w = _writer([_field("x", wtype, nullable=wnull)])
    rf = _field("x", rtype, nullable=rnull)
    if rdefault is not None:
        rf["default"] = rdefault
    r = _reader([rf])
    return check_compatibility(w, r)


# ---------------------------------------------------------------------------
# Type parsing / validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [8, 16, 32, 64])
def test_int_bits_accepted(bits):
    assert IntType(bits).bits == bits


@pytest.mark.parametrize("bits", [0, 7, 128, -8, True, 8.0])
def test_int_bits_rejected(bits):
    with pytest.raises(ContractError):
        IntType(bits)


@pytest.mark.parametrize(
    "p,s",
    [(1, 0), (18, 18), (18, 0), (5, 3)],
)
def test_decimal_valid(p, s):
    assert DecimalType(p, s).scale == s


@pytest.mark.parametrize(
    "p,s",
    [(0, 0), (19, 0), (5, 6), (3, -1), (True, 0), (3, True)],
)
def test_decimal_invalid(p, s):
    with pytest.raises(ContractError):
        DecimalType(p, s)


def test_text_rejects_nullable_nonbool():
    with pytest.raises(ContractError):
        TextType(nullable=1)


# ---------------------------------------------------------------------------
# Core domain-containment cases from the brief
# ---------------------------------------------------------------------------

def test_int8_to_decimal_3_0_safe():
    assert compat_single({"kind": "INT", "bits": 8},
                         {"kind": "DECIMAL", "precision": 3, "scale": 0}).safe


def test_int8_to_decimal_2_0_unsafe_even_if_samples_fit():
    wtype = {"kind": "INT", "bits": 8}
    rtype = {"kind": "DECIMAL", "precision": 2, "scale": 0}
    report = compat_single(wtype, rtype)
    assert not report.safe
    # Witness must be a legal INT8 value outside DECIMAL(2,0).
    witness = report.columns[0].witness
    assert witness is not None
    assert -128 <= int(witness["value"]) <= 127
    assert abs(int(witness["value"])) > 99


def test_int16_to_decimal_3_0_unsafe():
    assert not compat_single(
        {"kind": "INT", "bits": 16},
        {"kind": "DECIMAL", "precision": 3, "scale": 0},
    ).safe


def test_int16_to_decimal_5_0_safe():
    assert compat_single(
        {"kind": "INT", "bits": 16},
        {"kind": "DECIMAL", "precision": 5, "scale": 0},
    ).safe


def test_int8_to_int16_safe_and_reverse_unsafe():
    assert compat_single({"kind": "INT", "bits": 8}, {"kind": "INT", "bits": 16}).safe
    assert not compat_single({"kind": "INT", "bits": 16}, {"kind": "INT", "bits": 8}).safe


def test_decimal_3_1_to_4_2_safe():
    report = compat_single(
        {"kind": "DECIMAL", "precision": 3, "scale": 1},
        {"kind": "DECIMAL", "precision": 4, "scale": 2},
    )
    assert report.safe
    assert report.columns[0].conversions == ["decimal_exact_rescale"]


def test_decimal_scale_decrease_unsafe():
    report = compat_single(
        {"kind": "DECIMAL", "precision": 3, "scale": 1},
        {"kind": "DECIMAL", "precision": 4, "scale": 0},
    )
    assert not report.safe
    # Witness is 1/10, unrepresentable at scale 0.
    assert report.columns[0].witness["value"] == "1/10"


def test_decimal_precision_shrink_same_scale_unsafe():
    assert not compat_single(
        {"kind": "DECIMAL", "precision": 3, "scale": 0},
        {"kind": "DECIMAL", "precision": 2, "scale": 0},
    ).safe


def test_decimal_to_int_requires_scale_zero_and_range():
    assert compat_single(
        {"kind": "DECIMAL", "precision": 2, "scale": 0},
        {"kind": "INT", "bits": 8},
    ).safe
    assert not compat_single(
        {"kind": "DECIMAL", "precision": 3, "scale": 0},
        {"kind": "INT", "bits": 8},
    ).safe
    assert not compat_single(
        {"kind": "DECIMAL", "precision": 2, "scale": 1},
        {"kind": "INT", "bits": 8},
    ).safe


def test_text_only_to_text():
    assert compat_single({"kind": "TEXT"}, {"kind": "TEXT"}).safe
    assert not compat_single({"kind": "TEXT"}, {"kind": "DECIMAL", "precision": 3, "scale": 0}).safe
    assert not compat_single({"kind": "INT", "bits": 8}, {"kind": "TEXT"}).safe


def test_nullable_writer_nonnullable_reader_unsafe():
    report = compat_single(
        {"kind": "INT", "bits": 8},
        {"kind": "DECIMAL", "precision": 3, "scale": 0},
        wnull=True,
        rnull=False,
    )
    assert not report.safe
    assert report.columns[0].witness["value"] is None


def test_nullable_to_nullable_safe():
    assert compat_single(
        {"kind": "INT", "bits": 8},
        {"kind": "DECIMAL", "precision": 3, "scale": 0},
        wnull=True,
        rnull=True,
    ).safe


def test_default_does_not_rescue_nullable_unsafe():
    # A reader default may fill a missing COLUMN, never a null value.
    report = compat_single(
        {"kind": "INT", "bits": 8},
        {"kind": "DECIMAL", "precision": 3, "scale": 0},
        wnull=True,
        rnull=False,
        rdefault=0,
    )
    assert not report.safe


# ---------------------------------------------------------------------------
# Missing columns, extra columns, duplicates
# ---------------------------------------------------------------------------

def test_missing_reader_column_without_default_unsafe():
    w = _writer([_field("a", {"kind": "INT", "bits": 8})])
    r = _reader([_field("b", {"kind": "INT", "bits": 8})])
    report = check_compatibility(w, r)
    assert not report.safe
    assert "no default" in report.columns[0].reasons[0]


def test_missing_reader_column_with_default_safe():
    w = _writer([_field("a", {"kind": "INT", "bits": 8})])
    r = _reader([_field("b", {"kind": "INT", "bits": 8}, default=7)])
    report = check_compatibility(w, r)
    assert report.safe
    assert report.columns[0].source == "default"
    assert report.adaptation_plan()["steps"][0]["default"] == "7"


def test_extra_writer_columns_ignored():
    w = _writer([
        _field("a", {"kind": "INT", "bits": 8}),
        _field("extra", {"kind": "TEXT"}),
    ])
    r = _reader([_field("a", {"kind": "DECIMAL", "precision": 3, "scale": 0})])
    report = check_compatibility(w, r)
    assert report.safe
    assert [f["field_id"] for f in report.ignored_writer_fields] == ["extra"]


def test_duplicate_field_id_rejected():
    with pytest.raises(ContractError):
        _reader([
            _field("a", {"kind": "INT", "bits": 8}),
            _field("a", {"kind": "INT", "bits": 16}),
        ])


def test_renamed_display_name_still_pairs():
    w = _writer([_field("amount", {"kind": "INT", "bits": 8}, name="old_amount")])
    r = _reader([_field("amount", {"kind": "DECIMAL", "precision": 3, "scale": 0}, name="amount")])
    report = check_compatibility(w, r)
    assert report.safe


def test_writer_default_illegal():
    with pytest.raises(ContractError):
        parse_contract(
            {
                "version": "w1",
                "fields": [_field("a", {"kind": "INT", "bits": 8}, default=1)],
            },
            role="writer",
        )


def test_default_outside_reader_domain_rejected():
    with pytest.raises(ContractError):
        _reader([_field("a", {"kind": "DECIMAL", "precision": 2, "scale": 0}, default=1000)])


def test_default_fraction_string_canonical():
    r = _reader([
        _field(
            "a",
            {"kind": "DECIMAL", "precision": 3, "scale": 1},
            default="1/10",
        )
    ])
    assert r.fields[0].default == Fraction(1, 10)


def test_default_with_scale_mismatch_rejected():
    with pytest.raises(ContractError):
        _reader([
            _field(
                "a",
                {"kind": "DECIMAL", "precision": 3, "scale": 1},
                default="1/100",  # not representable at scale 1
            )
        ])


# ---------------------------------------------------------------------------
# Revision ops from input.json
# ---------------------------------------------------------------------------

def test_reader_revision_narrow_unsafe():
    import json
    from pathlib import Path

    base = json.loads(
        Path("materials/745/reader-contract.json").read_text()
    )
    reader = parse_contract(base, role="reader")
    rev = replace_field_type(
        reader,
        field_id="amount",
        new_type_json={"kind": "DECIMAL", "precision": 2, "scale": 0},
    )
    writer = _writer([_field("amount", {"kind": "INT", "bits": 8})])
    assert not check_compatibility(writer, rev).safe


def test_writer_revision_int16_untrusted_by_r1():
    import json
    from pathlib import Path

    reader = parse_contract(
        json.loads(Path("materials/745/reader-contract.json").read_text()),
        role="reader",
    )
    w = _writer([_field("amount", {"kind": "INT", "bits": 16})], version="w2")
    assert not check_compatibility(w, reader).safe


# ---------------------------------------------------------------------------
# Domain completeness spot checks against exhaustive enumeration
# ---------------------------------------------------------------------------

def test_int8_to_decimal_exhaustive():
    """INT8 -> DECIMAL(p,0): safe iff all of -128..127 fit."""
    for p in range(1, 5):
        expected = all(abs(n) <= 10**p - 1 for n in range(-128, 128))
        got = compat_single(
            {"kind": "INT", "bits": 8},
            {"kind": "DECIMAL", "precision": p, "scale": 0},
        ).safe
        assert got == expected, p


def test_decimal_to_decimal_exhaustive_scale_preserved():
    for pw, sw, pr in [(1, 0, 1), (2, 0, 2), (2, 1, 2), (3, 1, 3), (1, 1, 2)]:
        for sr in range(sw, sw + 2):
            if sr > pr:
                continue
            worst = (10**pw - 1) * 10 ** (sr - sw)
            expected = worst <= 10**pr - 1
            got = compat_single(
                {"kind": "DECIMAL", "precision": pw, "scale": sw},
                {"kind": "DECIMAL", "precision": pr, "scale": sr},
            ).safe
            assert got == expected, (pw, sw, pr, sr)


def test_default_nonminimal_fraction_normalized():
    r = _reader([
        _field("a", {"kind": "DECIMAL", "precision": 3, "scale": 0}, default="6/2")
    ])
    assert r.fields[0].default == Fraction(3)


def test_default_floats_and_exponents_rejected():
    for bad in [1.5, "1e3", "0.5", " 1", "1 ", "1_0"]:
        with pytest.raises(ContractError):
            _reader([
                _field("a", {"kind": "DECIMAL", "precision": 3, "scale": 1}, default=bad)
            ])


def test_decimal_scale_increase_but_precision_too_small_unsafe():
    # (10^2-1)*10^(1) = 990 > 99 -> DECIMAL(2,1) cannot hold all of DECIMAL(2,0)
    assert not compat_single(
        {"kind": "DECIMAL", "precision": 2, "scale": 0},
        {"kind": "DECIMAL", "precision": 2, "scale": 1},
    ).safe
    # DECIMAL(2,0) -> DECIMAL(3,1): 99*10=990 <= 999, safe.
    assert compat_single(
        {"kind": "DECIMAL", "precision": 2, "scale": 0},
        {"kind": "DECIMAL", "precision": 3, "scale": 1},
    ).safe


def test_witness_is_always_a_domain_member():
    """Every reported witness must itself be legal for the writer type."""
    from extcontract.types import parse_type, decode_writer_value

    cases = [
        ({"kind": "INT", "bits": 16}, {"kind": "DECIMAL", "precision": 3, "scale": 0}),
        ({"kind": "INT", "bits": 8}, {"kind": "DECIMAL", "precision": 2, "scale": 0}),
        ({"kind": "DECIMAL", "precision": 3, "scale": 1}, {"kind": "DECIMAL", "precision": 4, "scale": 0}),
    ]
    for wtype, rtype in cases:
        report = compat_single(wtype, rtype)
        assert not report.safe
        w = parse_type(wtype)
        witness = report.columns[0].witness
        decoded = decode_writer_value(w, witness["value"])
        # and it must NOT fit the reader
        rt = parse_type(rtype)
        from extcontract.types import is_legal_value
        assert not is_legal_value(rt, decoded)
