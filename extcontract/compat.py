"""Independent finite-type compatibility analysis and adaptation planning.

The planner answers one question only:

    Is *every* legal value of each declared writer column exactly representable
    in the paired reader column, and is every reader column supplied?

It never looks at a data file, never infers a type from samples, and never
requires a target row id.  A sample that happens to fit says nothing; the
analysis is a proof over the complete declared domains.

Safety proofs (all exact, no rounding/truncation/clamping):

  nullability     nullable writer -> non-nullable reader is unsafe (witness null)
  INT(bw)->INT(br)          safe iff br >= bw
  INT(bw)->DECIMAL(p,s)     every integer n with |n| <= 2**(bw-1) must have
                            |n|*10**s <= 10**p-1
  DECIMAL(pw,sw)->INT(br)   safe iff sw == 0 and 10**pw-1 <= 2**(br-1)-1
  DECIMAL(pw,sw)->DECIMAL(pr,sr)
                            safe iff sr >= sw and
                            (10**pw-1)*10**(sr-sw) <= 10**pr-1
  TEXT->TEXT                safe (any string)
  anything else             unsafe (no implicit coercion, in particular
                            TEXT is never parsed as a number)

For every unsafe pair a concrete legal writer-domain witness is produced, so
"unsafe" is falsifiable rather than merely asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .contract import Contract, Field
from .errors import ContractError
from .types import (
    DecimalType,
    IntType,
    TextType,
    Type,
    encode_reader_value,
    rational_string,
    type_to_json,
)


@dataclass
class ColumnPlan:
    field_id: str
    source: str  # "writer" | "default"
    reader_type: Type
    safe: bool
    reasons: list[str] = field(default_factory=list)
    conversions: list[str] = field(default_factory=list)
    writer_field_id: str | None = None
    writer_name: str | None = None
    writer_type: Type | None = None
    default_encoded: Any = None
    witness: dict | None = None
    witness_explanation: str | None = None

    def to_json(self) -> dict:
        out: dict[str, Any] = {
            "field_id": self.field_id,
            "source": self.source,
            "safe": self.safe,
            "reader_type": type_to_json(self.reader_type),
            "reader_nullable": self.reader_type.nullable,
            "reasons": list(self.reasons),
            "conversions": list(self.conversions),
        }
        if self.writer_field_id is not None:
            out["writer_field_id"] = self.writer_field_id
            out["writer_name"] = self.writer_name
            out["writer_type"] = type_to_json(self.writer_type)
            out["writer_nullable"] = self.writer_type.nullable
        if self.source == "default":
            out["default"] = self.default_encoded
        if self.witness is not None:
            out["witness"] = self.witness
            out["witness_explanation"] = self.witness_explanation
        return out


@dataclass
class CompatibilityReport:
    safe: bool
    writer_version: str
    reader_version: str
    columns: list[ColumnPlan]
    ignored_writer_fields: list[dict]
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        state = "SAFE" if self.safe else "INCOMPATIBLE"
        return f"{state}: writer {self.writer_version} -> reader {self.reader_version}"

    def to_json(self) -> dict:
        return {
            "status": "SAFE" if self.safe else "INCOMPATIBLE",
            "writer_version": self.writer_version,
            "reader_version": self.reader_version,
            "errors": list(self.errors),
            "columns": [c.to_json() for c in self.columns],
            "ignored_writer_fields": list(self.ignored_writer_fields),
            "plan": self.adaptation_plan() if self.safe else None,
        }

    def adaptation_plan(self) -> dict | None:
        """Public, executable adaptation plan (only for a SAFE contract)."""
        if not self.safe:
            return None
        return {
            "writer_version": self.writer_version,
            "reader_version": self.reader_version,
            "steps": [
                {
                    "field_id": c.field_id,
                    "source": c.source,
                    **(
                        {"writer_field_id": c.writer_field_id}
                        if c.source == "writer"
                        else {"default": c.default_encoded}
                    ),
                    "conversions": list(c.conversions),
                    "reader_type": type_to_json(c.reader_type),
                    "reader_nullable": c.reader_type.nullable,
                }
                for c in self.columns
            ],
        }


def check_compatibility(writer: Contract, reader: Contract) -> CompatibilityReport:
    """Analyse the full writer->reader contract pair.  Pure, row-free."""
    if writer.role != "writer":
        raise ContractError("writer contract must be parsed with role='writer'")
    if reader.role != "reader":
        raise ContractError("reader contract must be parsed with role='reader'")

    wfields = writer.field_by_id()
    columns: list[ColumnPlan] = []
    errors: list[str] = []

    for rf in reader.fields:
        wf = wfields.get(rf.field_id)
        columns.append(_plan_column(rf, wf))

    if not reader.fields:
        errors.append("reader contract declares no columns")

    matched = {rf.field_id for rf in reader.fields}
    ignored = [
        {
            "field_id": wf.field_id,
            "name": wf.name,
            "type": type_to_json(wf.type),
            "reason": "not consumed by reader",
        }
        for wf in writer.fields
        if wf.field_id not in matched
    ]

    safe = not errors and all(c.safe for c in columns)
    return CompatibilityReport(
        safe=safe,
        writer_version=writer.version,
        reader_version=reader.version,
        columns=columns,
        ignored_writer_fields=ignored,
        errors=errors,
    )


def _plan_column(rf: Field, wf: Field | None) -> ColumnPlan:
    reader_type = rf.type

    if wf is None:
        if rf.has_default:
            return ColumnPlan(
                field_id=rf.field_id,
                source="default",
                reader_type=reader_type,
                safe=True,
                reasons=[
                    "writer omits this column; reader declares an in-domain default"
                ],
                conversions=["fill_default"],
                default_encoded=encode_reader_value(reader_type, rf.default),
            )
        reason = (
            "writer omits this column and the reader declares no default; "
            "a missing column cannot be invented (nullable does not imply a "
            "supplied value)"
        )
        return ColumnPlan(
            field_id=rf.field_id,
            source="writer",
            reader_type=reader_type,
            safe=False,
            reasons=[reason],
            witness={"missing_column": rf.field_id},
            witness_explanation="no writer value exists for a required reader column",
        )

    # Writer column exists: a default is never consulted for it, even when the
    # writer is nullable or a particular row is null.
    safe, reasons, witness, why, conversions = _pair_analysis(wf.type, reader_type)
    notes = []
    if rf.has_default:
        notes.append(
            "reader default is not used because the writer declares this field_id"
        )
    if wf.name != rf.name:
        notes.append(
            f"display name differs ({wf.name!r} vs {rf.name!r}); "
            f"pairing is by field_id only"
        )
    return ColumnPlan(
        field_id=rf.field_id,
        source="writer",
        reader_type=reader_type,
        safe=safe,
        reasons=reasons + notes,
        conversions=conversions if safe else [],
        writer_field_id=wf.field_id,
        writer_name=wf.name,
        writer_type=wf.type,
        witness=witness,
        witness_explanation=why,
    )


def _pair_analysis(
    w: Type, r: Type
) -> tuple[bool, list[str], dict | None, str | None, list[str]]:
    reasons: list[str] = []

    # 1. Nullability: a null is a legal writer value, so it must be legal for
    #    the reader.  No default may substitute for it.
    if w.nullable and not r.nullable:
        reasons.append(
            "nullable writer vs non-nullable reader: null is a legal writer value"
        )
        return (
            False,
            reasons,
            {"value": None, "encoded": None},
            "writer admits null; reader does not, and defaults never replace an "
            "existing column's row-level null",
            [],
        )
    if w.nullable:
        reasons.append("both nullable: null passes through as null")

    if isinstance(w, TextType) and isinstance(r, TextType):
        reasons.append("TEXT->TEXT: every JSON string is preserved verbatim")
        return True, reasons, None, None, ["text_identity"]

    if isinstance(w, IntType) and isinstance(r, IntType):
        if r.bits >= w.bits:
            reasons.append(
                f"INT({w.bits}) domain [{w.minimum},{w.maximum}] is contained in "
                f"INT({r.bits}) domain [{r.minimum},{r.maximum}]"
            )
            conv = ["int_identity"] if r.bits == w.bits else ["int_widen"]
            return True, reasons, None, None, conv
        witness = r.maximum + 1
        reasons.append(
            f"INT({w.bits}) exceeds INT({r.bits}): {witness} is a legal "
            f"INT({w.bits}) value but outside INT({r.bits})"
        )
        return (
            False,
            reasons,
            {"value": witness, "encoded": rational_string(witness)},
            f"{witness} fits the writer but not the reader",
            [],
        )

    if isinstance(w, IntType) and isinstance(r, DecimalType):
        bound = 2 ** (w.bits - 1)  # max |n| over the symmetric INT domain
        capacity = r.max_coefficient
        need = bound * (10 ** r.scale)
        if need <= capacity:
            reasons.append(
                f"all |n| <= 2**{w.bits - 1} = {bound}; at scale {r.scale} the "
                f"largest coefficient is {need} <= 10**{r.precision}-1 = {capacity}"
            )
            return True, reasons, None, None, ["int_to_decimal_exact"]
        # Pick a concrete legal writer value that fails.
        witness = -(2 ** (w.bits - 1))  # abs is maximal
        coeff = abs(witness) * 10 ** r.scale
        reasons.append(
            f"largest |n| = {bound} needs coefficient {need} at scale {r.scale}, "
            f"but reader capacity is {capacity}"
        )
        return (
            False,
            reasons,
            {"value": witness, "encoded": rational_string(witness)},
            f"coefficient {coeff} > {capacity} in {r.describe()}",
            [],
        )

    if isinstance(w, DecimalType) and isinstance(r, IntType):
        if w.scale == 0:
            capacity = w.max_coefficient
            if capacity <= r.maximum:
                reasons.append(
                    f"DECIMAL(_,0) integers |c| <= {capacity} fit INT({r.bits}) "
                    f"[{r.minimum},{r.maximum}]"
                )
                return True, reasons, None, None, ["decimal_to_int_exact"]
            witness = capacity
            reasons.append(
                f"DECIMAL({w.precision},0) can carry {witness}, outside "
                f"INT({r.bits}) maximum {r.maximum}"
            )
            return (
                False,
                reasons,
                {"value": rational_string(witness), "encoded": rational_string(witness)},
                f"{witness} > {r.maximum}",
                [],
            )
        witness = Fraction(1, 10 ** w.scale)
        reasons.append(
            f"scale {w.scale} > 0: {rational_string(witness)} is a legal writer "
            f"value that is not an integer"
        )
        return (
            False,
            reasons,
            {"value": rational_string(witness), "encoded": rational_string(witness)},
            "exact conversion would require truncation/rounding to an integer",
            [],
        )

    if isinstance(w, DecimalType) and isinstance(r, DecimalType):
        if r.scale < w.scale:
            witness = Fraction(1, 10 ** w.scale)
            reasons.append(
                f"scale decrease {w.scale}->{r.scale} loses values: "
                f"{rational_string(witness)} has no exact representation at "
                f"scale {r.scale}"
            )
            return (
                False,
                reasons,
                {"value": rational_string(witness), "encoded": rational_string(witness)},
                "lowering scale generally loses values; no rounding is allowed",
                [],
            )
        need = w.max_coefficient * (10 ** (r.scale - w.scale))
        capacity = r.max_coefficient
        if need <= capacity:
            reasons.append(
                f"scale preserved/increased {w.scale}->{r.scale}; largest "
                f"rescaled coefficient {need} <= {capacity}"
            )
            conv = (
                ["decimal_identity"]
                if r.scale == w.scale
                else ["decimal_exact_rescale"]
            )
            return True, reasons, None, None, conv
        witness = Fraction(w.max_coefficient, 10 ** w.scale)
        reasons.append(
            f"rescaled coefficient {need} exceeds reader capacity {capacity}"
        )
        return (
            False,
            reasons,
            {"value": rational_string(witness), "encoded": rational_string(witness)},
            f"{rational_string(witness)} does not fit {r.describe()}",
            [],
        )

    # Everything else is an explicit non-coercion boundary.
    reasons.append(
        f"no exact declared conversion from {w.describe()} to {r.describe()}; "
        f"TEXT is never implicitly parsed as numeric and numerics are not "
        f"implicitly stringified"
    )
    return (
        False,
        reasons,
        {"value": _sample_of(w), "encoded": _encoded_sample_of(w)},
        "no declared exact conversion exists between these kinds",
        [],
    )


def _sample_of(t: Type) -> Any:
    if isinstance(t, TextType):
        return "1"
    if isinstance(t, IntType):
        return 1
    return "1/10" if t.scale else "1"


def _encoded_sample_of(t: Type) -> Any:
    if isinstance(t, TextType):
        return "1"
    if isinstance(t, IntType):
        return "1"
    return rational_string(Fraction(_sample_of(t)))


# ---------------------------------------------------------------------------
# Contract revisions
# ---------------------------------------------------------------------------

def replace_field_type(
    contract: Contract, *, field_id: str, new_type_json: dict
) -> Contract:
    """Return a contract with one field's type replaced (structural revision).

    Used to model the ``reader_revision`` / ``writer_revision`` operations in
    the supplied input.  The version string is changed by the caller.
    """
    from .types import parse_type

    found = False
    new_fields = []
    for f in contract.fields:
        if f.field_id == field_id:
            found = True
            t = parse_type(new_type_json, nullable=f.type.nullable)
            new_fields.append(
                Field(
                    field_id=f.field_id,
                    name=f.name,
                    type=t,
                    default=f.default,
                    has_default=f.has_default,
                )
            )
        else:
            new_fields.append(f)
    if not found:
        raise ContractError(f"cannot revise unknown field_id {field_id!r}")
    return Contract(version=contract.version, fields=tuple(new_fields), role=contract.role)
