"""Exact type model: INT, DECIMAL, TEXT, plus nullable.

A type is one of:

    IntType(bits, nullable=False)
    DecimalType(precision, scale, nullable=False)
    TextType(nullable=False)

The *domain* of a type is the set of abstract values it can carry:

    INT(bits)      -> Python ``int`` in [-2**(bits-1), 2**(bits-1) - 1]
    DECIMAL(p, s)  -> exact ``Fraction`` c / 10**s with |c| <= 10**p - 1
    TEXT           -> Python ``str``

Plus the single value ``None`` when ``nullable`` is true.

Nothing in this module infers a type from sample data.  A declared type is
authoritative; sample values are only ever checked *against* a declaration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .errors import ContractError, DomainError

INT_BITS = (8, 16, 32, 64)
DECIMAL_PRECISION_RANGE = range(1, 19)  # 1..18 inclusive
MAX_PRECISION = 18

# A strict JSON-string fraction: optional sign, digits, and an optional
# "/" fraction.  No exponent, underscores, decimals, or surrounding space.
_CANONICAL_FRACTION_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?$")
# A contracted writer value string: the same grammar but spaces around the
# slash are tolerated (still exact; normalised later).  We keep it strict
# enough to reject exponents and float notation.
_WRITER_FRACTION_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\s*/\s*[1-9][0-9]*)?$")


# ---------------------------------------------------------------------------
# Type objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntType:
    bits: int
    nullable: bool = False

    kind = "INT"

    def __post_init__(self) -> None:
        if type(self.bits) is not int:
            raise ContractError(f"INT bits must be an int, got {self.bits!r}")
        if self.bits not in INT_BITS:
            raise ContractError(f"INT bits must be one of {INT_BITS}, got {self.bits!r}")
        if not isinstance(self.nullable, bool):
            raise ContractError("nullable must be a bool")

    @property
    def minimum(self) -> int:
        return -(2 ** (self.bits - 1))

    @property
    def maximum(self) -> int:
        return 2 ** (self.bits - 1) - 1

    def describe(self) -> str:
        return f"INT({self.bits})"


@dataclass(frozen=True)
class DecimalType:
    precision: int
    scale: int
    nullable: bool = False

    kind = "DECIMAL"

    def __post_init__(self) -> None:
        if not isinstance(self.precision, int) or isinstance(self.precision, bool):
            raise ContractError("DECIMAL precision must be an int")
        if not isinstance(self.scale, int) or isinstance(self.scale, bool):
            raise ContractError("DECIMAL scale must be an int")
        if not 1 <= self.precision <= MAX_PRECISION:
            raise ContractError(
                f"DECIMAL precision must be in 1..{MAX_PRECISION}, got {self.precision}"
            )
        if not 0 <= self.scale <= self.precision:
            raise ContractError(
                f"DECIMAL scale must be in 0..precision, got {self.scale}>{self.precision}"
            )
        if not isinstance(self.nullable, bool):
            raise ContractError("nullable must be a bool")

    @property
    def max_coefficient(self) -> int:
        return 10 ** self.precision - 1

    def describe(self) -> str:
        return f"DECIMAL({self.precision},{self.scale})"


@dataclass(frozen=True)
class TextType:
    nullable: bool = False

    kind = "TEXT"

    def __post_init__(self) -> None:
        if not isinstance(self.nullable, bool):
            raise ContractError("nullable must be a bool")

    def describe(self) -> str:
        return "TEXT"


Type = IntType | DecimalType | TextType


# ---------------------------------------------------------------------------
# Parsing / dumping type objects
# ---------------------------------------------------------------------------

def parse_type(raw: Any, nullable: bool = False) -> Type:
    """Build a type object from a ``{"kind": ...}`` JSON object.

    ``nullable`` is a sibling of ``type`` in a contract field, so it is
    passed separately by the contract parser.  Extra/unknown keys are
    rejected to catch silent typos.
    """
    if not isinstance(raw, dict):
        raise ContractError(f"type must be an object, got {type(raw).__name__}")
    kind = raw.get("kind")
    if kind == "INT":
        _expect_keys(raw, {"kind", "bits"})
        return IntType(bits=raw["bits"], nullable=nullable)
    if kind == "DECIMAL":
        _expect_keys(raw, {"kind", "precision", "scale"})
        return DecimalType(
            precision=raw["precision"], scale=raw["scale"], nullable=nullable
        )
    if kind == "TEXT":
        _expect_keys(raw, {"kind"})
        return TextType(nullable=nullable)
    raise ContractError(f"unknown type kind {kind!r}")


def _expect_keys(raw: dict, allowed: set[str]) -> None:
    extra = set(raw) - allowed
    if extra:
        raise ContractError(f"unexpected type keys: {sorted(extra)}")
    missing = allowed - set(raw)
    if missing:
        raise ContractError(f"missing type keys: {sorted(missing)}")


def type_to_json(t: Type) -> dict:
    if isinstance(t, IntType):
        return {"kind": "INT", "bits": t.bits}
    if isinstance(t, DecimalType):
        return {"kind": "DECIMAL", "precision": t.precision, "scale": t.scale}
    return {"kind": "TEXT"}


# ---------------------------------------------------------------------------
# Rational helpers
# ---------------------------------------------------------------------------

def rational_string(value: Fraction | int) -> str:
    """Canonical minimal rational string (matches reader_client expectations).

    ``Fraction`` is always normalised, so ``str`` already yields the minimal
    ``n`` or ``n/d`` form.  ``None`` is deliberately not accepted here.
    """
    f = value if isinstance(value, Fraction) else Fraction(value)
    return str(f)


def canonical_fraction_text(text: str) -> bool:
    """True iff ``text`` is already the minimal rational string of its value."""
    if not isinstance(text, str) or not _CANONICAL_FRACTION_RE.match(text):
        return False
    try:
        return str(Fraction(text)) == text
    except (ValueError, ZeroDivisionError):
        return False


def parse_exact_fraction(text: str, *, where: str = "fraction") -> Fraction:
    """Parse a schema-supplied exact fraction that may be non-minimal.

    Accepts ``"3"``, ``"6/2"``, ``"-1/10"`` and similar; rejects floats,
    exponents, underscores, decimals and surrounding whitespace.  The value is
    normalised by :class:`Fraction`, so ``"6/2"`` and ``"3"`` are equal.
    """
    if not isinstance(text, str) or not _CANONICAL_FRACTION_RE.match(text):
        raise ContractError(f"{where}: noncanonical fraction {text!r}")
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:  # pragma: no cover - regex guards
        raise ContractError(f"{where}: invalid fraction {text!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Domain membership
# ---------------------------------------------------------------------------

def int_range(bits: int) -> tuple[int, int]:
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def in_int_domain(value: int, bits: int) -> bool:
    if type(value) is not int:  # bool is a subtype of int; reject it explicitly
        return False
    lo, hi = int_range(bits)
    return lo <= value <= hi


def decimal_coefficient(value: Fraction, scale: int) -> int | None:
    """Return the exact coefficient if ``value`` is representable at ``scale``.

    ``value * 10**scale`` must be an integer; otherwise ``None``.
    """
    scaled = value * (10 ** scale)
    if scaled.denominator != 1:
        return None
    return scaled.numerator


def in_decimal_domain(value: Fraction, precision: int, scale: int) -> bool:
    if not isinstance(value, Fraction):
        return False
    coeff = decimal_coefficient(value, scale)
    if coeff is None:
        return False
    return abs(coeff) <= 10 ** precision - 1


def is_legal_value(t: Type, value: Any) -> bool:
    """Check a *decoded* Python value against a type's domain (incl. null)."""
    if value is None:
        return t.nullable
    if isinstance(t, IntType):
        return in_int_domain(value, t.bits)
    if isinstance(t, DecimalType):
        return isinstance(value, Fraction) and in_decimal_domain(
            value, t.precision, t.scale
        )
    # TEXT
    return type(value) is str


def decode_writer_value(t: Type, raw: Any, *, where: str = "value") -> Any:
    """Decode a raw JSON writer value into the abstract domain, or raise.

    Writer encodings (contracted in README):
      INT     -> JSON integer
      DECIMAL -> JSON fraction string (exact; no float literals)
      TEXT    -> JSON string (never coerced to a number)
      null    -> JSON null, only when nullable

    Booleans are never accepted as numbers.  The returned value is an ``int``,
    a ``Fraction``, a ``str`` or ``None``.
    """
    if raw is None:
        if not t.nullable:
            raise DomainError(f"{where}: null in non-nullable {t.describe()}")
        return None
    if isinstance(t, IntType):
        if type(raw) is not int:
            raise DomainError(
                f"{where}: INT requires a JSON integer, got {_jtype(raw)}"
            )
        if not in_int_domain(raw, t.bits):
            raise DomainError(
                f"{where}: {raw} outside INT({t.bits}) range "
                f"[{t.minimum},{t.maximum}]"
            )
        return raw
    if isinstance(t, DecimalType):
        if not isinstance(raw, str):
            raise DomainError(
                f"{where}: DECIMAL requires a JSON fraction string, got "
                f"{_jtype(raw)} (exact transport form; floats are never accepted)"
            )
        if not _WRITER_FRACTION_RE.match(raw):
            raise DomainError(
                f"{where}: DECIMAL requires a fraction string, got {raw!r}"
            )
        try:
            value = Fraction(raw)
        except (ValueError, ZeroDivisionError):
            raise DomainError(f"{where}: invalid fraction {raw!r}") from None
        coeff = decimal_coefficient(value, t.scale)
        if coeff is None:
            raise DomainError(
                f"{where}: {rational_string(value)} not representable at "
                f"scale {t.scale} (would require rounding)"
            )
        if abs(coeff) > t.max_coefficient:
            raise DomainError(
                f"{where}: {rational_string(value)} outside "
                f"{t.describe()} (|coefficient| {abs(coeff)} > {t.max_coefficient})"
            )
        return value
    # TEXT
    if type(raw) is not str:
        raise DomainError(f"{where}: TEXT requires a JSON string, got {_jtype(raw)}")
    return raw


def encode_reader_value(t: Type, value: Any) -> Any:
    """Encode an abstract domain value for reader-records/v1.

    INT / DECIMAL -> minimal rationale string; TEXT -> string; null -> null.
    """
    if value is None:
        return None
    if isinstance(t, TextType):
        return value
    return rational_string(value)


def decode_contract_default(t: Type, raw: Any, *, where: str) -> Any:
    """Decode a contract ``default`` (schema-side) into the abstract domain.

    Schema defaults may be given as a JSON integer or an exact fraction
    string for numeric fields, and a JSON string for TEXT.  A default is a
    *value*; it is never allowed to be null (a null default would silently
    manufacture business data), and it must lie inside the reader's domain.
    """
    if raw is None:
        raise ContractError(f"{where}: default may not be null")
    if isinstance(t, TextType):
        if type(raw) is not str:
            raise ContractError(f"{where}: TEXT default must be a string")
        return raw
    if type(raw) is int:
        value = Fraction(raw)
    elif isinstance(raw, str):
        value = parse_exact_fraction(raw, where=where)
    else:
        raise ContractError(
            f"{where}: numeric default must be an integer or exact fraction string"
        )
    if isinstance(t, IntType):
        if value.denominator != 1 or not in_int_domain(value.numerator, t.bits):
            raise ContractError(
                f"{where}: default {rational_string(value)} outside {t.describe()}"
            )
        return value.numerator
    # DECIMAL
    if not in_decimal_domain(value, t.precision, t.scale):
        raise ContractError(
            f"{where}: default {rational_string(value)} outside {t.describe()}"
        )
    return value


def _jtype(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__
