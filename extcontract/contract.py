"""Writer and reader contract documents.

A contract has:

    {
      "version": "<stable version string>",
      "fields": [
        {"field_id": "<stable id>", "name": "<display name>",
         "type": {...}, "nullable": <bool>,
         "default": <optional, reader contracts only>}
      ]
    }

Rules enforced here:
  * ``field_id`` is the only identity; it must be unique (duplicates reject).
  * ``name`` is display-only and may change without affecting anything.
  * ``type`` follows :mod:`extcontract.types`.
  * ``default`` is only legal on reader fields.  It is a *value* in the
    reader's own domain; it can only fill an entirely missing writer column
    (never a per-row null).
  * Extra top-level/field keys are rejected to avoid silent typos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ContractError
from .types import Type, decode_contract_default, parse_type, type_to_json

_TOP_KEYS = {"version", "fields"}
_FIELD_KEYS = {"field_id", "name", "type", "nullable", "default"}


@dataclass(frozen=True)
class Field:
    field_id: str
    name: str
    type: Type
    default: Any = None
    has_default: bool = False

    def signature(self) -> tuple:
        """Structural identity used for equality between contract versions."""
        return (
            self.field_id,
            self.type.kind,
            type_to_json(self.type).get("bits"),
            type_to_json(self.type).get("precision"),
            type_to_json(self.type).get("scale"),
            self.type.nullable,
            self.has_default,
            self.default,
        )


@dataclass(frozen=True)
class Contract:
    version: str
    fields: tuple[Field, ...]
    role: str = "reader"  # "writer" or "reader"

    def field_by_id(self) -> dict[str, Field]:
        return {f.field_id: f for f in self.fields}

    def signatures(self) -> dict[str, tuple]:
        return {f.field_id: f.signature() for f in self.fields}

    def canonical_json(self) -> dict:
        return {
            "version": self.version,
            "fields": [
                {
                    "field_id": f.field_id,
                    "name": f.name,
                    "type": type_to_json(f.type),
                    "nullable": f.type.nullable,
                    **({"default": f.default} if f.has_default else {}),
                }
                for f in self.fields
            ],
        }


def parse_contract(raw: Any, *, role: str) -> Contract:
    if role not in ("writer", "reader"):
        raise ContractError(f"role must be writer/reader, got {role!r}")
    if not isinstance(raw, dict):
        raise ContractError("contract must be a JSON object")
    extra = set(raw) - _TOP_KEYS
    missing = _TOP_KEYS - set(raw)
    if extra:
        raise ContractError(f"unexpected contract keys: {sorted(extra)}")
    if missing:
        raise ContractError(f"missing contract keys: {sorted(missing)}")

    version = raw["version"]
    if not isinstance(version, str) or not version:
        raise ContractError("contract version must be a non-empty string")
    fields_raw = raw["fields"]
    if not isinstance(fields_raw, list) or not fields_raw:
        raise ContractError("contract fields must be a non-empty list")

    fields: list[Field] = []
    seen: set[str] = set()
    for index, item in enumerate(fields_raw):
        f = _parse_field(item, role=role, index=index)
        if f.field_id in seen:
            raise ContractError(f"duplicate field_id {f.field_id!r}")
        seen.add(f.field_id)
        fields.append(f)
    return Contract(version=version, fields=tuple(fields), role=role)


def _parse_field(item: Any, *, role: str, index: int) -> Field:
    if not isinstance(item, dict):
        raise ContractError(f"field #{index} must be an object")
    extra = set(item) - _FIELD_KEYS
    if extra:
        raise ContractError(f"field #{index}: unexpected keys {sorted(extra)}")
    required = {"field_id", "name", "type"}
    if required - set(item):
        raise ContractError(f"field #{index}: missing keys {sorted(required - set(item))}")
    field_id = item["field_id"]
    if not isinstance(field_id, str) or not field_id:
        raise ContractError(f"field #{index}: field_id must be a non-empty string")
    name = item["name"]
    if not isinstance(name, str):
        raise ContractError(f"field {field_id!r}: name must be a string")

    # ``nullable`` may live either inside the type object or as a sibling.
    type_raw = dict(item["type"]) if isinstance(item["type"], dict) else item["type"]
    nullable = item.get("nullable", False)
    if isinstance(type_raw, dict) and "nullable" in type_raw:
        if "nullable" in item and item["nullable"] != type_raw["nullable"]:
            raise ContractError(f"field {field_id!r}: conflicting nullable flags")
        nullable = type_raw.pop("nullable")
    if not isinstance(nullable, bool):
        raise ContractError(f"field {field_id!r}: nullable must be a bool")
    t = parse_type(type_raw, nullable=nullable)

    has_default = "default" in item
    default: Any = None
    if has_default:
        if role == "writer":
            raise ContractError(
                f"field {field_id!r}: defaults are reader-side fills and are not "
                f"legal in a writer contract"
            )
        default = decode_contract_default(t, item["default"], where=f"field {field_id!r}")
    return Field(
        field_id=field_id,
        name=name,
        type=t,
        default=default,
        has_default=has_default,
    )


def load_contract(path: str | Path, *, role: str) -> Contract:
    import json

    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ContractError(f"contract file not found: {p}") from None
    except json.JSONDecodeError as exc:
        raise ContractError(f"contract file {p} is not valid JSON: {exc}") from None
    return parse_contract(raw, role=role)
