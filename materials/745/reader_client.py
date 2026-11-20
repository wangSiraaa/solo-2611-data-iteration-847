"""Consume converted reader records; this client does not infer writer compatibility."""
import json
import sys
from fractions import Fraction
from pathlib import Path


def check_value(value, field):
    if value is None:
        if not field["nullable"]:
            raise ValueError("null in required field")
        return None
    kind = field["type"]["kind"]
    if kind == "TEXT":
        if type(value) is not str:
            raise ValueError("non-text value")
        return None
    if type(value) is not str:
        raise ValueError("reader numeric values must be exact fraction strings")
    number = Fraction(value)
    if str(number) != value:
        raise ValueError("noncanonical rational")
    if kind == "INT":
        bits = field["type"]["bits"]
        if number.denominator != 1 or not -(2**(bits-1)) <= number < 2**(bits-1):
            raise ValueError("integer outside reader domain")
    elif kind == "DECIMAL":
        precision, scale = field["type"]["precision"], field["type"]["scale"]
        coefficient = number * 10**scale
        if coefficient.denominator != 1 or abs(coefficient) > 10**precision-1:
            raise ValueError("decimal outside reader domain")
    else:
        raise ValueError("unknown reader type")
    return number


def consume(contract_path, records_path):
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    doc = json.loads(Path(records_path).read_text(encoding="utf-8"))
    if doc.get("format") != "reader-records/v1" or doc.get("reader_version") != contract["version"]:
        raise ValueError("reader contract mismatch")
    fields = {f["field_id"]: f for f in contract["fields"]}
    if len(fields) != len(contract["fields"]):
        raise ValueError("duplicate reader field identity")
    totals = {key: Fraction(0) for key, f in fields.items() if f["type"]["kind"] != "TEXT"}
    seen = set()
    for row in doc["rows"]:
        if row["row_id"] in seen or set(row["values"]) != set(fields):
            raise ValueError("row identity or reader column mismatch")
        seen.add(row["row_id"])
        for key, field in fields.items():
            number = check_value(row["values"][key], field)
            if number is not None:
                totals[key] += number
    return {"row_count": len(seen), "numeric_totals": {k: str(v) for k, v in totals.items()}}


if __name__ == "__main__":
    print(json.dumps(consume(sys.argv[1], sys.argv[2]), ensure_ascii=False, sort_keys=True))
