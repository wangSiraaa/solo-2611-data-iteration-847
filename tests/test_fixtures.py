"""End-to-end test driving the exact F-BAD / F-NEW documents from fixtures/."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from extcontract import IngestionLibrary, load_contract, write_records

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "materials" / "745"
FIXTURES = ROOT / "fixtures"
CLIENT = MATERIALS / "reader_client.py"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _client(records_path):
    proc = subprocess.run(
        [sys.executable, str(CLIENT), str(MATERIALS / "reader-contract.json"), str(records_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_fixture_scenario(tmp_path):
    writer = load_contract(MATERIALS / "writer-contract.json", role="writer")
    reader = load_contract(MATERIALS / "reader-contract.json", role="reader")
    library = IngestionLibrary(reader)

    assert library.ingest(writer, _load(FIXTURES / "bootstrap.json")).accepted
    assert library.numeric_totals() == {"amount": "-1"}

    bad = library.ingest(writer, _load(FIXTURES / "F-BAD.json"))
    assert not bad.accepted
    assert library.numeric_totals() == {"amount": "-1"}
    assert library.get_row("three") is None
    assert library.get_row("one").canonical["amount"] == 1

    new = library.ingest(writer, _load(FIXTURES / "F-NEW.json"))
    assert new.accepted, new.errors
    assert library.numeric_totals() == {"amount": "7"}

    out = tmp_path / "published.json"
    write_records(library.publish(), out)
    assert _client(out) == {"numeric_totals": {"amount": "7"}, "row_count": 2}


def test_fixture_file_replay_after_new(tmp_path):
    writer = load_contract(MATERIALS / "writer-contract.json", role="writer")
    reader = load_contract(MATERIALS / "reader-contract.json", role="reader")
    library = IngestionLibrary(reader)
    library.ingest(writer, _load(FIXTURES / "bootstrap.json"))
    library.ingest(writer, _load(FIXTURES / "F-NEW.json"))

    # Replaying the exact bootstrap after F-NEW is accepted (file-level replay)
    # but must not roll the target back: revisions are still 2.
    replay = library.ingest(writer, _load(FIXTURES / "bootstrap.json"))
    assert replay.accepted
    assert replay.receipt["replay"] is True
    assert library.numeric_totals() == {"amount": "7"}
