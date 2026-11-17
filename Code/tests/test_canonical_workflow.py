"""Production entry points cannot select a different SEC dataset."""
import importlib.util
from pathlib import Path
import sys

import pytest
from evidenceroute.ingestion.downloader import load_canonical_manifest

ROOT = Path(__file__).resolve().parents[2]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "Code/scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_downloader_acquires_every_canonical_record(monkeypatch, tmp_path):
    module = script("download_filings")
    calls = []

    class OfflineDownloader:
        def __init__(self, **kwargs):
            pass

        def acquire(self, filing, **kwargs):
            calls.append(filing.document_id)
            return {"origin": "test", "size_bytes": 1024, "sha256": "0" * 64, "s3_status": "not_requested"}

    monkeypatch.setattr(module, "FilingDownloader", OfflineDownloader)
    assert module.main(["--raw-dir", str(tmp_path)]) == 0
    expected = [f.document_id for f in load_canonical_manifest()]
    assert calls == expected
    assert len(set(calls)) == 75


@pytest.mark.parametrize("name", ["download_filings", "process_filings"])
@pytest.mark.parametrize("option", ["--manifest", "--ticker", "--year", "--dataset"])
def test_entry_points_reject_dataset_selection(name, option, monkeypatch):
    module = script(name)
    monkeypatch.setattr(sys, "argv", [name, option, "invalid"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
