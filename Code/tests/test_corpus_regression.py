"""Integration coverage of the canonical 75-filing corpus and production processor."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from evidenceroute.ingestion.downloader import load_canonical_manifest, relative_filing_path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw"
OUTPUT = ROOT / "data/processed/full-v1"
HASHES = {
    "sections.parquet": "4a703731cb31a4c3a7bb4f4113b7fcd0203119fc9fd5fed072b8f6d920350b7d",
    "chunks.parquet": "edc98caf9600da7bebc457cbcd35f64b172027f1af0729e141b5d491b75279fc",
    "dataset_manifest.json": "2f8e5f3bccd1eb679446e712668d80b8f84ddad49861b3ce0c8c09d3c5199e39",
}


def assert_corpus(output):
    filings = {f.document_id: f for f in load_canonical_manifest()}
    for name, expected in HASHES.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    metadata = json.loads((output / "dataset_manifest.json").read_text())
    assert (metadata["filing_count"], metadata["section_count"], metadata["chunk_count"]) == (75, 416, 24766)
    assert {r["document_id"] for r in metadata["inputs"]} == set(filings)
    assert metadata["storage_prefix"] == "processed/full-v1/"
    sections = pq.read_table(output / "sections.parquet").to_pylist()
    chunks = pq.read_table(output / "chunks.parquet").to_pylist()
    lookup = {(s["document_id"], s["section_id"]): s["text"] for s in sections}
    assert len(lookup) == len(sections) == 416
    assert len({c["chunk_id"] for c in chunks}) == len(chunks) == 24766
    assert {c["document_id"] for c in chunks} == set(filings)
    assert sum(s["section_id"] == "item_1c" for s in sections) == 41
    for doc in filings:
        assert {"item_1", "item_1a", "item_7", "item_8", "item_15"} <= {sid for did, sid in lookup if did == doc}
    for chunk in chunks:
        text = lookup[chunk["document_id"], chunk["section_id"]]
        assert text[chunk["char_start"]:chunk["char_end"]] == chunk["text"]
        filing = filings[chunk["document_id"]]
        assert chunk["source_url"] == filing.source_url
        assert chunk["ticker"] == filing.ticker
        assert chunk["fiscal_year"] == filing.fiscal_year


def test_exported_canonical_corpus_integrity():
    if not all((OUTPUT / name).is_file() for name in HASHES):
        pytest.skip("Canonical processed corpus unavailable")
    assert_corpus(OUTPUT)


def test_processor_regenerates_all_75_filings_with_identical_hashes(tmp_path):
    filings = load_canonical_manifest()
    if any(not (RAW_DIR / relative_filing_path(f)).is_file() for f in filings):
        pytest.skip("Original SEC HTML unavailable")
    spec = importlib.util.spec_from_file_location("process_filings", ROOT / "Code/scripts/process_filings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "full-v1"
    module.process(raw_dir=RAW_DIR, output_dir=output)
    assert_corpus(output)


def test_s3_publish_is_idempotent_and_rejects_conflicts(
    tmp_path, monkeypatch
):
    import importlib.util
    import io
    import sys
    import types

    script = ROOT / "Code/scripts/process_filings.py"
    spec = importlib.util.spec_from_file_location(
        "process_filings_s3_test", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "full-v1"
    # Exercise publication independently of downloaded SEC HTML or AWS access.
    import hashlib
    import json

    output.mkdir()
    files = {}
    for name in ("sections.parquet", "chunks.parquet"):
        content = f"integrity-test bytes for {name}".encode()
        (output / name).write_bytes(content)
        files[name] = {"sha256": hashlib.sha256(content).hexdigest()}
    (output / "dataset_manifest.json").write_text(json.dumps({
        "dataset_type": "full", "storage_prefix": "processed/full-v1/", "files": files,
    }))

    objects = {}

    class FakeClientError(Exception):
        def __init__(self, code):
            self.response = {"Error": {"Code": code}}

    class FakeStreamingBody(io.BytesIO):
        def iter_chunks(self, chunk_size=1024 * 1024):
            while chunk := self.read(chunk_size):
                yield chunk

    class FakeS3:
        def get_object(self, *, Bucket, Key):
            key = (Bucket, Key)
            if key not in objects:
                raise FakeClientError("NoSuchKey")
            return {"Body": FakeStreamingBody(objects[key])}

        def put_object(
            self, *, Bucket, Key, Body, IfNoneMatch, Metadata
        ):
            assert IfNoneMatch == "*"
            key = (Bucket, Key)
            if key in objects:
                raise FakeClientError("PreconditionFailed")
            objects[key] = Body.read()
            assert Metadata["sha256"] == hashlib.sha256(objects[key]).hexdigest()

    fake_boto3 = types.SimpleNamespace(
        client=lambda service: FakeS3()
    )
    fake_exceptions = types.SimpleNamespace(
        ClientError=FakeClientError
    )

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(
        sys.modules, "botocore",
        types.ModuleType("botocore")
    )
    monkeypatch.setitem(
        sys.modules, "botocore.exceptions",
        fake_exceptions
    )

    kwargs = {
        "output_dir": output,
    }

    first = module.publish_to_s3(**kwargs)
    assert first == [
        f"s3://dats-nlp/syedibrahim.hamza@gwu.edu/EvidenceRoute/processed/full-v1/{name}"
        for name in ("sections.parquet", "chunks.parquet", "dataset_manifest.json")
    ]
    assert len(objects) == 3

    second = module.publish_to_s3(**kwargs)
    assert first == second
    assert len(objects) == 3

    key = (
        "dats-nlp",
        "syedibrahim.hamza@gwu.edu/EvidenceRoute/processed/full-v1/chunks.parquet",
    )
    objects[key] = b"corrupted"

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        module.publish_to_s3(**kwargs)

    # Local corruption must be rejected before any network mutation.
    (output / "chunks.parquet").write_bytes(b"local corruption")
    with pytest.raises(ValueError, match="Local checksum mismatch"):
        module.publish_to_s3(**kwargs)
