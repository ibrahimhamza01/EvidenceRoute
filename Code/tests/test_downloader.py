"""Offline tests: no SEC requests, AWS credentials, or S3 writes required."""

from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from evidenceroute.ingestion.downloader import (
    AcquisitionError,
    FilingDownloader,
    load_manifest,
    publish_s3,
    relative_filing_path,
    sha256_file,
)

MANIFEST = Path(__file__).resolve().parents[2] / "data/filings_manifest.csv"
VALID_HTML = b"<!DOCTYPE html><html><head><title>Form 10-K</title></head><body>" + b"report content " * 120 + b"</body></html>"


class FakeResponse:
    def __init__(self, data=VALID_HTML, status=200, url="https://www.sec.gov/test"):
        self.data = data
        self.status_code = status
        self.url = url
        self.headers = {}
        self.closed = False

    def iter_content(self, chunk_size=131072):
        for i in range(0, len(self.data), chunk_size):
            yield self.data[i:i + chunk_size]

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"http {self.status_code}")

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses=None):
        self.responses = responses or [FakeResponse()]
        self.calls = 0

    def get(self, *args, **kwargs):
        result = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return result


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.put_calls = 0

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        body, metadata = self.objects[(Bucket, Key)]
        return {"Metadata": metadata, "ContentLength": len(body)}

    def put_object(self, *, Bucket, Key, Body, ContentLength, ContentType, Metadata, IfNoneMatch):
        assert ContentType == "text/html"
        assert IfNoneMatch == "*"
        self.put_calls += 1
        if (Bucket, Key) in self.objects:
            raise ClientError({"Error": {"Code": "412", "Message": "Already exists"}}, "PutObject")
        data = Body.read()
        assert ContentLength == len(data)
        self.objects[(Bucket, Key)] = (data, Metadata)


def first_filing():
    return load_manifest(MANIFEST)[0]


def test_pilot_manifest_exactly_four_filings():
    filings = load_manifest(MANIFEST)
    assert len(filings) == 4
    assert {(f.ticker, f.fiscal_year) for f in filings} == {
        ("NVDA", 2023), ("NVDA", 2024), ("AMD", 2023), ("AMD", 2024)
    }
    assert filings[0].cik == "0001045810"
    assert relative_filing_path(filings[0]).parts[0] == "sec-filings"


def test_bad_manifest_document_id_is_rejected(tmp_path):
    raw = MANIFEST.read_text()
    bad = tmp_path / "bad.csv"
    bad.write_text(raw.replace("0001045810_0001045810-23-000017", "not-the-right-id"))
    with pytest.raises(AcquisitionError, match="document_id"):
        load_manifest(bad)


def test_bad_manifest_source_url_is_rejected(tmp_path):
    raw = MANIFEST.read_text()
    bad = tmp_path / "bad.csv"
    bad.write_text(raw.replace("https://www.sec.gov/Archives", "https://bad.example/Archives", 1))
    with pytest.raises(AcquisitionError, match="canonical SEC"):
        load_manifest(bad)


def build_downloader(tmp_path, session):
    return FilingDownloader(
        user_agent="EvidenceRoute Research student@gwu.edu",
        raw_dir=tmp_path / "raw",
        results_path=tmp_path / "raw" / "acquisition_results.jsonl",
        session=session,
        sleeper=lambda _: None,
    )


def test_first_download_and_repeat_are_idempotent(tmp_path):
    session = FakeSession()
    obj = build_downloader(tmp_path, session)
    filing = first_filing()
    first = obj.acquire(filing)
    second = obj.acquire(filing)
    assert session.calls == 1
    assert first["origin"] == "downloaded"
    assert second["origin"] == "cached"
    assert first["sha256"] == second["sha256"]
    assert first["size_bytes"] == len(VALID_HTML)
    assert (tmp_path / "raw" / relative_filing_path(filing)).read_bytes() == VALID_HTML


def test_resume_after_restart_uses_ledger(tmp_path):
    filing = first_filing()
    first_session = FakeSession()
    build_downloader(tmp_path, first_session).acquire(filing)
    second_session = FakeSession()
    result = build_downloader(tmp_path, second_session).acquire(filing)
    assert result["origin"] == "cached"
    assert second_session.calls == 0


def test_corrupt_cache_is_not_overwritten(tmp_path):
    filing = first_filing()
    downloader = build_downloader(tmp_path, FakeSession())
    downloader.acquire(filing)
    dest = tmp_path / "raw" / relative_filing_path(filing)
    dest.write_bytes(b"modified data")
    with pytest.raises(AcquisitionError, match="Local file disagrees"):
        downloader.acquire(filing)
    assert dest.read_bytes() == b"modified data"


def test_invalid_html_does_not_create_raw_file(tmp_path):
    session = FakeSession([FakeResponse(data=b"not an SEC filing" * 200)])
    downloader = build_downloader(tmp_path, session)
    with pytest.raises(AcquisitionError, match="did not look like HTML"):
        downloader.acquire(first_filing())
    assert not (tmp_path / "raw" / relative_filing_path(first_filing())).exists()


def test_retries_transient_503(tmp_path):
    session = FakeSession([FakeResponse(status=503), FakeResponse()])
    result = build_downloader(tmp_path, session).acquire(first_filing())
    assert session.calls == 2
    assert result["status"] == "done"


def test_s3_first_upload_and_repeat(tmp_path):
    s3 = FakeS3()
    local = build_downloader(tmp_path, FakeSession())
    first = local.acquire(first_filing(), s3_client=s3, bucket="test", prefix="my/EvidenceRoute")
    second = local.acquire(first_filing(), s3_client=s3, bucket="test", prefix="my/EvidenceRoute")
    assert first["s3_status"] == "uploaded"
    assert second["s3_status"] == "already_present"
    assert s3.put_calls == 1
    assert first["s3_uri"].startswith("s3://test/my/EvidenceRoute/raw/sec-filings/")


def test_s3_mismatched_existing_key_refuses_overwrite(tmp_path):
    s3 = FakeS3()
    path = tmp_path / "file.htm"
    path.write_bytes(VALID_HTML)
    sha, size = sha256_file(path)
    s3.objects[("test", "raw/doc")] = (b"another filing", {"sha256": "bad"})
    with pytest.raises(AcquisitionError, match="Refusing overwrite"):
        publish_s3(s3, "test", "raw/doc", path, sha, size, "doc")
    assert s3.put_calls == 0
