"""Reproducible, manifest-driven acquisition of original SEC 10-K HTML bytes.

Input: committed data/filings_manifest.csv, validated with evidenceroute.schemas.Filing.
Output: local original bytes, append-only JSONL acquisition log, optional S3 copy.
This module deliberately does NOT discover filings or parse/chunk HTML.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import requests
from botocore.exceptions import ClientError

from evidenceroute.schemas import Filing

MANIFEST_COLUMNS = (
    "document_id", "company_name", "ticker", "cik", "accession_number",
    "form_type", "fiscal_year", "report_period_end", "filing_date",
    "primary_document", "source_url",
)
RETRY_STATUSES = {429, 500, 502, 503, 504}
HTML_SIGNATURE = re.compile(rb"<\s*(?:!doctype\s+html|html\b)", re.IGNORECASE)
HEX_SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")
ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
PRIMARY_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.html?$", re.IGNORECASE)
MAX_HTML_BYTES = 50 * 1024 * 1024


class AcquisitionError(Exception):
    """A manifest, download, cache, or storage integrity problem."""


def validate_filing(filing: Filing) -> None:
    """Fail before network access if a row identifies an inconsistent SEC document."""
    if not (len(filing.cik) == 10 and filing.cik.isascii() and filing.cik.isdigit()):
        raise AcquisitionError(f"{filing.ticker}: CIK must contain exactly ten digits")
    if not ACCESSION.fullmatch(filing.accession_number):
        raise AcquisitionError(f"{filing.ticker}: malformed accession number")
    if filing.accession_number[:10] != filing.cik:
        raise AcquisitionError(f"{filing.ticker}: accession CIK does not match filing CIK")
    if filing.document_id != f"{filing.cik}_{filing.accession_number}":
        raise AcquisitionError(f"{filing.ticker}: incorrect deterministic document_id")
    if filing.form_type != "10-K":
        raise AcquisitionError(f"{filing.ticker}: only original Form 10-K is supported")
    if not PRIMARY_FILE.fullmatch(filing.primary_document) or ".." in filing.primary_document:
        raise AcquisitionError(f"{filing.ticker}: unsafe or non-HTML primary_document")
    expected = (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(filing.cik)}/{filing.accession_number.replace('-', '')}/"
        f"{filing.primary_document}"
    )
    if filing.source_url != expected:
        raise AcquisitionError(
            f"{filing.ticker} {filing.fiscal_year}: source_url is not the "
            f"canonical SEC archive URL. Expected {expected}"
        )
    if filing.raw_sha256 is not None and not HEX_SHA256.fullmatch(filing.raw_sha256):
        raise AcquisitionError(f"{filing.ticker}: invalid raw_sha256")


def load_manifest(path: Path) -> list[Filing]:
    """Read CSV without pandas (preserves leading zeroes in CIKs)."""
    if not path.is_file():
        raise AcquisitionError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise AcquisitionError("Manifest has no header")
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise AcquisitionError(f"Manifest missing columns: {', '.join(sorted(missing))}")
        filings = []
        seen_ids: set[str] = set()
        seen_pairs: set[tuple[str, int]] = set()
        for lineno, row in enumerate(reader, start=2):
            try:
                if None in row:
                    raise ValueError("extra unlabelled CSV values")
                record = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                filing = Filing.model_validate(record)
                validate_filing(filing)
                pair = (filing.cik, filing.fiscal_year)
                if filing.document_id in seen_ids or pair in seen_pairs:
                    raise AcquisitionError("duplicate document_id or CIK/fiscal_year")
                seen_ids.add(filing.document_id)
                seen_pairs.add(pair)
                filings.append(filing)
            except (ValueError, AcquisitionError) as exc:
                raise AcquisitionError(f"Manifest row {lineno}: {exc}") from exc
    if not filings:
        raise AcquisitionError("Manifest contains no filings")
    return filings


def relative_filing_path(filing: Filing) -> Path:
    return Path("sec-filings") / filing.cik / filing.accession_number.replace("-", "") / filing.primary_document


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def validate_html(prefix: bytes, size: int) -> None:
    if size < 512:
        raise AcquisitionError(f"Filing HTML is unexpectedly small ({size} bytes)")
    if not HTML_SIGNATURE.search(prefix):
        raise AcquisitionError("Response did not look like HTML")
    text = prefix.lower()
    if any(signature in text for signature in (
        b"request rate threshold exceeded", b"access denied", b"unusual traffic",
        b"your request has been denied", b"automated requests from your network",
    )):
        raise AcquisitionError("SEC returned an access/rate-limit page instead of a filing")


def load_known_checksums(path: Path) -> dict[str, str]:
    """Latest confirmed local SHA per document. Never use failed attempt as a trust anchor."""
    known: dict[str, str] = {}
    if not path.is_file():
        return known
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AcquisitionError(f"Corrupt acquisition log line {index}: {exc}") from exc
            if record.get("status") in {"local_ok", "done"}:
                digest = record.get("sha256", "")
                if not HEX_SHA256.fullmatch(str(digest)):
                    raise AcquisitionError(f"Bad checksum in acquisition log line {index}")
                document_id = record["document_id"]
                if document_id in known and known[document_id] != digest:
                    raise AcquisitionError(f"Checksum drift in acquisition log for {document_id}")
                known[document_id] = digest
    return known


def append_result(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(record)
    row["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


class FilingDownloader:
    """One-process SEC downloader with a global minimum interval between requests."""

    def __init__(
        self,
        *,
        user_agent: str,
        raw_dir: Path,
        results_path: Path,
        interval: float = 0.5,
        max_attempts: int = 4,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if "@" not in user_agent or "example." in user_agent.lower():
            raise AcquisitionError("Provide a real contact email in SEC_USER_AGENT / --user-agent")
        if interval < 0.5:
            raise AcquisitionError("SEC request interval must be at least 0.5 seconds (<=2 req/s)")
        if max_attempts < 1:
            raise AcquisitionError("max_attempts must be >= 1")
        self.user_agent = user_agent
        self.raw_dir = raw_dir
        self.results_path = results_path
        self.interval = interval
        self.max_attempts = max_attempts
        self.session = session or requests.Session()
        self.sleep = sleeper
        self.clock = clock
        self._last_request: float | None = None
        self.known = load_known_checksums(results_path)

    def _throttle(self) -> None:
        if self._last_request is not None:
            remaining = self.interval - (self.clock() - self._last_request)
            if remaining > 0:
                self.sleep(remaining)
        self._last_request = self.clock()

    def _download(self, filing: Filing, dest: Path, expected_sha: str | None) -> tuple[str, int]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, self.max_attempts + 1):
            self._throttle()
            response = None
            temp_path: Path | None = None
            try:
                response = self.session.get(
                    filing.source_url,
                    headers={"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"},
                    stream=True,
                    timeout=(10, 60),
                )
                status = response.status_code
                if status in RETRY_STATUSES:
                    if attempt == self.max_attempts:
                        raise AcquisitionError(f"SEC HTTP {status} after {attempt} attempts")
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        delay = min(max(float(retry_after), 0.0), 60.0)
                    except ValueError:
                        delay = min(2 ** attempt, 30)
                    self.sleep(max(delay, 2 ** attempt))
                    continue
                if status == 403:
                    raise AcquisitionError(
                        "SEC HTTP 403: check User-Agent, source URL and rate limits; "
                        "do not bypass SEC access controls"
                    )
                response.raise_for_status()
                final_url = urlsplit(response.url)
                if final_url.scheme != "https" or final_url.hostname not in {"sec.gov", "www.sec.gov"}:
                    raise AcquisitionError("SEC request redirected to a non-SEC destination")

                digest = hashlib.sha256()
                total = 0
                prefix = b""
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=dest.parent, prefix=".partial-", suffix=".html", delete=False
                ) as temp:
                    temp_path = Path(temp.name)
                    for block in response.iter_content(chunk_size=128 * 1024):
                        if not block:
                            continue
                        total += len(block)
                        if total > MAX_HTML_BYTES:
                            raise AcquisitionError("HTML exceeds 50 MiB safety limit")
                        digest.update(block)
                        if len(prefix) < 65536:
                            prefix += block[: 65536 - len(prefix)]
                        temp.write(block)
                validate_html(prefix, total)
                actual_sha = digest.hexdigest()
                if expected_sha and actual_sha.lower() != expected_sha.lower():
                    raise AcquisitionError(
                        f"Checksum mismatch for {filing.document_id}: "
                        f"expected {expected_sha}, got {actual_sha}. "
                        "Review before updating the frozen corpus."
                    )
                temp_path.replace(dest)
                return actual_sha, total
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ContentDecodingError,
            ) as exc:
                if attempt == self.max_attempts:
                    raise AcquisitionError(f"SEC network error after {attempt} attempts: {exc}") from exc
                self.sleep(min(2 ** attempt, 30))
            finally:
                if response is not None:
                    response.close()
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        raise AcquisitionError("Unreachable download retry state")

    def ensure_local(self, filing: Filing) -> tuple[Path, str, int, str]:
        dest = self.raw_dir / relative_filing_path(filing)
        manifest_sha = filing.raw_sha256
        ledger_sha = self.known.get(filing.document_id)
        if manifest_sha and ledger_sha and manifest_sha.lower() != ledger_sha.lower():
            raise AcquisitionError(f"Manifest and ledger checksum disagree: {filing.document_id}")
        expected_sha = manifest_sha or ledger_sha
        if dest.is_file() and expected_sha:
            actual_sha, size = sha256_file(dest)
            if actual_sha.lower() == expected_sha.lower():
                with dest.open("rb") as stream:
                    validate_html(stream.read(65536), size)
                return dest, actual_sha, size, "cached"
            # Never overwrite a corrupt or modified cache file without investigation.
            raise AcquisitionError(
                f"Local file disagrees with recorded checksum: {dest}. "
                "Investigate or move the file aside before retrying."
            )
        # Without a previously recorded digest, an existing file cannot be
        # trusted as a cache: download fresh bytes and atomically replace it.
        sha, size = self._download(filing, dest, expected_sha)
        self.known[filing.document_id] = sha
        return dest, sha, size, "downloaded"

    def acquire(
        self,
        filing: Filing,
        *,
        s3_client: Any = None,
        bucket: str | None = None,
        prefix: str = "",
    ) -> dict[str, Any]:
        validate_filing(filing)
        dest, sha, size, origin = self.ensure_local(filing)
        relative = relative_filing_path(filing).as_posix()
        base = {
            "document_id": filing.document_id,
            "ticker": filing.ticker,
            "fiscal_year": filing.fiscal_year,
            "source_url": filing.source_url,
            "local_path": str(dest),
            "sha256": sha,
            "size_bytes": size,
            "origin": origin,
        }
        # Persist local success *before* attempting S3 so S3 failures can be
        # retried without downloading from SEC again.
        append_result(self.results_path, {**base, "status": "local_ok"})
        s3_uri = None
        s3_result = "disabled"
        if s3_client is not None:
            if not bucket:
                raise AcquisitionError("S3 bucket required when S3 is enabled")
            key = "/".join(part for part in (prefix.strip("/"), "raw", relative) if part)
            s3_result = publish_s3(s3_client, bucket, key, dest, sha, size, filing.document_id)
            s3_uri = f"s3://{bucket}/{key}"
        final = {**base, "status": "done", "s3_uri": s3_uri, "s3_status": s3_result}
        append_result(self.results_path, final)
        return final


def publish_s3(
    client: Any, bucket: str, key: str, path: Path, sha: str, size: int, document_id: str
) -> str:
    """Check existing object and use conditional PUT to avoid overwriting it."""

    def matching_remote() -> bool:
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise AcquisitionError(
                f"S3 HEAD failed for s3://{bucket}/{key}: {code}. "
                "Check permissions to this prefix."
            ) from exc
        remote_sha = head.get("Metadata", {}).get("sha256", "").lower()
        remote_size = int(head.get("ContentLength", -1))
        if remote_sha != sha.lower() or remote_size != size:
            raise AcquisitionError(
                f"S3 object already exists with missing/different hash or size: "
                f"s3://{bucket}/{key}. Refusing overwrite."
            )
        return True

    if matching_remote():
        return "already_present"

    try:
        with path.open("rb") as stream:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=stream,
                ContentLength=size,
                ContentType="text/html",
                Metadata={"sha256": sha, "document-id": document_id},
                IfNoneMatch="*",
            )
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"412", "PreconditionFailed", "ConditionalRequestConflict", "409"}:
            # A second writer may have uploaded the exact same content.
            if matching_remote():
                return "already_present"
        raise AcquisitionError(f"S3 conditional upload failed: {code}") from exc
    if not matching_remote():
        raise AcquisitionError("S3 upload returned success but object could not be verified")
    return "uploaded"
