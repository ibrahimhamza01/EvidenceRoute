"""Process SEC filings and automatically generate Parquet and dataset metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evidenceroute.ingestion.downloader import load_manifest, relative_filing_path
from evidenceroute.ingestion.parser import extract_sections
from evidenceroute.ingestion.processor import chunk_sections, write_parquet


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_to_s3(
    *,
    output_dir: Path,
    dataset: str,
    bucket: str,
    prefix: str,
) -> list[str]:
    """Publish verified dataset artifacts without overwriting S3 objects."""
    import boto3
    from botocore.exceptions import ClientError

    if dataset not in ("pilot", "full"):
        raise ValueError(f"Unsupported dataset: {dataset}")

    dataset_prefix = (
        "processed/pilot-v1/" if dataset == "pilot"
        else "processed/"
    )
    base_key = f"{prefix.strip('/')}/{dataset_prefix}".lstrip("/")

    manifest_path = output_dir / "dataset_manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))

    if metadata["dataset_type"] != dataset:
        raise ValueError("Dataset mode does not match generated manifest")
    if metadata["storage_prefix"] != dataset_prefix:
        raise ValueError("Manifest S3 prefix does not match dataset mode")

    filenames = (
        "sections.parquet",
        "chunks.parquet",
        "dataset_manifest.json",
    )

    expected_hashes = {
        name: sha256_file(output_dir / name)
        for name in filenames
    }

    for name in filenames[:2]:
        if metadata["files"][name]["sha256"] != expected_hashes[name]:
            raise ValueError(f"Local checksum mismatch: {name}")

    s3 = boto3.client("s3")

    def remote_hash(key: str) -> str | None:
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in (
                "NoSuchKey", "404"
            ):
                return None
            raise

        digest = hashlib.sha256()
        body = response["Body"]
        try:
            for block in body.iter_chunks(chunk_size=1024 * 1024):
                if block:
                    digest.update(block)
        finally:
            body.close()
        return digest.hexdigest()

    # Preflight: detect conflicts before uploading anything.
    existing = {}
    for name in filenames:
        key = base_key + name
        actual = remote_hash(key)
        if actual is not None and actual != expected_hashes[name]:
            raise RuntimeError(
                f"Refusing to overwrite different S3 object: "
                f"s3://{bucket}/{key}"
            )
        existing[name] = actual is not None

    published = []

    # Upload the manifest last so it marks a complete dataset.
    for name in filenames:
        key = base_key + name
        uri = f"s3://{bucket}/{key}"

        if not existing[name]:
            try:
                with (output_dir / name).open("rb") as handle:
                    s3.put_object(
                        Bucket=bucket,
                        Key=key,
                        Body=handle,
                        IfNoneMatch="*",
                        Metadata={"sha256": expected_hashes[name]},
                    )
            except ClientError as exc:
                # A concurrent publisher may have created the object.
                if exc.response["Error"]["Code"] not in (
                    "PreconditionFailed", "412"
                ):
                    raise

            if remote_hash(key) != expected_hashes[name]:
                raise RuntimeError(
                    f"S3 checksum verification failed: {uri}"
                )

        published.append(uri)

    return published


def process(*, manifest: Path, raw_dir: Path, output_dir: Path,
            dataset: str, max_chars: int, overlap_chars: int) -> dict:
    if max_chars <= 0 or not 0 <= overlap_chars < max_chars:
        raise ValueError("Require max_chars > 0 and 0 <= overlap_chars < max_chars")
    filings = list(load_manifest(manifest))
    if not filings:
        raise ValueError("Refusing to process an empty filing manifest")
    if dataset == "pilot" and len(filings) != 4:
        raise ValueError("Pilot mode requires the four-filing pilot manifest")

    sections, chunks, inputs, failures = [], [], [], []
    for filing in filings:
        path = raw_dir / relative_filing_path(filing)
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            extracted = extract_sections(path, filing.document_id, filing.source_url)
            generated = chunk_sections(extracted, filing, max_chars=max_chars,
                                       overlap_chars=overlap_chars)
            sections.extend(extracted)
            chunks.extend(generated)
            inputs.append({"document_id": filing.document_id,
                           "relative_path": relative_filing_path(filing).as_posix(),
                           "sha256": sha256_file(path)})
        except (OSError, ValueError) as exc:
            failures.append({"document_id": filing.document_id, "error": str(exc)})
    if failures:
        raise RuntimeError(f"Processing incomplete; no outputs written: {json.dumps(failures)}")

    section_lookup = {(s.document_id, s.section_id): s.text for s in sections}
    if len(section_lookup) != len(sections):
        raise ValueError("Duplicate section IDs within a filing")
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs")
    for chunk in chunks:
        text = section_lookup[(chunk.document_id, chunk.section_id)]
        if text[chunk.char_start:chunk.char_end] != chunk.text:
            raise ValueError(f"Invalid chunk offsets: {chunk.chunk_id}")

    write_parquet(sections, chunks, output_dir)
    files = {}
    for name, count in (("sections.parquet", len(sections)),
                        ("chunks.parquet", len(chunks))):
        files[name] = {"records": count, "sha256": sha256_file(output_dir / name)}

    metadata = {
        "dataset_name": "pilot-v1" if dataset == "pilot" else "full",
        "dataset_type": dataset,
        "storage_prefix": "processed/pilot-v1/" if dataset == "pilot" else "processed/",
        "production_storage_prefix": "processed/",
        "filing_count": len(filings),
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "extracted_items": ["1", "1A", "7", "8", "15"],
        "chunking": {"max_chars": max_chars, "overlap_chars": overlap_chars},
        "inputs": inputs,
        "files": files,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--manifest", type=Path, default=Path("data/filings_manifest.csv"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--overlap-chars", type=int, default=200)
    ap.add_argument(
        "--s3",
        action="store_true",
        help="Publish validated output to S3",
    )
    ap.add_argument(
        "--s3-bucket",
        default="dats-nlp",
    )
    ap.add_argument(
        "--s3-prefix",
        default="syedibrahim.hamza@gwu.edu/EvidenceRoute",
    )
    args = ap.parse_args()
    output_dir = args.output_dir or (Path("data/processed/pilot-v1")
                                     if args.dataset == "pilot"
                                     else Path("data/processed"))
    result = process(manifest=args.manifest, raw_dir=args.raw_dir,
                     output_dir=output_dir, dataset=args.dataset,
                     max_chars=args.max_chars, overlap_chars=args.overlap_chars)
    print(json.dumps({"output_dir": str(output_dir), "filings": result["filing_count"],
                      "sections": result["section_count"],
                      "chunks": result["chunk_count"],
                      "files": ["sections.parquet", "chunks.parquet", "dataset_manifest.json"]},
                     indent=2))
    print("Validate Item boundaries and financial tables before publishing to S3.")

    if args.s3:
        published = publish_to_s3(
            output_dir=output_dir,
            dataset=args.dataset,
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
        )
        print("S3 publication verified:")
        for uri in published:
            print(f"  {uri}")


if __name__ == "__main__":
    main()