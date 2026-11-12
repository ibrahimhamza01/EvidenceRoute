"""Shared constants for retrieval tests (document IDs from data/filings_manifest.csv)."""
from pathlib import Path

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "retrieval_dev_chunks.jsonl"

NVDA_2023 = "0001045810_0001045810-23-000017"
NVDA_2024 = "0001045810_0001045810-24-000029"
AMD_2023 = "0000002488_0000002488-24-000012"
AMD_2024 = "0000002488_0000002488-25-000012"
