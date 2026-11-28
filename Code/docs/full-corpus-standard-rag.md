# Full-Corpus Standard RAG — Issue #15

## Overview

Standard RAG retrieves SEC filing passages and generates answers
with citations linked to retrieved chunk IDs.

Corpus:
- 25 companies
- Fiscal years 2022–2024
- 75 Form 10-K filings
- 416 sections
- 24,766 chunks

## Setup

From the repository root:

    source Code/.venv/bin/activate
    export PYTHONPATH=Code/src

## Offline Standard RAG

    python Code/scripts/run_standard_rag.py --question "What semiconductor supply chain risks are reported?" --top-k 5 --provider mock

Mock mode tests the pipeline and citations without making an LLM API request.
It does not generate a factual synthesis of SEC evidence.

## Company Filtering

    python Code/scripts/run_standard_rag.py --question "What supply chain risks does NVIDIA report?" --ticker NVDA --top-k 5 --provider mock

## Gemini Generation

Set GEMINI_API_KEY in the environment without committing credentials.

    python Code/scripts/run_standard_rag.py --question "What supply chain risks does NVIDIA report?" --ticker NVDA --provider gemini

## Python Integration

The reusable function is:

    from evidenceroute.research.pipeline import run_standard_rag

    response = run_standard_rag(
        question=question,
        retriever=retriever,
        llm_client=llm_client,
        top_k=5,
        filters=filters,
    )

The function returns a ResearchResponse with an answer, citations,
strategy, and verification status.

The optional metrics_callback receives RAGMetrics containing
retrieval latency, generation latency, total latency, request status,
evidence count, and citation count.

## Benchmarking

The runner uses the reviewed benchmark merged in Issue #18:

- data/benchmark/questions.csv
- data/benchmark/gold_evidence.csv

Retrieval metrics reuse the shared evidence-group evaluation functions.
Overlapping chunks within one evidence group count as alternative evidence.

Run the development split:

    python Code/scripts/benchmark_standard_rag.py --split dev --top-k 5

The runner reports recall, precision, MRR, full evidence coverage,
and retrieval latency.

Only approved, answerable questions with complete verified evidence
are included. Retrieval metrics do not establish answer correctness.

## Handoff to Agentic RAG

Ibrahim can reuse:
1. The SEC corpus and shared retrieval interfaces.
2. run_standard_rag() as a baseline workflow.
3. ResearchResponse for consistent output schemas.
4. Citation validation and abstention behavior.
5. RAGMetrics for observability.
6. The benchmark runner for future comparisons.

Agentic RAG should preserve original SEC chunk IDs and source URLs
when performing additional retrieval or reasoning steps.

## Known Limitations

- BM25 may miss semantically related evidence.
- Mock responses do not establish factual correctness.
- Valid citation IDs do not prove that the cited passage supports a claim.
- Gemini requires API credentials and network access.
- Claude integration requires confirmed provider configuration.
- Canonical corpus hashes must be checked independently.
