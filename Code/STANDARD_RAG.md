
# Standard RAG — EvidenceRoute

## Overview

Standard RAG generates answers to financial research questions using
retrieved evidence from SEC filings.

This implementation belongs to Issue #5.

## Workflow

1. Receive a research question.
2. Receive retrieved SEC filing chunks with metadata.
3. Construct a grounded prompt containing the evidence.
4. Send the prompt to a language model.
5. Extract chunk citations from the generated answer.
6. Validate citation IDs against the supplied evidence.
7. Return a structured `ResearchResponse`.

The response includes the answer, strategy, citations, and
verification status.

## Main Components

| File | Responsibility |
|------|----------------|
| `research/prompts.py` | Build evidence-grounded prompts |
| `research/llm_client.py` | LLM interface and Gemini client |
| `research/citations.py` | Extract and validate citation IDs |
| `research/standard_rag.py` | Orchestrate Standard RAG |

## Citation Format

Answers must cite supplied evidence using:

`[chunk:CHUNK_ID]`

Example:

NVIDIA reported approximately 29,600 employees in
36 countries. [chunk:8b7deb3e2b688baf905dd233]

The citation must refer to a chunk included in the retrieved evidence.

## Safety and Abstention

The implementation:

- Rejects empty questions.
- Abstains when no evidence is supplied.
- Abstains when the LLM produces an empty answer.
- Rejects unknown or fabricated chunk IDs.
- Abstains when an answer contains no valid citations.
- Returns a service-error response for handled LLM errors and timeouts.

A valid citation ID does not prove that the cited text supports
the answer. Therefore, successful citation validation returns
`verification_status="not_checked"`.

Semantic claim verification is outside the scope of this module.

## Running the Demo

From the `Code` directory:

```powershell
$env:PYTHONPATH = "src"
python examples/demo_standard_rag.py
```

The demo uses a verified NVIDIA SEC filing excerpt and a mock LLM.

It requires no API key, network connection, or Qdrant database.

## Running Tests

From the `Code` directory:

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests/test_rag_sec_regression.py -v
python -m pytest -q
```

The SEC regression tests use verified NVIDIA and AMD filing
excerpts with simulated LLM responses.

## Gemini Configuration

`research/llm_client.py` provides a Gemini client for live
model generation.

Install the project's dependencies and configure the required
Gemini API credentials according to the client's implementation.

The Standard RAG function accepts an LLM client implementing
the `generate(prompt)` interface.

Live model usage may incur API charges.

## Integration Notes

`generate_rag_answer(question, evidence, llm_client)` expects:

- `question`: a nonempty research question.
- `evidence`: a sequence of `RetrievalResult` objects.
- `llm_client`: an object implementing the LLM client interface.

The retrieval component is responsible for providing evidence.
Standard RAG does not perform retrieval itself.

Other project modules can call this function after retrieving
SEC filing chunks.

## Current Limitations

- Citation validation checks chunk identifiers, not factual support.
- The demo uses a mock LLM rather than live Gemini.
- The regression fixtures contain verified excerpts, not full
  processed filing chunks.
- Retrieval integration and end-to-end live evaluation must be
  tested separately.
