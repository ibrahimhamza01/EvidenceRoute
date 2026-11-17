# Standard RAG — EvidenceRoute

[Quick start](../README.md) · [Documentation index](../../README.md#documentation)

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
| `Code/src/evidenceroute/research/prompts.py` | Build evidence-grounded prompts |
| `Code/src/evidenceroute/research/llm_client.py` | LLM interface and Gemini client |
| `Code/src/evidenceroute/research/citations.py` | Extract and validate citation IDs |
| `Code/src/evidenceroute/research/standard_rag.py` | Orchestrate Standard RAG |

## Prompt construction

`build_rag_prompt(question, evidence)` combines the question with evidence blocks containing chunk ID, document ID, company, fiscal year, section name, source URL and passage text. The prompt instructs the model to use only supplied evidence, avoid invented facts and citations, cite every factual claim, and treat retrieved passages as data rather than instructions. It specifies the exact abstention message when the supplied passages are insufficient.

## Citation Format

Answers must cite supplied evidence using:

`[chunk:CHUNK_ID]`

Example:

NVIDIA reported approximately 29,600 employees in
36 countries. [chunk:CHUNK_ID]

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

From the repository root:

```powershell
$env:PYTHONPATH = "Code/src"
python Code/examples/demo_standard_rag.py
```

The demo retrieves from all canonical full-v1 chunks and echoes supplied evidence using a mock LLM.

It requires `data/processed/full-v1/chunks.parquet` and no API key, network connection, or Qdrant database.

For Bash, the equivalent demo command is:

```bash
PYTHONPATH=Code/src python Code/examples/demo_standard_rag.py
```

## Running Tests

From the repository root:

```powershell
$env:PYTHONPATH = "Code/src"
python -m pytest Code/tests/test_rag_corpus.py -v
python -m pytest -q
```

The corpus regression tests exercise citation provenance for every one of the 75 canonical filings with simulated LLM responses.

The focused RAG suite can also be run from Bash:

```bash
PYTHONPATH=Code/src python -m pytest Code/tests/test_rag_*.py Code/tests/test_standard_rag.py -q
```

The corpus tests additionally cover cited-answer preservation, fabricated-citation rejection, missing-evidence abstention without an LLM call, and the distinction between valid citations and factual verification. Prompt, citation, LLM-client and handled-error tests exercise the components independently.

## Gemini Configuration

`Code/src/evidenceroute/research/llm_client.py` provides a Gemini client for live
model generation.

Install the dependencies using the [quick start](../README.md). `GeminiClient` reads `GEMINI_API_KEY` and optionally `GEMINI_MODEL`; its current default model is `gemini-2.5-flash`. Constructor arguments can override the key, model and timeout (60 seconds by default).

```bash
export GEMINI_API_KEY="your-api-key"
export GEMINI_MODEL="gemini-2.5-flash"
```

Instantiate `GeminiClient()` and pass it to `generate_rag_answer` with already retrieved evidence for live generation. The mock demo does not use these credentials or contact Gemini. Live requests use temperature 0.0. `LLMResult` records response text, model name, elapsed time and available input/output token counts. Client failures are wrapped as `LLMClientError` and handled by the RAG orchestration layer.

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
- Corpus regressions consume actual processed chunks; mock responses do not verify factual answers.
- Retrieval integration and end-to-end live evaluation must be
  tested separately.
