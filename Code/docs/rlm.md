# Recursive language model research

## Architecture and control flow

1. `RootController` sends the question, tool schemas, previous observation, and remaining time to a model client. The scripted client plays back predetermined decisions for offline tests.
2. A provider reply is one model decision. It can contain one tool call, an ordered batch of up to four tool calls, or an exclusive finish. The next request replays the original assistant message and sends one user message containing a result for each native tool-use ID. [Bedrock Converse tool use](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-client-side.html) · [Anthropic tool result ordering](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use)
3. `CorpusToolDispatcher` validates the name and arguments against the same models that publish the schemas. It calls only allowlisted local operations and returns a bounded JSON success or error. Invalid arguments become feedback the model can correct.
4. The controller accepts a structured finish only when its cited chunk IDs were delivered to the model. It records requested decisions, validated arguments, observations, original evidence IDs, provider attempts, timing, and stop reason.

`CorpusEnvironment` provides validated, read-only access to the canonical SEC corpus of **25 companies, 75 filings, 416 sections, and 24,766 chunks**. It loads `data/filings_manifest.csv` and the section and chunk Parquet files under `data/processed/full-v1/`. The tools list companies, filings, and sections, fetch filing metadata or original chunks, page through section text, and search with BM25. The corpus stays local. The provider receives the question and the observations accumulated during the run.

The caller can supply required filing and section IDs through `ResearchScope`. An unspecified requirement remains unknown.

### Research scope planning

The interpreter identifies companies, fiscal years, section IDs, the topic and objective, and whether the question requires comprehensive review. It marks inferred requirements and explains its assumptions. The host preserves the original question. Scripted callers can still pass an `IntentInterpreter` to `interpret_intent`. The natural-language entry point requests requirements and discovery phrases from Bedrock and charges interpretation to the overall research allowance. [Natural-language question interpretation](#natural-language-question-interpretation)

Python pages through the filing and section metadata without search. Company selectors match a ticker first, then an exact name recorded in the manifest, ignoring case and extra whitespace. Matched aliases collapse to their canonical ticker. Targets are unique and sorted by ticker, fiscal year, and section ID. Unknown companies, ambiguous matches, and unavailable filings or sections remain visible as unresolved company/year/section slots in the required count.

Each company group keeps its resolved fiscal years and the original question together. The [live semantic review](#reading-the-report) shows why that matters. Independently choosing excerpts for each year can make different topics look like a change in disclosure. A shared group lets the analyst check the same risk in each filing. Even matched excerpts need adequate coverage before making an absence claim.

The plan stays host-side. `execute_research_plan` validates provider reservations and admits whole groups in ticker order within the group, shared, and root limits. Unresolved groups and groups outside those limits retain a pending reason. Comprehensive plans default to delegation-first execution. Admission reserves enough root decisions for one assignment per group and enough shared delegation and partition slots for every admitted group. The full plan remains the reporting denominator. Provider, token, and time limits can still stop execution after admission.

### Bounded recursive execution

`RecursiveResearch` opts into recursive execution. Its factory creates a fresh budget-aware model client per invocation, and `RootController` handles each conversation. The model supplies a question, a goal, and a nonempty set of unique canonical document/section pairs to `delegate_research`. The root can assign all its authorized pairs to a fresh child to investigate a narrower question over the same evidence. With no explicit authorization, it can select any valid canonical pairs. Below the root, every successful delegation must remove at least one pair. The scope therefore shrinks along each branch, and a singleton cannot be subdivided under the current pair representation.

Each child starts with its own question, goal, authorized pairs, and an empty model history. A Python corpus view constrains metadata, reads, lookups, and search. Filtering exact pairs **before counting and slicing** makes totals, offsets, and coverage denominators refer to the authorized collection. For example, if ranked hits are `[outside, A, B]`, a one-hit first page returns `A` with total `2` and next offset `1`. Filtering after slicing could leave an empty page or retain the wrong total. BM25 scoring statistics still come from the corpus-wide index.

A bounded delegation observation carries the child's stop reason, analysis status, partition coverage, and validated findings. A failed child can return retained findings with its failed status. Only findings present in an acknowledged observation grant access to the delivered excerpts. The trace records retention, offers, attempted delivery, acknowledgement, and terminal selection. A declared finding must contribute evidence to the answer or a new finding.

| Signal | What it establishes |
| --- | --- |
| Evidence access and coverage | Direct chunk access supplies the whole original chunk. A delegated finding supplies only its delivered excerpt. A section window advances text coverage but supplies no chunk IDs. The per-invocation tracker still reports acknowledged pagination. Planned execution also rolls up original chunk IDs and section-window ranges across the tree against the full plan. |
| Analysis status | The model declares its assessment, uncertainty, and conflicts. The controller defaults the invocation to `incomplete` when no status or supporting references are supplied. A validated `finished` action can still have partial coverage or incomplete analysis. A search miss cannot establish that a disclosure was absent. |
| Citation and quotation validation | Citation IDs must refer to evidence the invocation received. The host resolves selected span IDs to exact canonical text and metadata. Legacy quotes must occur in the original chunk. Evidence drawn solely from a delivered finding stays within its delivered excerpts and requires selecting or declaring that finding. |
| Semantic support | Exact citations establish source and wording. An optional separate invocation assesses entailment. [Semantic verification](#independent-semantic-verification) |

This is **RLM-inspired recursive long-context processing**. The original paper uses a programmable REPL to inspect, partition, and recurse over external input. EvidenceRoute exposes typed corpus and delegation tools for this SEC workflow. [Recursive Language Models](https://arxiv.org/abs/2512.24601)

#### Shared budget and termination

Root and descendants draw from one `SharedBudget`. A model session is a fresh client, a decision is one invocation of that client, a provider attempt is one transport request including a retry, and a corpus attempt is a dispatcher call. Local controller iteration and tool-attempt limits still apply. A delegation consumes a local tool slot, and shared delegation and partition budgets are charged before the child starts. Repeated assignment of a partition consumes another assignment slot.

| Condition | Effect |
| --- | --- |
| Valid finish | Ends the invocation as `finished`. Invalid actions, citations, or findings return feedback for correction while decisions remain available. |
| Depth, scope, or shared delegation and partition limits | Reject the proposed child with feedback to the parent. Shared defaults allow depth 2 below the root, 8 admitted children, and 16 cumulative pair assignments. The assignment cap also bounds explicit root authorization. The delegation tool is omitted at the depth ceiling, and runtime validation still guards any attempted call. |
| Shared provider or corpus attempt limits | The defaults are 24 provider attempts and 32 corpus attempts. Provider exhaustion ends an invocation when the adapter needs another request or retry. Corpus exhaustion rejects that dispatch and can still leave room for the model to finish. |
| Local controller limits | Exhausting iterations ends with `iteration_limit`. With protected finalization, the local tool cap or last decision triggers finish-only mode. Without it, another single tool request after the local tool budget is spent ends with `tool_call_limit`. |
| Measured token threshold | At 60,000 reported tokens by default, further decisions, requests, and tool work are blocked. An already returned valid finish can still be accepted. Successful and failed attempts contribute available input, output, and cache counters. Missing input or output counters are tracked separately, and a response can cross the threshold before usage is known. |
| Cooperative deadline | The controller uses the earliest of the shared deadline (90 seconds by default), its local deadline, and the parent's deadline. Providers cap request timeouts by remaining time. Late decisions are discarded. Synchronous SDK calls can overrun the deadline. |
| Model or provider failure | Retryable provider errors can retry within configured attempt and time limits. Truncated investigation replies can transition to protected finalization when a finish attempt remains. Other unhandled failures end the invocation, normally as `model_error`. The parent receives the child's failed outcome and any retained findings that fit the observation. |

Batch operations execute sequentially in provider order. One batch spends one model decision, and each admitted operation spends its own local tool slot and applicable corpus or delegation budget. The controller checks those limits before each operation. An invalid operation can return an error while its peers continue. A request skipped at a local or shared limit gets its own error result when a follow-up is allowed. The local tool limit can skip the rest of a batch and still allow a later finish. Running out of provider attempts blocks further batch work that would require another provider turn.

Four operations is the execution limit. A well-formed native batch of five through sixteen is rejected in full and receives a `tool_batch_limit` result for every ID. The original assistant message stays in history for that follow-up. Sixteen is a protocol safety ceiling for sending complete rejection feedback. Batches above it and malformed or duplicate IDs stop the invocation. Rejection spends the current model decision and provider attempt but performs no corpus work or delegation. Any correction uses the remaining budget, including the existing finish-only rules once protected finalization begins.

The observation byte limit applies to each result and the combined batch envelope. The controller reserves space for all IDs and compact errors before execution. If the envelope cannot fit, it stops without executing the batch. Otherwise earlier results get first use of the remaining bytes. An oversized result is replaced by an error and leaves the original evidence and pagination intact. Coverage and citation eligibility require both a successful result and acknowledged delivery of that result. An acknowledgement of the batch's error block grants no evidence.

#### Offline example

Run the scripted example with local canonical files. It delegates FY2023 and FY2024 evidence collection, and the FY2024 child asks a grandchild to inspect Item 1A. It prints the tree, findings, original citations, coverage, and shared counters. The scripts illustrate the control flow. Semantic comparison and provider inference are outside this example.

```bash
PYTHONPATH=Code/src python Code/examples/demo_rlm_recursive.py
```

## Benchmark and router integration

Use `evidenceroute.research.rlm.natural_language.generate_rlm_answer` for the shared `ResearchResponse` defined in `schemas.py`. Use `research_question` when the caller also needs diagnostics. Reuse a `CorpusEnvironment` across questions. Configure a `RecursiveResearch` runner with a fresh client factory, such as `lambda _: BedrockRootClient.from_env(disable_retries=True)`, and explicit `CorpusLimits`. [Question demo](../examples/demo_rlm_question.py)

```python
from evidenceroute.research.rlm.natural_language import research_question

result = research_question(question, runner, limits=limits, max_groups=2)
response = result.response
research = result.research
```

Allow for interpretation and at least one full tree allocation in `limits`. Use `interpreter_factory` to substitute a scripted interpreter offline. The natural-language path handles descriptive SEC reviews and supported two-year AI-infrastructure comparisons. If `research` is absent, record `result.interpretation.outcome` and overall usage. Ambiguity, unsupported scope, failure and exhausted resources have different meanings from an evidence-based abstention.

For a fixed-scope experiment, resolve a caller-supplied `ResearchIntent` into a `ResearchPlan` and call `execute_corpus_research(plan, runner, limits=limits, discovery=spec)` with a `DiscoverySpec` for the scoped answer path. Record which entry point was used since fixing scope bypasses interpretation.

| Integration field | Source |
| --- | --- |
| Question and run identity | Keep the harness's `question_id` and the original question. Attach `research.run_id` when a research result exists. |
| Answer and original SEC citations | Use `response.answer` and `response.citations`. The response uses `strategy="rlm"` and resolves answer citations to canonical `Chunk` records. Generic conclusions cite their bound chunks. Exact excerpts, span IDs, and the wider citation inventory remain in `research`. |
| Verification status | The question API returns `not_checked`. Provenance checks confirm source identity and delivery. Independent semantic review runs under its own allowance and has its own report. |
| Scope, coverage and uncertainty | Keep `research.coverage.required_target_count`, per-target access, missing requirements and `research.groups` with pending reasons. Read `Finding.uncertainty`, `Finding.conflict`, analysis status and comparison outcomes. Retained findings and discovered chunks may extend beyond the answer's citations. |
| Completion and errors | Keep `execution_status`, `research_status`, `stop_reason` and errors. Execution is `finished` when at least one batch ran and no required group remains pending. Research may still be `incomplete`. `unverified` records that the pipeline found no tracked incompleteness and leaves semantic certainty unresolved. |
| Usage and latency | Use `result.usage` for interpretation plus research, or `research.usage` for research alone. Record reported tokens, estimates, missing-counter counts, provider attempts and `elapsed_seconds`. The overall timer starts after the corpus is loaded. |

Verification is optional. When a research result exists, call `create_checkpoint(corpus, result)`, then `verify_checkpoint(checkpoint, corpus, verifier, limits=verification_limits)` under a separate allowance. Attach the result with `record_verification` to obtain `verification_report` and combined usage. Preserve the original result separately if the interface needs its exact answer text. The checkpoint stores structured research state and coverage. Use explicit continuation for a chosen evidence gap.

Map the result inside the existing benchmark by its question ID. Compare original citation IDs with supporting evidence groups and score answer quality and abstention separately. Citation IDs are sorted for reporting and carry no retrieval rank for MRR or ranked top-K metrics. Keep the question inputs, budget and verification policy fixed across comparable trials. The router can treat `corpus_wide` as a candidate for RLM and use the existing strategy dispatch. Routing confidence describes predicted scope. Use held-out answer, evidence, latency and usage results to assess the strategy choice. [Benchmark guide](evaluation.md) · [Routing guide](routing.md)

## Course setup with Amazon Bedrock

Bedrock is AWS's managed model API. EvidenceRoute uses Boto3's `bedrock-runtime` client to sign and send Converse requests. AWS authentication and **Claude Sonnet 4.5** access were verified for the course account in `us-east-1`. The account's IAM permissions denied Sonnet 4.6. [AWS Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)

| Setting | Behavior |
| --- | --- |
| `BEDROCK_MODEL_ID` | Required. The verified Sonnet 4.5 profile is `us.anthropic.claude-sonnet-4-5-20250929-v1:0`. |
| AWS region | `BEDROCK_REGION`, then `AWS_REGION`, then the Boto3 session region, which can come from `AWS_DEFAULT_REGION` or an AWS profile. |
| AWS credentials | Boto3's standard provider chain. For manually supplied temporary environment credentials, supply `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` together. Use `AWS_PROFILE` for an externally configured profile. Environment credentials can take precedence over a profile. |
| `CLAUDE_TIMEOUT_SECONDS` | 20 seconds by default. Each attempt caps its connect and read timeouts by remaining run time. |
| `CLAUDE_MAX_OUTPUT_TOKENS` | 1,024 generated tokens per response by default. |
| `CLAUDE_MAX_RETRIES` | One retry per decision by default, configurable from zero to two. |
| `CLAUDE_TEMPERATURE`, `CLAUDE_TOP_P` | Omitted by default. At most one can be set when supported by the selected model. |

Other accounts and profiles require their own region and `bedrock:InvokeModel` permissions. [Converse API permissions](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)

Course credentials reportedly expire about every four hours. Manually supplied environment credentials do not refresh automatically. On expiry, obtain a fresh credential set from the course issuer, replace all three values in the shell environment or externally managed profile, and start a new process. Remove stale environment credentials if switching to a profile. The application delegates credential retrieval and any provider-supported refresh to Boto3. Keep credentials outside the repository and leave SDK wire debugging off when using real credentials. Expired or missing credentials, invalid authentication, access denial, and provider request configuration errors end the affected invocation without retries and appear as sanitized trace codes. [Boto3 credential provider chain](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html)

The optional demo starts with one search hit for a NVIDIA FY2024 risk disclosure. It reports the configured limits, coverage, citations, request attempts, tokens, latency, and stop reason. The `--trace` flag adds requested actions, validated arguments, errors, evidence IDs, and observation delivery status. It omits raw tool-result text. The `--live` flag authorizes provider calls.

```bash
BEDROCK_REGION=us-east-1 BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  PYTHONPATH=Code/src python Code/examples/demo_rlm_claude.py --provider bedrock --live --trace
```

Direct Anthropic Messages remains available through `--provider anthropic --live` for environments with a separate `ANTHROPIC_API_KEY`. It uses optional `ANTHROPIC_BASE_URL` and `CLAUDE_MODEL` (default `claude-sonnet-4-6`) and shares the `CLAUDE_*` request limits above.

## Design decisions and trade-offs

| Decision | Trade-off |
| --- | --- |
| RLM-01 · Validated local snapshot | Canonical schemas and defensive copies preserve text and offsets. Loading the snapshot uses more memory than streaming. |
| RLM-02 · Exact pages and BM25 | Results are reproducible with chunk-ID tie breaking and a lazy index. Lexical search can miss different wording. |
| RLM-03 · Typed, allowlisted tools | The same argument models define schemas and validate requests. Adding operations requires a registry change. |
| RLM-04 · Whole-result byte handling | Rejecting oversize results preserves offsets and provenance. A smaller page may require another turn. |
| RLM-05 · Separate access, scope and analysis | Acknowledged pagination measures access. An unspecified requirement stays unknown, and analysis is model-reported. |
| RLM-06 · Native tool batches | Preserve assistant messages and result IDs in order. A recovery response consumes a turn, and each admitted operation consumes its own tool slot. |
| RLM-07 · SDK credential and timeout handling | Boto3 resolves credentials and the application owns retries. Blocking calls can overrun the cooperative deadline. |
| RLM-08 · Per-attempt telemetry | Report available input, output, cache and latency counters. A lost provider response can leave usage and delivery unknown. |
| RLM-09 · One-chunk defaults | Small observations conserve context and can require more turns. Explicit chunk pages can request up to ten. |
| RLM-10 · Sequential delegation with shared ceilings | The budget spans siblings, and document/section pairs are indivisible below the root. |
| RLM-11 · Original excerpts and declared use | Children share validated excerpts. Parents can read more within their scope when the excerpt is insufficient. |
| RLM-12 · Reserved finish capacity | Fixed shares protect siblings and can leave capacity unused. Unused child slots return to root investigation. |
| RLM-13 · Protected finish and replay | Named tool choice is provider-dependent. Replaying valid history after truncation preserves context and ends investigation early. |
| RLM-14 · Metadata-defined scope | Scheduling can cover a bounded slice. Unresolved and pending targets stay in the denominator. |
| RLM-15 · Canonical coverage union | Union whole chunks by ID and character ranges by document/section. Parent excerpts carry narrower access. |

## Resource limits and verification

The default root budget is 20 model iterations, 12 tool attempts, 24,000 UTF-8 bytes per observation, and 30 seconds. Environment pages allow up to 25 metadata rows, 10 original chunks, or 4,000 characters of section text. Search queries allow up to 1,000 characters.

| Limit | What it controls |
| --- | --- |
| Model iterations and corpus tool calls | The demo defaults to 6 decisions and 4 local tool attempts. Change them with `--max-iterations` and `--max-tool-calls`. An oversized response spends a tool attempt. A valid finish can still be accepted after the tool budget is spent. |
| Observation bytes | The demo defaults to 6,000 UTF-8 bytes per serialized tool result. Change it with `--max-response-bytes`. The dispatcher rejects an oversized result as a whole. Retry with the suggested limit at the same offset and filters. Advance using `next_offset` from a successful page. |
| Tokens | `CLAUDE_MAX_OUTPUT_TOKENS` caps generated tokens per provider response. Input tokens include the question, tool schemas, and conversation history. Observation bytes affect how much text enters that history, and token counts are measured separately by the provider. |
| Provider quotas | Bedrock enforces service quotas independently of the demo. Raising a local limit does not change the account's throughput or access. |
| Deadline | `--deadline-seconds` defaults to 45 seconds. The loop passes the remaining time to provider attempts and discards late decisions. |

## Engineering case study · Resource-aware tool calling

### From a failed run to a testable hypothesis

The original live Sonnet 4.5 demo found ten chunks about NVIDIA FY2024 supply-chain risk, delivered zero chunks, and exhausted four corpus tool attempts without an answer. Its five API requests succeeded. The controller reported `tool_call_limit`, the application stop for a spent local tool budget. Bedrock accepted the requests, and the run recorded no provider quota rejection. That evidence identifies the observed termination as local even though Bedrock independently enforces request and token quotas.

The discrepancy between `search_discovered` and `visible_chunks` located the problem between retrieval and model delivery. The dispatcher can record evidence IDs from a search before it serializes the result. It checks the entire JSON observation in UTF-8 and returns a bounded error if it exceeds the cap. The original search default requested ten chunks, while the demo allowed 6,000 bytes. Discovery could therefore succeed while the model received an error observation with no evidence.

1. **Narrow the live request.** A diagnostic explicitly asked for `limit=1` on the existing implementation with the same budgets. It finished with two model decisions, one corpus tool call, zero step errors, and original chunk `d21883065d62ee6692558dfe` cited in an answer about NVIDIA supply-chain risks.
2. **Reproduce locally at the dispatcher.** The canonical NVIDIA FY2024 Item 1A search produced the following measurements. Serialization at a larger cap exposed the size of the full candidate observation, and repeating the dispatch at 6,000 bytes tested what would reach the model.

| Search page | Serialized success observation | Evidence returned under the 6,000-byte cap |
| --- | ---: | ---: |
| Ten hits | 20,285 bytes | 0 chunks |
| One hit | 2,219 bytes | 1 chunk |

The original live summary did not record individual tool errors. The offline reproduction establishes a mechanism consistent with the original symptoms. It cannot reconstruct the exact sequence of failures in that run.

### Solution and trade-offs

The model-facing `search_chunks` and `read_chunks` defaults now request one whole chunk. Explicit pages up to ten remain available. The tool descriptions expose the run's byte cap, and the demo question explicitly asks for a one-chunk search. If a page exceeds the cap, the dispatcher reports `response_too_large` with the measured size, cap, and a smaller suggested limit where possible. With a very small error budget, the dispatcher drops details as needed to keep even the error bounded.

The model can retry at the **same offset and with the same filters**. It advances only with `next_offset` from a successful page. Rejecting the whole result keeps original chunk text and provenance intact, avoids skipping results, and leaves rejected hits out of citation eligibility. If a single whole chunk cannot fit, a short section window can still help inspect the text.

Raising the tool-call budget would allow more attempts without addressing whether a page fits. Raising the observation cap could make larger pages fit and would allow more evidence to be replayed in subsequent requests. Starting small suits a question that needs one cited disclosure. Broad research may need more calls, or it can explicitly ask for larger pages when the budget permits. The demo retains configurable limits and adds `--trace` for per-step diagnostics. Regression tests verify the byte cap, whole-result rejection, same-offset recovery, original chunk contents, pagination, and the difference between discovered and delivered evidence.

### Measured live runs

The original demo and the updated committed demo at `a0ab13f` used Claude Sonnet 4.5 through Amazon Bedrock in `us-east-1` for the NVIDIA FY2024 supply-chain question. Both allowed **6 model iterations, 4 corpus tool attempts, 6,000 bytes per observation, a 45-second overall deadline, and 1,024 maximum output tokens per provider response**. The updated run kept these limits and used `--live --trace`.

| Metric | Original demo | Updated demo | Reduction |
| --- | ---: | ---: | ---: |
| API calls | 5 | 2 | 60.0% |
| Corpus tool calls | 4 | 1 | 75.0% |
| Input tokens | 13,843 | 6,116 | 55.8% |
| Output tokens | 685 | 491 | 28.3% |
| Total reported tokens | 14,528 | 6,607 | 54.5% |
| Execution time | 11.9599s | 9.0706s | 24.2% |
| Accepted original citations | 0 | 1 | N/A |
| Stop reason | `tool_call_limit` | `finished` | N/A |

The updated demo delivered and cited original SEC chunk `d21883065d62ee6692558dfe` with zero provider or tool errors. These are two individual live runs, with no controlled benchmark, repeated trials, or statistical significance. The updated question explicitly requested a one-chunk search, and several implementation changes shipped together. The percentages describe this pair of runs only. They establish neither the effect of any one change nor measured monetary savings. Cost would require applicable pricing and billing data. The accepted citation validates provenance and delivery. Independent semantic correctness of the answer's claims remains unverified.

### Token consumption and context growth

| Model request | Input tokens | Output tokens | Total tokens | Purpose |
| --- | ---: | ---: | ---: | --- |
| 1 | 2,720 | 156 | 2,876 | Tool selection |
| 2 | 3,396 | 335 | 3,731 | Evidence-based answer synthesis |
| Total | 6,116 | 491 | 6,607 | Complete run |

The first request already carries system instructions, tool names and schemas, the user question, and provider-specific framing before the model retrieves any SEC text. Subsequent requests resend that context and replay assistant messages, tool-use IDs, and observations. A larger evidence page adds text and metadata to later inputs. Even error-only observations grow the history. The original failed demo's input token counts progressed **2,410 → 2,599 → 2,772 → 2,946 → 3,116**. These totals illustrate growth without isolating the contribution of any one prompt component.

Character limits bound source-text windows. UTF-8 byte limits measure the full serialized observation, including JSON and metadata. Input tokens count the material the model receives after provider tokenization. Output tokens count what it generates. Multibyte characters, JSON framing, and tokenization prevent a fixed conversion between these units. Local tool and byte caps govern application work, the output-token limit applies to one provider response, and Bedrock quotas govern service throughput separately.

Both compared runs reported zero cache-read and cache-write tokens. Provider-managed **KV caching** stores attention key and value state during inference. **API prompt caching** exposes reuse of eligible prompt prefixes across requests. The adapters replay normal conversation history and omit explicit cache controls, so prompt caching remains outside the measured baseline. Repeated system instructions and tool schemas make stable prefixes candidates for a later experiment, subject to model support, cache lifetime, and billing rules. Neither run demonstrates a caching-related improvement. [Bedrock prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)

### Reproduce the size check offline

With the local canonical manifest and Parquet files available, run from the repository root. This calls the local dispatcher only. The live command in the course setup section is the opt-in path for the committed demo.

```bash
PYTHONPATH=Code/src python - <<'PY'
import json
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.tools import CorpusToolDispatcher

corpus = CorpusEnvironment()
filing = corpus.list_filings(ticker="NVDA", fiscal_year=2024).items[0]
for limit in (10, 1):
    arguments = {
        "query": "supply chain", "limit": limit,
        "filters": {"document_ids": [filing.document_id], "section_ids": ["item_1a"]},
    }
    full = CorpusToolDispatcher(corpus, max_response_bytes=100_000).dispatch("search_chunks", arguments)
    capped = CorpusToolDispatcher(corpus, max_response_bytes=6_000).dispatch("search_chunks", arguments)
    print(limit, len(full.content.encode("utf-8")), json.loads(capped.content)["ok"],
          [ref.chunk_id for ref in capped.evidence])
PY
```

## Recursive milestone verification

The canonical offline demonstration collected one NVIDIA Item 1A excerpt for each fiscal year. It ran this actual execution tree:

```text
root (depth 0) · collect FY2023 and FY2024 evidence
├── child-1 (depth 1) · FY2023 Item 1A, read original chunk 2453efcba37d6147f94551b6
└── child-2 (depth 1) · FY2024 Item 1A and Item 1 assigned, return grandchild excerpt
    └── child-3 (depth 2) · FY2024 Item 1A, read original chunk 0bbb6e2ac7257dd27e1b84f5
```

The run used **4 model sessions, 9 decisions, 3 delegations, 4 partition assignments, 2 corpus tool attempts, and 0 provider attempts**. Maximum observed depth was **2**. The root finished with the two original citations shown above and **partial coverage**. FY2024 Item 1 remained unread. Scripted decisions verified control flow, fresh conversations, scope checks, evidence transfer, and accounting without relying on model behavior. Effective recursive research with a real LLM still needs live validation against appropriate baselines.

| Verification | Recorded result |
| --- | --- |
| Focused RLM suite | 95 passed, 1 warning in 2.06s |
| BM25, corpus retrieval, RAG provenance, and canonical corpus integrity | 40 passed, 1 warning in 17.65s |
| Scripted canonical recursive demo | Exit 0, stop reason `finished` |
| Separate full-corpus SentenceTransformer integration run | 1 passed, 2 warnings in 119.77s, outside the focused and related runs |

The focused and related runs each reported the existing AnyIO pytest deprecation. No live or paid model calls were made during recursive implementation. From the repository root, reproduce the test suites with:

```bash
export PYTHONPATH=Code/src AWS_EC2_METADATA_DISABLED=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest -q Code/tests/test_rlm_{recursive,controller,environment,claude,bedrock}.py
python -m pytest -q Code/tests/test_retrieval_bm25.py Code/tests/test_retrieval_corpus.py \
  Code/tests/test_rag_corpus.py Code/tests/test_corpus_regression.py::test_exported_canonical_corpus_integrity
```

### Corpus-wide planning verification

```bash
PYTHONPATH=Code/src python Code/examples/demo_rlm_plan.py --max-depth 1 --max-delegations 8 --max-partitions 16
```

The [issue #16 question](https://github.com/ibrahimhamza01/EvidenceRoute/issues/16) asks which companies introduced an AI-related infrastructure risk in FY2024 that was absent in FY2023. The scripted interpretation keeps those years and infers all corpus companies, Item 1A, and comprehensive review. The local snapshot contains 25 companies and 75 filings. The planner resolved **50 of 50** required Item 1A targets in **25 groups**, with **zero missing requirements**. The example limits of 8 delegations and 16 pair assignments cannot admit the full plan in one run. The demo made zero provider attempts and assessed no evidence.

`python -m pytest -q Code/tests/test_rlm_*.py` passed with **159 passed, 1 warning in 3.39s**. The related retrieval, provenance, and canonical integrity regressions recorded **40 passed, 1 warning in 18.59s** using the commands above. Both warnings were the existing AnyIO pytest deprecation. The existing scripted recursive demo also exited successfully with two original citations and zero provider attempts.

### Bounded multi-company execution and coverage

```bash
PYTHONPATH=Code/src python Code/examples/demo_rlm_planned_research.py
```

The scripted run admitted AAPL and AMD and kept 23 groups pending. The two children ran the same targeted search once per fiscal year and delivered four findings that the root cited using original SEC chunk IDs. The report counted **4 partial and 46 untouched targets** out of **50**, **4 of 3,038** unique whole chunks, and **0 of 4,005,903** characters delivered through section windows. The runner recorded 3 sessions, 9 decisions, 2 delegations, 4 corpus attempts, 4 pair assignments, and zero provider attempts. The root stopped `finished` while the research answer remained `incomplete`. The scripts demonstrate candidate retrieval and synthesis plumbing. The excerpts leave the novelty question unanswered.

The ledger distinguishes search discovery, acknowledged whole chunks, acknowledged section windows, and delivered excerpts. It records findings generated, acknowledged, and used, alongside assignment and model-reported analysis outcomes. Access is `complete` when either the canonical chunk inventory or the full section character range was delivered, with separate completeness flags for both measures. An acknowledged read by a child that later fails still contributes to access telemetry. Root citations still depend on evidence delivered to the root and the runner's existing validation. Complete access establishes neither semantic correctness nor that a risk was absent.

Verification recorded **167 focused RLM tests passed, 1 warning in 7.26s** and **40 related regressions passed, 1 warning in 17.86s**. Both warnings were the existing AnyIO pytest deprecation. The original scripted NVIDIA demo still exited successfully with two citations and zero provider attempts.

#### Opt-in live two-company experiment

Scope interpretation remains scripted. The plan keeps all 25 companies and 50 Item 1A targets, while the runner initially admits AAPL and AMD for both years. Bedrock Claude chooses the retrieval queries and investigation steps. Each child receives guidance to compare the same risk across years, return short verbatim quotes with original IDs, disclose uncertainty and partial coverage, and avoid inferring absence from a search miss. The root is asked to limit its synthesis to the assigned evidence.

```bash
BEDROCK_REGION=us-east-1 \
  BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  CLAUDE_MAX_OUTPUT_TOKENS=1024 PYTHONPATH=Code/src \
  python Code/examples/demo_rlm_planned_research.py --live --trace
```

Defaults allow depth 1, two delegations, four pair assignments, 12 shared corpus attempts, 18 provider attempts, a 100,000 reported-token threshold, and 180 seconds for both shared and local deadlines. The root and each child initially get six provider attempts. Each allocation now reserves two for protected finalization, leaving four for investigation. Local defaults are six decisions and six tool attempts per invocation. `--help` lists configurable limits. The experiment keeps the 6,000-byte observation cap and zero provider retries. Investigation defaults to 1,024 output tokens via `CLAUDE_MAX_OUTPUT_TOKENS`. Protected finalization defaults to 2,048 output tokens and requires named finish-tool choice. The existing reservation preflight rejects infeasible allocations before any provider client starts.

The JSON reports the full and admitted scopes, invocation outcomes, pending groups and per-target coverage, provider tokens and latency, and original citations. Findings and their acknowledged delivery are recorded separately from declared use. `--trace` adds per-step provider and tool details. Exit `0` requires both whole-company delegations, a finished child and acknowledged findings for both years per company, and a finished root synthesis citing original evidence from those findings and declaring the finding IDs used. An unsupported or missing year produces structural status `incomplete` and exit `1` even if the root finishes. Structural success still leaves the corpus-wide `answer_status` incomplete and `semantic_claims_verified` false.

##### First live run and quotation correction

Run `2a38409af02e4f4fbdbd54b21c3afba0` delegated both whole company groups and exercised research-wide access accounting. Of 25 companies, 23 remained pending. Four of 50 targets had partial access and 46 were untouched. The ledger recorded 13 unique acknowledged whole chunks out of 3,038 required, with 292 chunks in the admitted inventory. The run used 3 sessions, 16 decisions, 16 provider attempts and responses (13 investigation and 3 protected finish), 12 corpus attempts, and 4 pair assignments. It reported 77,835 input and 5,018 output tokens (82,853 total), 62.6930 seconds elapsed, and 58.409 seconds summed provider latency. No provider errors, retries, or truncation occurred.

Each child used five investigation attempts and its single protected finish. Both finishes failed exact quotation validation. The root then made two direct Apple searches, exhausted the shared corpus allowance, and also failed quote validation. All three protected requests selected `finish_research`. The run ended structurally `incomplete`, with zero validated findings or final citations and root stop `budget_exhausted` / `finalization_attempt_limit`.

| Submitted evidence | Canonical comparison |
| --- | --- |
| Apple `1b31d472397f437ab1975947` and `e4bf3d03444b5f8b4e1614e3` | The inserted `...` skipped the list from `antitrust` through `intellectual property ownership and infringement` before `digital platforms`. Neither quote was contiguous. |
| AMD `483314af3b949a29920abbbf` | The only difference was the straight apostrophe in `customers' ability`. The source uses `customers’ ability`. |
| AMD `e8e013b9b919159fb6168bca` and `fb3d8728e8dba18029424d7a` | Both submitted excerpts were exact contiguous matches. The rejected first quotation prevented accepting the finish. |
| Root `d79cf93171c2e9d56c5a2613`, `9872ded1fcb8eedd58fb2d78`, and `e4bf3d03444b5f8b4e1614e3` | The first quote matched. The other two used `Company's` where the source uses `Company’s`. |

The AMD FY2024 power-and-water disclosure is a candidate infrastructure risk. FY2023 search misses leave its novelty unestablished. The revised instructions distinguish an observed FY2024 risk, comparable FY2023 evidence, possible change needing verification, and insufficient evidence to assess novelty. General AI references alone do not establish an infrastructure risk.

Validation still requires an exact contiguous quote, preserving punctuation and whitespace. Feedback identifies the first failing finding and evidence item and names its chunk only when the invocation can cite it. It supplies no replacement text. The second protected attempt can correct or omit the finding, or return an explicitly incomplete result. It receives the validation error under the original Bedrock tool-use ID and cannot resume investigation. The shared attempt ceiling and output caps are unchanged.

Stubbed Bedrock tests verified correction and repeated rejection within the two finish slots, with original citations and partial coverage preserved. The focused RLM suite recorded **173 passed, 1 warning in 9.22s**. Related retrieval, provenance, and canonical integrity checks recorded **40 passed, 1 warning in 20.73s**. Both suites reported the existing AnyIO pytest deprecation. The scripted NVIDIA and planned demos still ran with zero provider attempts.

##### Second live run and comparative evidence

Run `3925e7f8bc014cb4a3a6cb22a89e6312` recorded 17 provider attempts and responses, 12 corpus attempts, 3 model sessions, and 2 delegations. It reported 84,910 input and 6,590 output tokens (91,500 total) over 70.2815 seconds. The ledger recorded 10 discovered chunks and 9 unique acknowledged whole chunks. One oversized search result was rejected and its undelivered evidence received no access credit. Coverage remained at four partial targets and 46 untouched targets.

The root acknowledged three validated Apple child findings. AMD failed quotation validation on both protected finishes. The root failed its finish on the same AMD quotation issue and reached its six-decision limit with provider capacity unused. The run ended structurally `incomplete` with zero final citations. Compared with the first run, the second delivered three findings instead of zero, acknowledged fewer whole chunks (9 vs. 13), and used more reported tokens (91,500 vs. 82,853) and time (70.2815 vs. 62.6930 seconds). Both runs used 12 corpus attempts and neither produced final citations. The two individual runs do not isolate the effect of the quotation guidance or reservation change.

- **AMD:** The canonical FY2023 Item 1A chunk `043757f52188a8c6fb2a2857` already states that data centers depend on clean water and reliable energy and that shortages could impair customer capacity expansion and use of AMD products and services. FY2024 chunk `483314af3b949a29920abbbf` repeats that risk and explicitly connects it to potential harm to AMD revenue. The FY2023 passage is candidate counterevidence to the claim that the underlying constraint was new in FY2024. These excerpts leave open whether the revenue language changed elsewhere in the filings.
- **Apple:** The child reported insufficient evidence to determine infrastructure-risk novelty. The root inferred novelty from safety and regulatory disclosures. The validated FY2023 quotation itself does not mention AI, and the retrieved evidence did not establish the proposed infrastructure-risk change.

The two-company demo optimization cycle ends here. The next milestone is systematic comparison of the same risk across years, including an explicit search for prior-year counterevidence before making a novelty claim.

At this checkpoint, durable finding storage, independent semantic verification, automated reinvestigation scheduling, live intent interpretation, and comprehensive corpus execution were still pending. This demo executes one bounded slice.

### Systematic comparative research

The original comparative demo scanned the resolved inventory for case-insensitive literal matches. Children receive up to three candidate IDs per authorized section with span hints and a cursor. `list_candidates` returns up to ten IDs for a scoped section. Comparative `search_chunks` returns ranked candidate IDs and span hints. Each call uses one local tool slot and one shared corpus attempt. Candidate IDs grant no evidence or access credit. `get_spans` supplies up to four exact passages from one scoped chunk. `get_chunk` supplies the full text when context is needed.

Comparative chunk observations include compact span IDs alongside the original text. An ID is a pair of character offsets within one canonical chunk. The model selects the pair and Python copies the exact source characters. Selection is limited to acknowledged span IDs or the identical span from a declared, acknowledged child finding. This preserves Unicode and prevents a parent from expanding a delegated excerpt. The existing observation byte cap applies and the legacy exact-quote interface remains available.

Children submit structured relevance and entailment assessments over selected spans. The root can reassess the delivered supporting and counterevidence spans in its existing budgeted session. An assessed prior-year match can support a `comparable_both_years` result. Contradictory assessments can yield `conflicting`. Current-year support with inconclusive prior-year evidence can at most yield `possible_new_needs_review`, and a prior-year search miss alone leaves the result `insufficient`. This path issues no verified novelty or absence verdict. It reports model-assisted judgments without independent semantic verification. Protected finish attempts also reserve local decisions, so returned child provider slots cannot consume those decisions.

```bash
PYTHONPATH=Code/src python Code/examples/demo_rlm_comparative.py
```

With the original eight literal terms, the offline demo scanned **3,038 chunks in 50 Item 1A sections across 25 companies** using 328 local pages and 4,603,503 chunk characters, including overlap. It identified **87 candidate chunks and 118 matching spans**. FY2023/FY2024 candidate chunk counts were **1/1 for AAPL** and **4/6 for AMD**. The AMD script read the FY2024 water-and-power passage and then the canonical FY2023 counterpart. Two chunks were delivered and assessed, both original citations reached the root, and the scripted root classified the constraint as comparable across both years. The FY2024 excerpt explicitly adds a potential revenue impact. The selected passages concern data centers generally and leave the AI-specific link unestablished. This demonstration used 3 sessions, 7 decisions, 2 corpus attempts, and zero provider attempts. Corpus-wide research remained incomplete.

The baseline live runs below used scripted intent, a two-group cap, 18 provider attempts, 12 corpus attempts, protected finalization, and zero provider retries.

The first live run, `fdd6cd9d72d14c708bfdc36e2b3b05a1`, scanned all **3,038 chunks across 50 sections and 25 companies** in **0.0568015 seconds**, finding **87 candidate chunks and 118 matching spans**. The root used one model session, one decision, and one Bedrock request with one response. The request took **3.1933 seconds** and reported **5,190 input and 374 output tokens**. The run stopped with `model_error` before any delegation or corpus operation, with zero delivered chunks, findings, or citations and no budget exhaustion. The saved report contains neither the decoder error nor the provider tool requests, so the exact live failure remains unknown.

The adapter did reject Converse tool-use IDs containing periods or colons, even though AWS allows both in requests and results. A stubbed comparative run reproduces that rejection with a native delegation batch. The adapter and batch validator now preserve those IDs through execution and result delivery. The saved report does not establish that this defect caused the live failure. Comparative reports now include per-step error codes and allowlisted provider stop reasons by default. `--trace` reuses the recursive trace for sanitized tool requests, validation errors, and invocation stops, without serializing private reasoning or raw provider messages. [AWS tool-use IDs](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolUseBlock.html) [AWS tool-result IDs](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html)

The second live run, `6aa5276f4e65486484f85090bfc01ac7`, completed the same 25-company, 50-section, 3,038-chunk scan. Bedrock returned `tool_use` and the decoder counted five operations in the first reply. The first four recorded requests were `get_chunk`. The decoder raised `tool_batch_limit`, confirming the cause of this run's `model_error`. No tools ran, no children started, and no evidence was delivered. The request reported 5,190 input tokens, 322 output tokens, and 2.902 seconds of provider wall time.

The fix rejects all five requests and sends a matching error for every ID before asking for a corrected turn. The investigation prompt also states the four-operation limit. If necessary, feedback drops the prose and retains the error code and numeric limit for each ID. If even that complete reply cannot fit the configured byte cap, the invocation stops. The experimental budgets and evidence rules are unchanged. Recovery has been exercised with native Bedrock stubs and remains unverified live.

The third live run, `5ae02fdd1d154776801963e39e521f57`, finished in **19.6014 seconds** with **3 provider attempts, 19,512 reported tokens, 6 corpus attempts, and 4 original citations**. It verified live source-backed span selection and structured comparative finalization. The root investigated directly with zero delegations, so recursive comparative execution remains unverified live. Both investigation batches had four requests. Oversized tool-request recovery was not exercised. Two executed corpus observations exceeded the batch observation limit, and two later requests were skipped at the local tool limit.

Both company comparisons returned `insufficient` rather than asserting novelty. Apple cited both years, while AMD cited only FY2023. One AMD assessment labeled AI accelerator demand `infrastructure` and `supports` even though its rationale described a market opportunity rather than an infrastructure risk. The run finished with research status `incomplete`. Scanning all 3,038 chunks for lexical candidates did not establish comprehensive semantic coverage.

Offline verification recorded **198 focused tests passed, 1 warning in 10.91s** and **40 related retrieval, provenance, and canonical integrity tests passed, 1 warning in 17.71s**. The warning was the existing AnyIO pytest deprecation. Bedrock stubs covered complete rejection and correction at five and sixteen calls, repeated rejections through protected finalization, duplicate and malformed requests, and the protocol and byte ceilings. The scripted comparative demo still scanned 3,038 chunks, finished with two citations and `incomplete` research status, and made zero provider attempts.

#### Delegation-first execution and budget profiles

The baseline at `3605854` produced four citations with zero delegations and no AMD FY2024 citation. The root now starts comprehensive plans with `delegate_research` as its only advertised tool. Python rejects direct reads, premature finishes, split groups, and duplicate initial assignments. The assignment restriction applies to the entire provider reply, including a batch that attempts the last unassigned group. An initial attempt counts when the host creates the child invocation. A failed child stays visible and does not block subsequent assignments. After all groups have an attempt, the root can synthesize or use remaining capacity for scoped reads and whole-group reinvestigation. When assignment capacity runs out, the root can finalize an incomplete answer if finish capacity remains. Set `delegation_first=False` to use the previous model-directed behavior.

| Setting | `diagnostic` (default) | `expanded` |
| --- | ---: | ---: |
| Company groups / depth / delegations / pair assignments | 2 / 1 / 2 / 4 | 2 / 1 / 2 / 4 |
| Shared provider attempts / attempts per child | 18 / 6 | 36 / 12 |
| Shared corpus attempts / reported-token threshold | 12 / 100,000 | 30 / 200,000 |
| Shared and local deadline | 180 s | 300 s |
| Local decisions / tool calls / response byte cap | 6 / 6 / 6,000 | 12 / 12 / 16,000 |
| Protected finish attempts / output tokens / tool choice | 2 / 2,048 / required | 2 / 2,048 / required |
| Root / child finalization token headroom | 12,000 / 12,000 | 24,000 / 24,000 |
| Protected request timeout / timeout retries | `CLAUDE_TIMEOUT_SECONDS` / 0 | 60 s / 1 |
| Post-assignment root tool calls | 2 | 2 |

The diagnostic profile retains the baseline shared ceilings. The expanded profile starts the root and each child with ten investigation attempts and two reserved finish attempts. Its 30 shared corpus calls can accommodate both children's maximum of 12 local tools each and the root's two follow-up calls. The existing allocation can return unused child attempts to the root within the configured ceiling. Execution batches remain capped at four and protocol recovery at sixteen.

Use `--profile-file` to load JSON with `max_groups`, `shared_limits`, `local_limits`, and `finalization`. The optional `max_root_reinvestigation_calls` defaults to two. `finalization` accepts `root_token_headroom`, `child_token_headroom`, `finalization_timeout_seconds`, and `finalization_timeout_retries`. Validation checks provider and token reservations, root decisions and tool slots, depth, pair capacity, and the minimum corpus allowance of one retrieval per year per group before the runner creates a provider client. The three-group offline run used the expanded profile with `max_groups=3`, `max_delegations=3`, `max_partitions=6`, and `max_provider_attempts=42`.

`execution_status` records whether the root accepted a finish. `recursive_comparative_status` also requires an acknowledged child result for each admitted group and declared use of delegated evidence when any child delivers findings. An investigator may abstain. The ledger retains failed initial attempts and any successful reinvestigation. `evidence_years` and `both_years_contributed` describe the assessed spans in each comparison. `root_cited_years_by_company` shows which fiscal years reached the answer's citations. The report also shows delivery failures, provider errors, unassigned groups, and pending scope.

The comparison resolver treats an infrastructure label without a stated risk connection as uncertain. An accelerator-demand passage needs a source-backed link to the risk mechanism. A final response accepts up to three structured comparisons and eight findings. Broader reporting needs consolidation across bounded runs.

Run the expanded profile manually from the repository root with authorized Bedrock credentials.

```bash
BEDROCK_REGION=us-east-1 \
  BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  CLAUDE_MAX_OUTPUT_TOKENS=1024 PYTHONPATH=Code/src \
  python Code/examples/demo_rlm_comparative.py --profile expanded --live --trace
```

**Offline verification.** The focused RLM suite passed **218 tests** and the related regressions passed **53 tests**. Each reported the existing AnyIO pytest deprecation warning. In the expanded Bedrock stub, the root recovered from a truncated assignment reply, delegated both companies, and synthesized four original citations covering both years for each company. The stub recorded eight provider attempts and four corpus attempts. The scripted recursive, planning, planned research, comparative, and Standard RAG demos exited successfully with zero provider calls. Both comparative profiles scanned 3,038 chunks, attempted two groups, and returned two citations. The custom profile attempted three groups. Corpus-wide comparative research remained incomplete.

```bash
export PYTHONPATH=Code/src AWS_EC2_METADATA_DISABLED=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest -q Code/tests/test_rlm_*.py
python -m pytest -q Code/tests/test_retrieval_bm25.py Code/tests/test_retrieval_corpus.py \
  Code/tests/test_rag_corpus.py Code/tests/test_rag_citations.py Code/tests/test_rag_prompts.py \
  Code/tests/test_standard_rag.py Code/tests/test_corpus_regression.py::test_exported_canonical_corpus_integrity
python Code/examples/demo_rlm_recursive.py
python Code/examples/demo_rlm_plan.py
python Code/examples/demo_rlm_planned_research.py
python Code/examples/demo_rlm_comparative.py --profile diagnostic
python Code/examples/demo_rlm_comparative.py --profile expanded
python Code/examples/demo_standard_rag.py
```

#### Grounded partial results under shared limits

Expanded live run `2f2882839aca4a7b9c61e7a26463237c`, saved as `rlm-comparative-expanded-v3.json`, delegated both companies in the initial batch. It used **21 provider attempts, 20 responses, 29 corpus attempts, and 202,093 reported tokens in 147.95 seconds**. Apple acknowledged 26 whole-chunk deliveries covering 16 unique chunks. Its protected finish timed out after 20.04 seconds and returned no findings or usage. AMD acknowledged 11 whole-chunk deliveries covering nine unique chunks and returned two source-backed findings that the root acknowledged. Its structure classified the underlying water and power risk as `comparable_both_years`. Its prose opened with a claim of novelty, then stated that the underlying risk appeared in both years. The root requested six follow-up searches for Apple. Five reached the corpus, and the root received 14 whole-chunk deliveries covering 11 unique chunks. A response crossed the 200,000-token threshold by 2,093 and the root stopped without finalizing or citing evidence.

- **Token admission.** Estimate each complete request from its serialized UTF-8 bytes divided by three, rounded up, plus the requested output cap. Investigation must leave the configured reserves for the root and active or initially unassigned children. A blocked investigation can replay its assembled history in finish-only mode. A child finish can use its own reserve while protecting the root and later children. A root finish may use the remaining shared headroom. Estimates count message history and tool schemas. Provider tokenization, request framing, and cache accounting can differ, and missing usage or a single response can still push actual consumption past the threshold. Measured usage stays separate and the configured ceiling is fixed.
- **Timeout and replay.** `CLAUDE_TIMEOUT_SECONDS` defaults to 20 seconds. The expanded override applies 60 seconds to protected finish requests. Bedrock sets both socket timeouts to the smaller of the applicable setting and remaining invocation time. One protected timeout retry can reuse the same request within the attempt, token, and deadline limits. The SDK performs one request per application attempt. The trace records each attempt, including missing usage, and acknowledges tool results only once. A timed-out server request may still have done work. The deadline remains cooperative.
- **Context and follow-up.** The adapters keep native assistant messages and tool-result IDs in order. Comparative searches send IDs and span hints. Acknowledged whole chunks returned again in the same invocation are referenced by ID. Two post-assignment root tool slots bound follow-up even across a batch. The trace records estimated request bytes, effective timeout, and generated observation bytes. The historical report lacks request-size telemetry. Measured locally, the two cited AMD whole-chunk observations were 1,479 and 1,660 bytes. Their exact-span observations were 566 and 636 bytes, and each duplicate whole-chunk reference was 85 bytes.
- **Conclusions and fallback.** The comparative answer renders validated outcomes instead of publishing the model's draft prose. General infrastructure support and the reported AI-specific connection are separate fields. The same-risk finding for the cited AMD spans supports a comparison across fiscal years and leaves the AI-specific link unestablished. If the root cannot finish, `partial_result` exposes only acknowledged child findings and comparisons with original spans, failed groups, and access counts. Root citations and synthesis status remain separate. Formatting and exact-span checks establish provenance and consistency. Independent semantic verification and corpus-wide conclusions remain open.

Offline verification at this stage passed **229 focused RLM tests** and **53 related regressions**, each with the existing AnyIO deprecation warning. Bedrock stubs exercised timeout replay, projected token starvation, measured overshoot, and acknowledged-child fallback.

#### Expanded V4 live checkpoint

V4 (`7489d1f7a57949f5b99880eba68bd149`, `rlm-comparative-expanded-v4.json`) lexically scanned **3,038 chunks across 50 sections and 25 companies**, finding **87 candidate chunks and 118 matching spans**. Both children and the root finished. AMD's first protected finish combined evidence from multiple documents in individual findings, violating the one-document-and-section rule. Its second protected attempt succeeded. The root acknowledged and used four AMD findings, returned four original chunk citations, and classified the assessed data-center power and water constraint as `comparable_both_years`. The exact spans leave the AI-specific connection unestablished. Apple finished with an incomplete assessment and no validated findings. The model did not investigate the other 23 companies.

| Measure | Expanded V3 | Expanded V4 | Observed change |
| --- | ---: | ---: | --- |
| Reported input tokens | 196,764 | 133,818 | 62,946 fewer |
| Reported output tokens | 5,329 | 7,703 | 2,374 more |
| Total reported tokens | 202,093 | 141,521 | 60,572 fewer (30.0%) |
| Recursive wall time | 147.9479 s | 86.4068 s | 61.5411 s less (41.6%) |
| Provider attempts / responses | 21 / 20 | 19 / 19 | 2 fewer attempts |
| Corpus operations | 29 | 26 | 3 fewer |
| Timeouts / attempts missing token usage | 1 / 1 | 0 / 0 | Both zero in V4 |

Both runs used the shared and local ceilings in the expanded column above, including 36 provider attempts, 30 corpus operations, a 200,000 reported-token threshold, and 300-second deadlines. Both reserved two finish attempts with a 2,048-token output cap and required finish-tool choice. V3 used whole-chunk search delivery and shared/local limits for root follow-up, with no dedicated token headroom or finish-timeout override. V4 used compact evidence delivery, two root follow-up calls, 24,000 tokens of headroom each for root and child finalization, and the 60-second protected timeout with one allowed retry. V4 used no retries.

Compact results reduce the filing text repeated in provider history. V4 delivered exact spans from 13 chunks and whole text from two. The root follow-up cap limits repeated investigation after delegation. Token headroom and the longer protected timeout leave room for synthesis and validation correction, as exercised by AMD's second finish. The lower reported input and wall time are consistent with these changes. One run per configuration cannot isolate causes or establish statistical significance. V3's missing usage prevents a complete token accounting, and the reported token difference cannot establish a monetary cost reduction.

The live command above applies to V4. To reproduce the V3 configuration, run the same `--profile expanded --live --trace` command with the earlier implementation and the V3 settings recorded here. Use the same local corpus for both configurations. Results and usage can vary across runs.

`execution_status` and `recursive_comparative_status` both succeeded in V4. `research_status` remained incomplete and independent semantic verification remained false. The public answer now identifies evidence-supported comparisons, states when the validated outcomes leave novelty unestablished, and lists admitted companies with unresolved assessments and companies outside the admitted scope. Possible novelty still requires review. The same scope disclosure accompanies acknowledged-child fallback, and the validated comparisons and original citations remain the source of the answer's claims.

Checkpoint verification passed **230 focused RLM tests** and **53 related regressions**, each with the existing AnyIO deprecation warning. Both offline comparative profiles finished with two original citations, zero provider attempts, and incomplete research status. The regression cases cover an unresolved admitted company, non-admitted scope, tentative novelty, and acknowledged-child fallback.

## Bounded live recursive experiment · Four observed runs

The first live recursive experiment asks Claude to compare one documented supply-chain risk disclosure from NVIDIA's FY2023 and FY2024 filings. It should identify similarities and differences supported by the retrieved evidence and disclose limitations. The demo resolves the two canonical filings and their Item 1A sections locally. These exact document/section pairs are the entire authorized scope. The root is prompted to delegate one separate investigation per year. Each child chooses its own queries and tool actions, starts with small evidence pages, and returns a compact finding with an exact quotation and original chunk ID. The parent synthesizes the delivered findings.

The default demo still runs the scripted offline tree. The `--live` flag opts in to Bedrock. The factory creates a fresh `BedrockRootClient` for each invocation, and the existing `RecursiveResearch` runner binds the shared budget. Depth one permits the two expected children and leaves grandchildren outside the experiment. The controller omits the delegation tool at the depth ceiling and keeps the runtime depth check. Starting with this shallow tree makes failures in delegation, evidence delivery, and synthesis easier to distinguish. A scripted live tool sequence would bypass model-directed decisions. Giving the root a wider corpus would make it harder to attribute evidence and resource use to the two target filings.

### Run comparison

All four runs used Amazon Bedrock with Claude Sonnet 4.5 (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`) in `us-east-1`.

| Metric | Initial | Batching | Reservations | Latest |
| --- | ---: | ---: | ---: | ---: |
| Provider attempts | 1 | 12 | 10 | 9 |
| Corpus attempts | 0 | 8 | 6 | 5 |
| Input tokens | 3,835 | 54,703 | 43,451 | 36,417 |
| Output tokens | 416 | 2,392 | 3,190 | 2,866 |
| Total reported tokens | 4,251 | 57,095 | 46,641 | 39,283 |
| Wall time (seconds) | 4.0379 | 35.6985 | 43.8476 | 37.2452 |
| Delivered findings | 0 | 1 | 1 | 2 |
| Final original citations | 0 | 0 | 0 | 2 |
| Outcome | `multiple_tool_uses` | `budget_exhausted` | `truncated_response` | `success` |

The successful run used about 31.2% fewer reported tokens than the budget-exhausted run and 15.8% fewer than the preceding reservations run. The reservations run itself used 18.3% fewer reported tokens and took 22.8% longer than the batching run. These are individual runs with changes to prompts and finalization policies. The comparisons do not isolate the effect of any one change, support statistical generalization, or establish measured billing savings.

### First live result

The single-tool adapter rejected the response with `multiple_tool_uses`. The run ended as `incomplete` with stop reason `model_error`. It recorded one session, one decision, one provider response, 3.952s provider-reported latency, no delegation, and no exhausted budget. The exact tool names and arguments were not recorded, so the trace does not establish which requests were grouped.

### Batch-enabled live result and finalization starvation

Run `f3e66af80dab48179df5627829f56e7f` recorded 3 sessions, 13 decisions, 12 provider responses, and a maximum depth of 1. The root requested two delegations in one reply. Both children ran sequentially and the root acknowledged both outcomes. No provider protocol errors occurred.

- **FY2023:** Four searches delivered four chunks. The fifth API call finished with a validated finding on geopolitical tensions and supply continuity. The parent acknowledged original chunk `bd263969e91358073d0ea0a1`. The child used 24,198 tokens. Per-request input was **3,328 → 4,000 → 4,637 → 5,270 → 5,838**.
- **FY2024:** Four retrieval operations delivered four chunks, including `d21883065d62ee6692558dfe`, the chunk used by the earlier targeted experiment. The fifth API call requested another retrieval and hit `tool_call_limit`. The child used 23,689 tokens and submitted no validated finding, so the root acquired no citation access from it. Per-request input was **3,328 → 4,000 → 4,591 → 5,231 → 5,855**.
- **Root:** Its additional retrieval hit `corpus_attempt_limit`. Its next decision was blocked by `provider_attempt_limit`, ending with `budget_exhausted`. Root usage was 8,625 input and 583 output tokens.

This run demonstrated live recursive delegation and child-to-parent evidence transfer. Investigations consumed the capacity needed to finalize. The child token sequences also show that fresh conversations isolate parent context while each child's own history still accumulates.

### Protected finalization

The live demo keeps the same shared limits. With the default 12 provider attempts and two child slots of four each, the root initially has four attempts. Each pool holds **three investigation attempts and one finish-only attempt**. Prospective child pools are held before admission, so neither the root nor an early child can spend a later child's share. Unused attempts from an admitted child return to root investigation on exit, including failure. Slots for children that never launch stay set aside until the run ends. A voluntary early finish during investigation uses an investigation attempt.

Run `71927fc9d79b444589f7aecdd244fc57` recorded 3 sessions and 10 decisions, completed both depth-one delegations, and made nine investigation attempts and one protected finish attempt. No shared attempt, reported-token, or deadline limit was exhausted.

- **FY2023:** Three provider attempts and two searches delivered four unique chunks. The child finished during investigation with a validated finding citing `bd263969e91358073d0ea0a1`. The parent acknowledged the finding. The child used 15,343 reported tokens and returned one unused provider slot.
- **FY2024:** Three investigation requests delivered three chunks. On the protected attempt, Claude requested `search_chunks` despite the finish-only tool list. The runtime rejected it as `finalization_only` and the child stopped with `budget_exhausted` / `finalization_attempt_limit`. It used 15,930 reported tokens and granted no findings or citations to the parent.
- **Root:** The root acknowledged both child outcomes and directly retrieved FY2024 chunk `948f04ad7acdaaa298cc3d01`. Its third provider reply reached the 1,024 output-token cap. The adapter rejected it as `truncated_response` without recording a complete action. The root stopped with `model_error` despite remaining provider capacity.

#### Finish request controls and truncation recovery

Investigation turns show the remaining local and shared budgets and leave tool choice to the model. The controller enters finish-only mode when investigation attempts or local tool slots run out, when it reaches the last local decision, or when neither corpus work nor delegation is available. It replaces the investigation prompt and offers only `finish_research`. The default `auto` choice can still yield a prohibited request, as observed in the protected run. `--finalization-tool-choice required` requests the named finish tool in Bedrock Converse or direct Anthropic Messages. Runtime enforcement rejects further research and validates the finish, citations, and quotations. The prompt asks for an incomplete result or abstention when evidence is insufficient.

The course Sonnet 4.5 profile accepted named choice in the successful run below. Converse support varies by model and configuration. The AWS `SpecificToolChoice` reference lists Claude 3, while the AWS Claude guide describes the Converse setting more broadly. The demo defaults to `auto`. A validation rejection with named choice is reported as `aws_required_finish_validation` (or `http_required_finish_validation` for a direct Messages HTTP 400) without retry or automatic fallback. The adapters do not enable extended thinking. [AWS SpecificToolChoice](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_SpecificToolChoice.html) [AWS Claude tool use](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-tool-use.html#forced-tool-use) [Anthropic tool choice](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use)

Reservations protect request slots. The finish output cap is separately configurable and defaults to 2,048 tokens, while investigation defaults to 1,024. The experiment asks each child for a one-sentence answer plus one supported finding with the quote given once. The root is asked for a concise comparison with original citations, the IDs of findings it used, and coverage limits. Queries remain model-driven. These prompt and cap changes affect comparability with earlier live runs.

A `max_tokens` investigation reply supplies usage but no executable action. The implementation at this milestone could replay a valid history in a reserved finish slot after investigation truncation. A truncated protected finish still ended the invocation. The later corpus checkpoint extends that recovery to remaining protected slots and records the live outcome. See [protected replay](#protected-replay-and-its-limit).

### First successful end-to-end live run

Run `1b803beef0f948d9943f23e582c57cd7` ended with status `success` and stop reason `finished`. All six structural criteria passed. The root requested both fiscal-year delegations in one batch. The run recorded 3 sessions, 9 decisions, 9 provider responses, zero retries, 2 partition assignments, and a maximum depth of 1. The nine attempts comprised eight investigation attempts and one protected finish. Provider-reported latency summed to 33.601 seconds. Cache reads and writes reported zero tokens, and there were no provider or validation errors.

| Invocation | Provider / corpus attempts | Reported tokens | Evidence and finish |
| --- | ---: | ---: | --- |
| FY2023 child | 4 / 3 | 16,171 | Received two unique chunks, with one repeated direct lookup. Exhausted its investigation allowance, then used `finish_research` with required tool choice and 332 output tokens. Delivered `child-1:finding-1`, citing `a19fccaa6f0cd45a5fb8c0a8`. |
| FY2024 child | 3 / 2 | 12,825 | Two searches delivered two unique chunks. Finished during investigation and released one unused provider slot. Delivered `child-2:finding-1`, citing `02834a6d106b2138d0425352`. |
| Root | 2 / 0 | 10,287 | Received and declared both findings used. Its final response used 976 output tokens to synthesize a comparison citing both original chunks. Both partitions retained partial coverage. |

This establishes live model-directed delegation, protected finalization, child-to-parent evidence transfer, and synthesis with Claude. The protected finish used fewer than 1,024 output tokens, so this run does not establish a need for the increased finish cap. At this milestone, depth-two delegation and truncation recovery had only been tested offline.

### Configuration and manual reproduction

With the local canonical files available and temporary AWS credentials configured outside the repository, run from the repository root:

```bash
BEDROCK_REGION=us-east-1 \
  BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  CLAUDE_MAX_OUTPUT_TOKENS=1024 PYTHONPATH=Code/src \
  python Code/examples/demo_rlm_recursive.py --live --trace \
    --finalization-tool-choice required --finalization-max-output-tokens 2048
```

The successful run used the configuration above. Use `--finalization-tool-choice auto` for a separate run with the default policy if the provider rejects a named request. The demo disables automatic retries even if `CLAUDE_MAX_RETRIES` is set. The Bedrock SDK permits one transport attempt per call. The request timeout defaults to 20 seconds and is capped by the remaining time.

| Limit | Default | Option |
| --- | ---: | --- |
| Shared recursion depth | 1 | `--max-depth` |
| Shared delegated invocations | 2 | `--max-delegations` |
| Shared provider attempts | 12 | `--max-provider-attempts` |
| Shared corpus attempts | 8 | `--max-corpus-attempts` |
| Shared partition assignments | 2 | `--max-partitions` |
| Shared reported-token threshold | 60,000 | `--max-reported-tokens` |
| UTF-8 bytes per observation | 6,000 | `--max-response-bytes` |
| Shared and per-invocation deadline | 120 seconds | `--deadline-seconds` |
| Model decisions per invocation | 6 | `--max-iterations` |
| Tool attempts per invocation | 4 | `--max-tool-calls` |
| Provider attempts per child | 4 | `--child-provider-attempts` |
| Reserved finish-only provider attempts per invocation | 1 | `--finalization-attempts` (`0` restores the unreserved policy) |
| Output tokens per investigation response | 1,024 | `CLAUDE_MAX_OUTPUT_TOKENS` |
| Output tokens per protected finish response | 2,048 | `--finalization-max-output-tokens` |
| Finish-only tool choice | `auto` | `--finalization-tool-choice {auto,required}` |

The shared counts are checked before new work. Both deadline settings use the same flag so the root's usual 30-second controller default cannot shorten the experiment. Children inherit the time remaining. The deadline remains cooperative for blocking SDK calls. Token admission uses reported usage after a request, so an individual response can cross the threshold and missing counters can hide usage. The run never raises its budgets in response to an error.

With reservations enabled, the configuration must leave at least one investigation attempt after reserving finalization for the root and for each configured child. Reduce the child count or allocation explicitly when lowering the provider cap.

### Reading the report

The recursive demo exits `0` for a finished synthesis using acknowledged child findings from both years, `1` for incomplete execution, and `2` for invalid setup. Direct root citations cannot substitute for the required delegated findings in this demo. Partial analysis and coverage remain visible even when the execution criteria pass.

The JSON records requested tools, executed operations, delivery, original citations, partition coverage, errors and shared usage. Add `--trace` for per-decision and per-attempt detail. Private reasoning and raw provider payloads stay outside the report. Use the root elapsed time for wall-clock comparisons. Per-invocation times include descendants, and summing them double-counts time.

**Semantic review of the successful run.** The final answer inferred a shift toward geopolitical risks in FY2024 from two independently selected excerpts. An earlier FY2023 run retrieved `bd263969e91358073d0ea0a1`, which explicitly discusses geopolitical tensions involving Taiwan and China and their implications for supply continuity. The latest excerpts cannot establish that geopolitical risk was new or received greater emphasis in FY2024. The latest FY2023 quotation also does not independently support its finding's additional claim about demand estimation challenges. That child returned `finished` with a finding marked `analyzed`, while its invocation-level analysis status remained `incomplete`.

Both NVIDIA partitions remained partial, and the demo reports per-invocation coverage.

### Offline verification of the live path

Stubbed Bedrock and Anthropic responses verify both-years delegation in one reply, provider result IDs, sequential execution, malformed and duplicate IDs, mixed finish rejection, mid-batch budget exhaustion, and the aggregate byte cap. The default scripted demo still runs without provider attempts. These checks do not estimate live latency, token savings, or answer quality.

The starvation regression reproduces 12 provider attempts, 13 model decisions, 8 corpus attempts, and no final citations with reservations disabled. Other stubbed tests inspect named finish choice and output caps in both provider payloads, preserve pending single and batched results across truncation, and verify usage, runtime rejection, evidence validation, and budget stops. A combined regression recovers the root after truncation and still reports the experiment incomplete when the FY2024 child's finish attempt requests a prohibited search. A direct FY2024 citation cannot satisfy the delegated-finding criterion.

| Check | Result |
| --- | --- |
| Focused RLM suite, including starvation, finalization, and truncation recovery | 138 passed, 1 warning in 2.51s |
| Related retrieval, RAG provenance, and canonical corpus regressions | 40 passed, 1 warning in 17.41s |
| Default scripted demo on the canonical corpus | Exit 0, two original chunk citations, zero provider attempts |

Both test runs reported the existing AnyIO pytest deprecation. Reproduction uses the same offline test commands from the recursive milestone above.

## Sequential corpus-wide research

The V4 checkpoint finished one two-company tree and left 23 companies outside model investigation. `execute_corpus_research()` takes the original resolved `ResearchPlan` and uses the same admission policy to schedule whole company groups in ticker order. Each batch starts a fresh delegation-first tree. A group keeps its requested years and sections together. Unresolved requirements, groups too large to fit, and batches left pending at a resource stop remain in the original denominator. Explicit company subsets use the same scheduler.

`CorpusLimits` sets the maximum batches, provider attempts, reported tokens, corpus tool calls, delegations, partition assignments, and elapsed time. Admission requires enough remaining allowance for one full configured tree, including the final smaller batch. The scheduler charges actual counters after each tree, including work already charged when a tree raises an exception. This conservative policy can leave unused capacity below the next tree's allocation. The existing profiles, protected finish reservations, and local limits apply within each tree. Each tree inherits the earlier of the overall deadline and its own deadline. The deadline is cooperative and cannot interrupt a blocking request or filesystem read.

The optional lexical scan runs once inside the overall deadline with separate chunk and time caps. Its pages, scanned chunks, candidates, and matching spans are reported separately from model tool calls and assessed evidence. A group with zero lexical matches remains eligible. Estimates from attempted provider requests are reported separately from measured tokens and missing counters. A response can cross a reported-token threshold, and missing usage can hide consumption. The next tree starts only if the remaining reported-token allowance can cover its allocation.

Aggregation runs in Python without a corpus-sized model conversation. Invocation and finding IDs carry a run and batch namespace. Coverage unions original chunk IDs, delivered span IDs, and section character ranges against the full plan. Reports track retained root findings, acknowledged child findings, terminal selections, citations, and comparisons. Findings from a failed child contribute delivered evidence only after acknowledgement. Exact source excerpts and SEC provenance survive deduplication. Conflicting assessments and uncertainty stay visible. The deterministic answer reports completed execution, unresolved assessments, and pending scope. Partial access or lexical coverage cannot establish absence or novelty.

Run from the repository root. The first command uses scripted providers. The limited command deliberately exits `1` after returning its partial report. A live invocation requires both `--live` and an explicit JSON `--limits-file` containing `CorpusLimits`.

```bash
PYTHONPATH=Code/src python Code/examples/demo_rlm_corpus_research.py
PYTHONPATH=Code/src python Code/examples/demo_rlm_corpus_research.py --limited
PYTHONPATH=Code/src python Code/examples/demo_rlm_corpus_research.py --company AMD --company NVDA
```

| Offline canonical run | Full | Limited |
| --- | ---: | ---: |
| Required / resolved filing-section targets | 50 / 50 | 50 / 50 |
| Batches launched | 13 (12 pairs and one singleton) | 1 |
| Company groups attempted / finished / pending | 25 / 25 / 0 | 2 / 2 / 23 |
| Lexical chunks / pages | 3,038 / 328 | 3,038 / 328 |
| Model corpus calls / partition assignments | 2 / 50 | 2 / 4 |
| Original citations / provider attempts | 2 / 0 | 2 / 0 |

The scripted AMD example delivers the two original water-and-power spans and one batch-root comparison. The other company scripts abstain and remain semantically unassessed. Both reports retain incomplete research status. The limited run stops at the overall partition allowance and keeps the first batch's evidence. Independent semantic verification and live corpus-wide reliability remain unmeasured. Scheduling and aggregation state is in memory for this milestone.

Verification passed **246 focused RLM tests** and **53 related retrieval, provenance, and corpus integrity regressions** with the existing AnyIO deprecation warning. The offline corpus-wide, limited, explicit-subset, planning, planned research, recursive, and comparative demos ran with zero provider attempts. The focused and related test commands are in the earlier verification section.

### Six-company live checkpoint and bounded recovery

Run `4255c0412c34480882f8fde99e5fcace` (`rlm-corpus-six-expanded.json`) used the expanded profile on AAPL, AMD, AMZN, GOOGL, MSFT, and NVDA. All **6 requested companies were scheduled and attempted**. All **12 resolved Item 1A targets** had partial access. Of three batch roots, one finished and two failed. Five children finished and one failed. Five companies had pending work. The report held **17 acknowledged child findings, 4 root findings, 20 retained evidence spans in the accepted inventory, and 3 final root citations**. Execution and research both remained `incomplete`.

| Measure | Two-company V4 | Simple V4 × 3 | Six-company run |
| --- | ---: | ---: | ---: |
| Provider attempts | 19 | 57 | 52 |
| Reported tokens | 141,521 | 424,563 | 401,314 |
| Corpus operations | 26 | 78 | 76 |
| Elapsed seconds | 86.4068 | 259.2204 | 308.505648646 |
| Finished roots | 1 | 3 | 1 |

The six-company run estimated **438,919 tokens** and had **zero missing usage records**. Batches reported **127,700**, **137,172**, and **136,442** tokens. The lexical scan covered **762 chunks across 82 pages** in **0.016620522 seconds**. The six-company timer includes scan and coordination, while the V4 figure measures its recursive tree. The extrapolation is arithmetic. Different filings, tool choices, findings, and failed invocations prevent causal efficiency comparisons or cost conclusions.

#### Six-company reliability validation

Run `c00c4eda1b6640719a18e4ee87790f4d` (`rlm-corpus-six-expanded-validation.json`) used the same expanded ceilings and company pairs. The corpus demo now shares the standalone synthesis guidance. Both runs scheduled and attempted all six companies, scanned all 762 requested chunks across 82 pages, and recorded partial access for all 12 sections. Execution and research remained `incomplete` with independent semantic verification `false`.

| Measure | First six-company run | Reliability validation |
| --- | ---: | ---: |
| Finished children / roots | 5 / 1 | 6 / 2 |
| Companies with pending work | 5 | 2 (AMZN, GOOGL) |
| Acknowledged child findings / root findings / final root citations | 17 / 4 / 3 | 19 / 10 / 10 |
| Provider attempts / corpus operations | 52 / 76 | 59 / 74 |
| Reported / estimated tokens | 401,314 / 438,919 | 460,211 / 505,122 |
| Elapsed seconds | 308.51 | 315.32 |

The validation reported no provider timeout, missing usage, or shared-budget exhaustion. The number of finished roots rose alongside token and attempt counts. Different retrieval and generation workloads and incomplete roots limit performance comparisons.

#### Shared observation-byte delivery

The first run logged `batch_observation_limit` and `tool_call_limit` at the Batch 1 root. AMD finished with four validated findings and no acknowledgements. Code inspection confirmed that an overflowing batch member was replaced wholesale with an error. The original telemetry cannot identify AMD as the overflowing member or prove that overflow caused the missing acknowledgements.

The controller reserves error envelopes for native tool IDs, then apportions remaining bytes across remaining operations. The comparative fallback packs whole validated findings and comparison summaries whose referenced findings fit in the same result. The full byte cap still applies. Omitted findings retain their original IDs and exact SEC provenance in the ledger. Only offered findings in a successful, acknowledged observation enter the parent's evidence. Submitted IDs mark an attempted request. The stub regression delivers both siblings when their individual payloads fit and their combined full payloads exceed the cap. Failure-isolation tests keep unacknowledged excerpts outside accepted evidence and citations.

In validation, the three delegation envelopes measured **10,438**, **14,302**, and **12,741 bytes**, below the **16,000-byte** cap. Full and offered payload sizes matched for all children. All **19** generated findings were offered, submitted, and acknowledged. The finished roots declared **10** child finding IDs used. The unacknowledged inventory was empty. This verifies live delivery of the full results. The compact fallback remains covered by offline tests, and it can still omit findings when the byte share is small.

#### Protected replay and its limit

The first run logged terminal truncations at the first two roots and the NVIDIA child. The old controller rejected replay in the protected phase even with a finish slot available. This was a code defect, but the original telemetry cannot assign each truncation to a phase or establish that another slot remained.

The controller now permits replay in either phase when valid request history and a protected slot remain. Recovery discards partial output, keeps pending native tool results and IDs, and credits acknowledged evidence once. Finish-only restrictions and decision, provider, token, and deadline checks still apply. The comparative prompt asks for shorter output on replay. Root and child Bedrock stubs verify recovery using a second protected slot, and the multi-batch stub verifies policy propagation and sanitized diagnostics.

In validation, Batch 1 recovered from an investigation truncation and finished with **1,990 output tokens**. Batch 2 truncated during investigation, then reached **2,048 output tokens** on both protected attempts. Each protected request measured **24,496 bytes** and **7,554 input tokens**. The root stopped on the second truncation, leaving Amazon and Alphabet pending despite acknowledged child findings. This verifies bounded protected replay in live execution, with successful protected recovery still unobserved. Default diagnostics record phases, stop reasons, output caps, recovery, observation sizes, and delivery status. `--trace` adds tool IDs, timing, and usage without raw messages or reasoning blocks.

The validation search ledger recovers Microsoft passages despite zero fixed-term candidates and Amazon's semiconductor supply excerpts in both years despite both relevant chunks missing from the shortlist. Public comparisons identify the risk and cite assessed chunk IDs. The explicit AI category requires an AI reference in an exact span, and contextual claims can use a separate same-company, same-year exact span. These checks preserve attribution for model assessments.

Microsoft FY2024 adds water to the reviewed operational risk wording. Alphabet's prior-year excerpt describes general data-center supply interruption, while its later excerpts name AI accelerator supply and deployment dependencies. NVIDIA's reviewed earlier excerpt concerns its own facilities, while the later excerpt describes customer expansion constraints. The model labeled these comparisons `conflicting` by treating missing support or differently scoped evidence as contradiction. The excerpts support differences in reviewed disclosure. Establishing contradiction requires evidence that opposes the same proposition, and establishing novelty requires stronger prior-year coverage.

Reproduce the targeted validation from the repository root with authorized Bedrock credentials. The existing root-level `evidenceroute-six-company-limits.json` specifies 3 batches, 108 provider attempts, 90 corpus operations, 6 delegations, 12 partition assignments, 600,000 reported tokens, and a 900-second deadline.

```bash
BEDROCK_REGION=us-east-1 \
  BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  CLAUDE_TIMEOUT_SECONDS=20 CLAUDE_MAX_OUTPUT_TOKENS=1024 PYTHONPATH=Code/src \
  python Code/examples/demo_rlm_corpus_research.py --live --trace --profile expanded \
    --limits-file evidenceroute-six-company-limits.json \
    --company AAPL --company AMD --company AMZN --company GOOGL --company MSFT --company NVDA \
    > rlm-corpus-six-expanded-validation-rerun.json
```

The two protected truncations motivate testing a more compact finish representation under the existing ceilings. The trace measures saturation and leaves the reason for the long generation unresolved.

Verification passed **254 focused RLM tests** and **53 related regressions**, each with the existing AnyIO deprecation warning. Bedrock stubs exercised compact sibling delivery, replay from a truncated protected finish, and shared policy propagation across trees. Offline six-company, full and limited corpus, comparative, planning, recursive, and Standard RAG demos passed their expected checks with zero provider attempts.

### Compact root synthesis by finding reference

In validation run `c00c4eda1b6640719a18e4ee87790f4d`, the root regenerated findings to assess evidence already validated by its children. Batch 1 finished at 1,990 output tokens and Batch 3 at 1,956. Batch 2 had seven acknowledged findings and exhausted both 2,048-token protected attempts. Total usage was 460,211 reported tokens across 59 attempts and 315.32 seconds. The new representation targets redundant root output under the same resource ceilings.

Comparative roots advertise `finish_research` with a compact `synthesis` payload. Results label delivered child findings with invocation-local `dN` references. The root reassesses those findings directly and may add independently read spans as `nN` findings, where the number is the one-based position in `new_findings`. The host renders the answer and derives the sorted, deduplicated original chunk citations and used child finding IDs. One rationale per assessment records the root's judgment. Child assessments and conflict labels do not determine the compact root outcome. Child and non-comparative finish schemas retain their existing formats. Legacy root decisions can still be validated for compatibility.

References become eligible only after acknowledgement of the delivered result. Compaction can leave gaps in `dN` numbering, and gaps grant no authority. Comparison evidence must stay within one company and the requested years. Contextual AI references require an exact AI span from the same company and year. Root-created findings require directly acknowledged spans in one document and section. Rejected synthesis leaves root selections and citations empty while the acknowledged-child and retained-unacknowledged ledgers remain available. Coverage and corpus reports distinguish original child findings selected by the root from findings generated by the root. Native tool IDs, caps, and protected replay rules are unchanged.

A four-excerpt fixture reconstructs the recorded Batch 1 Apple regulatory and AMD capacity assessments using canonical SEC spans. The raw validation JSON was absent from the checkout. For equivalent outcomes and assessed chunks, serialized finish input fell from **2,085 to 868 UTF-8 bytes (58.4% smaller)**. This measures the reconstructed tool-input JSON only. The available offline tokenizer was for the retrieval embedding model, so no Claude token estimate is reported. Live provider-token usage and latency remain unmeasured.

```bash
PYTHONPATH=Code/src python -m pytest -q -s Code/tests/test_rlm_comparison.py::test_serialized_compact_finish_fixture
PYTHONPATH=Code/src python Code/examples/demo_rlm_corpus_research.py
```

Verification passed **260 focused RLM tests** and **53 related regressions**, with the existing AnyIO warning. The Bedrock stub replayed a truncated compact finish with the same tool-result IDs and used two original child findings without regenerating either one. Offline corpus-wide, limited, six-company, comparative, planning, recursive, and Standard RAG demos passed with zero provider attempts.

Aliases add a small amount to child observations, which still fit or compact under the existing byte limit. The compact form does not guarantee that a protected finish fits its output cap.

#### Targeted live reference synthesis

Run `d34a62a374c647e18f50f3c257f36cf5` (`rlm-reference-synthesis-validation.json`) repeated the Amazon–Alphabet scope from Batch 2 of `c00c4eda1b6640719a18e4ee87790f4d` with the same per-tree ceilings. Both children finished in each run.

| Measure | Earlier Batch 2 | Reference synthesis |
| --- | ---: | ---: |
| Root | Failed | Finished |
| Provider attempts | 22 | 20 |
| Reported tokens | 179,036 | 156,778 |
| Elapsed seconds (approximate) | 126.47 | 83.04 |
| Root reported tokens | 31,710 | 11,403 |

The earlier root had three truncated responses, including two protected finishes. The new root finished in investigation with **843 output tokens** under the ordinary 1,024-token cap. It selected all six acknowledged child findings, generated zero new root findings, and cited seven original chunks. The run recorded zero errors. Execution finished and research remained incomplete. The 12,790-byte delegation envelope carried both full results, and the root finished without replay. These are two observations with different investigation and synthesis call sequences, which preclude attribution of the time and token differences to the compact format alone. The earlier batch's metrics come from the previously reviewed baseline. Its raw report was absent from this checkout.

From the repository root with authorized Bedrock credentials, use the expanded profile and the one-tree limits file from the experiment (`max_batches` 1, `max_provider_attempts` 36, `max_reported_tokens` 200000, `max_corpus_attempts` 30, `max_delegations` 2, `max_partitions` 4, `deadline_seconds` 300, `max_scan_chunks` 30000, and `scan_deadline_seconds` 30).

```bash
BEDROCK_REGION=us-east-1 \
  BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  CLAUDE_TIMEOUT_SECONDS=20 CLAUDE_MAX_OUTPUT_TOKENS=1024 PYTHONPATH=Code/src \
  python Code/examples/demo_rlm_corpus_research.py --live --trace --profile expanded --fixed-terms \
    --limits-file rlm-two-company-limits.json --company AMZN --company GOOGL \
    > rlm-reference-synthesis-validation-rerun.json
```

### Question-dependent candidate discovery

The targeted live run scanned **208 chunks** and found **15** initial candidates with the fixed terms. Five of its seven assessed chunks were outside that set. Subsequent scoped exploration found the missing passages, including both Amazon semiconductor-supply chunks and Alphabet's references to third-party infrastructure. The earlier six-company run likewise recovered Microsoft evidence after a zero-candidate scan.

`DiscoverySpec` validates the intent topic and up to 32 caller-supplied phrases, including spelling, acronym, and word-form variants. When phrases are omitted, the leading clause of the topic supplies the search phrase. The scanner reuses the BM25 tokenizer's case, Unicode compatibility, and punctuation normalization with stop words preserved, then matches contiguous token phrases. This catches formatting variants such as hyphenation. The original phrase-only scan required explicit variants for fused words, abbreviations, and inflections. In the original phrase-only scan, ranking favors the longest matched phrase, then the number of distinct matches, then the original chunk ID. The expansion in the corpus-wide follow-up adds contextual anchors and inflection handling. Each section retains at most 16 candidates by default and reports its total matches separately. The example phrase set lives in the demo. The engine works with any resolved plan, including the one-year review below.

Reads interleave exact filing sections and divide the remaining chunk allowance across the current round. A large filing or high candidate count cannot take another section's candidate allowance. The scanner records section metadata reads, page reads, scanned chunks, unvisited targets, and deadline or limit stops. The page cap defaults to the chunk cap and can be set lower with `max_scan_pages` in corpus limits. Metadata lookups have a separate upper bound of one per resolved section, and all discovery runs inside the existing scan and overall deadlines. A budget too small for one round can leave some sections unscanned. The deadline is cooperative at corpus-call boundaries.

The initial scan makes zero provider calls or extra BM25 passes. BM25 remains the scoped `search_chunks` follow-up under the existing model tool and corpus limits. Its token-based ranking can surface passages that lack a contiguous configured phrase. Zero-candidate groups stay eligible for investigation. Candidate IDs and span hints confer no citation authority. Exact span delivery, reference-based synthesis, and source verification still use their existing paths. Missing or differently scoped prior-year evidence still cannot establish contradiction or novelty.

| Offline canonical scope | Scan | Chunks / page reads / metadata reads | Candidate chunks | Known targets recovered |
| --- | --- | ---: | ---: | ---: |
| Amazon, Alphabet, Microsoft FY2023–FY2024 Item 1A | Eight literal terms | 321 / 35 / 6 | 15 | 2 of 9 |
| Same sections | Normalized example phrases | 321 / 35 / 6 | 40 | 9 of 9 |
| Apple and NVIDIA FY2024 Item 1A | Foreign currency phrases | 133 / 14 / 2 | 6 | No labeled set |

The offline AI check uses seven assessed chunks from the Amazon–Alphabet validation and two Microsoft excerpts from the earlier investigation. The known targets informed the example phrases, so the comparison measures regression-target recovery. It does not estimate corpus recall, precision, or semantic correctness. The broader example produced more candidates and can surface incidental matches. The unrelated review found five candidates for Apple and one for NVIDIA without model assessment.

```bash
PYTHONPATH=Code/src python Code/examples/demo_rlm_discovery.py
PYTHONPATH=Code/src python Code/examples/demo_rlm_comparative.py
PYTHONPATH=Code/src python Code/examples/demo_rlm_comparative.py --fixed-terms
```

The research demos use normalized discovery by default. Repeat `--term` to supply phrases, or use `--fixed-terms` for the literal baseline. Verification passed **266 focused RLM tests** and **53 related regressions**, each with the existing AnyIO deprecation warning. Discovery, both comparative modes, full and limited corpus-wide research, the Amazon–Alphabet subset, planning, recursive, and Standard RAG demos passed offline.

#### Live discovery validation

Run `7ae604dd29d14d53bd01326cba9f0394` (`rlm-discovery-live-validation.json`, code at `c784491`) repeated the Amazon–Alphabet reference-synthesis scope of `d34a62a374c647e18f50f3c257f36cf5`. Both runs required and resolved four FY2023–FY2024 Item 1A targets and scanned 208 chunks under the expanded per-tree ceilings.

| Measure | Reference synthesis, literal terms | Normalized discovery |
| --- | ---: | ---: |
| Initial candidate chunks | 15 | 19 |
| Assessed chunks among initial candidates | 2 of 7 | 5 of 5 |
| Provider attempts | 20 | 17 |
| Reported tokens | 156,778 | 127,726 |
| Corpus tool attempts | 24 | 24 |
| Elapsed seconds | 83.04 | 79.40 |

The normalized metrics were checked against the saved report. The reference column uses the recorded checkpoint and the historical corpus-attempt count supplied for this comparison. The scanner recovered all seven known regression targets in those two companies using 23 chunk-page reads and four metadata reads in 0.0192 seconds. The offline count of 40 normalized candidates also included Microsoft and measures a larger scope. Known targets informed the example phrases, and the two observed runs had different investigation sequences. General recall and causal effects on latency, token usage, and retrieval quality remain unmeasured.

All three invocations finished. The root acknowledged and selected five child findings, added zero findings, and cited five original chunks supporting seven evidence spans. It finished during investigation with 924 output tokens under the 1,024-token cap. The Alphabet child logged one local tool-limit rejection. The report recorded zero provider truncations and terminal failures. Execution finished and research remained incomplete.

The Alphabet assessments treated missing prior-year accelerator or energy specificity as contradiction. The prior-year excerpt describes a more general dependency and does not oppose either current-year proposition. The FY2024 accelerator excerpt supports intense competition for supply capacity. It does not fully support the assessment that competitors could claim capacity first. Independent semantic verification remains open.

To reproduce this checkpoint with authorized Bedrock credentials, use the reference-synthesis command above with `--fixed-terms` omitted, `--limits-file rlm-discovery-limits.json`, and output redirected to `rlm-discovery-live-validation-rerun.json`. The saved limits file uses the same overall ceilings as the reference-synthesis run.

### Natural-language question interpretation

`natural_language.research_question(question, runner, limits=...)` accepts the original question and an existing recursive runner. `BedrockIntentInterpreter` requests a forced Converse tool use for both requirements and discovery phrases. It sends recorded company names and tickers, available fiscal years per company, and the set of filing section IDs and titles. The request contains no filing excerpts. The host validates the native tool response, constructs a `ResearchIntent`, checks 1–32 lexical phrases through `DiscoverySpec`, and uses the existing deterministic planner and canonical corpus IDs. The result includes the original question, scope assumptions, unresolved requirements, and execution telemetry.

| Question or requirement | Interpretation policy |
| --- | --- |
| Explicit company, year, or filing section | Preserve named requirements, including unavailable ones. Claude maps unambiguous aliases using the inventory, and Python resolves the resulting ticker or exact recorded name. Unknown names and unavailable filing or section slots stay in the plan. A group with an unresolved requirement remains pending under the existing whole-group admission policy. |
| Corpus-wide scope | Apply all corpus companies to explicit corpus-wide requests and unqualified “which companies” questions. Mark the latter company universe as inferred. Mark comprehensiveness for questions that require evaluating every requested target. |
| Unspecified or relative fiscal years | A descriptive review with no year uses the latest fiscal year in the snapshot, FY2024 here, as an inferred requirement even when it is unavailable for a named company. Relative periods and missing comparison years call for clarification. |
| Unspecified or ambiguous section or company | For risk-factor questions with no named section, Python defaults to Item 1A and records the inferred scope. Ask for clarification when company scope or identity is ambiguous or when the appropriate section is unclear. Requests for full filings or all sections are unsupported because the corpus contains selected sections. |
| Objective | Descriptive SEC disclosure reviews use scoped candidate pages and canonical evidence spans. Supported AI infrastructure risk comparisons require exactly two distinct fiscal years and comprehensive review within the requested scope, including yes/no novelty questions. Other comparison topics, predictions, recommendations, and objectives requiring external data return an unsupported outcome. |

The resolved intent records defaults in `inferred_fields` with a concise `scope_notes` explanation. The host materializes the default section, validates inferred years, and checks that the execution mode matches the objective. Only a valid executable interpretation reaches plan resolution. Schema and corpus checks establish shape and resolvability. Fidelity to the wording of the question, including alias judgments and scope extraction, still depends on the model.

Interpretation defaults to at most two attempts, 1,024 output tokens per attempt within the configured Claude cap, a 16,000-token charge threshold, and a 30-second cooperative deadline. The phase uses the lower of its limits and the overall allowances. The existing Converse transport applies the configured retry policy, credential chain, timeout caps, and sanitized error codes. Each attempt records transport latency, available provider latency and reported token counters, estimated tokens, output cap, stop reason, and errors. A missing input or output counter charges the larger of the request estimate or the available reported usage. Admission checks the estimate before each request, and an observed overage blocks research. Estimates use the existing request-byte heuristic. A blocking SDK operation can overrun the deadline.

Research receives the remaining provider-attempt and token allowances and the original absolute deadline. The scheduler applies the configured tree reservations, protected finalization, and corpus and scan ceilings. Overall usage includes the interpretation attempts and actual reported and estimated counters. The interpretation report also shows the token charge used for admission. The timer starts after the caller has loaded the corpus. Discovery searches only resolved filing sections. Citation authority requires acknowledged evidence reads. A miss cannot establish absence or novelty, and research can remain incomplete after every invocation finishes.

Run the scripted verification from the repository root. Each question produces a different mocked structured reply through the same Bedrock adapter and runs scoped research over one canonical FY2024 Item 1A section. The Amazon review scanned 44 chunks and the Apple foreign exchange review scanned 53. Each finished execution with one original citation and incomplete research. The CLI records one mocked interpretation attempt and zero live provider attempts per question.

```bash
PYTHONPATH=Code/src python Code/examples/demo_rlm_question.py \
  "What AI infrastructure supply risks does Amazon disclose in FY2024 Item 1A?"
PYTHONPATH=Code/src python Code/examples/demo_rlm_question.py \
  "What foreign exchange risks does Apple disclose in FY2024?"
```

For a future authorized Bedrock run, add `--live --limits-file rlm-discovery-limits.json` to either command with the Bedrock environment from the course setup section. The default diagnostic profile leaves room within those overall ceilings for interpretation. An expanded tree requires an overall file that also covers interpretation. `--interpretation-limits-file` can set a smaller phase allowance. The prior scripted demos remain reproducible.

Verification passed **294 focused RLM tests** (28 for the question path) and **53 related retrieval, provenance, and corpus integrity regressions**, each with the existing AnyIO deprecation warning. Use the focused and related commands in the earlier verification section. The planning, planned research, recursive, discovery, both comparative modes, full and limited corpus research, Amazon–Alphabet subset, and Standard RAG demos also ran offline. The limited corpus demo returned its expected partial result and exit code 1.

Mocked responses verify request construction, validation, budget settlement, scoped candidate access, and original evidence provenance. The following checkpoint records the first live observations. Independent semantic verification and durable findings remain future work.

#### First live question experiments

The two saved reports use Bedrock Claude Sonnet 4.5 at `91b9cce`, with the expanded research profile and the root-level `rlm-question-limits.json` and `rlm-interpretation-limits.json` files. The overall ceiling was 38 attempts, 216,000 reported tokens, and 330 seconds. Interpretation could use two attempts, 16,000 charged tokens, and 30 seconds, with a 1,024-output-token cap per request.

| Measure | A · Amazon–Alphabet novelty comparison | B · Apple foreign exchange review |
| --- | ---: | ---: |
| Interpretation attempts | 1 | 1 |
| Interpretation reported / charged tokens | 3,298 / 3,298 | 3,243 / 3,243 |
| Interpretation output tokens | 329 | 282 |
| Interpretation elapsed seconds | 4.59 | 4.01 |
| Research attempts / reported tokens | 23 / 167,651 | 0 / 0 |
| Research corpus attempts / elapsed seconds | 26 / 103.26 | 0 / — |
| Overall attempts / reported tokens / elapsed seconds | 24 / 170,949 / 107.85 | 1 / 3,243 / 4.01 |

**A.** Run `b26afad5f7b2444a8231971a83e648b4` in `rlm-question-comparative-validation.json` resolved AMZN and GOOGL, FY2023–FY2024, inferred Item 1A, and selected the year-over-year objective. Its 20 generated phrases produced 33 candidates from 208 chunks, covering six of the seven known two-company regression targets. The matched scope allows a descriptive comparison with the earlier 19-candidate, seven-target discovery run. The phrase sets, investigation sequences, and finalization outcomes differed, so these observations do not establish general recall or causal effects on retrieval, cost, or latency.

Amazon accessed relevant evidence in both years. It exhausted its protected finishes after a truncation at 2,048 output tokens and an `invalid_findings` rejection at 1,328. Alphabet and the root finished. The root recovered from `invalid_findings` on its second protected finish, acknowledged and selected five Alphabet findings, added zero findings, and returned seven evidence spans from four original citation chunks. Execution and research remained incomplete. Alphabet's child treated the third-party equipment and infrastructure dependency for AI deployment as comparable in both years. The root treated it as insufficient and judged the evidence to concern different risks. The interpreter returned `comprehensive=false` even though resolving novelty requires review across the requested scope.

**B.** In `rlm-question-foreign-exchange-validation.json`, Bedrock returned `tool_use` and the interpreter stopped at `invalid_interpretation`. The response was neither truncated nor a transport failure. The saved report contains no rejected structured payload or field-level diagnostic, so the exact validation failure is unknown. Plan resolution, discovery, and recursive research never started.

#### Interpretation validation and correction

The old `invalid_interpretation` code covered JSON and schema errors, inconsistent scope requirements, and invalid discovery phrases. The report now links up to four unique diagnostics to each rejected attempt. Each diagnostic uses a known schema path and a fixed category and code. Extra fields identify their parent path. The diagnostic excludes submitted values, unknown keys, provider messages, and reasoning.

A response with a valid native tool envelope and invalid structured input can receive one correction request. The new Converse conversation uses the original question, inventory, and forced schema with sanitized feedback. Rejected assistant content is discarded. The host validates the new submission in full and checks it against the recoverable scope from the first response. A repair can fix invalid inferred defaults or complete missing fields. Previously validated company, year, and section requirements and the comparative objective are protected from replacement. Interpreting the original wording remains a model judgment.

The correction uses the remaining attempt, estimated-token, output, and time allowances. Initial transport retries use the same attempt pool, and the correction has one transport send. The report records the correction attempt number when a send occurs, and the overall totals include its usage even when it fails. Valid ambiguous or unsupported outcomes, truncation, and native protocol failures end interpretation without a correction. Research starts only after successful validation and budget settlement.

Experiment A exposed that the original scope rule allowed `comprehensive=false` for a novelty question. The interpreter now requires comprehensive review for the supported two-year comparison mode within the requested companies, years, and sections. Descriptive reviews retain their existing policy. The flag describes the evidence needed to answer, while execution limits and incomplete-research semantics still govern what the run can establish.

```bash
PYTHONPATH=Code/src python -m pytest -q Code/tests/test_rlm_natural_language.py -k 'correction or two_year_ai'
```

The mocked Apple review exercises a missing-field repair and passes the corrected plan into scoped research, ending with one original citation. Verification passed 313 focused RLM tests and 53 related regressions, each with the existing AnyIO warning. Both question demos and the earlier planning, recursive, discovery, comparative, corpus, and Standard RAG demos ran offline. The limited corpus demo returned its expected partial result and exit code 1.

The exact rejected payload from the first live Apple run remains unknown. Amazon's protected finalization failures and the root–child disagreement on the Alphabet comparison remain open.

#### Apple correction validation

At `58078c4`, the rerun in `rlm-question-foreign-exchange-correction-validation.json` returned `tool_use` without truncation on both attempts. Correction ran on attempt 2. Both attempts reported `invalid_inferred_sections` in the `inconsistent_requirements` category at `requirements.section_ids`. The saved report omits the rejected section IDs. The run consumed 6,715 reported and charged tokens in 7.56 seconds. Plan resolution, discovery, and recursive research never started.

The previous contract asked Claude to generate inferred section IDs, then validated them against a fixed Item 1A default. The provider now selects `explicit` with all requested `section_ids`, or `risk_factors_default` with an empty list. Python supplies `item_1a` for the default, adds `section_ids` to `inferred_fields`, and records that the plan covers only Item 1A. Explicit IDs reach the existing planner, where an unavailable section such as Item 7A stays unresolved.

Correction feedback spells out both forms of the contract. A malformed default can be corrected by clearing its IDs. Correction preserves an explicit section selection even if the default would resolve to the same Item 1A target. Requests with unclear scope can still return a clarification. The existing retry and resource ceilings apply.

The offline Amazon demo supplies explicit Item 1A. The Apple demo selects the Python default. The mocked correction regression submits a representative malformed default, repairs it, and exercises the resulting scope through research. Verification passed 323 focused RLM tests and 53 related regressions with the existing AnyIO warning. Both question demos and the legacy demos ran offline.

#### Live Apple validation with deterministic section selection

Run `ee6e51027fb348ed9cf351fbf070ef30` interpreted the Apple foreign-exchange question on the first attempt as an AAPL FY2024 review with inferred Item 1A. The interpreter generated 13 discovery phrases and used 3,367 reported and charged tokens in approximately 3.09 seconds, with no validation failures or correction. The scan examined 53 chunks and found five candidates. The child received all five candidate chunks in full.

Research used 12 provider attempts, 74,172 reported tokens, and nine corpus operations in approximately 72.99 seconds. The overall run used 13 attempts and 77,539 reported tokens in 76.08 seconds. The child had a truncated investigation response followed by two `invalid_findings` protected finishes. The root made two recovery searches, then received `invalid_action` and `invalid_findings` on its protected finishes. Both exhausted their finalization allocations.

Interpretation succeeded. Research execution and research completeness remained incomplete, with no accepted findings or citations and no independent semantic verification.

### Incremental finding retention

At `d7b2d7c`, both finish paths saved findings only after validating the entire terminal payload. One rejection could discard every draft in that payload, and a truncated reply never reached validation. Descriptive research advertised quote-based findings and rejected span references. The Apple trace kept only the rejection codes, so the exact failure is unknown. The earlier Amazon investigator also returned zero accepted findings after protected finalization failed.

- `submit_findings` validates up to four findings per investigation operation and retains each valid item immediately. It takes a claim, the model's analysis status, optional uncertainty or conflict, and up to three acknowledged chunk/span pairs from one document and section. Python extracts the exact text and supplies SEC URLs, company identity, year, section and the partition's access status at acceptance. Discovery hints grant no evidence access. A span read grants only the excerpts delivered, and a read earlier in the same tool batch still needs acknowledgement.
- Each invocation can retain eight unique local findings across investigation and finish. Repeating equivalent evidence and claim fields returns the existing ID. A submission uses one local tool slot and one shared corpus attempt, counted under `finding_submission_attempts`. Receipts and child results use the existing byte and batch allowances. The update keeps the configured provider, retry, token, time and tool ceilings.
- Both objectives use the existing reference synthesis path. `sN` selects an accepted local finding after its receipt is acknowledged. `dN` selects an acknowledged child finding. Comparative synthesis can assess both reference types. Python derives original citations from the selected evidence. Legacy quote finishes and terminal `nN` span drafts still work and validate atomically. A failed finish leaves previously retained findings available and terminal selections empty.
- A failed child can offer retained findings with the original stop reason and incomplete status. The parent can use only excerpts that fit and were acknowledged. The reports distinguish retained, offered, submitted, acknowledged and selected findings. Validation diagnostics store fixed categories, codes and field paths, such as `synthesis.cite_refs[0]` for an unacknowledged reference. They exclude rejected values and private provider content.

The two scripted Apple runs each used one mocked interpretation request, scanned 53 canonical FY2024 Item 1A chunks, and found five candidates. Each child read three exact spans from one chunk and recorded one finding in two charged corpus operations. The successful run finished both invocations. With `--fail-finish`, the controller rejected an unknown child reference and the child stopped. The parent acknowledged the retained finding, finished by selecting `d1` and produced one original citation. Overall execution and research remained incomplete, so the command returned exit code 1. An additional regression fails the root finish as well and verifies that partial evidence remains available with zero terminal citations.

The comparative demos with normalized and fixed-term discovery each returned two original citations and a `comparable_both_years` assessment. The offline comparison test also uses submitted local references in a child assessment and delivered references at the root. Bedrock stubs exercised truncation, timeout and finalization exhaustion after acceptance. Verification passed 336 focused RLM tests and 53 related regressions, each with the existing AnyIO deprecation warning. No paid providers were called.

```bash
export PYTHONPATH=Code/src AWS_EC2_METADATA_DISABLED=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python Code/examples/demo_rlm_question.py "What foreign exchange risks does Apple disclose in FY2024?" --trace
python Code/examples/demo_rlm_question.py "What foreign exchange risks does Apple disclose in FY2024?" --fail-finish --trace
python Code/examples/demo_rlm_comparative.py
python -m pytest -q Code/tests/test_rlm_*.py
```

Submitting during investigation spends a tool operation and can consume a provider decision within the same ceilings. A small observation may defer receipts or omit child findings while the retained records stay available in the trace.

#### Live Apple validation

Run `09a6ea22e5cf4784a6b892ab73bb870f` in `rlm-apple-incremental-live-validation.json` interpreted the AAPL FY2024 Item 1A review on its first attempt. Interpretation reported and charged 3,368 tokens in 3.31 seconds and generated 15 discovery phrases. The scan examined all 53 chunks and found five candidates. The child received four candidates and one additional chunk as whole text.

The child submitted `s1`–`s4` at decision five and `s5` at decision six. All five were accepted and acknowledged. The child finished by reference with 104 output tokens, and the root's protected finish used 79. The 6,726-byte child observation delivered all five findings. The root acknowledged and selected all five, added zero new findings, and cited seven exact spans from three original chunks. Of the three submission operations, two accepted child findings and one rejected four root drafts with `unacknowledged_span`. Two subsequent root reads were skipped at the reinvestigation tool limit.

Research used 11 provider attempts, 84,343 reported tokens and 12 corpus operations in 37.81 seconds. Overall usage was 12 attempts and 87,711 reported tokens in 41.12 seconds. Both invocations and execution finished, and research remained incomplete. The previous Apple run used 13 overall attempts and 77,539 reported tokens in 76.08 seconds and returned zero accepted findings or citations. These two runs used different discovery phrases and investigation sequences. The observed differences do not isolate a performance effect. Failed-child retention was exercised offline because the live child finished.

The selected spans include fragments such as “dollar value of the gross margins” and “dollar, while generally beneficial”. The strengthening-currency finding also claims increased component costs and adverse margin effects, while its selected excerpt describes international pricing and derivative losses. The component-cost consequence appears in the adjacent, unselected span. These cases motivate independent entailment checks for fragments and each part of compound claims.

### Durable checkpoints and targeted continuation

The API in `evidenceroute.research.rlm.checkpoint` captures a settled corpus or question result with `create_checkpoint`, writes it with `save_checkpoint`, and validates it with `load_checkpoint`. Version 1 keeps the original intent and resolved plan, accepted findings with stable IDs, recorded evidence deliveries and selections, coverage, comparisons, assignments, citations, and per-run allowances and usage. A question checkpoint also records the interpretation charge. The UTF-8 JSON format is capped at 16 MiB and 64 run records. Saves sync a private temporary file beside the destination and replace the destination atomically.

Loading checks the version and schema, compares a fingerprint of the loaded canonical filing, section and chunk records, and resolves the stored intent against that snapshot. It verifies each excerpt, citation, span and filing-section identity against canonical data and checks the consistency of recorded access and invocation scope. The fingerprint identifies the snapshot. Historical acknowledgement remains a record from a trusted local artifact, and claim entailment remains unverified.

`plan_continuation` selects unresolved executions in plan order. Completed groups with partial semantic coverage stay out of the default selection. Explicit company selection and an optional goal can direct investigation of a gap within a resolved whole group. Missing requirements remain pending against the original denominator. Each `continue_research` call requires an explicit `CorpusLimits` allowance and uses the existing scheduler and runner. The checkpoint preserves prior successes and failures, unions evidence and access by identity, and reports current, historical and cumulative usage.

Historical findings enter a new invocation through `read_prior_findings`. The operation charges a local tool slot and a shared corpus attempt, offers up to four whole findings from that invocation's authorized scope within the existing byte limits, and grants reference access after delivery is acknowledged. A historical excerpt grants access to its own text and source. Coverage counts acknowledged chunks, spans and ranges once across runs.

```bash
export PYTHONPATH=Code/src AWS_EC2_METADATA_DISABLED=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python Code/examples/demo_rlm_checkpoint.py --initial-checkpoint rlm-checkpoint-initial.json \
  --continued-checkpoint rlm-checkpoint-continued.json > rlm-checkpoint-offline-report.json
python -m pytest -q Code/tests/test_rlm_checkpoint.py
```

The offline demo saved and reloaded a two-company FY2024 Item 1A review after Apple finished and Amazon's child failed. It resumed Amazon alone, delivered its retained finding to the new child, and added a second Amazon span. The merged state preserved Apple's finding ID, two required targets, three findings and two original citation chunks. Initial and continuation runs charged four and three corpus attempts, respectively. Overall execution finished and measured coverage stayed partial. The continuation tests also exercised a zero-allowance run and a comparative round trip with counted provider stubs. Verification passed 32 checkpoint tests within 368 RLM tests, plus 53 related regressions. The test runs reported the existing AnyIO deprecation warning. All work for this checkpoint used zero paid provider calls.

Version 1 checkpoints settled runs through explicit saves. Conversation history, automatic crash recovery, cross-snapshot migration, concurrent writer coordination and semantic verification are outside this format. The merged research status remains incomplete after a focused continuation.

#### Live checkpoint validation

`rlm-checkpoint-live.py` ran each phase in a separate Python process with the expanded per-tree profile and a separately authorized one-batch corpus allowance. The FY2024 Item 1A review required Amazon and Alphabet. Loading both checkpoints against the canonical corpus confirmed that the complete first run record and all three original finding IDs survived in the continued checkpoint.

| Stage | Execution and evidence | Attempts / reported tokens | Corpus operations / seconds | Lexical chunks scanned |
| --- | --- | ---: | ---: | ---: |
| Initial | AMZN finished with three retained findings and two citation chunks. GOOGL was pending at `batch_limit`. | 11 / 78,882 | 14 / 44.30 | 107 |
| Continuation | The planner selected GOOGL alone. It finished with four additional findings and four citation chunks. | 10 / 71,734 | 14 / 37.50 | 63 |
| Merged | Both assignments completed against the original two-target denominator. Seven findings and six unique citation chunks survived. | 21 / 150,616 | 28 / 81.81 | 170 |

The initial scan examined 44 Amazon and 63 Alphabet chunks before the batch cap stopped investigation. The continuation scanned the same 63 Alphabet chunks again. The cumulative count measures scan work across 19 pages and three metadata reads. The records also show 176,056 estimated tokens, two delegations, two partition assignments and two finding-submission attempts. Execution finished and research remained incomplete, with partial access for each filing section. Both continuation invocations recorded zero acknowledged prior finding IDs. Same-company historical delivery was established offline.

### Independent semantic verification

`verify_checkpoint` reviews retained findings and comparisons in separately authorized batches. Each batch is a fresh Bedrock Converse invocation with one native `submit_verification` tool. The verifier receives the claim or comparative conclusion and numbered exact selected excerpts with fiscal years. Investigator assessments, rationales and conversation history stay outside the request. An invocation using the same underlying model can share its systematic errors. The results record a second model judgment, with statistical independence and semantic correctness still unestablished.

The verifier must list each material component, including qualifiers, causal links and consequences. Each component receives a support, contradiction or insufficient-evidence assessment. Support and contradiction require selected evidence references. Insufficiency requires a concise gap. Python derives the finding result from the components. An explicit contradiction yields `contradicted`. A mix of supported and insufficient components yields `partially_supported`. Uniform support yields `supported`, and uniform insufficiency yields `insufficient_evidence`. Failed, skipped and missing assessments remain `unverified`. The host checks references and result consistency. Completeness of the component list still depends on the verifier.

Comparative reviews assess the original conclusion and separately classify the risk relationship as shared, more specifically worded, different or unclear. A resolved relationship requires selected excerpts from both years. Earlier-year nonmention and an earlier statement that explicitly opposes the risk have separate fields. The host rejects contradiction attributed solely to a year marked nonmention or insufficient. A different risk in the selected excerpts leaves filing-wide novelty unresolved.

The current span splitter divides at periods followed by whitespace, including after `U.S.`. Apple's strengthening-currency span `848:1098` therefore begins mid-sentence and omits both the opening condition and the adjacent component-cost sentence. The hedge-effectiveness span `1442:1667` omits the preceding forwards-and-options sentence. The verifier sees selected evidence only. Historical span IDs and the segmentation algorithm retain their existing meanings.

`VerificationLimits` bounds attempts, estimated admission and charged usage, generated output, cases per run and batch, selected-evidence payload bytes, and elapsed time. A batch uses at most one provider attempt. Missing usage charges at least the request estimate, and measured overage or late replies retain their charges and leave the batch unverified. Reports show current and cumulative verification usage alongside research and combined usage. Version 2 checkpoints attach review results to stable finding and comparison identities, exact subject digests and the source run prefix. Version 1 files still load. Coverage and original findings remain available, and gap recommendations identify authorized filing sections for optional continuation.

```bash
export PYTHONPATH=Code/src AWS_EC2_METADATA_DISABLED=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python Code/examples/demo_rlm_verification.py > rlm-verification-offline-report.json
python -m pytest -q Code/tests/test_rlm_verification.py
```

The offline fixture uses Apple's two recorded compound claims and reconstructs the Alphabet comparison from canonical FY2023–FY2024 excerpts. In two mocked attempts, the Apple fixture produced two partial results for component costs and hedge contract types, one supported assessment of limited hedge effectiveness, and one contradicted hedge guarantee. In three mocked attempts, the Alphabet fixture produced five supported assessments and one insufficient assessment of the historical conflict claim. Its comparative results distinguished more-specific accelerator wording from a different product-recall risk while leaving filing-wide novelty unresolved. Both fixtures kept research incomplete.

The verifier CLI requires `--live --checkpoint FILE --limits-file FILE` for provider requests and accepts `--kind comparisons` and `--save-checkpoint FILE`.

#### Live semantic-verification results

The saved Apple and Alphabet runs sent scripted research controls to the live verifier. A third run reviewed the actual two-company continuation checkpoint. All three used `rlm-verification-limits.json`. The six source and reviewed files loaded against the canonical corpus. The reviewed files preserved the original research runs, findings, comparisons, citations and coverage. The three JSON reports matched the reviewed records.

| Saved report | Cases | Provider attempts | Reported and charged tokens | Seconds | Recorded verdicts |
| --- | ---: | ---: | ---: | ---: | --- |
| `rlm-verification-aapl-live-report.json` | 4 findings | 2 | 4,286 | 6.26 | One unverified, one partial, one supported, one contradicted |
| `rlm-verification-googl-live-report.json` | 3 findings and 3 comparisons | 3 | 6,484 | 9.93 | Five supported, one contradicted |
| `rlm-verification-checkpoint-live-report.json` | 7 finding records | 4 | 9,487 | 15.42 | Six supported, one partial |
| **Total** | **17 case records** | **9** | **20,257** | **31.62** | **16 valid structured assessments** |

- **Apple.** The hedging review identified the contract-type evidence gap and supported limited hedge effectiveness. The positive limited-effectiveness control and negative full-offset guarantee received their expected verdicts. The strengthening-currency case remained `unverified` with `invalid_assessment`. The exact rejection reason is unavailable. The valid hedging assessment in the same batch survived.
- **Alphabet.** The verifier assigned `contradicted` to the original conflict conclusion and classified the accelerator wording as `more_specific_wording`. It cited the FY2023 general supply-risk passage as support for the underlying risk. It distinguished product recalls from infrastructure supply risks and recorded earlier-year nonmention with filing-wide novelty unresolved. The supported verdict for that comparison applies to its conclusion that the available excerpts leave novelty uncertain.
- **Actual checkpoint.** Alphabet child finding 4 received partial support. Its selected excerpts describe general systems vulnerabilities, while the earthquake-location and AI-specific infrastructure assertions require additional evidence. Amazon root finding 1 and child finding 1 have distinct finding IDs and the same subject digest. The seven case records represent six distinct verification subjects. Research remained incomplete after verification.

The actual checkpoint used 21 research attempts, 150,616 reported tokens and 81.81 seconds. Adding verification brought the recorded sums to 25 attempts, 160,103 reported tokens and 97.23 seconds. Verification added approximately **6.3%** in reported tokens and **18.9%** in measured execution time relative to the recorded research run. These figures describe one checkpoint. The aggregate 16 of 17 measures structured-assessment completion and includes repeated subject matter.

The verdicts are model assessments. For example, the actual Alphabet accelerator finding received support for a causal link between competitors claiming capacity first and difficulty negotiating terms. Its selected SEC sentence presents those as alternative conditions that could cause supply constraints. Control agreement and valid evidence references leave this kind of entailment error possible.

## Corpus-wide acceptance follow-up

The saved Sonnet 4.5 run `ce45e24c55744535b81e80a0756e3428` in `rlm-acceptance-corpus-live.json` asked which companies introduced a new AI-related infrastructure risk in FY2024. It resolved all **25 companies and 50 Item 1A targets**. The scan took **0.250 s** over 3,038 chunks and 4,603,503 characters, including overlap. The 22 candidates were distributed across AMD (15), Amazon (4), and NVIDIA (3). Three batches attempted six companies. Five assignments completed, AMD failed, and 19 groups stopped at the batch limit.

The trace records **55 attempts**, 442,230 input and 23,559 output tokens (**465,789 reported**), 76 corpus operations, and **240.72 s** including interpretation. Research requests summed to 1,357,610 serialized bytes, with a 26,040-byte median and 45,918-byte maximum. The acknowledged retrieval operations delivered 58 chunk-ID occurrences across invocations, 14 of which repeated an ID within the invocation. This count includes distinct spans from the same chunk. The report lacks enough detail to recover the number of verbatim repeat excerpts. The root acknowledged 12 findings, selected nine and cited nine original chunks. It recorded zero finding submissions, zero established novelty conclusions and no separate verification. Execution and research remained incomplete.

AMD exceeded `comparisons[1].assessments[0].rationale` and then combined sources from different filings in one finding. Apple hit the mixed-scope rule. Bank of America exceeded `infrastructure_connection` before recovering and treated prior-year nonmention as contradiction. Some Apple and Amazon claims described both years using excerpts from one year. ConocoPhillips' root and child disagreed on a comparison that conflated support for the underlying risk with support for novelty.

### Discovery, comparative capture and final reporting

- **Discovery.** Apply normalized question phrases and conservative singular/plural variants. For AI infrastructure topics, rank contextual co-occurrence of AI, infrastructure and risk anchors. Strong infrastructure/risk co-occurrence can surface prior-year counterparts with no AI wording. Specific phrase and contextual matches outrank generic terms. Each section keeps its own deterministic candidate cap. Scoped BM25 remains available for gaps. Candidates and lexical misses grant no evidence authority or absence verdict.
- **Capture.** The `full-corpus` profile reserves one existing investigator decision and tool opportunity before the protected finish slots. A dedicated turn uses named native `submit_findings` when the investigator has yet to complete a submission. An empty list is valid. Each accepted finding is retained immediately, with canonical filing scope and exact excerpts derived in Python. The controller records submission attempts, accepted findings, rejected records and unused dedicated turns. Provider failure, token admission or the deadline can still stop an invocation before capture. Finishing by reference avoids regenerating accepted evidence. Legacy profiles retain their prior finish contracts.
- **Comparisons.** Atomic captures cite spans from one filing and section. The capture path rejects explicit comparisons against the other requested year and common relative-year formulations in the claim. The comparative finish links year-specific references. It separates passage support, risk category, AI specificity and disclosure relationship. Nonmention, different risks and explicit opposing evidence have separate representations. Python derives source years and requires appropriate evidence from both years for a resolved relationship. Risk categories distinguish AI infrastructure dependencies, AI-enabled threats, general infrastructure risks, unrelated risks and uncertainty. Invalid reference assessments produce sanitized per-record diagnostics while valid peers remain usable. Semantic errors in unrestricted claim text can still require verification.
- **Reporting.** `create_checkpoint`, `review_checkpoint`, and `verification_report` give a supported investigation-and-review path. The verifier receives the selected exact canonical excerpts and adjudicates components. Original findings and comparisons stay in the audit inventory. The final answer identifies unverified, insufficient, partial, supported and contradicted comparisons. It presents supported components from selected evidence with SEC references, and leaves flagged investigator disagreements unresolved. Related comparisons are matched by normalized risk label or overlapping evidence. Rephrasings over disjoint passages can require human review. Filing-wide novelty remains unresolved. Select a gap's company and goal for a caller-authorized continuation and then authorize a separate verification pass. New comparison identities require their own review. No automatic investigation-verification loop runs.

### Full-corpus scheduling and accounting

`full-corpus` uses two company groups per batch and the diagnostic profile's per-tree ceilings. The tree allows depth 1, 18 provider attempts, 6 child attempts, 12 corpus operations, 100,000 reported tokens and a 180-second deadline. Local limits are 6 decisions, 6 tool attempts and 6,000 observation bytes. Two finish attempts per invocation have a 2,048-token output cap and named finish selection. Root and child token headroom are 12,000 each. The profile disables root follow-up retrieval and uses explicit targeted continuation for evidence gaps. Singleton batches reserve only one delegation and the group's two filing sections. Other per-tree ceilings still apply.

Scheduling prioritizes groups with candidates across both years and higher candidate counts, with deterministic ticker tie breaking. All resolved groups, including zero-match groups, remain eligible. The overall allowance must cover the next tree's configured resources before admission. Reported-token thresholds can be exceeded by an in-flight response, and the deadline is cooperative. Checkpoint version 3 stores the complete, validated initial candidate inventory for compact comparative runs and preserves verification history from the earlier format. Resume reuses the inventory without another lexical scan. Earlier checkpoint versions still load. The original denominator and historical finding IDs survive continuation. Completed assignments stay out of default reinvestigation even if root synthesis failed. The report retains pending execution, failures, unresolved semantic outcomes and separate research, verification and combined usage. Save at bounded pass boundaries. An interruption inside a pass resumes from the last saved checkpoint.

### Offline measurements

Both scans below used the original interpretation phrases and the same 50 sections. Toggle `DiscoverySpec.expand_context` to reproduce the phrase-only policy. Times are local medians of five runs. Known evidence contains seven historical Amazon and Alphabet chunks plus two Microsoft chunks. It is a regression set.

| Measure | Original phrase matching | Contextual discovery |
| --- | ---: | ---: |
| Chunks / characters scanned | 3,038 / 4,603,503 | 3,038 / 4,603,503 |
| Matches before caps / retained candidates | 22 / 22 | 172 / 162 |
| Companies / sections with candidates | 3 / 6 | 15 / 29 |
| Known evidence found | 2 of 9 | 9 of 9 |
| Alphabet / Microsoft candidates | 0 / 0 | 12 / 12 |
| Median scan time | 0.250 s | 0.553 s |

Contextual discovery adds incidental candidates. Distribution is uneven. NVIDIA had 30, AMD 28, Intel 21, Meta 19 and Amazon 17. Ten companies had zero candidates. The foreign-exchange regression retained the same candidates under both modes. This comparison measures candidate recall on known passages. It leaves retrieval precision and held-out semantic quality unmeasured.

For the same AMD excerpt, the dispatcher serialized 1,479 bytes for a whole chunk, 566 for a single exact span and 166 for a repeated span reference. The last saves 400 bytes each time that result appears in native history, or 1,200 bytes across three subsequent requests. Exact duplicates within an authorized section can reference a previously acknowledged canonical span in an overlapping chunk. New chunk IDs receive no access credit from such references. Native tool ordering, original result IDs, and acknowledgement rules are preserved. The JSON finish schema fell from 4,707 bytes in the legacy child or 3,772 in the legacy compact root to 2,273 bytes in the capture workflow. These are serialized sizes. They exclude any live token or latency effect and depend on the actual turn sequence.

Bedrock documents explicit prompt caching for Sonnet 4.5 through Converse, including cross-region inference, with a 1,024-token prefix minimum and up to four checkpoints. The installed SDK exposes `cachePoint` for `tools`, `system` and message content. The adapter sends no cache checkpoints. Tool sets and budget notices change at phase boundaries, so a later trial should measure eligible stable prefixes within phases and examine actual cache reads and writes. The current implementation claims no cache benefit. [Sonnet 4.5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-5.html) · [Bedrock prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)

The canonical scripted profile scheduled **25 companies in 13 batches**, including a singleton, and retained all **50 required targets**. The constrained pass used six delegations and 12 partitions before admission stopped it with `insufficient_delegations`. A fresh process resumed the other 19 groups across ten batches from the saved checkpoint, scanning zero additional lexical chunks. The merged report preserved two historical findings and two original SEC citation chunks. It recorded 28 corpus operations, 25 submission attempts including empty submissions, and 50 partition assignments. A separate AMD failure fixture accepted two findings and cited both after two deliberately invalid finishes. A native stub also covered truncation followed by an invalid reference. The separate verifier fixture used two mocked requests and 1,500 synthetic reported tokens. The merged answer reported partial support for the AMD comparison and printed the supported general infrastructure component while leaving the AI link and novelty unresolved. The remaining company outcomes were scripted abstentions. These experiments exercised execution and accounting with zero paid provider calls.

### Live validation commands

Use the [README budget generator](../README.md#recursive-research-rlm) and the Sonnet 4.5 AWS environment printed there. The CLI uses fixed scope for reproducibility. The Python integration adds natural-language interpretation and charges it to the overall allowance. These live commands await validation.

```bash
export PYTHONPATH=Code/src

# AMD capture and finish regression
python Code/examples/demo_rlm_corpus_research.py --profile full-corpus --live --trace \
  --company AMD --limits-file rlm-milestone-single-limits.json \
  --save-checkpoint rlm-amd-live-checkpoint.json > rlm-amd-live.json

# Small comparison with separately budgeted verification
python Code/examples/demo_rlm_corpus_research.py --profile full-corpus --live --trace \
  --company AMD --company GOOGL --limits-file rlm-milestone-single-limits.json \
  --verification-limits-file rlm-milestone-verify-limits.json \
  --save-checkpoint rlm-small-verified-live-checkpoint.json > rlm-small-verified-live.json

# Full denominator, bounded first pass
python Code/examples/demo_rlm_corpus_research.py --profile full-corpus --live --trace \
  --limits-file rlm-milestone-constrained-limits.json \
  --save-checkpoint rlm-full-live-checkpoint.json > rlm-full-live-initial.json

# In a new process, continue from the settled checkpoint
python Code/examples/demo_rlm_corpus_research.py --profile full-corpus --live \
  --resume rlm-full-live-checkpoint.json --limits-file rlm-milestone-continuation-limits.json \
  --verification-limits-file rlm-milestone-verify-limits.json \
  --save-checkpoint rlm-full-live-checkpoint.json > rlm-full-live-continued.json
```

Use `rlm-milestone-full-limits.json` on the first call for an allowance covering all 13 trees in one pass. Inspect pending assignments after any live allowance stops. For a specific verification gap, add `--company TICKER --goal "Find earlier-year evidence for the same mechanism"` to a resume call and supply fresh allowances. A separate verifier invocation, even with the same model, provides a semantic assessment whose accuracy remains subject to review. Full-corpus live execution, live cost and latency changes, and held-out quality remain open.


## AMD live capture follow-up

The focused AMD run `4837bdbd50f34b999b2543b88f8b0e20` at `71c6733` used the economical `full-corpus` profile. The saved report and validated checkpoint record **6 attempts, 23,300 reported tokens, 16.96 seconds, 187 scanned chunks and 20 candidates**. Four calls delivered six exact spans. One scoped search subsequently returned candidate IDs, and the other search was skipped at the local read cap. The named capture tool received a valid empty list. The investigator and root both finalized by reference. The checkpoint holds zero findings, comparisons and citations, and no submission records were rejected. Execution finished with unresolved analysis.

### What the delivered evidence supported

| Fiscal year | Original chunk and span | Supported observation |
| --- | --- | --- |
| 2023 | `a13ada334ad8f009b2a43f01` `359:520`, `521:724` | Demand varies across AMD end markets. AMD reported significant AI accelerator demand in Q4 2023. |
| 2024 | `e7fbf3d91de1c9c912d9ab11` `970:1167` | AMD reported significant AI accelerator demand since Q4 2023. |
| 2024 | `e8e013b9b919159fb6168bca` `626:787`, `788:999` | Similar demand language recurred in a second chunk. |
| 2024 | `26bd14970249e9ca31c54908` `1319:1486` | AMD expected intense competition in the supply of GPUs and other AI accelerators. |

These are valid year-specific observations. They establish demand variability, demand history and expected competition. The selected sentences and their surrounding chunks leave a particular infrastructure dependency unresolved. FY2023 rank 2 repeats the FY2024 competition sentence. Neither demand nor competition establishes a new infrastructure constraint.

The stronger mechanism was already in discovery. FY2023 rank 3 (`043757f52188a8c6fb2a2857`, `296:513`) says that power or water shortages could impair customers' data-center expansion and consumption of AMD products. FY2024 rank 5 (`483314af3b949a29920abbbf`, `393:680`) repeats the mechanism and adds a potential revenue impact. The executed FY2024 search surfaced that later chunk ID after the allotted reads, and no exact text from either mechanism reached the model. Those two spans support separate general infrastructure findings and a shared underlying constraint. The AI-specific causal link remains unresolved. Other retained candidates concern third-party operations, supplier constraints, export controls and regulation. The root-level `rlm-amd-live-audit.json` contains the six verbatim excerpts, their chunk context, and the ranked candidates with canonical URLs.

### Capture request and targeted changes

The live request contracted from **15,084 to 12,156 bytes** and **5,166 to 4,310 input tokens** at capture. The Converse adapter appends the latest native tool results to its message list and sends the full history with the new tool list. The six span IDs were already acknowledged on the previous turn. The capture request offered only `submit_findings`, used named tool choice, and replaced the investigation budget notice with the capture instruction. An offline replay at `71c6733` retained the earlier messages as an exact prefix. Its tool configuration shrank by 4,805 bytes and its system prompt by 176 bytes, while messages grew by 1,044 bytes. The replay used substitute search arguments because the artifact omitted the original arguments and raw payload. It demonstrates request construction and evidence retention. It cannot determine how the model evaluated those excerpts or why it chose an empty list.

The capture prompt and schema now ask for atomic supported observations from one filing. An evidenced infrastructure risk remains eligible while AI linkage, another fiscal year or novelty is unresolved. The investigator records those gaps as uncertainty. Demand alone supports a demand observation. An empty list remains valid when the excerpts support no useful scoped observation.

The comparative shortlist surfaces one infrastructure mechanism hint per year when the retained lexical candidates contain both a strong infrastructure anchor and a disruption or constraint term. The existing ranking and candidate inventory remain available. The investigator starts with original text from each year, can combine up to four nearby spans from one chunk in one call, and can follow a lower-ranked mechanism or use scoped search as needed. For a resolved two-year assignment, the controller reserves one scoped original-text attempt for each year before the final exploration slots can be used elsewhere. The reservation also applies to a year with zero candidates. This protects an opportunity. Provider behavior, tool errors, deadline and token admission still affect actual coverage.

### Quality profile and continuation

The economical `full-corpus` profile retains its ceilings. Use `full-corpus-quality` for the next live validation. Its nine local tool slots allow four investigation operations per year and one capture. Its nine child provider slots allow six adaptive investigation turns, a capture turn and two protected finishes. Two child pools plus two root assignment turns and two root finishes require 22 shared attempts. The 160,000-token threshold provides room for the longer conversation history. It is a configured ceiling subject to in-flight reported usage, and the actual costs still require live measurement.

| Per-tree setting | Economical | Quality |
| --- | ---: | ---: |
| Shared provider attempts / corpus attempts | 18 / 12 | 22 / 18 |
| Local decisions / tools / child attempts | 6 / 6 / 6 | 9 / 9 / 9 |
| Reported token threshold | 100,000 | 160,000 |
| Observation bytes / deadline | 6,000 / 180 s | 6,000 / 180 s |
| Depth / root follow-up tools / protected finish attempts | 1 / 0 / 2 | 1 / 0 / 2 |

Read `assignment_evidence` for execution state, retained finding counts and years, and findings with verifier support. A completed zero-finding assignment has unresolved analysis. Default continuation selects unresolved execution, and an explicit `--company AMD --goal "Inspect the water and energy dependency in both filings"` selects the completed AMD assignment for a bounded revisit. `outstanding_verification` counts unreviewed or previously unverified current cases. The verifier CLI's `--outstanding` uses `review_checkpoint` to advance the saved review inventory. Settled reviews, including insufficient-evidence judgments, are reused. Each pass has a fresh explicit allowance. Cases that remain unverified may need smaller batches, revised evidence, or explicitly selected IDs.

### Next live validation

The prepared JSON files are generated by `python rlm-quality-limits.py` in this workspace. Their explicit caps are below. All research files also set 3,038 scan chunks, 328 scan pages, and a 10-second scan allowance. The continuation has capacity for all 25 groups even if the whole first pass fails. Completed groups remain outside its default selection.

| Limits file | Batches | Attempts | Reported tokens | Corpus calls | Delegations / partitions | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rlm-quality-tree-limits.json` | 1 | 22 | 160,000 | 18 | 2 / 4 | 180 |
| `rlm-quality-initial-limits.json` | 3 | 66 | 480,000 | 54 | 6 / 12 | 540 |
| `rlm-quality-full-limits.json` | 13 | 286 | 2,080,000 | 234 | 25 / 50 | 2,340 |

`rlm-quality-verification-limits.json` allows 16 requests, 64,000 charged tokens, 32 items, two items per request, 1,024 output tokens, 16,000 evidence input bytes, and 240 seconds per pass.

```bash
export PYTHONPATH=Code/src BEDROCK_REGION=us-east-1
export BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
export CLAUDE_TIMEOUT_SECONDS=20 CLAUDE_MAX_OUTPUT_TOKENS=1024

python Code/examples/demo_rlm_corpus_research.py --profile full-corpus-quality --live --trace \
  --company AMD --limits-file rlm-quality-tree-limits.json \
  --save-checkpoint rlm-amd-quality-live-checkpoint.json > rlm-amd-quality-live.json

python Code/examples/demo_rlm_corpus_research.py --profile full-corpus-quality --live --trace \
  --company AMZN --company GOOGL --limits-file rlm-quality-tree-limits.json \
  --verification-limits-file rlm-quality-verification-limits.json \
  --save-checkpoint rlm-amzn-googl-quality-live-checkpoint.json > rlm-amzn-googl-quality-live.json

python Code/examples/demo_rlm_corpus_research.py --profile full-corpus-quality --live --trace \
  --limits-file rlm-quality-initial-limits.json \
  --save-checkpoint rlm-full-quality-live-checkpoint.json > rlm-full-quality-initial-live.json

python Code/examples/demo_rlm_corpus_research.py --profile full-corpus-quality --live \
  --resume rlm-full-quality-live-checkpoint.json --limits-file rlm-quality-full-limits.json \
  --save-checkpoint rlm-full-quality-live-checkpoint.json > rlm-full-quality-continued-live.json

python Code/examples/demo_rlm_verification.py --live --outstanding \
  --checkpoint rlm-full-quality-live-checkpoint.json --limits-file rlm-quality-verification-limits.json \
  --save-checkpoint rlm-full-quality-live-checkpoint.json > rlm-full-quality-reviewed-live.json
```

The first full-corpus pass can exit 1 after saving partial progress. Each additional verification pass requires a new invocation with the explicit limits file. For the Amazon–Alphabet run, compare Amazon's paired semiconductor supplier excerpts (`b79aa2561327a953e141e2aa` and `4d1474550060251037c5f981`) and Alphabet's paired third-party equipment disclosures (`15787429383b3e9839f69716` and `62bad042e2e9d8334c2e6267`). Amazon states substantially the same AI supply dependency in both years. Alphabet states the deployment dependency in both years and names more cost drivers in FY2024. `python rlm-quality-positive-control.py rlm-amzn-googl-quality-live.json` audits candidate ranking, acknowledged text and retained evidence for those controls. The controls do not establish novelty.

Native offline regressions replayed the six AMD excerpts and exercised abstention, both-year capture, a missing earlier-year mechanism, and failed finalization after capture. The Amazon–Alphabet fixture retained four year-specific findings and independently reviewed six cases across two authorized stub passes. The second pass reused the first pass's two settled comparisons. The quality workflow retained the 25-company and 50-target denominator across restart. A separate failure simulation left 20 companies eligible after the first six attempts. Continuation investigated all 20 under the new allowance, reusing the initial scan and preserving the failed child's accepted findings. No paid requests were made. At this stage only the offline profile had been measured. The final experiment below records the subsequent live results.


## Final stabilization before corpus-wide evaluation

The quality runs at `b437d97` spent **8 attempts, 40,335 reported tokens, and 29.75 seconds** on AMD and **12 attempts, 63,847 tokens, and 40.68 seconds** on Amazon and Alphabet. Reviewing the second checkpoint added **3 attempts, 6,733 tokens, and 10.44 seconds**. AMD retained three findings from both years. One cross-year draft was rejected. A child risk label exceeded 120 characters and a root assessment used invalid AI context. The general water and power dependency was disclosed in both years, with a potential revenue consequence in FY2024. The selected evidence leaves the AI link unresolved.

Amazon's acknowledged excerpts included the limited semiconductor supplier statement in both years. The exact sentence appeared twice per year in overlapping chunks. The selected offsets at the end of the earlier overlapping chunks had no following text. The offsets in the later chunks ended just before the consequence sentence, which says availability constraints could adversely affect the ability to develop and operate AI technology. The explicit supplier dependency was already grounded. The trace shows an empty submission, zero rejected finding records, and later references to aliases that had never been issued. Native Converse replay kept the delivered results in history. Long history, repeated text, alias confusion, and novelty-oriented assignment wording remain hypotheses about the model's choice. The local audit includes the verbatim spans and their canonical context.

Alphabet retained four findings. The bundled root reply reduced its child result from **4,563 to 2,443 bytes** and offered only the first two FY2024 findings. The FY2023 finding remained in checkpoint state. Four historical finding reviews returned supported and two comparisons returned `invalid_assessment`. The original verification arguments were not recorded. Schema bounds, unknown evidence aliases, wrong-year references, and relationship/evidence inconsistency can produce that code. Historical marks of support also overstated some exact wording. Competitive accelerator capacity does not establish an actual shortage, and the deployment disclosure says it **may** depend on availability and pricing.

The stabilization keeps the quality ceilings unchanged.

- **Capture** indexes up to six acknowledged span locators per year and explicitly invites shared risks as separate, useful observations. Single or two-span reads can supply a following canonical span where it fits the existing byte share. Exact text, offsets, and aliases remain stable. An empty capture remains valid.
- **Root delivery** sends a local alias, fiscal year, claim, and whole selected quotes with canonical chunk and span IDs. Host-side findings keep full metadata. When space is tight, the packer tries a complete two-year comparison, including its context references, then a related cross-year pair. The exact Alphabet finding replay offered all four records in **2,879 bytes** with a two-child envelope of **4,384 bytes**. A pressure fixture offered the central pair in **1,630 bytes** and logged the other two as omitted. Acknowledgement remains restricted to delivered records.
- **Synthesis** gives explicit alias and label instructions. Invalid optional AI context is stripped and its category downgraded where needed, preserving general evidence. When all attempted comparisons fail on correctable inputs and usable aliases exist, a native invocation can spend one existing protected finish attempt on correction. Already valid comparisons stay available. No provider ceiling increases.
- **Recovery and verification** offer at most one lexically related pair per selected company, from up to twelve distinct retained excerpts or excerpt sets per year. A fresh review judges the relationship and the original year-specific claims. Proposals, results, diagnostics, and usage are recorded separately from investigator comparisons. Previously omitted findings retain their original access history. Unclear one-year reviews can provide no earlier evidence and request a gap. Invalid cases retain bounded field and code diagnostics while valid peer results survive. Narrow checks flag categorical dependency language and shortage claims unsupported by competitive-capacity wording. Finding reports can quote reviewed evidence even when a comparison fails. Historical review records are preserved.
- **Evaluation** keeps investigator metrics in new checkpoints so continuation can report company-level capture, rejection, provider usage, and latency. Root and shared verification usage stay in overall totals. Historical invocation telemetry can be supplied with the original detailed report.

Offline checkpoint-only recovery loaded the two actual checkpoints and selected AMD's water/power pair and Alphabet's deployment-cost pair, including its previously omitted FY2023 finding. The scripted verifier assessed both as more specific later wording with the underlying risk already present. Reload preserved the original findings, access ledgers and verification history. The canonical demonstration completed **25 companies, 50 targets and 13 batches**, including the singleton. An initial failure simulation resumed 20 assignments with zero additional scan and a separate verification pass. These are offline control results. The subsequent full-corpus experiment measures live capture and resource use. Independent verifier accuracy remains unmeasured.

### Final live commands

Use the Sonnet 4.5 environment and explicit quality limit files documented in the preceding section. `rlm-stabilization-recovery-limits.json` allows two requests, 12,000 charged tokens, two items, one item per request, 1,024 output tokens, 10,000 evidence input bytes and 60 seconds per recovery pass. The dated historical checkpoints remain valid input.

```bash
export PYTHONPATH=Code/src BEDROCK_REGION=us-east-1
export BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
export CLAUDE_TIMEOUT_SECONDS=20 CLAUDE_MAX_OUTPUT_TOKENS=1024

python Code/examples/demo_rlm_verification.py --live --recover --company AMD \
  --checkpoint rlm-amd-quality-live-checkpoint.json --limits-file rlm-stabilization-recovery-limits.json \
  --save-checkpoint rlm-amd-recovered-live-checkpoint.json > rlm-amd-recovered-live.json

python Code/examples/demo_rlm_verification.py --live --recover --company GOOGL \
  --checkpoint rlm-amzn-googl-quality-live-checkpoint.json --limits-file rlm-stabilization-recovery-limits.json \
  --save-checkpoint rlm-amzn-googl-recovered-live-checkpoint.json > rlm-amzn-googl-recovered-live.json

python Code/examples/demo_rlm_corpus_research.py --profile full-corpus-quality --live \
  --resume rlm-amzn-googl-recovered-live-checkpoint.json --company AMZN \
  --goal "Compare reliance on semiconductor suppliers and the effects of availability in both filings. Capture year-specific shared risks." \
  --limits-file rlm-quality-tree-limits.json \
  --save-checkpoint rlm-amzn-targeted-live-checkpoint.json > rlm-amzn-targeted-live.json

python Code/examples/demo_rlm_corpus_research.py --profile full-corpus-quality --live --trace \
  --limits-file rlm-quality-initial-limits.json \
  --save-checkpoint rlm-final-corpus-live-checkpoint.json > rlm-final-corpus-initial-live.json

python Code/examples/demo_rlm_corpus_research.py --profile full-corpus-quality --live \
  --resume rlm-final-corpus-live-checkpoint.json --limits-file rlm-quality-full-limits.json \
  --save-checkpoint rlm-final-corpus-continued-checkpoint.json > rlm-final-corpus-continued-live.json
```

The historical verification chain used the following inputs and outputs. Each row represents a separate authorization of the same limits. The command template describes the recorded experiment.

```bash
python Code/examples/demo_rlm_verification.py --live --outstanding \
  --checkpoint "$input" --limits-file rlm-quality-verification-limits.json \
  --save-checkpoint "$output" > "$report"
```

| Input checkpoint | Output checkpoint | Report |
| --- | --- | --- |
| `rlm-final-corpus-continued-checkpoint.json` | `rlm-final-corpus-reviewed-checkpoint.json` | `rlm-final-corpus-reviewed-live.json` |
| `rlm-final-corpus-reviewed-checkpoint.json` | `rlm-final-review-pass2-checkpoint.json` | `rlm-final-review-pass2.json` |
| `rlm-final-review-pass2-checkpoint.json` | `rlm-final-review-pass3-checkpoint.json` | `rlm-final-review-pass3.json` |

## Final full-corpus experiment

At the audited revision `44d9eba50e4c6111aaed63cac908576b2d783133`, the Sonnet 4.5 experiment used the `full-corpus-quality` profile and the Bedrock model configuration above. The artifacts record the profile and limits. The checkpoint does not attest the Git revision or deployed model ID. The question asked which companies introduced a new AI-related infrastructure risk in FY2024 relative to FY2023. The demo supplied the intent, discovery phrases and Item 1A scope programmatically. There was no natural-language interpretation request. The denominator was all 25 companies and both years, or 50 filing sections. Other sections could also contain relevant disclosures.

Each tree allowed two company investigators, depth one, four partitions, 22 provider attempts, 18 corpus attempts, 160,000 tokens and 180 seconds. The initial allowance admitted three batches. Five of six investigators completed. AMD retained three findings before its protected finish failed schema validation and reached `finalization_attempt_limit`. The initial checkpoint retained 11 findings overall. Continuation admitted ten two-company batches covering AMD and the 19 untouched companies. No completed assignment repeated. All **13 actual batches were pairs**, with 26 child invocations and 52 partition assignments against the original 25 companies and 50 targets. The singleton described in the offline demonstration belongs to a different execution.

The complete discovery cache matched across continuation. The first research run was an unchanged prefix of the continued checkpoint, including stable finding IDs, evidence references, quotations and access history. All three review checkpoints preserved the entire continued research state, including original comparisons, citations and counters. The canonical loader checked the snapshot identity, source metadata, chunk text, span offsets and access provenance. The final state was `execution_status=finished`, `research_status=incomplete`.

| Execution | Provider attempts | Reported tokens | Estimated tokens | Measured seconds | Newly settled reviews |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial research | 39 | 222,632 | 263,866 | 124.30 | — |
| Continuation increment | 141 | 794,092 | 941,207 | 401.83 | — |
| Research subtotal | 180 | 1,016,724 | 1,205,073 | 526.14 | — |
| Initial review | 16 | 40,636 | 48,729 | 68.60 | 29 |
| Review pass 2 | 16 | 37,217 | 46,542 | 48.11 | 29 |
| Review pass 3 | 10 | 24,001 | 29,331 | 34.87 | 18 |
| Review subtotal | 42 | 101,854 | 124,602 | 151.57 | 76 |
| Combined | 222 | 1,118,578 | 1,329,675 | 677.71 | 76 |

Provider attempts count requests. Reported tokens sum available input, output and any recorded cache-token counters. Estimates are separate request-budget estimates. No attempt lacked usage. The initial detailed trace records 211,303 input and 11,329 output tokens. Continuation and verification totals omit that split, so full-run monetary cost is unavailable. Seconds sum measured run intervals, excluding time between authorizations. Subtotals use unrounded times.

Research charged 54 + 180 corpus tool attempts and 6 + 20 finding submission calls. Accepted finding references numbered 11 + 33, with 1 + 1 rejected record diagnostics. The lexical inventory scan charged 328 pages, 50 section metadata reads and 3,038 scanned chunks initially and zero on continuation. Scanning a chunk does not measure delivery or understanding.

### Evidence yield and assessment inventory

Eleven companies retained both years (AMD, AMZN, BAC, GOOGL, HD, JPM, META, MSFT, PFE, PG and V). COST retained FY2024 alone. The other 13 completed with zero retained findings. All 50 merged target access records remained partial and none recorded complete section text. All 25 novelty conclusions remain unresolved. No observed trace establishes the prevalence of risks among the zero-finding companies.

| Cases | Supported | Partially supported | Insufficient evidence | Unverified | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Findings | 34 | 7 | 3 | 0 | 44 |
| Comparisons | 27 | 5 | 0 | 1 | 33 |
| Total | 61 | 12 | 3 | 1 | 77 |

Assessment completion was 76/77 (98.7%). These are model judgments on selected evidence. No independent accuracy measurement or exhaustive coverage estimate follows. `findings_with_verifier_support` includes both full and partial support (41/44). Historical comparisons deduplicate identical serialized records. Current comparisons retain the latest run's set per ticker and normalized risk label. Both inventories contain 33 records here, across 29 ticker/label keys. Related labels and multiple assessments of the same passages preclude interpreting the count as unique semantic claims. Later passes selected only outstanding cases and preserved all settled reviews.

Selected passages received `same_risk` for AMD power/water, AMZN semiconductor suppliers and META technical infrastructure, and `more_specific_wording` for AMD. HD, MSFT and PG each had a `different_risk` relationship between selected passages. That outcome cannot establish absence from an earlier filing. A separate HD assessment of AI implementation and integration remains unverified, ID `552b49c73bd1e1044a5c0c240c19178cfe4e87eb7ebe3dc58af0ba1df23ba687`. All three attempts recorded `invalid_assessment`, successively diagnosing `assessment/value_error`, `components.2.statement/string_too_long` and `comparison.earlier_year/inconsistent_support`.

The local evaluator intersects configured control chunk IDs with direct-span or whole-chunk access for receipt, and with retained finding evidence chunk IDs for retention. Amazon accepts either overlapping chunk in each year. Citation membership is a separate test. Checking the offsets confirms that AMD's water/power sentence and Amazon's supplier disclosure were delivered and retained in both years. Alphabet's FY2023 control had neither receipt nor retention. Its FY2024 deployment sentence was delivered at `62bad042e2e9d8334c2e6267:634:926` without a retained finding on that chunk. A chunk hit alone would also match another sentence from the same chunk.

The **separate earlier recoveries** gave AMD a supported shared water/power dependency with a more specific FY2024 potential revenue consequence. Alphabet's recovery supported the FY2023 component. Both FY2024 components were contradicted, with the same recorded gap, “e2 states 'may depend' not 'depends'”. The raw relationship was more specific wording, but the report withheld a verified relationship. These recoveries contribute zero to the full-corpus totals.

For offline reproduction with the saved root artifacts, run the following from the repository root. It uses the canonical loader, merged coverage and report functions. The earlier `rlm-final-corpus-evaluation.json` describes the first review stage.

```bash
set -o pipefail
PYTHONPATH=Code/src python rlm-stabilization-evaluate.py rlm-final-review-pass3-checkpoint.json | diff -u rlm-final-evaluation.json -
```

### Answer and integration decision

The initial research answer reports batch progress and defers substantive conclusions. The continued output has `result=null`; its checkpoint report lists company yield and unverified risk labels. After three reviews the answer contains qualified excerpts and supported components, interleaved with 25 assignment summaries. For example, “Verifier-supported component. Reliance on a limited group of suppliers for semiconductor products related to AI infrastructure such as GPUs”. The structured relationship and evidence fields are more informative than the text for some comparisons.

* **Verified publication.** The saved answer used one sentence for two relationships, omitted the different-risk judgment and collected ordinary comparison sources from components alone. Recoveries collected relationship sources without printing their IDs. The corrected renderer gives each eligible relationship distinct wording, preserves its evidence pointers in ordinary conclusions, and cites the original chunks inline for both ordinary and recovered relationships. Different-risk judgments leave filing-wide novelty unresolved. Existing verification and disagreement rules still govern publication.
* **Integration blockers for general dispatch.** Interpretation supports descriptive disclosure review and an explicit two-fiscal-year AI-infrastructure comparison. The demo's scripted scope and comparison taxonomy constrain comparative financial and other arbitrary benchmark questions. Discovery gates domain-specific context ranking on the topic. Scope-aware routing needs to surface unsupported questions. Before the general publication milestone, the quality-profile answer deferred substantive evidence to optional verification.
* **Citation contract.** Original `citations` belong to research. `finding_citations` covers supported finding excerpts, including those available only in structured data. `conclusion_citations` covers published comparative components and relationships. The additive `answer_citations` contains exactly the sources printed in the verified answer, deduplicated by canonical chunk ID in first citation order. Exact quotes and spans remain available through finding evidence. The shared `ResearchResponse` expects canonical `Chunk` citations and one verification status, so an adapter still needs to resolve `EvidenceRef` and preserve interpretation outcome, failed/pending execution, evidence gaps and per-case support. `research_status` has no dedicated evidence-abstention value.
* **Empirical limitations.** Sparse two-year capture, missed positive controls, overlapping claims, model judgments, 1.12 million reported tokens and 678 measured seconds make general routing and cost-effectiveness unproven. Comparative evaluation requires reviewed benchmark questions, including answerable and abstention cases. Presentation brevity is an optional optimization after the citation and result contracts are settled.

The offline verified-publication replay increased distinct printed IDs from 27 to 31 and conclusion citations from 17 to 21, including MSFT's earlier water-risk chunk `4815881d3fdefcf0a17e31d7`. AMD's supported recovery now cites both years inline. Alphabet's contradicted recovery remains unresolved. Every printed ID matched the answer citation collection and canonical evidence. The 44 findings, 33 historical comparisons, judgments, one outstanding comparison, incomplete research status, 25 unresolved novelty conclusions and resource totals matched the saved reports. The focused verification, comparison, checkpoint and corpus-reporting suites passed **146 tests** offline.

### General research publication

Corpus research publishes atomic investigator findings with company ticker, fiscal year, original claim text, qualifications and inline canonical chunk IDs. Descriptive runs use the same projection with or without lexical discovery. Finished roots contribute their recorded selections. Unfinished roots can contribute their own retained findings and acknowledged child findings under a partial-evidence heading. Unselected and undelivered findings remain inspectable in the ledger. The projection flags incomplete research, insufficient eligible evidence and the limits of lexical misses. Its claims are semantically unverified, including when viewed alongside a separate verified report.

`answer_citations` contains exactly the sources cited in `answer`, deduplicated by canonical chunk ID in first-citation order. Research-wide `citations` retain their original meaning. Exact quotes and span pointers remain in the finding evidence. `publication_state` reports `selected_evidence`, `partial_evidence` or `insufficient_evidence`. Legacy comparison text uses `comparison_evidence` when it cites assessed passages. These states describe publication yield. Execution status, research status, usage and coverage retain their existing semantics. Separate work is required to assess whether the evidence answers the question, reconcile cross-source operands and broaden natural-language admission.

`checkpoint.publication` reconstructs the same atomic evidence view across saved runs and continuations. Historical findings without reconstructible selections stay in the ledger and the answer flags the gap. Compact comparative research retains its structured comparisons and leaves filing-wide novelty unresolved. The verified renderer and its citation fields remain independent of this view.

Offline, `PYTHONPATH=Code/src PYTHONDONTWRITEBYTECODE=1 python rlm-publication-replay.py` replayed the saved artifacts without provider calls. The continued full-corpus output previously listed risk labels with zero printed citation IDs. The new view renders 36 selected findings with 30 aligned chunks. The earlier descriptive checkpoints render 3 findings with 2 chunks and 7 findings with 6 chunks. The final verified report still has 44 retained findings, 33 historical comparisons, 31 answer chunks, 21 conclusion chunks, 10 published relationships, one outstanding comparison, incomplete research status and unresolved novelty for all 25 companies. Saved judgments and usage matched.

The six focused research, natural-language, checkpoint, verification and comparison test modules passed 210 tests offline. Regression cases cover discovery parity, terminal and partial selections, missing historical selection records, empty and ineligible findings, conflicting claims, canonical evidence and citation order.

## Issue #16 acceptance review

The criteria below refer to the [issue checklist](https://github.com/ibrahimhamza01/EvidenceRoute/issues/16). Implementation evidence is relative to `Code/src/evidenceroute/research/rlm/` unless specified.

| Criterion | Verdict | Implementation and demonstration | Qualification |
| --- | --- | --- | --- |
| Document architecture and operational definition | Satisfied | Architecture and control flow above describes the external environment, root, children, aggregation and stopping. | Bounded RLM-inspired strategy. |
| Read-only access to all 75 filings | Satisfied | `environment.py:71`; environment and canonical hash/provenance regressions. | The experiment selected 50 sections. |
| Recursively analyze selected subsets | Satisfied | `recursive.py:88`; scripted grandchild aggregation in `test_rlm_recursive.py:365`, 26 live children. | Live profile depth one. |
| Preserve original SEC provenance | Satisfied | `findings.py:108`, `checkpoint_validation.py:57`; reloaded all five research/review snapshots. | Source identity and quotation checks do not measure entailment. |
| Aggregate companies and fiscal years | Satisfied | `execute_corpus_research`, `verification_report`; live multi-company cross-year records. | Only 11 companies yielded both years. |
| Inspect corpus coverage | Satisfied | `ResearchCheckpoint.merged_coverage`; 50-target ledger with discovery, access and finding references. | Lexical and semantic coverage differ. |
| Trigger additional investigation on uncertainty/conflict | Satisfied | `verification.py:487` derives scoped gaps, `checkpoint.py:232` selects follow-up, bounded reinvestigation tests and earlier live recoveries. | Final reviews selected saved evidence. No automatic commitment to close every gap. |
| Track depth, tokens, calls and latency | Satisfied | Invocation tree, `CorpusUsage`, `VerificationUsage`; five run intervals and zero missing usage records. | No complete directional billing totals. |
| Bound execution with configurable limits | Satisfied | `budget.py:78`, `CorpusLimits`, verification caps; live batch and finalization stops and mocked budget regressions. | Cooperative time checks and transport timeouts. |
| Trace final claims to original passages | Satisfied | The offline publication replay resolved all 31 printed IDs, including relationship citations, to canonical evidence. Structured pointers preserve exact spans. | Traceability applies to the evidence cited by the renderer. Semantic judgment quality is a separate evaluation question. |
| Qualify incomplete coverage | Satisfied | `checkpoint.py:146`, `verification.py:569`; saved final answer explicitly leaves novelty unresolved. | Completion applies to assignments or assessments only. |
| Run automated tests without paid calls | Satisfied | Scripted clients, mock Converse transports and native regression tests, including continuation and verification. | No independent human validation. |
| Document reproducible Sonnet 4.5 demonstration | Satisfied | Demo, profile, historical invocation chain, limit files/values and offline evaluation above. | Access and local artifacts are prerequisites; live judgments can vary. |
| Preserve ingestion, retrieval and RAG regressions | Partially satisfied | Canonical regeneration, BM25, RAG, provenance and selected dense/hybrid tests passed. | The full dense/hybrid sweep was not certified in this audit. |

The earlier closeout recorded 399 focused RLM, 53 retrieval/RAG/canonical and 40 ingestion/benchmark/routing passes. The final offline audit ran **572 distinct passing cases**, with 30 dense/hybrid cases deselected. No implementation changes or live requests were made for this audit.

## Milestone 2 — evidence-bound generic synthesis

A generic recursive root can use its existing reference finish to answer a question about several disclosures. For example, the offline Apple–Amazon fixture distinguishes conditional exchange-rate exposure in component costs from reliance on a limited group of semiconductor suppliers. The finish contains this optional array inside `synthesis`:

```json
{
  "conclusions": [{
    "statement": "Apple describes exchange-rate exposure in component costs, while Amazon describes reliance on a limited group of chip suppliers.",
    "disposition": "answered",
    "qualification": "The passages describe exposures rather than realized losses.",
    "bindings": [
      {"ref": "d2", "ticker": "AMZN", "filing_year": 2024, "section_id": "item_1a"},
      {"ref": "d1", "ticker": "AAPL", "filing_year": 2024, "section_id": "item_1a"}
    ]
  }]
}
```

A finish accepts up to three conclusions. Each binds two to eight operands covering at least two distinct acknowledged findings. An `sN` alias requires an acknowledged submission receipt. A `dN` alias requires delivery from a child or a scoped prior-finding read. A binding selects all original evidence for its finding. `filing_year` identifies the source filing. Optional `period_text` anchors a reporting period literally in a quoted excerpt, including a historical period discussed in a later filing. Preserve units, conditions and conflicts in the statement and qualification. Descriptive answers can continue using `cite_refs` alone.

Structured callers can set up to eight `ResearchIntent.required_operands` using the binding source fields. Each conclusion must cover them all. A generic review plan can span several filing years and use ordinary discovery. The existing specialized year-over-year discovery path retains its own comparison schema. The host checks source identity, authorized partitions, delivery, required bindings and existing citation limits. A failed proposal is withheld within the same finish. Valid sibling conclusions, separately selected findings and the retained evidence ledger survive.

`published_conclusions` attributes each record to its finished root. Bindings contain stable finding IDs linked to the original quotes and spans. `answer_citations` follows first mention in the answer, beginning with conclusion citations. `answer_disposition` is an **unverified** assessment of the requested answer. A single root may report `answered` within the full authorized scope when its conclusion binds the required operands. Unread chunks in an authorized section keep research incomplete, but they do not by themselves change that disposition. Conditional or conflicting findings can reduce the disposition to `partial`. A rejected conclusion keeps its selections and records a sanitized reason such as `wrong_source` or `unacknowledged_finding_reference`. Missing required bindings can yield `insufficient_evidence` alongside eligible atomic findings. A descriptive answer without an assessment retains a null disposition. Separate roots publish their own conclusions in ledger order. Relationships requiring evidence split across roots remain unresolved.

Checkpoint v4 stores structured requirements and conclusion state in the plan and invocation ledger, including the local receipt IDs needed to check bindings. Loading rechecks root ownership, selection, acknowledgement, canonical evidence and operands. Continuations retain earlier records and may bind historical findings after scoped delivery. Versions 1–3 retain their original meaning and receive no inferred selections or conclusions. Finding-level reviews leave generic conclusions semantically unverified. Entailment, numerical accuracy, sufficiency and filing-wide absence remain outside the structural checks.

From the repository root, run `PYTHONPATH=Code/src python -m pytest -q Code/tests/test_rlm_synthesis.py` for the scripted cases, or use `Code/tests/test_rlm*.py` for the full RLM regression. The tests exercise companies, source years, sections, reporting-period anchors, conflicts, rejected bindings, split batches, citation order, checkpoint tampering and continuation. These offline cases measure contract behavior. Live model accuracy remains open. Natural-language operand extraction is specified in the next milestone and is still scripted, not live.

## Milestone 3 — generalized natural-language interpretation

### Problem and baseline

Milestone 2 can bind a generic conclusion to canonical ticker, filing year, and section operands, but the natural-language interpreter only admitted a descriptive review or a two-fiscal-year AI-infrastructure comparison. An ordinary cross-company, cross-year, or cross-section filing question was rejected or would have been forced onto the specialized comparison executor. At `08324d70` that executor still owns novelty judgments, literal two-year scans, and the historical ranking path.

### Hypothesis

A separate interpretation mode can preserve a comparison through validation and correction while the stored research objective remains `review`. The existing generic discovery, research, synthesis, and publication path can then answer filing-grounded comparisons without a new executor.

### Solution and alternatives

`generic_synthesis` is an interpretation mode, not an execution mode. Its `ResearchIntent.objective` stays `review`, so discovery remains the ordinary phrase scan and the root stays a generic root. The specialized `ai_infrastructure_comparison` mode still requires objective `year_over_year`, exactly two fiscal years, and comprehensive review. Adding a third research objective was rejected because `year_over_year` already selects comparative execution whenever discovery is present.

The interpreter may now use `section_selection: inferred` for inventory section IDs when the question names none. The host records `section_ids` as inferred and appends the inventory explanation. Unnamed risk-factor questions still use the Item 1A default. Named sections, including unavailable ones, stay explicit. Inferred IDs that are not in the inventory are rejected rather than replaced.

For cross-source questions the interpreter fills `required_operands` with two to eight `SourceOperand` bindings. The host canonicalizes an unambiguous company alias to its ticker, checks each operand against the requested companies, filing years, and sections, and keeps unavailable operands on the intent. It does not invent `period_text`. A bare year or any anchor that is not a literal substring of the question is rejected. One section that already discloses several periods stays a review of that filing year, with no required operands and no second finding. Two named sections keep an operand each, even in that same filing. More than eight companies, filing years, explicit sections, or operands — including a corpus-wide comparison on this 25-company snapshot — returns `unsupported_operand_capacity` instead of a shorter operand list.

A correction cannot change `generic_synthesis` into a review, drop a preserved company, year, section, or operand, or replace an inferred section. Ranking, forecasting, external data, and arbitrary calculation stay unsupported when the interpreter says so. The host does not rewrite those judgments into a descriptive review.

Root guidance and the existing generic-root prompt both receive the original question and the required operands. Generic discovery does not add AI-infrastructure context anchors unless the topic itself contains both signals. That boost and the compact comparative prompt remain on the specialized path.

### Supported question contract

| Question | Mode | Execution |
| --- | --- | --- |
| Descriptive disclosure, including a corpus-wide description | `review` | Generic research. No required operands. |
| Risk factors with no named section | `review` or `generic_synthesis` | Item 1A is supplied and marked inferred. |
| Named section, available or not | either ready mode | Explicit IDs are preserved. Unavailable targets stay missing. |
| Section implied by the inventory but not named | either ready mode | `inferred`, limited to inventory IDs, with an explanation. |
| Cross-company, cross-year, or cross-section comparison within eight operands | `generic_synthesis` | Generic research. Objective remains `review`. |
| Explicit question over two or more sections, including one filing | `generic_synthesis` | One operand per requested section. Sufficiency does not require reading every chunk. |
| Comparative figures stated inside one section | `review` | That filing year only. One atomic finding can answer. |
| Two-year AI-infrastructure novelty | `ai_infrastructure_comparison` | Existing comparative executor. |
| Ranking, forecast, external data, arbitrary calculation, or more than eight required sources | `unsupported` | No plan. |
| Missing company, year, or section identity | `ambiguous` | No plan. |

### Interpretation assumptions

Named comparison years are filing fiscal years. A year discussed inside a filing is not a second filing year. Relative periods are ambiguous. The latest snapshot year is still the default only for a review that names no year. Company aliases map to one inventory ticker or stay verbatim. Corpus-wide scope is still `company_selection: all` and does not silently drop a named company. Comprehensive review is required for a generic synthesis and for the AI comparison, and it does not add companies, years, or sections. An operand is required only for a source the conclusion must bind, not for every partition in the scan. `period_text` is empty until a quoted excerpt can contain a literal anchor, and the interpreter may supply one only when that non-year phrase already appears in the question. Conclusions remain `semantic_status: unverified`. A missing operand or a lexical miss does not establish filing-wide absence.

### Verification

Verification used the local full-v1 snapshot (25 companies, fiscal years through 2024, sections `item_1`, `item_1a`, `item_1c`, `item_7`, `item_8`, and `item_15`). The interpreter was a mocked Converse client. Research was a scripted recursive client. No paid provider call and no live model trial were run. Development questions `q045`, `q029`, `q042`, and `q024` are not in this checkout, so they were not executed and no gold evidence was loaded. Each scripted case is one trial. Live token, latency, and accuracy metrics are unavailable.

The scripted cases cover an inferred business-section review, a non-AI FY2023–FY2024 foreign-exchange comparison, a three-year business comparison, an Apple–Amazon comparison, a business-versus-MD&A comparison, comparative periods disclosed inside one FY2024 filing, unavailable operands, ambiguous and unsupported objectives, operand-capacity refusal, a correction that drops a year or operand or the synthesis mode, generic discovery without AI anchors, the legacy AI comparison, a token-budget stop, and an unbound conclusion. The Apple–Amazon result reloaded from checkpoint v4 with the same operands, unverified conclusion, and absence qualification.

From the repository root:

```bash
PYTHONPATH=Code/src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  Code/tests/test_rlm_natural_language.py \
  Code/tests/test_rlm_planning.py \
  Code/tests/test_rlm_discovery.py \
  Code/tests/test_rlm_synthesis.py \
  Code/tests/test_rlm_corpus_research.py \
  Code/tests/test_rlm_checkpoint.py
```

That focused set passed **177** tests. `test_rlm_comparison.py` and `test_rlm_recursive.py` passed **152** more. `rlm-publication-replay.py` and `rlm-stabilization-evaluate.py rlm-final-review-pass3-checkpoint.json` reproduced the saved historical publication: 44 findings, 33 comparisons, 31 answer chunks, 21 conclusion chunks, 10 relationships, one outstanding comparison, incomplete research, 25 unresolved novelty conclusions, and unchanged judgments and usage. `git diff --check` reported no whitespace errors.

### Before and after

Before, a comparison of Apple's FY2023 and FY2024 foreign-exchange risk was unsupported unless it was restated as the AI-infrastructure task. After, a scripted interpretation plans a generic review of AAPL Item 1A for 2023 and 2024, requires those two operands, and publishes one unverified conclusion. A three-year plan keeps three operands. A nine-company comparison is refused with the eight-operand limit instead of keeping eight companies. The historical AI checkpoint is unchanged.

### Trade-offs and limitations

The host trusts the interpreter’s mode. It will not convert an explicit unsupported or ambiguous judgment into a review, and it will not parse the question itself to decide that one filing already contains the comparison. Live prompt compliance is unmeasured. Generic conclusions are structurally bound and semantically unverified. A bound conclusion can be `answered` while research stays incomplete because the section was not read exhaustively. Relationships split across roots, full-filing requests, and comparisons above eight operands remain out of scope. Ranking, forecasting, external data, and arbitrary calculation are still refused. Live validation is blocked until a provider run is separately authorized; this milestone did not start one.

## Live q045 stabilization

The first live Cisco fiscal-2024 Splunk question used 6 provider requests, 28,610 reported tokens, and 25.11 seconds. It found the relevant passages, spent the child’s six tool calls on weakly ranked mentions, reported product revenue for a total-revenue question, omitted the two section operands, and withheld the root conclusion with only `withheld_conclusion`.

Generic discovery still admits candidates by lexical phrase, then orders each section with the existing BM25 formula over the matching passages. The query is the question clause for that section, so a passage that covers the requested terms outranks an incidental mention of a longer phrase. A component, segment, or balance the question did not name ranks after the same terms without that extra measure. The specialized AI-infrastructure rank is unchanged. Investigator prompts tell the model to read an unresolved part before extra hits and to keep the requested metric, unit, period, and qualifiers. Explicit multi-section questions, including two sections of one filing, keep one operand per section. A single section that already states several periods stays a review. Answer disposition follows the bound conclusion; unread chunks remain an incomplete-research note. Rejected conclusions keep provenance checks and publish the path and code, not the model text. Generic child deliveries send the alias, source identity, claim, and quote instead of repeating filing metadata.

Offline checks are `test_rlm_discovery.py`, `test_rlm_synthesis.py`, and `test_rlm_natural_language.py`. No provider call was made. The saved q045 artifacts were not rerun. A local tool cap can still end investigation while shared budget remains, BM25 cannot recognize a role description that does not use the question’s words, and conclusions stay semantically unverified. The next live trial can reuse the q045 limits. The useful change is the code above, not a larger tool budget.

The second live run answered both parts and then withheld the conclusion as `synthesis.conclusions[0] string_too_long`. Schema diagnostics now name the field, such as `statement` or `qualification`. A conclusion statement may be 1200 characters and a qualification 600, and those limits are in the schema and the finish instruction. One schema rejection can use the remaining reserved finalization attempt; provenance failures still withhold the conclusion and keep the atomic findings. Reading text from an assigned document and section does not close retrieval. Once submitted findings cover every assigned document-section pair, the investigator may finish or keep reading within budget.

The q029 AMD gaming-revenue question failed both interpretation attempts at `requirements.inferred_fields[0]` with `literal_error`. Section scope is only `section_selection`: `explicit`, `risk_factors_default`, or `inferred`. `inferred_fields` accepts only `company_selection`, `fiscal_years`, `objective`, `topic`, and `comprehensive`. A redundant section token in that list is dropped before validation, so an otherwise valid plan is kept. Companies, years, sections, objectives, and required operands stay strictly checked. A remaining literal error names the permitted values in the correction feedback.

## Final stabilization and integration

The following results come from the saved Claude Sonnet 4.5 runs on the full-v1 corpus. Each row is one live trial with investigator assessments and no independent semantic review.

| Saved report | Attempts / reported tokens | Observed result |
| --- | ---: | --- |
| `rlm-generalization-q045-v2-live.json` (`19d7475`) | 8 / 41,752 | Cisco Item 1 findings described how Splunk fits into security and observability. Item 7 reported approximately $1.4 billion in total FY2024 revenue from Splunk. The root exceeded the conclusion length cap and the report retained the four findings. |
| `rlm-generalization-q029-v2-live.json` (`f6cd934`) | 8 / 46,511 | AMD's FY2023 and FY2024 MD&A supported a drop from $6.2 billion to $2.6 billion, about 58%. The FY2023 report explained its own 9% decline and the FY2024 report explained its 58% decline. Both pointed to lower semi-custom product sales or revenue. A search result overflowed the tool batch, and an invalid finding reference required recovery. |
| `rlm-generalization-q042-live.json` (`f6cd934`) | 13 / 76,506 | Both children searched Item 1. The root received NVIDIA's fabless strategy and Intel's selective use of outside foundries. The NVIDIA finding that named TSMC and Samsung stayed in the child ledger. The root's qualification about Intel's four years of investment lacked that period in its bound excerpts. |
| `rlm-generalization-q024-live.json` (`f6cd934`) | 8 / 34,551 | The cited table recorded iPhone revenue. The investigator inferred filing-wide nondisclosure and a change in reporting practice in FY2019 from that evidence. The terminal conclusion also failed the two-binding minimum. |

For generic research, bounded child delivery now ranks whole findings by question terms and discovery phrases, with early representation for separate required sources. Inferred scope can include several plausible sections. A required filing can be bound through its authorized candidate sections. If the highest-priority sections fit the partition allowance and still cover each resolved filing, lower-priority inferred alternatives can be left out and recorded in the scope notes. Explicitly named sections remain required.

A single binding can support a direct answer. An abstention can identify a gap in the reviewed evidence with optional citations. An answer that makes a cross-source assertion still requires distinct acknowledged findings for all required operands and any literal reporting-period anchors. The host rejects common ungrounded disclosure-absence and historical assertions in generic findings and conclusions. Investigators must tie a historical explanation to the change and source that states it.

Generic search pages fit the observation allowance and return a continuation offset. Repeated acknowledged pages and spans use compact receipts. A page or span that has yet to be delivered remains eligible for retrieval. The host keeps the same investigation ceilings and leaves unresolved parts open while tool slots remain. The specialized corpus-wide comparative path retains its capture, continuation, and verification workflow.

The closeout used scripted regression checks and separately reloaded the four saved checkpoints against the canonical corpus. The older live reports remain historical observations. Lexical relevance can miss paraphrases, inferred breadth depends on interpretation and available partitions, and evidence spread across separate roots can remain unresolved. The assertion checks catch common wording and scope errors. They cannot determine entailment for every paraphrase or establish filing-wide absence. Live answer quality, token savings, and latency after these changes remain unmeasured.

## Ongoing engineering record

For each future milestone, add a short entry to this guide with the **problem and baseline**, **hypothesis**, **solution and alternatives**, **verification method**, **before-and-after results**, and **trade-offs and limitations**. Include a reproduction command and, where applicable, the model and provider, corpus scope, execution limits, commit reference, and whether verification was scripted, offline, or live. Identify which variables changed, how many trials ran, and which metrics were unavailable. Link to the relevant design decision when it already explains the architecture.
