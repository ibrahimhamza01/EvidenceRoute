# Recursive language model research

## Architecture and control flow

1. `RootController` sends the question, tool schemas, previous observation, and remaining time to a model client. The scripted client plays back predetermined decisions for offline tests.
2. A provider reply is one model decision. It can contain one tool call, an ordered batch of up to four tool calls, or an exclusive finish. The next request replays the original assistant message and sends one user message containing a result for each native tool-use ID. [Bedrock Converse tool use](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-client-side.html) · [Anthropic tool result ordering](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use)
3. `CorpusToolDispatcher` validates the name and arguments against the same models that publish the schemas. It calls only allowlisted local operations and returns a bounded JSON success or error. Invalid arguments become feedback the model can correct.
4. The controller accepts a structured finish only when its cited chunk IDs were delivered to the model. It records requested decisions, validated arguments, observations, original evidence IDs, provider attempts, timing, and stop reason.

`CorpusEnvironment` provides validated, read-only access to the canonical SEC corpus of **25 companies, 75 filings, 416 sections, and 24,766 chunks**. It loads `data/filings_manifest.csv` and the section and chunk Parquet files under `data/processed/full-v1/`. The tools list companies, filings, and sections, fetch filing metadata or original chunks, page through section text, and search with BM25. The corpus stays local. The provider receives the question and the observations accumulated during the run.

The caller can supply required filing and section IDs through `ResearchScope`. An unspecified requirement remains unknown.

### Research scope planning

The interpreter identifies companies, fiscal years, section IDs, the topic and objective, and whether the question requires comprehensive review. It marks inferred requirements and explains its assumptions or uncertainty. Invalid or incomplete structured output raises an interpretation error. The offline interpreter is injected, and the host preserves the original question. A future provider-backed interpreter must charge attempts and reported tokens to the same budget and deadline as execution.

Python pages through the filing and section metadata without search. Company selectors match a ticker first, then an exact name recorded in the manifest, ignoring case and extra whitespace. Matched aliases collapse to their canonical ticker. Targets are unique and sorted by ticker, fiscal year, and section ID. Unknown companies, ambiguous matches, and unavailable filings or sections remain visible as unresolved company/year/section slots in the required count.

Each company group keeps its resolved fiscal years and the original question together. The [live semantic review](#reading-the-report) shows why that matters. Independently choosing excerpts for each year can make different topics look like a change in disclosure. A shared group lets the analyst check the same risk in each filing. Even matched excerpts need adequate coverage before making an absence claim.

The plan stays host-side. The admission check tests whether one child per nonempty group fits the shared depth, delegation, and pair-assignment limits. It leaves the plan unchanged. The runner can already take the pairs for a selected group or bounded slice through `partitions=`. A future scheduler can keep the full plan as the denominator and expose counts and paged group IDs to the root, resolving each ID to authorized pairs before delegation. Provider and local limits still apply at execution. Inventory resolution does not measure evidence coverage. Research-wide roll-up remains pending because pagination offsets from different scoped views can refer to different underlying items and cannot be merged directly.

### Bounded recursive execution

`RecursiveResearch` opts into recursive execution. Its factory creates a fresh budget-aware model client per invocation, and `RootController` handles each conversation. The model supplies a question, a goal, and a nonempty set of unique canonical document/section pairs to `delegate_research`. The root can assign all its authorized pairs to a fresh child to investigate a narrower question over the same evidence. With no explicit authorization, it can select any valid canonical pairs. Below the root, every successful delegation must remove at least one pair. The scope therefore shrinks along each branch, and a singleton cannot be subdivided under the current pair representation.

Each child starts with its own question, goal, authorized pairs, and an empty model history. A Python corpus view constrains metadata, reads, lookups, and search. Filtering exact pairs **before counting and slicing** makes totals, offsets, and coverage denominators refer to the authorized collection. For example, if ranked hits are `[outside, A, B]`, a one-hit first page returns `A` with total `2` and next offset `1`. Filtering after slicing could leave an empty page or retain the wrong total. BM25 scoring statistics still come from the corpus-wide index.

A bounded delegation observation carries the child's stop reason, analysis status, partition coverage, and validated findings. Failed, oversized, or unacknowledged observations grant no citations. The trace separates findings generated, acknowledged, and declared used. A declared finding must have been delivered and must contribute a chunk to the answer or a new finding.

| Signal | What it establishes |
| --- | --- |
| Evidence access and coverage | Direct chunk access supplies the whole original chunk. A delegated finding supplies only its delivered excerpt. A section window advances text coverage but supplies no chunk IDs. The tracker computes per-invocation pagination from acknowledged ranges. A complete section read or chunk traversal can mark a partition complete, while a search hit or delegated excerpt alone shows partial access. Child pagination stays in the child trace. Exhaustive research-wide coverage aggregation across descendants is pending. |
| Analysis status | The model declares its assessment, uncertainty, and conflicts. The controller defaults the invocation to `incomplete` when no status or supporting references are supplied. A validated `finished` action can still have partial coverage or incomplete analysis. A search miss cannot establish that a disclosure was absent. |
| Citation and quotation validation | Citation IDs must refer to evidence the invocation received. Citations based solely on delegated evidence require the delivered finding ID. For structured findings, the host resolves canonical source metadata and checks exact quote membership in the original chunk. A finding based solely on a delegated excerpt must also stay within that excerpt. |
| Semantic correctness | Independent claim verification is pending. A valid citation and exact quote establish the source and wording, while the claim's interpretation still needs review. |

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
| Model or provider failure | Retryable provider errors can retry within configured attempt and time limits. Truncated investigation replies can transition to protected finalization when a finish attempt remains. Other unhandled failures end the invocation, normally as `model_error`. A failed child returns an error outcome to the parent, which can continue within its remaining budget. |

Batch operations execute sequentially in provider order. One batch spends one model decision, and each admitted operation spends its own local tool slot and applicable corpus or delegation budget. The controller checks those limits before each operation. An invalid operation can return an error while its peers continue. A request skipped at a local or shared limit gets its own error result when a follow-up is allowed. The local tool limit can skip the rest of a batch and still allow a later finish. Running out of provider attempts blocks further batch work that would require another provider turn.

The observation byte limit applies to each result and the combined batch envelope. The controller reserves space for all IDs and compact errors before execution. If the envelope cannot fit, it stops without executing the batch. Otherwise earlier results get first use of the remaining bytes. An oversized result is replaced by an error and leaves the original evidence and pagination intact. Coverage and citation eligibility require both a successful result and acknowledged delivery of that result. An acknowledgement of the batch's error block grants no evidence.

#### Offline example

Run the scripted example with local canonical files. It delegates FY2023 and FY2024 evidence collection, and the FY2024 child asks a grandchild to inspect Item 1A. It prints the tree, findings, original citations, coverage, and shared counters. The scripts illustrate the control flow. Semantic comparison and provider inference are outside this example.

```bash
PYTHONPATH=Code/src python Code/examples/demo_rlm_recursive.py
```

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

## Design Decisions & Trade-offs

| Decision | Choice and reason | Alternatives and limitations |
| --- | --- | --- |
| RLM-01 · Local validated snapshot | Reuse filing, section, chunk, and corpus schemas. Validate provenance and text offsets and return defensive copies of mutable records. | Streaming Parquet would reduce peak memory. The current snapshot pays the loading cost up front for straightforward lookups. |
| RLM-02 · Exact pages and lexical search | Character windows and whole chunks preserve source text. Existing BM25 lookup and chunk-ID tie breaking give repeatable search. | Whole sections can be too large for a tool turn. Dense search could find different wording at the cost of another model and index. Keyword search can miss relevant evidence. The initial BM25 build is lazy. |
| RLM-03 · Allowlisted tool schemas | Generate schemas and validate requests using the same argument models. The dispatcher owns execution. | Separate schemas and validators can drift. An unrestricted function interface would broaden model-controlled execution. Adding a tool requires an explicit registry change. |
| RLM-04 · Bounded decisions and observations | Reuse the root-client interface. Reject oversized results as a whole so evidence and offsets remain consistent. | Parsing prose would need extra conventions. Truncating JSON could remove provenance. Oversized requests may need a smaller page or section window. |
| RLM-05 · Separate scope, access, and analysis | Merge acknowledged ranges per invocation and preserve child pagination in its own trace. | Rolling up summaries as coverage would overstate how much the parent inspected. Unspecified root requirements stay unknown. |
| RLM-06 · Preserve native messages and execute batches sequentially | Bedrock can return multiple tool-use blocks in one reply. Preserve the complete assistant message and pair each result with its original ID in one user message. Validate IDs before dispatch and reject mixed batches containing `finish_research`. The direct Messages transport requests single-tool turns and can handle a batch safely. | Prompting alone did not enforce single-call replies in the first live run. Sequential execution keeps budget checks and delivery accounting ordered. Batches over four calls, malformed or duplicate IDs, and partial replies never reach dispatch. |
| RLM-07 · AWS SDK, bounded attempts, and opt-in | Use the existing Boto3 dependency and standard credential chain. Disable SDK retries for Converse so the shared loop owns retry accounting and time checks. Reuse the session and apply the remaining time to new request clients. | Handwritten signing and credential rotation would duplicate SDK behavior. A new client per attempt adds setup cost. Socket timeouts and credential resolution are synchronous and cannot guarantee a hard deadline. Late decisions are discarded. |
| RLM-08 · Per-attempt usage and acknowledged delivery | Record input/output and cache counters when supplied, wall latency, Bedrock latency when supplied, and sanitized errors. Credit a provider observation only after an assistant response acknowledges the turn. | An iteration count hides retries. A lost response leaves delivery and usage unknown. Neither adapter requests prompt caching or estimates cost. |
| RLM-09 · Small chunk pages by default | The model-facing `search_chunks` and `read_chunks` tools default to one whole chunk and still accept an explicit limit up to ten. Tool descriptions show the run's byte cap. Oversized responses report the attempted size and cap and suggest a smaller page or text window when possible. | A larger default can return more evidence per successful call but exceeded the demo budget for the NVIDIA search. Starting with one chunk can require more turns for broad questions. Silently dropping results would obscure pagination. Increasing the byte cap also increases the amount of evidence replayed in later model requests. |
| RLM-10 · Sequential delegation with one budget | Scope shrinks at every delegation below the root. A shared budget also caps work across sibling branches. | Per-child budgets would multiply resources. Parallel delegation would need synchronized admission and trace handling. Pairs are the current indivisible scope units. |
| RLM-11 · Carry original excerpts with findings | Validate source quotes against accessible chunks, track generated and delivered findings separately, and require declared finding use for delegated citations. Preserve child coverage, failures and conflict notes. | Sending whole child histories would expose unrelated context and use more tokens. Excerpts limit what parents can quote without fetching more evidence. Declared use is a model report and does not verify the claim. |
| RLM-12 · Reserve provider slots for finalization | Hold provider capacity for each configured child ahead of admission, and protect finish-only slots for each invocation. Charge every transport attempt, including retries, to its active phase and the shared cap. | Prompting alone leaves the last slot spendable on retrieval. Fixed shares isolate siblings but may leave capacity unused for children that never launch. Child slots left over at exit return to root investigation. |
| RLM-13 · Constrain and size protected finish requests | Use a finish-specific prompt and output cap. Allow native named tool choice where supported and keep runtime validation in both modes. After a truncated investigation reply, reuse the last valid history for protected finalization. | The course Sonnet 4.5 profile accepted named choice on a protected finish that used 332 output tokens. The need for the higher cap remains untested. Truncation recovery is offline-tested and ends investigation early to preserve capacity for synthesis. |
| RLM-14 · Keep required scope independent of execution | Enumerate required slots from metadata before applying limits. Group years by company and retain unresolved slots. | Top-K retrieval could skip entire companies. Trimming to the budget would shrink the coverage denominator. One run may need to take a bounded slice, with its results still measured against the original plan. |

See the [Converse tool contract](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-client-side.html) and [Boto3 request configuration](https://botocore.amazonaws.com/v1/documentation/api/latest/reference/config.html) for wire and timeout details.

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

## Engineering achievements and recursive implications

| Engineering challenge | Implemented approach and verification | Trade-off or remaining question |
| --- | --- | --- |
| Put the SEC corpus outside the model context while keeping evidence auditable | Read-only manifest and Parquet access preserves document, section, and chunk IDs, offsets, text, and original SEC URLs. The canonical integrity regression checks corpus inventory, offsets, and provenance. | Loading a local snapshot costs memory up front. Lexical search supports repeatable selection and can miss disclosures phrased differently. |
| Separate research policy from provider behavior | The controller owns decisions and coverage, the dispatcher owns validation and tool execution, and the environment owns corpus access. Adapters translate tool messages and usage, while transports handle provider calls. Shared controller limits are provider-independent, and provider configuration comes from the environment. Scripted clients and stubbed transports verify both boundaries offline. | Each adapter still needs provider-specific tool-message handling and an explicit policy for failed or unacknowledged turns. |
| Diagnose AWS access and bring up a real provider | Successful authentication narrowed the Sonnet 4.6 failure to authorization for that model. Updating the environment to the authorized Sonnet 4.5 inference profile in `us-east-1` used the existing Bedrock path. The successful live trace confirmed tool calling, observation delivery, and a cited finish. | The verified profile is account- and region-dependent. Credential retrieval and refresh stay with Boto3 and the course issuer. |
| Keep coverage and findings grounded across sessions | Separate discovered, delivered, and declared-used evidence. Verify original quotations and retain incomplete coverage, conflict notes, and failures. Regression tests reject citations from oversized, failed, or undelivered child results. | Quote validation confirms that the words appear in evidence the model received. It does not establish that the quotation entails the claim. |

The failed demo exposed a cost of accumulated context, and the subsequent live trace showed the overhead of repeating inputs. This informed three aspects of the recursive design:

- **Context isolation and scoped access.** Each child starts with its task, goal, and authorized document/section pairs. It does not inherit the parent's transcript. Python checks every corpus operation, so the scope holds even if the model requests an out-of-scope lookup. Exact pairs make the search universe and coverage denominators reproducible. They also limit which evidence can support a finding. Fresh sessions avoid inherited history while still sending system instructions and schemas, so more sessions can increase aggregate token use.
- **Shared budget.** A fresh session may produce several decisions. One decision can cause multiple provider attempts if the transport retries, and a decision can also execute a corpus tool or request a delegation. Admitted child invocations and assigned document/section pairs are counted separately. A global attempt ledger prevents per-child budgets from multiplying across the tree. Request counts can be checked before sending. Exact token usage is reported afterward, and synchronous deadlines rely on timeouts and late-result handling.
- **Compact findings.** Children return validated excerpts, citation IDs, and coverage status rather than complete transcripts. Only acknowledged results qualify as evidence in the parent conversation. The parent still has to decide how to synthesize the findings, handle disagreements, and disclose gaps.

These are engineering choices motivated by observed behavior. The scripted run has not established that recursion improves research quality or reduces token use against a non-recursive baseline.

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

Corpus-wide execution scheduling, exhaustive coverage roll-up, durable finding storage, systematic conflict resolution, semantic claim verification, and corpus-wide evaluation remain pending.

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

A `max_tokens` investigation reply supplies usage but no executable action. If a protected slot and the other limits permit, the controller makes one transition to finish-only. It resends the last valid history with the original pending tool results and IDs, including batches. It never appends the partial assistant reply or duplicates result messages. Previously unacknowledged evidence becomes eligible only after the provider acknowledges delivery. The recovery attempt spends the existing finish allowance. A truncated finish ends the invocation, and a transport retry still spends an attempt from the current phase. The shared 60,000-token admission threshold and cooperative deadline still apply. This recovery is verified offline only.

### First successful end-to-end live run

Run `1b803beef0f948d9943f23e582c57cd7` ended with status `success` and stop reason `finished`. All six structural criteria passed. The root requested both fiscal-year delegations in one batch. The run recorded 3 sessions, 9 decisions, 9 provider responses, zero retries, 2 partition assignments, and a maximum depth of 1. The nine attempts comprised eight investigation attempts and one protected finish. Provider-reported latency summed to 33.601 seconds. Cache reads and writes reported zero tokens, and there were no provider or validation errors.

| Invocation | Provider / corpus attempts | Reported tokens | Evidence and finish |
| --- | ---: | ---: | --- |
| FY2023 child | 4 / 3 | 16,171 | Received two unique chunks, with one repeated direct lookup. Exhausted its investigation allowance, then used `finish_research` with required tool choice and 332 output tokens. Delivered `child-1:finding-1`, citing `a19fccaa6f0cd45a5fb8c0a8`. |
| FY2024 child | 3 / 2 | 12,825 | Two searches delivered two unique chunks. Finished during investigation and released one unused provider slot. Delivered `child-2:finding-1`, citing `02834a6d106b2138d0425352`. |
| Root | 2 / 0 | 10,287 | Received and declared both findings used. Its final response used 976 output tokens to synthesize a comparison citing both original chunks. Both partitions retained partial coverage. |

This establishes live model-directed delegation, protected finalization, child-to-parent evidence transfer, and synthesis with Claude. The protected finish used fewer than 1,024 output tokens, so this run does not establish a need for the increased finish cap. Depth-two delegation and truncation recovery remain offline-tested.

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

Exit `0` means the recursive experiment met its structural criteria. Exit `1` means the run produced a report and was incomplete. Invalid setup or arguments exit `2`. The report checks provider responses, separate singleton delegations for both years, validated child findings, acknowledged delivery, a finished synthesis declaring which findings it used, and citations from those used findings covering both years. A root that cites both filings directly still misses the delegation criterion. A validated excerpt can contribute even when the child reports incomplete analysis or partial coverage. The report preserves those statuses. An empty finding, failed child, or unacknowledged observation cannot satisfy that year's evidence requirement.

The JSON shows authorized pairs, the invocation tree, budget stops, errors, per-invocation counts, citations, and coverage. Requested tool counts include the native finish request. Operation records separate requests, execution outcomes, and delivery. The report also retains bounded, sanitized tool IDs, names, and arguments for rejected responses when the tool blocks can be decoded. Each excerpt records whether the parent declared its finding used. `--trace` adds per-decision and per-attempt telemetry. The report omits private reasoning, raw provider payloads, authentication headers, and whole observation bodies.

The report takes shared counts and measured token usage from the root's final budget snapshot. It separates investigation attempts, finish-only attempts, and unused child slots returned to the root. Per-invocation traces identify the change to finalization and its reason. Provider usage is aggregated once over the invocation tree, with missing-counter counts retained. Per-invocation wall times include their children, so adding them would double-count time. Summed provider latency and attempt wall time describe provider work separately from elapsed run time.

**Semantic review of the successful run.** The final answer inferred a shift toward geopolitical risks in FY2024 from two independently selected excerpts. An earlier FY2023 run retrieved `bd263969e91358073d0ea0a1`, which explicitly discusses geopolitical tensions involving Taiwan and China and their implications for supply continuity. The latest excerpts cannot establish that geopolitical risk was new or received greater emphasis in FY2024. The latest FY2023 quotation also does not independently support its finding's additional claim about demand estimation challenges. That child returned `finished` with a finding marked `analyzed`, while its invocation-level analysis status remained `incomplete`.

Citation provenance identifies the original chunk and quotation validation checks its wording. Coverage describes the evidence accessed, and analysis status is model-reported. In this run, both partitions remained partial and exhaustive descendant coverage roll-up is still pending. Independent semantic verification would have to test whether the evidence supports each claim and comparison.

### Offline verification of the live path

Stubbed Bedrock and Anthropic responses verify both-years delegation in one reply, provider result IDs, sequential execution, malformed and duplicate IDs, mixed finish rejection, mid-batch budget exhaustion, and the aggregate byte cap. The default scripted demo still runs without provider attempts. These checks do not estimate live latency, token savings, or answer quality.

The starvation regression reproduces 12 provider attempts, 13 model decisions, 8 corpus attempts, and no final citations with reservations disabled. Other stubbed tests inspect named finish choice and output caps in both provider payloads, preserve pending single and batched results across truncation, and verify usage, runtime rejection, evidence validation, and budget stops. A combined regression recovers the root after truncation and still reports the experiment incomplete when the FY2024 child's finish attempt requests a prohibited search. A direct FY2024 citation cannot satisfy the delegated-finding criterion.

| Check | Result |
| --- | --- |
| Focused RLM suite, including starvation, finalization, and truncation recovery | 138 passed, 1 warning in 2.51s |
| Related retrieval, RAG provenance, and canonical corpus regressions | 40 passed, 1 warning in 17.41s |
| Default scripted demo on the canonical corpus | Exit 0, two original chunk citations, zero provider attempts |

Both test runs reported the existing AnyIO pytest deprecation. Reproduction uses the same offline test commands from the recursive milestone above.

## Ongoing engineering record

For each future milestone, add a short entry to this guide with the **problem and baseline**, **hypothesis**, **solution and alternatives**, **verification method**, **before-and-after results**, and **trade-offs and limitations**. Include a reproduction command and, where applicable, the model and provider, corpus scope, execution limits, commit reference, and whether verification was scripted, offline, or live. Identify which variables changed, how many trials ran, and which metrics were unavailable. Link to the relevant design decision when it already explains the architecture.
