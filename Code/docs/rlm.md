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

The plan stays host-side. `execute_research_plan` validates provider reservations and admits whole groups in ticker order within the group, shared, and root limits. Unresolved groups and groups outside those limits retain a pending reason. Comprehensive plans default to delegation-first execution. Admission reserves enough root decisions for one assignment per group and enough shared delegation and partition slots for every admitted group. The full plan remains the reporting denominator. Provider, token, and time limits can still stop execution after admission.

### Bounded recursive execution

`RecursiveResearch` opts into recursive execution. Its factory creates a fresh budget-aware model client per invocation, and `RootController` handles each conversation. The model supplies a question, a goal, and a nonempty set of unique canonical document/section pairs to `delegate_research`. The root can assign all its authorized pairs to a fresh child to investigate a narrower question over the same evidence. With no explicit authorization, it can select any valid canonical pairs. Below the root, every successful delegation must remove at least one pair. The scope therefore shrinks along each branch, and a singleton cannot be subdivided under the current pair representation.

Each child starts with its own question, goal, authorized pairs, and an empty model history. A Python corpus view constrains metadata, reads, lookups, and search. Filtering exact pairs **before counting and slicing** makes totals, offsets, and coverage denominators refer to the authorized collection. For example, if ranked hits are `[outside, A, B]`, a one-hit first page returns `A` with total `2` and next offset `1`. Filtering after slicing could leave an empty page or retain the wrong total. BM25 scoring statistics still come from the corpus-wide index.

A bounded delegation observation carries the child's stop reason, analysis status, partition coverage, and validated findings. Failed, oversized, or unacknowledged observations grant no citations. The trace separates findings generated, acknowledged, and declared used. A declared finding must have been delivered and must contribute a chunk to the answer or a new finding.

| Signal | What it establishes |
| --- | --- |
| Evidence access and coverage | Direct chunk access supplies the whole original chunk. A delegated finding supplies only its delivered excerpt. A section window advances text coverage but supplies no chunk IDs. The per-invocation tracker still reports acknowledged pagination. Planned execution also rolls up original chunk IDs and section-window ranges across the tree against the full plan. |
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

Four operations is the execution limit. A well-formed native batch of five through sixteen is rejected in full and receives a `tool_batch_limit` result for every ID. The original assistant message stays in history for that follow-up. Sixteen is a protocol safety ceiling for sending complete rejection feedback. Batches above it and malformed or duplicate IDs stop the invocation. Rejection spends the current model decision and provider attempt but performs no corpus work or delegation. Any correction uses the remaining budget, including the existing finish-only rules once protected finalization begins.

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
| RLM-06 · Preserve native messages and execute batches sequentially | Bedrock can return multiple tool-use blocks in one reply. Preserve the complete assistant message and pair each result with its original ID in one user message. Validate IDs before dispatch and reject mixed batches containing `finish_research`. The direct Messages transport requests single-tool turns and can handle a batch safely. | Four calls are executable per decision. Oversized batches up to the protocol ceiling can be rejected with one error per ID. This costs a model turn and allows recovery without executing part of the batch or breaking the provider history. |
| RLM-07 · AWS SDK, bounded attempts, and opt-in | Use the existing Boto3 dependency and standard credential chain. Disable SDK retries for Converse so the shared loop owns retry accounting and time checks. Reuse the session and apply the remaining time to new request clients. | Handwritten signing and credential rotation would duplicate SDK behavior. A new client per attempt adds setup cost. Socket timeouts and credential resolution are synchronous and cannot guarantee a hard deadline. Late decisions are discarded. |
| RLM-08 · Per-attempt usage and acknowledged delivery | Record input/output and cache counters when supplied, wall latency, Bedrock latency when supplied, and sanitized errors. Credit a provider observation only after an assistant response acknowledges the turn. | An iteration count hides retries. A lost response leaves delivery and usage unknown. Neither adapter requests prompt caching or estimates cost. |
| RLM-09 · Small chunk pages by default | The model-facing `search_chunks` and `read_chunks` tools default to one whole chunk and still accept an explicit limit up to ten. Tool descriptions show the run's byte cap. Oversized responses report the attempted size and cap and suggest a smaller page or text window when possible. | A larger default can return more evidence per successful call but exceeded the demo budget for the NVIDIA search. Starting with one chunk can require more turns for broad questions. Silently dropping results would obscure pagination. Increasing the byte cap also increases the amount of evidence replayed in later model requests. |
| RLM-10 · Sequential delegation with one budget | Scope shrinks at every delegation below the root. A shared budget also caps work across sibling branches. | Per-child budgets would multiply resources. Parallel delegation would need synchronized admission and trace handling. Pairs are the current indivisible scope units. |
| RLM-11 · Carry original excerpts with findings | Validate source quotes against accessible chunks, track generated and delivered findings separately, and require declared finding use for delegated citations. Preserve child coverage, failures and conflict notes. | Sending whole child histories would expose unrelated context and use more tokens. Excerpts limit what parents can quote without fetching more evidence. Declared use is a model report and does not verify the claim. |
| RLM-12 · Reserve provider slots for finalization | Hold provider capacity for each configured child ahead of admission, and protect finish-only slots for each invocation. Charge every transport attempt, including retries, to its active phase and the shared cap. | Prompting alone leaves the last slot spendable on retrieval. Fixed shares isolate siblings but may leave capacity unused for children that never launch. Child slots left over at exit return to root investigation. |
| RLM-13 · Constrain and size protected finish requests | Use a finish-specific prompt and output cap. Allow native named tool choice where supported and keep runtime validation in both modes. After a truncated investigation reply, reuse the last valid history for protected finalization. | The course Sonnet 4.5 profile accepted named choice on a protected finish that used 332 output tokens. The need for the higher cap remains untested. Truncation recovery is offline-tested and ends investigation early to preserve capacity for synthesis. |
| RLM-14 · Keep required scope independent of execution | Enumerate required slots from metadata before applying limits. Group years by company and retain unresolved slots. | Top-K retrieval could skip entire companies. Trimming to the budget would shrink the coverage denominator. One run may need to take a bounded slice, with its results still measured against the original plan. |
| RLM-15 · Roll up evidence by canonical IDs and section offsets | Deduplicate acknowledged whole-chunk IDs across invocations. Merge acknowledged character ranges only within the same canonical document/section pair. | Document-wide pagination offsets can refer to different items in different scopes. Counting child excerpts as whole chunks would overstate root access. |

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

Durable finding storage, independent semantic verification, automated reinvestigation scheduling, live intent interpretation, and comprehensive corpus execution remain pending. This demo executes one bounded slice and does not launch follow-up runs.

### Systematic comparative research

The comparative demo scans the full resolved inventory in canonical company, year, section, and chunk order. It counts case-insensitive literal matches for configurable terms under separate local chunk and time limits. Children receive up to three candidate IDs per authorized section with span hints and a cursor. `list_candidates` returns up to ten IDs for a scoped section. Comparative `search_chunks` returns ranked candidate IDs and span hints. Each call uses one local tool slot and one shared corpus attempt. Candidate IDs grant no evidence or access credit. `get_spans` supplies up to four exact passages from one scoped chunk. `get_chunk` supplies the full text when context is needed.

Comparative chunk observations include compact span IDs alongside the original text. An ID is a pair of character offsets within one canonical chunk. The model selects the pair and Python copies the exact source characters. Selection is limited to acknowledged span IDs or the identical span from a declared, acknowledged child finding. This preserves Unicode and prevents a parent from expanding a delegated excerpt. The existing observation byte cap applies and the legacy exact-quote interface remains available.

Children submit structured relevance and entailment assessments over selected spans. The root can reassess the delivered supporting and counterevidence spans in its existing budgeted session. An assessed prior-year match can support a `comparable_both_years` result. Contradictory assessments can yield `conflicting`. Current-year support with inconclusive prior-year evidence can at most yield `possible_new_needs_review`, and a prior-year search miss alone leaves the result `insufficient`. This path issues no verified novelty or absence verdict. It reports model-assisted judgments without independent semantic verification. Protected finish attempts also reserve local decisions, so returned child provider slots cannot consume those decisions.

```bash
PYTHONPATH=Code/src python Code/examples/demo_rlm_comparative.py
```

With the default terms, the offline demo scanned **3,038 chunks in 50 Item 1A sections across 25 companies** using 328 local pages and 4,603,503 chunk characters, including overlap. It identified **87 candidate chunks and 118 matching spans**. FY2023/FY2024 candidate chunk counts were **1/1 for AAPL** and **4/6 for AMD**. The AMD script read the FY2024 water-and-power passage and then the canonical FY2023 counterpart. Two chunks were delivered and assessed, both original citations reached the root, and the scripted root classified the constraint as comparable across both years. The FY2024 excerpt explicitly adds a potential revenue impact. The selected passages concern data centers generally and leave the AI-specific link unestablished. This demonstration used 3 sessions, 7 decisions, 2 corpus attempts, and zero provider attempts. Corpus-wide research remained incomplete.

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

Exit `0` means the recursive experiment met its structural criteria. Exit `1` means the run produced a report and was incomplete. Invalid setup or arguments exit `2`. The report checks provider responses, separate singleton delegations for both years, validated child findings, acknowledged delivery, a finished synthesis declaring which findings it used, and citations from those used findings covering both years. A root that cites both filings directly still misses the delegation criterion. A validated excerpt can contribute even when the child reports incomplete analysis or partial coverage. The report preserves those statuses. An empty finding, failed child, or unacknowledged observation cannot satisfy that year's evidence requirement.

The JSON shows authorized pairs, the invocation tree, budget stops, errors, per-invocation counts, citations, and coverage. Requested tool counts include the native finish request. Operation records separate requests, execution outcomes, and delivery. The report also retains bounded, sanitized tool IDs, names, and arguments for rejected responses when the tool blocks can be decoded. Each excerpt records whether the parent declared its finding used. `--trace` adds per-decision and per-attempt telemetry. The report omits private reasoning, raw provider payloads, authentication headers, and whole observation bodies.

The report takes shared counts and measured token usage from the root's final budget snapshot. It separates investigation attempts, finish-only attempts, and unused child slots returned to the root. Per-invocation traces identify the change to finalization and its reason. Provider usage is aggregated once over the invocation tree, with missing-counter counts retained. Per-invocation wall times include their children, so adding them would double-count time. Summed provider latency and attempt wall time describe provider work separately from elapsed run time.

**Semantic review of the successful run.** The final answer inferred a shift toward geopolitical risks in FY2024 from two independently selected excerpts. An earlier FY2023 run retrieved `bd263969e91358073d0ea0a1`, which explicitly discusses geopolitical tensions involving Taiwan and China and their implications for supply continuity. The latest excerpts cannot establish that geopolitical risk was new or received greater emphasis in FY2024. The latest FY2023 quotation also does not independently support its finding's additional claim about demand estimation challenges. That child returned `finished` with a finding marked `analyzed`, while its invocation-level analysis status remained `incomplete`.

Citation provenance identifies the original chunk and quotation validation checks its wording. Coverage describes the evidence accessed, and analysis status is model-reported. In this run, both partitions remained partial. The original NVIDIA demo still reports per-invocation coverage. The planned execution ledger is exercised separately above. Independent semantic verification would have to test whether the evidence supports each claim and comparison.

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

Aggregation runs in Python without a corpus-sized model conversation. Invocation and finding IDs carry a run and batch namespace. Coverage unions original chunk IDs, delivered span IDs, and section character ranges against the full plan. Accepted root findings, citations, and comparisons remain separate from acknowledged child findings and comparisons. A failed or unacknowledged result can appear in the diagnostic ledger but contributes no accepted findings. Exact source excerpts and SEC provenance survive deduplication. Conflicting assessments and uncertainty stay visible. The deterministic answer separates completed execution, unresolved assessments, and pending scope. Partial access or lexical coverage cannot establish absence or novelty.

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

Remaining work includes compact reference-based root synthesis to reduce repeated evidence and generated structure, generalized candidate discovery beyond the fixed shortlist, and independent semantic verification of relevance, entailment, and disclosure changes. The two protected truncations motivate testing a more compact finish representation under the existing ceilings. The trace measures saturation and leaves the reason for the long generation unresolved.

Verification passed **254 focused RLM tests** and **53 related regressions**, each with the existing AnyIO deprecation warning. Bedrock stubs exercised compact sibling delivery, replay from a truncated protected finish, and shared policy propagation across trees. Offline six-company, full and limited corpus, comparative, planning, recursive, and Standard RAG demos passed their expected checks with zero provider attempts.

## Ongoing engineering record

For each future milestone, add a short entry to this guide with the **problem and baseline**, **hypothesis**, **solution and alternatives**, **verification method**, **before-and-after results**, and **trade-offs and limitations**. Include a reproduction command and, where applicable, the model and provider, corpus scope, execution limits, commit reference, and whether verification was scripted, offline, or live. Identify which variables changed, how many trials ran, and which metrics were unavailable. Link to the relevant design decision when it already explains the architecture.
