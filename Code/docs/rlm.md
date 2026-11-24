# Recursive language model research

## Architecture and control flow

1. `RootController` sends the question, tool schemas, previous observation, and remaining time to a model client. The scripted client plays back predetermined decisions for offline tests.
2. The model client maps a provider reply into one internal tool or finish decision. The next request replays the original assistant message and pairs the tool result with the provider's tool-use ID.
3. `CorpusToolDispatcher` validates the name and arguments against the same models that publish the schemas. It calls only allowlisted local operations and returns a bounded JSON success or error. Invalid arguments become feedback the model can correct.
4. The controller accepts a structured finish only when its cited chunk IDs were delivered to the model. It records requested decisions, validated arguments, observations, original evidence IDs, provider attempts, timing, and stop reason.

`CorpusEnvironment` loads `data/filings_manifest.csv` and the section and chunk Parquet files under `data/processed/full-v1/`. The tools list companies, filings, and sections, fetch filing metadata or original chunks, page through section text, and search with BM25. The corpus stays local. The provider receives the question and the observations accumulated during the run.

The caller can supply required filing and section IDs through `ResearchScope`. An unspecified requirement is unknown. Pagination is complete when delivered ranges cover the requested collection or text. The trace distinguishes search discoveries, delivered chunks, and pagination. Claim verification remains a separate step.

## Course setup with Amazon Bedrock

Bedrock is AWS's managed model API. Its Converse interface uses AWS authentication and a common message format for models such as Claude. EvidenceRoute uses Boto3's `bedrock-runtime` client to sign and send these requests. Bedrock is the expected course provider based on the temporary AWS credentials. The course account, model access, and region still need live verification. [AWS Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)

| Setting | Behavior |
| --- | --- |
| `BEDROCK_MODEL_ID` | Required. Use the model ID or inference profile authorized for the course account. AWS lists `us.anthropic.claude-sonnet-4-6` and `global.anthropic.claude-sonnet-4-6` among the profile IDs. |
| AWS region | `BEDROCK_REGION`, then `AWS_REGION`, then the Boto3 session region, which can come from `AWS_DEFAULT_REGION` or an AWS profile. |
| AWS credentials | Boto3's standard provider chain. For manually supplied temporary environment credentials, supply `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` together. Use `AWS_PROFILE` for an externally configured profile. Environment credentials can take precedence over a profile. |
| `CLAUDE_TIMEOUT_SECONDS` | 20 seconds by default. Each attempt caps its connect and read timeouts by remaining run time. |
| `CLAUDE_MAX_OUTPUT_TOKENS` | 1,024 generated tokens per response by default. |
| `CLAUDE_MAX_RETRIES` | One retry per decision by default, configurable from zero to two. |
| `CLAUDE_TEMPERATURE`, `CLAUDE_TOP_P` | Omitted by default. At most one can be set when supported by the selected model. |

Confirm the course-approved region, inference profile, and `bedrock:InvokeModel` permission before enabling live calls. [Claude Sonnet 4.6 on Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html) · [Converse API permissions](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)

Course credentials reportedly expire about every four hours. Manually supplied environment credentials do not refresh automatically. On expiry, obtain a fresh credential set from the course issuer, replace all three values in the shell environment or externally managed profile, and start a new process. Remove stale environment credentials if switching to a profile. The application delegates credential retrieval and any provider-supported refresh to Boto3. Keep credentials outside the repository and leave SDK wire debugging off when using real credentials. Expired or missing credentials, invalid authentication, access denial, and configuration errors stop the run without retries and appear as sanitized trace codes. [Boto3 credential provider chain](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html)

The optional demo asks for one cited NVIDIA FY2024 disclosure and reports coverage, citations, request attempts, tokens, latency, and stop reason. Run it after confirming the course configuration. The `--live` flag authorizes provider calls.

```bash
PYTHONPATH=Code/src python Code/examples/demo_rlm_claude.py --provider bedrock --live
```

Direct Anthropic Messages remains available through `--provider anthropic --live` for environments with a separate `ANTHROPIC_API_KEY`. It uses optional `ANTHROPIC_BASE_URL` and `CLAUDE_MODEL` (default `claude-sonnet-4-6`) and shares the `CLAUDE_*` request limits above.

## Design Decisions & Trade-offs

| Decision | Choice and reason | Alternatives and limitations |
| --- | --- | --- |
| RLM-01 · Local validated snapshot | Reuse filing, section, chunk, and corpus schemas. Validate provenance and text offsets and return defensive copies of mutable records. | Streaming Parquet would reduce peak memory. The current snapshot pays the loading cost up front for straightforward lookups. |
| RLM-02 · Exact pages and lexical search | Character windows and whole chunks preserve source text. Existing BM25 lookup and chunk-ID tie breaking give repeatable search. | Whole sections can be too large for a tool turn. Dense search could find different wording at the cost of another model and index. Keyword search can miss relevant evidence. The initial BM25 build is lazy. |
| RLM-03 · Allowlisted tool schemas | Generate schemas and validate requests using the same argument models. The dispatcher owns execution. | Separate schemas and validators can drift. An unrestricted function interface would broaden model-controlled execution. Adding a tool requires an explicit registry change. |
| RLM-04 · Bounded decisions and observations | Reuse the root-client interface. Reject oversized results as a whole so evidence and offsets remain consistent. | Parsing prose would need extra conventions. Truncating JSON could remove provenance. Oversized requests may need a smaller page or section window. |
| RLM-05 · Separate scope, access, and analysis | Require caller-established scope, merge delivered ranges, and accept citations only to delivered chunk IDs. | Inferring scope or counting the final page alone can overstate coverage. Citation checks establish access to evidence. Claim verification is pending. |
| RLM-06 · Preserve native provider messages | Replay assistant content, including reasoning signatures when present. Translate Bedrock `toolUse` and `toolResult` at the edge and use `finish_research` for validated completion. Bedrock Converse has no single-call switch in its tool choice. The adapter instructs the model to use one tool and rejects multiple calls before dispatch. The direct Messages transport requests single-tool turns. | A second controller would duplicate validation and coverage. Accepting prose as a finish would bypass those checks. Tool result errors travel in the existing JSON body. Unexpected or truncated responses leave the run incomplete. |
| RLM-07 · AWS SDK, bounded attempts, and opt-in | Use the existing Boto3 dependency and standard credential chain. Disable SDK retries for Converse so the shared loop owns retry accounting and time checks. Reuse the session and apply the remaining time to new request clients. | Handwritten signing and credential rotation would duplicate SDK behavior. A new client per attempt adds setup cost. Socket timeouts and credential resolution are synchronous and cannot guarantee a hard deadline. Late decisions are discarded. |
| RLM-08 · Per-attempt usage and acknowledged delivery | Record input/output and cache counters when supplied, wall latency, Bedrock latency when supplied, and sanitized errors. Credit a provider observation only after an assistant response acknowledges the turn. | An iteration count hides retries. A lost response leaves delivery and usage unknown. Neither adapter requests prompt caching or estimates cost. |

See the [Converse tool contract](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-client-side.html) and [Boto3 request configuration](https://botocore.amazonaws.com/v1/documentation/api/latest/reference/config.html) for wire and timeout details.

## Resource limits and verification

The default root budget is 20 model iterations, 12 tool attempts, 24,000 UTF-8 bytes per observation, and 30 seconds. Environment pages allow up to 25 metadata rows, 10 original chunks, or 4,000 characters of section text. Search queries allow up to 1,000 characters. Invalid decisions spend iterations. A valid finish can still be accepted after the tool budget is spent. The demo uses tighter iteration, tool, and response-size limits.

Fixture tests cover corpus provenance, pagination, dispatch, coverage, both provider adapters, and controller integration. Bedrock tests inject or stub responses, including through the real Boto3 Converse operation model. No live provider calls were made. The full-corpus SentenceTransformer integration test passed separately with `1 passed, 2 warnings in 119.77s`. That test was outside the focused RLM run.

Live verification still needs the course-approved AWS region, inference profile, and permissions. Automatic scope planning, recursive sub-model calls, structured findings, evidence-preserving aggregation, claim verification, and recursion-wide budgets are pending.
