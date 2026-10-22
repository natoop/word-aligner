# Multilingual Word Alignment API

English | [简体中文](README.zh-CN.md)

A translation word-alignment service built with FastAPI and [SimAlign](https://github.com/cisnlp/simalign). It accepts source/target language codes and corresponding sentence pairs, then returns contextual token-embedding links with semantic similarity, estimated confidence, character offsets, and grouped one-to-one, one-to-many, many-to-one, and many-to-many relationships.

SentencePiece is useful for subword tokenization and model encoding, but it does not establish correspondence between source and translated text. This service follows a review-oriented alignment pipeline:

```text
Source/target text -> word tokenization -> contextual subword embeddings
-> mean-pooled word embeddings -> similarity matrix -> SimAlign matching
-> relative confidence scoring and span refinement -> grouped relations and character offsets
```

Chinese (`zh`, `zh-Hans`, and `zh-Hant`) uses `jieba` for display-level word tokenization. Placeholders, markup, and index-arrow expressions such as `[[T1504_1]]`, `${name}`, `{{name}}`, HTML tags, `5→3`, and `6->4` are preserved as complete tokens and force-aligned by identical text and occurrence order.

## Getting Started

Python 3.10–3.12 is recommended. `requirements.txt` explicitly includes the Python `sentencepiece` package required by the XLM-R tokenizer. On the first real alignment request, SimAlign downloads the XLM-R model from Hugging Face. Once a complete cache exists, the service automatically switches to offline mode and stops checking Hugging Face for updates.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/demo` for the interactive demo, or `http://localhost:8000/docs` for Swagger UI. The demo lets you choose the local model or an LLM provider, loads language choices from `GET /api/v1/languages`, and highlights grouped alignments. It uses `/api/v1/align` for the local model and `/api/v1/llm/align` for an LLM. The LLM path does not load XLM-R; the existing local-model endpoint remains available. The current local test address is `http://localhost:8000`. Serve the page through `/demo` instead of opening the HTML file directly, since browsers may block `file://` requests to the API. You can also run the service with Docker Compose:

```powershell
docker volume create word-aligner-models
docker compose up --build
```

The Compose project is named `word-aligner`, and the image is `word-aligner:latest`. Models are mounted through the logical `model-cache` volume. To avoid downloading XLM-R again after the project rename, the physical `word-aligner-models` volume is declared external; `docker volume create` is idempotent and does not clear an existing cache.

The checked-in Compose configuration targets an NVIDIA GPU: it builds PyTorch from the CUDA 12.8 wheel index, assigns all available GPUs to the container, and sets `ALIGNER_DEVICE=cuda`. Docker Desktop or Docker Engine must expose the NVIDIA runtime to containers. The Dockerfile itself keeps the CPU wheel index as its default, so a direct `docker build .` remains CPU-oriented. To convert the Compose deployment to CPU, change the build argument to `https://download.pytorch.org/whl/cpu`, remove `gpus: all`, and set `ALIGNER_DEVICE=cpu`.

To configure LLM providers, copy `.env.example` to `.env` and fill in the provider settings described below. Docker Compose 2.24 or newer passes this optional file into the container; without `.env`, the service still starts with no LLM providers. The app also reads the project-root `.env` directly when run outside Docker. Process environment variables take precedence, and changes require an application restart. If you only want to try LLM alignment in the Compose deployment, set `ALIGNER_EAGER_LOAD=false` to skip the local XLM-R warm-up at startup.

Keep the service at one worker per container because every worker loads its own copy of the model. For horizontal scaling, run multiple single-worker containers.

### Model Cache and Update Policy

The default `HF_MODEL_UPDATE_POLICY=if-missing` checks the Hugging Face cache before model initialization:

- If configuration, weight, or tokenizer files are missing, network downloads and incomplete-download recovery remain enabled.
- If the cache is complete, the service sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Future starts use only local files and do not send HEAD update requests.
- The Compose volume persists the cache, so rebuilding or recreating the container does not download the model again.

Set the policy to `always` to allow update checks on every start. Set it to `offline` to prohibit all network access even when the cache is incomplete; application startup then fails if required model files are unavailable.

## API

### `GET /api/v1/languages`

Returns the languages that this service publicly supports for word alignment. Any listed language can be used as either the source or target language, and any listed pair can be combined. Calling this endpoint does not load the model.

```json
{
  "model": "xlmr",
  "pairing": "any-to-any",
  "total": 48,
  "languages": [
    {
      "code": "en",
      "name": "English",
      "native_name": "English",
      "tokenizer": "unicode-regex"
    },
    {
      "code": "zh-Hans",
      "name": "Simplified Chinese",
      "native_name": "简体中文",
      "tokenizer": "jieba"
    }
  ]
}
```

The public list includes only languages for which the service has a usable word-tokenization strategy. XLM-R also covers languages such as Japanese and Thai, but they require additional language-specific tokenizers and therefore are not currently advertised as supported.

### `GET /api/v1/llm/providers`

Returns the configured LLM provider choices for the demo. The response contains only provider IDs, display names, and model names; API keys and base URLs are never exposed.

```json
{
  "providers": [
    {"id": "deepseek", "name": "DeepSeek", "model": "deepseek-chat"},
    {"id": "local", "name": "Local server", "model": "your-local-model"}
  ],
  "default_provider": "deepseek"
}
```

### `POST /api/v1/llm/align`

This independent endpoint aligns already-translated text through a configured OpenAI-compatible chat-completions API. It does not use the XLM-R model, and its request does not accept the local endpoint's `method` or `repair` fields.

```json
{
  "provider": "deepseek",
  "source_language": "en",
  "target_language": "zh-Hans",
  "sentence_pairs": [
    {"id": "demo", "source": "He gave up.", "target": "他放弃了。"}
  ]
}
```

`provider` is optional when a default provider is configured. The language fields are optional and default to `auto`. Submit 1–10 corresponding sentence pairs; each side may contain up to 10,000 characters. The server splits text into alignment units: Chinese and other scripts without spaces use fine-grained character or grapheme units, while ordinary word runs remain grouped. These are display units, not necessarily linguistic words. Each side of a pair is limited to 1,024 units, with a 4,096-unit total limit across the request; requests over these limits return `422`.

The model returns unit IDs only. The server validates that every source and target unit appears exactly once, either in an alignment or in the explicit unaligned list, and rejects unknown, repeated, or omitted IDs. It then builds the token text, UTF-16 offsets, and grouped alignment response. An unaligned unit means the model did not assign it a counterpart; it does not by itself mean that the translation omitted its meaning.

Response shape (alignment choices and usage below are illustrative):

```json
{
  "source_language": "en",
  "target_language": "zh-Hans",
  "provider": "deepseek",
  "provider_name": "DeepSeek",
  "model": "deepseek-chat",
  "method": "llm",
  "offset_unit": "utf16",
  "confidence_method": null,
  "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
  "sentence_alignments": [
    {
      "index": 0,
      "id": "demo",
      "source": "He gave up.",
      "target": "他放弃了。",
      "source_tokens": [
        {"index": 0, "text": "He", "start": 0, "end": 2, "is_protected": false},
        {"index": 1, "text": "gave", "start": 3, "end": 7, "is_protected": false},
        {"index": 2, "text": "up", "start": 8, "end": 10, "is_protected": false},
        {"index": 3, "text": ".", "start": 10, "end": 11, "is_protected": false}
      ],
      "target_tokens": [
        {"index": 0, "text": "他", "start": 0, "end": 1, "is_protected": false},
        {"index": 1, "text": "放", "start": 1, "end": 2, "is_protected": false},
        {"index": 2, "text": "弃", "start": 2, "end": 3, "is_protected": false},
        {"index": 3, "text": "了", "start": 3, "end": 4, "is_protected": false},
        {"index": 4, "text": "。", "start": 4, "end": 5, "is_protected": false}
      ],
      "links": [],
      "alignment_groups": [
        {
          "type": "one-to-one",
          "origin": "llm",
          "similarity": null,
          "confidence": null,
          "source_indices": [0],
          "target_indices": [0],
          "source_tokens": ["He"],
          "target_tokens": ["他"],
          "links": []
        },
        {
          "type": "many-to-many",
          "origin": "llm",
          "similarity": null,
          "confidence": null,
          "source_indices": [1, 2],
          "target_indices": [1, 2, 3],
          "source_tokens": ["gave", "up"],
          "target_tokens": ["放", "弃", "了"],
          "links": []
        },
        {
          "type": "one-to-one",
          "origin": "llm",
          "similarity": null,
          "confidence": null,
          "source_indices": [3],
          "target_indices": [4],
          "source_tokens": ["."],
          "target_tokens": ["。"],
          "links": []
        }
      ],
      "unaligned_source_indices": [],
      "unaligned_target_indices": [],
      "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}
    }
  ]
}
```

LLM token offsets are zero-based, half-open UTF-16 code-unit offsets into the unchanged input text, so JavaScript can use them directly for slicing. The response has `method: "llm"`, `offset_unit: "utf16"`, and `confidence_method: null`; groups use `origin: "llm"`, null similarity/confidence, and empty `links`. Per-sentence `usage` comes from the provider. Top-level `usage` sums the provider-reported counts and is `null` if any pair has no usage data.

Errors use `400` for an unknown provider, `422` for invalid or oversized input, `502` for upstream failures or invalid model output, `503` when no provider is configured, and `504` for a provider timeout. A provider rate limit can return `429`.

### `POST /api/v1/align`

Request:

```json
{
  "source_language": "en",
  "target_language": "zh-Hans",
  "method": "itermax",
  "repair": {
    "enabled": true,
    "strategy": "conservative",
    "max_position_distance": 0.35,
    "min_similarity": 0.45,
    "min_confidence": 0.35,
    "max_source_span": 3,
    "max_target_span": 6,
    "min_score_gain": 0.05,
    "min_span_coverage": 0.75
  },
  "sentence_pairs": [
    {
      "id": "sentence-1",
      "source": "New York is very beautiful.",
      "target": "纽约非常美丽。"
    },
    {
      "id": "sentence-2",
      "source": "Keep [[T1504_1]] unchanged.",
      "target": "保持 [[T1504_1]] 不变。"
    }
  ]
}
```

Field constraints:

- `source_language` and `target_language`: BCP 47-style language codes such as `en`, `zh-Hans`, and `de`.
- `sentence_pairs`: 1–100 already-corresponding sentence pairs. Each source or target text can contain up to 10,000 characters.
- `method`: defaults to `itermax`, which usually provides a balanced precision/recall tradeoff. `inter` is more conservative, while `mwmf` returns maximum-weight matching results.
- `repair`: optional. When omitted, the service returns the original model result. When present, missing-link repair and bounded phrase refinement are enabled.
- `repair.strategy`: `conservative` (default) or `span-aware`.
- `repair.max_position_distance`: maximum normalized source/target token-position distance, from `0` to `1`; defaults to `0.35`.
- `repair.min_similarity`: minimum normalized cosine similarity for a repaired link; defaults to `0.45`.
- `repair.min_confidence`: minimum estimated confidence for a repaired link; defaults to `0.35`.
- `repair.max_source_span`: maximum source-token count in a refined span; defaults to `3`.
- `repair.max_target_span`: maximum target-token count in a refined span; defaults to `6`.
- `repair.min_score_gain`: minimum score improvement for `span-aware` expansion and normalized composite repair-island expansion; defaults to `0.05`.
- `repair.min_span_coverage`: coverage threshold used to detect asymmetric clauses and accept expansions; defaults to `0.75`.

Conservative repair does not force every unaligned token onto a neighboring token. It first considers only content for which both sides remain unaligned, uses `mwmf` from the same embedding inference as a candidate, and filters by similarity, confidence, positional distance, and neighboring alignment anchors. Each repaired token can participate in only one new relationship.

It then checks clauses bounded by aligned punctuation or protected-token anchors. If the clause has usable alignment evidence but either side remains below `min_span_coverage`, ambiguous lexical links are refined without crossing those anchors. A clause that fits within `max_source_span` and `max_target_span` becomes one explicit `origin: "refined"` phrase group. For example, `北方的北方 ↔ The far north of the North` is represented as a `many-to-many` group rather than preserving a high-scoring but semantically wrong `北方 ↔ The` link.

When the full clause is larger than either limit, the limits apply to each local candidate span rather than disabling refinement. A bounded monotonic optimizer combines mean-pooled span similarity, neighborhood-relative evidence, and balanced token-position boundaries. It preserves self-contained model groups and rewrites only spans containing omissions, expansions, or cross-group links. Thus `This is a machine translation example. ↔ 这是一个机器翻译示例。` becomes `This is ↔ 这是`, `a ↔ 一个`, `machine translation ↔ 机器翻译`, and the unchanged model group `example ↔ 示例`. Local optimization is capped at 64 content tokens across both sides. Punctuation links remain atomic, and tokens covered by explicit phrase groups are not reported as unaligned.

Before an anchored clause is refined, a low-confidence lexical link that crosses an aligned punctuation anchor is discarded unless it is a locked rule relation or reaches the boundary-lock confidence. If this exposes unaligned content on both sides between the same neighboring monotonic links, only that bounded gap is refined. Two adjacent gap islands may merge across a weak internal link when their pooled embedding score passes the normalized `min_score_gain` test. Stable surrounding links remain atomic, and genuinely untranslated function words can remain unaligned.

Full coverage does not automatically imply correct token roles. Within an anchored clause, a contiguous one-to-one chain that runs in exact reverse target order is collapsed into one local refined phrase group. A sparse strictly descending run with at least three evidence links may use its contiguous bounding span when it fits within two configured local spans. This handles both `原子/链接 → links/atomic` as `原子链接 ↔ atomic links` and reordered names such as `Programme for the Endorsement of Forest ↔ 森林认证认可计划`, while leaving surrounding stable links intact.

English/Chinese temporal-clause reordering is handled at word granularity. A unique `before ↔ 之前/以前` or `after ↔ 之后/以后` marker can bound exchanged clauses. Each bounded block is aligned into individual words or small one-to-many/many-to-one groups using its XLM-R span embeddings; a two-word block may also reverse its word order. English articles and auxiliary words without independent translations may remain unaligned. For example, `Callback triggered before the cloned node is inserted into the original tree.` produces `Callback ↔ 回调`, `triggered ↔ 触发的`, `before ↔ 之前`, `cloned ↔ 克隆`, `node ↔ 节点`, `inserted ↔ 插入`, and `original tree ↔ 原树`. These groups have `origin: "refined"` and expose their explicit token links as `origin: "repaired"` in both group-level and sentence-level `links`. Link similarities retain the original token-pair cosine score, while their confidence uses the refined span evidence. Punctuation anchors remain atomic. The service does not collapse this ambiguous region into a whole-sentence group. This bounded rule covers the listed temporal constructions, not arbitrary syntactic reordering.

Before missing-link repair and span refinement, deterministic temporal normalization handles non-monotonic year/month formats. English abbreviated or full month names are normalized to `month:1` through `month:12`; four-digit years and Chinese `number + 年` spans are normalized to `year:YYYY`; Chinese `number + 月` spans are normalized to the same month key. Matching spans replace conflicting model links with complete `origin: "rule"` relations. For example, `Feb 2025 ↔ 2025年2月` returns `Feb ↔ [2, 月]` and `2025 ↔ [2025, 年]`, both with `1.0` rule scores and no unaligned tokens.

Deterministic quantity normalization also aligns equivalent percentages when tokenizer boundaries differ. A split source span such as `[70, %]` and a combined target token `[70%]` become one complete `many-to-one`, `origin: "rule"` relation with `1.0` scores. Repeated values are paired in occurrence order, so `70%`, `30%`, and `100%` in the same sentence remain distinct.

For an anchorless sentence with one-sided omissions, conservative repair builds soft anchors from monotonic one-to-one links whose confidence is at least `max(repair.min_confidence, 0.45)`. Intervals that are fully covered remain unchanged, even when an internal link is below the anchor threshold. A small one-sided gap is attached to the better neighboring anchor only when its pooled span score improves. The improvement is normalized by the remaining score headroom, `(expanded - base) / (1 - base)`, before comparison with `min_score_gain`.

Larger intervals containing omissions and weak placeholder links become composite repair islands. One adjacent anchor may be absorbed when the same normalized gain test passes and the anchor confidence is below the boundary lock threshold. A composite island can combine up to two configured local spans, while the global 64-token safety bound still applies. For example, `Palm oil ↔ 棕榈油` becomes one refined group, and `and ingredients derived from palm oils ↔ 及其衍生成分` becomes a second refined group; the stable suffix `must / be / RSPO / certified` keeps its model links.

`span-aware` includes the same conservative behavior and can additionally expand only one side of an existing group to contiguous, currently unaligned tokens when mean-pooled span embeddings improve by at least `min_score_gain`. Expansions cannot overlap, cross punctuation or protected-token anchors, or exceed the configured span limits.

Response structure example (alignment content is illustrative; actual results depend on the model):

```json
{
  "source_language": "en",
  "target_language": "zh-Hans",
  "model": "xlmr",
  "embedding_layer": 8,
  "method": "itermax",
  "confidence_method": "bidirectional-margin-span-v2",
  "sentence_alignments": [
    {
      "index": 0,
      "id": "sentence-1",
      "source": "New York is very beautiful.",
      "target": "纽约非常美丽。",
      "source_tokens": [
        {"index": 0, "text": "New", "start": 0, "end": 3, "is_protected": false},
        {"index": 1, "text": "York", "start": 4, "end": 8, "is_protected": false}
      ],
      "target_tokens": [
        {"index": 0, "text": "纽约", "start": 0, "end": 2, "is_protected": false}
      ],
      "links": [
        {"source_index": 0, "target_index": 0, "origin": "model", "similarity": 0.91, "confidence": 0.86},
        {"source_index": 1, "target_index": 0, "origin": "model", "similarity": 0.88, "confidence": 0.81}
      ],
      "alignment_groups": [
        {
          "type": "many-to-one",
          "origin": "model",
          "similarity": 0.895,
          "confidence": 0.835,
          "source_indices": [0, 1],
          "target_indices": [0],
          "source_tokens": ["New", "York"],
          "target_tokens": ["纽约"],
          "links": [
            {"source_index": 0, "target_index": 0, "origin": "model", "similarity": 0.91, "confidence": 0.86},
            {"source_index": 1, "target_index": 0, "origin": "model", "similarity": 0.88, "confidence": 0.81}
          ]
        }
      ],
      "unaligned_source_indices": [],
      "unaligned_target_indices": []
    }
  ]
}
```

Link `origin` identifies how each atomic relation was created:

- `model`: returned by the selected SimAlign method.
- `rule`: created by exact protected-token matching or normalized temporal matching.
- `repaired`: added by conservative repair.

Groups also expose `origin`, `similarity`, and `confidence`. A group with one link origin inherits that origin, a component containing multiple origins is `mixed`, and an explicit phrase span is `refined`. A conservative collapsed group has an empty `links` array because the service is intentionally withholding ambiguous token-level claims; a `span-aware` expansion retains its underlying evidence links.

Atomic-link `similarity` is `(cosine + 1) / 2` for contextual word-token embeddings. `confidence` is an uncalibrated v2 estimate combining bidirectional probability (`35%`), relative candidate margins (`30%`), agreement across enabled matching methods (`20%`), and local span/order consistency (`15%`). The candidate evidence is CSLS-style and neighborhood-relative, so uniformly high XLM-R cosine values do not dominate the estimate. Rule links use `1.0` for both values. The response exposes `confidence_method` so clients can distinguish this estimate from a future probability calibrated on labeled alignments.

`start` and `end` are zero-based, half-open offsets measured in Unicode code points. In Python, `text[start:end]` returns the original token. The service preserves leading and trailing whitespace, so offsets always refer to the exact request text.

JavaScript string indices use UTF-16 code units. If a non-BMP character such as an emoji appears before a token, convert the code-point offset to a UTF-16 offset before slicing. Frontends should normally consume `alignment_groups` for bilingual highlighting and use the two `unaligned_*_indices` fields to display unaligned content. Explicit refined spans count as aligned even when their `links` array is empty.

### Health Checks

- `GET /health/live`: process liveness.
- `GET /health/ready`: model name, load state, and eager/lazy load mode.

## Configuration

| Environment variable | Default | Description |
| --- | --- | --- |
| `ALIGNER_MODEL` | `xlmr` | SimAlign alias or full Hugging Face model ID, such as `microsoft/xlm-align-base` |
| `ALIGNER_TOKEN_TYPE` | `word` | Must be `word`; model subwords are mean-pooled into display-level word embeddings |
| `ALIGNER_LAYER` | `8` | Hidden-state layer used for contextual token embeddings |
| `ALIGNER_MATCHING_METHODS` | `mai` | Enables `mwmf`, `inter`, and `itermax` |
| `ALIGNER_CONFIDENCE_TEMPERATURE` | `0.1` | Positive softmax temperature used by bidirectional confidence scoring |
| `ALIGNER_DEVICE` | `cpu` | Inference device, such as `cpu` or `cuda`; the checked-in Compose file overrides it to `cuda` |
| `ALIGNER_EAGER_LOAD` | `false` | Load the model during startup when `true`; otherwise load it on the first alignment request |
| `HF_HOME` | Hugging Face default | Model cache location |
| `HF_MODEL_UPDATE_POLICY` | `if-missing` | `if-missing` automatically goes offline for a complete cache; also supports `always` and `offline` |
| `LLM_PROVIDERS` | auto-discovered | Optional comma-separated provider IDs and ordering; when empty or omitted, providers are discovered from their `LLM_<ID>_*` settings. LLM support stays disabled when no providers are configured |
| `LLM_DEFAULT_PROVIDER` | first configured provider | Optional default provider used when a request omits `provider`; discovered providers are ordered alphabetically |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-request upstream timeout |
| `LLM_MAX_OUTPUT_TOKENS` | `8192` | Maximum output tokens requested from the provider |

Each configured provider requires `LLM_<ID>_NAME`, `LLM_<ID>_BASE_URL`, and `LLM_<ID>_MODEL`. It may also set `LLM_<ID>_API_KEY` (optional for local servers) and `LLM_<ID>_RESPONSE_FORMAT` (`json_object`, `json_schema`, or `none`; default `json_object`). If `LLM_PROVIDERS` is empty or omitted, provider IDs are discovered from these uppercase variables and sorted alphabetically; an explicit list takes precedence and preserves its order. Use a base URL such as `https://api.deepseek.com/v1` or `https://api.openai.com/v1`; do not include `/chat/completions`, which the server appends. For OpenAI, set `LLM_OPENAI_MAX_TOKENS_PARAMETER=max_completion_tokens` when needed; the default is `max_tokens`.

Example configuration (use your own secret in the ignored `.env` file, never commit it):

```dotenv
LLM_PROVIDERS=deepseek,openai,local
# Optional; defaults to the first listed or discovered provider.
# LLM_DEFAULT_PROVIDER=deepseek
LLM_TIMEOUT_SECONDS=120
LLM_MAX_OUTPUT_TOKENS=8192

LLM_DEEPSEEK_NAME=DeepSeek
LLM_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_DEEPSEEK_MODEL=deepseek-chat
LLM_DEEPSEEK_API_KEY=
LLM_DEEPSEEK_RESPONSE_FORMAT=json_object

LLM_OPENAI_NAME=OpenAI
LLM_OPENAI_BASE_URL=https://api.openai.com/v1
LLM_OPENAI_MODEL=gpt-4.1-mini
LLM_OPENAI_API_KEY=
LLM_OPENAI_RESPONSE_FORMAT=json_schema
LLM_OPENAI_MAX_TOKENS_PARAMETER=max_completion_tokens

LLM_LOCAL_NAME=Local server
LLM_LOCAL_BASE_URL=http://host.docker.internal:1234/v1
LLM_LOCAL_MODEL=your-local-model
LLM_LOCAL_API_KEY=
LLM_LOCAL_RESPONSE_FORMAT=json_object
```

The local base URL must be reachable from the API process. When the API runs in Compose, `localhost` refers to the container itself; use a host address such as `host.docker.internal` or a service name on the same Docker network.

For a quick PowerShell call, the demo is available at `/demo`; the API accepts the same JSON shown above:

```powershell
$body = @{
  provider = 'deepseek'
  source_language = 'en'
  target_language = 'zh-Hans'
  sentence_pairs = @(@{ id = 'demo'; source = 'He gave up.'; target = '他放弃了。' })
} | ConvertTo-Json -Depth 6
Invoke-RestMethod -Uri 'http://localhost:8000/api/v1/llm/align' -Method Post -ContentType 'application/json' -Body $body
```

## Testing

Tests use a fake alignment backend and do not download XLM-R:

```powershell
pip install -r requirements-dev.txt
pytest
ruff check .
```

LLM client tests use a mock HTTP transport. No live provider call or real model quality check is implied by this local test suite.
