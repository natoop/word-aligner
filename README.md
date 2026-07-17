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

Open `http://localhost:8000/docs` for Swagger UI. You can also run the service with Docker Compose:

```powershell
docker volume create word-aligner-models
docker compose up --build
```

The Compose project is named `word-aligner`, and the image is `word-aligner:latest`. Models are mounted through the logical `model-cache` volume. To avoid downloading XLM-R again after the project rename, the physical `word-aligner-models` volume is declared external; `docker volume create` is idempotent and does not clear an existing cache.

The checked-in Compose configuration targets an NVIDIA GPU: it builds PyTorch from the CUDA 12.8 wheel index, assigns all available GPUs to the container, and sets `ALIGNER_DEVICE=cuda`. Docker Desktop or Docker Engine must expose the NVIDIA runtime to containers. The Dockerfile itself keeps the CPU wheel index as its default, so a direct `docker build .` remains CPU-oriented. To convert the Compose deployment to CPU, change the build argument to `https://download.pytorch.org/whl/cpu`, remove `gpus: all`, and set `ALIGNER_DEVICE=cpu`.

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
- `repair.min_score_gain`: minimum pooled-embedding score improvement for a `span-aware` expansion; defaults to `0.05`.
- `repair.min_span_coverage`: coverage threshold used to detect asymmetric clauses and accept expansions; defaults to `0.75`.

Conservative repair does not force every unaligned token onto a neighboring token. It first considers only content for which both sides remain unaligned, uses `mwmf` from the same embedding inference as a candidate, and filters by similarity, confidence, positional distance, and neighboring alignment anchors. Each repaired token can participate in only one new relationship.

It then checks clauses bounded by aligned punctuation or protected-token anchors. If the clause has usable alignment evidence but either side remains below `min_span_coverage`, ambiguous lexical links are refined without crossing those anchors. A clause that fits within `max_source_span` and `max_target_span` becomes one explicit `origin: "refined"` phrase group. For example, `北方的北方 ↔ The far north of the North` is represented as a `many-to-many` group rather than preserving a high-scoring but semantically wrong `北方 ↔ The` link.

When the full clause is larger than either limit, the limits apply to each local candidate span rather than disabling refinement. A bounded monotonic optimizer combines mean-pooled span similarity, neighborhood-relative evidence, and balanced token-position boundaries. It preserves self-contained model groups and rewrites only spans containing omissions, expansions, or cross-group links. Thus `This is a machine translation example. ↔ 这是一个机器翻译示例。` becomes `This is ↔ 这是`, `a ↔ 一个`, `machine translation ↔ 机器翻译`, and the unchanged model group `example ↔ 示例`. Local optimization is capped at 64 content tokens across both sides. Punctuation links remain atomic, and tokens covered by explicit phrase groups are not reported as unaligned.

Full coverage does not automatically imply correct token roles. Within an anchored clause, a contiguous one-to-one chain that runs in exact reverse target order is collapsed into one local refined phrase group. This handles swaps such as `原子/链接 → links/atomic` as `原子链接 ↔ atomic links` while leaving the surrounding reordered but semantically valid links intact. Wider or non-contiguous reordering is not rewritten by this conservative crossing rule.

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
- `rule`: created by exact placeholder or markup matching.
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

## Testing

Tests use a fake alignment backend and do not download XLM-R:

```powershell
pip install -r requirements-dev.txt
pytest
ruff check .
```
