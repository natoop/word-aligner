# Multilingual Word Alignment API

English | [简体中文](README.zh-CN.md)

A translation word-alignment service built with FastAPI and [SimAlign](https://github.com/cisnlp/simalign). It accepts source/target language codes and corresponding sentence pairs, then returns contextual token-embedding links with semantic similarity, estimated confidence, character offsets, and grouped one-to-one, one-to-many, many-to-one, and many-to-many relationships.

SentencePiece is useful for subword tokenization and model encoding, but it does not establish correspondence between source and translated text. This service follows a review-oriented alignment pipeline:

```text
Source/target text -> word tokenization -> contextual subword embeddings
-> mean-pooled word embeddings -> similarity matrix -> SimAlign matching
-> confidence scoring and conservative repair -> grouped relations and character offsets
```

Chinese (`zh`, `zh-Hans`, and `zh-Hant`) uses `jieba` for display-level word tokenization. Placeholders and markup such as `[[T1504_1]]`, `${name}`, `{{name}}`, and HTML tags are preserved as complete tokens and force-aligned when their text is identical.

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
    "min_confidence": 0.35
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
- `repair`: optional. When omitted, the service returns the original model result. When present, conservative missing-link repair is enabled.
- `repair.strategy`: currently supports only `conservative`.
- `repair.max_position_distance`: maximum normalized source/target token-position distance, from `0` to `1`; defaults to `0.35`.
- `repair.min_similarity`: minimum normalized cosine similarity for a repaired link; defaults to `0.45`.
- `repair.min_confidence`: minimum estimated confidence for a repaired link; defaults to `0.35`.

Conservative repair does not force every unaligned token onto a neighboring token. It considers only content for which both sides remain unaligned, uses `mwmf` from the same embedding inference as a candidate, and filters by similarity, confidence, positional distance, and neighboring alignment anchors. Each repaired token can participate in only one new relationship. For example, if `itermax` misses `machine ↔ آلة` while `mwmf` finds a sufficiently strong candidate, the service adds it with `origin: "repaired"`. A token remains unaligned when only one side has a viable candidate or the score thresholds are not met.

Response structure example (alignment content is illustrative; actual results depend on the model):

```json
{
  "source_language": "en",
  "target_language": "zh-Hans",
  "model": "xlmr",
  "embedding_layer": 8,
  "method": "itermax",
  "confidence_method": "bidirectional-softmax-margin-v1",
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

`origin` identifies how each link was created:

- `model`: returned by the selected SimAlign method.
- `rule`: created by exact placeholder or markup matching.
- `repaired`: added by conservative repair.

`similarity` is `(cosine + 1) / 2` for the contextual word-token embeddings. `confidence` is an uncalibrated estimate combining bidirectional softmax probabilities, source/target candidate margins, mutual-best status, and agreement across enabled matching methods. Rule links use `1.0` for both values. The response exposes `confidence_method` so clients can distinguish this estimate from a future probability calibrated on labeled alignments.

`start` and `end` are zero-based, half-open offsets measured in Unicode code points. In Python, `text[start:end]` returns the original token. The service preserves leading and trailing whitespace, so offsets always refer to the exact request text.

JavaScript string indices use UTF-16 code units. If a non-BMP character such as an emoji appears before a token, convert the code-point offset to a UTF-16 offset before slicing. Frontends should normally consume `alignment_groups` for bilingual highlighting and use the two `unaligned_*_indices` fields to display unaligned content.

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
