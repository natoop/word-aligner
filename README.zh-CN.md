# Multilingual Word Alignment API

[English](README.md) | 简体中文

基于 FastAPI 和 [SimAlign](https://github.com/cisnlp/simalign) 的翻译词对齐服务。它接收原文/译文语种和句对列表，逐句返回词、字符位置、原始对齐边，以及聚合后的一对一、一对多、多对一和多对多关系。

SentencePiece 适合做子词切分和模型编码，但不负责建立原文与译文之间的对应关系。本服务按照实际审校链路处理：

```text
原文/译文 -> 词级切分 -> SimAlign 对齐 -> 聚合关系 -> 字符位置映射
```

中文（`zh`、`zh-Hans`、`zh-Hant`）使用 `jieba` 做展示层词切分。`[[T1504_1]]`、`${name}`、`{{name}}` 和 HTML 标签会作为完整词保留，并按完全相同的内容强制对齐。

## 启动

建议使用 Python 3.10～3.12。`requirements.txt` 已显式包含 XLM-R tokenizer 需要的 `sentencepiece` Python 包。首次真正执行对齐时，SimAlign 会从 Hugging Face 下载 XLM-R 模型，下载耗时和磁盘占用取决于网络环境；完整缓存建立后，服务会自动切换到离线模式，不再向 Hugging Face 检查更新。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开 `http://localhost:8000/docs` 可使用 Swagger UI。也可以用 Docker Compose：

```powershell
docker compose up --build
```

Compose 项目名为 `word-aligner`，镜像名为 `word-aligner:latest`。模型缓存在逻辑卷 `model-cache` 中；为避免项目改名后重复下载 XLM-R，当前 Compose 会继续复用已有的物理模型卷。Docker 镜像显式安装 CPU-only PyTorch，不会携带无用的 CUDA 运行库。请保持单 worker，多个 worker 会各自加载一份大模型；需要横向扩容时，更适合启动多个单 worker 容器。

### 模型缓存与更新策略

默认的 `HF_MODEL_UPDATE_POLICY=if-missing` 会在模型初始化前检查 Hugging Face 缓存：

- 缺少配置、权重或 tokenizer 文件：允许联网下载或继续未完成的下载。
- 缓存完整：自动设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，后续启动只读本地缓存，不发送 HEAD 更新请求。
- 缓存由 Compose 命名卷持久化，重新构建或创建容器不会重复下载。

也可以设置为 `always`，每次启动都允许检查更新；设置为 `offline` 则无论缓存是否完整都禁止联网，缓存缺失时应用会启动失败。

## API

### `GET /api/v1/languages`

返回服务公开支持的词对齐语种。列表中的语种可作为原文或译文，并支持任意互配；调用该接口不会触发大模型加载。

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

当前公开列表只包含已有可用词级切分策略的语种。XLM-R 虽然还覆盖日语、泰语等语种，但这些语言需要额外的专用分词器，因此暂不在接口中声明为受支持。

### `POST /api/v1/align`

请求：

```json
{
  "source_language": "en",
  "target_language": "zh-Hans",
  "method": "itermax",
  "repair": {
    "enabled": true,
    "strategy": "conservative",
    "max_position_distance": 0.35
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

字段约束：

- `source_language`、`target_language`：BCP 47 风格语种代码，例如 `en`、`zh-Hans`、`de`。
- `sentence_pairs`：1～100 个已经互相对应的句对；单侧文本最长 10,000 个字符。
- `method`：默认 `itermax`，通常在召回率与准确率之间较均衡；`inter` 更保守；`mwmf` 是最大权匹配结果。
- `repair`：可选；不传时保持原始模型结果。传入后启用保守漏词修正。
- `repair.strategy`：当前仅支持 `conservative`。
- `repair.max_position_distance`：源词和目标词的最大归一化位置距离，范围 `0～1`，默认 `0.35`。

保守修正不会把所有漏词强制附着到邻词。它只处理源端和目标端都尚未对齐的内容，以同次推理得到的 `mwmf` 作为语义候选，再用位置距离过滤，并保证每个修正词只参与一条新增关系。例如 `itermax` 漏掉 `machine ↔ آلة`、且 `mwmf` 找到了该关系时，会将它补充为 `origin: "repaired"`；只有单侧漏词时仍允许保持未对齐。

响应结构示例（对齐内容仅用于解释结构，真实结果由模型决定）：

```json
{
  "source_language": "en",
  "target_language": "zh-Hans",
  "model": "xlmr",
  "method": "itermax",
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
        {"source_index": 0, "target_index": 0, "origin": "model"},
        {"source_index": 1, "target_index": 0, "origin": "model"}
      ],
      "alignment_groups": [
        {
          "type": "many-to-one",
          "source_indices": [0, 1],
          "target_indices": [0],
          "source_tokens": ["New", "York"],
          "target_tokens": ["纽约"],
          "links": [
            {"source_index": 0, "target_index": 0, "origin": "model"},
            {"source_index": 1, "target_index": 0, "origin": "model"}
          ]
        }
      ],
      "unaligned_source_indices": [],
      "unaligned_target_indices": []
    }
  ]
}
```

`origin` 用于区分链接来源：`model` 是所选 SimAlign 方法的原始关系，`rule` 是占位符或标签的精确规则关系，`repaired` 是保守修正补充的关系。

`start`/`end` 是 Unicode 码点维度的零基、左闭右开字符位置，可在 Python 中直接用 `text[start:end]` 取回原词。服务不会裁掉输入文本首尾的空白，因此位置始终相对于原始请求文本。JavaScript 使用 UTF-16 下标；当文本在目标词之前含有 emoji 等非 BMP 字符时，需要先把码点位置换算成 UTF-16 位置。前端做双语高亮时应优先消费 `alignment_groups`；需要显示未对齐内容时使用两个 `unaligned_*_indices` 字段。

### 健康检查

- `GET /health/live`：进程存活检查。
- `GET /health/ready`：返回模型名、是否已经加载以及加载模式。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ALIGNER_MODEL` | `xlmr` | SimAlign 模型别名 |
| `ALIGNER_TOKEN_TYPE` | `bpe` | 模型 token 类型 |
| `ALIGNER_MATCHING_METHODS` | `mai` | 同时启用 `mwmf`、`inter`、`itermax` |
| `ALIGNER_DEVICE` | `cpu` | 推理设备，如 `cpu` 或 `cuda` |
| `ALIGNER_EAGER_LOAD` | `false` | `true` 时在应用启动阶段加载模型；否则第一次对齐时加载 |
| `HF_HOME` | Hugging Face 默认目录 | 模型缓存位置 |
| `HF_MODEL_UPDATE_POLICY` | `if-missing` | `if-missing` 缓存完整后自动离线；也可设为 `always` 或 `offline` |

## 测试

测试使用假的对齐后端，因此不会下载 XLM-R 模型：

```powershell
pip install -r requirements-dev.txt
pytest
ruff check .
```
