# Multilingual Word Alignment API

[English](README.md) | 简体中文

基于 FastAPI 和 [SimAlign](https://github.com/cisnlp/simalign) 的翻译词对齐服务。它接收原文/译文语种和句对列表，逐句返回带语义相似度和估算置信度的上下文 Token Embedding 对齐、字符位置，以及聚合后的一对一、一对多、多对一和多对多关系。

SentencePiece 适合做子词切分和模型编码，但不负责建立原文与译文之间的对应关系。本服务按照实际审校链路处理：

```text
原文/译文 -> 词级切分 -> 上下文子词 Embedding
-> 子词均值聚合为词级 Embedding -> 相似度矩阵 -> SimAlign 匹配
-> 相对置信度评分和 Span 精炼 -> 聚合关系及字符位置映射
```

中文（`zh`、`zh-Hans`、`zh-Hant`）使用 `jieba` 做展示层词切分。`[[T1504_1]]`、`${name}`、`{{name}}`、HTML 标签以及 `5→3`、`6->4` 这样的索引箭头表达式会作为完整 token 保留，并按相同文本和出现顺序强制对齐。

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
docker volume create word-aligner-models
docker compose up --build
```

Compose 项目名为 `word-aligner`，镜像名为 `word-aligner:latest`。模型缓存在逻辑卷 `model-cache` 中；为避免项目改名后重复下载 XLM-R，物理卷 `word-aligner-models` 被声明为 external。`docker volume create` 可以安全重复执行，不会清空已有缓存。

仓库中的 Compose 配置默认使用 NVIDIA GPU：从 CUDA 12.8 PyTorch wheel 源构建镜像，将可用 GPU 分配给容器，并设置 `ALIGNER_DEVICE=cuda`。Docker Desktop 或 Docker Engine 必须已经向容器开放 NVIDIA runtime。Dockerfile 自身仍以 CPU wheel 源作为默认值，因此直接执行 `docker build .` 时默认构建 CPU 镜像。若要把 Compose 改为 CPU，需要将构建参数改成 `https://download.pytorch.org/whl/cpu`、移除 `gpus: all`，并把 `ALIGNER_DEVICE` 改为 `cpu`。

请保持单 worker，多个 worker 会各自加载一份大模型；需要横向扩容时，更适合启动多个单 worker 容器。

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

字段约束：

- `source_language`、`target_language`：BCP 47 风格语种代码，例如 `en`、`zh-Hans`、`de`。
- `sentence_pairs`：1～100 个已经互相对应的句对；单侧文本最长 10,000 个字符。
- `method`：默认 `itermax`，通常在召回率与准确率之间较均衡；`inter` 更保守；`mwmf` 是最大权匹配结果。
- `repair`：可选；不传时保持原始模型结果。传入后启用漏词修正和有边界的短语 Span 精炼。
- `repair.strategy`：支持默认的 `conservative` 和更积极的 `span-aware`。
- `repair.max_position_distance`：源词和目标词的最大归一化位置距离，范围 `0～1`，默认 `0.35`。
- `repair.min_similarity`：修正链接允许的最低归一化余弦相似度，默认 `0.45`。
- `repair.min_confidence`：修正链接允许的最低估算置信度，默认 `0.35`。
- `repair.max_source_span`：精炼 Span 最多包含的源端 token 数，默认 `3`。
- `repair.max_target_span`：精炼 Span 最多包含的目标端 token 数，默认 `6`。
- `repair.min_score_gain`：`span-aware` 扩展及软修复岛边界扩展所需的最小分数增益，默认 `0.05`。
- `repair.min_span_coverage`：识别覆盖不对称分句和接受扩展时使用的覆盖率阈值，默认 `0.75`。

保守修正不会把所有漏词强制附着到邻词。它先只处理源端和目标端都尚未对齐的内容，以同次 Embedding 推理得到的 `mwmf` 作为候选，再按相似度、置信度、位置距离和相邻对齐锚点过滤，并保证每个修正词只参与一条新增关系。

随后，它会检查由已对齐标点或受保护 token 锚定的分句。如果分句已有可用对齐证据，但任一侧仍低于 `min_span_coverage`，服务会在不跨越这些锚点的前提下精炼有歧义的词级链接。整个分句未超过 `max_source_span` 和 `max_target_span` 时，会作为一个显式 `origin: "refined"` 短语组返回。例如 `北方的北方 ↔ The far north of the North` 会表示为 `many-to-many` 组，而不再保留分数虽高、语义却错误的 `北方 ↔ The`。

当整个分句超过任一长度限制时，这两个限制约束的是每个局部候选 Span，而不是直接关闭精炼。受限的单调优化器会综合均值池化 Span 相似度、邻域相对证据和均衡的 token 位置边界；已经完整自洽的模型组会保留，只重写包含漏词、扩展或跨组错连的 Span。因此 `This is a machine translation example. ↔ 这是一个机器翻译示例。` 会得到 `This is ↔ 这是`、`a ↔ 一个`、`machine translation ↔ 机器翻译`，并保留原模型的 `example ↔ 示例`。局部优化的两侧内容 token 总数上限为 64。标点链接仍按原子链接保留，显式短语组覆盖的 token 也不会列入未对齐索引。

两侧覆盖率达到 100% 也不代表 token 角色一定正确。在已有锚点限定的分句内，如果一段连续一对一链接的目标端顺序严格反向，服务会把它收拢成一个局部精炼短语组。例如 `原子/链接 → links/atomic` 会改为 `原子链接 ↔ atomic links`，周围虽然发生短语换序但语义正确的链接仍会保留。跨度更大或目标位置不连续的换序不会被这条保守 crossing 规则重写。

在漏词修复和 Span 精炼之前，确定性的日期规范化规则会处理年月格式换序。英文月份缩写和全称会规范化为 `month:1`～`month:12`，四位年份以及中文的“数字 + 年”Span 会规范化为 `year:YYYY`，中文的“数字 + 月”Span 会规范化为相同月份键。匹配成功后会移除冲突的模型链接，并生成完整的 `origin: "rule"` 关系。例如 `Feb 2025 ↔ 2025年2月` 会得到 `Feb ↔ [2, 月]` 和 `2025 ↔ [2025, 年]`，规则分数均为 `1.0`，且没有未对齐 token。

对于没有硬锚点且存在单侧漏词的句子，保守修复会从置信度不低于 `max(repair.min_confidence, 0.45)` 的单调一对一链接中建立软锚点。两侧已经完整覆盖的区间保持不变，即使其中某条链接低于锚点阈值。较小的单侧缺口只有在池化 Span 分数提高时才会归入左右候选中更好的一侧。与 `min_score_gain` 比较前，增益按剩余分数空间规范化为 `(expanded - base) / (1 - base)`。

包含连续漏词和弱占位链接的较大区间会形成复合修复岛。当相同的规范化增益测试通过、且相邻锚点未达到边界锁定阈值时，修复岛可以吸收一侧锚点。复合修复岛最多组合两个配置的局部 Span，同时仍受 64-token 全局安全上限约束。例如 `Palm oil ↔ 棕榈油` 会形成一个精炼组，`and ingredients derived from palm oils ↔ 及其衍生成分` 会形成第二个精炼组；稳定后缀 `must / be / RSPO / certified` 保留模型链接。

`span-aware` 包含相同的保守行为，并允许在池化 Span Embedding 至少提高 `min_score_gain` 时，把已有组的单侧扩展到相邻且尚未对齐的 token。扩展不能重叠，不能跨越标点或受保护 token 锚点，也不能超过配置的 Span 长度。

响应结构示例（对齐内容仅用于解释结构，真实结果由模型决定）：

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

原子链接的 `origin` 用于区分来源：`model` 是所选 SimAlign 方法的原始关系，`rule` 是受保护 token 精确匹配或日期规范化生成的规则关系，`repaired` 是保守修正补充的关系。

组级结果同样提供 `origin`、`similarity` 和 `confidence`。只包含一种链接来源的组继承该来源，包含多种来源的连通分量为 `mixed`，显式短语 Span 为 `refined`。保守收拢组的 `links` 为空，表示服务有意不声称有歧义的词级对应；`span-aware` 扩展组会保留其基础证据链接。

原子链接的 `similarity` 是上下文词级 Token Embedding 的 `(cosine + 1) / 2`。`confidence` 是尚未用人工标注集校准的 v2 估算值，综合双向概率（`35%`）、相对候选差距（`30%`）、多种匹配方法的一致性（`20%`）和局部 Span/顺序一致性（`15%`）。候选证据采用类似 CSLS 的邻域相对分数，避免 XLM-R 普遍偏高的绝对余弦值主导结果；规则链接的两个值均为 `1.0`。响应中的 `confidence_method` 用于区分当前估算算法与未来基于标注对齐数据校准后的概率。

`start`/`end` 是 Unicode 码点维度的零基、左闭右开字符位置，可在 Python 中直接用 `text[start:end]` 取回原词。服务不会裁掉输入文本首尾的空白，因此位置始终相对于原始请求文本。JavaScript 使用 UTF-16 下标；当文本在目标词之前含有 emoji 等非 BMP 字符时，需要先把码点位置换算成 UTF-16 位置。前端做双语高亮时应优先消费 `alignment_groups`；需要显示未对齐内容时使用两个 `unaligned_*_indices` 字段。显式精炼 Span 即使 `links` 为空也会被视为已对齐。

### 健康检查

- `GET /health/live`：进程存活检查。
- `GET /health/ready`：返回模型名、是否已经加载以及加载模式。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ALIGNER_MODEL` | `xlmr` | SimAlign 模型别名或完整 Hugging Face 模型 ID，如 `microsoft/xlm-align-base` |
| `ALIGNER_TOKEN_TYPE` | `word` | 必须为 `word`；模型子词会均值聚合为展示层词向量 |
| `ALIGNER_LAYER` | `8` | 用于上下文 Token Embedding 的隐藏层编号 |
| `ALIGNER_MATCHING_METHODS` | `mai` | 同时启用 `mwmf`、`inter`、`itermax` |
| `ALIGNER_CONFIDENCE_TEMPERATURE` | `0.1` | 双向置信度 Softmax 使用的正温度参数 |
| `ALIGNER_DEVICE` | `cpu` | 推理设备，如 `cpu` 或 `cuda`；仓库中的 Compose 会覆盖为 `cuda` |
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
