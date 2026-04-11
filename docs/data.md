# Data: Download and Preprocessing

> 本文档介绍如何完成所有数据集的下载与预处理。

---

## 1. 数据集总览

| 数据集 | slug | 实体数 | 关系数 | 文本来源 | 用途 |
|--------|------|--------|--------|----------|------|
| WikiData5M | `wikidata_5m` | 4,594,485 | 822 | Wikipedia 首段描述 | 预训练 / 同域 KGC / T2G / GAD |
| WikiData5M Mini | `wikidata_5m_mini` | ~7,800 | 822 | 同上 (子集) | 工程验证 (dataset / model / loss / trainer) |
| FB15k-237 | `fb15k_237` | 14,541 | 237 | Freebase → Wikipedia 映射 | 跨域 KGC（主实验） |
| CoDEx-L | `codex_l` | 77,951 | 69 | WikiData 标签 + 描述 | 跨域 KGC（主实验） |
| YAGO3-10 | `yago3_10` | 123,182 | 37 | 实体名清洗 | 跨域 KGC（主实验） |
| WN18RR | `wn18rr` | 40,943 | 11 | WordNet synset 定义 + 同义词 | 跨域 KGC（附录） |
| Wiki27K | `wiki27k` | 27,112 | 62 | WikiData 标签 + 描述 | 跨域 KGC（附录） |

---

## 2. 目录结构

数据分为 `raw/`（download 产出）和 `processed/`（preprocess 产出）：

```text
data/wikidata_5m/
├── raw/                              # download 产出
│   ├── train_triples.txt             #   head_id<TAB>rel_id<TAB>tail_id
│   ├── val_triples.txt
│   ├── test_triples.txt
│   ├── entity_texts.tsv              #   entity_id<TAB>text
│   └── relation_texts.tsv            #   relation_id<TAB>text
└── processed/                        # preprocess 产出
    ├── entity2id.tsv                 #   string_id<TAB>int_id
    ├── relation2id.tsv
    ├── train_triples.pt              #   int tensor (N, 3)
    ├── val_triples.pt
    ├── test_triples.pt
    └── text_embeddings/
        ├── entity_emb_sbert_384.pt
        └── relation_emb_sbert_384.pt

data/wikidata_5m_mini/                # 从 wikidata_5m 采样的小型验证子集
├── raw/                              #   格式与 wikidata_5m 完全一致
└── processed/

data/fb15k_237/
├── raw/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   ├── entities.dict
│   ├── relations.dict
│   ├── entity_texts.tsv
│   └── relation_texts.tsv
└── processed/
    └── ...
```

`data/` 目录已在 `.gitignore` 中排除。

---

## 3. 环境变量

项目使用 `.env` 文件管理不想被 Git 追踪的配置（API 地址、密钥等）。

```bash
cp .env.example .env   # 首次使用，从模板创建
```

`.env` 中的变量会在 CLI 启动时通过 `python-dotenv` 加载，
Hydra 配置通过 `${oc.env:VAR,default}` 引用，优先级为：

```text
命令行 Hydra override  >  环境变量 / .env  >  YAML 默认值
```

当前支持的变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `OLLAMA_EMBEDDING_MODEL_NAME` | `qwen3-embedding:8b` | Ollama embedding 模型名 |
| `OLLAMA_EMBEDDING_DIM` | `768` | Ollama embedding 输出维度 (MRL 截断) |
| `OPENAI_API_KEY` | (无) | OpenAI API 密钥 |

---

## 4. Download：下载原始数据

download 负责获取图结构和实体/关系文本，写入 `raw/`。不需要 GPU。

### 4.1 调用方式

```bash
# 预训练数据集
uv run python -m riemannfm.cli.download data=wikidata_5m

# 下游 KGC 数据集
uv run python -m riemannfm.cli.download data=fb15k_237
uv run python -m riemannfm.cli.download data=wn18rr
uv run python -m riemannfm.cli.download data=codex_l
uv run python -m riemannfm.cli.download data=yago3_10
uv run python -m riemannfm.cli.download data=wiki27k              # Google Drive via gdown

# 下载全部数据集
uv run python -m riemannfm.cli.download download.all=true

# 强制重新下载
uv run python -m riemannfm.cli.download data=wn18rr download.force=true
```

### 4.2 Hydra 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data=` | `wikidata_5m` | 选择数据集（对应 `configs/data/*.yaml`） |
| `download.force` | `false` | 强制重新下载 |
| `download.all` | `false` | 下载全部数据集 |

### 4.3 三阶段流水线

**Stage 1 — 下载图结构** → `raw/{train,val,test}_triples.txt`

| 数据集 | 来源 |
|--------|------|
| WikiData5M | Dropbox `wikidata5m_transductive.tar.gz` |
| FB15k-237, WN18RR | GitHub `villmow/datasets_knowledge_embedding` 逐文件下载 |
| CoDEx-L | GitHub `tsafavi/codex` 逐 split 下载 |
| YAGO3-10 | GitHub `DeepGraphLearning/KnowledgeGraphEmbedding` 逐文件下载 |
| Wiki27K | Google Drive (THU-KEG/PKGC, gdown) |

跳过条件：`raw/train.txt` 或 `raw/train_triples.txt` 已存在。

**Stage 2 — 提取实体文本** → `raw/entity_texts.tsv`（`entity_id<TAB>text`）

| 数据集 | 策略 |
|--------|------|
| WikiData5M | 下载 `wikidata5m_text.txt.gz`，取每个实体的首句 |
| FB15k-237 | 下载 `entity2textlong.txt` 映射文件 |
| WN18RR | NLTK WordNet synset 定义 + 同义词 |
| CoDEx-L, Wiki27K | WikiData 实体描述 JSON |
| YAGO3-10 | 实体名清洗（`Albert_Einstein` → `Albert Einstein`） |

**Stage 3 — 提取关系文本** → `raw/relation_texts.tsv`（`relation_id<TAB>text`）

| 数据集 | 策略 |
|--------|------|
| WikiData5M | 下载 `wikidata5m_alias.tar.gz`，解析 relation 别名 |
| WN18RR | `_hypernym` → `hypernym` |
| FB15k-237 | `/film/film/genre` → `film film genre` |
| CoDEx-L | `relations.dict` |

每个阶段结束后自动运行 `validate_raw()` 检查完整性。

---

## 5. Build Mini：构建工程验证数据集

从完整 WikiData5M 中采样一个小型子集 (`wikidata_5m_mini`)，格式与原始数据集完全一致，
可直接用于测试 dataset、model、loss、trainer 等组件。

### 5.1 采样策略

采用 **relation-aware sampling**：

1. 对全部 822 种 relation，每种至少采样若干 triples → 保证 100% relation 覆盖
2. 收集 phase 1 涉及的所有实体，如未达到目标数则补充更多 triples
3. 对已采样实体集内的所有边做 densify（不增加实体，只增加边密度）
4. 随机 shuffle 后 80/10/10 分为 train/val/test
5. 过滤 `entity_texts.tsv` 和 `relation_texts.tsv` 到子集

**输出规模**: ~7,800 entities, 822 relations, ~22,000 triples (deterministic, seed=42)

### 5.2 调用方式

前提：已完成 WikiData5M 下载 (`data/wikidata_5m/raw/` 存在)。

```bash
uv run python -m riemannfm.cli.preprocess data=wikidata_5m preprocess.build_mini=true

# 强制重新构建
uv run python -m riemannfm.cli.preprocess data=wikidata_5m preprocess.build_mini=true preprocess.force=true
```

产出写入 `data/wikidata_5m_mini/raw/`，格式与 `data/wikidata_5m/raw/` 完全一致。

---

## 6. Preprocess：ID 映射 + 文本嵌入

preprocess 读取 `raw/` 产出两类文件到 `processed/`：

1. **ID 映射**: 字符串 ID → 连续整数，生成 `entity2id.tsv`、`relation2id.tsv` 和 `{split}_triples.pt`
2. **文本嵌入**: 对 entity/relation texts 编码为向量，保存为 `.pt` 文件

### 6.1 调用方式

```bash
# 预训练数据集
uv run python -m riemannfm.cli.preprocess data=wikidata_5m_mini embedding=sbert
uv run python -m riemannfm.cli.preprocess data=wikidata_5m embedding=sbert

# 下游 KGC 数据集
uv run python -m riemannfm.cli.preprocess data=fb15k_237 embedding=sbert
uv run python -m riemannfm.cli.preprocess data=wn18rr embedding=sbert
uv run python -m riemannfm.cli.preprocess data=codex_l embedding=sbert
uv run python -m riemannfm.cli.preprocess data=yago3_10 embedding=sbert
uv run python -m riemannfm.cli.preprocess data=wiki27k embedding=sbert

# 只做 ID 映射（不做文本嵌入）
uv run python -m riemannfm.cli.preprocess data=wikidata_5m embedding=none

# 强制重新处理
uv run python -m riemannfm.cli.preprocess data=wikidata_5m embedding=sbert preprocess.force=true
```

### 6.2 Hydra 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data=` | `wikidata_5m` | 数据集 |
| `embedding=` | `qwen3` | 编码器（对应 `configs/embedding/*.yaml`） |
| `preprocess.batch_size` | `256` | 编码 batch size |
| `preprocess.device` | `cuda` | 编码设备 |
| `preprocess.force` | `false` | 强制重新处理 |
| `preprocess.build_mini` | `false` | 构建 mini 验证集而非做 preprocess |

### 6.3 Embedding 配置

| 配置 | Provider | Ollama 模型名 | HuggingFace 模型名 | d_c | 用途 |
|------|----------|--------------|-------------------|-----|------|
| `embedding=sbert` | huggingface | `all-minilm:l6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 轻量 baseline / 快速调试 |
| `embedding=nomic` | ollama | `nomic-embed-text` | `nomic-ai/nomic-embed-text-v1.5` | 768 | wikidata_5m_mini 默认 |
| `embedding=qwen3` | ollama | `qwen3-embedding:8b` | `Qwen/Qwen3-Embedding-8B` | 768 (MRL 截断) | 完整预训练 / 最终论文 |
| `embedding=none` | — | — | — | 0 | 消融对照（纯结构，无文本） |

不同编码器的嵌入文件命名为 `{entity|relation}_emb_{encoder}_{dim}.pt`，互不干扰，支持 ablation。

Ollama 配置通过 `.env` 覆盖（见第 3 节），也可在命令行直接 override：

```bash
uv run python -m riemannfm.cli.preprocess data=wikidata_5m embedding=qwen3 \
    embedding.api_base=http://gpu-server:11434
```

---

## 7. 数据验证

下载后自动运行验证。也可手动执行：

```bash
uv run python -m riemannfm.data.pipeline.validate data/wikidata_5m
uv run python -m riemannfm.data.pipeline.validate data/wikidata_5m_mini
```

检查项：

1. 三元组文件格式和数量
2. Entity/relation text 覆盖率
3. Split 之间无泄漏 (train ∩ val = ∅, etc.)
4. Relation 分布统计

---

## 8. 典型工作流

### 快速验证 (mini 数据集)

```bash
# 1. 下载 WikiData5M 原始数据
uv run python -m riemannfm.cli.download data=wikidata_5m

# 2. 从完整数据采样 mini 子集
uv run python -m riemannfm.cli.preprocess data=wikidata_5m preprocess.build_mini=true

# 3. 对 mini 做预处理（ID 映射 + 文本嵌入）
uv run python -m riemannfm.cli.preprocess data=wikidata_5m_mini embedding=sbert
```

### 全量下载 + 预处理

```bash
# 1. 下载全部数据集
uv run python -m riemannfm.cli.download download.all=true

# 2. 预处理 WikiData5M
uv run python -m riemannfm.cli.preprocess data=wikidata_5m embedding=qwen3

# 3. 追加其他编码器（ID 映射不重复，只追加嵌入文件）
uv run python -m riemannfm.cli.preprocess data=wikidata_5m embedding=sbert

# 4. 预处理全部下游数据集
uv run python -m riemannfm.cli.preprocess data=fb15k_237 embedding=qwen3
uv run python -m riemannfm.cli.preprocess data=wn18rr embedding=qwen3
uv run python -m riemannfm.cli.preprocess data=codex_l embedding=qwen3
uv run python -m riemannfm.cli.preprocess data=yago3_10 embedding=qwen3
uv run python -m riemannfm.cli.preprocess data=wiki27k embedding=qwen3
```

> 预训练和微调命令见 [cli.md](cli.md)。
