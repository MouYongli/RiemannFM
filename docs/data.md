# Data: Download and Preprocessing

> 本文档介绍如何完成所有数据集的下载与预处理。

---

## 1. 数据集总览

| 数据集 | slug | 实体数 | 关系数 | 文本来源 | 用途 |
|--------|------|--------|--------|----------|------|
| WikiData5M | `wikidata_5m` | 4,594,485 | 822 | Wikipedia 标题 + 首段描述 | 预训练 / 同域 KGC / T2G / GAD |
| FB15k-237 | `fb15k_237` | 14,541 | 237 | Freebase → Wikipedia 映射 | 跨域 KGC |
| WN18RR | `wn18rr` | 40,943 | 11 | WordNet synset 定义 + 同义词 | 跨域 KGC |
| CoDEx-L | `codex_l` | 77,951 | 69 | WikiData 标签 + 描述 | 跨域 KGC |
| Wiki27K | `wiki27k` | ~27,000 | 214 | WikiData 标签 + 描述 | 跨域 KGC |
| YAGO3-10 | `yago3_10` | 123,182 | 37 | Wikipedia 映射 | 跨域 KGC |

---

## 2. 目录结构

数据分为 `raw/`（download 产出）和 `processed/`（preprocess 产出）：

```text
data/fb15k_237/
├── raw/                              # make download
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   ├── entities.dict
│   ├── relations.dict
│   └── entity_texts.tsv
└── processed/                        # make preprocess（后续实现）
    └── text_embeddings/
        ├── entity_emb_sbert_384.pt
        └── entity_emb_xlm_roberta_1024.pt

data/wikidata_5m/
├── raw/
│   ├── train_triples.txt
│   ├── val_triples.txt
│   ├── test_triples.txt
│   ├── entities.json
│   └── entity_texts.tsv
└── processed/
    └── text_embeddings/
        └── entity_emb_sbert_384.pt
```

`data/` 目录已在 `.gitignore` 中排除。

---

## 3. Download：下载原始数据

download 负责获取图结构和实体文本，写入 `raw/`。不需要 GPU，不需要 text_encoder。

conda 用户请先 `conda activate riemannfm`，uv 用户直接执行。

三种等价的调用方式：

```bash
# 方式一：make（推荐）
make download ARGS="data=fb15k237"
make download ARGS="download.all=true"
make download ARGS="data=wn18rr download.force=true"

# 方式二：shell 脚本
bash scripts/download.sh data=fb15k237
bash scripts/download.sh download.all=true

# 方式三：python 模块
python -m riemannfm.cli.download data=fb15k237
python -m riemannfm.cli.download download.all=true
```

### Hydra 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data=` | `wikidata_5m` | 选择数据集（对应 `configs/data/*.yaml`） |
| `download.force` | `false` | 强制重新下载 |
| `download.all` | `false` | 下载全部 6 个数据集 |

### 两阶段流水线

**Stage 1：下载图结构** → `raw/{train,valid,test}.txt` + `entities.dict` + `relations.dict`

- FB15k-237, WN18RR: 从 `villmow/datasets_knowledge_embedding` 下载 tar.gz
- WikiData5M: 优先从 HuggingFace `datasets` 库下载，失败时提示手动获取
- CoDEx-L: 从 `tsafavi/codex` GitHub 仓库下载
- Wiki27K, YAGO3-10: 需手动放置文件（暂无自动下载源）

跳过条件：`raw/train.txt`（或 `raw/train_triples.txt`）已存在。

**Stage 2：提取实体文本** → `raw/entity_texts.tsv`（`entity_id\ttext`）

| 数据集 | 提取方式 |
|--------|----------|
| WikiData5M | 从 `entities.json` 拼接 label + description |
| FB15k-237, YAGO3-10 | 下载 `entity2textlong.txt`（StAR_KGC 项目） |
| WN18RR | 使用 `nltk.corpus.wordnet` 提取 synset 定义和同义词 |
| CoDEx-L, Wiki27K | 下载 WikiData 实体描述 JSON |

跳过条件：`raw/entity_texts.tsv` 已存在。

---

## 4. Preprocess：ID 映射 + 文本嵌入

preprocess 做两件事：
1. **ID 映射**：字符串 ID (Q-id/P-id) → 连续整数，生成 `entity2id.tsv`、`relation2id.tsv` 和 int tensor 三元组
2. **文本嵌入**：用指定编码器对 `entity_texts.tsv` / `relation_texts.tsv` 编码

```bash
# HuggingFace 本地编码器
make preprocess ARGS="data=wikidata_5m embedding=sbert"
make preprocess ARGS="data=fb15k237 embedding=xlm_roberta"

# Ollama API（需本地运行 Ollama 服务）
make preprocess ARGS="data=wikidata_5m embedding=qwen3_embed"

# 只做 ID 映射，不做文本嵌入
make preprocess ARGS="data=wikidata_5m embedding=none"

# 强制重新处理
make preprocess ARGS="data=wikidata_5m embedding=sbert preprocess.force=true"
```

### Hydra 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data=` | `wikidata_5m` | 数据集 |
| `embedding=` | `sbert` | 编码器（对应 `configs/embedding/*.yaml`） |
| `preprocess.batch_size` | `256` | 编码 batch size |
| `preprocess.device` | `cuda` | 编码设备 |
| `preprocess.force` | `false` | 强制重新处理 |

### Embedding provider

| Provider | 说明 | 配置 |
|----------|------|------|
| `huggingface` | 本地 transformer 模型 | 需要 GPU，model_name 为 HuggingFace 路径 |
| `ollama` | Ollama 的 OpenAI 兼容接口 | 需要运行 Ollama 服务，配置 api_base |
| `openai` | OpenAI API | 需要 OPENAI_API_KEY 环境变量 |
| `none` | 跳过文本嵌入 | 只做 ID 映射 |

不同编码器的嵌入文件命名为 `{entity|relation}_emb_{encoder}_{dim}.pt`，互不干扰，支持 ablation。

---

## 5. Hydra 配置集成

数据配置通过 Hydra 插值自动获取 embedding 参数：

```yaml
# configs/data/fb15k237.yaml
slug: fb15k_237
dataset: kg_benchmark
data_dir: data/fb15k_237
num_edge_types: 238
text_source: wikipedia_mapping
text_encoder: ${embedding.model_name}
dim_text_emb: ${embedding.output_dim}
```

可用的 embedding 配置（`configs/embedding/`）：

| 配置文件 | Provider | 模型 | 输出维度 |
|----------|----------|------|----------|
| `embedding/sbert.yaml` | huggingface | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `embedding/xlm_roberta.yaml` | huggingface | `xlm-roberta-large` | 1024 |
| `embedding/qwen3_embed.yaml` | ollama | `qwen3-embedding` | 4096 |
| `embedding/none.yaml` | none | — | 0 |

---

## 6. 训练时的数据检查

预训练脚本在构建数据集前自动检查 `raw/` 下是否有图文件，缺失时报错：

```text
FileNotFoundError: Dataset not found at data/fb15k_237/raw/.
Run `make download ARGS='data=fb15k237'` first.
```

---

## 7. 典型工作流

```bash
# 1. 下载全部原始数据（一次性）
make download ARGS="download.all=true"

# 2. 预处理：ID 映射 + 文本嵌入
make preprocess ARGS="data=wikidata_5m embedding=sbert"
make preprocess ARGS="data=fb15k237 embedding=sbert"

# 追加不同编码器（ID 映射不会重复，只追加嵌入文件）
make preprocess ARGS="data=wikidata_5m embedding=xlm_roberta"
make preprocess ARGS="data=wikidata_5m embedding=qwen3_embed"

# 3. 训练
make pretrain ARGS="data=wikidata_5m embedding=sbert"
make finetune ARGS="data=fb15k237 embedding=sbert"
```
