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

## 2. 快速上手

conda 用户请先 `conda activate riemannfm`，uv 用户直接执行。

有三种等价的调用方式：

### 方式一：make（推荐）

```bash
# 下载单个数据集 + 用 SBERT 预计算文本嵌入
make download ARGS="data=fb15k237 text_encoder=sbert"

# 下载全部数据集
make download ARGS="download.all=true text_encoder=sbert"

# 用不同编码器生成额外嵌入（图结构和文本不会重复下载）
make download ARGS="data=fb15k237 text_encoder=xlm_roberta"

# 指定 GPU 和 batch size
make download ARGS="data=wikidata_5m text_encoder=qwen3_embed download.device=cuda:0 download.batch_size=128"

# 强制重新下载和编码
make download ARGS="data=wn18rr text_encoder=sbert download.force=true"
```

### 方式二：shell 脚本

```bash
bash scripts/download.sh data=fb15k237 text_encoder=sbert
bash scripts/download.sh download.all=true text_encoder=xlm_roberta
```

### 方式三：python 模块

```bash
python -m riemannfm.cli.download data=fb15k237 text_encoder=sbert
python -m riemannfm.cli.download download.all=true text_encoder=sbert
```

三种方式参数完全相同，都是 Hydra override 语法。

---

## 3. Hydra 参数

下载参数定义在 `configs/download/default.yaml`，通过 Hydra override 修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data=` | `wikidata_5m` | 选择数据集（对应 `configs/data/*.yaml`） |
| `text_encoder=` | `sbert` | 选择文本编码器（对应 `configs/text_encoder/*.yaml`） |
| `download.batch_size` | `256` | 编码 batch size |
| `download.device` | `cuda` | 编码设备（`cuda`, `cpu`, `cuda:N`） |
| `download.force` | `false` | 强制重新下载和编码 |
| `download.all` | `false` | 下载全部 6 个数据集 |

---

## 4. 三阶段流水线

每个数据集按顺序执行三个阶段，各阶段独立检查输出文件是否存在：

### Stage 1: 下载图结构

下载三元组（train/valid/test splits）和实体/关系映射。

- **FB15k-237, WN18RR**: 从 `villmow/datasets_knowledge_embedding` 下载 tar.gz
- **WikiData5M**: 优先从 HuggingFace `datasets` 库下载，失败时提示手动获取
- **CoDEx-L**: 从 `tsafavi/codex` GitHub 仓库下载
- **Wiki27K, YAGO3-10**: 需手动放置文件（暂无自动下载源）

跳过条件：`train.txt`（或 `train_triples.txt`）已存在。

### Stage 2: 提取实体文本

为每个实体生成文本描述，保存为 `entity_texts.tsv`（`entity_id\ttext`）。

| 数据集 | 提取方式 |
|--------|----------|
| WikiData5M | 从 `entities.json` 拼接 label + description |
| FB15k-237, YAGO3-10 | 下载 `entity2textlong.txt`（StAR_KGC 项目） |
| WN18RR | 使用 `nltk.corpus.wordnet` 提取 synset 定义和同义词 |
| CoDEx-L, Wiki27K | 下载 WikiData 实体描述 JSON |

跳过条件：`entity_texts.tsv` 已存在。

### Stage 3: 预计算文本嵌入

用指定的文本编码器对 `entity_texts.tsv` 中所有实体编码，保存为 `.pt` 文件。

嵌入文件命名：`entity_emb_{encoder}_{dim}.pt`

不同编码器/维度的嵌入共存于同一目录，支持 ablation 实验：

```text
text_embeddings/
├── entity_emb_sbert_384.pt
├── entity_emb_xlm_roberta_1024.pt
└── entity_emb_qwen3_embed_4096.pt
```

跳过条件：对应的 `.pt` 文件已存在。

---

## 5. 目录结构

所有数据集统一放在 `data/` 下（flat 结构）：

```text
data/
├── wikidata_5m/
│   ├── train_triples.txt          # Stage 1
│   ├── val_triples.txt
│   ├── test_triples.txt
│   ├── entities.json
│   ├── entity_texts.tsv           # Stage 2
│   └── text_embeddings/           # Stage 3
│       └── entity_emb_sbert_384.pt
├── fb15k_237/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   ├── entities.dict
│   ├── relations.dict
│   ├── entity_texts.tsv
│   └── text_embeddings/
│       ├── entity_emb_sbert_384.pt
│       └── entity_emb_xlm_roberta_1024.pt
├── wn18rr/
├── codex_l/
├── wiki27k/
└── yago3_10/
```

`data/` 目录已在 `.gitignore` 中排除，不会被提交。

---

## 6. Hydra 配置集成

数据配置通过 Hydra 插值自动获取 `text_encoder` 和 `dim_text_emb`：

```yaml
# configs/data/fb15k237.yaml
slug: fb15k_237
dataset: kg_benchmark
data_dir: data/fb15k_237
num_edge_types: 238
text_source: wikipedia_mapping
text_encoder: ${text_encoder.model_name}    # 来自 configs/text_encoder/*.yaml
dim_text_emb: ${text_encoder.output_dim}    # 自动匹配
```

可用的文本编码器配置：

| 配置文件 | 模型 | 输出维度 |
|----------|------|----------|
| `text_encoder/sbert.yaml` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `text_encoder/xlm_roberta.yaml` | `xlm-roberta-large` | 1024 |
| `text_encoder/qwen3_embed.yaml` | `Alibaba-NLP/Qwen3-Embedding-8B` | 4096 |
| `text_encoder/none.yaml` | — | 0 |

训练时通过 Hydra override 切换编码器：

```bash
make pretrain ARGS="text_encoder=xlm_roberta data=fb15k237"
```

---

## 7. 训练时的数据检查

预训练脚本在构建数据集前会自动检查：

1. 图结构文件是否存在（`train.txt` / `train_triples.txt`）
2. 如果配置了 `text_encoder` 和 `dim_text_emb`，检查对应的嵌入文件是否存在（仅警告，不阻塞）

缺失图结构时会报错并提示：

```text
FileNotFoundError: Dataset not found at data/fb15k_237.
Run `make download ARGS='data=fb15k237'` first.
```

---

## 8. Ablation 工作流

不同文本编码器的嵌入文件互不干扰，可按需预计算：

```bash
# 先用默认编码器下载全部数据
make download ARGS="download.all=true text_encoder=sbert"

# 后续只需追加不同编码器的嵌入（图和文本不会重复下载）
make download ARGS="download.all=true text_encoder=xlm_roberta"
make download ARGS="download.all=true text_encoder=qwen3_embed download.device=cuda:1"
```

对应的实验配置只需切换 `text_encoder`，数据路径自动解析：

```bash
# sbert ablation
make finetune ARGS="data=fb15k237 text_encoder=sbert"

# xlm-roberta ablation
make finetune ARGS="data=fb15k237 text_encoder=xlm_roberta"

# no text ablation
make finetune ARGS="data=fb15k237 text_encoder=none"
```
