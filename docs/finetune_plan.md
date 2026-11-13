# KGC Finetune + Evaluate 实现计划

## Context

Pretrain 基础设施已完成，但 4 个下游 task (kgc_lp, kgc_rp, t2g, gad) 的 finetune 和 evaluate 均为空 placeholder。用户决定先实现 **KGC Link Prediction + Relation Prediction**，T2G/GAD 后续再做。Finetune 策略通过 config 控制（freeze backbone vs full finetune）。

当前缺失：task heads、finetune LightningModule、KGC dataset/datamodule、评估 metrics、CLI、configs。

---

## 实现范围

### 新建文件

| 文件 | 用途 |
|------|------|
| `src/riemannfm/tasks/kgc.py` | KGC scoring heads (LP + RP) |
| `src/riemannfm/tasks/metrics.py` | MRR, Hits@k, Accuracy 等 metrics |
| `src/riemannfm/data/datasets/kgc_dataset.py` | KGC finetune/eval dataset（triple-level） |
| `src/riemannfm/models/finetune_module.py` | KGC finetune LightningModule |
| `configs/task/kgc_lp.yaml` | LP task config |
| `configs/task/kgc_rp.yaml` | RP task config |
| `configs/training/finetune.yaml` | Finetune training config |
| `configs/experiment/kgc_fb15k237.yaml` | FB15K-237 实验 recipe（示例） |

### 修改文件

| 文件 | 改动 |
|------|------|
| `src/riemannfm/cli/finetune.py` | 替换 placeholder，实现 Hydra CLI |
| `src/riemannfm/cli/evaluate.py` | 替换 placeholder，实现评估 CLI |
| `src/riemannfm/tasks/__init__.py` | 导出 task heads |

---

## Step 1: KGC Scoring Heads — `src/riemannfm/tasks/kgc.py`

### Link Prediction Head

思路：利用 pretrained entity embeddings + backbone 的 node hidden states，对 (h, r, t) 三元组打分。

```python
class RiemannFMKGCLinkHead(nn.Module):
    """Score (h, r, t) triples for link prediction.
    
    支持两种 scoring 模式:
    1. manifold_dist: score = -d_M(h + r, t)  (流形测地距离)
    2. bilinear: score = h^T W_r t  (每 relation 一个变换)
    """
```

- 输入: `h_emb (B, D)`, `r_id (B,)`, `t_emb (B, D)` — 从 entity_emb 查出的流形坐标
- 输出: `score (B,)` — 标量分数
- 复用: `RiemannFMProductManifold.geodesic_distance()` 做 manifold-based scoring

### Relation Prediction Head

```python
class RiemannFMKGCRelHead(nn.Module):
    """Classify relation type given (h, t) pair.
    
    score_k = MLP([h; t; h*t]) -> (K,) logits
    """
```

- 输入: `h_emb (B, D)`, `t_emb (B, D)`
- 输出: `logits (B, K)` — K 个 relation 的分数
- Loss: CrossEntropyLoss

---

## Step 2: KGC Dataset — `src/riemannfm/data/datasets/kgc_dataset.py`

**不同于 pretrain 的子图采样**，KGC 用 triple-level 数据：

```python
class RiemannFMKGCDataset(Dataset):
    """Triple-level dataset for KGC finetune and evaluation.
    
    每个 sample 返回 (head_id, rel_id, tail_id, label).
    Training: 正样本 + 负采样 (corrupt head 或 tail).
    Evaluation: 返回正样本，ranking 在 eval loop 中进行.
    """
```

- 加载 `{split}_triples.pt` — shape (num_triples, 3)
- Training: 对每个正三元组，随机 corrupt head/tail 生成负样本
- Eval: 返回 (h, r, t)，scoring 所有 candidate entity
- 需要构建 `all_true_triples` set 用于 filtered ranking

### Collator

简单的 stack collator，不需要子图 padding：

```python
def kgc_collate(batch) -> dict:
    # {"head": (B,), "rel": (B,), "tail": (B,), "label": (B,)}
```

---

## Step 3: Finetune LightningModule — `src/riemannfm/models/finetune_module.py`

```python
class RiemannFMKGCModule(L.LightningModule):
    """KGC fine-tuning module (LP and RP).
    
    加载 pretrained checkpoint -> 添加 task head -> finetune.
    
    Config 控制:
    - backbone_frozen: bool — 冻结 backbone + entity_emb
    - backbone_lr: float — backbone 学习率 (0 = frozen)
    - head_lr: float — task head 学习率
    """
```

**关键设计**:

1. **从 pretrained checkpoint 加载**:
   - 加载 `RiemannFMPretrainModule` checkpoint
   - 提取 `manifold`, `entity_emb`, `model` (RieFormer backbone)
   - 不需要 `flow`, `loss_fn`（pretrain 专用）

2. **Freeze 控制** (通过 config):
   - `backbone_frozen=true`: 冻结整个 backbone + entity_emb，只训练 head
   - `backbone_frozen=false`: backbone 用 `backbone_lr`，head 用 `head_lr`

3. **Training step (LP)**:
   - Lookup entity embeddings for h, t
   - Score (h, r, t) via KGCLinkHead
   - Loss: MarginRankingLoss 或 BCE (正样本 vs 负样本)

4. **Training step (RP)**:
   - Lookup entity embeddings for h, t
   - Classify relation via KGCRelHead
   - Loss: CrossEntropyLoss

5. **Validation/Test step**:
   - LP: 对每个 (h, r, ?) 或 (?, r, t)，score 所有 entity，计算 filtered ranking
   - RP: 对每个 (h, ?, t)，score 所有 relation，计算 accuracy

6. **Optimizer**: 
   - 两组 param groups: `[{backbone, lr=backbone_lr}, {head, lr=head_lr}]`
   - Scheduler: linear warmup + cosine (复用 pretrain 的 pattern)

### Wikidata5M 的特殊情况

Wikidata5M transductive 的 KGC **不需要 finetune**（pretrain 就在此图上训练）。直接用 pretrained model evaluate：
- 加载 pretrained checkpoint
- 用 val/test triples 做 filtered ranking
- 报告 MRR, Hits@1/3/10

---

## Step 4: Evaluation Metrics — `src/riemannfm/tasks/metrics.py`

```python
def compute_ranking_metrics(
    ranks: Tensor,  # (N,) — 每个 query 的正确 entity rank (1-based)
) -> dict[str, float]:
    """MRR, Hits@1, Hits@3, Hits@10."""

def filtered_ranking(
    scores: Tensor,        # (num_entities,)
    target_idx: int,
    true_targets: set[int],  # 需要过滤的已知正样本
) -> int:
    """Filtered rank: 正确答案在去掉其他正样本后的排名."""
```

---

## Step 5: CLI — `src/riemannfm/cli/finetune.py` & `evaluate.py`

### finetune.py

```python
@hydra.main(config_path="../../configs", config_name="config")
def main(cfg):
    # 1. Setup data (KGC dataset)
    # 2. Load pretrained checkpoint
    # 3. Build KGC module (LP or RP)
    # 4. Lightning Trainer.fit()
    # 5. Save best checkpoint
```

### evaluate.py

```python
@hydra.main(config_path="../../configs", config_name="config")
def main(cfg):
    # 1. Load model (pretrained or finetuned checkpoint)
    # 2. Load test triples + all_true_triples (for filtering)
    # 3. For each test triple:
    #    - Score all candidate entities (batched)
    #    - Compute filtered rank
    # 4. Aggregate metrics, log to wandb
```

**注意**: evaluation 需要 score 所有 entity（Wikidata5M 有 460 万），需要分批处理。

---

## Step 6: Configs

### `configs/task/kgc_lp.yaml`
```yaml
task: kgc_lp
head:
  _target_: riemannfm.tasks.kgc.RiemannFMKGCLinkHead
  scoring: manifold_dist  # or bilinear
neg_samples: 256
loss: margin_ranking  # or bce
margin: 1.0
eval_batch_size: 256  # candidates per batch during eval
```

### `configs/task/kgc_rp.yaml`
```yaml
task: kgc_rp
head:
  _target_: riemannfm.tasks.kgc.RiemannFMKGCRelHead
loss: cross_entropy
```

### `configs/training/finetune.yaml`
```yaml
backbone_frozen: false
backbone_lr: 1e-5
head_lr: 1e-3
warmup_steps: 500
max_steps: 10000
batch_size: 512
weight_decay: 0.01
max_grad_norm: 1.0
use_riemannian_optim: true
```

### `configs/experiment/kgc_fb15k237.yaml`
```yaml
# @package _global_
defaults:
  - override /data: fb15k_237
  - override /task: kgc_lp
  - override /training: finetune
  - _self_

pretrain_ckpt: ???  # path to pretrained checkpoint
seed: 42
```

---

## Step 7: 实现顺序

1. **metrics** — 独立模块，可先写 + 单测
2. **kgc.py (task heads)** — 依赖 manifold，可单测
3. **kgc_dataset.py** — 依赖已有的 triples.pt，可单测
4. **finetune_module.py** — 依赖 1-3
5. **configs** — yaml 文件
6. **cli/finetune.py + evaluate.py** — 依赖 4-5
7. **集成测试** — smoke test on mini dataset

---

## Verification

1. **单元测试**: `make test` — metrics、scoring heads、dataset 的独立测试
2. **Smoke test (finetune)**:
   ```bash
   uv run python -m riemannfm.cli.finetune \
     data=wikidata_5m_mini task=kgc_lp training=finetune \
     pretrain_ckpt=<path> training.max_steps=10
   ```
3. **Smoke test (evaluate)**:
   ```bash
   uv run python -m riemannfm.cli.evaluate \
     data=wikidata_5m_mini task=kgc_lp \
     ckpt_path=<path>
   ```
4. **完整评估**: FB15K-237 test set, 报告 MRR + Hits@1/3/10
5. `make lint && make typecheck && make test`
