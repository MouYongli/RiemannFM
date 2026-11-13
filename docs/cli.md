# CLI 用法参考

所有 CLI 命令基于 [Hydra](https://hydra.cc/)，配置文件位于 `configs/`。

---

## 1. 数据下载

```bash
uv run python -m riemannfm.cli.download data=wikidata_5m
```

## 2. Build Mini (工程验证子集)

```bash
uv run python -m riemannfm.cli.preprocess data=wikidata_5m preprocess.build_mini=true
```

## 3. 数据预处理 (文本嵌入)

```bash
uv run python -m riemannfm.cli.preprocess data=wikidata_5m_mini embedding=qwen3
uv run python -m riemannfm.cli.preprocess data=wikidata_5m embedding=qwen3
```

---

## 4. 预训练

### 硬件约束 (1x H100 80GB)

- `max_nodes=128`（256 会 OOM）
- `batch_size=4`（8 会 OOM）
- `gradient_accumulation_steps=8`（effective batch = 32）

### 模型规模（实测参数量）

| Config | Layers | node_dim | edge_dim | Heads | Arch Params | Total (wiki5m) |
|--------|--------|----------|----------|-------|-------------|----------------|
| small  | 6      | 384      | 128      | 6     | 22.5M       | 23.6M (mini) / 473M (full) |
| base   | 10     | 512      | 192      | 8     | 66.9M       | 517.7M |
| large  | 14     | 768      | 256      | 12    | 200.7M      | 651.4M |

Entity embedding: 4,594,485 x 98 = 450.3M params (wikidata_5m), 7,833 x 98 = 0.8M (mini).

---

### Phase 0: Smoke Test

验证代码能跑通、loss 三路都在下降。mini 数据 + small 模型，~5 min。

```bash
uv run python -m riemannfm.cli.pretrain experiment=smoke_test
```

| 参数 | 值 |
|------|----|
| Model | small (6L/384d/128e, 22.5M) |
| Data | wikidata_5m_mini (7,833 entities) |
| max_nodes | 64 |
| max_steps | 1,000 |
| warmup_steps | 200 |
| val_check_interval | 500 |
| batch_size x accum | 8 x 1 = 8 |
| lambda_disc / mu_align / tau | 10.0 / 0.5 / 0.07 |

通过标准：无 NaN/Inf，L_disc < 0.5，L_align 有下降趋势，kappa 变化 > 0.01。

---

### Phase 1: Validation Run

全量数据 + base 模型，10k steps，~8h。

```bash
uv run python -m riemannfm.cli.pretrain experiment=validation_run
```

| 参数 | 值 |
|------|----|
| Model | base (10L/512d/192e, 66.9M) |
| Data | wikidata_5m (4.6M entities) |
| max_nodes | 128 |
| max_steps | 10,000 |
| warmup_steps | 500 |
| val_check_interval | 500 |
| batch_size x accum | 4 x 8 = 32 |
| lambda_disc / mu_align / tau | 10.0 / 0.5 / 0.07 |

通过标准：val_loss 稳定不回升，L_disc < 0.6，curvature 变化 > 0.05。

---

### Phase 2: HP Search（两阶段）

#### Stage 1: 广搜（small 模型，20 trials x 10k steps）

在 small 模型上搜索 loss 权重 + 学习率，找到让三路 loss 均有效下降的组合。

单 GPU（~4 天）：
```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m riemannfm.cli.pretrain experiment=pretrain_search sweep=pretrain --multirun
```

多 GPU 并行（3 GPU ~1.3 天）——3 个终端同时启动，共享 Optuna SQLite study：
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=1 uv run python -m riemannfm.cli.pretrain experiment=pretrain_search sweep=pretrain --multirun
# Terminal 2
CUDA_VISIBLE_DEVICES=2 uv run python -m riemannfm.cli.pretrain experiment=pretrain_search sweep=pretrain --multirun
# Terminal 3
CUDA_VISIBLE_DEVICES=3 uv run python -m riemannfm.cli.pretrain experiment=pretrain_search sweep=pretrain --multirun
```

三个进程共享 `hp_search_s1.db`，Optuna TPE 自动同步采样结果，总共仍是 20 trials。

| 参数 | 值 |
|------|----|
| Model | small (6L/384d/128e, 22.5M) |
| Data | wikidata_5m |
| max_nodes | 128 |
| max_steps | 10,000 |
| warmup_steps | 1,000 |
| val_check_interval | 2,000 |
| batch_size x accum | 4 x 8 = 32 |
| Sampler | TPE, 20 trials, minimize val_loss |

搜索空间（8 个参数）：

| 参数 | 范围 | 说明 |
|------|------|------|
| `training.lr` | 3e-5, 1e-4, 3e-4, 1e-3 | 主学习率，跨 1.5 个数量级 |
| `training.curvature_lr` | 1e-5, 5e-5, 1e-4, 5e-4 | 曲率学习率，与 main lr 解耦 |
| `training.lambda_disc` | 5, 10, 20, 50 | 离散 flow loss 权重（补偿 L_cont/L_disc ~40x 量级差） |
| `training.mu_align` | 0.1, 0.5, 1.0, 2.0 | 对比对齐 loss 权重 |
| `training.temperature` | 0.05, 0.07, 0.1, 0.2 | InfoNCE 温度 |
| `training.weight_decay` | 0.01, 0.05, 0.1 | 权重衰减 |
| `training.warmup_steps` | 500, 1000, 2000 | 学习率预热步数 |
| `data.max_nodes` | 64, 128 | 子图最大节点数 |

搜索完成后，锁定 Stage 1 最优的 `lambda_disc`、`mu_align`、`temperature`。

#### Stage 2: 精调（base 模型，6 trials x 10k steps）

在 base 模型上精调学习率（loss 权重从 Stage 1 锁定）。

运行前，将 Stage 1 最优结果填入 `configs/experiment/pretrain_search_base.yaml`:

```yaml
training:
  lambda_disc: ???   # Stage 1 最优值
  mu_align: ???      # Stage 1 最优值
  temperature: ???   # Stage 1 最优值
```

单 GPU（~2 天）：
```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m riemannfm.cli.pretrain experiment=pretrain_search_base sweep=pretrain_base --multirun
```

多 GPU 并行（3 GPU ~0.7 天）：
```bash
# Terminal 1/2/3 — 同上，各指定不同 CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=1 uv run python -m riemannfm.cli.pretrain experiment=pretrain_search_base sweep=pretrain_base --multirun
CUDA_VISIBLE_DEVICES=2 uv run python -m riemannfm.cli.pretrain experiment=pretrain_search_base sweep=pretrain_base --multirun
CUDA_VISIBLE_DEVICES=3 uv run python -m riemannfm.cli.pretrain experiment=pretrain_search_base sweep=pretrain_base --multirun
```

共享 `hp_search_s2.db`，总共仍是 6 trials。

| 参数 | 值 |
|------|----|
| Model | base (10L/512d/192e, 66.9M) |
| Data | wikidata_5m |
| max_nodes | 128 |
| max_steps | 10,000 |
| warmup_steps | 1,000 |
| val_check_interval | 2,000 |
| batch_size x accum | 4 x 8 = 32 |
| Sampler | TPE, 6 trials, minimize val_loss |

搜索空间（3 个参数）：

| 参数 | 范围 | 说明 |
|------|------|------|
| `training.lr` | 3e-5, 5e-5, 1e-4, 1.5e-4 | 围绕 Stage 1 最优值缩放 |
| `training.warmup_steps` | 500, 1000, 2000 | 大模型可能需要更长 warmup |
| `training.weight_decay` | 0.01, 0.05 | 微调正则化 |

迁移原则：
- **直接迁移**（loss 特性，与模型大小无关）：lambda_disc, mu_align, temperature, max_nodes
- **需重新搜索**（与模型规模耦合）：lr, warmup_steps, weight_decay

---

### E1: Main Pretrain (base, 3 seeds x 100k steps)

正式预训练。运行前将 Stage 2 最优 HP 填入 `configs/experiment/pretrain_wiki5m.yaml`。
每个 seed 单 GPU 单进程（base 模型峰值显存 <10GB，无需多卡 DDP）。

单 seed 串行：
```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=42
```

3 seeds 多 GPU 并行（3.5 天全部完成，而非串行 10.5 天）：
```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=42 &
CUDA_VISIBLE_DEVICES=2 uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=123 &
CUDA_VISIBLE_DEVICES=3 uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=456 &
```

| 参数 | 值 |
|------|----|
| Model | base (10L/512d/192e, 66.9M) |
| Data | wikidata_5m |
| max_nodes | 128 |
| max_steps | 100,000 |
| warmup_steps | 2,000 |
| val_check_interval | 2,000 |
| batch_size x accum | 4 x 8 = 32 |
| HP | Stage 2 最优（待填入） |
| Est. time | ~3.5 days/seed (1x H100) |

报告 3 seeds 的 mean +/- std: val/loss, val/L_cont, val/L_disc, val/L_align, kappa_h, kappa_s。

---

### E2-E4: Ablation (small, 50k steps each)

Multirun 变体串行执行。如有多 GPU，可将不同消融实验分配到不同 GPU 并行：

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m riemannfm.cli.pretrain experiment=ablation_manifold --multirun &      # E2: 7 variants
CUDA_VISIBLE_DEVICES=2 uv run python -m riemannfm.cli.pretrain experiment=ablation_architecture --multirun &   # E3: 8 variants
CUDA_VISIBLE_DEVICES=3 uv run python -m riemannfm.cli.pretrain experiment=ablation_loss --multirun &           # E4: 3 variants
```

所有消融共享参数：small 模型 (22.5M)，50k steps，warmup 1k，val_check 2k，batch 4x8=32，~2 days/variant。

#### E2: Manifold Ablation (7 variants)

Multirun 变体（总维度 = 96，保持公平比较）：

| Variant | H dim | S dim | E dim | Learnable curvature |
|---------|-------|-------|-------|---------------------|
| h_only | 96 | 0 | 0 | yes |
| s_only | 0 | 96 | 0 | yes |
| e_only | 0 | 0 | 96 | n/a |
| h_e | 48 | 0 | 48 | yes |
| s_e | 0 | 48 | 48 | yes |
| product_h_s_e | 32 | 32 | 32 | yes |
| fixed_curvature | 32 | 32 | 32 | **no** |

---

#### E3: Architecture Ablation (8 variants)

Multirun 变体：

| Variant | 移除的组件 |
|---------|-----------|
| full | 无（baseline） |
| no_geok | Geodesic distance kernel |
| no_mrope | Manifold RoPE |
| no_mrope_geok | M-RoPE + Geodesic kernel |
| no_ath | ATH-Norm（回退为 LayerNorm） |
| no_edge_self | Edge self-update |
| no_cross | Dual-stream cross-interaction |
| no_text_cond | Text conditioning |

---

#### E4: Flow Ablation (3 variants)

Multirun 变体：

| Variant | Continuous Flow | Discrete Flow |
|---------|----------------|---------------|
| joint | yes | yes |
| continuous_only | yes | no |
| discrete_only | no | yes |

---

### E5: Scaling (small/base/large, 100k steps)

```bash
uv run python -m riemannfm.cli.pretrain experiment=scaling --multirun
```

| 参数 | 值 |
|------|----|
| max_steps | 100,000 |
| warmup_steps | 2,000 |
| val_check_interval | 2,000 |
| batch_size x accum | 4 x 8 = 32 |

Multirun 变体：

| Model | Arch Params | Layers | node_dim | edge_dim | Est. time |
|-------|-------------|--------|----------|----------|-----------|
| small | 22.5M | 6 | 384 | 128 | ~2 days |
| base | 66.9M | 10 | 512 | 192 | ~3.5 days |
| large | 200.7M | 14 | 768 | 256 | ~5.5 days |

---

## 5. 通用选项

### 从 Checkpoint 恢复训练

```bash
uv run python -m riemannfm.cli.pretrain \
    paths.ckpt_path=/path/to/outputs/<run_dir>/checkpoints/last.ckpt
```

恢复内容：模型权重、优化器状态、lr scheduler、global_step、callbacks。
若原始 run 使用了 wandb，会自动检测 run id 并续接同一 wandb run。

### Logger 选择

```bash
uv run python -m riemannfm.cli.pretrain logger=default   # wandb + csv (默认)
uv run python -m riemannfm.cli.pretrain logger=wandb     # wandb only
uv run python -m riemannfm.cli.pretrain logger=csv       # csv only (离线)
uv run python -m riemannfm.cli.pretrain logger=none      # 不记录
```

### 多 GPU 并行

当前所有实验均为**单 GPU 单进程**（base 模型峰值显存 <10GB），通过 `CUDA_VISIBLE_DEVICES` 指定 GPU。
多 GPU 加速的方式是**多进程并行**，而非单 run 多卡 DDP。

| 场景 | 并行方式 | 原理 |
|------|----------|------|
| HP Search | 多进程共享 Optuna SQLite study | 同一 study_name + .db 文件，TPE 自动同步 |
| E1 多 seed | 每个 seed 一个 GPU | 完全独立，run_name 含 seed 不冲突 |
| E2-E4 消融 | 每个消融实验一个 GPU | 完全独立，不同 experiment config |

```bash
# 查看可用 GPU
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# 指定单个 GPU
CUDA_VISIBLE_DEVICES=1 uv run python -m riemannfm.cli.pretrain experiment=validation_run

# 多进程后台并行（& 放入后台）
CUDA_VISIBLE_DEVICES=1 uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=42 &
CUDA_VISIBLE_DEVICES=2 uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=123 &
CUDA_VISIBLE_DEVICES=3 uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=456 &
```

> 如需对单个 run 使用多卡 DDP：`uv run python -m riemannfm.cli.pretrain accelerator=ddp`（4 GPU）。
> 但当前模型规模下单卡足够，多进程并行是更高效的策略。

---

## 6. Experiment Config 总览（已验证）

以下参数经 `uv run python -c` Hydra compose + 模型实例化测试确认：

| Config | Phase | Model | Data | N_max | Steps | Warmup | ValInt | BS x Acc |
|--------|-------|-------|------|-------|-------|--------|--------|----------|
| `smoke_test` | P0 | 6L/384d | mini | 64 | 1k | 200 | 500 | 8 x 1 |
| `validation_run` | P1 | 10L/512d | full | 128 | 10k | 500 | 500 | 4 x 8 |
| `pretrain_search` | P2a | 6L/384d | full | 128 | 10k | 1k | 2k | 4 x 8 |
| `pretrain_search_base` | P2b | 10L/512d | full | 128 | 10k | 1k | 2k | 4 x 8 |
| `pretrain_wiki5m` | E1 | 10L/512d | full | 128 | 100k | 2k | 2k | 4 x 8 |
| `ablation_manifold` | E2 | 6L/384d | full | 128 | 50k | 1k | 2k | 4 x 8 |
| `ablation_architecture` | E3 | 6L/384d | full | 128 | 50k | 1k | 2k | 4 x 8 |
| `ablation_loss` | E4 | 6L/384d | full | 128 | 50k | 1k | 2k | 4 x 8 |
| `scaling` | E5 | all* | full | 128 | 100k | 2k | 2k | 4 x 8 |

*scaling 默认 base，通过 `--multirun` 遍历 small / base / large。

所有 experiment 共享 loss 权重默认值 (lambda_disc=10.0, mu_align=0.5, tau=0.07)，
P2a/P2b 中由 Optuna sweep 覆盖，E1-E5 使用 HP search 最终结果。
