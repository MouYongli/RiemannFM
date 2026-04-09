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

## 4. 预训练

### 硬件约束 (1x H100 80GB)

所有实验基于单张 H100 80GB 调参。base 模型 + wikidata_5m 的关键约束：

- `max_nodes=128`（256 会 OOM）
- `batch_size=4`（8 会 OOM）
- `gradient_accumulation_steps=8`（effective batch = 32）
- 实测吞吐：base 模型 ~0.82 s/step（batch=4, accum=1, max_nodes=128）

### 参数联动规则

缩短 `max_steps` 时，以下参数必须同步调整：

| 参数 | 经验比例 | Phase 0 (1k) | Phase 1 (10k) | Phase 2 (20k) | E1-E5 (500k) |
|------|---------|:---:|:---:|:---:|:---:|
| `warmup_steps` | 10-20% of max_steps | 200 | 1,000 | 2,000 | 10,000 (默认) |
| `val_check_interval` | max_steps / 2~5 | 500 | 2,000 | 5,000 | 10,000 (默认) |
| `gradient_accumulation_steps` | — | 1 | 8 | 8 | 8 |

### Phase 0: Smoke Test

验证代码能跑通，loss 能下降。mini 数据 + small 模型，~4 分钟。

```bash
uv run python -m riemannfm.cli.pretrain experiment=smoke_test
```

等价 CLI 覆盖:
```bash
uv run python -m riemannfm.cli.pretrain \
    model=small data=wikidata_5m_mini \
    training.max_steps=1000 \
    training.warmup_steps=200 \
    training.val_check_interval=500 \
    training.gradient_accumulation_steps=1
```

通过标准：无 NaN/Inf，loss 单调下降，单 GPU < 24GB。

### Phase 1: Validation Run

验证完整 pipeline 在真实数据上的训练动态，~18 小时。

```bash
uv run python -m riemannfm.cli.pretrain experiment=validation_run
```

等价 CLI 覆盖:
```bash
uv run python -m riemannfm.cli.pretrain \
    model=base data=wikidata_5m \
    data.max_nodes=128 \
    training.max_steps=10000 \
    training.warmup_steps=1000 \
    training.val_check_interval=2000 \
    training.batch_size=4 \
    training.gradient_accumulation_steps=8
```

通过标准：multi-t val 稳定，L_cont/L_disc/L_align 比例合理，curvature 值在演化。

### Phase 2: HP Search

Optuna TPE 搜索最优超参，15 trials x 20k steps，~10 天。

```bash
uv run python -m riemannfm.cli.pretrain \
    experiment=pretrain_search sweep=pretrain
```

搜索完成后，将最优超参写入 `configs/training/pretrain.yaml` 或对应 experiment config。

### E1: Main Pretrain (3 seeds)

正式预训练，所有参数用默认值（或 HP search 最优结果）。

```bash
uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=42
uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=123
uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=456
```

### E2: Manifold Ablation (7 runs)

```bash
uv run python -m riemannfm.cli.pretrain experiment=ablation_manifold --multirun
```

### E3: Architecture Ablation (8 runs)

```bash
uv run python -m riemannfm.cli.pretrain experiment=ablation_architecture --multirun
```

### E4: Flow Ablation (3 runs)

```bash
uv run python -m riemannfm.cli.pretrain experiment=ablation_loss --multirun
```

### E5: Scaling (3 sizes)

```bash
uv run python -m riemannfm.cli.pretrain experiment=scaling --multirun
```

### 从 Checkpoint 恢复训练

```bash
uv run python -m riemannfm.cli.pretrain \
    paths.ckpt_path=/path/to/outputs/<run_dir>/checkpoints/last.ckpt
```

恢复内容包括：模型权重、优化器状态、lr scheduler、global_step、callbacks 状态。
若原始 run 使用了 wandb，会自动检测 run id 并续接同一个 wandb run。

### Logger 选择

```bash
# default (wandb+csv) | wandb | csv (离线) | none
uv run python -m riemannfm.cli.pretrain logger=default
```

## 5. Experiment Config 总览

所有 experiment configs 位于 `configs/experiment/`，每个实验封装了完整参数：

| Config | Phase | 用途 | 关键覆盖 |
|--------|-------|------|---------|
| `smoke_test` | 0 | 代码验证 | small, mini, 1k steps, accum=1 |
| `validation_run` | 1 | 训练动态验证 | base, 10k steps, max_nodes=128 |
| `pretrain_search` | 2 | HP 搜索 | base, 20k steps, 15 trials |
| `pretrain_wiki5m` | E1 | 正式预训练 | base, 500k steps, 3 seeds |
| `ablation_manifold` | E2 | 流形消融 | 7 manifold variants |
| `ablation_architecture` | E3 | 架构消融 | 8 ablation flags |
| `ablation_loss` | E4 | Flow 消融 | 3 flow configs |
| `scaling` | E5 | 模型缩放 | small/base/large |
