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

### Phase 0: Smoke Test

```bash
uv run python -m riemannfm.cli.pretrain \
    model=small data=wikidata_5m_mini training.max_steps=1000
```

### Phase 1: Validation Run

```bash
uv run python -m riemannfm.cli.pretrain \
    model=base data=wikidata_5m training.max_steps=10000 \
    training.warmup_steps=1000 training.val_check_interval=2000
```

### Phase 2: HP Search

```bash
uv run python -m riemannfm.cli.pretrain \
    experiment=pretrain_search sweep=pretrain
```

### E1: Main Pretrain (3 seeds)

```bash
uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=42
uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=123
uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=456
```

### E2: Manifold Ablation

```bash
uv run python -m riemannfm.cli.pretrain experiment=ablation_manifold --multirun
```

### E3: Architecture Ablation

```bash
uv run python -m riemannfm.cli.pretrain experiment=ablation_architecture --multirun
```

### E4: Flow Ablation

```bash
uv run python -m riemannfm.cli.pretrain experiment=ablation_loss --multirun
```

### E5: Scaling

```bash
uv run python -m riemannfm.cli.pretrain experiment=scaling --multirun
```

### Logger 选择

```bash
uv run python -m riemannfm.cli.pretrain logger=default   # wandb + csv (默认)
uv run python -m riemannfm.cli.pretrain logger=wandb     # 仅 wandb
uv run python -m riemannfm.cli.pretrain logger=csv       # 仅 csv (离线)
uv run python -m riemannfm.cli.pretrain logger=none      # 无日志
```
