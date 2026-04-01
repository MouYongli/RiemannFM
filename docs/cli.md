# CLI 用法参考

所有 CLI 命令基于 [Hydra](https://hydra.cc/)，配置文件位于 `configs/`。

---

## 1. 数据下载

```bash
uv run python -m riemannfm.cli.download data=wikidata_5m
uv run python -m riemannfm.cli.download download.all=true        # 全部 7 个
```

## 2. Build Mini (工程验证子集)

```bash
uv run python -m riemannfm.cli.preprocess data=wikidata_5m preprocess.build_mini=true
```

## 3. 数据预处理 (文本嵌入)

```bash
uv run python -m riemannfm.cli.preprocess data=wikidata_5m_mini embedding=nomic
uv run python -m riemannfm.cli.preprocess data=wikidata_5m embedding=qwen3  # 换编码器
```

## 4. 预训练

```bash
# Smoke test (100 步, CSV 日志)
uv run python -m riemannfm.cli.pretrain data=wikidata_5m_mini training.max_steps=100 training.batch_size=4 logger=csv

# MVP (1000 步)
uv run python -m riemannfm.cli.pretrain training.max_steps=1000 training.batch_size=8

# 超参搜索
uv run python -m riemannfm.cli.pretrain +experiment=pretrain_search --multirun

# 完整预训练
uv run python -m riemannfm.cli.pretrain +experiment=pretrain_wiki5m

# 消融
uv run python -m riemannfm.cli.pretrain +experiment=ablation_architecture \
    ablation=no_mrope,no_geok,no_ath,no_edge_self,no_cross,no_text_cond --multirun
```

### Logger 选择

```bash
uv run python -m riemannfm.cli.pretrain logger=default      # wandb + csv (默认)
uv run python -m riemannfm.cli.pretrain logger=wandb    # 仅 wandb
uv run python -m riemannfm.cli.pretrain logger=csv      # 仅 csv (离线)
uv run python -m riemannfm.cli.pretrain logger=none           # 无日志
```

## 5. 微调 (未实现)

```bash
uv run python -m riemannfm.cli.finetune +experiment=kgc_fb15k237 \
    training.pretrained_ckpt=/path/to/ckpt.pt
```

## 6. 评估 (未实现)

```bash
uv run python -m riemannfm.cli.evaluate +experiment=kgc_fb15k237 \
    eval.checkpoint=/path/to/finetuned.pt
```

## 7. 生成 / 推理 (未实现)

```bash
uv run python -m riemannfm.cli.generate task=t2g \
    eval.checkpoint=/path/to/t2g.pt eval.num_generation_samples=1000
```

## 8. GAD 零样本评估 (未实现)

```bash
uv run python -m riemannfm.cli.evaluate +experiment=gad \
    eval.checkpoint=/path/to/pretrained.pt
```
