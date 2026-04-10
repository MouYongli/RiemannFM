# Experiment Plan — RiemannFM Pretraining

## Core Claim

Joint continuous-discrete flow matching on product Riemannian manifolds (H x S x E), with geodesic attention and text conditioning, learns high-quality generative representations for knowledge graphs.

## Model Sizes

| Config | Layers | node_dim | edge_dim | Heads | Arch Params | Use |
|--------|--------|----------|----------|-------|-------------|-----|
| small  | 6      | 384      | 128      | 6     | ~22M        | HP search, ablations |
| base   | 10     | 512      | 192      | 8     | ~65M        | Main experiment (E1) |
| large  | 14     | 768      | 256      | 12    | ~180M       | Scaling study (E5) |

Entity embedding (4.6M x 97) = 446M params shared across all sizes.

## Experiment Overview

| ID | Name | Model | Runs | Steps | Est. Time (1xH100) |
|----|------|-------|------|-------|---------------------|
| P0 | Smoke Test | small (mini data) | 1 | 1k | 5 min |
| P1 | Validation Run | base | 1 | 10k | 8h |
| P2a | HP Search Stage 1 | small | 20 | 10k | 4 days |
| P2b | HP Search Stage 2 | base | 6 | 10k | 2 days |
| E1 | Main Pretrain | base | 3 (seeds) | 100k | 10 days |
| E2 | Manifold Ablation | small | 7 | 50k | 14 days |
| E3 | Architecture Ablation | small | 8 | 50k | 16 days |
| E4 | Flow Ablation | small | 3 | 50k | 6 days |
| E5 | Scaling | small/base/large | 3 | 100k | 11 days |

---

## Execution Flow

```
Phase 0: Smoke Test (mini data + small model, 1k steps, ~5 min)
   |   Pass: no NaN, L_cont/L_disc/L_align all decreasing, kappa evolving
   v
Phase 1: Validation Run (full data + base model, 10k steps, ~8h)
   |   Pass: val_loss stable, L_disc < 0.6, L_align trending down
   v
Phase 2a: HP Search Stage 1 (small model, 20 trials x 10k steps, ~4 days)
   |   Search: lr, curvature_lr, lambda_disc, mu_align, temperature,
   |           weight_decay, warmup_steps, max_nodes
   |   Lock: loss weights (lambda_disc, mu_align, temperature)
   v
Phase 2b: HP Search Stage 2 (base model, 6 trials x 10k steps, ~2 days)
   |   Search: lr, warmup_steps, weight_decay (loss weights locked from stage 1)
   |   Output: final HP set for base model
   v
   |--------- best HP injected into all subsequent experiments ---------|
   v                                                                    v
Phase 3a: Ablations (small model, parallelizable)     Phase 3b: Main + Scaling
   |                                                           |
   |-- E2: Manifold     (7 variants x 50k)                    |-- E1: Main Pretrain (3 seeds x 100k)
   |-- E3: Architecture (8 variants x 50k)                    |-- E5: Scaling (3 sizes x 100k)
   |-- E4: Flow         (3 variants x 50k)
```

---

## Phase 0: Smoke Test

| Item | Setting |
|------|---------|
| Dataset | wikidata_5m_mini (7.8k entities) |
| Model | small (~22M arch params) |
| Steps | 1,000 |
| Command | `uv run python -m riemannfm.cli.pretrain experiment=smoke_test` |
| Pass criteria | No NaN/Inf, L_disc < 0.5 at end, L_align trending down, kappa shift > 0.01 |

## Phase 1: Validation Run

| Item | Setting |
|------|---------|
| Dataset | wikidata_5m (max_nodes=128) |
| Model | base (~65M arch params) |
| Steps | 10,000 (val every 500) |
| Batch | 4 x 8 accum = 32 effective |
| Command | `uv run python -m riemannfm.cli.pretrain experiment=validation_run` |
| Pass criteria | val_loss not increasing after step 2000, L_disc < 0.6, curvature shift > 0.05 |

## Phase 2: HP Search (Two-Stage)

### Stage 1: Broad search on small model

| Item | Setting |
|------|---------|
| Model | small (fast iteration) |
| Steps/trial | 10,000 |
| Trials | 20 (TPE) |
| Command | `uv run python -m riemannfm.cli.pretrain experiment=pretrain_search sweep=pretrain --multirun` |

Search space:

| Hyperparameter | Range | Rationale |
|---------------|-------|-----------|
| lr | [3e-5, 1e-4, 3e-4, 1e-3] | 1.5 orders of magnitude |
| curvature_lr | [1e-5, 5e-5, 1e-4, 5e-4] | Independent from main lr |
| lambda_disc | [5, 10, 20, 50] | Compensate L_cont/L_disc ~40x gap |
| mu_align | [0.1, 0.5, 1.0, 2.0] | Post gradient-fix, can be larger |
| temperature | [0.05, 0.07, 0.1, 0.2] | InfoNCE sharpness |
| weight_decay | [0.01, 0.05, 0.1] | Regularization |
| warmup_steps | [500, 1000, 2000] | Fraction of 10k budget |
| max_nodes | [64, 128] | Subgraph size |

**Transfers well to base**: lambda_disc, mu_align, temperature, max_nodes (loss-level properties).
**Needs re-tuning on base**: lr, warmup_steps, weight_decay (model-size-dependent).

### Stage 2: lr fine-tune on base model

| Item | Setting |
|------|---------|
| Model | base |
| Steps/trial | 10,000 |
| Trials | 6 (TPE) |
| Command | `uv run python -m riemannfm.cli.pretrain experiment=pretrain_search_base sweep=pretrain_base --multirun` |

Search space (loss weights locked from stage 1):

| Hyperparameter | Range |
|---------------|-------|
| lr | [3e-5, 5e-5, 1e-4, 1.5e-4] |
| warmup_steps | [500, 1000, 2000] |
| weight_decay | [0.01, 0.05] |

---

## Paper Experiments

### E1: Main Pretrain

> **RQ1**: Can RiemannFM pretrain stably on large-scale KGs?

| Item | Setting |
|------|---------|
| Model | base (~65M arch) |
| Steps | 100,000 |
| Seeds | 42, 123, 456 |
| HP | best from stage 2 |
| Command | `uv run python -m riemannfm.cli.pretrain experiment=pretrain_wiki5m seed=42` |
| Metrics | val/loss, val/L_cont, val/L_disc, val/L_align, kappa_h, kappa_s |

### E2: Manifold Ablation

> **RQ2**: How much does product H x S x E improve over single/dual manifolds?

| Variant | H dim | S dim | E dim | Learnable curvature |
|---------|-------|-------|-------|---------------------|
| h_only | 96 | 0 | 0 | yes |
| s_only | 0 | 96 | 0 | yes |
| e_only | 0 | 0 | 96 | n/a |
| h_e | 48 | 0 | 48 | yes |
| s_e | 0 | 48 | 48 | yes |
| product_h_s_e | 32 | 32 | 32 | yes |
| fixed_curvature | 32 | 32 | 32 | **no** |

| Item | Setting |
|------|---------|
| Model | small (7 variants x 50k steps) |
| Command | `uv run python -m riemannfm.cli.pretrain experiment=ablation_manifold --multirun` |

### E3: Architecture Ablation

> **RQ3**: How much does each architectural component contribute?

| Variant | Removed component |
|---------|-------------------|
| full | none (baseline) |
| no_geok | Geodesic distance kernel |
| no_mrope | Manifold RoPE |
| no_mrope_geok | M-RoPE + geodesic kernel |
| no_ath | ATH-Norm (fallback: LayerNorm) |
| no_edge_self | Edge self-update |
| no_cross | Dual-stream cross-interaction |
| no_text_cond | Text conditioning |

| Item | Setting |
|------|---------|
| Model | small (8 variants x 50k steps) |
| Command | `uv run python -m riemannfm.cli.pretrain experiment=ablation_architecture --multirun` |

### E4: Flow Ablation

> **RQ4**: Does joint flow matching outperform single-mode flow?

| Variant | Continuous | Discrete |
|---------|-----------|----------|
| joint | yes | yes |
| continuous_only | yes | no |
| discrete_only | no | yes |

| Item | Setting |
|------|---------|
| Model | small (3 variants x 50k steps) |
| Command | `uv run python -m riemannfm.cli.pretrain experiment=ablation_loss --multirun` |

### E5: Scaling

> **RQ5**: Does scaling model capacity yield consistent improvement?

| Model | Arch Params | Layers | node_dim | edge_dim |
|-------|-------------|--------|----------|----------|
| small | ~22M | 6 | 384 | 128 |
| base | ~65M | 10 | 512 | 192 |
| large | ~180M | 14 | 768 | 256 |

| Item | Setting |
|------|---------|
| Steps | 100,000 (same budget for all sizes) |
| Command | `uv run python -m riemannfm.cli.pretrain experiment=scaling --multirun` |
| Metrics | val/loss, throughput (samples/sec), peak GPU memory |

---

## Expected Paper Outputs

| Output | Source | Content |
|--------|--------|---------|
| Table 1 | E1 | Main results: 3-seed mean +/- std |
| Table 2 | E2 | Manifold ablation: 7 rows |
| Table 3 | E3 | Architecture ablation: 8 rows, delta% vs full |
| Table 4 | E4 | Flow ablation: 3 rows |
| Figure 1 | E5 | Scaling curve: val/loss vs arch params |
| Figure 2 | E1 | Training dynamics: loss + curvature over steps |

---

## Compute Budget (1x H100, sequential)

| Phase | GPU-hours | Calendar (1 GPU) |
|-------|-----------|-------------------|
| P0 Smoke | < 0.1h | 5 min |
| P1 Validation | 8h | 8h |
| P2a HP Stage 1 | ~90h | 4 days |
| P2b HP Stage 2 | ~47h | 2 days |
| E1 Main (3 seeds) | ~234h | 10 days |
| E2 Manifold (7) | ~156h | 6.5 days |
| E3 Architecture (8) | ~178h | 7.4 days |
| E4 Flow (3) | ~67h | 2.8 days |
| E5 Scaling (3) | ~264h | 11 days |
| **Total** | **~1044h** | **~52 days (1 GPU)** |

With 2 GPUs (GPU 1 + GPU 3): **~26 days**, parallelizing E1/E5 with E2-E4.

### Scheduling (2x H100)

```
GPU 1                               GPU 3
──────                               ──────
Day 1:   P0 + P1 (9h)
Day 2-5: P2a Stage 1 (4d)
Day 6-7: P2b Stage 2 (2d)
Day 8-17:  E1 seed=42 (3.5d)        E2 Manifold (6.5d)
           E1 seed=123 (3.5d)       E3 Architecture (7.4d)
           E1 seed=456 (3.5d)
Day 18-28: E5 Scaling (11d)         E4 Flow (2.8d)
```
