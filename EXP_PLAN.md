# Experiment Plan — ISWC 2026 (Pretrain + Ablation)

## Core Claim

Joint continuous-discrete flow matching on product Riemannian manifolds (H x S x E), with geodesic attention and text conditioning, learns high-quality generative representations for knowledge graphs.

## Experiment Overview

| ID | RQ | Name | Purpose | Runs |
|----|------|------|---------|------|
| E1 | RQ1 | Main Pretrain | Full pretrain with 3 seeds | 3 |
| E2 | RQ2 | Manifold Ablation | H x S x E vs single/dual manifolds vs fixed curvature | 7 |
| E3 | RQ3 | Architecture Ablation | Remove one component at a time | 8 |
| E4 | RQ4 | Flow Ablation | joint vs continuous-only vs discrete-only | 3 |
| E5 | RQ5 | Scaling | small / base / large | 3 |
| | | | **Total** | **24** |

---

## Experiment Dependencies & Execution Order

```
Phase 0: Smoke Test
   |      wikidata_5m_mini + small, 1k steps
   |      Pass criteria: no NaN, loss decreasing
   v
Phase 1: Validation Run
   |      wikidata_5m + base, 10k steps
   |      Pass criteria: multi-t val stable, loss components balanced, curvature evolving
   v
Phase 2: HP Search
   |      wikidata_5m + base, 50k steps x 30 trials (Optuna TPE)
   |      Output: best_hp = {lr, curvature_lr, warmup, lambda_disc, mu_align, tau, wd, max_nodes}
   v
   |--- best_hp injected into ALL subsequent experiments ---|
   v                                                        v
Phase 3: Ablation (independent, parallelizable)     Phase 3: Main + Scaling
   |                                                        |
   |-- E2: Manifold Ablation   (7 runs)                     |-- E1: Main Pretrain (3 seeds)
   |-- E3: Architecture Ablation (8 runs)                    |-- E5: Scaling       (3 sizes)
   |-- E4: Flow Ablation       (3 runs)
```

### Dependency Table

| Experiment | Depends on | Reason |
|------------|-----------|--------|
| Phase 0: Smoke test | none | First step, verify code |
| Phase 1: Validation | Phase 0 pass | Confirm training dynamics |
| Phase 2: HP search | Phase 1 pass | Pipeline must be stable before large-scale search |
| E1: Main pretrain | HP search done | Use best HP |
| E2: Manifold ablation | HP search done | Control variables: same HP, vary manifold only |
| E3: Architecture ablation | HP search done | Control variables: same HP, vary architecture only |
| E4: Flow ablation | HP search done | Control variables: same HP, vary flow only |
| E5: Scaling | HP search done | Use best HP with lr scaling rule |

---

## Development Phases (not in paper)

### Phase 0: Smoke Test

| Item | Setting |
|------|---------|
| Dataset | wikidata_5m_mini (7.8k entities, 822 relations) |
| Model | small (22.7M) |
| Steps | 1,000 |
| Command | `uv run python -m riemannfm.cli.pretrain model=small data=wikidata_5m_mini training.max_steps=1000` |
| Pass criteria | No NaN/Inf, loss monotonically decreasing, no OOM |

### Phase 1: Validation Run

| Item | Setting |
|------|---------|
| Dataset | wikidata_5m |
| Model | base (174.7M) |
| Steps | 10,000 |
| Command | `uv run python -m riemannfm.cli.pretrain model=base data=wikidata_5m training.max_steps=10000` |
| Pass criteria | Multi-t val stable, L_cont/L_disc/L_align ratios reasonable, curvature values evolving |

### Phase 2: HP Search

| Item | Setting |
|------|---------|
| Dataset | wikidata_5m |
| Model | base (fixed) |
| Steps/trial | 50,000 |
| Trials | 30 |
| Sampler | Optuna TPE |
| Objective | minimize val/loss |
| Seed | 42 (single seed during search) |
| Config | `experiment=pretrain_search` + `sweep=pretrain` |

#### Search Space

| Hyperparameter | Search Space | Current Default |
|---------------|-------------|-----------------|
| training.lr | [5e-5, 1e-4, 3e-4, 5e-4] | 1e-4 |
| training.curvature_lr | [1e-6, 5e-6, 1e-5, 5e-5] | 1e-5 |
| training.warmup_steps | [5000, 10000, 20000] | 10000 |
| training.lambda_disc | [0.1, 0.5, 1.0, 2.0] | 1.0 |
| training.mu_align | [0.01, 0.05, 0.1, 0.5] | 0.1 |
| training.temperature | [0.05, 0.07, 0.1, 0.2] | 0.07 |
| training.weight_decay | [0.01, 0.05, 0.1] | 0.01 |
| data.max_nodes | [32, 64, 128] | 256 |

**Not searched** (fixed by model size): num_layers, node_dim, edge_dim, num_heads, edge_heads.

---

## Paper Experiments

### E1: Main Pretrain

> **RQ1**: Can RiemannFM pretrain stably on large-scale KGs with well-behaved loss dynamics?

| Item | Setting |
|------|---------|
| Dataset | wikidata_5m (4.8M entities, 822 relations) |
| Model | base (174.7M) |
| Steps | 500,000 |
| Seeds | 42, 123, 456 |
| HP | best from HP search |
| Metrics | val/loss, val/L_cont, val/L_disc, val/L_align, kappa_h, kappa_s |
| Config | `experiment=pretrain_wiki5m` |

### E2: Manifold Ablation

> **RQ2**: How much does the product manifold H x S x E improve over single/dual manifolds? Does learnable curvature help?

| Run | Manifold | H dim | S dim | E dim | Learnable curvature |
|-----|----------|-------|-------|-------|-------------------|
| E2a | h_only | 96 | 0 | 0 | yes |
| E2b | s_only | 0 | 96 | 0 | yes |
| E2c | e_only | 0 | 0 | 96 | n/a |
| E2d | h_e | 48 | 0 | 48 | yes |
| E2e | s_e | 0 | 48 | 48 | yes |
| E2f | product_h_s_e | 32 | 32 | 32 | yes |
| E2g | fixed_curvature | 32 | 32 | 32 | **no** |

| Item | Setting |
|------|---------|
| Dataset | wikidata_5m |
| Model | base |
| Steps | 500,000 |
| Seed | 42 (total ambient dim = 96 for fair comparison) |
| HP | best from HP search |
| Metrics | val/loss, val/L_cont, val/L_disc |
| Config | `experiment=ablation_manifold` |

### E3: Architecture Ablation

> **RQ3**: How much does each architectural innovation contribute?

| Run | Config | Component Removed |
|-----|--------|------------------|
| E3a | full | none (upper bound) |
| E3b | no_geok | Geodesic distance kernel |
| E3c | no_mrope | Manifold RoPE |
| E3d | no_mrope_geok | M-RoPE + Geodesic kernel (both) |
| E3e | no_ath | ATH-Norm (falls back to LayerNorm) |
| E3f | no_edge_self | Edge self-update (Def 5.11) |
| E3g | no_cross | Dual-stream cross-interaction (Def 5.13) |
| E3h | no_text_cond | Text conditioning (globally disabled) |

| Item | Setting |
|------|---------|
| Dataset | wikidata_5m |
| Model | base |
| Steps | 500,000 |
| Seed | 42 |
| HP | best from HP search |
| Metrics | val/loss, val/L_cont, val/L_disc, delta% vs full |
| Config | `experiment=ablation_architecture` |

### E4: Flow Ablation

> **RQ4**: Does joint flow matching outperform continuous-only or discrete-only?

| Run | Flow config | Continuous | Discrete |
|-----|------------|-----------|----------|
| E4a | joint | yes | yes |
| E4b | continuous_only | yes | no |
| E4c | discrete_only | no | yes |

| Item | Setting |
|------|---------|
| Dataset | wikidata_5m |
| Model | base |
| Steps | 500,000 |
| Seed | 42 |
| HP | best from HP search |
| Metrics | val/loss, val/L_cont (E4a,b), val/L_disc (E4a,c) |
| Config | `experiment=ablation_loss` |

### E5: Scaling

> **RQ5**: Does scaling model capacity yield consistent improvement?

| Run | Model | Params | Layers | node_dim |
|-----|-------|--------|--------|----------|
| E5a | small | 22.7M | 6 | 384 |
| E5b | base | 174.7M | 12 | 768 |
| E5c | large | 628M | 24 | 1024 |

| Item | Setting |
|------|---------|
| Dataset | wikidata_5m |
| Steps | 500,000 (same step budget) |
| Seed | 42 |
| HP | best from HP search, lr scaled per size: lr_s = lr_base * sqrt(params_base / params_s) |
| Metrics | val/loss, val/L_cont, val/L_disc, throughput (samples/sec), peak GPU memory |

---

## Expected Paper Outputs

| Output | Source | Content |
|--------|--------|---------|
| Table 1 | E1 | Main results: 3-seed mean +/- std for all metrics |
| Table 2 | E2 | Manifold ablation: 7 rows, val/loss + learned curvatures |
| Table 3 | E3 | Architecture ablation: 8 rows, val/loss + delta% vs full |
| Table 4 | E4 | Flow ablation: 3 rows |
| Figure 1 | E5 | Scaling curve: val/loss vs params (log-log) |
| Figure 2 | E1 | Training dynamics: loss curves + curvature evolution over steps |

---

## Compute Budget

| Phase | Experiment | Runs | Steps/run | Est. GPU-hours (A100) |
|-------|-----------|------|-----------|----------------------|
| 0 | Smoke test | 1 | 1k | < 0.5h |
| 1 | Validation | 1 | 10k | ~2h |
| 2 | HP search | 30 | 50k | ~60h |
| 3 | E1 Main (3 seeds) | 3 | 500k | ~72h |
| 3 | E2 Manifold (7 runs) | 7 | 500k | ~168h |
| 3 | E3 Architecture (8 runs) | 8 | 500k | ~192h |
| 3 | E4 Flow (3 runs) | 3 | 500k | ~72h |
| 3 | E5 Scaling (3 sizes) | 3 | 500k | ~120h |
| | **Total** | **56** | | **~687h (~28.5 GPU-days)** |

### Scheduling (4x A100)

```
Week 1 Day 1-2:   Phase 0 + 1 + 2 (HP search)           ~1.5 days
Week 1-2 Day 2-5: E1 + E4 + E5 in parallel (9 runs)     ~2.5 days
Week 2-3 Day 5-10: E2 + E3 in parallel (15 runs)         ~5 days
                                                   Total: ~9-10 days
```
