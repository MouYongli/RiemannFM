# RiemannFM 预训练阶段实施计划

## 项目概况

RiemannFM 是一个深度图生成基础模型，在积黎曼流形 (H x S x E) 上执行联合连续-离散流匹配，用于知识图谱生成及下游任务（KGC、T2G、GAD）。目标会议：ISWC 2025。

## 技术栈

| 类别 | 工具 |
|------|------|
| 语言 | Python 3.12+ |
| 核心框架 | PyTorch 2.x, PyTorch Geometric |
| 训练框架 | Lightning 2.x |
| 黎曼优化 | geoopt |
| ODE 求解 | torchdiffeq |
| 配置 | Hydra / OmegaConf |
| 文本编码 | transformers (SBERT / Qwen3 / XLM-R / Nomic) |
| 实验追踪 | wandb |
| 包管理 | uv (PEP 735) |
| 质量 | ruff + mypy + pytest |

## 当前目录结构

```
configs/                        # 59 个 Hydra YAML 配置
├── model/                      #   rieformer_{small,base,large}
├── manifold/                   #   product_h_s_e + 6 ablation variants
├── flow/                       #   joint, continuous_only, discrete_only
├── embedding/                  #   sbert, qwen3, xlm_roberta, nomic, none
├── data/                       #   wikidata_5m, wikidata_5m_mini, fb15k237, ...
├── training/                   #   pretrain, finetune
├── task/                       #   kgc_lp, kgc_rp, t2g, gad
├── ablation/                   #   8 architecture ablation configs
├── experiment/                 #   14 full experiment recipes
└── sweep/                      #   4 Optuna search spaces

src/riemannfm/
├── data/                       # ✅ 已实现 (~1,900 LOC)
│   ├── graph.py                #   RiemannFMGraphData (Def 3.4-3.8)
│   ├── sampler.py              #   BFS 子图采样 + multi-hot E
│   ├── collator.py             #   Batch padding to N_max
│   ├── datamodule.py           #   Lightning DataModule
│   ├── datasets/
│   │   └── pretrain_dataset.py #   KG 子图数据集
│   └── pipeline/
│       ├── download.py         #   多格式数据集下载
│       ├── preprocess.py       #   ID 映射 + mini 数据集构建
│       ├── embed.py            #   多后端文本编码 (HF/Ollama/OpenAI)
│       └── validate.py         #   数据完整性验证
├── manifolds/                  # ⬜ placeholder
├── flow/                       # ⬜ placeholder
├── models/                     # ⬜ placeholder
├── losses/                     # ⬜ placeholder
├── optim/                      # ⬜ placeholder
├── tasks/                      # ⬜ placeholder
├── utils/                      # ⬜ placeholder
└── cli/
    ├── download.py             # ✅ 已实现
    ├── preprocess.py           # ✅ 已实现
    ├── pretrain.py             # ⬜ placeholder
    ├── finetune.py             # ⬜ placeholder
    ├── evaluate.py             # ⬜ placeholder
    └── generate.py             # ⬜ placeholder

tests/
├── test_datamodule.py          # ✅ 15 个测试全部通过
└── conftest.py
```

## 已实现的功能

| 功能 | 对应数学定义 | 状态 |
|------|------------|------|
| 数据下载 (7 个 KG 数据集) | — | ✅ |
| 预处理 (ID 映射 + 文本编码) | — | ✅ |
| wikidata_5m_mini 构建 | — | ✅ |
| RiemannFMGraphData 5-tuple | Def 3.4-3.8 | ✅ |
| Multi-hot 边类型 E in {0,1}^{NxNxK} | Def 3.4 | ✅ |
| 虚节点 padding (edges=0, text=0, mask=0) | Def 3.7-3.8 | ✅ |
| BFS 子图采样 + 重要性采样 | — | ✅ |
| Lightning DataModule | — | ✅ |
| 预计算文本嵌入 C_V, C_R | Def 3.5-3.6 | ✅ |

## 预训练阶段：待实现功能

### Layer 0: manifolds/ — 黎曼几何基础

| 组件 | 数学定义 | 说明 |
|------|---------|------|
| Lorentz 双曲流形 | Def 2.1 | exp/log map, 测地距离, 切空间投影, 曲率缩放 |
| 球面流形 | Def 2.2 | 同上 |
| 欧氏流形 | Def 2.3 | trivial, 但需统一接口 |
| 积流形 H x S x E | Def 2.4 | split/combine, 可学习 kappa, origin point |
| 流形采样 | Def 6.1 | 均匀采样 (hyperbolic ball, sphere, Gaussian) |

**文件**: `manifolds/{base,lorentz,spherical,euclidean,product,utils}.py`
**验证**: exp/log 往返、距离公理、切向正交性、kappa!=1 测试

### Layer 1: flow/ — 流匹配

| 组件 | 数学定义 | 说明 |
|------|---------|------|
| 连续噪声先验 | Def 6.1 | 双曲/球面/欧氏空间的均匀采样 |
| 离散噪声先验 | Def 6.2 | per-relation Bernoulli(rho_k) |
| 测地线插值 x_t | Def 6.3 | exp(t * log(x_1)) 从噪声到数据 |
| 离散插值 E_t | Def 6.4 | shared z_ij Bernoulli mask |
| 向量场目标 u_t | Def 6.5 | (1/(1-t)) * log_{x_t}(x_1) |

**文件**: `flow/{continuous_flow,discrete_flow,joint_flow,noise}.py`
**验证**: t=0 得噪声, t=1 得数据; 向量场方向正确; z_ij 在 K 个关系间共享

### Layer 2: models/ — RieFormer 神经网络

| 组件 | 数学定义 | 说明 |
|------|---------|------|
| 输入编码 | Def 5.3-5.4 | MLP([pi(x) ‖ c_i ‖ m_i]) + edge MLP([E_t*W_rel ‖ E_t*C_R]) |
| 时间嵌入 | Def 5.2 | 正弦编码 + 可学习投影 |
| Module A: Manifold Attention | Def 5.5-5.8 | M-RoPE + 测地核 + edge bias |
| Module B: ATH-Norm | Def 5.9 | 自适应切空间归一化 |
| Module C: Edge Self-Update | Def 5.10-5.11 | factorized head/tail 聚合 |
| Module D: Cross-Interaction | Def 5.12-5.13 | Edge<->Node 双向 |
| Module E: Text Injection | Def 5.14-5.15 | cross-attention with C_V |
| VF Head | Def 5.16-5.17 | MLP -> 切空间投影 (含曲率因子) |
| Edge Head | Def 5.18-5.19 | per-relation MLP + relation Transformer + sigmoid |

**文件**: `models/{rieformer,rieformer_block,riemannfm,heads,normalization,positional}.py`, `models/attention/{geodesic,edge,text_cross}.py`
**验证**: forward pass shape 测试、梯度流测试、参数量检查

### Layer 3: losses/ — 损失函数

| 组件 | 数学定义 | 说明 |
|------|---------|------|
| L_cont | Def 6.6 | 黎曼范数 MSE + m_i masking |
| L_disc | Def 6.8 | weighted BCE, w_k+ = min((1-rho_k)/rho_k, w_max) |
| L_align | Def 6.9-6.10 | 节点级 graph-text contrastive loss |

**文件**: `losses/{flow_matching_loss,contrastive_loss,combined_loss}.py`
**验证**: loss 有限且正、梯度流到曲率参数

### Layer 4: optim/ + cli/pretrain.py — 训练循环

| 组件 | 数学定义 | 说明 |
|------|---------|------|
| Riemannian Adam | Algo 1 step 10 | geoopt 封装 + 双参数组 (模型 vs 曲率) |
| 曲率投影 | Algo 1 step 11 | kappa_h <- min(kappa_h, -eps), kappa_s <- max(kappa_s, eps) |
| Lightning Module | — | training_step + configure_optimizers |
| wandb logging | — | loss curves, curvature tracking |

**文件**: `optim/riemannian.py`, `cli/pretrain.py`, 新增 `training/lightning_module.py`
**验证**: wikidata_5m_mini 上 1000 步 loss 下降、无 NaN/Inf

## 实施顺序与依赖

```
Layer 0: manifolds ──────────────────────────────────┐
    依赖: 无                                          │
    输出: exp/log/dist/proj_tangent/sample/origin      │
                                                      ▼
Layer 1: flow ─────────────────── 依赖 manifolds 的 exp/log/sample
    输出: noise sampling, interpolation, VF target
                                                      │
Layer 2: models ─────────────── 依赖 manifolds + flow │
    输出: RieFormer forward: (x_t,E_t,t,C_V,C_R,m) -> (V_hat, P_hat)
                                                      │
Layer 3: losses ─────────────── 依赖 manifolds 的 norm │
    输出: L_cont + lambda*L_disc + mu*L_align          │
                                                      ▼
Layer 4: optim + pretrain ──── 组装所有组件
    输出: MVP — wikidata_5m_mini loss 下降 ⭐
```

每层完成后可独立测试:
- manifolds: 几何性质 (exp/log 往返, 距离三角不等式)
- flow: 插值端点 (t=0 噪声, t=1 数据)
- models: forward pass shape + 梯度流
- losses: 有限正值 + 梯度到 kappa
- pretrain: 端到端 loss 下降

## MVP 验证标准

在 wikidata_5m_mini (822 relations, 7833 entities, 22862 triples) 上:
- `make pretrain ARGS="model=rieformer_small data=wikidata_5m_mini"`
- 1000 步内 loss 单调下降
- 无 NaN/Inf
- 单 GPU < 24GB
- checkpoint/resume 正常