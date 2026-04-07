# RiemannFM 实施计划

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
configs/                        # 70 个 Hydra YAML 配置
├── model/                      #   rieformer_{small,base,large}
├── manifold/                   #   product_h_s_e + 6 ablation variants
├── flow/                       #   joint, continuous_only, discrete_only
├── embedding/                  #   sbert, qwen3, xlm_roberta, nomic, none
├── data/                       #   wikidata_5m, wikidata_5m_mini, fb15k237, ...
├── training/                   #   pretrain, finetune
├── task/                       #   kgc_lp, kgc_rp, t2g, gad
├── ablation/                   #   8 architecture ablation configs
├── experiment/                 #   14 full experiment recipes
├── sweep/                      #   4 Optuna search spaces
├── accelerator/                #   gpu, cpu, mps, ddp, auto
├── logger/                     #   wandb, csv
├── paths/                      #   路径配置
├── download/                   #   下载配置
├── preprocess/                 #   预处理配置
└── eval/                       #   评估配置

src/riemannfm/
├── data/                       # ✅ 已实现 (~2,280 LOC)
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
├── manifolds/                  # ✅ 已实现 (~1,303 LOC)
│   ├── base.py                 #   抽象基类 RiemannFMManifold
│   ├── lorentz.py              #   Lorentz 双曲模型 (Def 2.1)
│   ├── spherical.py            #   球面流形 (Def 2.2)
│   ├── euclidean.py            #   欧氏空间 (Def 2.3)
│   ├── product.py              #   积流形 H x S x E (Def 2.4)
│   └── utils.py                #   辅助函数 (clamp, atanh, etc.)
├── flow/                       # ✅ 已实现 (~438 LOC)
│   ├── continuous_flow.py      #   测地线插值 + 向量场目标 (Def 6.3, 6.5)
│   ├── discrete_flow.py        #   离散插值 E_t (Def 6.4)
│   ├── joint_flow.py           #   联合流匹配编排 (Algorithm 1)
│   └── noise.py                #   连续/离散噪声采样 (Def 6.1, 6.2)
├── models/                     # ✅ 已实现 (~1,690 LOC)
│   ├── riemannfm.py            #   顶层模型: 文本投影 → 编码 → RieFormer → 预测头
│   ├── rieformer.py            #   L 层 Transformer 骨干
│   ├── rieformer_block.py      #   单 Transformer block (A-E 五模块)
│   ├── input_encoding.py       #   节点/边输入编码 (Def 5.3-5.4)
│   ├── positional.py           #   时间嵌入 (Def 5.2)
│   ├── normalization.py        #   ATH-Norm (Def 5.9) + Pre-Norm
│   ├── heads.py                #   VF Head (Def 5.16-5.17) + Edge Head (Def 5.18-5.19)
│   ├── lightning_module.py     #   Lightning 训练模块 (Algorithm 1 封装)
│   └── attention/
│       ├── geodesic.py         #   M-RoPE + 测地核 (Def 5.5-5.8)
│       └── edge.py             #   Edge Self-Update (Def 5.10-5.11)
├── losses/                     # ✅ 已实现 (~330 LOC)
│   ├── flow_matching_loss.py   #   L_cont (Def 6.6) + L_disc (Def 6.8)
│   ├── contrastive_loss.py     #   L_align (Def 6.9-6.10) + MLP 投影层
│   └── combined_loss.py        #   多任务加权: L_cont + λ·L_disc + μ·L_align
├── optim/                      # ✅ 已实现 (~98 LOC)
│   └── riemannian.py           #   双参数组 Riemannian Adam + 曲率投影
├── tasks/                      # ⬜ placeholder
├── utils/                      # ⬜ placeholder (metrics/ 子目录已创建)
└── cli/
    ├── download.py             # ✅ 已实现
    ├── preprocess.py           # ✅ 已实现
    ├── pretrain.py             # ✅ 已实现 (Hydra + wandb/CSV logging)
    ├── finetune.py             # ⬜ stub (NotImplementedError)
    ├── evaluate.py             # ⬜ stub (NotImplementedError)
    └── generate.py             # ⬜ stub (NotImplementedError)

tests/
├── conftest.py                 # 共享 fixtures
├── test_datamodule.py          # ✅ 339 行, data 模块测试
├── test_manifolds.py           # ✅ 499 行, 流形几何性质测试
├── test_flow.py                # ✅ 282 行, 流匹配测试
├── test_models.py              # ✅ 360 行, 模型 forward/gradient 测试
└── test_losses.py              # ✅ 236 行, 损失函数测试
```

## 已实现的功能

### 数据层 (data/)

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

### 流形层 (manifolds/)

| 功能 | 对应数学定义 | 状态 |
|------|------------|------|
| Lorentz 双曲流形 (exp/log/dist/proj_tangent) | Def 2.1 | ✅ |
| 球面流形 | Def 2.2 | ✅ |
| 欧氏流形 | Def 2.3 | ✅ |
| 积流形 H x S x E (split/combine, 可学习 kappa) | Def 2.4 | ✅ |
| 流形均匀采样 (hyperbolic ball, sphere, Gaussian) | Def 6.1 | ✅ |
| 曲率符号约束 (kappa_h < 0, kappa_s > 0) | — | ✅ |
| 6 种消融变体 (h_only, s_only, e_only, h_e, s_e, fixed) | — | ✅ |

### 流匹配层 (flow/)

| 功能 | 对应数学定义 | 状态 |
|------|------------|------|
| 连续噪声先验采样 | Def 6.1 | ✅ |
| 离散噪声先验 (per-relation Bernoulli) | Def 6.2 | ✅ |
| 测地线插值 x_t = exp(t * log(x_1)) | Def 6.3 | ✅ |
| 离散插值 E_t (shared z_ij Bernoulli mask) | Def 6.4 | ✅ |
| 向量场目标 u_t = (1/(1-t)) * log_{x_t}(x_1) | Def 6.5 | ✅ |
| 联合流匹配编排 (Algorithm 1) | — | ✅ |
| 消融支持 (continuous_only / discrete_only) | — | ✅ |

### 模型层 (models/)

| 功能 | 对应数学定义 | 状态 |
|------|------------|------|
| 输入编码 MLP([pi(x) ‖ c_i ‖ m_i]) | Def 5.3-5.4 | ✅ |
| 时间嵌入 (正弦 + 可学习投影) | Def 5.2 | ✅ |
| Module A: Manifold Attention (M-RoPE + 测地核) | Def 5.5-5.8 | ✅ |
| Module B: ATH-Norm | Def 5.9 | ✅ |
| Module C: Edge Self-Update | Def 5.10-5.11 | ✅ |
| Module D: Cross-Interaction (Edge<->Node) | Def 5.12-5.13 | ✅ |
| Module E: Text Injection (cross-attention) | Def 5.14-5.15 | ✅ |
| VF Head (MLP -> 切空间投影) | Def 5.16-5.17 | ✅ |
| Edge Head (per-relation MLP + sigmoid) | Def 5.18-5.19 | ✅ |
| Lightning Module (training_step + optimizers) | — | ✅ |

### 损失层 (losses/)

| 功能 | 对应数学定义 | 状态 |
|------|------------|------|
| L_cont 黎曼范数 MSE + m_i masking | Def 6.6 | ✅ |
| L_disc weighted BCE (w_k+ 自动平衡) | Def 6.8 | ✅ |
| L_align 节点级 graph-text contrastive + MLP 投影 | Def 6.9-6.10 | ✅ |
| 多任务加权组合损失 | — | ✅ |

### 优化层 (optim/)

| 功能 | 对应数学定义 | 状态 |
|------|------------|------|
| Riemannian Adam 双参数组 (模型 vs 曲率) | Algo 1 step 10 | ✅ |
| 曲率投影 (kappa_h <- min, kappa_s <- max) | Algo 1 step 11 | ✅ |

### 训练入口 (cli/)

| 功能 | 状态 |
|------|------|
| riemannfm-download | ✅ |
| riemannfm-preprocess | ✅ |
| riemannfm-pretrain (Hydra + wandb/CSV + checkpoint) | ✅ |
| riemannfm-finetune | ⬜ |
| riemannfm-evaluate | ⬜ |
| riemannfm-generate | ⬜ |

## 预训练阶段实施状态

```
Layer 0: manifolds ✅ 已完成 (~1,303 LOC, 499 行测试)
    exp/log/dist/proj_tangent/sample/origin, 积流形, 可学习曲率

Layer 1: flow ✅ 已完成 (~438 LOC, 282 行测试)
    noise sampling, interpolation, VF target, joint orchestration

Layer 2: models ✅ 已完成 (~1,690 LOC, 360 行测试)
    RieFormer forward: (x_t,E_t,t,C_V,C_R,m) -> (V_hat, P_hat)

Layer 3: losses ✅ 已完成 (~330 LOC, 236 行测试)
    L_cont + lambda*L_disc + mu*L_align

Layer 4: optim + pretrain ✅ 已完成 (~98 + 167 LOC)
    Riemannian Adam, curvature projection, Hydra CLI, wandb logging
```

## 测试覆盖

| 模块 | 测试文件 | 行数 | 覆盖范围 |
|------|---------|------|---------|
| data | test_datamodule.py | 339 | DataModule, collation, sampling, graph data |
| manifolds | test_manifolds.py | 499 | exp/log 往返, distance, tangent, curvature, product |
| flow | test_flow.py | 282 | continuous, discrete, joint, noise, time sampling |
| models | test_models.py | 360 | 组件测试, RieFormer forward, shape, gradients |
| losses | test_losses.py | 236 | continuous loss, discrete loss, combined loss |
| **合计** | **5 个文件** | **1,716** | **Layer 0-3 全覆盖** |

## MVP 验证标准

在 wikidata_5m_mini (822 relations, 7833 entities, 22862 triples) 上:
- `make pretrain ARGS="model=small data=wikidata_5m_mini"`
- 1000 步内 loss 单调下降
- 无 NaN/Inf
- 单 GPU < 24GB
- checkpoint/resume 正常

## 下一步：微调与评估阶段

### 待实现功能

#### tasks/ — 下游任务头

| 组件 | 说明 |
|------|------|
| KGC-LP | 链接预测 (Link Prediction) |
| KGC-RP | 关系预测 (Relation Prediction) |
| T2G | 文本到图生成 (Text-to-Graph) |
| GAD | 零样本图异常检测 (Graph Anomaly Detection) |

#### utils/ — 工具函数

| 组件 | 说明 |
|------|------|
| metrics | MRR, Hits@K, AUROC, F1 等评估指标 |
| logging | 分布式训练工具, 日志辅助 |

#### cli/ — 入口命令

| 命令 | 说明 |
|------|------|
| riemannfm-finetune | 下游任务微调 |
| riemannfm-evaluate | 模型评估 |
| riemannfm-generate | 图生成推理 |

### 依赖关系

```
pretrain checkpoint ─────────────────────────────┐
                                                  ▼
tasks/ (KGC/T2G/GAD heads) ──── 依赖 models/ + manifolds/
                                                  │
utils/metrics ───────────────── 依赖 task 定义     │
                                                  ▼
cli/finetune + evaluate + generate ── 组装所有组件
```
