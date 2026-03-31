# RiemannFM 实验设计框架

## 1. 预训练

### 1.1 数据

**数据集**：WikiData5M（Wang et al., 2019）

| 统计量 | 数值 |
|--------|------|
| 实体数 | $\|\mathcal{V}\|$ =  4,594,485 |
| 关系类型数 $K$ | 822 |
| 训练三元组 | 20,614,279 |
| 验证三元组 | 5,163 |
| 测试三元组 | 5,133 |

每个实体附带 WikiData 标签和描述文本，关系类型同理。文本编码器 $\phi_{\mathrm{text}}$ 使用冻结的 Sentence-BERT（或同等规模的预训练语言模型），输出维度 $d_c = 768$。

### 1.2 子图采样策略

预训练时从 $\mathcal{K}$ 中在线采样子图：

1. 均匀随机选取种子节点 $v_{\mathrm{seed}} \sim \mathrm{Uniform}(\mathcal{V})$；
2. 以 $v_{\mathrm{seed}}$ 为中心做 $k$-hop 邻域扩展（$k = 2$）；
3. 若邻域节点数超过 $N_{\max}$，随机下采样至 $N_{\max}$；若不足，填充虚节点至 $N_{\max}$；
4. 诱导子图的边集为 $\mathcal{E}_\mathcal{G} = \{(v_i, r_k, v_j) \in \mathcal{E} \mid v_i, v_j \in \mathcal{V}_\mathcal{G}\}$。

### 1.3 训练配置

**Epoch 定义**：由于子图采样是有放回的随机过程，不定义传统 epoch。训练按**总步数**衡量，每步采样一个 mini-batch 的子图，每个子图独立采样时间步 $t \sim \mathrm{Uniform}(0, 1 - \epsilon_t)$。

| 配置项 | 值 |
|--------|------|
| 总训练步数 | 500K |
| 优化器 | Riemannian Adam |
| 学习率调度 | 线性 warmup（10K steps）+ 余弦衰减 |
| 梯度裁剪 | 最大范数 1.0 |
| 曲率投影 | 每步执行，$\epsilon_\kappa = 10^{-5}$ |
| $\epsilon_t$ | $10^{-3}$ |
| $\epsilon_p$ | $10^{-8}$ |

### 1.4 预训练验证

每 10K steps 在验证子图集（从 WikiData5M 官方 validation split 对应的子图）上计算：

$$\mathcal{L}_{\mathrm{val}} = \mathcal{L}_{\mathrm{cont}} + \lambda\,\mathcal{L}_{\mathrm{disc}}$$

保存验证损失最低的 checkpoint 用于下游任务初始化。

---

## 2. 下游任务一：知识图谱补全（KGC）

### 2.1 问题设定

给定知识图谱中已知的事实集合，预测缺失的事实 $(v_i, r_k, v_j)$。

### 2.2 数据集

WikiData5M transductive split（官方拆分）：

| 拆分 | 三元组数 |
|------|---------|
| Train | 20,614,279 |
| Validation | 5,163 |
| Test | 5,133 |

### 2.3 掩码策略

训练时对每个子图构造任务掩码：

- **边掩码**：对每条存在的边 $(i, j)$（$\mathbf{E}_{1,ij} \neq \mathbf{0}_K$），独立以概率 $p_{\mathrm{mask}}^E$ 设置 $M_{ij}^{\mathrm{task}} = 1$；
- **节点掩码**：对每个真实节点 $i$（$m_i = 1$），独立以概率 $p_{\mathrm{mask}}^V$ 设置 $m_i^{\mathrm{task}} = 1$，同时掩码其所有关联边。

### 2.4 微调配置

| 配置项 | 值 |
|--------|------|
| 初始化 | 预训练最佳 checkpoint |
| 微调步数 | 50K |
| 优化器 | Riemannian Adam |
| 学习率 | 从超参搜索确定 |
| 损失 | $\mathcal{L}_{\mathrm{KGC}} = \mathcal{L}_{\mathrm{cont}}^{\mathrm{task}} + \lambda_{\mathrm{KGC}}\,\mathcal{L}_{\mathrm{disc}}^{\mathrm{task}}$ |
| 验证频率 | 每 5K steps |
| Early stopping | patience = 5（基于验证 MRR） |

### 2.5 评估协议

对每个测试三元组 $(v_i, r_k, v_j)$，构造查询 $(v_i, r_k, ?)$，以 $\hat{\mathbf{P}}_{ij'}^{(k)}$ 对所有候选实体 $v_{j'}$ 排序。采用 **filtered setting**：排序时移除训练/验证集中已知的正确答案（待评估的三元组除外）。

**指标**：

| 指标 | 定义 |
|------|------|
| MRR | $\frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \frac{1}{\mathrm{rank}_q}$ |
| Hits@1 | $\frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \mathbb{1}[\mathrm{rank}_q \leq 1]$ |
| Hits@3 | $\frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \mathbb{1}[\mathrm{rank}_q \leq 3]$ |
| Hits@10 | $\frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \mathbb{1}[\mathrm{rank}_q \leq 10]$ |

### 2.6 Baselines

| 类别 | 方法 |
|------|------|
| 结构方法 | TransE, RotatE, CompGCN, NBFNet |
| 文本增强 | KEPLER, SimKGC, KGC-ERC |
| 几何方法 | MuRP（乘积流形），AttH（双曲注意力） |

---

## 3. 下游任务二：文本条件子图生成（T2G）

### 3.1 问题设定

给定文本查询 $q$（如"法国的首都及其主要河流"），生成一个与查询语义匹配的知识子图 $\mathcal{G}$。

### 3.2 数据集（自建）

从 WikiData5M 中构造评估数据集：

1. 设计三个难度层级的查询模板；
2. 对每个查询，从 WikiData5M 中提取对应的真实子图作为 ground truth；
3. 人工审核确保查询与子图的对应关系正确。

| 难度 | 查询示例 | 节点数范围 | 查询数 |
|------|---------|-----------|--------|
| 简单 | "法国的首都" | 2–5 | 200 |
| 中等 | "法国及其邻国的首都" | 5–10 | 200 |
| 复杂 | "欧洲主要国家的政治体制" | 10–20 | 100 |

拆分：训练查询 400 / 验证查询 50 / 测试查询 50。

### 3.3 微调配置

| 配置项 | 值 |
|--------|------|
| 初始化 | 预训练最佳 checkpoint |
| 微调步数 | 30K |
| 掩码策略 | 全部掩码（$m_i^{\mathrm{task}} = 1,\, M_{ij}^{\mathrm{task}} = 1,\, \forall i, j$） |
| 文本条件 | $\mathbf{C}_\mathcal{V} = \mathbf{1}_N \mathbf{c}_q^\top$（所有节点共享查询嵌入） |
| 损失 | $\mathcal{L}_{\mathrm{T2G}} = \mathcal{L}_{\mathrm{cont}} + \lambda_{\mathrm{T2G}}\,\mathcal{L}_{\mathrm{disc}} + \mu_{\mathrm{T2G}}\,\mathcal{L}_{\mathrm{align}}$ |
| 推理步数 $T$ | 从超参搜索确定 |
| 验证频率 | 每 5K steps |
| Early stopping | patience = 5（基于验证关系 F1） |

### 3.4 评估协议

推理时从噪声采样，经 $T$ 步去噪生成子图。将生成子图与真实子图通过**匈牙利算法**对齐（基于节点文本嵌入的余弦相似度做二部图最大匹配）。

**指标**：

| 指标 | 定义 |
|------|------|
| 节点 F1 | 匹配节点集合的 precision / recall / F1（匹配阈值：BERTScore > 0.85） |
| 关系 F1 | 在匹配的节点对上，预测边类型集合与真实边类型集合的 F1 |
| BERTScore | 生成子图节点描述与真实子图节点描述的平均 BERTScore |

### 3.5 Baselines

| 类别 | 方法 |
|------|------|
| 图生成 | DeFoG（离散流匹配）+ 文本条件适配，DiGress（离散扩散）+ 文本条件适配 |
| LLM 方法 | GPT-4 直接从文本生成三元组，再构建子图 |

---

## 4. 下游任务三：图异常检测（GAD）

### 4.1 问题设定

给定一个子图，检测其中的异常边和异常节点。**零样本**使用预训练模型，无需微调。

### 4.2 数据集

在 WikiData5M 上构造异常检测测试集：

**异常注入策略**：

| 异常类型 | 注入方式 | 说明 |
|---------|---------|------|
| 关系类型替换 | 将 $(v_i, r_k, v_j)$ 的 $r_k$ 随机替换为 $r_{k'} \neq r_k$ | 实体正确但关系错误 |
| 尾实体替换 | 将 $(v_i, r_k, v_j)$ 的 $v_j$ 随机替换为 $v_{j'} \neq v_j$ | 关系正确但指向错误实体 |
| 节点异常 | 将某节点的所有出边随机重连到随机实体 | 整个节点的邻域异常 |

**测试集构造**：

| 配置 | 值 |
|------|------|
| 测试子图数 | 500 |
| 每个子图节点数 | $N_{\max}$ |
| 异常比例 | 5%, 10%（构造两组测试集） |

### 4.3 推理配置

| 配置项 | 值 |
|--------|------|
| 模型 | 预训练 checkpoint（不微调） |
| 时间步集 $\mathcal{T}$ | 均匀网格，$|\mathcal{T}|$ 从超参搜索确定 |
| 噪声采样 | 对每个测试子图，采样多组 $(\mathbf{X}_0, \mathbf{E}_0)$ 取平均以降低方差 |
| 噪声采样次数 | 10 |

**异常分数计算**：

- 边异常分数：$S_{ij}^{(k)} = 1 - \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \hat{\mathbf{P}}_{ij}^{(k)}$
- 节点异常分数：$S_i = \frac{1}{|\mathcal{N}_i|}\sum_{j \in \mathcal{N}_i} \max_{k:\mathbf{E}_{1,ij}^{(k)}=1} S_{ij}^{(k)}$

### 4.4 评估协议

**指标**：

| 指标 | 定义 | 级别 |
|------|------|------|
| AUROC | ROC 曲线下面积 | 边级、节点级分别报告 |
| AP | 平均精度（Average Precision） | 边级、节点级分别报告 |

对每种异常类型和异常比例分别报告。

### 4.5 Baselines

| 类别 | 方法 |
|------|------|
| 经典图异常检测 | DOMINANT, AnomalyDAE, GAAN |
| 泛化型检测器 | ARC（NeurIPS 2024） |
| 简单基线 | 随机分数, PPR（Personalized PageRank） |

---

## 5. 超参数搜索

### 5.1 搜索策略

分两阶段：

**阶段 1（预训练超参）**：用小规模预训练（50K steps）+ 验证损失做网格搜索，确定关键超参后做完整 500K steps 预训练。

**阶段 2（微调超参）**：固定预训练 checkpoint，在各下游任务上分别搜索任务特定超参。

### 5.2 搜索空间

#### 预训练超参

| 超参数 | 搜索范围 | 说明 |
|--------|----------|------|
| 学习率 | $\{1\text{e-}4, 3\text{e-}4, 5\text{e-}4\}$ | Riemannian Adam |
| $\lambda$（边损失权重） | $\{0.1, 0.5, 1.0, 2.0\}$ | |
| $\mu$（对比损失权重） | $\{0.01, 0.1, 0.5\}$ | |
| $N_{\max}$（子图大小） | $\{32, 64, 128\}$ | |
| $L$（RieFormer 层数） | $\{4, 6, 8\}$ | |
| $d_v$（节点隐藏维度） | $\{128, 256, 512\}$ | |
| $n_h$（注意力头数） | $\{4, 8\}$ | |
| $\tau$（对比温度） | $\{0.05, 0.1, 0.2\}$ | |

#### KGC 微调超参

| 超参数 | 搜索范围 |
|--------|----------|
| 学习率 | $\{5\text{e-}5, 1\text{e-}4, 3\text{e-}4\}$ |
| $\lambda_{\mathrm{KGC}}$ | $\{0.5, 1.0, 2.0\}$ |
| $p_{\mathrm{mask}}^E$ | $\{0.1, 0.2, 0.3\}$ |
| $p_{\mathrm{mask}}^V$ | $\{0.0, 0.05, 0.1, 0.2\}$ |

#### T2G 微调超参

| 超参数 | 搜索范围 |
|--------|----------|
| 学习率 | $\{5\text{e-}5, 1\text{e-}4, 3\text{e-}4\}$ |
| $\lambda_{\mathrm{T2G}}$ | $\{0.5, 1.0, 2.0\}$ |
| $\mu_{\mathrm{T2G}}$ | $\{0.01, 0.1, 0.5\}$ |
| 推理步数 $T$ | $\{50, 100, 200, 500\}$ |

#### GAD 超参

| 超参数 | 搜索范围 |
|--------|----------|
| $|\mathcal{T}|$（时间步数） | $\{5, 10, 20\}$ |
| 噪声采样次数 | $\{5, 10, 20\}$ |

---

## 6. Ablation Studies

### 6.1 流形几何消融（回答：乘积流形是否必要？）

| 变体 | 描述 | 评估任务 |
|------|------|---------|
| Full | $\mathbb{H} \times \mathbb{S} \times \mathbb{R}$（完整模型） | KGC, T2G, GAD |
| $\mathbb{H}$ only | 仅双曲空间 | KGC, T2G, GAD |
| $\mathbb{S}$ only | 仅球面空间 | KGC |
| $\mathbb{R}$ only | 仅欧氏空间（退化为标准 Transformer） | KGC, T2G, GAD |
| $\mathbb{H} \times \mathbb{R}$ | 去掉球面 | KGC |
| $\mathbb{S} \times \mathbb{R}$ | 去掉双曲 | KGC |
| 固定曲率 | $\kappa_h, \kappa_s$ 不可学习 | KGC |

### 6.2 架构组件消融（回答：每个子模块贡献了多少？）

| 变体 | 描述 | 评估任务 |
|------|------|---------|
| − Manifold RoPE | 移除 Manifold RoPE，保留 Geodesic Kernel | KGC |
| − Geodesic Kernel | 移除 Geodesic Kernel，保留 Manifold RoPE | KGC |
| − 两者都移除 | 注意力分数仅用标准 QK 内积 + 边偏置 | KGC |
| − ATH-Norm | 用标准 LayerNorm 替代 | KGC |
| − 边流自更新 | 移除子模块 C，边嵌入仅通过节点→边注入更新 | KGC |
| − 文本条件注入 | 移除子模块 E，无文本信息 | KGC, T2G |

### 6.3 训练目标消融（回答：联合训练是否优于单独训练？）

| 变体 | 描述 | 评估任务 |
|------|------|---------|
| − $\mathcal{L}_{\mathrm{align}}$ | 去掉对比损失 | KGC, T2G |
| − $\mathcal{L}_{\mathrm{disc}}$ | 去掉边类型损失 | KGC, GAD |
| − $\mathcal{L}_{\mathrm{cont}}$ | 去掉向量场损失 | KGC |

### 6.4 效率与规模分析

| 实验 | 变量 | 评估任务 |
|------|------|---------|
| 推理步数 $T$ | $\{10, 50, 100, 200, 500\}$ | T2G（质量 vs 速度 trade-off） |
| 预训练数据规模 | 10%, 25%, 50%, 100% WikiData5M 子图 | KGC（scaling curve） |
| 子图大小 $N_{\max}$ | $\{16, 32, 64, 128, 256\}$ | KGC（容量 vs 计算 trade-off） |

---

## 7. 可复现性

| 项目 | 说明 |
|------|------|
| 随机种子 | 固定 3 个种子（42, 123, 456），报告均值 ± 标准差 |
| 硬件 | 报告 GPU 型号、数量、训练时间 |
| 代码 | 公开发布代码和预训练 checkpoint |
| 数据 | T2G 和 GAD 的自建数据集公开发布 |
| 超参 | 附录中报告所有最终选定的超参数值 |

---

## 8. 预估计算资源

| 阶段 | 估计资源 |
|------|---------|
| 预训练（500K steps） | 4–8 GPU × 2–5 天 |
| 超参搜索（预训练） | 约 10 组 × 50K steps ≈ 预训练成本的 1× |
| KGC 微调 | 单 GPU × 数小时 |
| T2G 微调 | 单 GPU × 数小时 |
| GAD 推理 | 单 GPU × 数小时 |
| Ablation（~15 个变体） | 约预训练成本的 3–5× |
| **总计** | 约 4–8 GPU × 2–3 周 |