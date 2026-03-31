# RiemannFM 完整实验设计

---

## 0. 预训练与下游任务的关系

### 预训练数据
WikiData5M 全部训练三元组（~2061 万），通过在线子图采样训练流匹配模型。

### 下游任务数据的区分

| 下游数据集 | 与预训练的关系 | 说明 |
|-----------|-------------|------|
| WikiData5M | 同域评估 | 官方 test split（~5K 三元组）从未作为预训练目标出现。预训练学习子图的生成分布，不记忆具体三元组。 |
| CoDEx-L, Wiki27K | 跨域（WikiData 子集） | 实体集和关系集与预训练有部分重叠，但子图结构、拆分方式完全独立。需替换关系嵌入和分类头。 |
| FB15k-237 | 跨域（Freebase） | 完全不同的实体集和关系类型集。需域适配层（新输入编码器 + 新关系嵌入 + 新分类头）。 |
| WN18RR | 跨域（WordNet） | 完全不同的域（词汇语义网络 vs 百科知识图谱）。关系仅 11 种，结构高度层次化。 |
| YAGO3-10 | 跨域（YAGO） | 实体源自 WikiData/Wikipedia，但关系类型集和拆分独立。 |
| T2G (自建) | 同域 | 从 WikiData5M 构造，查询和 ground truth 子图与预训练子图在构造方式上不同。 |
| GAD (自建) | 同域 | 零样本使用预训练模型，异常为合成注入。 |

### 论文中需明确声明
1. 预训练目标是子图的生成分布（流匹配），不是记忆具体三元组
2. WikiData5M 官方 test 三元组从未在预训练中作为监督信号
3. 跨域数据集（FB15k-237, WN18RR 等）提供了真正的迁移能力验证

---

## 1. 数据集总览

### 1.1 各数据集统计

| 数据集 | 实体数 | 关系数 | 训练 | 验证 | 测试 | 自带文本 | 角色 |
|--------|--------|--------|------|------|------|---------|------|
| WikiData5M | 4,594,485 | 822 | 20,614,279 | 5,163 | 5,133 | ✓ (Wikipedia) | 预训练 + 同域 KGC/T2G/GAD |
| CoDEx-L | 77,951 | 69 | 551,193 | 30,622 | 30,622 | ✓ (WikiData 描述) | 跨域 KGC |
| Wiki27K | ~27,000 | 214 | ~167,000 | — | — | ✓ (WikiData 描述) | 跨域 KGC |
| FB15k-237 | 14,541 | 237 | 272,115 | 17,535 | 20,466 | 需映射 | 跨域 KGC（经典 benchmark） |
| WN18RR | 40,943 | 11 | 86,835 | 3,034 | 3,134 | ✓ (WordNet 定义) | 跨域 KGC（层次结构） |
| YAGO3-10 | 123,182 | 37 | 1,079,040 | 5,000 | 5,000 | 部分 | 跨域 KGC（补充） |

### 1.2 文本获取策略

| 数据集 | 文本来源 | 处理方式 |
|--------|---------|---------|
| WikiData5M | 每个实体对应 Wikipedia 页面标题 + 首段描述 | 直接使用 |
| CoDEx-L | WikiData 标签 + 描述 | 直接使用 |
| Wiki27K | WikiData 标签 + 描述 | 直接使用 |
| FB15k-237 | Freebase 实体映射至 WikiData/Wikipedia | 使用 KEPLER/SimKGC 已有的文本映射 |
| WN18RR | WordNet synset 定义 + 同义词列表 | 拼接定义和同义词作为文本输入 |
| YAGO3-10 | YAGO 实体描述 / Wikipedia 映射 | 通过实体链接获取 |

---

## 2. 预训练

### 2.1 数据

**数据集**：WikiData5M（Wang et al., 2019）

| 统计量 | 数值 |
|--------|------|
| 实体数 $|\mathcal{V}|$ | 4,594,485 |
| 关系类型数 $K$ | 822 |
| 训练三元组 | 20,614,279 |
| 验证三元组 | 5,163 |
| 测试三元组 | 5,133 |

每个实体附带 WikiData 标签和描述文本，关系类型同理。

**文本编码器**：$\phi_{\mathrm{text}}$ 使用冻结的 Sentence-BERT（或同等规模的预训练语言模型），输出维度 $d_c = 768$。全程冻结，不参与梯度更新。

### 2.2 子图采样策略

预训练时从 $\mathcal{K}$ 中在线采样子图，每步独立采样一个 mini-batch：

1. 均匀随机选取种子节点 $v_{\mathrm{seed}} \sim \mathrm{Uniform}(\mathcal{V})$
2. 以 $v_{\mathrm{seed}}$ 为中心做 $k$-hop 邻域扩展（$k = 2$）
3. 若邻域节点数超过 $N_{\max}$，按度数加权随机下采样至 $N_{\max}$
4. 若不足 $N_{\max}$，填充虚节点
5. 诱导子图的边集为 $\mathcal{E}_\mathcal{G} = \{(v_i, r_k, v_j) \in \mathcal{E} \mid v_i, v_j \in \mathcal{V}_\mathcal{G}\}$

每个子图编码为五元组 $(\mathbf{X}, \mathbf{E}, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m})$。

### 2.3 训练配置

**Epoch 定义**：子图采样是有放回的随机过程，不定义传统 epoch。训练按**总步数**衡量。每步采样一个 mini-batch 的子图，每个子图独立采样时间步 $t \sim \mathrm{Uniform}(0, 1 - \epsilon_t)$。

| 配置项 | 值 |
|--------|------|
| 总训练步数 | 500K |
| 优化器 | Riemannian Adam |
| 学习率调度 | 线性 warmup（10K steps）+ 余弦衰减 |
| 梯度裁剪 | 最大范数 1.0 |
| 曲率投影 | 每步执行，$\kappa_h \leftarrow \min(\kappa_h, -\epsilon_\kappa)$，$\kappa_s \leftarrow \max(\kappa_s, \epsilon_\kappa)$ |
| $\epsilon_t$ | $10^{-3}$ |
| $\epsilon_p$ | $10^{-8}$ |
| $\epsilon_\kappa$ | $10^{-5}$ |
| 损失 | $\mathcal{L} = \mathcal{L}_{\mathrm{cont}} + \lambda\,\mathcal{L}_{\mathrm{disc}} + \mu\,\mathcal{L}_{\mathrm{align}}$ |

### 2.4 预训练验证

每 10K steps 在验证子图集上计算：

$$\mathcal{L}_{\mathrm{val}} = \mathcal{L}_{\mathrm{cont}} + \lambda\,\mathcal{L}_{\mathrm{disc}}$$

验证子图从 WikiData5M 官方 validation split 对应的实体邻域中采样（确保验证三元组包含在验证子图中）。保存验证损失最低的 checkpoint 用于下游任务初始化。

### 2.5 预训练超参数搜索空间

分两阶段搜索：先用小规模预训练（50K steps）+ 验证损失做网格搜索，确定关键超参后做完整 500K steps 预训练。

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
| $d_h, d_s, d_e$（子空间维度分配） | 总维度固定，比例搜索 | 如 $d_h : d_s : d_e = 2:1:1$ |
| $\sigma_0$（欧氏噪声标准差） | $\{0.5, 1.0, 2.0\}$ | |
| $R_{\mathbb{H}}$（双曲截断半径） | $\{2.0, 5.0, 10.0\}$ | |

---

## 3. 任务一：知识图谱补全——链接预测（KGC-LP）

### 3.1 任务定义
给定查询 $(v_i, r_k, ?)$ 或 $(?, r_k, v_j)$，从所有候选实体中预测缺失的实体。

### 3.2 实验矩阵

| 数据集 | 设定 | 论文位置 |
|--------|------|---------|
| WikiData5M (transductive) | 同域评估 | 主实验 |
| CoDEx-L | 跨域迁移（有文本） | 主实验 |
| FB15k-237 | 跨域迁移（经典） | 主实验 |
| WN18RR | 跨域迁移（层次结构） | 主实验 |
| YAGO3-10 | 跨域迁移（补充） | 附录 |
| Wiki27K | 跨域迁移（补充） | 附录 |

### 3.3 微调配置

| 配置项 | WikiData5M（同域） | 跨域数据集 |
|--------|-------------------|-----------|
| 初始化 | 预训练最佳 checkpoint | 预训练最佳 checkpoint |
| 域适配 | 无需 | 新 $\mathbf{W}_{\mathrm{rel}}^{\mathrm{new}} \in \mathbb{R}^{K_{\mathrm{new}} \times d_r}$，新分类头 |
| 骨干学习率 | $\eta_{\mathrm{base}}$ | $\eta_{\mathrm{base}} / 50$（或冻结） |
| 新参数学习率 | — | $\eta_{\mathrm{base}}$ |
| 输出头学习率 | $\eta_{\mathrm{base}} / 5$ | $\eta_{\mathrm{base}} / 5$ |
| 曲率学习率 | $\eta_{\mathrm{base}} / 10$ | $\eta_{\mathrm{base}} / 10$ |
| 微调步数 | 50K | 30K |
| 损失 | $\mathcal{L}_{\mathrm{KGC}} = \mathcal{L}_{\mathrm{cont}}^{\mathrm{task}} + \lambda_{\mathrm{KGC}}\,\mathcal{L}_{\mathrm{disc}}^{\mathrm{task}}$ | 同左 |
| 验证频率 | 每 5K steps | 每 2K steps |
| Early stopping | patience = 5（基于验证 MRR） | patience = 5 |

### 3.4 微调超参数搜索空间

| 超参数 | 搜索范围 |
|--------|----------|
| $\eta_{\mathrm{base}}$ | $\{5\text{e-}5, 1\text{e-}4, 3\text{e-}4\}$ |
| $\lambda_{\mathrm{KGC}}$ | $\{0.5, 1.0, 2.0\}$ |
| $p_{\mathrm{mask}}^E$（边掩码概率） | $\{0.1, 0.2, 0.3\}$ |
| $p_{\mathrm{mask}}^V$（节点掩码概率） | $\{0.0, 0.05, 0.1, 0.2\}$ |

### 3.5 评估指标

所有指标在 **filtered setting** 下计算，对 $(v_i, r_k, ?)$ 和 $(?, r_k, v_j)$ 两个方向取平均。

| 指标 | 定义 |
|------|------|
| **MRR** | 正确实体排名的倒数的均值（主指标） |
| **Hits@1** | 正确实体排在第 1 位的比例 |
| **Hits@3** | 正确实体排在前 3 位的比例 |
| **Hits@10** | 正确实体排在前 10 位的比例 |

### 3.6 Baselines

#### 组 1：结构嵌入方法

| 方法 | 发表 | 空间 |
|------|------|------|
| TransE | NeurIPS 2013 | 欧氏平移 |
| RotatE | ICLR 2019 | 复数旋转 |
| DistMult | ICLR 2015 | 双线性 |
| ComplEx | ICML 2016 | 复数双线性 |

#### 组 2：几何感知方法（核心对比组）

| 方法 | 发表 | 空间 | 对比意义 |
|------|------|------|---------|
| MuRP | NeurIPS 2019 | 双曲 × 欧氏 | 最直接的乘积流形对比 |
| AttH | NeurIPS 2020 | 双曲 + 注意力 | 双曲注意力机制 |
| UltraE | NeurIPS 2022 | 超双曲 | 统一几何框架 |

#### 组 3：文本增强方法

| 方法 | 发表 | 对比意义 |
|------|------|---------|
| KEPLER | TACL 2021 | KGE + PLM 联合训练 |
| SimKGC | ACL 2022 | 对比学习文本 KGC |
| CSProm-KG | ACL 2023 | 条件软提示 |
| SKG-KGC | TACL 2024 | 多级共享知识引导 |
| KGC-ERC | arXiv 2025 | 实体邻域 + 关系上下文（当前 Wikidata5M SOTA） |

#### 组 4：GNN 方法

| 方法 | 发表 | 对比意义 |
|------|------|---------|
| CompGCN | ICLR 2020 | 关系感知 GCN |
| NBFNet | NeurIPS 2021 | 神经路径推理 |

### 3.7 已知 SOTA 参考（WikiData5M Transductive）

| 方法 | MRR | H@1 | H@10 |
|------|-----|-----|------|
| DistMult | 0.253 | — | — |
| SimKGC | 0.358 | 0.313 | 0.448 |
| CSProm-KG | ~0.380 | — | — |
| KGC-ERC + Desc. | **0.433** | **0.412** | — |

---

## 4. 任务二：知识图谱补全——关系预测（KGC-RP）

### 4.1 任务定义
给定查询 $(v_i, ?, v_j)$，预测实体对之间的关系类型。

### 4.2 与链接预测的区别
- LP：固定关系，预测缺失实体（从 $|\mathcal{V}|$ 个候选中排序）
- RP：固定实体对，预测缺失关系（从 $K$ 个候选中排序/多标签分类）
- 我们的模型天然支持 RP：$\hat{\mathbf{P}}_{ij}^{(k)}$ 直接给出每种关系类型的概率

### 4.3 实验矩阵

| 数据集 | 论文位置 |
|--------|---------|
| WikiData5M | 主实验 |
| FB15k-237 | 主实验 |
| WN18RR | 附录 |

### 4.4 微调配置

复用 KGC-LP 的微调 checkpoint（同一次微调同时支持 LP 和 RP 评估，无需额外训练）。

### 4.5 评估指标

| 指标 | 定义 |
|------|------|
| **MRR** | 正确关系排名的倒数的均值 |
| **Hits@1** | 正确关系排在第 1 位的比例 |
| **Macro-F1** | 多标签分类的 Macro-F1（同一对实体可有多种关系） |

### 4.6 Baselines

复用 KGC-LP 的所有 baselines（对其打分函数适配为关系排序）。额外加入：

| 方法 | 说明 |
|------|------|
| KG-BERT | 将三元组建模为句子分类 |
| RP-ISS | 结构 + 语义融合的关系预测专用方法 |

---

## 5. 任务三：文本条件子图生成（T2G）

### 5.1 任务定义
给定文本查询 $q$，生成一个语义匹配的知识子图。

### 5.2 数据集（自建，基于 WikiData5M）

#### 构建流程
1. 从 WikiData5M 中选取高频实体和关系模式
2. 设计三个难度层级的查询模板
3. 程序化提取对应的 ground truth 子图
4. 人工审核查询与子图的对应关系

#### 数据集规模

| 难度 | 查询示例 | 目标子图规模 | 训练 | 验证 | 测试 |
|------|---------|------------|------|------|------|
| 简单 | "Albert Einstein 的国籍和出生地" | 2–5 节点 | 210 | 45 | 45 |
| 中等 | "法国及其邻国的首都" | 5–12 节点 | 140 | 30 | 30 |
| 复杂 | "诺贝尔物理学奖近五届获奖者及其机构" | 10–20 节点 | 70 | 15 | 15 |
| **合计** | | | **420** | **90** | **90** |

数据集将作为 ISWC 的资源贡献公开发布，包含查询文本、ground truth 子图、评估脚本。

### 5.3 微调配置

| 配置项 | 值 |
|--------|------|
| 初始化 | 预训练最佳 checkpoint |
| 微调步数 | 30K |
| 掩码策略 | 全部掩码（$m_i^{\mathrm{task}} = 1,\, M_{ij}^{\mathrm{task}} = 1,\, \forall i, j$） |
| 文本条件 | $\mathbf{C}_\mathcal{V} = \mathbf{1}_N \mathbf{c}_q^\top$（所有节点共享查询嵌入） |
| 损失 | $\mathcal{L}_{\mathrm{T2G}} = \mathcal{L}_{\mathrm{cont}} + \lambda_{\mathrm{T2G}}\,\mathcal{L}_{\mathrm{disc}} + \mu_{\mathrm{T2G}}\,\mathcal{L}_{\mathrm{align}}$ |
| 验证频率 | 每 5K steps |
| Early stopping | patience = 5（基于验证关系 F1） |

### 5.4 微调超参数搜索空间

| 超参数 | 搜索范围 |
|--------|----------|
| 学习率 | $\{5\text{e-}5, 1\text{e-}4, 3\text{e-}4\}$ |
| $\lambda_{\mathrm{T2G}}$ | $\{0.5, 1.0, 2.0\}$ |
| $\mu_{\mathrm{T2G}}$ | $\{0.01, 0.1, 0.5\}$ |
| 推理步数 $T$ | $\{50, 100, 200, 500\}$ |

### 5.5 评估协议

**Step 1：节点匹配**
用 $\phi_{\mathrm{text}}$ 嵌入的余弦相似度构造代价矩阵，匈牙利算法求最优节点匹配。

**Step 2：指标计算**

| 指标 | 定义 | 说明 |
|------|------|------|
| **节点 F1** | 匹配成功节点的 F1（阈值 BERTScore > 0.85） | 结构完整性 |
| **关系 F1** | 匹配节点对上，边类型集合的 F1 | 关系正确性 |
| **关系类型准确率** | 正确预测存在边的节点对上，关系类型的准确率 | 区分"有边"和"边类型对" |
| **Graph BERTScore** | 所有匹配节点对的 BERTScore 均值 | 语义质量 |

按难度分层报告 + 总体报告。

### 5.6 Baselines

#### 组 1：图生成方法 + 文本条件适配

| 方法 | 原始场景 | 适配方式 |
|------|---------|---------|
| DeFoG | 离散流匹配图生成（ICML 2025） | 添加文本交叉注意力模块 |
| DiGress | 离散扩散图生成（ICLR 2023） | 添加文本交叉注意力模块 |
| GGFlow | 离散流匹配 + 最优传输（ICLR 2025） | 添加文本交叉注意力模块 |

所有适配方法使用同一个冻结 $\phi_{\mathrm{text}}$，确保公平。

#### 组 2：LLM 直接生成

| 方法 | 描述 |
|------|------|
| GPT-4 | 提示 LLM 输出三元组列表 |
| GPT-4 + CoT | Chain-of-Thought 提示后生成 |

#### 组 3：消融基线

| 方法 | 描述 |
|------|------|
| RiemannFM (w/o text) | 无条件生成 + 事后文本匹配 |
| RiemannFM ($\mathbb{R}$ only) | 欧氏流匹配 + 文本条件 |

---

## 6. 任务四：图异常检测（GAD）

### 6.1 任务定义
给定一个子图，检测异常边和异常节点。**零样本**使用预训练模型，不微调。

### 6.2 数据集（自建，基于 WikiData5M）

#### 异常注入策略

| 类型 | 注入方式 | 占比 |
|------|---------|------|
| 关系替换 | $(v_i, r_k, v_j) \to (v_i, r_{k'}, v_j)$，$r_{k'}$ 随机 | 40% |
| 尾实体替换 | $(v_i, r_k, v_j) \to (v_i, r_k, v_{j'})$，$v_{j'}$ 随机 | 40% |
| 合成节点 | 新节点 + 随机边 | 20% |

替换确保不产生 WikiData5M 中已存在的三元组（避免假阳性）。

#### 测试集规模

| 配置 | 值 |
|------|------|
| 测试子图数 | 500 |
| 异常比例 | 2%, 5%, 10%（三组） |

### 6.3 推理配置

| 配置项 | 值 |
|--------|------|
| 模型 | 预训练 checkpoint（不微调） |
| 噪声采样次数 | 10（对每个测试子图取平均以降低方差） |

### 6.4 GAD 超参数搜索空间

| 超参数 | 搜索范围 |
|--------|----------|
| $|\mathcal{T}|$（时间步数） | $\{5, 10, 20\}$ |
| 噪声采样次数 | $\{5, 10, 20\}$ |

### 6.5 评估指标

| 指标 | 级别 |
|------|------|
| **AUROC** | 边级 + 节点级 |
| **AP** (Average Precision) | 边级 + 节点级 |
| **F1@best** | 边级 + 节点级 |

按异常类型 × 异常比例分别报告。

### 6.6 Baselines

#### 组 1：KGE 能量函数

| 方法 | 异常分数 |
|------|---------|
| TransE-score | $\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|$ |
| RotatE-score | $\|\mathbf{h} \circ \mathbf{r} - \mathbf{t}\|$ |
| DistMult-score | $-\mathbf{h}^\top \mathrm{diag}(\mathbf{r}) \mathbf{t}$ |

在 WikiData5M 上训练好的 KGE 模型，用打分函数的负值作为异常分数。

#### 组 2：图异常检测方法

| 方法 | 发表 |
|------|------|
| DOMINANT | SDM 2019 |
| AnomalyDAE | ICDE 2020 |
| ARC | NeurIPS 2024 |

#### 组 3：简单基线

| 方法 | 描述 |
|------|------|
| Random | 随机分数 |
| Degree-inv | 节点度数倒数 |
| PPR | Personalized PageRank |

---

## 7. Ablation Studies

### 7.1 流形几何消融

**问题**：乘积流形 $\mathbb{H} \times \mathbb{S} \times \mathbb{R}$ 是否比单一几何空间更好？

| 变体 | 空间 |
|------|------|
| Full | $\mathbb{H}^{d_h}_{\kappa_h} \times \mathbb{S}^{d_s}_{\kappa_s} \times \mathbb{R}^{d_e}$ |
| H-only | $\mathbb{H}^D$ |
| S-only | $\mathbb{S}^D$ |
| R-only | $\mathbb{R}^D$（标准 Transformer） |
| H+R | $\mathbb{H}^{d_h} \times \mathbb{R}^{d_e}$ |
| S+R | $\mathbb{S}^{d_s} \times \mathbb{R}^{d_e}$ |
| Fixed-κ | $\mathbb{H} \times \mathbb{S} \times \mathbb{R}$，$\kappa_h = -1, \kappa_s = 1$ 固定 |

评估：WikiData5M KGC-LP MRR（50K steps 缩减预训练 + 微调）。

### 7.2 架构组件消融

**问题**：RieFormer 的每个子模块贡献了多少？

| 变体 | 移除组件 |
|------|---------|
| Full | — |
| −MRoPE | Manifold RoPE |
| −GeoK | Geodesic Kernel |
| −MRoPE−GeoK | 两者都移除 |
| −ATH | ATH-Norm → 标准 LayerNorm |
| −EdgeSelf | 边流自更新（子模块 C） |
| −Cross | 双向交叉交互（子模块 D） |
| −TextCond | 文本条件注入（子模块 E） |

评估：KGC-LP MRR + T2G 关系 F1（对 −TextCond）。

### 7.3 训练目标消融

**问题**：三个损失的联合训练是否优于单独训练？

| 变体 | 损失配置 |
|------|---------|
| Full | $\mathcal{L}_{\mathrm{cont}} + \lambda\mathcal{L}_{\mathrm{disc}} + \mu\mathcal{L}_{\mathrm{align}}$ |
| −Align | 去掉对比损失 |
| −Disc | 去掉边类型损失 |
| −Cont | 去掉向量场损失 |
| Disc-only | 仅边类型损失 |

评估：KGC-LP MRR + GAD AUROC。

### 7.4 效率与规模分析

| 实验 | 变量 | 指标 |
|------|------|------|
| 推理步数 | $T \in \{10, 50, 100, 200, 500\}$ | T2G 关系 F1 + 推理时间 |
| 预训练规模 | 50K / 125K / 250K / 500K steps | KGC-LP MRR（scaling curve） |
| 子图大小 | $N_{\max} \in \{16, 32, 64, 128\}$ | KGC-LP MRR + 训练吞吐量 |

---

## 8. 可复现性

| 项目 | 说明 |
|------|------|
| 随机种子 | 固定 3 个种子（42, 123, 456），报告均值 ± 标准差 |
| 硬件 | 报告 GPU 型号、数量、训练时间 |
| 代码 | 公开发布代码和预训练 checkpoint |
| 数据 | T2G 和 GAD 的自建数据集公开发布（含构建脚本和评估脚本） |
| 超参 | 附录中报告所有最终选定的超参数值 |

---

## 9. 预估计算资源

| 阶段 | 估计资源 |
|------|---------|
| 预训练（500K steps） | 4–8 GPU × 2–5 天 |
| 超参搜索（预训练） | 约 10 组 × 50K steps ≈ 预训练成本的 1× |
| KGC-LP 微调（6 个数据集） | 单 GPU × 每数据集数小时 |
| KGC-RP 评估 | 无额外训练（复用 LP 的 checkpoint） |
| T2G 微调 | 单 GPU × 数小时 |
| GAD 推理 | 单 GPU × 数小时 |
| Ablation（~20 个变体） | 约预训练成本的 3–5× |
| **总计** | 约 4–8 GPU × 2–4 周 |

---

## 10. 论文结果呈现规划

### 主论文（15 页正文）

| 编号 | 类型 | 内容 |
|------|------|------|
| 表 1 | 主结果 | KGC-LP：WikiData5M + CoDEx-L + FB15k-237 + WN18RR，所有 baselines |
| 表 2 | 主结果 | KGC-RP：WikiData5M + FB15k-237 |
| 表 3 | 主结果 | T2G：按难度分层 + 总体，所有 baselines |
| 表 4 | 主结果 | GAD：按异常类型 × 异常比例，边级 + 节点级 |
| 表 5 | 消融 | 流形几何消融（A1） |
| 表 6 | 消融 | 架构组件消融（A2） |
| 图 1 | 分析 | 推理步数 $T$ vs 质量/速度 trade-off |
| 图 2 | 分析 | 预训练规模 scaling curve |
| 图 3 | 定性 | T2G 生成示例（2–3 个查询） |

### 附录

| 内容 |
|------|
| YAGO3-10, Wiki27K 上的 KGC-LP 结果 |
| WN18RR 上的 KGC-RP 结果 |
| 训练目标消融（A3）完整结果 |
| 子图大小消融完整结果 |
| T2G 更多定性案例（5–10 个） |
| GAD 定性案例 |
| 所有超参数最终选定值 |
| T2G 数据集构建详细说明 |
| GAD 异常注入详细说明 |

---

## 11. ISWC 审稿人关注点应对

| 可能质疑 | 应对策略 |
|---------|---------|
| "预训练和 KGC 评估用同一个 WikiData5M，有数据泄露吗？" | 官方 test 三元组从未在预训练中出现。预训练学的是子图生成分布，不记忆三元组。另有 4 个跨域数据集验证泛化。 |
| "这是 ML 论文，和语义网有什么关系？" | (1) 在 WikiData 上预训练，(2) 保留多关系 KG 语义结构，(3) 文本条件利用实体/关系描述，(4) T2G 直接服务于知识图谱构建，(5) GAD 服务于 KG 质量保证。 |
| "T2G 和 GAD 数据集是自建的" | 公开发布数据集 + 评估脚本。T2G 数据集本身是一个资源贡献。所有 baselines 用完全相同的数据和协议。 |
| "为什么不做 inductive KGC？" | 可在 WikiData5M inductive split 上补充。但 transductive 已是主流评估方式，且跨域迁移实验已展示泛化能力。 |
| "关系预测是否多余？" | LP 和 RP 考察模型不同维度的能力。LP 看实体排序，RP 看关系分类。我们的模型天然输出 $\hat{\mathbf{P}}_{ij}^{(k)}$，报告 RP 几乎零额外成本。 |
| "和 LLM-based KGC 方法比如何？" | 将 GPT-4 作为 T2G baseline。LLM-based KGC 方法（如 AgREE）作为 KGC-LP 的参考（注意它们调用外部 API，不完全可比）。 |
