# RiemannFM 下游任务技术规范

**核心思想**：RiemannFM 是子图级联合生成模型，下游任务通过**条件化流匹配**统一适配——掩码待预测部分，保留已知部分作为条件，模型通过 ODE 推理补全缺失结构。

---

## 目录

- [0. 预训练模型接口](#0-预训练模型接口)
- [1. 统一条件化框架](#1-统一条件化框架)
  - [1.1 任务掩码](#11-任务掩码)
  - [1.2 条件化含噪输入](#12-条件化含噪输入)
  - [1.3 条件化损失](#13-条件化损失)
  - [1.4 条件化推理](#14-条件化推理)
- [2. 知识图谱补全（KGC）](#2-知识图谱补全kgc)
  - [2.1 Link Prediction](#21-link-prediction)
  - [2.2 Relation Prediction](#22-relation-prediction)
  - [2.3 评估协议](#23-评估协议)
- [3. 文本条件子图生成（T2G）](#3-文本条件子图生成t2g)
  - [3.1 问题设定](#31-问题设定)
  - [3.2 微调](#32-微调)
  - [3.3 推理](#33-推理)
  - [3.4 实体解码](#34-实体解码)
  - [3.5 评估协议](#35-评估协议)
- [4. 图异常检测（GAD）](#4-图异常检测gad)
  - [4.1 问题设定](#41-问题设定)
  - [4.2 异常分数](#42-异常分数)
  - [4.3 评估协议](#43-评估协议)
- [5. Few-shot 关系学习](#5-few-shot-关系学习)
- [6. 流形可解释性分析](#6-流形可解释性分析)

---

## 0. 预训练模型接口

预训练完成后，模型提供以下可复用组件：

| 组件 | 符号 / 代码 | 用途 |
|------|------------|------|
| 乘积流形 | $\mathcal{M} = \mathbb{H} \times \mathbb{S} \times \mathbb{R}$ (`manifold`) | 几何运算（dist, exp, log, proj） |
| 实体嵌入 | $\mathbf{e}_i \in \mathbb{R}^D$ (`entity_emb`) | 实体的流形坐标（经 `proj_manifold` 投影后 $\in \mathcal{M}$） |
| RieFormer | $f_\theta$ (`model`) | backbone + 输出头，接收 $(\mathbf{X}_t, \mathbf{E}_t, t, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m})$，输出 $(\hat{\mathbf{V}}, \hat{\mathbf{P}}, \mathbf{H})$ |
| 联合流 | `flow` | 噪声采样 + 插值（微调时复用） |
| 文本投影 | $\mathbf{W}_{\mathrm{text}}$ (`text_proj`) | $d_c \to d_p$ 维度解耦 |

下游任务可选择：
- **使用完整 backbone**（子图级推理，适用于 KGC/T2G/GAD）
- **仅使用实体嵌入**（三元组级打分，轻量级 KGC baseline）

---

## 1. 统一条件化框架

所有下游任务共享同一个条件化流匹配框架，仅在**掩码策略**和**损失权重**上有所不同。

### 1.1 任务掩码

**定义 1.1（任务掩码）。**
- 节点任务掩码 $\mathbf{m}^{\mathrm{task}} \in \{0, 1\}^N$：$m_i^{\mathrm{task}} = 1$ 表示**待预测**，$= 0$ 表示**已知条件**。
- 边任务掩码 $\mathbf{M}^{\mathrm{task}} \in \{0, 1\}^{N \times N}$：$M_{ij}^{\mathrm{task}} = 1$ 表示待预测，$= 0$ 表示已知条件。

不同任务对应不同的掩码模式：

| 任务 | 节点掩码 $\mathbf{m}^{\mathrm{task}}$ | 边掩码 $\mathbf{M}^{\mathrm{task}}$ | 掩码比例 |
|------|--------|--------|----------|
| KGC-LP | 稀疏（仅掩码目标节点） | 稀疏（仅掩码目标边 + 关联边） | 低 |
| KGC-RP | $\mathbf{0}$ | 稀疏（仅掩码目标边） | 低 |
| T2G | $\mathbf{1}$ | $\mathbf{1}$ | 100% |
| GAD | $\mathbf{0}$ | $\mathbf{0}$ | 0%（不掩码） |

### 1.2 条件化含噪输入

**定义 1.2（条件化插值）。** 待预测部分正常走流匹配插值，已知部分保持数据端真实值：

$$\mathbf{x}_{t,i}^{\mathrm{cond}} = \begin{cases} \exp_{\mathbf{x}_{0,i}}(t \cdot \log_{\mathbf{x}_{0,i}}(\mathbf{x}_{1,i})) & m_i^{\mathrm{task}} = 1 \\ \mathbf{x}_{1,i} & m_i^{\mathrm{task}} = 0 \end{cases}$$

$$\mathbf{E}_{t,ij}^{\mathrm{cond}} = \begin{cases} z_{ij} \cdot \mathbf{E}_{1,ij} + (1 - z_{ij}) \cdot \mathbf{E}_{0,ij} & M_{ij}^{\mathrm{task}} = 1 \\ \mathbf{E}_{1,ij} & M_{ij}^{\mathrm{task}} = 0 \end{cases}$$

### 1.3 条件化损失

**定义 1.3（条件化微调损失）。** 损失仅在待预测部分上计算：

$$\mathcal{L}_{\mathrm{cont}}^{\mathrm{task}} = \frac{1}{\sum_i m_i^{\mathrm{task}} \cdot m_i} \sum_{i=1}^N m_i^{\mathrm{task}} \cdot m_i \cdot \|\hat{\mathbf{v}}_i - \mathbf{u}_{t,i}\|_{T_{\mathbf{x}_{t,i}}\mathcal{M}}^2$$

$$\mathcal{L}_{\mathrm{disc}}^{\mathrm{task}} = \frac{1}{|\mathcal{S}^{\mathrm{task}}|}\sum_{(i,j) \in \mathcal{S}^{\mathrm{task}}}\sum_{k=1}^K \mathrm{BCE}(\hat{\mathbf{P}}_{ij}^{(k)}, \mathbf{E}_{1,ij}^{(k)})$$

其中 $\mathcal{S}^{\mathrm{task}} \subseteq \{(i,j) : M_{ij}^{\mathrm{task}} = 1,\, m_i = m_j = 1\}$ 为待预测的真实节点边对集合（可配合负采样）。$\mathcal{L}_{\mathrm{cont}}^{\mathrm{task}}$ 同时要求 $m_i^{\mathrm{task}} = 1$（待预测）和 $m_i = 1$（非虚节点）。

### 1.4 条件化推理

**算法 1：条件化 ODE 推理（通用）**

**输入**：已知条件 $(\mathbf{X}_1^{\mathrm{known}}, \mathbf{E}_1^{\mathrm{known}})$，任务掩码 $(\mathbf{m}^{\mathrm{task}}, \mathbf{M}^{\mathrm{task}})$，文本条件 $\mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}$，步数 $T$
**输出**：补全后的 $(\mathbf{X}_1, \mathbf{E}_1)$

1. 待预测部分采样噪声：$\mathbf{X}_0, \mathbf{E}_0$
2. 已知部分设为真值：$\mathbf{X}_0^{\mathrm{known}} = \mathbf{X}_1^{\mathrm{known}}$
3. **for** $s = 0, 1, \ldots, T-1$ **do**：
   - $t = s / T$，$\Delta t = 1/T$
   - 构造条件化输入 $\mathbf{X}_t^{\mathrm{cond}}, \mathbf{E}_t^{\mathrm{cond}}$（定义 1.2）
   - $(\hat{\mathbf{V}}, \hat{\mathbf{P}}) = f_\theta(\mathbf{X}_t^{\mathrm{cond}}, \mathbf{E}_t^{\mathrm{cond}}, t, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m})$
   - 待预测节点：$\mathbf{x}_{t+\Delta t, i} = \exp_{\mathbf{x}_{t,i}}(\Delta t \cdot \hat{\mathbf{v}}_i)$ 若 $m_i^{\mathrm{task}} = 1$
   - 已知节点：$\mathbf{x}_{t+\Delta t, i} = \mathbf{x}_{1,i}$
   - 待预测边：以概率 $p_{\mathrm{flip}} = \min(\Delta t / (1-t),\, 1)$ 翻转为 $\mathbb{1}[\hat{\mathbf{P}}_{ij}^{(k)} > 0.5]$ 若 $M_{ij}^{\mathrm{task}} = 1$
   - 已知边：$\mathbf{E}_{t+\Delta t, ij} = \mathbf{E}_{1,ij}$
4. 返回 $(\mathbf{X}_T, \mathbf{E}_T)$，其中已知部分不变，待预测部分为生成结果

---

## 2. 知识图谱补全（KGC）

### 2.1 Link Prediction

**问题**：给定查询 $(v_i, r_k, ?)$，从所有候选实体中排序找到正确的尾实体。

#### 2.1.1 子图级方案（推荐）

利用 RieFormer backbone 的子图推理能力，将 LP 转化为**条件化边补全**。

**数据构造**：对查询 $(v_i, r_k, ?)$：
1. BFS 采样 $v_i$ 的 $h$-hop 邻域子图 $\mathcal{G}_q$，包含最多 $N_{\max}$ 个节点
2. 将候选尾实体 $v_j$ 加入子图（若不在邻域内，作为额外节点添加）

**掩码策略**：
- $M_{ij}^{\mathrm{task}} = 1$ 仅对查询节点 $v_i$ 到候选实体 $v_j$ 的边
- 其余边和所有节点保持已知：$M_{pq}^{\mathrm{task}} = 0$，$m_p^{\mathrm{task}} = 0$

**打分**：
$$\mathrm{score}(v_i, r_k, v_j) = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \hat{\mathbf{P}}_{ij}^{(k)}(f_\theta(\mathbf{X}_t^{\mathrm{cond}}, \mathbf{E}_t^{\mathrm{cond}}, t, \ldots))$$

在多个时间步 $\mathcal{T} = \{t_1, \ldots, t_{|\mathcal{T}|}\}$ 取平均，提升鲁棒性。

**批量候选评估**：对所有候选 $v_j \in \mathcal{V}$，需要高效计算 score。两种策略：
- **子图共享**：固定查询子图，逐批替换候选节点位置的实体嵌入，复用 backbone 中间状态
- **单步近似**：仅取 $t = 0$（纯噪声端），此时模型输出相当于"从噪声重建目标边的概率"，可一次性对所有候选打分

#### 2.1.2 三元组级方案（轻量 baseline）

不走 backbone，直接在预训练实体嵌入上打分（当前代码实现）：

$$\mathrm{score}(h, r, t) = \begin{cases} -d_\mathcal{M}(\exp_{\mathbf{e}_h}(\mathbf{r}_k),\, \mathbf{e}_t) & \text{manifold\_dist 模式} \\ \sum_d e_{h,d} \cdot w_{r,d} \cdot e_{t,d} & \text{bilinear 模式} \end{cases}$$

其中 $\mathbf{e}_h, \mathbf{e}_t = \mathrm{proj}_\mathcal{M}(\text{entity\_emb}(h)), \mathrm{proj}_\mathcal{M}(\text{entity\_emb}(t))$，$\mathbf{r}_k \in T_{\mathbf{e}_h}\mathcal{M}$ 为可学习关系向量。

**优势**：推理速度快（无 backbone 开销），适合大规模候选集。
**劣势**：不使用图结构上下文，等价于流形版 TransE/DistMult。

#### 2.1.3 微调

**子图级方案的微调损失**：
$$\mathcal{L}_{\mathrm{KGC\text{-}LP}} = \mathcal{L}_{\mathrm{cont}}^{\mathrm{task}} + \lambda\,\mathcal{L}_{\mathrm{disc}}^{\mathrm{task}}$$

**训练策略**：
- 每个训练三元组 $(h, r, t)$：采样 $h$ 的邻域子图，掩码目标边
- 随机以 $p_{\mathrm{mask}}^E$ 额外掩码其他边，增加训练信号
- 随机以 $p_{\mathrm{mask}}^V$ 掩码部分节点及其关联边，学习节点预测

**微调超参**：backbone 使用低学习率（$\sim 10^{-5}$），任务头使用高学习率（$\sim 10^{-3}$）。可选冻结 backbone（head-only 微调）。

### 2.2 Relation Prediction

**问题**：给定 $(v_i, ?, v_j)$，预测关系类型。

**方案**：子图级推理中，$\hat{\mathbf{P}}_{ij} \in [0,1]^K$ 天然就是 $K$ 维关系预测，**不需要额外分类头**。

**掩码策略**：
- $M_{ij}^{\mathrm{task}} = 1$ 仅对查询边 $(v_i, v_j)$
- 节点全部已知：$\mathbf{m}^{\mathrm{task}} = \mathbf{0}$

**打分**：
$$\mathrm{score}(v_i, r_k, v_j) = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \hat{\mathbf{P}}_{ij}^{(k)}$$

取 $\arg\max_k$ 为预测结果（单标签），或阈值化为多标签预测。

**微调损失**：
$$\mathcal{L}_{\mathrm{KGC\text{-}RP}} = \mathcal{L}_{\mathrm{disc}}^{\mathrm{task}}$$

RP 不涉及节点位置预测，$\mathcal{L}_{\mathrm{cont}}^{\mathrm{task}} = 0$。

### 2.3 评估协议

**Link Prediction**：

对每个测试三元组 $(h, r, t)$，分别构造尾预测查询 $(h, r, ?)$ 和头预测查询 $(?, r, t)$：

1. 对所有候选实体 $v_j \in \mathcal{V}$ 计算 $\mathrm{score}(h, r, v_j)$
2. 按得分降序排列
3. **Filtered 设定**：从排名中移除训练/验证集中已知为真的三元组（避免惩罚正确预测）
4. 记录正确答案 $t$ 的排名 $\mathrm{rank}_q$

指标：
- $\mathrm{MRR} = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \frac{1}{\mathrm{rank}_q}$
- $\mathrm{Hits}@n = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \mathbb{1}[\mathrm{rank}_q \leq n]$，$n \in \{1, 3, 10\}$

**Relation Prediction**：

- **Accuracy**：$\mathrm{Acc} = \frac{1}{|\mathcal{Q}|}\sum_q \mathbb{1}[\hat{r}_q = r_q]$
- **Hits@k**：前 $k$ 个预测中包含正确关系的比例
- **MRR**：与 LP 相同公式，但排序对象是关系类型

---

## 3. 文本条件子图生成（T2G）

### 3.1 问题设定

给定文本查询 $q \in \Sigma^*$（如 "Albert Einstein's academic career"），生成一个语义匹配的知识子图 $\mathcal{G} = (\mathcal{V}_\mathcal{G}, \mathcal{E}_\mathcal{G})$。

### 3.2 微调

**掩码策略**：全部掩码（无条件生成）：
$$m_i^{\mathrm{task}} = 1 \;\;\forall i, \qquad M_{ij}^{\mathrm{task}} = 1 \;\;\forall (i,j)$$

**文本条件替换**：所有节点共享查询文本嵌入：
$$\mathbf{c}_q = \phi_{\mathrm{text}}(q) \in \mathbb{R}^{d_c}, \qquad \mathbf{C}_\mathcal{V} = \mathbf{1}_N \mathbf{c}_q^\top \in \mathbb{R}^{N \times d_c}$$

$\mathbf{C}_\mathcal{R}$ 保持不变。

**微调损失**：全部掩码时退化为预训练形式，加入对齐损失引导文本语义：
$$\mathcal{L}_{\mathrm{T2G}} = \mathcal{L}_{\mathrm{cont}} + \lambda\,\mathcal{L}_{\mathrm{disc}} + \mu\,\mathcal{L}_{\mathrm{align}}$$

$\mathcal{L}_{\mathrm{align}}$ 鼓励生成子图的节点隐藏状态与查询文本 $\mathbf{c}_q$ 在对齐空间中接近。

**训练数据构造**：对训练集中每个子图 $\mathcal{G}$，用模板或 LLM 从子图内容生成文本摘要作为查询 $q$（逆向构造配对数据）。

### 3.3 推理

采用算法 1（条件化 ODE 推理），其中 $\mathbf{m}^{\mathrm{task}} = \mathbf{1}$，$\mathbf{M}^{\mathrm{task}} = \mathbf{1}$：

1. 采样 $\mathbf{X}_0 \sim p_0^\mathcal{M}$，$\mathbf{E}_0 \sim \prod_k \mathrm{Bernoulli}(\rho_k)$
2. ODE 积分 $T$ 步，文本条件为 $\mathbf{C}_\mathcal{V} = \mathbf{1}_N \mathbf{c}_q^\top$
3. 后处理（见 §3.4）

### 3.4 实体解码

ODE 输出流形坐标 $\mathbf{x}_{1,i} \in \mathcal{M}$，需要映射回离散实体：

**最近邻解码**：
$$\hat{v}_i = \arg\min_{v \in \mathcal{V}} d_\mathcal{M}(\mathbf{x}_{1,i},\, \mathrm{proj}_\mathcal{M}(\mathbf{e}_v))$$

**虚节点移除**：
$$m_{\mathrm{pred},i} = \mathbb{1}\!\left[d_\mathcal{M}(\mathbf{x}_{1,i}, \mathbf{x}_\varnothing) \geq \epsilon_{\mathrm{null}}\right]$$

流形原点附近的节点视为虚节点，移除后得到可变大小的子图。

**边二值化**：
$$\mathbf{E}_{1,ij}^{(k)} \leftarrow \mathbb{1}[\hat{\mathbf{P}}_{ij}^{(k)} > p_{\mathrm{thresh}}]$$

**去重**：多个生成节点可能解码到同一实体，合并重复节点并取并边。

### 3.5 评估协议

将生成子图 $\hat{\mathcal{G}}$ 与真实子图 $\mathcal{G}^*$ 对齐后评估：

**对齐**：使用最大权重二部匹配（Hungarian 算法），权重为节点间的余弦相似度（基于实体嵌入或文本嵌入）。

**指标**：
- **节点 F1**：匹配成功的节点比例（以 BERTScore 阈值判定）
  - Precision = |匹配节点| / |$\hat{\mathcal{V}}$|
  - Recall = |匹配节点| / |$\mathcal{V}^*$|
- **关系 F1**：在匹配节点对上，比较预测边集与真实边集
  - 分别对每种关系类型计算 micro-F1
- **BERTScore**：生成节点文本描述与真实节点文本的平均 BERTScore
- **图编辑距离（GED）**：对齐后的最小编辑操作数（归一化）

---

## 4. 图异常检测（GAD）

### 4.1 问题设定

给定子图 $\mathcal{G}$，检测异常边和异常节点。**零样本**：直接使用预训练模型，不需要微调。

核心直觉：预训练模型学会了正常 KG 的统计规律。对正常事实，模型能高置信度重建（$\hat{\mathbf{P}} \approx 1$）；对异常事实，重建置信度低。

### 4.2 异常分数

**算法 2：图异常检测**

**输入**：待检测子图 $(\mathbf{X}_1, \mathbf{E}_1, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m})$，时间步集 $\mathcal{T} = \{t_1, \ldots, t_{|\mathcal{T}|}\}$，采样次数 $S$
**输出**：边异常分数 $\{S_{ij}^{(k)}\}$，节点异常分数 $\{S_i\}$

1. **for** $s = 1, \ldots, S$ **do**：
   - 采样噪声 $\mathbf{X}_0^{(s)}, \mathbf{E}_0^{(s)}$
   - **for** $t \in \mathcal{T}$ **do**：
     - 构造含噪输入 $\mathbf{X}_t, \mathbf{E}_t$（标准插值，不使用任务掩码）
     - $(\hat{\mathbf{V}}, \hat{\mathbf{P}}) = f_\theta(\mathbf{X}_t, \mathbf{E}_t, t, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m})$
     - 记录 $\hat{\mathbf{P}}^{(s,t)}$

2. **边异常分数**：对每条存在的边 $(i, j, k)$（$\mathbf{E}_{1,ij}^{(k)} = 1$）：
$$S_{ij}^{(k)} = 1 - \frac{1}{S \cdot |\mathcal{T}|}\sum_{s=1}^{S}\sum_{t \in \mathcal{T}} \hat{\mathbf{P}}_{ij}^{(k),(s,t)}$$

3. **节点异常分数**：设 $\mathcal{N}_i = \{j : \mathbf{E}_{1,ij} \neq \mathbf{0}_K\}$ 为节点 $i$ 的出边邻居集：
$$S_i = \frac{1}{|\mathcal{N}_i|}\sum_{j \in \mathcal{N}_i} \max_{k:\,\mathbf{E}_{1,ij}^{(k)}=1} S_{ij}^{(k)}$$

**多次采样取平均**（$S > 1$）减少噪声采样的随机性。多时间步取平均覆盖不同去噪阶段的信号。

### 4.3 评估协议

**指标**：
- **AUROC**：异常分数作为预测值，真实标签为 0/1，计算 ROC 曲线下面积
- **AP (Average Precision)**：精确率-召回率曲线下面积
- 分别报告**边级别**和**节点级别**的指标

---

## 5. Few-shot 关系学习

**问题**：预训练时见过 $K$ 种关系，推理时出现新关系 $r_{\mathrm{new}} \notin \mathcal{R}$，仅给 $n$ 个示例三元组（$n \in \{1, 5, 10\}$）。

**RiemannFM 的天然优势**：文本条件链路使得新关系无需可学习参数即可接入。

**方案**：

1. **关系原型构造**：通过文本编码器获取新关系的嵌入：
$$\mathbf{c}_{r_{\mathrm{new}}} = \phi_{\mathrm{text}}(\mathrm{label}_{r_{\mathrm{new}}} \circ \mathrm{desc}_{r_{\mathrm{new}}}) \in \mathbb{R}^{d_c}$$

2. **模型适配（无需微调）**：
   - 边编码器：$\mathbf{E}_{t,ij} \bar{\mathbf{C}}_\mathcal{R}$ 中，新关系通过文本投影 $\mathbf{W}_{\mathrm{text}}$ 自动获得 $d_p$ 维表示
   - 边输出头：$\mathrm{MLP}_{\mathrm{rel\text{-}proto}}(\bar{\mathbf{c}}_{r_{\mathrm{new}}})$ 生成新关系的原型向量
   - 仅需扩展 $\mathbf{W}_{\mathrm{rel}}$ 多一行（或用零初始化）和 $\mathbf{b}$ 多一维

3. **可选微调**：在 $n$ 个示例上做 few-step 梯度更新：
   - 冻结 backbone 和所有已知关系的参数
   - 仅更新 $\mathbf{W}_{\mathrm{rel}}$ 的新行和 $\mathbf{b}$ 的新维
   - 避免在少量样本上过拟合

**评估**：在 $\mathcal{R}_{\mathrm{novel}}$ 上报告 MRR 和 Hits@k，与 meta-learning 方法（GMatching, MetaR）对比。

---

## 6. 流形可解释性分析

乘积流形的三个分量具有明确的几何语义，可用于下游任务的可解释性分析。

### 6.1 分量异常分解

对 GAD 任务，将节点的流形坐标分解为三个分量，分析异常来源：

$$d_\mathcal{M}(\mathbf{x}_i, \mathbf{x}_j)^2 = d_\mathbb{H}(\mathbf{x}_i^\mathbb{H}, \mathbf{x}_j^\mathbb{H})^2 + d_\mathbb{S}(\mathbf{x}_i^\mathbb{S}, \mathbf{x}_j^\mathbb{S})^2 + d_\mathbb{R}(\mathbf{x}_i^\mathbb{R}, \mathbf{x}_j^\mathbb{R})^2$$

| 异常主要来源 | 分量 | 几何含义 | 解释 |
|-------------|------|---------|------|
| $d_\mathbb{H}$ 偏大 | 双曲 | 层级位置 | 实体在 ontology 树中的深度/位置异常（如具体实例被放在上位概念位置） |
| $d_\mathbb{S}$ 偏大 | 球面 | 社区归属 | 实体的循环/社区结构异常（如节点被错误归入不相关的社区） |
| $d_\mathbb{R}$ 偏大 | 欧氏 | 属性特征 | 实体的属性向量异常（如数值属性明显偏离同类实体） |

### 6.2 曲率与数据集特征

预训练学到的曲率 $\kappa_h, \kappa_s$ 编码了数据集的全局几何特征：

- $|\kappa_h|$ 大 → KG 层级结构强（如 ontology 丰富的 Wikidata）
- $\kappa_s$ 大 → KG 循环结构紧密（如社交网络型知识图谱）
- 不同数据集微调后曲率变化方向 → 反映下游任务对几何结构的需求差异
