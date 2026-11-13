# 5. 模型架构：RieFormer

## 目录

- [5. 模型架构：RieFormer](#5-模型架构rieformer)
  - [目录](#目录)
  - [5.1 设计哲学](#51-设计哲学)
    - [5.1.1 核心观察](#511-核心观察)
    - [5.1.2 设计原则](#512-设计原则)
    - [5.1.3 架构语义约定](#513-架构语义约定)
  - [5.2 总体定义](#52-总体定义)
  - [5.3 输入编码层](#53-输入编码层)
    - [5.3.1 文本投影](#531-文本投影)
    - [5.3.2 坐标投影](#532-坐标投影)
    - [5.3.3 结构位置编码（RWPE）](#533-结构位置编码rwpe)
    - [5.3.4 时间嵌入](#534-时间嵌入)
    - [5.3.5 节点初始嵌入](#535-节点初始嵌入)
    - [5.3.6 按需边特征（无状态边表示）](#536-按需边特征无状态边表示)
  - [5.4 RieFormer 块](#54-rieformer-块)
    - [5.4.1 子模块 A：结构感知稀疏流形注意力](#541-子模块-a结构感知稀疏流形注意力)
      - [5.4.1.1 Manifold RoPE](#5411-manifold-rope)
      - [5.4.1.2 Geodesic Kernel](#5412-geodesic-kernel)
      - [5.4.1.3 稀疏注意力候选集](#5413-稀疏注意力候选集)
      - [5.4.1.4 稀疏流形注意力](#5414-稀疏流形注意力)
    - [5.4.2 共享组件：ATH-Norm 与 FFN](#542-共享组件ath-norm-与-ffn)
      - [5.4.2.1 ATH-Norm](#5421-ath-norm)
      - [5.4.2.2 节点 FFN](#5422-节点-ffn)
    - [5.4.3 子模块 B：节点 FFN 残差](#543-子模块-b节点-ffn-残差)
    - [5.4.4 子模块 C：文本交叉注意力](#544-子模块-c文本交叉注意力)
  - [5.5 输出投影层](#55-输出投影层)
    - [5.5.1 切空间投影](#551-切空间投影)
    - [5.5.2 向量场输出头](#552-向量场输出头)
    - [5.5.3 Node→Edge 对表示](#553-nodeedge-对表示)
    - [5.5.4 分解式边预测头](#554-分解式边预测头)
      - [5.5.4.1 存在性头](#5541-存在性头)
      - [5.5.4.2 条件类型头](#5542-条件类型头)
      - [5.5.4.3 稀疏评估策略](#5543-稀疏评估策略)
  - [5.6 排列等变性](#56-排列等变性)
  - [5.7 复杂度分析](#57-复杂度分析)
    - [5.7.1 计算复杂度](#571-计算复杂度)
    - [5.7.2 显存占用](#572-显存占用)
    - [5.7.3 消融开关](#573-消融开关)

---

## 5.1 设计哲学

### 5.1.1 核心观察

知识图谱子图具有三个关键结构特征，共同约束了生成模型的合理架构形态：

**观察 1（拓扑稀疏性）** 平均度数 $\bar{d} \ll N$：Wikidata5M 采样子图中节点对存在边的概率 $< 1\%$。任何在 $N^2$ 节点对上维护层间密集状态的设计都会让 $>99\%$ 的存储/计算作用于"无边"空位。

**观察 2（关系高维度）** 关系类型数 $K$ 往往很大（Wikidata5M 中 $K \approx 822$）。若用单一 sigmoid 头在 $N \times N \times K$ 上同时承担"是否有边"与"边是何类型"，正类占比约 $10^{-5}$ 量级，BCE 梯度被无边对主导，训练损失会塌缩至 $\log K$ 的平凡基线。

**观察 3（局部性主导）** 子图内的语义信息流以**结构局部性**为主：KG 补全的主要信号来自 1-2 跳邻居；流形上"几何接近但拓扑未连"的节点对正是候选生成边。全连接节点注意力 $O(N^2)$ 在这种结构下易学出虚假长程依赖。

### 5.1.2 设计原则

基于上述观察，RieFormer 遵循四条设计原则：

**原则 A（边预测分解为"存在性 × 类型"）** 将边预测分解为两个语义独立的头：
$$\hat{\mathbf{P}}_{ij}^{(k)} = \underbrace{\hat{s}_{ij}}_{\text{边存在性}} \cdot \underbrace{\hat{\pi}_{ij}^{(k)}}_{\text{条件类型分布}}$$
其中 $\hat{s}_{ij} \in [0,1]$ 为 sigmoid 标量（承担稀疏拓扑学习），$\hat{\pi}_{ij}^{(k)}$ 为**给定存在边时**的类型分布（单热图用 softmax，多热图用 sigmoid）。两个头损失量级独立、优化方向清晰，类型头仅在正样本对上训练，避免梯度稀释。

**原则 B（节点侧主导，边表示无状态）** 不维护层间密集边隐藏状态。边信息以两种方式参与计算：(i) 在节点注意力中作为**低维偏置** $\boldsymbol{\phi}_{ij}$ 注入（无参聚合，参数都在共享文本投影中）；(ii) 在输出头通过端点节点 hidden state 的组合 $\mathbf{z}_{ij}$ 按需构造。这将模型主存储从 $O(N^2 d_{e'} L)$ 降至 $O(N d_v L)$。

**原则 C（结构感知稀疏注意力）** 节点注意力限制在三类候选上：
$$\mathcal{N}(i) = \{i\} \cup \underbrace{\mathcal{N}_{\mathrm{edge}}(i)}_{\text{1-hop 结构邻居}} \cup \underbrace{\mathcal{N}_{\mathrm{knn}}(i)}_{\text{top-}k_{\mathrm{geo}}\text{ 几何近邻}} \cup \underbrace{\mathcal{N}_{\mathrm{global}}}_{\text{全局 landmark}}$$
三类候选承担不同功能：结构邻居传播已知拓扑；几何近邻发现"流形相似但拓扑未知"的生成候选；landmark 承载长程全局信息。

**原则 D（流形约束仅作用于输出）** 节点隐藏嵌入 $\mathbf{h}_i^V$ 生活在 Euclidean 空间，承担信息流；几何感知通过三个接口注入：(i) 输入坐标投影；(ii) 注意力的 Manifold RoPE / Geodesic Kernel / 边偏置 / ATH-Norm 曲率 FiLM；(iii) 输出的切空间投影与几何距离 skip-connection。多头注意力本质为 Euclidean 算子，与节点隐藏空间的 Euclidean 性不冲突；流形约束闭合于最终 $\hat{\mathbf{v}} \in T_{\mathbf{x}_t}\mathcal{M}$ 的投影。

### 5.1.3 架构语义约定

- $\mathbf{x}_{t,i} \in \mathcal{M}$、$\mathbf{E}_{t,ij} \in \{0,1\}^K$ 为**几何/离散状态**（state），在 $L$ 层中保持不变，仅作为注意力偏置与归一化条件被反复读取。
- $\mathbf{h}_i^{V,(l)} \in \mathbb{R}^{d_v}$ 为**节点隐藏嵌入**（latent），完全生活在 Euclidean 空间，承担层间信息流。
- 边信息通过无参聚合 $\boldsymbol{\phi}_{ij}$ 与结构邻接 $\mathcal{N}_{\mathrm{edge}}$ 参与每层注意力，但**不保留层间隐藏状态**。

---

## 5.2 总体定义

**定义 5.1（RieFormer）** 参数为 $\theta$ 的映射
$$f_\theta(\mathbf{X}_t, \mathbf{E}_t, t, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m}) = \left(\hat{\mathbf{V}},\, \hat{\mathbf{s}},\, \hat{\boldsymbol{\pi}};\, \mathbf{H}^{V,(L)}\right)$$

**输入**：$\mathbf{X}_t \in \mathcal{M}^N$，$\mathbf{E}_t \in \{0,1\}^{N \times N \times K}$，$t \in [0,1]$ 或逐节点 $\mathbf{t}_{\mathrm{node}} \in [0,1]^N$，$\mathbf{C}_\mathcal{V} \in \mathbb{R}^{N \times d_c}$，$\mathbf{C}_\mathcal{R} \in \mathbb{R}^{K \times d_c}$，$\mathbf{m} \in \{0,1\}^N$。

**输出**：
- $\hat{\mathbf{V}} = (\hat{\mathbf{v}}_1, \ldots, \hat{\mathbf{v}}_N)$，$\hat{\mathbf{v}}_i \in T_{\mathbf{x}_{t,i}}\mathcal{M}$：节点切向量预测。
- $\hat{\mathbf{s}} \in [0,1]^{N \times N}$：节点对边存在性预测。
- $\hat{\boldsymbol{\pi}} \in [0,1]^{N \times N \times K}$：条件边类型分布预测（训练/推理时仅在候选对集合上实际评估）。
- $\mathbf{H}^{V,(L)} \in \mathbb{R}^{N \times d_v}$：backbone 最终隐藏状态（供对齐损失 $\mathcal{L}_{\mathrm{align}}$、语义掩码识别损失 $\mathcal{L}_{\mathrm{mask\_c}}$ 使用）。

$\theta$ 包含所有可学习参数，含曲率 $\kappa_h, \kappa_s$。

---

## 5.3 输入编码层

**记号**：$[\mathbf{a} \| \mathbf{b}]$ 表示向量拼接。

### 5.3.1 文本投影

**定义 5.2（文本投影）** 预训练文本编码器的输出维度 $d_c$ 随编码器选择而变化（如 Qwen3-Embedding 的 $d_c = 768$）。为解耦模型与文本编码器，引入共享线性投影 $\mathbf{W}_c \in \mathbb{R}^{d_p \times d_c}$ 将文本条件映射到模型内部维度 $d_p$：
$$\bar{\mathbf{C}}_\mathcal{V} = \mathbf{C}_\mathcal{V} \mathbf{W}_c^\top \in \mathbb{R}^{N \times d_p}, \qquad \bar{\mathbf{C}}_\mathcal{R} = \mathbf{C}_\mathcal{R} \mathbf{W}_c^\top \in \mathbb{R}^{K \times d_p}$$

节点文本与关系文本**共享**同一投影矩阵。后续简记 $\bar{\mathbf{c}}_i \triangleq \bar{\mathbf{C}}_{\mathcal{V},i}$，$\bar{\mathbf{c}}_{r_k} \triangleq \bar{\mathbf{C}}_{\mathcal{R},k}$。

### 5.3.2 坐标投影

**定义 5.3（坐标投影 $\pi$）** $\pi: \mathcal{M} \to \mathbb{R}^{D'}$，其中 $D' = d_h + (d_s + 1) + d_e$：
$$\pi(\mathbf{x}) = \Big[\,\mathbf{x}^{\mathbb{H}}_{1:d_h} \,\Big\|\, \mathrm{LN}_{\mathbb{S}}(\mathbf{x}^{\mathbb{S}}) \,\Big\|\, \mathrm{LN}_{\mathbb{R}}(\mathbf{x}^{\mathbb{R}})\,\Big]$$

- $\mathbf{x}^{\mathbb{H}}_{1:d_h}$：丢弃 Lorentz 时轴 $x_0 = \sqrt{\|\mathbf{x}^{\mathbb{H}}_{1:}\|^2 + 1/|\kappa_h|}$。时轴是空间坐标与 $\kappa_h$ 的确定性函数，小-tangent 区批间方差近零，作为 MLP 输入会注入 batch-wide DC 偏置，导致投影头下游塌缩。
- $\mathrm{LN}_{\mathbb{S}}, \mathrm{LN}_{\mathbb{R}}$：分别作用于球面、欧氏分量的独立 LayerNorm，剥离 anchor（$s_0 \approx 1/\sqrt{\kappa_s}$、$\mathbf{0}_{d_e}$）附近的尺度 DC。
- 不对 $\mathbf{x}^{\mathbb{H}}_{1:d_h}$ 施加 LN，以保留 Lorentz 几何签名 $(-,+,\ldots,+)$ 的结构。

**代价与补偿**：$\pi$ 截断了 $\kappa_h, \kappa_s$ 到 encoder 入口的梯度通路，由 ATH-Norm（定义 5.12）将 $[\kappa_h, \kappa_s]$ 作为 FiLM 条件逐层注入回补。

### 5.3.3 结构位置编码（RWPE）

**定义 5.4（RWPE）** $k$-步随机游走位置编码。对节点 $i$：
$$\mathbf{p}_i = \left[(\tilde{\mathbf{A}})_{ii},\, (\tilde{\mathbf{A}}^2)_{ii},\, \ldots,\, (\tilde{\mathbf{A}}^k)_{ii}\right] \in \mathbb{R}^{d_{\mathrm{pe}}}$$

其中 $\tilde{\mathbf{A}} = \mathbf{D}^{-1}\mathbf{A}$ 为行归一化邻接矩阵，$\mathbf{A}_{ij} = \mathbb{1}[\mathbf{E}_{t,ij} \neq \mathbf{0}_K \lor \mathbf{E}_{t,ji} \neq \mathbf{0}_K]$（**按存在性二值化**，不区分关系类型，对称化保证 $\tilde{\mathbf{A}}$ 归一化良态），$d_{\mathrm{pe}} = k$。

**作用**：对语义掩码节点（文本替换为共享 `mask_emb`），若几何位置 $\mathbf{x}$ 相似，backbone 无法区分同构位置上的不同实体，$\mathcal{L}_{\mathrm{mask\_c}}$ 塌缩至 $\log M$ 基线。RWPE 为 identity-free 的结构签名（仅依赖邻接），打破符号对称性而不泄漏实体身份。

### 5.3.4 时间嵌入

**定义 5.5（时间嵌入）** 设 $d_t \in \mathbb{Z}_{>0}$ 为偶数，正弦位置编码：
$$\boldsymbol{\psi}(t) = [\sin(\omega_1 t),\, \ldots,\, \sin(\omega_{d_t/2} t),\, \cos(\omega_1 t),\, \ldots,\, \cos(\omega_{d_t/2} t)] \in \mathbb{R}^{d_t}$$
其中 $\omega_l = 10000^{-2l/d_t}$。两层 MLP 投影：
$$\mathbf{t}_{\mathrm{emb}} = \mathbf{W}_2\,\mathrm{SiLU}(\mathbf{W}_1 \boldsymbol{\psi}(t) + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^{d_t}$$

支持**逐节点**时间：当输入为 $\mathbf{t}_{\mathrm{node}} \in [0,1]^N$ 时（对应节点三分区场景），$\mathbf{t}_{\mathrm{emb}} \in \mathbb{R}^{N \times d_t}$ 逐节点计算；否则为子图级标量并广播至 $N$。

### 5.3.5 节点初始嵌入

**定义 5.6（节点初始嵌入）**
$$\mathbf{h}_i^{V,(0)} = \mathrm{MLP}_{\mathrm{node}}\!\left([\pi(\mathbf{x}_{t,i}) \,\|\, \bar{\mathbf{c}}_i \,\|\, \mathbf{p}_i \,\|\, m_i]\right) + \mathbf{W}_{\mathrm{tp}}\,\mathbf{t}_{\mathrm{emb},i} \in \mathbb{R}^{d_v}$$

输入维度 $D' + d_p + d_{\mathrm{pe}} + 1$，$\mathrm{MLP}_{\mathrm{node}}$ 为两层 MLP，$\mathbf{W}_{\mathrm{tp}} \in \mathbb{R}^{d_v \times d_t}$ 为时间投影矩阵。此输入层时间投影提供全局时间锚点，与 ATH-Norm 的逐层时间条件互补：前者给出全局时间感，后者提供自适应调节。

### 5.3.6 按需边特征（无状态边表示）

**定义 5.7（按需边特征）** 对每对 $(i,j) \in [N]^2$，按需计算一个低维 edge feature：
$$\boldsymbol{\phi}_{ij} = \begin{cases}
\displaystyle \frac{1}{\max(1, \|\mathbf{E}_{t,ij}\|_1)} \sum_{k=1}^K \mathbf{E}_{t,ij}^{(k)} \bar{\mathbf{c}}_{r_k} \in \mathbb{R}^{d_p} & \text{若 } \mathbf{E}_{t,ij} \neq \mathbf{0}_K \\[0.5em]
\mathbf{e}_\varnothing \in \mathbb{R}^{d_p} & \text{否则}
\end{cases}$$

即激活关系文本嵌入的**平均**（单热时即该关系文本本身）。$\mathbf{e}_\varnothing \in \mathbb{R}^{d_p}$ 为可学习"无边" null embedding。

**性质**：
- **无参聚合**：$\boldsymbol{\phi}_{ij}$ 本身无独立参数，所有参数共享自 $\mathbf{W}_c$ 与 $\mathbf{e}_\varnothing$。
- **无层间状态**：$\boldsymbol{\phi}_{ij}$ 仅依赖于 $\mathbf{E}_t$ 与关系文本，在每层注意力中按需重新使用（广播），不随层更新。
- **稀疏友好**：实际实现中只对 $\mathbf{E}_{t,ij} \neq \mathbf{0}_K$ 的对显式计算，其余对共享 $\mathbf{e}_\varnothing$ 广播。显存开销为 $O(E d_p + d_p)$，其中 $E$ 为激活边数（典型 $O(N)$）。

---

## 5.4 RieFormer 块

RieFormer 由 $L \in \mathbb{Z}_{>0}$ 个相同结构的块堆叠而成。第 $l$ 层（$l \in [L]$）接收节点嵌入 $\mathbf{H}^{V,(l-1)} \in \mathbb{R}^{N \times d_v}$，输出 $\mathbf{H}^{V,(l)} \in \mathbb{R}^{N \times d_v}$。每块按以下顺序执行三个子模块（Pre-Norm 风格，归一化在子模块内部、变换之前执行）：

- **A**：结构感知稀疏流形注意力（ATH-Norm + Sparse Manifold Attention）
- **B**：节点 FFN 残差
- **C**：文本交叉注意力 + FFN 残差

### 5.4.1 子模块 A：结构感知稀疏流形注意力

设 $n_h \in \mathbb{Z}_{>0}$ 为头数，$d_{\mathrm{head}} = d_v / n_h$（要求 $n_h \mid d_v$ 且 $2 \mid d_{\mathrm{head}}$）。

#### 5.4.1.1 Manifold RoPE

**定义 5.8（Manifold RoPE）** 对第 $s \in [n_h]$ 个头，频率 $\omega_l^{(s)} = 10000^{-2l/d_{\mathrm{head}}}$，$l \in [d_{\mathrm{head}}/2]$。节点对 $(i,j)$ 的角度由乘积流形测地距离生成：
$$\theta_{ij,l}^{(s)} = \omega_l^{(s)} \cdot d_\mathcal{M}(\mathbf{x}_{t,i}, \mathbf{x}_{t,j})$$

旋转矩阵 $\mathbf{R}(\boldsymbol{\theta}_{ij}^{(s)}) \in \mathbb{R}^{d_{\mathrm{head}} \times d_{\mathrm{head}}}$ 为 $d_{\mathrm{head}}/2$ 个 $2 \times 2$ 旋转块的块对角矩阵：
$$\mathbf{R}(\boldsymbol{\theta}_{ij}^{(s)}) = \mathrm{diag}\!\left(\begin{pmatrix}\cos\theta_{ij,l}^{(s)} & -\sin\theta_{ij,l}^{(s)} \\ \sin\theta_{ij,l}^{(s)} & \cos\theta_{ij,l}^{(s)}\end{pmatrix}\right)_{l=1}^{d_{\mathrm{head}}/2}$$

#### 5.4.1.2 Geodesic Kernel

**定义 5.9（Geodesic Kernel）** 对第 $s$ 个头：
$$\kappa^{(s)}(\mathbf{x}_{t,i}, \mathbf{x}_{t,j}) = w_\mathbb{H}^{(s)} \kappa_\mathbb{H}(\mathbf{x}_{t,i}^\mathbb{H}, \mathbf{x}_{t,j}^\mathbb{H}) + w_\mathbb{S}^{(s)} \kappa_\mathbb{S}(\mathbf{x}_{t,i}^\mathbb{S}, \mathbf{x}_{t,j}^\mathbb{S}) + w_\mathbb{R}^{(s)} \kappa_\mathbb{R}(\mathbf{x}_{t,i}^\mathbb{R}, \mathbf{x}_{t,j}^\mathbb{R})$$
其中：
- $\kappa_\mathbb{H}(\mathbf{a},\mathbf{b}) = -d_\mathbb{H}(\mathbf{a},\mathbf{b})$
- $\kappa_\mathbb{S}(\mathbf{a},\mathbf{b}) = \kappa_s \cdot \mathbf{a}^\top \mathbf{b}$
- $\kappa_\mathbb{R}(\mathbf{a},\mathbf{b}) = -\|\mathbf{a}-\mathbf{b}\|_2^2$

$w_\mathbb{H}^{(s)}, w_\mathbb{S}^{(s)}, w_\mathbb{R}^{(s)} \in \mathbb{R}$ 为每头独立的可学习权重。

#### 5.4.1.3 稀疏注意力候选集

**定义 5.10（稀疏注意力候选集）** 节点 $i$ 的注意力候选集
$$\mathcal{N}(i) = \{i\} \cup \mathcal{N}_{\mathrm{edge}}(i) \cup \mathcal{N}_{\mathrm{knn}}(i) \cup \mathcal{N}_{\mathrm{global}}$$

- **结构邻居**：$\mathcal{N}_{\mathrm{edge}}(i) = \{j \in [N] : j \neq i,\, m_j = 1,\, (\mathbf{E}_{t,ij} \neq \mathbf{0}_K \lor \mathbf{E}_{t,ji} \neq \mathbf{0}_K)\}$，传播已知拓扑信息。
- **几何近邻**：$\mathcal{N}_{\mathrm{knn}}(i) = \mathrm{TopK}_{j \in [N] \setminus (\{i\} \cup \mathcal{N}_{\mathrm{edge}}(i)),\, m_j = 1}\!\left(-d_\mathcal{M}(\mathbf{x}_{t,i}, \mathbf{x}_{t,j}),\, k_{\mathrm{geo}}\right)$，发现"流形相似但拓扑未知"的生成候选。从 $\mathcal{N}_{\mathrm{edge}}(i)$ 补集中选取以避免重复。
- **全局 landmark**：$\mathcal{N}_{\mathrm{global}} \subseteq [N]$ 为按节点度数选取的 top-$k_g$ 高中心性节点，所有节点可注意到，承载长程信息。以度数（排列不变量）选择保证排列等变性。

所有候选需满足 $m_j = 1$（屏蔽虚节点）。$|\mathcal{N}(i)| \leq 1 + |\mathcal{N}_{\mathrm{edge}}(i)| + k_{\mathrm{geo}} + k_g$。默认超参 $k_{\mathrm{geo}} = 8$，$k_g = 4$。

#### 5.4.1.4 稀疏流形注意力

**定义 5.11（稀疏流形注意力，Pre-Norm）** Pre-Norm 归一化：
$$\bar{\mathbf{h}}_i^V = \mathrm{ATH\text{-}Norm}(\mathbf{h}_i^{V,(l-1)},\, \mathbf{t}_{\mathrm{emb},i},\, [\kappa_h, \kappa_s])$$

QKV 投影（$s \in [n_h]$）：
$$\mathbf{q}_i^{(s)} = \mathbf{W}_Q^{(s)}\bar{\mathbf{h}}_i^V,\quad \mathbf{k}_j^{(s)} = \mathbf{W}_K^{(s)}\bar{\mathbf{h}}_j^V,\quad \mathbf{v}_j^{(s)} = \mathbf{W}_V^{(s)}\bar{\mathbf{h}}_j^V \in \mathbb{R}^{d_{\mathrm{head}}}$$
其中 $\mathbf{W}_Q^{(s)}, \mathbf{W}_K^{(s)}, \mathbf{W}_V^{(s)} \in \mathbb{R}^{d_{\mathrm{head}} \times d_v}$。

注意力分数（仅对 $j \in \mathcal{N}(i)$ 计算）：
$$a_{ij}^{(s)} = \underbrace{\frac{(\mathbf{R}(\boldsymbol{\theta}_{ij}^{(s)})\mathbf{q}_i^{(s)})^\top \mathbf{k}_j^{(s)}}{\sqrt{d_{\mathrm{head}}}}}_{\text{RoPE 几何-语义匹配}} + \underbrace{\beta^{(s)}\kappa^{(s)}(\mathbf{x}_{t,i}, \mathbf{x}_{t,j})}_{\text{测地核偏置}} + \underbrace{\mathbf{w}_\phi^{(s)\top}\boldsymbol{\phi}_{ij} + \mathbf{w}_{\phi'}^{(s)\top}\boldsymbol{\phi}_{ji}}_{\text{有向边类型偏置}} + \underbrace{b_\mathcal{N}^{(s)}(i,j)}_{\text{候选集类型偏置}}$$

- $\beta^{(s)} \in \mathbb{R}$：可学习核缩放。
- $\mathbf{w}_\phi^{(s)}, \mathbf{w}_{\phi'}^{(s)} \in \mathbb{R}^{d_p}$：前向/反向边偏置权重，支持有向边的非对称注入。
- $b_\mathcal{N}^{(s)}(i,j)$：根据 $j$ 属于 $\{i\}$、edge、knn、global 哪个子集的可学习标量偏置（四个值），使模型学习不同来源的相对重要性。重叠时按 edge > self > knn > global 优先级归类。

$j \notin \mathcal{N}(i)$ 时 $a_{ij}^{(s)} = -\infty$（softmax 零权重）。注意力权重与输出：
$$\alpha_{ij}^{(s)} = \mathrm{softmax}_{j \in \mathcal{N}(i)}(a_{ij}^{(s)}),\qquad \mathbf{o}_i^{(s)} = \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(s)} \mathbf{v}_j^{(s)}$$

多头拼接与输出投影：
$$\mathrm{MHA}_i = \mathbf{W}_O [\mathbf{o}_i^{(1)} \| \cdots \| \mathbf{o}_i^{(n_h)}] \in \mathbb{R}^{d_v},\quad \mathbf{W}_O \in \mathbb{R}^{d_v \times d_v}$$

残差连接（跳过归一化前的原始嵌入）：
$$\tilde{\mathbf{h}}_i^V = \mathbf{h}_i^{V,(l-1)} + \mathrm{Drop}(\mathrm{MHA}_i)$$

**设计说明**：有向边偏置 $\boldsymbol{\phi}_{ij}$ 与 $\boldsymbol{\phi}_{ji}$ 的分离注入让节点在聚合时感知前向/反向两种关系语义（区分"$i$ 指向 $j$"与"$j$ 指向 $i$"）；$\mathcal{N}_{\mathrm{edge}}$ 候选集限制承担了"边定义邻接"的结构约束。二者共同实现边→节点的信息流，无需独立模块。

### 5.4.2 共享组件：ATH-Norm 与 FFN

#### 5.4.2.1 ATH-Norm

**定义 5.12（ATH-Norm，自适应时间条件归一化）** 设特征维度 $d \in \mathbb{Z}_{>0}$，时间嵌入维度 $d_t$，可选辅助条件维度 $d_{\mathrm{cond}} \geq 0$（如曲率标量 $[\kappa_h, \kappa_s]$，$d_{\mathrm{cond}} = 2$；$d_{\mathrm{cond}} = 0$ 时该通道关闭）。对输入 $\mathbf{h} \in \mathbb{R}^d$、时间嵌入 $\mathbf{t}_{\mathrm{emb}} \in \mathbb{R}^{d_t}$、可选条件 $\mathbf{c}_{\mathrm{cond}} \in \mathbb{R}^{d_{\mathrm{cond}}}$：
$$\mathrm{ATH\text{-}Norm}(\mathbf{h},\, \mathbf{t}_{\mathrm{emb}},\, \mathbf{c}_{\mathrm{cond}}) = \boldsymbol{\gamma}(\mathbf{t}_{\mathrm{emb}}, \mathbf{c}_{\mathrm{cond}}) \odot \mathrm{LN}(\mathbf{h}) + \boldsymbol{\beta}(\mathbf{t}_{\mathrm{emb}}, \mathbf{c}_{\mathrm{cond}})$$

其中 $\mathrm{LN}$ 为无仿射参数的 LayerNorm（均值方差归一化后不乘可学习 $\gamma, \beta$）。自适应仿射通过 FiLM 风格线性层给出：
$$[\boldsymbol{\gamma} \| \boldsymbol{\beta}] = \mathbf{W}_a [\mathbf{t}_{\mathrm{emb}} \| \mathbf{c}_{\mathrm{cond}}] + \mathbf{b}_a$$

- $\mathbf{W}_a \in \mathbb{R}^{2d \times (d_t + d_{\mathrm{cond}})}$ 初始化为 $\mathbf{0}$。
- $\mathbf{b}_a \in \mathbb{R}^{2d}$ 前 $d$ 维初始化为 $1$（$\gamma$），后 $d$ 维初始化为 $0$（$\beta$）。

训练开始时近似恒等 LayerNorm，保证训练稳定性。支持子图级 $\mathbf{t}_{\mathrm{emb}} \in \mathbb{R}^{d_t}$（广播至 $N$）或逐节点 $\mathbf{t}_{\mathrm{emb}} \in \mathbb{R}^{N \times d_t}$（对应节点三分区）。消融开关 `use_ath_norm=false` 时退化为标准 LayerNorm。

#### 5.4.2.2 节点 FFN

**定义 5.13（节点 FFN）** 两层前馈网络，SiLU 激活，扩张因子 $r_{\mathrm{ffn}} = 4$：
$$\mathrm{FFN}_V(\mathbf{h}) = \mathbf{W}_2^V \mathrm{Drop}(\mathrm{SiLU}(\mathbf{W}_1^V \mathbf{h} + \mathbf{b}_1^V)) + \mathbf{b}_2^V$$
其中 $\mathbf{W}_1^V \in \mathbb{R}^{r_{\mathrm{ffn}} d_v \times d_v}$，$\mathbf{W}_2^V \in \mathbb{R}^{d_v \times r_{\mathrm{ffn}} d_v}$。Pre-Norm 残差连接（归一化位于残差前）。

### 5.4.3 子模块 B：节点 FFN 残差

**定义 5.14（节点 FFN 残差）**
$$\mathbf{h}_i^{V,\mathrm{mid}} = \tilde{\mathbf{h}}_i^V + \mathrm{FFN}_V\!\left(\mathrm{ATH\text{-}Norm}(\tilde{\mathbf{h}}_i^V,\, \mathbf{t}_{\mathrm{emb},i},\, [\kappa_h, \kappa_s])\right)$$

### 5.4.4 子模块 C：文本交叉注意力

**定义 5.15（文本交叉注意力）** Pre-Norm 归一化：
$$\hat{\mathbf{h}}_i^V = \mathrm{ATH\text{-}Norm}(\mathbf{h}_i^{V,\mathrm{mid}},\, \mathbf{t}_{\mathrm{emb},i},\, [\kappa_h, \kappa_s])$$

以节点嵌入为 query，投影后节点文本为 key/value：
$$\mathbf{q}_i^{\mathrm{text}} = \mathbf{W}_Q^{\mathrm{text}} \hat{\mathbf{h}}_i^V,\quad \mathbf{k}_j^{\mathrm{text}} = \mathbf{W}_K^{\mathrm{text}} \bar{\mathbf{c}}_j,\quad \mathbf{v}_j^{\mathrm{text}} = \mathbf{W}_V^{\mathrm{text}} \bar{\mathbf{c}}_j$$
其中 $\mathbf{W}_Q^{\mathrm{text}} \in \mathbb{R}^{d_v \times d_v}$，$\mathbf{W}_K^{\mathrm{text}}, \mathbf{W}_V^{\mathrm{text}} \in \mathbb{R}^{d_v \times d_p}$。文本侧使用**全量** $\bar{\mathbf{C}}_\mathcal{V}$（虚节点通过掩码屏蔽）：
$$\mathrm{CrossAttn}_i = \sum_{j=1}^N m_j \cdot \mathrm{softmax}_j\!\left(\frac{\mathbf{q}_i^{\mathrm{text}\top} \mathbf{k}_j^{\mathrm{text}}}{\sqrt{d_v}}\right) \mathbf{v}_j^{\mathrm{text}}$$

交叉注意力残差 + FFN 残差得到第 $l$ 层输出：
$$\mathbf{h}_i^{V,\mathrm{text}} = \mathbf{h}_i^{V,\mathrm{mid}} + \mathrm{Drop}(\mathrm{CrossAttn}_i)$$
$$\mathbf{h}_i^{V,(l)} = \mathbf{h}_i^{V,\mathrm{text}} + \mathrm{FFN}_V\!\left(\mathrm{ATH\text{-}Norm}(\mathbf{h}_i^{V,\mathrm{text}},\, \mathbf{t}_{\mathrm{emb},i},\, [\kappa_h, \kappa_s])\right)$$

---

## 5.5 输出投影层

### 5.5.1 切空间投影

**定义 5.16（切空间投影）** 对任意 $\hat{\mathbf{u}} \in \mathbb{R}^D$（$D = (d_h+1) + (d_s+1) + d_e$），按环境维度拆分为 $\hat{\mathbf{u}}^\mathbb{H} \in \mathbb{R}^{d_h+1}$，$\hat{\mathbf{u}}^\mathbb{S} \in \mathbb{R}^{d_s+1}$，$\hat{\mathbf{u}}^\mathbb{R} \in \mathbb{R}^{d_e}$。投影到 $T_{\mathbf{x}_t}\mathcal{M}$：
$$\hat{\mathbf{v}}^\mathbb{H} = \hat{\mathbf{u}}^\mathbb{H} - \kappa_h \langle \hat{\mathbf{u}}^\mathbb{H}, \mathbf{x}_t^\mathbb{H} \rangle_\mathrm{L} \cdot \mathbf{x}_t^\mathbb{H} \in T_{\mathbf{x}_t^\mathbb{H}}\mathbb{H}$$
$$\hat{\mathbf{v}}^\mathbb{S} = \hat{\mathbf{u}}^\mathbb{S} - \kappa_s (\mathbf{x}_t^{\mathbb{S}\top} \hat{\mathbf{u}}^\mathbb{S}) \cdot \mathbf{x}_t^\mathbb{S} \in T_{\mathbf{x}_t^\mathbb{S}}\mathbb{S}$$
$$\hat{\mathbf{v}}^\mathbb{R} = \hat{\mathbf{u}}^\mathbb{R} \in \mathbb{R}^{d_e}$$

### 5.5.2 向量场输出头

**定义 5.17（向量场预测）** 对节点 $i$：
$$\hat{\mathbf{u}}_i = \mathrm{MLP}_{\mathrm{vec}}(\mathbf{h}_i^{V,(L)}) \in \mathbb{R}^D$$
$\mathrm{MLP}_{\mathrm{vec}}: \mathbb{R}^{d_v} \to \mathbb{R}^D$ 为两层 MLP（Linear → SiLU → Linear）。经定义 5.16 投影得 $\hat{\mathbf{v}}_i = (\hat{\mathbf{v}}_i^\mathbb{H}, \hat{\mathbf{v}}_i^\mathbb{S}, \hat{\mathbf{v}}_i^\mathbb{R}) \in T_{\mathbf{x}_{t,i}}\mathcal{M}$。

### 5.5.3 Node→Edge 对表示

**定义 5.18（几何距离特征）** 对节点对 $(i,j)$：
$$\boldsymbol{\delta}_{ij} = \left[d_\mathbb{H}(\mathbf{x}_{t,i}^\mathbb{H}, \mathbf{x}_{t,j}^\mathbb{H}),\, d_\mathbb{S}(\mathbf{x}_{t,i}^\mathbb{S}, \mathbf{x}_{t,j}^\mathbb{S}),\, d_\mathbb{R}(\mathbf{x}_{t,i}^\mathbb{R}, \mathbf{x}_{t,j}^\mathbb{R}),\, d_\mathcal{M}(\mathbf{x}_{t,i}, \mathbf{x}_{t,j})\right] \in \mathbb{R}^4$$

显式注入几何先验作为 skip-connection：hidden state 经 $L$ 层非线性后可能丢失精确距离信息，$\boldsymbol{\delta}_{ij}$ 直接从当前时刻的流形坐标提供距离信号。

**定义 5.19（Node→Edge 对表示）** 对节点对 $(i,j)$：
$$\mathbf{z}_{ij} = \mathrm{MLP}_{\mathrm{pair}}\!\left(\left[\mathbf{h}_i^{V,(L)} \,\|\, \mathbf{h}_j^{V,(L)} \,\|\, \mathbf{h}_i^{V,(L)} \odot \mathbf{h}_j^{V,(L)} \,\|\, \boldsymbol{\delta}_{ij}\right]\right) \in \mathbb{R}^{d_{\mathrm{pair}}}$$

$\mathrm{MLP}_{\mathrm{pair}}: \mathbb{R}^{3d_v + 4} \to \mathbb{R}^{d_{\mathrm{pair}}}$ 为两层 MLP。

**说明**：$\mathbf{z}_{ij} \neq \mathbf{z}_{ji}$（通过 $\mathbf{h}_i \| \mathbf{h}_j$ 的有序拼接保证有向性），Hadamard 积 $\mathbf{h}_i \odot \mathbf{h}_j$ 对称分量提供稳健的对交互信号，$\boldsymbol{\delta}_{ij}$ 本身对称。有向性完全由拼接项承担。

### 5.5.4 分解式边预测头

#### 5.5.4.1 存在性头

**定义 5.20（边存在性）** 对节点对 $(i,j)$：
$$\hat{s}_{ij} = \sigma\!\left(\mathbf{w}_s^\top \mathbf{z}_{ij} + b_s^{\mathrm{ne}} \cdot \mathbb{1}[i \neq j] + b_s^{\mathrm{self}} \cdot \mathbb{1}[i = j]\right)$$

- $\mathbf{w}_s \in \mathbb{R}^{d_{\mathrm{pair}}}$：可学习线性投影。
- $b_s^{\mathrm{ne}}, b_s^{\mathrm{self}} \in \mathbb{R}$：分离自环与非自环的基线偏置（自环在知识图谱中的先验概率显著不同于普通边）。

#### 5.5.4.2 条件类型头

**定义 5.21（关系原型）** 关系原型向量由关系文本生成：
$$\mathbf{p}_k = \mathrm{MLP}_{\mathrm{proto}}(\bar{\mathbf{c}}_{r_k}) \in \mathbb{R}^{d_{e'}},\quad k \in [K]$$
$\mathrm{MLP}_{\mathrm{proto}}: \mathbb{R}^{d_p} \to \mathbb{R}^{d_{e'}}$ 为两层 MLP。**由文本生成**使模型支持**零样本关系扩展**——新关系只需提供文本描述即可参与推理。记 $\mathbf{P}_{\mathrm{proto}} = (\mathbf{p}_1, \ldots, \mathbf{p}_K)^\top \in \mathbb{R}^{K \times d_{e'}}$。

**定义 5.22（条件类型分布）** 双线性打分：
$$\ell_{ij}^{(k)} = \langle \mathbf{W}_{\mathrm{type}} \mathbf{z}_{ij},\, \mathbf{p}_k \rangle + b_k$$

- $\mathbf{W}_{\mathrm{type}} \in \mathbb{R}^{d_{e'} \times d_{\mathrm{pair}}}$：对表示到关系空间的投影。
- $\mathbf{b} = (b_1, \ldots, b_K)^\top \in \mathbb{R}^K$：可学习逐关系偏置，初始化为零，用于非对称阈值校准。

激活函数根据数据特性选择：
$$\hat{\pi}_{ij}^{(k)} = \begin{cases}
\mathrm{softmax}_k(\ell_{ij}^{(k)}) & \text{单热图（如 wikidata5m）：正对恰有一条边} \\[0.3em]
\sigma(\ell_{ij}^{(k)}) & \text{多热图（如 fb15k-237）：正对可有多条边}
\end{cases}$$

最终边预测（训练损失与推理解码时使用）：
$$\hat{\mathbf{P}}_{ij}^{(k)} = \hat{s}_{ij} \cdot \hat{\pi}_{ij}^{(k)}$$

**复杂度**：$K$ 维 softmax/sigmoid 对 $K$ 线性，复杂度 $O(|\mathcal{S}| K d_{e'})$；关系间的依赖通过 $L$ 层节点注意力的 $\boldsymbol{\phi}_{ij}$ 偏置隐式建模。

#### 5.5.4.3 稀疏评估策略

**训练**：仅对候选对集合 $\mathcal{S} = \mathcal{S}^+ \cup \mathcal{S}^-$ 评估 $\mathbf{z}_{ij}, \hat{s}_{ij}$；仅对正对 $\mathcal{S}^+$ 评估 $\hat{\pi}_{ij}$（负对上 $\hat{\pi}$ 无监督信号，其期望被 $\hat{s}_{ij} \to 0$ 吸收）。$\mathcal{S}^+$ 为至少存在一条边的真实节点对，$\mathcal{S}^-$ 为从无边真实节点对中按比例 $\eta_{\mathrm{neg}} > 0$ 采样的负样本。

**推理（两阶段解码）**：
1. 对所有满足 $m_i = m_j = 1$ 的 $(i,j) \in [N]^2$ 评估 $\hat{s}_{ij}$，选出 top-$B_{\mathrm{cand}}$ 候选对（或 $\hat{s}_{ij} > s_{\mathrm{thresh}}$）构成候选集 $\mathcal{S}_{\mathrm{dec}}$。
2. 仅对 $\mathcal{S}_{\mathrm{dec}}$ 评估 $\hat{\pi}_{ij}^{(k)}$，按激活类型决定：
   - 单热：$k^* = \arg\max_k \hat{\pi}_{ij}^{(k)}$，$\mathbf{E}_{1,ij}^{(k^*)} = 1$，其余 $0$。
   - 多热：$\mathbf{E}_{1,ij}^{(k)} = \mathbb{1}[\hat{\pi}_{ij}^{(k)} > p_{\mathrm{thresh}}]$。
3. $(i,j) \notin \mathcal{S}_{\mathrm{dec}}$ 时 $\mathbf{E}_{1,ij} = \mathbf{0}_K$。

阶段 1 为 $O(N^2 d_{\mathrm{pair}})$ 的 $\hat{s}$ 评估（不涉及 $K$），阶段 2 为 $O(B_{\mathrm{cand}} K d_{e'})$。总代价 $O(N^2 d_{\mathrm{pair}} + B_{\mathrm{cand}} K d_{e'})$，$B_{\mathrm{cand}} \ll N^2$ 时显著低于 $O(N^2 K d_{e'})$ 的密集评估。

---

## 5.6 排列等变性

**命题 5.1（排列等变性）** 对任意排列矩阵 $\boldsymbol{\Pi} \in \{0,1\}^{N \times N}$：
$$f_\theta(\boldsymbol{\Pi}\mathbf{X}_t,\, \boldsymbol{\Pi}\mathbf{E}_t\boldsymbol{\Pi}^\top,\, t,\, \boldsymbol{\Pi}\mathbf{C}_\mathcal{V},\, \mathbf{C}_\mathcal{R},\, \boldsymbol{\Pi}\mathbf{m}) = (\boldsymbol{\Pi}\hat{\mathbf{V}},\, \boldsymbol{\Pi}\hat{\mathbf{s}}\boldsymbol{\Pi}^\top,\, \boldsymbol{\Pi}\hat{\boldsymbol{\pi}}\boldsymbol{\Pi}^\top,\, \boldsymbol{\Pi}\mathbf{H}^{V,(L)})$$

其中 $\mathbf{C}_\mathcal{R}$ 与 $t$ 不受节点排列影响。

**证明要点**：
1. **逐节点算子**：节点初始嵌入、QKV 投影、FFN、文本交叉注意力均为逐节点算子，或对节点维度等变。
2. **RWPE 等变性**：RWPE $\mathbf{p}_i$ 由邻接矩阵的对角元定义。对排列后邻接 $\tilde{\mathbf{A}}' = \boldsymbol{\Pi}\tilde{\mathbf{A}}\boldsymbol{\Pi}^\top$，有 $(\tilde{\mathbf{A}}')^k_{ii} = (\tilde{\mathbf{A}}^k)_{\pi^{-1}(i)\pi^{-1}(i)}$，等变。
3. **稀疏候选集**：$\mathcal{N}_{\mathrm{edge}}(i)$ 由邻接对称关系定义，$\mathcal{N}_{\mathrm{knn}}(i)$ 由测地距离定义，二者均为节点对的等变函数。$\mathcal{N}_{\mathrm{global}}$ 按**度数排序**选取 top-$k_g$，度数为排列不变量，因此 landmark 集合作为整体被 $\boldsymbol{\Pi}$ 映射到对应位置。
4. **边偏置**：$\boldsymbol{\phi}_{ij}$ 由 $\mathbf{E}_{t,ij}$ 与 $\bar{\mathbf{C}}_\mathcal{R}$ 决定，$\mathbf{C}_\mathcal{R}$ 不随节点排列变化，故 $\boldsymbol{\phi}_{\pi(i)\pi(j)} = \boldsymbol{\phi}_{ij}$。
5. **边预测头**：$\mathbf{z}_{ij}$ 依赖 $(\mathbf{h}_i, \mathbf{h}_j, \boldsymbol{\delta}_{ij})$，均为节点对的等变函数；$\hat{s}_{ij}, \hat{\pi}_{ij}$ 由 $\mathbf{z}_{ij}$ 逐对独立计算。$\square$

---

## 5.7 复杂度分析

### 5.7.1 计算复杂度

设 $\bar{d}$ 为子图平均度数，$E = N\bar{d}$ 为总边数（典型 $E = O(N)$），$|\mathcal{S}|$ 为训练负采样集大小，$B_{\mathrm{cand}}$ 为推理时边候选数。

| 阶段 | 复杂度 |
|---|---|
| 节点注意力（单层） | $O(N(\bar{d} + k_{\mathrm{geo}} + k_g) d_v)$ |
| 节点 FFN（单层） | $O(N r_{\mathrm{ffn}} d_v^2)$ |
| 文本交叉注意力（单层） | $O(N^2 d_v + N d_v d_p)$ |
| 几何特征计算（每层需用） | $O(N(\bar{d} + k_{\mathrm{geo}} + k_g))$ |
| 边存在性（训练） | $O(\vert\mathcal{S}\vert d_{\mathrm{pair}})$ |
| 边类型（训练） | $O(\vert\mathcal{S}^+\vert K d_{e'})$ |
| 边存在性（推理） | $O(N^2 d_{\mathrm{pair}})$ |
| 边类型（推理） | $O(B_{\mathrm{cand}} K d_{e'})$ |

**典型配置估算**（$N = 64$，$K = 822$，$d_v = 128$，$d_{e'} = 128$，$d_{\mathrm{pair}} = 256$，$L = 6$，$\bar{d} = 4$，$k_{\mathrm{geo}} = 8$，$k_g = 4$，$B_{\mathrm{cand}} = 32$）：
- 节点注意力候选数 $\approx 17$，相对全连接 $N = 64$ 的缩减比 $\sim 3.8\times$。
- 推理边预测成本 $\approx N^2 d_{\mathrm{pair}} + B_{\mathrm{cand}} K d_{e'} \approx 10^6 + 3.4 \times 10^6 \approx 4.4 \times 10^6$ 次乘加，相对密集评估 $N^2 K d_{e'} \approx 4.3 \times 10^8$ 约 **100× 加速**。

### 5.7.2 显存占用

| 组件 | 显存（单子图） |
|---|---|
| 节点隐藏状态 $\mathbf{H}^{V,(l)}$ | $O(N d_v L)$ |
| 边偏置 $\boldsymbol{\phi}$（稀疏） | $O(E d_p)$ |
| 注意力矩阵（稀疏） | $O(N(\bar{d}+k_{\mathrm{geo}}+k_g) L n_h)$ |
| 文本交叉注意力矩阵 | $O(N^2 L n_h)$ |
| 边预测中间态 $\mathbf{z}_{ij}$（训练稀疏/推理候选） | $O(\vert\mathcal{S}\vert d_{\mathrm{pair}})$ / $O(B_{\mathrm{cand}} d_{\mathrm{pair}})$ |

**主成本** $O(N d_v L)$ 来自节点隐藏状态，与 $N$ 线性、与 $K$ 无关。这使得模型在 $K$ 极大的 KG（如 Wikidata 完整关系集 $K > 1000$）上仍可训练。

### 5.7.3 消融开关

为支持实验验证各设计组件的贡献，RieFormer 提供以下独立消融开关：

- `sparse_attention`：{`full`, `edge_knn_global`, `edge_only`, `knn_only`}——注意力候选集策略。`full` 退化为稠密注意力（$\mathcal{N}(i) = [N]$），可验证稀疏性的必要性。
- `edge_bias_in_attn`：{`true`, `false`}——是否在节点注意力中注入有向边偏置 $\boldsymbol{\phi}_{ij}, \boldsymbol{\phi}_{ji}$。false 时可验证边偏置对拓扑感知的贡献。
- `decompose_edge_head`：{`true`, `false`}——是否使用 $\hat{s} \cdot \hat{\pi}$ 分解。false 时退化为单头 $N^2 K$ 密集 sigmoid，可验证分解策略对稀疏标签的收益。
- `geo_features_in_pair`：{`true`, `false`}——是否在 $\mathbf{z}_{ij}$ 中拼接几何距离 $\boldsymbol{\delta}_{ij}$。false 时验证 hidden state 是否已充分保留距离信号。
- `relation_proto`：{`text`, `learned`}——关系原型来源。`text` 从 $\bar{\mathbf{c}}_{r_k}$ 生成（支持零样本），`learned` 为独立可学习参数（对关系数小、每关系样本丰富的 KG 更合适）。
- `use_ath_norm`：{`true`, `false`}——ATH-Norm vs 标准 LayerNorm，验证时间条件 FiLM 的贡献。
- `time_granularity`：{`subgraph`, `per_node`}——子图级时间 vs 逐节点时间，后者对应节点三分区预训练场景。

每项开关独立可验证，构成清晰的消融路径。



```
================================================================================
                    RieFormer: End-to-End Architecture
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                            INPUT: Subgraph @ time t                          │
│                                                                              │
│  Nodes X_t ∈ M^N          Edges E_t ∈ {0,1}^(N×N×K)      Mask m ∈ {0,1}^N   │
│  ┌──────────────┐         ┌──────────────────────┐       ┌────────────┐     │
│  │ x^H ∈ H^dh   │         │  sparse tensor       │       │ real/virt  │     │
│  │ x^S ∈ S^ds   │         │  >99% zeros          │       │ indicator  │     │
│  │ x^R ∈ R^de   │         │  K relation types    │       └────────────┘     │
│  └──────────────┘         └──────────────────────┘                           │
│                                                                              │
│  Node Text C_V ∈ R^(N×dc)      Relation Text C_R ∈ R^(K×dc)     time t      │
│  ┌───────────────────┐         ┌─────────────────────┐          ┌─────┐     │
│  │ Qwen3-Embedding   │         │ Qwen3-Embedding     │          │[0,1]│     │
│  │ (label+desc)      │         │ shared across sub-  │          └─────┘     │
│  │                   │         │ graphs              │                       │
│  └───────────────────┘         └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
        │                    │                      │                │
        │                    │                      │                │
        ▼                    ▼                      ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INPUT ENCODING (§5.3)                               │
│                                                                              │
│  π(x)        W_c·C_V→C̄_V    W_c·C_R→C̄_R    RWPE p_i    ψ(t)→MLP→t_emb     │
│  ┌──────┐    ┌─────────┐    ┌─────────┐    ┌───────┐    ┌────────────┐      │
│  │ drop │    │ shared  │    │ shared  │    │diag of│    │ sinusoidal │      │
│  │ x_0, │    │ linear  │    │ linear  │    │ Ã, Ã² │    │  +2-layer  │      │
│  │ LN_S,│    │ d_c→d_p │    │ d_c→d_p │    │ …, Ãᵏ │    │    MLP     │      │
│  │ LN_R │    └─────────┘    └─────────┘    └───────┘    └────────────┘      │
│  └──────┘         │              │              │              │            │
│       │           │              │              │              │            │
│       └───────────┴──────────────┴──────────────┘              │            │
│                         │                                       │            │
│                         ▼                                       │            │
│              ┌──────────────────────┐                           │            │
│              │ MLP_node([π‖c̄‖p‖m]) │◄──────────────────────────┘            │
│              └──────────────────────┘       + W_tp · t_emb                   │
│                         │                                                    │
│                         ▼                                                    │
│                    h_i^(0) ∈ R^d_v   (Euclidean latent)                     │
│                                                                              │
│  Edge-side: ON-DEMAND feature φ_ij = mean_k(E_t,ij^(k) · c̄_rk) or e_∅      │
│             NO layer-wise edge hidden state maintained                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 RieFormer BLOCK × L  (§5.4)   [stacked ×L]                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                                                                        │  │
│  │  ┌────────────────────────────────────────────────────────────────┐   │  │
│  │  │  A: SPARSE MANIFOLD ATTENTION                                   │   │  │
│  │  │                                                                  │   │  │
│  │  │   h^(l-1) ─► ATH-Norm(t_emb, [κ_h,κ_s]) ─► h̄                   │   │  │
│  │  │                                                                  │   │  │
│  │  │   Candidate set N(i) = {i} ∪ N_edge(i) ∪ N_knn(i) ∪ N_global   │   │  │
│  │  │              │             │              │              │      │   │  │
│  │  │     1-hop (via E_t)   top-k_geo      top-k_g by          │      │   │  │
│  │  │     known topology    geodesic       degree (landmarks)  │      │   │  │
│  │  │                       on M                                │      │   │  │
│  │  │                                                                  │   │  │
│  │  │   For j ∈ N(i), s ∈ [n_h]:                                       │   │  │
│  │  │                                                                  │   │  │
│  │  │   a_ij = (R(θ_ij)·Wq·h̄_i)ᵀ(Wk·h̄_j)/√d_head      ← M-RoPE     │   │  │
│  │  │        + β·κ(x_i, x_j)                             ← Geo-Kernel  │   │  │
│  │  │        + w_φ·φ_ij + w_φ'·φ_ji                      ← Edge bias   │   │  │
│  │  │        + b_N(i,j)                                  ← Set bias    │   │  │
│  │  │                                                                  │   │  │
│  │  │   α = softmax_N(i)(a);  o = Σ_{j∈N(i)} α_ij · v_j                │   │  │
│  │  │                                                                  │   │  │
│  │  │   h̃ = h^(l-1) + W_O·[o^(1)‖…‖o^(n_h)]       (residual)         │   │  │
│  │  └────────────────────────────────────────────────────────────────┘   │  │
│  │                                  │                                     │  │
│  │                                  ▼                                     │  │
│  │  ┌────────────────────────────────────────────────────────────────┐   │  │
│  │  │  B: NODE FFN RESIDUAL                                           │   │  │
│  │  │   h_mid = h̃ + FFN_V(ATH-Norm(h̃, t_emb, [κ_h,κ_s]))            │   │  │
│  │  │         (Linear → SiLU → Drop → Linear, expansion r=4)          │   │  │
│  │  └────────────────────────────────────────────────────────────────┘   │  │
│  │                                  │                                     │  │
│  │                                  ▼                                     │  │
│  │  ┌────────────────────────────────────────────────────────────────┐   │  │
│  │  │  C: TEXT CROSS-ATTENTION  +  FFN                                │   │  │
│  │  │                                                                  │   │  │
│  │  │   Q ← h_mid         K,V ← C̄_V (all N nodes, virt masked)       │   │  │
│  │  │   CrossAttn = softmax(QKᵀ/√d_v) · V                              │   │  │
│  │  │                                                                  │   │  │
│  │  │   h_text = h_mid + CrossAttn                                    │   │  │
│  │  │   h^(l)  = h_text + FFN_V(ATH-Norm(h_text))                     │   │  │
│  │  └────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                         h^(L) ∈ R^(N×d_v)  backbone hidden                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
┌─────────────────────┐  ┌──────────────────────┐  ┌──────────────────────────┐
│  VECTOR FIELD HEAD  │  │  PAIR REPRESENTATION │  │  BACKBONE HIDDEN         │
│       (§5.5.2)      │  │       (§5.5.3)       │  │  H^(L) ∈ R^(N×d_v)       │
│                     │  │                      │  │                          │
│  û = MLP_vec(h^(L)) │  │  δ_ij = [d_H, d_S,   │  │  → L_align (contrastive) │
│         ∈ R^D       │  │         d_R, d_M]    │  │  → L_mask_c (recognize)  │
│         │           │  │         ∈ R^4        │  │                          │
│         ▼           │  │         (skip-conn)  │  └──────────────────────────┘
│  Tangent projection │  │                      │
│   v̂^H = û^H         │  │  z_ij = MLP_pair(    │
│   - κ_h<û^H,x^H>_L  │  │    [h_i ‖ h_j        │
│     · x^H           │  │     ‖ h_i⊙h_j ‖ δ]) │
│   v̂^S = û^S         │  │     ∈ R^d_pair       │
│   - κ_s·(xᵀû)·x^S   │  │                      │
│   v̂^R = û^R         │  └──────────┬───────────┘
│                     │             │
│  v̂ ∈ T_{x_t}M       │             │
│         │           │   ┌─────────┴──────────┐
│         ▼           │   │                    │
│  L_cont / L_mask_x  │   ▼                    ▼
└─────────────────────┘  ┌──────────┐  ┌──────────────┐
                         │ EXISTENCE│  │ CONDITIONAL  │
                         │   HEAD   │  │  TYPE HEAD   │
                         │  §5.5.4.1│  │  §5.5.4.2    │
                         │          │  │              │
                         │ ŝ_ij =   │  │ p_k=MLP_proto│
                         │  σ(wsᵀz  │  │      (c̄_rk)  │
                         │  +b^ne   │  │              │
                         │  or b^self│ │ ℓ_ij^(k)=    │
                         │ )        │  │  <W_type·z,  │
                         │          │  │   p_k> + b_k │
                         │ ∈ [0,1]  │  │              │
                         └────┬─────┘  │ π̂ = softmax  │
                              │        │ (single-hot) │
                              │        │  or σ        │
                              │        │ (multi-hot)  │
                              │        │              │
                              │        └──────┬───────┘
                              │               │
                              └──────┬────────┘
                                     │
                              P̂_ij^(k) = ŝ_ij · π̂_ij^(k)
                                     │
                                     ▼
                              L_disc = L_exist + λ·L_type

================================================================================
                    TWO-STAGE INFERENCE DECODING (§5.5.4.3)
================================================================================

  Stage 1: compute ŝ_ij ∀(i,j) → select top-B_cand or threshold
             O(N² · d_pair)      ← no K dependency!
                      │
                      ▼
             candidate set S_dec
                      │
  Stage 2: compute π̂_ij^(k) only for (i,j) ∈ S_dec
             O(B_cand · K · d_e')
                      │
                      ▼
  Final E_1,ij:  argmax_k π̂ (single-hot) or threshold (multi-hot)

================================================================================
                          KEY DESIGN INVARIANTS
================================================================================

  • GEOMETRIC STATE {x_t, E_t}    frozen across L layers (read-only bias)
  • LATENT HIDDEN h_i^V            Euclidean, layer-wise updated
  • NO layer-wise edge hidden      φ_ij rebuilt on-demand each layer
  • Manifold constraint            enforced only at final tangent projection
  • Sparse attention O(N(d̄+k))    ≪ dense O(N²)
  • Factored edge head ŝ·π̂         decouples sparsity from type
  • Text-conditioned prototypes    enable zero-shot relations
```