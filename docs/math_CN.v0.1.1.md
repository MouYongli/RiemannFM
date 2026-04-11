# RiemannFM 数学形式化

**RiemannFM: Geometry- and Text-Aware Flow Matching on Product Manifolds with Joint Continuous-Discrete  Dynamics for Knowledge Graph Generation — 完整数学定义**

---

## 目录

- [RiemannFM 数学形式化](#riemannfm-数学形式化)
  - [目录](#目录)
  - [1. 基础集合与索引约定](#1-基础集合与索引约定)
  - [2. 子图的形式化定义](#2-子图的形式化定义)
  - [3. 子图的矩阵表示](#3-子图的矩阵表示)
    - [3.1 嵌入空间：乘积流形](#31-嵌入空间乘积流形)
    - [3.2 节点坐标](#32-节点坐标)
    - [3.3 边类型张量](#33-边类型张量)
    - [3.4 文本条件](#34-文本条件)
    - [3.5 虚节点填充](#35-虚节点填充)
    - [3.6 训练样本与数据集](#36-训练样本与数据集)
  - [4. 流形上的基本运算](#4-流形上的基本运算)
    - [4.1 测地距离](#41-测地距离)
    - [4.2 切空间](#42-切空间)
    - [4.3 对数映射](#43-对数映射)
    - [4.4 指数映射](#44-指数映射)
    - [4.5 切空间黎曼范数](#45-切空间黎曼范数)
  - [5. 模型架构：RieFormer](#5-模型架构rieformer)
    - [5.1 总体定义](#51-总体定义)
    - [5.2 输入编码层](#52-输入编码层)
    - [5.3 RieFormer 块](#53-rieformer-块)
      - [5.3.1 子模块 A：流形感知多头注意力](#531-子模块-a流形感知多头注意力)
      - [5.3.1 子模块 A：流形感知多头注意力](#531-子模块-a流形感知多头注意力-1)
      - [5.3.3 子模块 C：边流自更新](#533-子模块-c边流自更新)
      - [5.3.4 子模块 D：双向交叉交互](#534-子模块-d双向交叉交互)
      - [5.3.5 子模块 E：文本条件注入](#535-子模块-e文本条件注入)
    - [5.4 输出投影层](#54-输出投影层)
      - [5.4.1 向量场输出头](#541-向量场输出头)
      - [5.4.2 边类型输出头](#542-边类型输出头)
    - [5.5 排列等变性](#55-排列等变性)
  - [6. 离散-连续联合流匹配](#6-离散-连续联合流匹配)
    - [6.1 噪声先验分布](#61-噪声先验分布)
    - [6.2 前向插值过程](#62-前向插值过程)
    - [6.3 条件目标](#63-条件目标)
    - [6.4 训练损失函数](#64-训练损失函数)
    - [6.5 训练与推理算法](#65-训练与推理算法)
  - [7. 下游任务微调](#7-下游任务微调)
    - [7.1 统一掩码框架](#71-统一掩码框架)
    - [7.2 任务一：知识图谱补全](#72-任务一知识图谱补全)
    - [7.3 任务二：文本条件子图生成](#73-任务二文本条件子图生成)
    - [7.4 任务三：图异常检测](#74-任务三图异常检测)
  - [符号速查表](#符号速查表)
    - [基础集合与索引](#基础集合与索引)
    - [流形与几何](#流形与几何)
    - [子图表示](#子图表示)
    - [模型架构（RieFormer）](#模型架构rieformer)
    - [流匹配](#流匹配)
    - [损失函数](#损失函数)
    - [下游任务](#下游任务)

---

## 1. 基础集合与索引约定

**定义 1.1（基础集合）。**

- **实体集**：$\mathcal{V} = \{v_1, v_2, \ldots, v_{|\mathcal{V}|}\}$。
- **关系类型集**：$\mathcal{R} = \{r_1, r_2, \ldots, r_K\}$，$K \triangleq |\mathcal{R}|$。
- **知识图谱**：$\mathcal{K} = (\mathcal{V}, \mathcal{E})$，其中事实集 $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{R} \times \mathcal{V}$。一个事实 $(v_i, r_k, v_j) \in \mathcal{E}$ 表示实体 $v_i$ 通过关系 $r_k$ 有向地连接到实体 $v_j$。

**自环**：允许 $i = j$，即 $(v_i, r_k, v_i) \in \mathcal{E}$ 合法。

**多关系边**：同一有序实体对 $(v_i, v_j)$ 可关联多种关系类型，即可同时存在 $(v_i, r_a, v_j) \in \mathcal{E}$ 和 $(v_i, r_b, v_j) \in \mathcal{E}$，$a \neq b$。

**相反关系**：语义相反的关系（如"父亲"与"子女"）作为 $\mathcal{R}$ 中独立的元素存在，不人工添加反向关系。

---

## 2. 子图的形式化定义

**记号**：对任意正整数 $n$，记 $[n] \triangleq \{1, 2, \ldots, n\}$。

**定义 2.1（子图）。** 子图 $\mathcal{G} = (\mathcal{V}_\mathcal{G}, \mathcal{E}_\mathcal{G})$ 是 $\mathcal{K}$ 的诱导子图：
- $\mathcal{V}_\mathcal{G} \subseteq \mathcal{V}$，$|\mathcal{V}_\mathcal{G}| = N$；
- $\mathcal{E}_\mathcal{G} = \{(v_i, r_k, v_j) \in \mathcal{E} \mid v_i, v_j \in \mathcal{V}_\mathcal{G}\}$。

对 $\mathcal{V}_\mathcal{G}$ 中的节点重新编号为 $[N]$。后续 $i, j \in [N]$ 均指子图内的局部编号。

---

## 3. 子图的矩阵表示

子图 $\mathcal{G}$ 编码为五元组 $(\mathbf{X}, \mathbf{E}, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m})$，分别对应节点流形坐标、边类型张量、节点文本条件矩阵、关系文本条件矩阵和节点掩码。各分量在以下小节中依次定义。

### 3.1 嵌入空间：乘积流形

**定义 3.1（Lorentz 内积）。** 对 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^{d_h+1}$：
$$\langle \mathbf{a}, \mathbf{b} \rangle_{\mathrm{L}} = -a_0 b_0 + \sum_{l=1}^{d_h} a_l b_l$$

**定义 3.2（乘积流形）。**
$$\mathcal{M} = \mathbb{H}^{d_h}_{\kappa_h} \times \mathbb{S}^{d_s}_{\kappa_s} \times \mathbb{R}^{d_e}$$

其中：
- $\mathbb{H}^{d_h}_{\kappa_h} = \{\mathbf{z} \in \mathbb{R}^{d_h+1} \mid \langle \mathbf{z}, \mathbf{z} \rangle_{\mathrm{L}} = 1/\kappa_h,\; z_0 > 0\}$，$d_h$ 维双曲空间（Lorentz 模型），曲率 $\kappa_h < 0$；
- $\mathbb{S}^{d_s}_{\kappa_s} = \{\mathbf{z} \in \mathbb{R}^{d_s+1} \mid \|\mathbf{z}\|_2^2 = 1/\kappa_s\}$，$d_s$ 维球面，曲率 $\kappa_s > 0$；
- $\mathbb{R}^{d_e}$ 为 $d_e$ 维欧氏空间。

总环境空间维度 $D \triangleq (d_h + 1) + (d_s + 1) + d_e$。

**分量记号**：对 $\mathbf{x} \in \mathcal{M}$，记 $\mathbf{x}^{\mathbb{H}} \in \mathbb{R}^{d_h+1}$，$\mathbf{x}^{\mathbb{S}} \in \mathbb{R}^{d_s+1}$，$\mathbf{x}^{\mathbb{R}} \in \mathbb{R}^{d_e}$ 为三个子空间分量。

### 3.2 节点坐标

**定义 3.3（节点坐标）。**
$$\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N) \in \mathcal{M}^N$$
其中 $\mathbf{x}_i = (\mathbf{x}_i^{\mathbb{H}},\, \mathbf{x}_i^{\mathbb{S}},\, \mathbf{x}_i^{\mathbb{R}}) \in \mathcal{M}$，$i \in [N]$。

### 3.3 边类型张量

**定义 3.4（边类型张量）。**
$$\mathbf{E} \in \{0,1\}^{N \times N \times K}$$
$$\mathbf{E}_{ij}^{(k)} = \mathbb{1}[(v_i, r_k, v_j) \in \mathcal{E}_\mathcal{G}]$$

固定 $(i, j)$ 时，$\mathbf{E}_{ij} \in \{0,1\}^K$ 为多热向量：$\mathbf{E}_{ij} = \mathbf{0}_K$ 表示 $i$ 到 $j$ 无边，$\|\mathbf{E}_{ij}\|_1 \in \{0, 1, \ldots, K\}$ 表示 $i$ 到 $j$ 的关系类型数。

**性质**：
- **非对称**：一般地 $\mathbf{E}_{ij} \neq \mathbf{E}_{ji}$；
- **多重性**：$\|\mathbf{E}_{ij}\|_1$ 可大于 $1$。
- **稀疏性**：平均出度 $\bar{d} = \frac{1}{N}\sum_{i=1}^{N} |\{j \in [N] : \mathbf{E}_{ij} \neq \mathbf{0}_K\}| \ll N$。

### 3.4 文本条件

**记号**：设 $\Sigma$ 为字符表，$\Sigma^*$ 为 $\Sigma$ 上的有限字符串集，$\circ$ 为字符串拼接运算。

**定义 3.5（文本条件向量）。** 设 $\phi_{\mathrm{text}}: \Sigma^* \to \mathbb{R}^{d_c}$ 为预训练文本编码器。对每个节点 $v_i$：
$$\mathbf{c}_{v_i} = \phi_{\mathrm{text}}(\mathrm{label}_{v_i} \circ \mathrm{desc}_{v_i}) \in \mathbb{R}^{d_c}$$

对每个关系类型 $r_k$：
$$\mathbf{c}_{r_k} = \phi_{\mathrm{text}}(\mathrm{label}_{r_k} \circ \mathrm{desc}_{r_k}) \in \mathbb{R}^{d_c}$$

后续简记 $\mathbf{c}_i \triangleq \mathbf{c}_{v_i}$。

**定义 3.6（文本条件矩阵）。**
- 节点文本矩阵：$\mathbf{C}_\mathcal{V} = (\mathbf{c}_1, \ldots, \mathbf{c}_N) \in \mathbb{R}^{N \times d_c}$（子图级）；
- 关系文本矩阵：$\mathbf{C}_\mathcal{R} = (\mathbf{c}_{r_1}, \ldots, \mathbf{c}_{r_K}) \in \mathbb{R}^{K \times d_c}$（全局，所有子图共享）。

### 3.5 虚节点填充

**定义 3.7（虚节点填充）。** 设 $N_{\max} \in \mathbb{Z}_{>0}$ 为子图节点数上界（预设超参数）。对 $|\mathcal{V}_\mathcal{G}| \leq N_{\max}$ 的子图，填充 $N_{\max} - |\mathcal{V}_\mathcal{G}|$ 个虚节点，使总节点数为 $N_{\max}$：

- 虚节点坐标：$\mathbf{x}_i = \mathbf{x}_\varnothing \in \mathcal{M}$，其中锚点各分量为
$$\mathbf{x}_\varnothing^{\mathbb{H}} = \left(\frac{1}{\sqrt{|\kappa_h|}},\, 0,\, \ldots,\, 0\right) \in \mathbb{R}^{d_h+1}, \quad \mathbf{x}_\varnothing^{\mathbb{S}} = \left(\frac{1}{\sqrt{\kappa_s}},\, 0,\, \ldots,\, 0\right) \in \mathbb{R}^{d_s+1}, \quad \mathbf{x}_\varnothing^{\mathbb{R}} = \mathbf{0}_{d_e}$$
- 虚节点边：$\forall j \in [N_{\max}]$，$\mathbf{E}_{ij} = \mathbf{E}_{ji} = \mathbf{0}_K$；
- 虚节点文本：$\mathbf{c}_i = \mathbf{0}_{d_c}$。

可验证 $\mathbf{x}_\varnothing^{\mathbb{H}}$ 满足 $\langle \mathbf{x}_\varnothing^{\mathbb{H}}, \mathbf{x}_\varnothing^{\mathbb{H}} \rangle_{\mathrm{L}} = -1/|\kappa_h| = 1/\kappa_h$ 且首分量大于零；$\mathbf{x}_\varnothing^{\mathbb{S}}$ 满足 $\|\mathbf{x}_\varnothing^{\mathbb{S}}\|_2^2 = 1/\kappa_s$。

**定义 3.8（节点掩码）。**
$$\mathbf{m} \in \{0,1\}^{N_{\max}}, \quad m_i = \begin{cases} 1 & \text{真实节点} \\ 0 & \text{虚节点} \end{cases}$$

满足 $\sum_{i=1}^{N_{\max}} m_i = |\mathcal{V}_\mathcal{G}|$。

**符号约定**：自此以后，$N \triangleq N_{\max}$。后续所有公式中 $i, j \in [N]$ 包含真实节点与虚节点，虚节点通过 $\mathbf{m}$ 识别和屏蔽。

### 3.6 训练样本与数据集

**定义 3.9（训练样本）。** 一个训练样本即子图的五元组编码：
$$\left(\mathbf{X} \in \mathcal{M}^N,\; \mathbf{E} \in \{0,1\}^{N \times N \times K},\; \mathbf{C}_\mathcal{V} \in \mathbb{R}^{N \times d_c},\; \mathbf{C}_\mathcal{R} \in \mathbb{R}^{K \times d_c},\; \mathbf{m} \in \{0,1\}^N\right)$$

**定义 3.10（训练数据集）。** $\mathcal{D} = \{\mathcal{G}_1, \mathcal{G}_2, \ldots, \mathcal{G}_{|\mathcal{D}|}\}$ 为从知识图谱 $\mathcal{K}$ 中采样的子图集合。

---

## 4. 流形上的基本运算

### 4.1 测地距离

**定义 4.1（分量测地距离）。**

- 双曲距离 $d_{\mathbb{H}}: \mathbb{H}^{d_h}_{\kappa_h} \times \mathbb{H}^{d_h}_{\kappa_h} \to \mathbb{R}_{\geq 0}$：
$$d_{\mathbb{H}}(\mathbf{a}, \mathbf{b}) = \frac{1}{\sqrt{|\kappa_h|}} \operatorname{arccosh}\!\left(\kappa_h \cdot \langle \mathbf{a}, \mathbf{b} \rangle_{\mathrm{L}}\right)$$
其中 $\kappa_h \cdot \langle \mathbf{a}, \mathbf{b} \rangle_{\mathrm{L}} \geq 1$ 对所有 $\mathbf{a}, \mathbf{b} \in \mathbb{H}^{d_h}_{\kappa_h}$ 成立。

- 球面距离 $d_{\mathbb{S}}: \mathbb{S}^{d_s}_{\kappa_s} \times \mathbb{S}^{d_s}_{\kappa_s} \to \mathbb{R}_{\geq 0}$：
$$d_{\mathbb{S}}(\mathbf{a}, \mathbf{b}) = \frac{1}{\sqrt{\kappa_s}} \arccos\!\left(\kappa_s \cdot \mathbf{a}^\top \mathbf{b}\right)$$
其中 $\kappa_s \cdot \mathbf{a}^\top \mathbf{b} \in [-1, 1]$ 由 Cauchy-Schwarz 不等式保证。

- 欧氏距离 $d_{\mathbb{R}}: \mathbb{R}^{d_e} \times \mathbb{R}^{d_e} \to \mathbb{R}_{\geq 0}$：
$$d_{\mathbb{R}}(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2$$

**定义 4.2（乘积流形测地距离）。** $d_\mathcal{M}: \mathcal{M} \times \mathcal{M} \to \mathbb{R}_{\geq 0}$：
$$d_\mathcal{M}(\mathbf{x}, \mathbf{y}) = \sqrt{d_{\mathbb{H}}(\mathbf{x}^{\mathbb{H}}, \mathbf{y}^{\mathbb{H}})^2 + d_{\mathbb{S}}(\mathbf{x}^{\mathbb{S}}, \mathbf{y}^{\mathbb{S}})^2 + d_{\mathbb{R}}(\mathbf{x}^{\mathbb{R}}, \mathbf{y}^{\mathbb{R}})^2}$$

$d_\mathcal{M}$ 为 $\mathcal{M}$ 上的度量（各分量距离的 $\ell_2$ 组合保持度量公理）。

### 4.2 切空间

**定义 4.3（切空间）。** 对 $\mathbf{x} \in \mathcal{M}$，各分量切空间为：
- $T_{\mathbf{x}^{\mathbb{H}}}\mathbb{H} = \{\mathbf{v} \in \mathbb{R}^{d_h+1} \mid \langle \mathbf{v}, \mathbf{x}^{\mathbb{H}} \rangle_{\mathrm{L}} = 0\}$
- $T_{\mathbf{x}^{\mathbb{S}}}\mathbb{S} = \{\mathbf{v} \in \mathbb{R}^{d_s+1} \mid \mathbf{v}^\top \mathbf{x}^{\mathbb{S}} = 0\}$
- $T_{\mathbf{x}^{\mathbb{R}}}\mathbb{R}^{d_e} = \mathbb{R}^{d_e}$

乘积切空间：$T_\mathbf{x}\mathcal{M} = T_{\mathbf{x}^{\mathbb{H}}}\mathbb{H} \times T_{\mathbf{x}^{\mathbb{S}}}\mathbb{S} \times \mathbb{R}^{d_e}$。切向量记为 $\mathbf{v} = (\mathbf{v}^{\mathbb{H}}, \mathbf{v}^{\mathbb{S}}, \mathbf{v}^{\mathbb{R}}) \in T_\mathbf{x}\mathcal{M}$。

### 4.3 对数映射

对数映射 $\log_\mathbf{x}(\mathbf{y})$ 将流形上的点 $\mathbf{y}$ 拉回到 $\mathbf{x}$ 处的切空间。它回答的问题是："从 $\mathbf{x}$ 出发，沿哪个方向走多远能到达 $\mathbf{y}$？"返回的切向量编码了方向和距离。

**定义 4.4（对数映射）。** $\log_\mathbf{x}: \mathcal{M} \to T_\mathbf{x}\mathcal{M}$，各分量：

- 双曲分量：设 $\mathbf{u} = \mathbf{y}^{\mathbb{H}} - \kappa_h \langle \mathbf{x}^{\mathbb{H}}, \mathbf{y}^{\mathbb{H}} \rangle_{\mathrm{L}} \cdot \mathbf{x}^{\mathbb{H}}$，则
$$\log_{\mathbf{x}^{\mathbb{H}}}(\mathbf{y}^{\mathbb{H}}) = \frac{d_{\mathbb{H}}(\mathbf{x}^{\mathbb{H}}, \mathbf{y}^{\mathbb{H}})}{\|\mathbf{u}\|_{\mathrm{L}}} \cdot \mathbf{u}$$
其中 $\|\mathbf{u}\|_{\mathrm{L}} = \sqrt{|\langle \mathbf{u}, \mathbf{u} \rangle_{\mathrm{L}}|}$。可验证 $\langle \mathbf{u}, \mathbf{x}^{\mathbb{H}} \rangle_{\mathrm{L}} = 0$，即 $\mathbf{u} \in T_{\mathbf{x}^{\mathbb{H}}}\mathbb{H}$。

- 球面分量：设 $\mathbf{w} = \mathbf{y}^{\mathbb{S}} - \kappa_s (\mathbf{x}^{\mathbb{S}\top} \mathbf{y}^{\mathbb{S}}) \cdot \mathbf{x}^{\mathbb{S}}$，则
$$\log_{\mathbf{x}^{\mathbb{S}}}(\mathbf{y}^{\mathbb{S}}) = \frac{d_{\mathbb{S}}(\mathbf{x}^{\mathbb{S}}, \mathbf{y}^{\mathbb{S}})}{\|\mathbf{w}\|_2} \cdot \mathbf{w}$$
可验证 $\mathbf{w}^\top \mathbf{x}^{\mathbb{S}} = 0$，即 $\mathbf{w} \in T_{\mathbf{x}^{\mathbb{S}}}\mathbb{S}$。

- 欧氏分量：
$$\log_{\mathbf{x}^{\mathbb{R}}}(\mathbf{y}^{\mathbb{R}}) = \mathbf{y}^{\mathbb{R}} - \mathbf{x}^{\mathbb{R}}$$

乘积：$\log_\mathbf{x}(\mathbf{y}) = (\log_{\mathbf{x}^{\mathbb{H}}}(\mathbf{y}^{\mathbb{H}}),\, \log_{\mathbf{x}^{\mathbb{S}}}(\mathbf{y}^{\mathbb{S}}),\, \log_{\mathbf{x}^{\mathbb{R}}}(\mathbf{y}^{\mathbb{R}}))$。

**约定**：$\mathbf{y} = \mathbf{x}$ 时，$\log_\mathbf{x}(\mathbf{x}) = \mathbf{0} \in T_\mathbf{x}\mathcal{M}$。

### 4.4 指数映射

指数映射 $\exp_\mathbf{x}(\mathbf{v})$ 将切空间中的向量 $\mathbf{v}$ 推回到流形上。它回答的问题是："从 $\mathbf{x}$ 出发，沿切向量 $\mathbf{v}$ 的方向走 $\|\mathbf{v}\|$ 的测地距离，到达流形上的哪个点？"

**定义 4.5（指数映射）。** $\exp_\mathbf{x}: T_\mathbf{x}\mathcal{M} \to \mathcal{M}$，各分量：

- 双曲分量：
$$\exp_{\mathbf{x}^{\mathbb{H}}}(\mathbf{v}^{\mathbb{H}}) = \cosh(\sqrt{|\kappa_h|}\,\|\mathbf{v}^{\mathbb{H}}\|_{\mathrm{L}})\,\mathbf{x}^{\mathbb{H}} + \frac{\sinh(\sqrt{|\kappa_h|}\,\|\mathbf{v}^{\mathbb{H}}\|_{\mathrm{L}})}{\sqrt{|\kappa_h|}\,\|\mathbf{v}^{\mathbb{H}}\|_{\mathrm{L}}}\,\mathbf{v}^{\mathbb{H}}$$

- 球面分量：
$$\exp_{\mathbf{x}^{\mathbb{S}}}(\mathbf{v}^{\mathbb{S}}) = \cos(\sqrt{\kappa_s}\,\|\mathbf{v}^{\mathbb{S}}\|_2)\,\mathbf{x}^{\mathbb{S}} + \frac{\sin(\sqrt{\kappa_s}\,\|\mathbf{v}^{\mathbb{S}}\|_2)}{\sqrt{\kappa_s}\,\|\mathbf{v}^{\mathbb{S}}\|_2}\,\mathbf{v}^{\mathbb{S}}$$

- 欧氏分量：
$$\exp_{\mathbf{x}^{\mathbb{R}}}(\mathbf{v}^{\mathbb{R}}) = \mathbf{x}^{\mathbb{R}} + \mathbf{v}^{\mathbb{R}}$$

乘积：$\exp_\mathbf{x}(\mathbf{v}) = (\exp_{\mathbf{x}^{\mathbb{H}}}(\mathbf{v}^{\mathbb{H}}),\, \exp_{\mathbf{x}^{\mathbb{S}}}(\mathbf{v}^{\mathbb{S}}),\, \exp_{\mathbf{x}^{\mathbb{R}}}(\mathbf{v}^{\mathbb{R}}))$。

**约定**：$\mathbf{v} = \mathbf{0}$ 时，$\exp_\mathbf{x}(\mathbf{0}) = \mathbf{x}$（各分量中 $\sinh(0)/0$ 和 $\sin(0)/0$ 均取极限值 $1$）。

**命题 4.1（互逆性）。** 在 injectivity radius 内（双曲与欧氏：$\infty$；球面：$\pi/\sqrt{\kappa_s}$），$\exp_\mathbf{x} \circ \log_\mathbf{x} = \mathrm{id}_\mathcal{M}$ 且 $\log_\mathbf{x} \circ \exp_\mathbf{x} = \mathrm{id}_{T_\mathbf{x}\mathcal{M}}$。

### 4.5 切空间黎曼范数

**定义 4.6（切空间黎曼范数）。** 对 $\mathbf{v} = (\mathbf{v}^{\mathbb{H}}, \mathbf{v}^{\mathbb{S}}, \mathbf{v}^{\mathbb{R}}) \in T_\mathbf{x}\mathcal{M}$：
$$\|\mathbf{v}\|_{T_\mathbf{x}\mathcal{M}} = \sqrt{\|\mathbf{v}^{\mathbb{H}}\|_{\mathrm{L}}^2 + \|\mathbf{v}^{\mathbb{S}}\|_2^2 + \|\mathbf{v}^{\mathbb{R}}\|_2^2}$$

---

## 5. 模型架构：RieFormer

### 5.1 总体定义

**定义 5.1（RieFormer）。** RieFormer 为参数为 $\theta$ 的映射：
$$f_\theta(\mathbf{X}_t, \mathbf{E}_t, t, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m}) = (\hat{\mathbf{V}}, \hat{\mathbf{P}})$$

输入：$\mathbf{X}_t \in \mathcal{M}^N$，$\mathbf{E}_t \in \{0,1\}^{N \times N \times K}$，$t \in [0,1]$，$\mathbf{C}_\mathcal{V} \in \mathbb{R}^{N \times d_c}$，$\mathbf{C}_\mathcal{R} \in \mathbb{R}^{K \times d_c}$，$\mathbf{m} \in \{0,1\}^N$。

输出：
- $\hat{\mathbf{v}}_i \in T_{\mathbf{x}_{t,i}}\mathcal{M}$，节点 $i$ 的切向量预测，$i \in [N]$；
- $\hat{\mathbf{P}} \in [0,1]^{N \times N \times K}$，$\hat{\mathbf{P}}_{ij}^{(k)}$ 为边类型概率预测。

记 $\mathbf{X}_t = (\mathbf{x}_{t,1}, \ldots, \mathbf{x}_{t,N}) \in \mathcal{M}^N$，其中 $\mathbf{x}_{t,i} \in \mathcal{M}$ 为节点 $i$ 在时间 $t$ 的流形坐标，通过测地线插值从噪声 $\mathbf{x}_{0,i}$​ 到数据 $\mathbf{x}_{1,i}$​ 得到。

$\theta$ 包含所有可学习参数，含曲率 $\kappa_h, \kappa_s$。

### 5.2 输入编码层

**记号**：$[\mathbf{a} \| \mathbf{b}]$ 表示向量拼接（concatenation）。

**定义 5.2（坐标投影）。** $\pi: \mathcal{M} \to \mathbb{R}^D$：
$$\pi(\mathbf{x}) = [\mathbf{x}^{\mathbb{H}} \| \mathbf{x}^{\mathbb{S}} \| \mathbf{x}^{\mathbb{R}}]$$

**定义 5.3（节点初始嵌入）。**
$$\mathbf{h}_i^{V,(0)} = \mathrm{MLP}_{\mathrm{node}}\!\left([\pi(\mathbf{x}_{t,i}) \| \mathbf{c}_i \| m_i]\right) + \mathbf{W}_{\mathrm{tp}}\,\mathbf{t}_{\mathrm{emb}} \in \mathbb{R}^{d_v}$$
其中输入维度为 $D + d_c + 1$，$d_v$ 为节点隐藏维度，$\mathbf{W}_{\mathrm{tp}} \in \mathbb{R}^{d_v \times d_v}$ 为时间投影矩阵，$\mathbf{t}_{\mathrm{emb}}$ 为定义 5.5 的时间嵌入。此时间投影与 ATH-Norm（定义 5.9）的逐层时间条件互补：前者在输入层提供全局时间锚点，后者在每层自适应调节归一化参数。

**定义 5.4（边初始嵌入）。**
$$\mathbf{h}_{ij}^{E,(0)} = \mathrm{MLP}_{\mathrm{edge}}\!\left([\mathbf{E}_{t,ij}\mathbf{W}_{\mathrm{rel}} \| \mathbf{E}_{t,ij}\mathbf{C}_\mathcal{R}]\right) \in \mathbb{R}^{d_{e'}}$$
其中 $\mathbf{W}_{\mathrm{rel}} \in \mathbb{R}^{K \times d_r}$ 为可学习关系嵌入矩阵，$\mathbf{E}_{t,ij}\mathbf{W}_{\mathrm{rel}} \in \mathbb{R}^{d_r}$ 为激活关系类型的嵌入之和，$\mathbf{E}_{t,ij}\mathbf{C}_\mathcal{R} \in \mathbb{R}^{d_c}$ 为激活关系类型的文本嵌入之和。输入维度为 $d_r + d_c$。

**定义 5.5（时间嵌入）。** 设 $d_t \in \mathbb{Z}_{>0}$ 为偶数，正弦位置编码采用块拼接形式：
$$\boldsymbol{\psi}(t) = [\sin(\omega_1 t),\, \ldots,\, \sin(\omega_{d_t/2} t),\, \cos(\omega_1 t),\, \ldots,\, \cos(\omega_{d_t/2} t)] \in \mathbb{R}^{d_t}$$
其中 $\omega_l = 10000^{-2l/d_t}$，$l \in [d_t/2]$。经两层 MLP 投影：
$$\mathbf{t}_{\mathrm{emb}} = \mathbf{W}_2\,\sigma\!\left(\mathbf{W}_1 \boldsymbol{\psi}(t) + \mathbf{b}_1\right) + \mathbf{b}_2 \in \mathbb{R}^{d_t}$$
其中 $\mathbf{W}_1 \in \mathbb{R}^{d_t \times d_t}$，$\mathbf{W}_2 \in \mathbb{R}^{d_t \times d_t}$，$\sigma(\cdot) = \mathrm{SiLU}$。两层 MLP 相比单层线性提供更丰富的时间条件表达能力。

### 5.3 RieFormer 块

RieFormer 由 $L \in \mathbb{Z}_{>0}$ 个相同结构的块堆叠而成。第 $l$ 层（$l \in [L]$）接收节点嵌入 $\mathbf{H}^{V,(l-1)} \in \mathbb{R}^{N \times d_v}$ 和边嵌入 $\mathbf{H}^{E,(l-1)} \in \mathbb{R}^{N \times N \times d_{e'}}$，输出 $\mathbf{H}^{V,(l)} \in \mathbb{R}^{N \times d_v}$ 和 $\mathbf{H}^{E,(l)} \in \mathbb{R}^{N \times N \times d_{e'}}$。

每个块按以下顺序执行五个子模块（采用 Pre-Norm 风格，即归一化在子模块内部、变换之前执行）：

- A. ATH-Norm + 流形感知多头注意力（Manifold RoPE + Geodesic Kernel）→ 更新节点嵌入
- B.（已合并入 A，见下述 Pre-Norm 说明）
- C. 边流自更新 → 更新边嵌入
- D. 双向交叉交互 → 节点与边嵌入互相注入
- E. 文本条件注入 → 通过交叉注意力注入文本语义

#### 5.3.1 子模块 A：流形感知多头注意力

#### 5.3.1 子模块 A：流形感知多头注意力

设 $n_h \in \mathbb{Z}_{>0}$ 为注意力头数，$d_{\mathrm{head}} = d_v / n_h$（要求 $n_h \mid d_v$ 且 $2 \mid d_{\mathrm{head}}$）。

**定义 5.6（Manifold RoPE）。** 对第 $s$ 个注意力头（$s \in [n_h]$），频率 $\omega_l^{(s)} = 10000^{-2l/d_{\mathrm{head}}}$，$l \in [d_{\mathrm{head}}/2]$。对节点对 $(i, j)$，定义角度：
$$\theta_{ij,l}^{(s)} = \omega_l^{(s)} \cdot d_\mathcal{M}(\mathbf{x}_{t,i}, \mathbf{x}_{t,j})$$

旋转矩阵 $\mathbf{R}(\boldsymbol{\theta}_{ij}^{(s)}) \in \mathbb{R}^{d_{\mathrm{head}} \times d_{\mathrm{head}}}$ 为块对角矩阵，由 $d_{\mathrm{head}}/2$ 个 $2 \times 2$ 旋转块组成：
$$\mathbf{R}(\boldsymbol{\theta}_{ij}^{(s)}) = \mathrm{diag}\!\left(\begin{pmatrix} \cos\theta_{ij,1}^{(s)} & -\sin\theta_{ij,1}^{(s)} \\ \sin\theta_{ij,1}^{(s)} & \cos\theta_{ij,1}^{(s)} \end{pmatrix}, \ldots, \begin{pmatrix} \cos\theta_{ij,d_{\mathrm{head}}/2}^{(s)} & -\sin\theta_{ij,d_{\mathrm{head}}/2}^{(s)} \\ \sin\theta_{ij,d_{\mathrm{head}}/2}^{(s)} & \cos\theta_{ij,d_{\mathrm{head}}/2}^{(s)} \end{pmatrix}\right)$$

**定义 5.7（Geodesic Kernel）。** 对第 $s$ 个注意力头：
$$\kappa^{(s)}(\mathbf{x}_{t,i}, \mathbf{x}_{t,j}) = w_{\mathbb{H}}^{(s)} \kappa_{\mathbb{H}}(\mathbf{x}_{t,i}^{\mathbb{H}}, \mathbf{x}_{t,j}^{\mathbb{H}}) + w_{\mathbb{S}}^{(s)} \kappa_{\mathbb{S}}(\mathbf{x}_{t,i}^{\mathbb{S}}, \mathbf{x}_{t,j}^{\mathbb{S}}) + w_{\mathbb{R}}^{(s)} \kappa_{\mathbb{R}}(\mathbf{x}_{t,i}^{\mathbb{R}}, \mathbf{x}_{t,j}^{\mathbb{R}})$$
其中：
- $\kappa_{\mathbb{H}}(\mathbf{a}, \mathbf{b}) = -d_{\mathbb{H}}(\mathbf{a}, \mathbf{b})$
- $\kappa_{\mathbb{S}}(\mathbf{a}, \mathbf{b}) = \kappa_s \cdot \mathbf{a}^\top \mathbf{b}$
- $\kappa_{\mathbb{R}}(\mathbf{a}, \mathbf{b}) = -\|\mathbf{a} - \mathbf{b}\|_2^2$

$w_{\mathbb{H}}^{(s)}, w_{\mathbb{S}}^{(s)}, w_{\mathbb{R}}^{(s)} \in \mathbb{R}$ 为每头独立的可学习权重。

**定义 5.8（流形感知注意力，Pre-Norm 风格）。** 先对节点嵌入做 ATH-Norm（定义 5.9），再计算注意力：

$$\bar{\mathbf{h}}_i^{V} = \mathrm{ATH\text{-}Norm}(\mathbf{h}_i^{V,(l-1)},\, \mathbf{t}_{\mathrm{emb}})$$

查询/键/值投影（$s \in [n_h]$）作用于归一化后的嵌入：
$$\mathbf{q}_i^{(s)} = \mathbf{W}_Q^{(s)} \bar{\mathbf{h}}_i^{V},\quad \mathbf{k}_j^{(s)} = \mathbf{W}_K^{(s)} \bar{\mathbf{h}}_j^{V},\quad \mathbf{v}_j^{(s)} = \mathbf{W}_V^{(s)} \bar{\mathbf{h}}_j^{V} \in \mathbb{R}^{d_{\mathrm{head}}}$$
其中 $\mathbf{W}_Q^{(s)}, \mathbf{W}_K^{(s)}, \mathbf{W}_V^{(s)} \in \mathbb{R}^{d_{\mathrm{head}} \times d_v}$。

注意力分数：
$$a_{ij}^{(s)} = \frac{(\mathbf{R}(\boldsymbol{\theta}_{ij}^{(s)})\mathbf{q}_i^{(s)})^\top \mathbf{k}_j^{(s)}}{\sqrt{d_{\mathrm{head}}}} + \beta^{(s)} \kappa^{(s)}(\mathbf{x}_{t,i}, \mathbf{x}_{t,j}) + \mathbf{w}_b^{(s)\top} \mathbf{h}_{ij}^{E,(l-1)}$$
其中 $\beta^{(s)} \in \mathbb{R}$ 为可学习缩放系数，$\mathbf{w}_b^{(s)} \in \mathbb{R}^{d_{e'}}$ 为边偏置权重。

注意力权重：$\alpha_{ij}^{(s)} = \mathrm{softmax}_j(a_{ij}^{(s)})$。

欧氏聚合：
$$\mathbf{o}_i^{(s)} = \sum_{j=1}^N \alpha_{ij}^{(s)} \cdot \mathbf{v}_j^{(s)} \in \mathbb{R}^{d_{\mathrm{head}}}$$

多头拼接与投影：
$$\mathrm{MHA}_i = \mathbf{W}_O [\mathbf{o}_i^{(1)} \| \cdots \| \mathbf{o}_i^{(n_h)}] \in \mathbb{R}^{d_v}$$
其中 $\mathbf{W}_O \in \mathbb{R}^{d_v \times d_v}$。

残差连接（跳过归一化前的原始嵌入）：$\tilde{\mathbf{h}}_i^V = \mathbf{h}_i^{V,(l-1)} + \mathrm{MHA}_i$。

#### 5.3.3 子模块 C：边流自更新

边嵌入通过分解注意力独立更新：对边 $(i, j)$，分别聚合头节点 $i$ 的其他出边和尾节点 $j$ 的其他入边的信息。

**定义 5.11（分解边注意力）。** 对边 $(i, j)$：

头侧聚合（$i$ 的其他出边）：
$$\mathbf{g}_{ij}^{\mathrm{head}} = \sum_{p \in [N] \setminus \{j\}} \gamma_{ip \to ij}^{\mathrm{head}} \cdot \mathbf{W}_{\mathrm{Ev}}^{\mathrm{head}} \mathbf{h}_{ip}^{E,(l-1)}$$

尾侧聚合（$j$ 的其他入边）：
$$\mathbf{g}_{ij}^{\mathrm{tail}} = \sum_{p \in [N] \setminus \{i\}} \gamma_{pj \to ij}^{\mathrm{tail}} \cdot \mathbf{W}_{\mathrm{Ev}}^{\mathrm{tail}} \mathbf{h}_{pj}^{E,(l-1)}$$

注意力权重：
$$\gamma_{ip \to ij}^{\mathrm{head}} = \mathrm{softmax}_{p}\!\left(\frac{(\mathbf{W}_{\mathrm{Eq}}^{\mathrm{head}} \mathbf{h}_{ij}^{E,(l-1)})^\top (\mathbf{W}_{\mathrm{Ek}}^{\mathrm{head}} \mathbf{h}_{ip}^{E,(l-1)})}{\sqrt{d_{e'}}}\right)$$
尾侧 $\gamma_{pj \to ij}^{\mathrm{tail}}$ 类似，使用独立的权重矩阵 $\mathbf{W}_{\mathrm{Eq}}^{\mathrm{tail}}, \mathbf{W}_{\mathrm{Ek}}^{\mathrm{tail}}$。

所有权重矩阵 $\mathbf{W}_{\mathrm{Eq}}^{\mathrm{head}}, \mathbf{W}_{\mathrm{Ek}}^{\mathrm{head}}, \mathbf{W}_{\mathrm{Ev}}^{\mathrm{head}}, \mathbf{W}_{\mathrm{Eq}}^{\mathrm{tail}}, \mathbf{W}_{\mathrm{Ek}}^{\mathrm{tail}}, \mathbf{W}_{\mathrm{Ev}}^{\mathrm{tail}} \in \mathbb{R}^{d_{e'} \times d_{e'}}$。

残差更新：
$$\tilde{\mathbf{h}}_{ij}^E = \mathbf{h}_{ij}^{E,(l-1)} + \mathrm{MLP}_{\mathrm{E\text{-}up}}\!\left([\mathbf{h}_{ij}^{E,(l-1)} \| \mathbf{g}_{ij}^{\mathrm{head}} \| \mathbf{g}_{ij}^{\mathrm{tail}}]\right)$$
其中 $\mathrm{MLP}_{\mathrm{E\text{-}up}}: \mathbb{R}^{3d_{e'}} \to \mathbb{R}^{d_{e'}}$。

经 LayerNorm：$\bar{\mathbf{h}}_{ij}^E = \mathrm{LN}(\tilde{\mathbf{h}}_{ij}^E)$。

#### 5.3.4 子模块 D：双向交叉交互

**定义 5.12（边→节点聚合）。** 对节点 $i$，聚合其相关边嵌入：
$$\hat{\mathbf{h}}_i^V = \bar{\mathbf{h}}_i^V + \mathrm{MLP}_{E \to V}\!\left(\sum_{j=1}^N \alpha_{ij}^{E \to V} \cdot \mathbf{W}_{\mathrm{Ev2n}} \bar{\mathbf{h}}_{ij}^E\right)$$
其中 $\mathbf{W}_{\mathrm{Ev2n}} \in \mathbb{R}^{d_v \times d_{e'}}$，$\mathrm{MLP}_{E \to V}: \mathbb{R}^{d_v} \to \mathbb{R}^{d_v}$。注意力权重：
$$\alpha_{ij}^{E \to V} = \mathrm{softmax}_j\!\left(\frac{(\mathbf{W}_Q^{E \to V} \bar{\mathbf{h}}_i^V)^\top (\mathbf{W}_K^{E \to V} \bar{\mathbf{h}}_{ij}^E)}{\sqrt{d_v}}\right)$$
其中 $\mathbf{W}_Q^{E \to V} \in \mathbb{R}^{d_v \times d_v}$，$\mathbf{W}_K^{E \to V} \in \mathbb{R}^{d_v \times d_{e'}}$。

**定义 5.13（节点→边注入）。** 对边 $(i, j)$：
$$\hat{\mathbf{h}}_{ij}^E = \bar{\mathbf{h}}_{ij}^E + \mathrm{MLP}_{V \to E}\!\left([\hat{\mathbf{h}}_i^V \| \hat{\mathbf{h}}_j^V \| \hat{\mathbf{h}}_i^V \odot \hat{\mathbf{h}}_j^V]\right)$$
其中 $\mathrm{MLP}_{V \to E}: \mathbb{R}^{3d_v} \to \mathbb{R}^{d_{e'}}$。

#### 5.3.5 子模块 E：文本条件注入

**定义 5.14（文本交叉注意力）。** 对节点 $i$，以节点嵌入为 query、文本条件为 key/value：
$$\mathbf{q}_i^{\mathrm{text}} = \mathbf{W}_Q^{\mathrm{text}} \hat{\mathbf{h}}_i^V, \quad \mathbf{k}_j^{\mathrm{text}} = \mathbf{W}_K^{\mathrm{text}} \mathbf{c}_j, \quad \mathbf{v}_j^{\mathrm{text}} = \mathbf{W}_V^{\mathrm{text}} \mathbf{c}_j$$
其中 $\mathbf{W}_Q^{\mathrm{text}} \in \mathbb{R}^{d_v \times d_v}$，$\mathbf{W}_K^{\mathrm{text}}, \mathbf{W}_V^{\mathrm{text}} \in \mathbb{R}^{d_v \times d_c}$。

交叉注意力：
$$\mathrm{CrossAttn}_i = \sum_{j=1}^N \mathrm{softmax}_j\!\left(\frac{\mathbf{q}_i^{\mathrm{text}\top} \mathbf{k}_j^{\mathrm{text}}}{\sqrt{d_v}}\right) \cdot \mathbf{v}_j^{\mathrm{text}}$$

**定义 5.15（第 $l$ 层输出）。** 文本交叉注意力残差后，分别经 ATH-Norm + FFN 残差得到最终输出：

- 节点文本残差：$\mathbf{h}_i^{V,\mathrm{text}} = \hat{\mathbf{h}}_i^V + \mathrm{CrossAttn}_i$
- 节点 FFN：$\mathbf{h}_i^{V,(l)} = \mathbf{h}_i^{V,\mathrm{text}} + \mathrm{FFN}_V\!\left(\mathrm{ATH\text{-}Norm}(\mathbf{h}_i^{V,\mathrm{text}},\, \mathbf{t}_{\mathrm{emb}})\right)$
- 边 FFN：$\mathbf{h}_{ij}^{E,(l)} = \hat{\mathbf{h}}_{ij}^E + \mathrm{FFN}_E\!\left(\mathrm{LN}(\hat{\mathbf{h}}_{ij}^E)\right)$

其中 $\mathrm{FFN}_V: \mathbb{R}^{d_v} \to \mathbb{R}^{d_v}$ 和 $\mathrm{FFN}_E: \mathbb{R}^{d_{e'}} \to \mathbb{R}^{d_{e'}}$ 均为两层 MLP（Linear → SiLU → Linear），采用 Pre-Norm 残差连接。

### 5.4 输出投影层

#### 5.4.1 向量场输出头

**定义 5.16（切空间投影）。** 对任意 $\hat{\mathbf{u}} \in \mathbb{R}^D$，按坐标投影 $\pi$ 的逆序拆分为 $\hat{\mathbf{u}}^{\mathbb{H}} \in \mathbb{R}^{d_h+1}$，$\hat{\mathbf{u}}^{\mathbb{S}} \in \mathbb{R}^{d_s+1}$，$\hat{\mathbf{u}}^{\mathbb{R}} \in \mathbb{R}^{d_e}$。投影到 $T_{\mathbf{x}_t}\mathcal{M}$：
$$\hat{\mathbf{v}}^{\mathbb{H}} = \hat{\mathbf{u}}^{\mathbb{H}} - \kappa_h \langle \hat{\mathbf{u}}^{\mathbb{H}}, \mathbf{x}_t^{\mathbb{H}} \rangle_{\mathrm{L}} \cdot \mathbf{x}_t^{\mathbb{H}} \in T_{\mathbf{x}_t^{\mathbb{H}}}\mathbb{H}$$
$$\hat{\mathbf{v}}^{\mathbb{S}} = \hat{\mathbf{u}}^{\mathbb{S}} - \kappa_s (\mathbf{x}_t^{\mathbb{S}\top} \hat{\mathbf{u}}^{\mathbb{S}}) \cdot \mathbf{x}_t^{\mathbb{S}} \in T_{\mathbf{x}_t^{\mathbb{S}}}\mathbb{S}$$
$$\hat{\mathbf{v}}^{\mathbb{R}} = \hat{\mathbf{u}}^{\mathbb{R}} \in \mathbb{R}^{d_e}$$

**定义 5.17（向量场预测）。** 对节点 $i$：
$$\hat{\mathbf{u}}_i = \mathrm{MLP}_{\mathrm{vec}}(\mathbf{h}_i^{V,(L)}) \in \mathbb{R}^D$$
其中 $\mathrm{MLP}_{\mathrm{vec}}: \mathbb{R}^{d_v} \to \mathbb{R}^D$。经定义 5.16 投影后得 $\hat{\mathbf{v}}_i = (\hat{\mathbf{v}}_i^{\mathbb{H}}, \hat{\mathbf{v}}_i^{\mathbb{S}}, \hat{\mathbf{v}}_i^{\mathbb{R}}) \in T_{\mathbf{x}_{t,i}}\mathcal{M}$。

#### 5.4.2 边类型输出头

**定义 5.18（关系交互层）。** 对边 $(i, j)$ 和每个关系类型 $k \in [K]$，构造候选关系特征：
$$\mathbf{r}_{ij}^{(k)} = \mathrm{MLP}_{\mathrm{rel\text{-}proj}}\!\left([\mathbf{h}_{ij}^{E,(L)} \| \mathbf{c}_{r_k}]\right) \in \mathbb{R}^{d_{e'}}$$
其中 $\mathrm{MLP}_{\mathrm{rel\text{-}proj}}: \mathbb{R}^{d_{e'} + d_c} \to \mathbb{R}^{d_{e'}}$。

$K$ 个候选特征 $\{\mathbf{r}_{ij}^{(1)}, \ldots, \mathbf{r}_{ij}^{(K)}\}$ 经关系 Transformer（在关系维度上的自注意力）交互后得 $\tilde{\mathbf{r}}_{ij}^{(k)} \in \mathbb{R}^{d_{e'}}$，$k \in [K]$。

**定义 5.19（边类型概率）。**
$$\hat{\mathbf{P}}_{ij}^{(k)} = \sigma(\mathbf{w}_{\mathrm{cls}}^\top \tilde{\mathbf{r}}_{ij}^{(k)} + b_{\mathrm{cls}}) \in [0,1]$$
其中 $\sigma(\cdot)$ 为 sigmoid 函数，$\mathbf{w}_{\mathrm{cls}} \in \mathbb{R}^{d_{e'}}$，$b_{\mathrm{cls}} \in \mathbb{R}$。各关系类型独立预测（对应多热标签）。

### 5.5 排列等变性

**命题 5.1（排列等变性）。** 对任意排列矩阵 $\boldsymbol{\Pi} \in \{0,1\}^{N \times N}$：
$$f_\theta(\boldsymbol{\Pi}\mathbf{X}_t, \boldsymbol{\Pi}\mathbf{E}_t\boldsymbol{\Pi}^\top, t, \boldsymbol{\Pi}\mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \boldsymbol{\Pi}\mathbf{m}) = (\boldsymbol{\Pi}\hat{\mathbf{V}}, \boldsymbol{\Pi}\hat{\mathbf{P}}\boldsymbol{\Pi}^\top)$$

其中 $\mathbf{C}_\mathcal{R}$ 和 $t$ 不受排列影响。

---

## 6. 离散-连续联合流匹配

**约定**：$t = 0$ 对应噪声端，$t = 1$ 对应数据端。$\mathbf{X}_1, \mathbf{E}_1$ 为训练数据（来自子图），$\mathbf{X}_0, \mathbf{E}_0$ 为噪声采样。

### 6.1 噪声先验分布

**定义 6.1（连续噪声先验）。** 对每个节点 $i \in [N]$，各分量独立采样：
- 双曲分量：$\mathbf{x}_{0,i}^{\mathbb{H}} \sim \mathrm{Uniform}(B_{\mathbb{H}}(\mathbf{x}_\varnothing^{\mathbb{H}}, R_{\mathbb{H}}))$，即在以锚点 $\mathbf{x}_\varnothing^{\mathbb{H}}$ 为中心、测地半径 $R_{\mathbb{H}} > 0$ 的测地球内关于黎曼体积测度均匀采样；
- 球面分量：$\mathbf{x}_{0,i}^{\mathbb{S}} \sim \mathrm{Uniform}(\mathbb{S}^{d_s}_{\kappa_s})$，即关于球面体积测度的均匀分布；
- 欧氏分量：$\mathbf{x}_{0,i}^{\mathbb{R}} \sim \mathcal{N}(\mathbf{0}, \sigma_0^2 \mathbf{I}_{d_e})$。

其中 $R_{\mathbb{H}} > 0$ 和 $\sigma_0 > 0$ 为超参数。记连续噪声先验为 $p_0^{\mathcal{M}}$。

**定义 6.2（离散噪声先验）。** 对每对 $(i, j) \in [N]^2$ 和每个关系类型 $k \in [K]$，独立采样：
$$\mathbf{E}_{0,ij}^{(k)} \sim \mathrm{Bernoulli}(\rho_k)$$

其中边际频率：
$$\rho_k = \frac{\sum_{\mathcal{G} \in \mathcal{D}} \sum_{i=1}^N \sum_{j=1}^N \mathbf{E}_{1,ij}^{(k)}}{|\mathcal{D}| \cdot N^2}$$

即关系类型 $r_k$ 在训练集中的平均出现频率（虚节点边为零，不影响分子）。

### 6.2 前向插值过程

**定义 6.3（连续插值）。** 对每个节点 $i \in [N]$，在 $\mathbf{x}_{0,i}$（噪声）与 $\mathbf{x}_{1,i}$（数据）之间沿测地线插值：
$$\mathbf{x}_{t,i} = \exp_{\mathbf{x}_{0,i}}\!\left(t \cdot \log_{\mathbf{x}_{0,i}}(\mathbf{x}_{1,i})\right) \in \mathcal{M}, \quad t \in [0, 1]$$

边界条件：$\mathbf{x}_{0,i} = \mathbf{x}_{0,i}$（噪声），$\mathbf{x}_{1,i} = \mathbf{x}_{1,i}$（数据）。

**定义 6.4（离散插值）。** 对每对 $(i, j) \in [N]^2$，独立采样掩码：
$$z_{ij} \sim \mathrm{Bernoulli}(t)$$
$$\mathbf{E}_{t,ij} = z_{ij} \cdot \mathbf{E}_{1,ij} + (1 - z_{ij}) \cdot \mathbf{E}_{0,ij}$$

同一对 $(i, j)$ 的所有 $K$ 个关系类型共享同一个 $z_{ij}$，即整体取数据端或噪声端。不同方向 $z_{ij}$ 与 $z_{ji}$ 独立采样。

### 6.3 条件目标

**定义 6.5（条件向量场目标）。** 对每个节点 $i \in [N]$：
$$\mathbf{u}_{t,i} = \frac{1}{1-t}\log_{\mathbf{x}_{t,i}}(\mathbf{x}_{1,i}) \in T_{\mathbf{x}_{t,i}}\mathcal{M}$$

即从当前位置 $\mathbf{x}_{t,i}$ 指向数据点 $\mathbf{x}_{1,i}$ 的切向量，按剩余时间 $1 - t$ 缩放为速度：沿此速度走 $1 - t$ 时间恰好到达 $\mathbf{x}_{1,i}$。训练时 $t$ 的采样分布可选：

- **均匀分布**（默认）：$t \sim \mathrm{Uniform}(0, 1 - \epsilon_t)$，$\epsilon_t > 0$ 为小常数，避免 $t \to 1$ 时的奇异性；
- **Logit-Normal 分布**：$t = \sigma(\mu_{\mathrm{LN}} + \sigma_{\mathrm{LN}} \cdot z)$，$z \sim \mathcal{N}(0, 1)$，其中 $\sigma(\cdot)$ 为 sigmoid 函数。该分布将采样密度集中于 $t \approx 0.5$，指数级降低 $t \to 0$ 和 $t \to 1$ 极端区域的采样概率，从而抑制 $1/(1-t)$ 缩放因子引起的梯度方差。

**定义 6.6（边类型目标）。** 对每对 $(i, j) \in [N]^2$ 和每个关系类型 $k \in [K]$：
$$\mathbf{E}_{1,ij}^{(k)} \in \{0,1\}$$
即数据端的真实边类型标签。

### 6.4 训练损失函数

**定义 6.7（流形向量场损失）。**
$$\mathcal{L}_{\mathrm{cont}} = \frac{1}{N}\sum_{i=1}^N m_i \cdot \|\hat{\mathbf{v}}_i - \mathbf{u}_{t,i}\|_{T_{\mathbf{x}_{t,i}}\mathcal{M}}^2$$

其中 $\|\cdot\|_{T_{\mathbf{x}_{t,i}}\mathcal{M}}$ 为定义 4.6 的切空间黎曼范数。虚节点（$m_i = 0$）不贡献损失。

**定义 6.8（边类型损失）。** 设 $\mathcal{S} \subseteq [N]^2$ 为采样的边对集合。
$$\mathcal{L}_{\mathrm{disc}} = \frac{1}{|\mathcal{S}|}\sum_{(i,j) \in \mathcal{S}}\sum_{k=1}^K \mathrm{BCE}_{\mathrm{w}}(\hat{\mathbf{P}}_{ij}^{(k)}, \mathbf{E}_{1,ij}^{(k)})$$

其中加权二元交叉熵：
$$\mathrm{BCE}_{\mathrm{w}}(\hat{p}, y) = -\left[w_k^+ \cdot y \cdot \log(\hat{p} + \epsilon_p) + (1 - y) \cdot \log(1 - \hat{p} + \epsilon_p)\right]$$

正样本权重 $w_k^+ = \min\!\left(\frac{1 - \rho_k}{\rho_k},\, w_{\max}\right)$ 校正类别不平衡，$w_{\max} > 0$ 为截断上限，$\epsilon_p > 0$ 为数值稳定常数。

**定义 6.9（图-文对比损失）。** 将数据端坐标投影到对齐空间。由于 Lorentz 模型中时间坐标 $x_0 = \sqrt{\|\mathbf{x}_{\mathrm{sp}}\|^2 + 1/|\kappa_h|}$ 是空间坐标的确定性函数（冗余维度），且在训练初期对所有实体近似为常数，会淹没实体间的区分信号，因此在投影前将其剔除：
$$\mathbf{g}_i = \mathrm{MLP}_{\mathrm{proj}}\!\left(\pi_{\setminus x_0}(\mathbf{x}_{1,i})\right) \in \mathbb{R}^{d_\mathrm{a}}$$
其中 $\pi_{\setminus x_0}$ 表示去除 Lorentz 时间坐标 $x_0$ 后的环境坐标切片（维度 $D - 1$），$\mathrm{MLP}_{\mathrm{proj}}: \mathbb{R}^{D-1} \to \mathbb{R}^{d_\mathrm{a}}$，$d_\mathrm{a}$ 为对齐空间维度。

余弦相似度：$\mathrm{sim}(\mathbf{g}_i, \mathbf{c}_j) = \frac{\mathbf{g}_i^\top \mathbf{c}_j}{\|\mathbf{g}_i\|_2 \|\mathbf{c}_j\|_2}$

设 $\mathcal{B} \subseteq [N]$ 为 mini-batch 中的节点索引集。当 $|\mathcal{B}|$ 较大时（如 $BN_{\max}$ 量级），从中均匀随机子采样至多 $M_{\mathrm{align}}$ 个节点以控制 $|\mathcal{B}| \times |\mathcal{B}|$ 相似度矩阵的规模并降低 InfoNCE 任务难度。对称对比损失：
$$\mathcal{L}_{\mathrm{align}} = \frac{1}{2}\left(\mathcal{L}_{\mathrm{align}}^{g \to c} + \mathcal{L}_{\mathrm{align}}^{c \to g}\right)$$
其中：
$$\mathcal{L}_{\mathrm{align}}^{g \to c} = -\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \log \frac{\exp(\mathrm{sim}(\mathbf{g}_i, \mathbf{c}_i) / \tau)}{\sum_{j \in \mathcal{B}} \exp(\mathrm{sim}(\mathbf{g}_i, \mathbf{c}_j) / \tau)}$$
$\mathcal{L}_{\mathrm{align}}^{c \to g}$ 对称定义。$\tau > 0$ 为温度超参数。

**定义 6.10（总训练损失）。**
$$\mathcal{L} = \mathcal{L}_{\mathrm{cont}} + \lambda\,\mathcal{L}_{\mathrm{disc}} + \mu\,\mathcal{L}_{\mathrm{align}}$$
其中 $\lambda, \mu > 0$ 为损失权重超参数。

### 6.5 训练与推理算法

**算法 1：RiemannFM 单步训练**

**输入**：训练子图 $(\mathbf{X}_1, \mathbf{E}_1, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m})$，参数 $\theta$
**输出**：损失 $\mathcal{L}$

1. 采样 $t \sim p_t$（$p_t$ 为 Uniform 或 Logit-Normal，见定义 6.5 注释）
2. 对 $i \in [N]$：采样 $\mathbf{x}_{0,i} \sim p_0^{\mathcal{M}}$（定义 6.1）
3. 对 $(i, j) \in [N]^2$，$k \in [K]$：采样 $\mathbf{E}_{0,ij}^{(k)} \sim \mathrm{Bernoulli}(\rho_k)$（定义 6.2）
4. 对 $i \in [N]$：$\mathbf{x}_{t,i} = \exp_{\mathbf{x}_{0,i}}(t \cdot \log_{\mathbf{x}_{0,i}}(\mathbf{x}_{1,i}))$（定义 6.3）
5. 对 $(i, j) \in [N]^2$：采样 $z_{ij} \sim \mathrm{Bernoulli}(t)$，$\mathbf{E}_{t,ij} = z_{ij} \mathbf{E}_{1,ij} + (1 - z_{ij}) \mathbf{E}_{0,ij}$（定义 6.4）
6. $(\hat{\mathbf{V}}, \hat{\mathbf{P}}) = f_\theta(\mathbf{X}_t, \mathbf{E}_t, t, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m})$（定义 5.1）
7. 对 $i \in [N]$：$\mathbf{u}_{t,i} = \frac{1}{1-t}\log_{\mathbf{x}_{t,i}}(\mathbf{x}_{1,i})$（定义 6.5）
8. 计算 $\mathcal{L} = \mathcal{L}_{\mathrm{cont}} + \lambda\,\mathcal{L}_{\mathrm{disc}} + \mu\,\mathcal{L}_{\mathrm{align}}$（定义 6.10）
9. Riemannian Adam 更新 $\theta$：对欧氏参数用标准 Adam，对曲率参数用黎曼梯度
10. 曲率投影：$\kappa_h \leftarrow \min(\kappa_h, -\epsilon_\kappa)$，$\kappa_s \leftarrow \max(\kappa_s, \epsilon_\kappa)$

其中 $\epsilon_t > 0$，$\epsilon_\kappa > 0$ 为小常数。

**算法 2：RiemannFM 推理**

**输入**：文本条件 $\mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}$，步数 $T \in \mathbb{Z}_{>0}$，步长 $\Delta t = 1/T$
**输出**：$(\mathbf{X}_1, \mathbf{E}_1, \mathbf{m}_{\mathrm{pred}})$

1. 采样 $\mathbf{X}_0, \mathbf{E}_0$（定义 6.1, 6.2）
2. **for** $s = 0, 1, \ldots, T-1$ **do**：
   - $t = s \cdot \Delta t$
   - $(\hat{\mathbf{V}}, \hat{\mathbf{P}}) = f_\theta(\mathbf{X}_t, \mathbf{E}_t, t, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m})$
   - 对 $i \in [N]$：$\mathbf{x}_{t+\Delta t, i} = \exp_{\mathbf{x}_{t,i}}(\Delta t \cdot \hat{\mathbf{v}}_i)$
   - 对 $(i, j) \in [N]^2$，$k \in [K]$：以概率 $p_{\mathrm{flip}}(t, \Delta t) = \min\!\left(\frac{\Delta t}{1-t},\, 1\right)$ 更新 $\mathbf{E}_{t+\Delta t, ij}^{(k)} = \mathbb{1}[\hat{\mathbf{P}}_{ij}^{(k)} > 0.5]$，否则保持 $\mathbf{E}_{t, ij}^{(k)}$ 不变
3. 虚节点移除：$m_{\mathrm{pred},i} = \mathbb{1}[d_\mathcal{M}(\mathbf{x}_{1,i}, \mathbf{x}_\varnothing) \geq \epsilon_{\mathrm{null}}]$
4. 边二值化：$\mathbf{E}_{1,ij}^{(k)} \leftarrow \mathbb{1}[\hat{\mathbf{P}}_{ij}^{(k)} > p_{\mathrm{thresh}}]$

其中 $\epsilon_{\mathrm{null}} > 0$ 为虚节点检测阈值，$p_{\mathrm{thresh}} \in (0, 1)$ 为边二值化阈值。

---

## 7. 下游任务微调

### 7.1 统一掩码框架

**定义 7.1（任务掩码）。**
- 节点任务掩码 $\mathbf{m}^{\mathrm{task}} \in \{0, 1\}^N$：$m_i^{\mathrm{task}} = 1$ 表示待预测，$m_i^{\mathrm{task}} = 0$ 表示已知条件。
- 边任务掩码 $\mathbf{M}^{\mathrm{task}} \in \{0, 1\}^{N \times N}$：$M_{ij}^{\mathrm{task}} = 1$ 表示待预测，$M_{ij}^{\mathrm{task}} = 0$ 表示已知条件。

**定义 7.2（条件化含噪输入）。** 待预测部分按第 6 节正常插值，已知部分保持数据端真实值：
$$\mathbf{x}_{t,i}^{\mathrm{cond}} = \begin{cases} \exp_{\mathbf{x}_{0,i}}(t \cdot \log_{\mathbf{x}_{0,i}}(\mathbf{x}_{1,i})) & m_i^{\mathrm{task}} = 1 \\ \mathbf{x}_{1,i} & m_i^{\mathrm{task}} = 0 \end{cases}$$

$$\mathbf{E}_{t,ij}^{\mathrm{cond}} = \begin{cases} z_{ij} \cdot \mathbf{E}_{1,ij} + (1 - z_{ij}) \cdot \mathbf{E}_{0,ij} & M_{ij}^{\mathrm{task}} = 1 \\ \mathbf{E}_{1,ij} & M_{ij}^{\mathrm{task}} = 0 \end{cases}$$

**定义 7.3（条件化微调损失）。** 损失仅在待预测部分上计算：
$$\mathcal{L}_{\mathrm{cont}}^{\mathrm{task}} = \frac{1}{\sum_i m_i^{\mathrm{task}} \cdot m_i} \sum_{i=1}^N m_i^{\mathrm{task}} \cdot m_i \cdot \|\hat{\mathbf{v}}_i - \mathbf{u}_{t,i}\|_{T_{\mathbf{x}_{t,i}}\mathcal{M}}^2$$

$$\mathcal{L}_{\mathrm{disc}}^{\mathrm{task}} = \frac{1}{\sum_{i,j} M_{ij}^{\mathrm{task}}} \sum_{(i,j)} M_{ij}^{\mathrm{task}} \sum_{k=1}^K \mathrm{BCE}_{\mathrm{w}}(\hat{\mathbf{P}}_{ij}^{(k)}, \mathbf{E}_{1,ij}^{(k)})$$

注意 $\mathcal{L}_{\mathrm{cont}}^{\mathrm{task}}$ 中同时要求 $m_i^{\mathrm{task}} = 1$（待预测）和 $m_i = 1$（真实节点，非虚节点）。

### 7.2 任务一：知识图谱补全

**掩码策略。** 给定训练子图，按以下规则构造任务掩码：

- 边掩码：对每条存在的边 $(i, j)$（即 $\mathbf{E}_{1,ij} \neq \mathbf{0}_K$），独立以概率 $p_{\mathrm{mask}}^E$ 设置 $M_{ij}^{\mathrm{task}} = 1$；
- 节点掩码：对每个真实节点 $i$（$m_i = 1$），独立以概率 $p_{\mathrm{mask}}^V$ 设置 $m_i^{\mathrm{task}} = 1$，同时掩码其所有关联边：$\forall j \in [N]$，$M_{ij}^{\mathrm{task}} = M_{ji}^{\mathrm{task}} = 1$；
- 其余位置 $m_i^{\mathrm{task}} = 0$，$M_{ij}^{\mathrm{task}} = 0$。

其中 $p_{\mathrm{mask}}^E, p_{\mathrm{mask}}^V \in (0, 1)$ 为超参数。

**微调损失。**
$$\mathcal{L}_{\mathrm{KGC}} = \mathcal{L}_{\mathrm{cont}}^{\mathrm{task}} + \lambda_{\mathrm{KGC}}\,\mathcal{L}_{\mathrm{disc}}^{\mathrm{task}}$$
其中 $\lambda_{\mathrm{KGC}} > 0$ 为权重超参数。

**评估。** 对待预测事实 $(v_i, r_k, ?)$，以模型输出 $\hat{\mathbf{P}}_{ij}^{(k)}$ 对所有候选实体 $v_j$ 排序。指标：
- $\mathrm{MRR} = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \frac{1}{\mathrm{rank}_q}$，其中 $\mathcal{Q}$ 为测试查询集；
- $\mathrm{Hits}@n = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \mathbb{1}[\mathrm{rank}_q \leq n]$，$n \in \{1, 3, 10\}$。

### 7.3 任务二：文本条件子图生成

**问题设定。** 给定文本查询 $q \in \Sigma^*$，生成一个与查询语义匹配的子图 $\mathcal{G}$。

**掩码策略。** 全部掩码：
$$m_i^{\mathrm{task}} = 1 \quad \forall i \in [N], \qquad M_{ij}^{\mathrm{task}} = 1 \quad \forall (i, j) \in [N]^2$$

**文本条件替换。** 将所有节点的文本条件替换为查询嵌入：
$$\mathbf{c}_q = \phi_{\mathrm{text}}(q) \in \mathbb{R}^{d_c}, \qquad \mathbf{C}_\mathcal{V} = \mathbf{1}_N \mathbf{c}_q^\top \in \mathbb{R}^{N \times d_c}$$

其中 $\mathbf{1}_N \in \mathbb{R}^N$ 为全 $1$ 向量。$\mathbf{C}_\mathcal{R}$ 保持不变。

**微调损失。** 由于全部掩码，条件化损失退化为预训练损失形式：
$$\mathcal{L}_{\mathrm{T2G}} = \mathcal{L}_{\mathrm{cont}} + \lambda_{\mathrm{T2G}}\,\mathcal{L}_{\mathrm{disc}} + \mu_{\mathrm{T2G}}\,\mathcal{L}_{\mathrm{align}}$$
其中 $\lambda_{\mathrm{T2G}}, \mu_{\mathrm{T2G}} > 0$ 为权重超参数。对比损失 $\mathcal{L}_{\mathrm{align}}$ 鼓励生成的节点坐标与查询文本 $\mathbf{c}_q$ 在对齐空间中接近。

**评估。** 将生成子图与真实子图通过最大匹配对齐后计算：
- **节点 F1**：基于匹配节点的文本相似度，以 BERTScore 阈值判定匹配成功；
- **关系 F1**：在匹配的节点对上，比较预测的边类型集合与真实边类型集合的 F1；
- **BERTScore**：生成子图中节点文本描述与真实子图的平均 BERTScore。

### 7.4 任务三：图异常检测

**问题设定。** 给定一个子图 $\mathcal{G}$，检测其中的异常边和异常节点。本任务直接使用预训练模型，无需微调。

**异常分数构造。** 对待检测子图，以 $\mathbf{X}_1 = \mathbf{X}$，$\mathbf{E}_1 = \mathbf{E}$ 为数据端，按定义 6.1–6.2 采样噪声 $\mathbf{X}_0, \mathbf{E}_0$，按定义 6.3–6.4 构造含噪输入 $\mathbf{X}_t, \mathbf{E}_t$。

**定义 7.4（边异常分数）。** 设 $\mathcal{T} = \{t_1, \ldots, t_{|\mathcal{T}|}\} \subset (0, 1)$ 为预选时间步集合（如均匀网格）。对每条存在的边 $(i, j, k)$（$\mathbf{E}_{1,ij}^{(k)} = 1$）：
$$S_{ij}^{(k)} = 1 - \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \hat{\mathbf{P}}_{ij}^{(k)}\!\left(f_\theta(\mathbf{X}_t, \mathbf{E}_t, t, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m})\right)$$

直觉：预训练模型对正常事实给出高概率 $\hat{\mathbf{P}}_{ij}^{(k)} \approx 1$，异常分数 $S_{ij}^{(k)} \approx 0$；对异常事实给出低概率，异常分数接近 $1$。多时间步取平均提高鲁棒性。

**定义 7.5（节点异常分数）。** 设 $\mathcal{N}_i = \{j \in [N] : \mathbf{E}_{1,ij} \neq \mathbf{0}_K\}$ 为节点 $i$ 的出边邻居集。
$$S_i = \frac{1}{|\mathcal{N}_i|}\sum_{j \in \mathcal{N}_i} \max_{k:\,\mathbf{E}_{1,ij}^{(k)}=1} S_{ij}^{(k)}$$

即节点的异常程度为其所有出边中最可疑关系的平均异常分数。

**评估。** AUROC，AP（Average Precision）。

---

## 符号速查表

### 基础集合与索引

| 符号 | 类型 | 值域 | 含义 |
|------|------|------|------|
| $\mathcal{V}$ | 集合 | — | 实体集 |
| $\mathcal{R}$ | 集合 | — | 关系类型集 |
| $K$ | 标量 | $\mathbb{Z}_{>0}$ | 关系类型总数，$K \triangleq |\mathcal{R}|$ |
| $\mathcal{K}$ | 图 | — | 知识图谱 $(\mathcal{V}, \mathcal{E})$ |
| $\mathcal{E}$ | 集合 | $\subseteq \mathcal{V} \times \mathcal{R} \times \mathcal{V}$ | 事实集 |
| $\mathcal{G}$ | 图 | — | 子图 $(\mathcal{V}_\mathcal{G}, \mathcal{E}_\mathcal{G})$ |
| $N$ | 标量 | $\mathbb{Z}_{>0}$ | 子图节点数（填充后 $N = N_{\max}$） |
| $N_{\max}$ | 标量 | $\mathbb{Z}_{>0}$ | 子图节点数上界 |
| $[n]$ | 集合 | — | $\{1, 2, \ldots, n\}$ |
| $\mathcal{D}$ | 集合 | — | 训练数据集（子图集合） |

### 流形与几何

| 符号 | 类型 | 值域 | 含义 |
|------|------|------|------|
| $\mathcal{M}$ | 流形 | — | 乘积流形 $\mathbb{H}^{d_h}_{\kappa_h} \times \mathbb{S}^{d_s}_{\kappa_s} \times \mathbb{R}^{d_e}$ |
| $d_h, d_s, d_e$ | 标量 | $\mathbb{Z}_{>0}$ | 双曲、球面、欧氏子空间内在维度 |
| $D$ | 标量 | $\mathbb{Z}_{>0}$ | 总环境维度，$D = (d_h+1)+(d_s+1)+d_e$ |
| $\kappa_h$ | 标量 | $\mathbb{R}_{<0}$ | 双曲空间曲率（可学习） |
| $\kappa_s$ | 标量 | $\mathbb{R}_{>0}$ | 球面曲率（可学习） |
| $\langle \cdot, \cdot \rangle_{\mathrm{L}}$ | 运算 | $\mathbb{R}$ | Lorentz 内积 |
| $T_\mathbf{x}\mathcal{M}$ | 空间 | — | $\mathbf{x}$ 处的切空间 |
| $\exp_\mathbf{x}$ | 映射 | $T_\mathbf{x}\mathcal{M} \to \mathcal{M}$ | 指数映射 |
| $\log_\mathbf{x}$ | 映射 | $\mathcal{M} \to T_\mathbf{x}\mathcal{M}$ | 对数映射 |
| $d_\mathcal{M}$ | 函数 | $\mathcal{M} \times \mathcal{M} \to \mathbb{R}_{\geq 0}$ | 乘积流形测地距离 |
| $d_{\mathbb{H}}, d_{\mathbb{S}}, d_{\mathbb{R}}$ | 函数 | — | 各子空间测地距离 |
| $\pi$ | 映射 | $\mathcal{M} \to \mathbb{R}^D$ | 坐标投影 |
| $\mathbf{x}_\varnothing$ | 向量 | $\mathcal{M}$ | 虚节点锚点 |

### 子图表示

| 符号 | 类型 | 值域 | 含义 |
|------|------|------|------|
| $\mathbf{x}_i$ | 向量 | $\mathcal{M}$ | 节点 $i$ 的流形坐标 |
| $\mathbf{x}_i^{\mathbb{H}}, \mathbf{x}_i^{\mathbb{S}}, \mathbf{x}_i^{\mathbb{R}}$ | 向量 | $\mathbb{R}^{d_h+1}, \mathbb{R}^{d_s+1}, \mathbb{R}^{d_e}$ | 各子空间分量 |
| $\mathbf{X}$ | 元组 | $\mathcal{M}^N$ | 节点坐标序列 |
| $\mathbf{E}$ | 张量 | $\{0,1\}^{N \times N \times K}$ | 边类型张量 |
| $\mathbf{E}_{ij}$ | 向量 | $\{0,1\}^K$ | 节点 $i$ 到 $j$ 的多热边类型 |
| $\mathbf{C}_\mathcal{V}$ | 矩阵 | $\mathbb{R}^{N \times d_c}$ | 节点文本条件矩阵 |
| $\mathbf{C}_\mathcal{R}$ | 矩阵 | $\mathbb{R}^{K \times d_c}$ | 关系文本条件矩阵 |
| $\mathbf{c}_i$ | 向量 | $\mathbb{R}^{d_c}$ | 节点 $i$ 的文本条件（$\mathbf{c}_i \triangleq \mathbf{c}_{v_i}$） |
| $\mathbf{c}_{r_k}$ | 向量 | $\mathbb{R}^{d_c}$ | 关系 $r_k$ 的文本条件 |
| $\mathbf{m}$ | 向量 | $\{0,1\}^N$ | 虚节点掩码 |
| $\phi_{\mathrm{text}}$ | 映射 | $\Sigma^* \to \mathbb{R}^{d_c}$ | 预训练文本编码器 |
| $d_c$ | 标量 | $\mathbb{Z}_{>0}$ | 文本编码器输出维度 |

### 模型架构（RieFormer）

| 符号 | 类型 | 值域 | 含义 |
|------|------|------|------|
| $f_\theta$ | 映射 | — | RieFormer 模型 |
| $\theta$ | — | — | 所有可学习参数（含 $\kappa_h, \kappa_s$） |
| $L$ | 标量 | $\mathbb{Z}_{>0}$ | RieFormer 层数 |
| $d_v$ | 标量 | $\mathbb{Z}_{>0}$ | 节点隐藏维度 |
| $d_{e'}$ | 标量 | $\mathbb{Z}_{>0}$ | 边隐藏维度 |
| $d_r$ | 标量 | $\mathbb{Z}_{>0}$ | 关系嵌入维度 |
| $d_t$ | 标量 | $\mathbb{Z}_{>0}$（偶数） | 时间嵌入维度 |
| $d_{\mathrm{head}}$ | 标量 | $\mathbb{Z}_{>0}$（偶数） | 注意力头维度，$d_{\mathrm{head}} = d_v / n_h$ |
| $d_\mathrm{a}$ | 标量 | $\mathbb{Z}_{>0}$ | 对齐空间维度 |
| $n_h$ | 标量 | $\mathbb{Z}_{>0}$ | 注意力头数 |
| $\mathbf{h}_i^{V,(l)}$ | 向量 | $\mathbb{R}^{d_v}$ | 第 $l$ 层节点 $i$ 的嵌入 |
| $\mathbf{h}_{ij}^{E,(l)}$ | 向量 | $\mathbb{R}^{d_{e'}}$ | 第 $l$ 层边 $(i,j)$ 的嵌入 |
| $\mathbf{t}_{\mathrm{emb}}$ | 向量 | $\mathbb{R}^{d_v}$ | 时间嵌入 |
| $\hat{\mathbf{v}}_i$ | 向量 | $T_{\mathbf{x}_{t,i}}\mathcal{M}$ | 模型预测切向量 |
| $\hat{\mathbf{P}}_{ij}^{(k)}$ | 标量 | $[0,1]$ | 边类型概率预测 |
| $\kappa^{(s)}$ | 函数 | $\mathcal{M} \times \mathcal{M} \to \mathbb{R}$ | 第 $s$ 头 Geodesic Kernel |
| $\hat{\rho}_i$ | 标量 | $\mathbb{R}_{\geq 0}$ | 层级深度估计 |
| $\odot$ | 运算 | — | 逐元素乘积（Hadamard 积） |
| $\|$ | 运算 | — | 向量拼接 |

### 流匹配

| 符号 | 类型 | 值域 | 含义 |
|------|------|------|------|
| $t$ | 标量 | $[0,1]$ | 流匹配时间步（$0$=噪声，$1$=数据） |
| $\mathbf{X}_0, \mathbf{E}_0$ | — | — | 噪声端采样 |
| $\mathbf{X}_1, \mathbf{E}_1$ | — | — | 数据端（训练子图） |
| $\mathbf{x}_{t,i}$ | 向量 | $\mathcal{M}$ | 节点 $i$ 在时间 $t$ 的插值坐标 |
| $\mathbf{E}_{t,ij}$ | 向量 | $\{0,1\}^K$ | 边 $(i,j)$ 在时间 $t$ 的插值类型 |
| $z_{ij}$ | 标量 | $\{0,1\}$ | 边插值掩码，$z_{ij} \sim \mathrm{Bernoulli}(t)$ |
| $\mathbf{u}_{t,i}$ | 向量 | $T_{\mathbf{x}_{t,i}}\mathcal{M}$ | 条件向量场目标 |
| $\rho_k$ | 标量 | $[0,1]$ | 关系 $r_k$ 的边际频率 |
| $p_0^{\mathcal{M}}$ | 分布 | — | 连续噪声先验 |
| $R_{\mathbb{H}}$ | 标量 | $\mathbb{R}_{>0}$ | 双曲均匀分布截断半径 |
| $\sigma_0$ | 标量 | $\mathbb{R}_{>0}$ | 欧氏噪声标准差 |
| $T$ | 标量 | $\mathbb{Z}_{>0}$ | 推理步数 |
| $\Delta t$ | 标量 | $(0,1]$ | 推理步长，$\Delta t = 1/T$ |
| $p_{\mathrm{flip}}$ | 函数 | $[0,1]$ | 边更新翻转概率 |

### 损失函数

| 符号 | 类型 | 值域 | 含义 |
|------|------|------|------|
| $\mathcal{L}_{\mathrm{cont}}$ | 标量 | $\mathbb{R}_{\geq 0}$ | 流形向量场损失 |
| $\mathcal{L}_{\mathrm{disc}}$ | 标量 | $\mathbb{R}_{\geq 0}$ | 边类型损失 |
| $\mathcal{L}_{\mathrm{align}}$ | 标量 | $\mathbb{R}_{\geq 0}$ | 图-文对比损失 |
| $\mathcal{L}$ | 标量 | $\mathbb{R}_{\geq 0}$ | 总训练损失 |
| $\lambda, \mu$ | 标量 | $\mathbb{R}_{>0}$ | 损失权重 |
| $\tau$ | 标量 | $\mathbb{R}_{>0}$ | 对比损失温度 |
| $w_k^+$ | 标量 | $\mathbb{R}_{>0}$ | 关系 $r_k$ 正样本权重 |
| $w_{\max}$ | 标量 | $\mathbb{R}_{>0}$ | 正样本权重截断上限 |
| $\mathcal{S}$ | 集合 | $\subseteq [N]^2$ | 边损失采样集 |
| $\mathcal{B}$ | 集合 | $\subseteq [N]$ | 对比损失 mini-batch 索引 |
| $M_{\mathrm{align}}$ | 标量 | $\mathbb{Z}_{>0}$ | 对比损失最大节点子采样数 |
| $\mu_{\mathrm{LN}}, \sigma_{\mathrm{LN}}$ | 标量 | $\mathbb{R}, \mathbb{R}_{>0}$ | Logit-Normal 时间分布参数 |
| $\epsilon_t, \epsilon_p, \epsilon_\kappa$ | 标量 | $\mathbb{R}_{>0}$ | 数值稳定/截断常数 |

### 下游任务

| 符号 | 类型 | 值域 | 含义 |
|------|------|------|------|
| $\mathbf{m}^{\mathrm{task}}$ | 向量 | $\{0,1\}^N$ | 节点任务掩码 |
| $\mathbf{M}^{\mathrm{task}}$ | 矩阵 | $\{0,1\}^{N \times N}$ | 边任务掩码 |
| $\mathcal{L}_{\mathrm{cont}}^{\mathrm{task}}, \mathcal{L}_{\mathrm{disc}}^{\mathrm{task}}$ | 标量 | $\mathbb{R}_{\geq 0}$ | 条件化微调损失 |
| $p_{\mathrm{mask}}^E, p_{\mathrm{mask}}^V$ | 标量 | $(0,1)$ | KGC 掩码概率 |
| $\mathbf{c}_q$ | 向量 | $\mathbb{R}^{d_c}$ | T2G 查询文本嵌入 |
| $S_{ij}^{(k)}$ | 标量 | $[0,1]$ | 边异常分数 |
| $S_i$ | 标量 | $[0,1]$ | 节点异常分数 |
| $\mathcal{T}$ | 集合 | $\subset (0,1)$ | GAD 预选时间步集 |
| $\mathcal{N}_i$ | 集合 | $\subseteq [N]$ | 节点 $i$ 的出边邻居集 |
| $\epsilon_{\mathrm{null}}$ | 标量 | $\mathbb{R}_{>0}$ | 虚节点检测阈值 |
| $p_{\mathrm{thresh}}$ | 标量 | $(0,1)$ | 边二值化阈值 |
