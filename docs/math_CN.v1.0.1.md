# RiemannFM：面向知识图谱生成的乘积流形上几何-文本感知流匹配

---

# Part I. 基础与表示

本部分建立文档所需的静态对象与几何原语。第 1-3 章形式化知识图谱、子图及其矩阵表示；第 4 章给出乘积流形上的几何运算；第 5 章引入关系的全局可学习嵌入。所有后续章节（流匹配框架、RieFormer 架构、训练与推理）均建立在本部分基础上。

**方法的核心对象**：三类随机变量协同演化，共享一个时间变量 $t \in [0, 1]$：

- 节点坐标 $\mathbf{X}$：住在乘积流形 $\mathcal{M}$ 上，做**连续流匹配**（黎曼测地线插值）；
- 边类型张量 $\mathbf{E}$：离散值，做**离散流匹配**（掩码调度解码）；
- 关系全局嵌入 $\mathbf{R}$：欧氏空间的可学习参数（不做流匹配），作为**条件**参与节点流和边流。

本部分的角色是静态描述：定义 $\mathbf{X}, \mathbf{E}, \mathbf{R}$ 所在的空间，以及 $\mathcal{M}$ 上的几何运算。动力学（$t$ 依赖的演化）在 Part II 引入。

---

## 第 1 章 基础集合与索引约定

### 1.1 基础集合

**定义 1.1（基础集合）。**

- **实体集**：$\mathcal{V} = \{v_1, v_2, \ldots, v_{|\mathcal{V}|}\}$。
- **关系类型集**：$\mathcal{R} = \{r_1, r_2, \ldots, r_K\}$，$K \triangleq |\mathcal{R}|$。
- **知识图谱**：$\mathcal{K} = (\mathcal{V}, \mathcal{E})$，其中事实集 $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{R} \times \mathcal{V}$。一个事实 $(v_i, r_k, v_j) \in \mathcal{E}$ 表示实体 $v_i$ 通过关系 $r_k$ 有向地连接到实体 $v_j$。

### 1.2 允许的边结构约定

**自环**：允许 $i = j$，即 $(v_i, r_k, v_i) \in \mathcal{E}$ 合法。

**多关系边**：同一有序实体对 $(v_i, v_j)$ 可关联多种关系类型，即可同时存在 $(v_i, r_a, v_j) \in \mathcal{E}$ 和 $(v_i, r_b, v_j) \in \mathcal{E}$，$a \neq b$。

**相反关系**：语义相反的关系（如"父亲"与"子女"）作为 $\mathcal{R}$ 中独立的元素存在，不人工添加反向关系。

### 1.3 记号约定

对任意正整数 $n$，记 $[n] \triangleq \{1, 2, \ldots, n\}$。

粗体字母（如 $\mathbf{X}, \mathbf{R}, \mathbf{E}$）表示向量、矩阵或张量；花体字母（如 $\mathcal{V}, \mathcal{R}, \mathcal{M}$）表示集合或流形；黑板体（如 $\mathbb{H}, \mathbb{S}, \mathbb{R}$）表示标准几何空间。$\mathcal{R}$（花体）指关系类型集，$\mathbb{R}$（黑板体）指实数空间，$\mathbf{R}$（粗体）指本文后续引入的关系全局嵌入矩阵，三者字符不同。

---

## 第 2 章 子图的形式化定义

**定义 2.1（诱导子图）。** 子图 $\mathcal{G} = (\mathcal{V}_\mathcal{G}, \mathcal{E}_\mathcal{G})$ 是 $\mathcal{K}$ 的诱导子图：

- $\mathcal{V}_\mathcal{G} \subseteq \mathcal{V}$，$|\mathcal{V}_\mathcal{G}| = N$；
- $\mathcal{E}_\mathcal{G} = \{(v_i, r_k, v_j) \in \mathcal{E} \mid v_i, v_j \in \mathcal{V}_\mathcal{G}\}$。

对 $\mathcal{V}_\mathcal{G}$ 中的节点重新编号为 $[N]$。后续 $i, j \in [N]$ 均指子图内的局部编号。

**设计理由。** 采用诱导子图而非任意子图，保证：(i) 子图内节点的邻接结构由全图唯一确定，避免采样歧义；(ii) 子图边集与节点集一致，利于 $N \times N$ 边张量的紧凑表示。

---

## 第 3 章 子图的矩阵表示

子图 $\mathcal{G}$ 编码为五元组 $(\mathbf{X}, \mathbf{E}, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m})$，分别对应节点流形坐标、边类型张量、节点文本条件矩阵、关系文本条件矩阵和节点掩码。各分量在以下小节中依次定义。

引入时间变量 $t \in [0, 1]$ 后（Part II），节点坐标 $\mathbf{X}$ 与边张量 $\mathbf{E}$ 扩展为时间索引版本 $\mathbf{X}_t, \mathbf{E}_t$，分别在 $\mathcal{M}^N$ 上做连续流匹配与在 $\{0,1,\mathrm{MASK}\}^{N \times N \times K}$ 上做离散流匹配。**关系侧不随 $t$ 演化**：关系仅通过一个全局可学习嵌入矩阵 $\mathbf{R} \in \mathbb{R}^{K \times d_r}$ 参与模型（详见第 5 章），作为条件而非被生成的变量。

### 3.1 嵌入空间：乘积流形

**定义 3.1（Lorentz 内积）。** 对 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^{d_h+1}$：

$$\langle \mathbf{a}, \mathbf{b} \rangle_{\mathrm{L}} = -a_0 b_0 + \sum_{l=1}^{d_h} a_l b_l$$

**定义 3.2（乘积流形）。**

$$\mathcal{M} = \mathbb{H}^{d_h}_{\kappa_h} \times \mathbb{S}^{d_s}_{\kappa_s} \times \mathbb{R}^{d_e}$$

其中：

- $\mathbb{H}^{d_h}_{\kappa_h} = \{\mathbf{z} \in \mathbb{R}^{d_h+1} \mid \langle \mathbf{z}, \mathbf{z} \rangle_{\mathrm{L}} = 1/\kappa_h,\; z_0 > 0\}$，$d_h$ 维双曲空间（Lorentz 模型），曲率 $\kappa_h < 0$；
- $\mathbb{S}^{d_s}_{\kappa_s} = \{\mathbf{z} \in \mathbb{R}^{d_s+1} \mid \|\mathbf{z}\|_2^2 = 1/\kappa_s\}$，$d_s$ 维球面，曲率 $\kappa_s > 0$；
- $\mathbb{R}^{d_e}$ 为 $d_e$ 维欧氏空间。

**环境空间维度**：$D \triangleq (d_h + 1) + (d_s + 1) + d_e$。

**分量记号**：对 $\mathbf{x} \in \mathcal{M}$，记 $\mathbf{x}^{\mathbb{H}} \in \mathbb{R}^{d_h+1}$，$\mathbf{x}^{\mathbb{S}} \in \mathbb{R}^{d_s+1}$，$\mathbf{x}^{\mathbb{R}} \in \mathbb{R}^{d_e}$ 为三个子空间分量。

**设计理由。** 乘积流形让三种几何偏好并存：双曲分量 $\mathbb{H}^{d_h}_{\kappa_h}$ 适合编码**层级结构**（指数体积增长匹配树状展开）；球面分量 $\mathbb{S}^{d_s}_{\kappa_s}$ 适合编码**周期/循环结构**；欧氏分量 $\mathbb{R}^{d_e}$ 承担**无层级的开放语义维度**。节点作为实体代表，同时具有层级（分类）、周期（时间相关）、开放（属性）三类特征，乘积流形为其提供足够的几何表达能力。关系不放流形的理由详见第 5 章。

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

- **有向性**：$\mathbf{E}$ 作为有向张量不强制对称，一般地 $\mathbf{E}_{ij} \neq \mathbf{E}_{ji}$；
- **近似对称子集**：存在语义对称关系子集 $\mathcal{R}_{\mathrm{sym}} \subseteq \mathcal{R}$（如 spouse、sibling、shares border with、twin city），在数据中对任意 $r_k \in \mathcal{R}_{\mathrm{sym}}$ 有 $\mathbf{E}_{ij}^{(k)} \approx \mathbf{E}_{ji}^{(k)}$（实测 Wikidata5M 上反向边覆盖率 $\sim 90\text{--}99\%$，其余为标注缺失）；
- **多重性**：$\|\mathbf{E}_{ij}\|_1$ 可大于 $1$；
- **稀疏性**：平均出度 $\bar{d} = \frac{1}{N}\sum_{i=1}^{N} |\{j \in [N] : \mathbf{E}_{ij} \neq \mathbf{0}_K\}| \ll N$。

**离散流匹配的预告**：Part II 中 $\mathbf{E}$ 将扩展为时间索引 $\mathbf{E}_t$，其状态空间扩展为 $\{0, 1, \mathrm{MASK}\}^{N \times N \times K}$。掩码状态由独立的掩码指示张量 $\boldsymbol{\mu}_t \in \{0, 1\}^{N \times N}$ 标注。

### 3.4 文本条件

**记号**：设 $\Sigma$ 为字符表，$\Sigma^*$ 为 $\Sigma$ 上的有限字符串集，$\circ$ 为字符串拼接运算。

**定义 3.5（文本条件向量）。** 设 $\phi_{\mathrm{text}}: \Sigma^* \to \mathbb{R}^{d_c}$ 为预训练文本编码器（如 Qwen3-Embedding，$d_c = 768$）。对每个节点 $v_i$：

$$\mathbf{c}_{v_i} = \phi_{\mathrm{text}}(\mathrm{label}_{v_i} \circ \mathrm{desc}_{v_i}) \in \mathbb{R}^{d_c}$$

对每个关系类型 $r_k$：

$$\mathbf{c}_{r_k} = \phi_{\mathrm{text}}(\mathrm{label}_{r_k} \circ \mathrm{desc}_{r_k}) \in \mathbb{R}^{d_c}$$

后续简记 $\mathbf{c}_i \triangleq \mathbf{c}_{v_i}$。

**定义 3.6（文本条件矩阵）。**

- 节点文本矩阵：$\mathbf{C}_\mathcal{V} = (\mathbf{c}_1, \ldots, \mathbf{c}_N)^\top \in \mathbb{R}^{N \times d_c}$（子图级）；
- 关系文本矩阵：$\mathbf{C}_\mathcal{R} = (\mathbf{c}_{r_1}, \ldots, \mathbf{c}_{r_K})^\top \in \mathbb{R}^{K \times d_c}$（全局，所有子图共享）。

**冻结性**：$\phi_{\mathrm{text}}$ 在预训练与微调阶段均冻结。$\mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}$ 可预计算并缓存。

### 3.5 虚节点填充

**定义 3.7（虚节点填充）。** 设 $N_{\max} \in \mathbb{Z}_{>0}$ 为子图节点数上界（预设超参数）。对 $|\mathcal{V}_\mathcal{G}| \leq N_{\max}$ 的子图，填充 $N_{\max} - |\mathcal{V}_\mathcal{G}|$ 个虚节点，使总节点数为 $N_{\max}$：

- 虚节点坐标：$\mathbf{x}_i = \mathbf{x}_\varnothing \in \mathcal{M}$，其中锚点各分量为

$$\mathbf{x}_\varnothing^{\mathbb{H}} = \left(\tfrac{1}{\sqrt{|\kappa_h|}},\, 0,\, \ldots,\, 0\right) \in \mathbb{R}^{d_h+1}, \quad \mathbf{x}_\varnothing^{\mathbb{S}} = \left(\tfrac{1}{\sqrt{\kappa_s}},\, 0,\, \ldots,\, 0\right) \in \mathbb{R}^{d_s+1}, \quad \mathbf{x}_\varnothing^{\mathbb{R}} = \mathbf{0}_{d_e}$$

- 虚节点边：$\forall j \in [N_{\max}]$，$\mathbf{E}_{ij} = \mathbf{E}_{ji} = \mathbf{0}_K$；
- 虚节点文本：$\mathbf{c}_i = \mathbf{0}_{d_c}$。

可验证 $\mathbf{x}_\varnothing^{\mathbb{H}}$ 满足 $\langle \mathbf{x}_\varnothing^{\mathbb{H}}, \mathbf{x}_\varnothing^{\mathbb{H}} \rangle_{\mathrm{L}} = -1/|\kappa_h| = 1/\kappa_h$ 且首分量大于零；$\mathbf{x}_\varnothing^{\mathbb{S}}$ 满足 $\|\mathbf{x}_\varnothing^{\mathbb{S}}\|_2^2 = 1/\kappa_s$。

**定义 3.8（节点掩码）。**

$$\mathbf{m} \in \{0,1\}^{N_{\max}}, \quad m_i = \begin{cases} 1 & \text{真实节点} \\ 0 & \text{虚节点} \end{cases}$$

满足 $\sum_{i=1}^{N_{\max}} m_i = |\mathcal{V}_\mathcal{G}|$。

**符号约定**：自此以后，$N \triangleq N_{\max}$。后续所有公式中 $i, j \in [N]$ 包含真实节点与虚节点，虚节点通过 $\mathbf{m}$ 识别和屏蔽。

**设计理由。** 虚节点填充使所有子图张量具有固定形状 $(N, N, \ldots)$，便于 batch 化处理；锚点 $\mathbf{x}_\varnothing$ 在各流形分量上是自然的"原点"（双曲时轴上、球面极点、欧氏零点），其切空间可正交规范化；通过 $\mathbf{m}$ 屏蔽虚节点，损失与注意力计算不受其影响。

---

## 第 4 章 流形上的基本运算

本章给出 $\mathcal{M}$ 上流匹配框架所需的全部几何原语：测地距离、切空间、对数映射、指数映射、切空间黎曼范数、切空间正交投影、流形约束投影。这些原语主要服务于**节点连续流匹配**——节点坐标在 $\mathcal{M}$ 上从先验流到数据分布的过程需要测地线插值（$\exp, \log$ 组合）、向量场表示（切空间）、模型预测的几何合法性（$\mathrm{Proj}$）以及推理时的数值稳定性（$\mathrm{Retract}$）。

边离散流匹配不涉及流形运算（边的演化发生在离散状态空间，见第 8 章）；关系在欧氏空间中作为可学习嵌入（见第 5 章），亦不涉及本章原语。

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

**实现注记**：切向量 $\mathbf{v}^\mathbb{H}$ 作为切空间元素满足 $\langle \mathbf{v}^\mathbb{H}, \mathbf{x}^\mathbb{H} \rangle_\mathrm{L} = 0$，意味着 $\mathbf{v}^\mathbb{H}$ 在 ambient $\mathbb{R}^{d_h+1}$ 空间中的 Lorentz 内积自范数 $\langle \mathbf{v}^\mathbb{H}, \mathbf{v}^\mathbb{H} \rangle_\mathrm{L}$ 非负，故 $\|\mathbf{v}^\mathbb{H}\|_\mathrm{L} = \sqrt{\langle \mathbf{v}^\mathbb{H}, \mathbf{v}^\mathbb{H} \rangle_\mathrm{L}}$ 良定。

### 4.6 切空间正交投影

切空间正交投影 $\mathrm{Proj}_\mathbf{x}$ 把 ambient 空间 $\mathbb{R}^D$ 中的任意向量映射到切空间 $T_\mathbf{x}\mathcal{M}$。在本文架构中，模型的节点切向量预测头输出 ambient 向量，需经此投影保证输出严格在切空间内（否则后续 $\exp_\mathbf{x}$ 不能使用）。

**定义 4.7（切空间正交投影）。** $\mathrm{Proj}_\mathbf{x}: \mathbb{R}^D \to T_\mathbf{x}\mathcal{M}$。对 $\mathbf{u} = (\mathbf{u}^\mathbb{H}, \mathbf{u}^\mathbb{S}, \mathbf{u}^\mathbb{R}) \in \mathbb{R}^{d_h+1} \times \mathbb{R}^{d_s+1} \times \mathbb{R}^{d_e}$：

- 双曲分量：

$$\mathrm{Proj}_{T_{\mathbf{x}^\mathbb{H}}\mathbb{H}}(\mathbf{u}^\mathbb{H}) = \mathbf{u}^\mathbb{H} + \kappa_h \langle \mathbf{u}^\mathbb{H}, \mathbf{x}^\mathbb{H} \rangle_\mathrm{L} \cdot \mathbf{x}^\mathbb{H}$$

- 球面分量：

$$\mathrm{Proj}_{T_{\mathbf{x}^\mathbb{S}}\mathbb{S}}(\mathbf{u}^\mathbb{S}) = \mathbf{u}^\mathbb{S} - (\mathbf{u}^{\mathbb{S}\top} \mathbf{x}^\mathbb{S}) \cdot \mathbf{x}^\mathbb{S}$$

- 欧氏分量：

$$\mathrm{Proj}_{T_{\mathbf{x}^\mathbb{R}}\mathbb{R}^{d_e}}(\mathbf{u}^\mathbb{R}) = \mathbf{u}^\mathbb{R}$$

乘积：$\mathrm{Proj}_\mathbf{x}(\mathbf{u}) = (\mathrm{Proj}_{T_{\mathbf{x}^\mathbb{H}}\mathbb{H}}(\mathbf{u}^\mathbb{H}),\, \mathrm{Proj}_{T_{\mathbf{x}^\mathbb{S}}\mathbb{S}}(\mathbf{u}^\mathbb{S}),\, \mathrm{Proj}_{T_{\mathbf{x}^\mathbb{R}}\mathbb{R}^{d_e}}(\mathbf{u}^\mathbb{R}))$。

**命题 4.2（投影的正交性）。** 设 $\hat{\mathbf{v}} = \mathrm{Proj}_\mathbf{x}(\mathbf{u})$。则：

- $\langle \hat{\mathbf{v}}^\mathbb{H}, \mathbf{x}^\mathbb{H} \rangle_\mathrm{L} = 0$
- $\hat{\mathbf{v}}^{\mathbb{S}\top} \mathbf{x}^\mathbb{S} = 0$
- $\hat{\mathbf{v}} \in T_\mathbf{x}\mathcal{M}$

*证明要点*：双曲分量 $\langle \hat{\mathbf{v}}^\mathbb{H}, \mathbf{x}^\mathbb{H} \rangle_\mathrm{L} = \langle \mathbf{u}^\mathbb{H}, \mathbf{x}^\mathbb{H} \rangle_\mathrm{L} + \kappa_h \langle \mathbf{u}^\mathbb{H}, \mathbf{x}^\mathbb{H} \rangle_\mathrm{L} \cdot \langle \mathbf{x}^\mathbb{H}, \mathbf{x}^\mathbb{H} \rangle_\mathrm{L} = \langle \mathbf{u}^\mathbb{H}, \mathbf{x}^\mathbb{H} \rangle_\mathrm{L} (1 + \kappa_h / \kappa_h) = 0$（使用 $\langle \mathbf{x}^\mathbb{H}, \mathbf{x}^\mathbb{H} \rangle_\mathrm{L} = 1/\kappa_h$）。球面分量类似代入 $\mathbf{x}^{\mathbb{S}\top}\mathbf{x}^\mathbb{S} = 1/\kappa_s$ 验证。∎

**命题 4.3（幂等性与最小化）。** $\mathrm{Proj}_\mathbf{x}$ 在 ambient 内积下是切空间的正交投影算子：

- 幂等：$\mathrm{Proj}_\mathbf{x} \circ \mathrm{Proj}_\mathbf{x} = \mathrm{Proj}_\mathbf{x}$
- 最小化：$\hat{\mathbf{v}} = \mathrm{Proj}_\mathbf{x}(\mathbf{u})$ 是 $T_\mathbf{x}\mathcal{M}$ 中距离 $\mathbf{u}$ 最近的点（在 ambient $\ell_2$ 度量下，欧氏和球面分量；双曲分量在 Lorentz 度量下）

**用途**：本投影算子在 RieFormer 的节点切向量头中显式调用——MLP 输出 ambient 向量，经 $\mathrm{Proj}_\mathbf{x}$ 映射到切空间后，才用于流匹配损失的黎曼范数计算与推理时的指数映射。

### 4.7 流形约束投影

推理阶段节点沿切向量场 $\exp_\mathbf{x}$ 积分（多步 ODE 求解）时，由于浮点累积误差，$\mathbf{x}_t$ 可能轻微偏离流形约束（$\langle \mathbf{x}_t^\mathbb{H}, \mathbf{x}_t^\mathbb{H} \rangle_\mathrm{L} \neq 1/\kappa_h$ 或 $\|\mathbf{x}_t^\mathbb{S}\|_2^2 \neq 1/\kappa_s$）。流形约束投影 $\mathrm{Retract}$ 把 ambient 空间中的任意点投影回流形，作为数值稳定性的安全网。

**定义 4.8（流形约束投影）。** $\mathrm{Retract}: \mathbb{R}^D \setminus Z \to \mathcal{M}$，其中 $Z$ 为各分量的退化集（原点等）。对 $\mathbf{z} = (\mathbf{z}^\mathbb{H}, \mathbf{z}^\mathbb{S}, \mathbf{z}^\mathbb{R}) \in \mathbb{R}^{d_h+1} \times \mathbb{R}^{d_s+1} \times \mathbb{R}^{d_e}$：

- 双曲分量：若 $\kappa_h \langle \mathbf{z}^\mathbb{H}, \mathbf{z}^\mathbb{H} \rangle_\mathrm{L} > 0$ 且 $z_0^\mathbb{H} > 0$，则

$$\mathrm{Retract}_\mathbb{H}(\mathbf{z}^\mathbb{H}) = \frac{\mathbf{z}^\mathbb{H}}{\sqrt{\kappa_h \langle \mathbf{z}^\mathbb{H}, \mathbf{z}^\mathbb{H} \rangle_\mathrm{L}}}$$

- 球面分量：若 $\mathbf{z}^\mathbb{S} \neq \mathbf{0}$，则

$$\mathrm{Retract}_\mathbb{S}(\mathbf{z}^\mathbb{S}) = \frac{\mathbf{z}^\mathbb{S}}{\sqrt{\kappa_s} \|\mathbf{z}^\mathbb{S}\|_2}$$

- 欧氏分量：

$$\mathrm{Retract}_\mathbb{R}(\mathbf{z}^\mathbb{R}) = \mathbf{z}^\mathbb{R}$$

乘积：$\mathrm{Retract}(\mathbf{z}) = (\mathrm{Retract}_\mathbb{H}(\mathbf{z}^\mathbb{H}),\, \mathrm{Retract}_\mathbb{S}(\mathbf{z}^\mathbb{S}),\, \mathrm{Retract}_\mathbb{R}(\mathbf{z}^\mathbb{R}))$。

**命题 4.4（投影后满足流形约束）。** 对 $\mathbf{z} \in \mathbb{R}^D \setminus Z$，$\mathrm{Retract}(\mathbf{z}) \in \mathcal{M}$。

*证明要点*：双曲分量取 $\tilde{\mathbf{z}} = \mathbf{z}^\mathbb{H} / \sqrt{\kappa_h \langle \mathbf{z}^\mathbb{H}, \mathbf{z}^\mathbb{H} \rangle_\mathrm{L}}$，有 $\langle \tilde{\mathbf{z}}, \tilde{\mathbf{z}} \rangle_\mathrm{L} = \langle \mathbf{z}^\mathbb{H}, \mathbf{z}^\mathbb{H} \rangle_\mathrm{L} / (\kappa_h \langle \mathbf{z}^\mathbb{H}, \mathbf{z}^\mathbb{H} \rangle_\mathrm{L}) = 1/\kappa_h$。$z_0$ 的正性由 $\mathbf{z}^\mathbb{H}$ 的原始正性保持。球面分量：$\|\tilde{\mathbf{z}}\|_2^2 = 1/\kappa_s$，满足球面约束。∎

**设计理由。** 相比测地投影（最近点投影），本算子**沿锚点-$\mathbf{z}$ 方向的径向缩放**，实现简单、数值稳定、可微分。对轻微的约束违反（如数值精度导致的 $\langle \mathbf{z}^\mathbb{H}, \mathbf{z}^\mathbb{H} \rangle_\mathrm{L} = 1/\kappa_h + \epsilon$），径向投影与测地投影几乎等价；对较大违反，径向投影仍能保证结果在流形上，作为安全网足够。

**用途**：本投影在推理时使用——节点 ODE 积分每步后对 $\mathbf{X}_t$ 做 $\mathrm{Retract}$ 矫正，防止数值漂移累积破坏流形约束。训练阶段由于每步都从先验重新采样 $\mathbf{X}_0$ 并闭式构造 $\mathbf{X}_t$（详见第 7 章），不存在累积漂移问题，故训练阶段不调用 $\mathrm{Retract}$。

### 4.8 本章小结：几何原语清单

| 原语 | 符号 | 类型签名 | 训练用途 | 推理用途 |
|------|------|---------|---------|---------|
| Lorentz 内积 | $\langle \cdot, \cdot \rangle_\mathrm{L}$ | $\mathbb{R}^{d_h+1} \times \mathbb{R}^{d_h+1} \to \mathbb{R}$ | 基础运算 | 基础运算 |
| 测地距离 | $d_\mathcal{M}$ | $\mathcal{M} \times \mathcal{M} \to \mathbb{R}_{\geq 0}$ | Geodesic Kernel | 无 |
| 对数映射 | $\log_\mathbf{x}$ | $\mathcal{M} \to T_\mathbf{x}\mathcal{M}$ | 测地插值、目标场 | 主干内几何信号 |
| 指数映射 | $\exp_\mathbf{x}$ | $T_\mathbf{x}\mathcal{M} \to \mathcal{M}$ | 测地插值 | ODE 积分步 |
| 黎曼范数 | $\|\cdot\|_{T_\mathbf{x}\mathcal{M}}$ | $T_\mathbf{x}\mathcal{M} \to \mathbb{R}_{\geq 0}$ | FM 损失 | 无 |
| 切空间投影 | $\mathrm{Proj}_\mathbf{x}$ | $\mathbb{R}^D \to T_\mathbf{x}\mathcal{M}$ | 切向量头 | 切向量头 |
| 流形约束投影 | $\mathrm{Retract}$ | $\mathbb{R}^D \setminus Z \to \mathcal{M}$ | 无 | ODE 每步数值修复 |

所有后续章节的流形运算（测地线插值、目标向量场、切向量头、推理 ODE）仅依赖此清单。闭式展开（$\cosh, \sinh, \cos, \sin$ 等）不再在正文重复出现，留给实现。

---

## 第 5 章 关系的全局可学习嵌入

### 5.1 动机与设计立场

节点（§3.2）和边（§3.3）作为子图级对象，随子图变化而变化，是流匹配建模的**目标分布**。关系则不同：关系类型集 $\mathcal{R}$ 对整个知识图谱 $\mathcal{K}$ 全局固定，所有子图共享同一组关系；关系本身不是"被生成的变量"，而是**类型标签**，用于标注边的性质。

由此出发，本文对关系采取"**条件参数**"的设计立场：关系通过一个全局可学习的嵌入矩阵 $\mathbf{R}$ 参与模型，$\mathbf{R}$ 作为与节点坐标 $\mathbf{X}$、边张量 $\mathbf{E}$ 对等的模型输入之一，但本身**不随时间 $t$ 演化、不经历流匹配过程**。关系的作用是：

1. **为边的类型预测提供条件**：$K+1$ 维边预测头（见第 17 章）通过与 $\mathbf{R}$ 的双线性交互计算每类关系的 logit；
2. **为节点 hidden 的演化提供关系上下文**：主干中节点通过交叉注意力读取 $\mathbf{R}$ 所在的关系 token 序列；
3. **承载关系间代数结构**：$\mathbf{R}$ 的行向量相对位置可隐式编码对称、层级、互斥等关系模式。

### 5.2 为何关系不放流形

将 $\mathbf{R}$ 放在欧氏空间而非乘积流形 $\mathcal{M}$ 上，经由如下三点权衡得出：

**(1) 关系数据的几何结构较弱。** 节点对应实体，实体间存在清晰的层级（"动物 → 哺乳动物 → 犬"）、周期（时间、季节）与开放属性三类几何偏好，乘积流形的三分量设计正为此服务。关系类型之间的几何结构远没有这么明显：大多数关系（如 `plays_sport`、`speaks_language`、`located_in`）是平行且互相独立的离散标签；少数关系具有弱层级（`father_of ⊂ parent_of`），但这种层级的深度与普遍性远不及实体层级。关系的语义结构主要由文本 $\mathbf{c}_{r_k}$ 承载，几何先验的边际贡献有限。

**(2) 关系的"生成分布"退化为单点。** 若强行让关系参与流匹配，数据端目标 $\mathbf{R}_1$ 只能是某个确定的全局矩阵 $\mathbf{R}^\star$（因为所有子图共享同一组关系），流匹配的生成目标从"复杂多模态数据分布"退化为"单点参数学习"。这样 FM 框架的主要威力（学习分布、支持条件生成）对关系来说无用武之地，反而引入 Riemannian 优化的工程复杂度（每步 $\mathrm{Retract}$、梯度切空间投影、学习率分组等）。

**(3) 欧氏参数化与现有优化生态兼容。** 作为欧氏可学习矩阵，$\mathbf{R}$ 可直接用 AdamW 优化，与主干参数一同更新，不需要特殊的黎曼优化器。初始化、梯度裁剪、权重衰减等标准技巧均直接适用。

**结论**：关系适合作为**欧氏可学习参数**，而非流形上的随机变量。本文据此设计。

### 5.3 关系嵌入矩阵的定义

**定义 5.1（关系全局嵌入矩阵）。** 设 $d_r \in \mathbb{Z}_{>0}$ 为关系 hidden 维度（与节点 hidden 维度 $d_v$ 独立的超参数，后续章节详细说明）。关系全局嵌入矩阵为

$$\mathbf{R} \in \mathbb{R}^{K \times d_r}$$

其第 $k$ 行 $\mathbf{r}_k \in \mathbb{R}^{d_r}$ 是关系 $r_k$ 的嵌入向量。

**可训练性**：$\mathbf{R}$ 作为模型参数，通过梯度反向传播更新。其梯度来自所有使用关系嵌入的下游组件——主要为边初始嵌入（定义见 Part III）与边预测头（第 17 章）。

**全局性**：$\mathbf{R}$ 在所有子图、所有训练样本、所有时间 $t$ 中**完全共享**。这与节点坐标 $\mathbf{X}_t$（子图级、时间相关）形成对比：

| 对象 | 作用域 | 时间相关 | 是否训练参数 |
|------|--------|---------|------------|
| $\mathbf{X}_t$ | 子图级 | 是 | 否（被生成的随机变量） |
| $\mathbf{E}_t$ | 子图级 | 是 | 否（被生成的随机变量） |
| $\mathbf{R}$ | 全局 | 否 | 是（可学习参数） |

### 5.4 关系嵌入与关系文本的互补关系

关系有两种静态表示：

- **语义表示**：文本嵌入 $\mathbf{c}_{r_k} \in \mathbb{R}^{d_c}$（定义 3.5，由冻结文本编码器 $\phi_\mathrm{text}$ 给出）；
- **结构表示**：可学习嵌入 $\mathbf{r}_k \in \mathbb{R}^{d_r}$（本章定义）。

两者**维度、来源、可训练性均不同**，承担互补的功能：

| 方面 | $\mathbf{c}_{r_k}$（文本） | $\mathbf{r}_k$（嵌入） |
|------|------------------------|---------------------|
| 维度 | $d_c$（由编码器决定，典型 768） | $d_r$（模型超参） |
| 来源 | 预训练文本编码器 | 可学习参数 |
| 训练 | 冻结 | 随模型更新 |
| 语义 | 通用自然语言语义 | 从边的统计模式中涌现的代数结构 |
| 角色 | 外部锚点（支持归纳） | 模型内部表示（支持精细判别） |

**互补性的具体体现**：

- 对**新出现的关系**（训练中未见），只能依赖 $\mathbf{c}_{r_k}$（文本仍可编码）；$\mathbf{r}_k$ 需通过某种策略从 $\mathbf{c}_{r_k}$ 推出（见 §5.6）。
- 对**文本含糊但行为独特的关系**（如 `part_of` vs `located_in`，文本近似但代数行为不同），$\mathbf{r}_k$ 可通过训练学到比 $\mathbf{c}_{r_k}$ 更精细的区分。
- 两者**共同输入**到模型（节点初始嵌入、边初始嵌入、主干中的关系 token 均同时 concat $\mathbf{c}_{r_k}$ 和 $\mathbf{r}_k$），模型可自主决定在不同下游场景下侧重哪一侧。

### 5.5 关系嵌入的用途汇总

$\mathbf{R}$ 在模型中的使用位置（详细定义见后续章节，此处预览）：

**(用途 1) 关系 token 的初始化输入**：主干中有 $K$ 个关系 token，其初始 hidden 由 $\mathbf{r}_k$ 与 $\mathbf{c}_{r_k}$ 共同 concat 经 MLP 得到（Part III）。

**(用途 2) 边初始嵌入的关系信息源**：位置 $(i, j)$ 的边初始嵌入 $\mathbf{h}_{ij}^{E,(0)}$ 从当前 $\mathbf{E}_{t,ij}$ 激活的关系类型聚合 $\mathbf{R}$ 的对应行向量（即 $\mathbf{E}_{t,ij}\mathbf{R} \in \mathbb{R}^{d_r}$，详见 Part III）。

**(用途 3) 边预测头的类型打分**：第 $k$ 类关系的 logit 通过边潜嵌入 $\mathbf{h}_{ij}^{E,(L)}$ 与 $\mathbf{r}_k$ 的双线性交互给出：

$$\hat{\ell}^{(k)}_{ij} = (\mathbf{h}_{ij}^{E,(L)})^\top \mathbf{W}_\mathrm{type} \mathbf{r}_k + b_k$$

其中 $\mathbf{W}_\mathrm{type}$ 为可学习打分矩阵（第 17 章）。这一形式让模型**自然支持归纳**：添加新关系只需给出其 $\mathbf{r}_k$，无需修改打分头结构。

### 5.6 初始化策略（预告）

$\mathbf{R}$ 的初始化对预训练收敛速度与最终性能有显著影响。本文采用**基于关系文本的语义感知初始化**：在预训练启动阶段，将 $\mathbf{r}_k$ 初始化为 $\mathbf{c}_{r_k}$ 的某个投影，使语义相近的关系在嵌入空间初始位置就接近。具体算法（含启动阶段的动态冻结策略）详见第 21 章。此处仅指出设计原则：

- **不推荐**随机初始化：关系数 $K$ 相对较小（$10^2$–$10^3$ 量级），从随机初始化收敛慢，且容易在训练早期被边预测损失拖向次优局部解；
- **推荐**文本驱动初始化：$\mathbf{r}_k \leftarrow \mathbf{W}_\mathrm{init}\mathbf{c}_{r_k}$，其中 $\mathbf{W}_\mathrm{init} \in \mathbb{R}^{d_r \times d_c}$ 可选为 (i) 冻结的 PCA 投影，(ii) 启动阶段训练的轻量 MLP，或 (iii) 简单随机矩阵（作为 baseline）。

### 5.7 本章小结

本章确立了关系在本文架构中的角色：

- **是**：全局可学习的欧氏嵌入矩阵 $\mathbf{R} \in \mathbb{R}^{K \times d_r}$，与节点坐标 $\mathbf{X}_t$、边张量 $\mathbf{E}_t$ 并列作为模型的三类主要对象。
- **不是**：随机变量、流匹配的生成目标、流形上的点。
- **与文本的关系**：与文本嵌入 $\mathbf{C}_\mathcal{R}$ 互补，共同作为关系的表示，一个承担通用语义、一个承担代数结构。
- **在模型中的使用**：作为关系 token 的初始化源、边初始嵌入的关系信息源、边类型预测头的打分条件。

至此 Part I 的所有静态表示已全部定义完毕：

| 对象 | 空间 | 定义位置 | 维度 |
|------|------|---------|------|
| 节点坐标 $\mathbf{X}$ | $\mathcal{M}^N$ | 定义 3.3 | $N \times D$ |
| 边类型张量 $\mathbf{E}$ | $\{0,1\}^{N \times N \times K}$ | 定义 3.4 | $N \times N \times K$ |
| 节点文本 $\mathbf{C}_\mathcal{V}$ | $\mathbb{R}^{N \times d_c}$ | 定义 3.6 | $N \times d_c$ |
| 关系文本 $\mathbf{C}_\mathcal{R}$ | $\mathbb{R}^{K \times d_c}$ | 定义 3.6 | $K \times d_c$ |
| 节点掩码 $\mathbf{m}$ | $\{0,1\}^N$ | 定义 3.8 | $N$ |
| 关系嵌入 $\mathbf{R}$ | $\mathbb{R}^{K \times d_r}$ | 定义 5.1 | $K \times d_r$ |

Part II 随后引入时间变量 $t \in [0, 1]$，让 $\mathbf{X} \to \mathbf{X}_t$ 做连续流匹配（第 7 章）、$\mathbf{E} \to \mathbf{E}_t$ 做离散流匹配（第 8 章）；$\mathbf{R}$ 作为条件参与而不随 $t$ 演化。

---

*Part I 结束。Part II 进入方法的理论核心——流匹配框架。*
# Part II. 时间动力学：Flow Matching 框架

本部分引入时间变量 $t \in [0,1]$，定义 RiemannFM 的核心生成动力学。第 6 章建立联合连续-离散流匹配的统一框架；第 7 章详细定义节点的黎曼流匹配；第 8 章定义边的离散流匹配（掩码调度）；第 9 章引入模态遮蔽策略作为预训练的数据增强机制。

本部分关注 **"训练信号从何而来"**——推出训练中每个 batch 的样本构造与监督目标的闭式解。具体损失函数形式、优化算法与推理过程详见 Part IV 与 Part V。

---

## 第 6 章 联合连续-离散流匹配框架

### 6.1 生成式建模目标

给定知识图谱 $\mathcal{K}$ 上采样得到的子图分布 $p_\mathrm{data}$，本文目标是建模联合分布

$$p_\mathrm{data}(\mathbf{X}, \mathbf{E} \mid \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R})$$

即"给定节点和关系的文本条件，生成节点坐标与边类型张量的联合分布"。关系全局嵌入 $\mathbf{R}$ 作为条件参数（第 5 章），不参与生成目标。

**为什么建模联合分布**：节点坐标 $\mathbf{X}$ 与边类型张量 $\mathbf{E}$ 在数据中高度耦合——相邻节点的坐标趋近、常见关系类型与节点语义对应。若仅分别建模边际 $p(\mathbf{X})$ 和 $p(\mathbf{E})$ 而独立采样合成，将破坏真实子图的结构一致性。

**流匹配框架的选择**：Flow Matching（Lipman et al. 2023, Chen & Lipman 2024）提供了一种不依赖变分下界、无需模拟扩散过程的生成式建模范式，支持连续变量（$\mathbf{X}$）与离散变量（$\mathbf{E}$）的统一处理。

### 6.2 时间变量与调度统一

引入时间变量 $t \in [0, 1]$ 作为"生成进度"的连续刻度：
- $t = 0$：先验端。节点坐标采样自先验分布 $p_\mathrm{prior}^\mathcal{M}$；边类型全部处于掩码态 MASK。
- $t = 1$：数据端。节点坐标与边类型皆为真实数据。
- $t \in (0, 1)$：中间状态。节点在流形上沿测地线从先验向数据插值；边按某种调度 $\alpha(t)$ 逐步从掩码解码为真实值。

**关键原则：节点和边共享同一个 $t$**。单次训练 step 从 $t$ 分布 $p_t$ 采样一个 $t$，同时用于构造 $\mathbf{X}_t$（连续插值）与 $\mathbf{E}_t$（掩码采样）。这保证模型主干在每个 $t$ 看到的节点和边处于**同等的生成进度**。

$t$ 分布 $p_t$ 的设计见 §7.5 与 §8.3。

### 6.3 三类变量的动力学角色

为清晰起见，汇总三类变量的动力学角色：

| 变量 | 空间 | 动力学 | 目标 $t=1$ | 先验 $t=0$ |
|------|------|-------|-----------|-----------|
| $\mathbf{X}_t$ 节点坐标 | $\mathcal{M}^N$ | 连续 FM（黎曼） | 真实数据 $\mathbf{X}_1$ | 采样自 $p_\mathrm{prior}^\mathcal{M}$ |
| $\mathbf{E}_t$ 边类型 | $\{0,1,\mathrm{MASK}\}^{N\times N\times K}$ | 离散 FM（掩码式） | 真实边 $\mathbf{E}_1$ | 全掩码 $\boldsymbol{\mu}_0 = \mathbf{1}$ |
| $\mathbf{R}$ 关系嵌入 | $\mathbb{R}^{K \times d_r}$ | 无动力学（参数） | — | — |

节点通过黎曼测地线连续演化；边通过独立位置的伯努利掩码过程离散演化；关系始终是固定的参数（但其值通过训练梯度更新）。

### 6.4 Conditional Flow Matching 基本定理

**定理 6.1（条件流匹配，欧氏情形，Lipman et al. 2023）。** 设 $p_0, p_1$ 为 $\mathbb{R}^d$ 上的概率分布。设 $\mathbf{x}_t \sim p_{t|0,1}(\cdot \mid \mathbf{x}_0, \mathbf{x}_1)$ 为连接 $\mathbf{x}_0, \mathbf{x}_1$ 的条件路径，$\mathbf{v}^\star(\mathbf{x}_t, t \mid \mathbf{x}_0, \mathbf{x}_1) = \tfrac{d \mathbf{x}_t}{dt}$ 为条件路径的速度。令

$$\mathcal{L}_\mathrm{CFM}(\theta) = \mathbb{E}_{t \sim p_t,\, \mathbf{x}_0 \sim p_0,\, \mathbf{x}_1 \sim p_1,\, \mathbf{x}_t \sim p_{t|0,1}} \big\|\mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{v}^\star(\mathbf{x}_t, t \mid \mathbf{x}_0, \mathbf{x}_1)\big\|^2$$

则其梯度与学习**边际向量场**（将 $p_0$ 推到 $p_1$ 的向量场）的均方损失的梯度相同：

$$\nabla_\theta \mathcal{L}_\mathrm{CFM}(\theta) = \nabla_\theta \mathbb{E}_{t,\, \mathbf{x}_t \sim p_t} \big\|\mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{v}^\star_\mathrm{marg}(\mathbf{x}_t, t)\big\|^2$$

**定理的意义**：用**可采样的条件路径和条件速度**作训练目标，就能学到不可采样的边际向量场。这让 FM 训练变得可行——每 step 采样 $(\mathbf{x}_0, \mathbf{x}_1, t)$，构造 $\mathbf{x}_t$，监督 $\mathbf{v}_\theta(\mathbf{x}_t, t) \approx \mathbf{v}^\star(\mathbf{x}_t, t \mid \mathbf{x}_0, \mathbf{x}_1)$。

### 6.5 黎曼推广：乘积流形上的 CFM

**定理 6.2（黎曼 CFM，Chen & Lipman 2024）。** 设 $\mathcal{M}$ 为黎曼流形，$p_0, p_1$ 为 $\mathcal{M}$ 上的概率分布。取条件路径为测地线：

$$\mathbf{x}_t = \gamma_{\mathbf{x}_0 \to \mathbf{x}_1}(t) = \exp_{\mathbf{x}_0}\!\big(t \cdot \log_{\mathbf{x}_0}(\mathbf{x}_1)\big)$$

相应的条件速度为

$$\mathbf{v}^\star(t \mid \mathbf{x}_0, \mathbf{x}_1) = \frac{\log_{\mathbf{x}_t}(\mathbf{x}_1)}{1 - t}, \quad t \in [0, 1)$$

令

$$\mathcal{L}_\mathrm{RCFM}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\Big[\big\|\mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{v}^\star(t \mid \mathbf{x}_0, \mathbf{x}_1)\big\|_{T_{\mathbf{x}_t}\mathcal{M}}^2\Big]$$

其中范数取切空间黎曼范数（定义 4.6）。则 $\mathcal{L}_\mathrm{RCFM}$ 的极小化得到将 $p_0$ 推到 $p_1$ 的边际向量场。

**乘积流形的分解**：在 $\mathcal{M} = \mathbb{H}^{d_h}_{\kappa_h} \times \mathbb{S}^{d_s}_{\kappa_s} \times \mathbb{R}^{d_e}$ 上，测地线 $\gamma(t)$ 由各分量独立的测地线组成：

$$\mathbf{x}_t = \big(\gamma_\mathbb{H}(t),\, \gamma_\mathbb{S}(t),\, \gamma_\mathbb{R}(t)\big)$$

条件速度相应地分解：

$$\mathbf{v}^\star(t) = \Big(\tfrac{\log_{\mathbf{x}_t^\mathbb{H}}(\mathbf{x}_1^\mathbb{H})}{1-t},\, \tfrac{\log_{\mathbf{x}_t^\mathbb{S}}(\mathbf{x}_1^\mathbb{S})}{1-t},\, \mathbf{x}_1^\mathbb{R} - \mathbf{x}_0^\mathbb{R}\Big)$$

（欧氏分量的 $\log$ 化简为差向量）。

**意义**：乘积流形的 FM **自然分解**——三个分量独立处理，无需额外耦合项。这是乘积流形几何作为方法基础的关键优势。

### 6.6 离散变量的流匹配

离散变量的流匹配不能直接用测地线和速度场——离散状态空间无连续插值。本文采用**掩码式离散流匹配**（Masked Discrete Flow Matching，参考 Sahoo et al. 2024 的 SEDD、Shi et al. 2024 的 Masked Diffusion）：

**核心思想**：扩展状态空间加入 MASK 态，前向过程从数据端开始逐步"掩蔽"部分位置（对应真实数据端到先验端），反向过程从全掩码开始逐步"解码"（对应先验端到数据端）。

对每个位置 $(i, j) \in [N]^2$，独立引入**掩码指示** $\mu_{ij}(t) \in \{0, 1\}$：
- $\mu_{ij}(t) = 1$：位置被掩码（未决定值）；
- $\mu_{ij}(t) = 0$：位置已解码（值为真实的 $\mathbf{E}_{ij}(1)$）。

前向过程：$\mu_{ij}(t) \sim \mathrm{Bernoulli}(1 - \alpha(t))$，$\alpha: [0,1] \to [0,1]$ 单调递增，$\alpha(0)=0, \alpha(1)=1$。

**训练目标**：模型在掩码位置预测**干净端边际分布** $p_{1|t}$（"看了当前部分观察，猜猜这个位置的真实值是什么"）。具体形式见第 8 章。

**与连续 FM 的统一**：掩码式离散 FM 与连续 FM 共享"生成从先验到数据的学习"这一目标，且通过共享 $t$ 实现同步演化。训练 step 的构造为：

```
采样 t ~ p_t
采样 X_0 ~ p_prior（节点）
采样 μ_t ~ Bernoulli(1 - α(t))（边掩码）
构造 X_t = exp_{X_0}(t · log_{X_0}(X_1))
构造 E_t：被掩码位置为 MASK，否则为 E_1
主干前向，同时计算 FM 损失（节点）和掩码预测损失（边）
```

### 6.7 "联合"的精确含义

本方法命名为"联合连续-离散流匹配"，其"联合"体现在三个层次：

**(1) 共享时间 $t$**：节点连续 FM 与边离散 FM 使用同一 $t$。主干在每个 $t$ 前向时，节点处于某个噪声水平（$t$ 决定），边处于对应的掩码覆盖率（$\alpha(t)$ 决定）。两者在"生成进度"上严格对齐。

**(2) 共享主干**：节点切向量预测与边 $p_{1|t}$ 预测都由同一 RieFormer 主干 $F_\theta$ 计算。主干的每一层都同时更新节点 hidden、边 hidden、关系 hidden。节点的演化信息通过主干传给边，边的状态通过 attention bias 影响节点。

**(3) 耦合的训练信号**：节点 FM 损失 $\mathcal{L}_X$ 与边离散 FM 损失 $\mathcal{L}_E$ 同时反传到主干，形成两路互补的梯度。节点 FM 要求主干预测正确的几何流动方向（与边结构无关），边 FM 要求主干利用节点几何预测正确的边类型。联合训练强制主干同时学好两种任务。

**非联合的备选**：若分别训练两个独立模型（一个做节点 FM，一个做边预测），则失去 (2)(3)，节点和边的生成独立进行，联合分布结构被破坏。本方法的联合设计正是为避免这种破坏。

### 6.8 本章小结

本章给出了本方法流匹配框架的五个关键决定：

1. 建模目标是 $p(\mathbf{X}, \mathbf{E} \mid \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R})$ 的联合条件分布；
2. 时间变量 $t \in [0, 1]$ 为节点连续流与边离散流的统一调度；
3. 节点 FM 采用黎曼 CFM（定理 6.2），测地线插值与对数映射给出闭式目标场；
4. 边 FM 采用掩码式离散 FM，通过独立 Bernoulli 掩码与 $p_{1|t}$ 预测；
5. 联合体现为共享 $t$、共享主干、耦合梯度三层次。

关系嵌入 $\mathbf{R}$ 不参与动力学，作为条件参数在主干中通过 token 形式与节点、边交互。

---

## 第 7 章 节点的黎曼流匹配

### 7.1 先验分布

**定义 7.1（乘积流形先验分布）。** 节点先验分布 $p_\mathrm{prior}^\mathcal{M}$ 由三个分量独立采样组成：

- **双曲分量**（wrapped normal on $\mathbb{H}$）：
  1. 从锚点切空间 $T_{\mathbf{x}_\varnothing^\mathbb{H}}\mathbb{H}$（同构于 $\mathbb{R}^{d_h}$）采样 $\tilde{\mathbf{v}} \sim \mathcal{N}(\mathbf{0}, \sigma_h^2 \mathbf{I}_{d_h})$；
  2. 嵌入到 $\mathbb{R}^{d_h+1}$（插入 $\tilde{v}_0 = 0$）：$\mathbf{v} = (0, \tilde{v}_1, \ldots, \tilde{v}_{d_h})$；
  3. 沿测地线推送：$\mathbf{x}_0^\mathbb{H} = \exp_{\mathbf{x}_\varnothing^\mathbb{H}}(\mathbf{v})$。

- **球面分量**（uniform on $\mathbb{S}$）：
  1. 采样 $\tilde{\mathbf{g}} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_s+1})$；
  2. 归一化：$\mathbf{x}_0^\mathbb{S} = \tilde{\mathbf{g}} / (\sqrt{\kappa_s} \|\tilde{\mathbf{g}}\|_2)$。

- **欧氏分量**（标准高斯）：$\mathbf{x}_0^\mathbb{R} \sim \mathcal{N}(\mathbf{0}, \sigma_e^2 \mathbf{I}_{d_e})$。

超参数 $\sigma_h, \sigma_e > 0$ 控制先验分布的集中度（典型 $\sigma_h = \sigma_e = 1$）。球面上使用均匀分布即可（无方差参数）。

**性质**：各分量先验对所在流形的等距群不变（双曲 wrapped normal 关于锚点对称；球面均匀分布完全旋转不变；欧氏高斯关于原点对称）。这种对称性简化分析并避免引入方向偏好。

### 7.2 测地线插值

给定节点 $i$ 的先验样本 $\mathbf{x}_{i,0}$ 与数据样本 $\mathbf{x}_{i,1}$，在时刻 $t$ 的插值坐标为：

$$\mathbf{x}_{i,t} = \exp_{\mathbf{x}_{i,0}}\!\big(t \cdot \log_{\mathbf{x}_{i,0}}(\mathbf{x}_{i,1})\big) \in \mathcal{M}$$

该公式在各流形分量独立计算（乘积流形的分解性质，§6.5），实现上直接调用第 4 章的 $\exp$ 与 $\log$ 原语。

**$t$ 的意义**：$t$ 为测地线上的比例参数。$t=0$ 在先验端，$t=1$ 在数据端，$t \in (0, 1)$ 为测地线段上的线性位置。

### 7.3 目标向量场

测地线在时刻 $t$ 的瞬时速度（即目标向量场）由定理 6.2 给出：

$$\mathbf{v}_i^\star(t) = \frac{\log_{\mathbf{x}_{i,t}}(\mathbf{x}_{i,1})}{1 - t} \in T_{\mathbf{x}_{i,t}}\mathcal{M}$$

**等价形式**：目标向量场也可写成初始速度的平行移动 $\mathrm{PT}_{\mathbf{x}_{i,0} \to \mathbf{x}_{i,t}}\!\big(\log_{\mathbf{x}_{i,0}}(\mathbf{x}_{i,1})\big)$。但上式仅依赖 $\log$ 原语，实现更简洁，推荐使用。

**欧氏分量的化简**：对欧氏分量 $\mathbf{x}_t^\mathbb{R} = (1-t)\mathbf{x}_0^\mathbb{R} + t\mathbf{x}_1^\mathbb{R}$，$\log_{\mathbf{x}_t^\mathbb{R}}(\mathbf{x}_1^\mathbb{R}) = \mathbf{x}_1^\mathbb{R} - \mathbf{x}_t^\mathbb{R} = (1-t)(\mathbf{x}_1^\mathbb{R} - \mathbf{x}_0^\mathbb{R})$，故 $\mathbf{v}^{\star,\mathbb{R}}(t) = \mathbf{x}_1^\mathbb{R} - \mathbf{x}_0^\mathbb{R}$（常向量）。

**数值稳定性**：$t \to 1$ 时分母 $1 - t \to 0$。训练实现加入数值保护：

$$\mathbf{v}_i^\star(t) = \frac{\log_{\mathbf{x}_{i,t}}(\mathbf{x}_{i,1})}{\max(1 - t,\, \epsilon)}, \quad \epsilon = 10^{-4}$$

此外 $t$ 从连续分布采样时 $t = 1$ 的概率为 0，实际训练中 $1 - t$ 极少接近 $\epsilon$。

### 7.4 FM 损失

**定义 7.2（节点连续流匹配损失）。** 记 $\hat{\mathbf{v}}_i^V(t)$ 为模型对第 $i$ 节点、时间 $t$ 的切向量预测（已经切空间投影 $\mathrm{Proj}_{\mathbf{x}_{i,t}}$），$m_i^\mathrm{coord}$ 为该节点的坐标遮蔽指示（1 = 参与 FM，0 = 遮蔽，见第 9 章；预训练默认全 1）。节点 FM 损失：

$$\mathcal{L}_X = \mathbb{E}_{t \sim p_t,\, \mathbf{X}_0 \sim p_\mathrm{prior},\, \mathbf{X}_1 \sim p_\mathrm{data}} \Bigg[\frac{1}{Z} \sum_{i=1}^{N} m_i \cdot m_i^\mathrm{coord} \cdot \big\|\hat{\mathbf{v}}_i^V(t) - \mathbf{v}_i^\star(t)\big\|_{T_{\mathbf{x}_{i,t}}\mathcal{M}}^2\Bigg]$$

其中 $Z = \sum_i m_i m_i^\mathrm{coord}$ 为有效节点数（归一化避免 batch 间方差）。

**范数的 ambient 实现**：切向量 $\hat{\mathbf{v}}_i^V(t)$ 与 $\mathbf{v}_i^\star(t)$ 均严格位于 $T_{\mathbf{x}_{i,t}}\mathcal{M}$ 内（前者通过 $\mathrm{Proj}$、后者通过 $\log$ 的性质）。此时黎曼范数（定义 4.6）等价于各分量在 ambient 空间的 $\ell_2$ 范数之和：

$$\big\|\hat{\mathbf{v}}_i - \mathbf{v}_i^\star\big\|^2 = \big\|\hat{\mathbf{v}}_i^\mathbb{H} - \mathbf{v}_i^{\star,\mathbb{H}}\big\|_2^2 + \big\|\hat{\mathbf{v}}_i^\mathbb{S} - \mathbf{v}_i^{\star,\mathbb{S}}\big\|_2^2 + \big\|\hat{\mathbf{v}}_i^\mathbb{R} - \mathbf{v}_i^{\star,\mathbb{R}}\big\|_2^2$$

（双曲分量的 Lorentz 范数在切空间内等于 ambient $\ell_2$ 范数，因切向量满足 $\langle \mathbf{v}^\mathbb{H}, \mathbf{x}^\mathbb{H} \rangle_\mathrm{L} = 0$）。

**分量加权（可选）**：不同流形分量的距离尺度可能差异显著。实现中可分别计算三个分量损失并加权：

$$\mathcal{L}_X = \lambda^\mathbb{H} \mathcal{L}_X^\mathbb{H} + \lambda^\mathbb{S} \mathcal{L}_X^\mathbb{S} + \lambda^\mathbb{R} \mathcal{L}_X^\mathbb{R}$$

默认 $\lambda^\mathbb{H} = \lambda^\mathbb{S} = \lambda^\mathbb{R} = 1$，可按维度倒数（$1/d_h$ 等）或实证调整。

### 7.5 时间分布 $p_t$

**定义 7.3（时间分布选项）。** 本文考虑三种 $p_t$：

- **Uniform**：$t \sim \mathrm{Uniform}(0, 1)$。简单、无偏。
- **Logit-Normal**：$t = \sigma(u), u \sim \mathcal{N}(0, 1)$。密度在 $t=0.5$ 集中，端点稀疏。适合"难在中段"的任务。
- **Beta**：$t \sim \mathrm{Beta}(\alpha, \beta)$，$\alpha, \beta > 0$。$\alpha=\beta=1$ 退化为 Uniform；$\alpha > \beta$ 偏向 1；$\alpha < \beta$ 偏向 0。

**默认选择**：预训练标准模式使用 **Logit-Normal**（Stable Diffusion 3 / Flux 的默认选择，在多种生成任务上实证稳健）。模态遮蔽场景（§9）使用不同的 $p_t$（详见第 9 章）。

### 7.6 先验采样与数据采样的耦合

**独立耦合**（默认）：每 step 独立采样 $(\mathbf{X}_0, \mathbf{X}_1)$，即 $\mathbf{X}_0 \sim p_\mathrm{prior}^{\otimes N}$ 与 $\mathbf{X}_1 \sim p_\mathrm{data}$ 无配对。CFM 定理在独立耦合下成立。

**OT 耦合**（可选增强）：在 batch 内通过某种 Optimal Transport 算法把 $\mathbf{X}_0$ 与 $\mathbf{X}_1$ 的每个样本配对，使配对距离最小化（Tong et al. 2024 的 OT-CFM）。OT 耦合可加快收敛、提高生成样本质量，但在乘积流形上实现复杂（需要流形 OT 算法）。

**本文立场**：默认使用独立耦合。OT 耦合作为未来扩展方向。

### 7.7 节点 FM 的训练信号生成算法

```
# 每次训练 step（节点部分）
for batch G in dataloader:
    # 1. 采样时间
    t ~ p_t (取决于当前模态遮蔽场景)
    
    # 2. 数据端
    X_1 = G.node_coords  ∈ M^N
    
    # 3. 先验端
    X_0 ~ p_prior^⊗N
    
    # 4. 测地线插值
    X_t = exp_map(X_0, t · log_map(X_0, X_1))
    
    # 5. 目标向量场
    v_star = log_map(X_t, X_1) / (1 - t + eps)
    
    # 6. 主干前向（含关系与边，详见后续章节）
    H_V, H_R, H_E = F_theta(X_t, E_t, R, C_V, C_R, t, ...)
    
    # 7. 节点切向量预测
    u_hat = MLP_V(H_V)  ∈ ambient space
    v_hat = Proj_{X_t}(u_hat)  ∈ T_{X_t}M
    
    # 8. FM 损失
    L_X = sum_i m_i · m_i^coord · ||v_hat_i - v_star_i||^2 / Z
```

注意**目标向量场的数据端 $\mathbf{X}_1$ 不参与梯度反传**（它是真实数据，不是模型参数）。损失梯度完全用于更新网络参数 $\theta$。

---

## 第 8 章 边的离散流匹配

### 8.1 状态空间扩展

边类型张量 $\mathbf{E}_t$ 的状态空间扩展为 $\{0, 1, \mathrm{MASK}\}^{N \times N \times K}$。每个位置 $(i, j)$ 的状态由两个张量联合描述：

- $\boldsymbol{\mu}_t \in \{0, 1\}^{N \times N}$：**掩码指示张量**，$\mu_{t,ij} = 1$ 表示位置被掩码（值未知），$\mu_{t,ij} = 0$ 表示已解码。
- $\mathbf{E}_t \in \{0, 1\}^{N \times N \times K}$：**边值张量**，仅在 $\mu_{t,ij} = 0$ 时有意义。被掩码位置的 $\mathbf{E}_{t,ij}$ 约定为占位值 $\mathbf{0}_K$（模型通过 $\mu$ 指示区分"真实无边"与"被掩码"）。例如当 $\mathbf{E}_{1,ij} = \mathbf{0}_K$（真实无边）时，未掩码态 $(\mathbf{E}_{t,ij}, \mu_{t,ij}) = (\mathbf{0}_K, 0)$，掩码态 $(\mathbf{0}_K, 1)$——二者边值张量相同，仅 $\mu$ bit 区分。

**边界条件**：
- $t = 0$：$\boldsymbol{\mu}_0 = \mathbf{1}_{N \times N}$（全掩码），$\mathbf{E}_0 = \mathbf{0}$（全占位）。
- $t = 1$：$\boldsymbol{\mu}_1 = \mathbf{0}_{N \times N}$（全解码），$\mathbf{E}_1$ 为真实数据。

### 8.2 前向调度函数

**定义 8.1（调度函数）。** 设 $\alpha: [0, 1] \to [0, 1]$ 为单调递增函数，满足 $\alpha(0) = 0, \alpha(1) = 1$。$\alpha(t)$ 表示时刻 $t$ 的**期望解码覆盖率**（即 $\Pr[\mu_{t,ij} = 0]$）。

**调度函数选项**：

- **线性**：$\alpha(t) = t$
- **余弦**：$\alpha(t) = 1 - \cos(\pi t / 2)$。小 $t$ 时解码慢（保守），大 $t$ 时快。
- **凹**：$\alpha(t) = 1 - (1 - t)^2$。小 $t$ 时解码快（积极）。

**默认选择**：**余弦调度**。其理由与直觉一致："生成初期先定骨架（少量解码），后期精修细节（快速解码）"。

### 8.3 前向过程的构造

给定时间 $t$，边状态的前向采样：

$$\mu_{t,ij} \sim \mathrm{Bernoulli}(1 - \alpha(t)), \quad \text{i.i.d. over } (i, j)$$

$$\mathbf{E}_{t,ij} = \begin{cases} \mathbf{E}_{1,ij} & \text{若 } \mu_{t,ij} = 0 \\ \mathbf{0}_K & \text{若 } \mu_{t,ij} = 1 \end{cases}$$

**i.i.d. 假设的合理性**：边位置之间的掩码采样**不联合采样、不考虑条件依赖**。这是训练期的简化假设——推理期实际解码是条件依赖的（下一位置的解码依赖已解码位置），存在 train-test mismatch。但文献实证（MaskGIT、Masked Diffusion）表明此 mismatch 在实践中可接受，且 i.i.d. 采样支持高度并行的训练。

### 8.4 $p_{1|t}$ 训练目标

**核心思想**：模型在当前噪声状态 $(\mathbf{X}_t, \mathbf{E}_t, \boldsymbol{\mu}_t)$ 下，预测**被掩码位置的干净端分布**。即对每个被掩码位置 $(i, j)$（$\mu_{t,ij} = 1$），模型输出 $\mathbf{E}_{1,ij}$ 的条件概率。

**$K+1$ 维分解**：$\mathbf{E}_{1,ij} \in \{0, 1\}^K$ 通过两层预测因子化：

$$\Pr[\mathbf{E}_{1,ij} = \mathbf{b}] = \Pr[e_{ij} = \mathbb{1}[\mathbf{b} \neq \mathbf{0}]] \cdot \Pr[\mathbf{E}_{1,ij} = \mathbf{b} \mid e_{ij} = \mathbb{1}[\mathbf{b} \neq \mathbf{0}]]$$

其中 $e_{ij} \in \{0, 1\}$ 为存在性指示（$e_{ij} = 1 \iff \|\mathbf{E}_{1,ij}\|_1 \geq 1$）。

预测头输出 $K+1$ 维 logits：
- $\hat{\ell}_{ij}^\mathrm{ex}(t)$：存在性 logit，预测 $e_{ij}$；
- $\hat{\ell}_{ij}^{(k)}(t)$，$k = 1, \ldots, K$：类型 logits，预测 $\mathbf{E}_{1,ij}^{(k)}$。

### 8.5 离散 FM 损失

**定义 8.2（边存在性损失）。** 仅在**被掩码位置**（$\mu_{t,ij} = 1$）计算：

$$\mathcal{L}_\mathrm{ex} = \mathbb{E}_{t, \mathbf{X}_t, \mathbf{E}_t, \boldsymbol{\mu}_t}\Bigg[\frac{1}{Z_\mathrm{ex}} \sum_{(i,j)} m_i m_j \cdot \mu_{t,ij} \cdot \mathrm{BCE}\big(\hat{\ell}_{ij}^\mathrm{ex}(t),\, e_{1,ij}\big)\Bigg]$$

其中 $Z_\mathrm{ex} = \sum_{(i,j)} m_i m_j \mu_{t,ij}$ 为有效位置数。

**定义 8.3（边类型损失）。** 门控型：仅在**被掩码且真实有边**位置计算：

$$\mathcal{L}_\mathrm{ty} = \mathbb{E}_{t, \ldots}\Bigg[\frac{1}{Z_\mathrm{ty}} \sum_{(i,j)} m_i m_j \cdot \mu_{t,ij} \cdot e_{1,ij} \cdot \sum_{k=1}^{K} \mathrm{BCE}\big(\hat{\ell}_{ij}^{(k)}(t),\, \mathbf{E}_{1,ij}^{(k)}\big)\Bigg]$$

其中 $Z_\mathrm{ty} = \sum_{(i,j)} m_i m_j \mu_{t,ij} e_{1,ij}$。

**为什么"仅在掩码位置"**：已解码位置是模型的**条件输入**（主干看到它们的真实值），非待预测目标。若在已解码位置也计算损失，等价于要求模型"忽略输入重建"，破坏掩码解码的语义。

**为什么类型损失**门控：绝大多数位置 $\mathbf{E}_{1,ij} = \mathbf{0}_K$（稀疏图），若所有位置都算 $K$ 个关系的 BCE，负样本会完全淹没正样本的梯度。门控 $e_{1,ij}$ 使类型损失只在真实存在边的位置计算，保证类型学习的信号质量。

### 8.6 时间分布 $p_t$ 在边 FM 中的作用

边 FM 使用的 $t$ 分布与节点 FM **共享**（§6.2 的原则）。同一个 $t$ 既用于构造 $\mathbf{X}_t$ 又用于构造 $\boldsymbol{\mu}_t$。

**$t$ 分布对下游适配的影响**：

- $t$ 接近 1（小掩码覆盖率）：对应 KGC 场景——已观察大部分图，预测少数缺失边。
- $t$ 接近 0（大掩码覆盖率）：对应生成场景——从稀疏上下文生成完整子图。

本文默认使用 **Logit-Normal**，两端平衡。若主要关注 KGC，可尝试偏向 1 的 Beta 分布（如 $\mathrm{Beta}(5, 2)$）；若主要关注生成，可尝试偏向 0（如 $\mathrm{Beta}(2, 5)$）。具体建议在第 34 章消融研究中讨论。

### 8.7 边 FM 训练算法

```
# 每次训练 step（边部分，与节点部分共享 t）
for batch G in dataloader:
    t ~ p_t  # 与节点 FM 共用
    
    # 1. 前向构造 μ_t 与 E_t
    alpha_t = schedule(t)  # e.g. 1 - cos(π t / 2)
    mu_t ~ Bernoulli(1 - alpha_t, shape=(N, N))
    E_t = where(mu_t == 0, G.E_1, 0)  # 未掩码取真值,掩码取 0
    
    # 2. 主干前向（与节点共用）
    _, _, H_E = F_theta(X_t, E_t, mu_t, R, C_V, C_R, t, ...)
    
    # 3. 边预测头
    ell_ex[i,j] = w_ex^T · h_E_{ij}^{(L)}
    ell_type[i,j,k] = h_E_{ij}^{(L) T} · W_type · R_k + b_k
    
    # 4. 损失（仅在掩码位置）
    L_ex = BCE(ell_ex, e_1, mask=m·m·mu_t)
    L_ty = BCE(ell_type, E_1, mask=m·m·mu_t·e_1)
```

### 8.8 训练-推理的微小 mismatch

训练时 $\boldsymbol{\mu}_t$ 独立 Bernoulli 采样，推理时逐步解码依赖已决定位置（详见第 25 章）。这种 mismatch 在掩码式扩散/FM 文献中是已知问题，实证影响小，不作额外处理。

---

## 第 9 章 模态遮蔽：数据增强与归纳泛化

### 9.1 动机

本章引入的**模态遮蔽**（modality masking）是预训练阶段的数据增强策略。其动机有三：

**(1) 防模态偷懒（modality shortcut）**：节点同时拥有几何坐标 $\mathbf{x}_i$ 与文本嵌入 $\mathbf{c}_i$。文本由冻结的强大编码器给出，信息非常丰富——模型可能发现"只看文本就能预测边"，从而**让几何坐标沦为装饰**，流匹配框架失去意义。强制遮蔽文本迫使模型必须从几何学习信号。

**(2) 支持下游异构场景**：
- 归纳 KGC（新实体）：推理时出现训练中未见的节点，只有文本、无学到的坐标；
- 冷启动生成：从文本描述生成子图（坐标未知）；
- 纯结构任务：只关心拓扑，文本无关或不可靠。

预训练时模拟这些"部分信息缺失"场景，下游迁移性能显著提升。

**(3) 隐式对齐**：被要求"遮文本时仍能预测边、遮坐标时仍能预测边"，模型被强制学会**两个模态之间的互相映射**。这比显式 InfoNCE 对齐更强——不是拉嵌入空间接近，而是要求"从 A 能推出 B 的下游信号"。

### 9.2 四种训练模式

**定义 9.1（训练模式与采样概率）。** 每个 batch 的子图在送入模型前，按如下概率分布选择一种模式：

| 模式 | 概率 | 文本 $\mathbf{c}_i$ | 坐标 $\mathbf{x}_i$（数据端）| $t$ 分布 |
|------|------|-------------------|---------------------------|---------|
| 全模态 | $p_\mathrm{full} = 0.70$ | 全保留 | 全使用 | Logit-Normal |
| 文本遮蔽 | $p_\mathrm{tm} = 0.15$ | 随机 $\rho_\mathrm{tm}$ 比例置为 `mask_emb` | 全使用 | $\mathrm{Beta}(5,1)$ |
| 坐标遮蔽 | $p_\mathrm{cm} = 0.15$ | 全保留 | 随机 $\rho_\mathrm{cm}$ 比例不使用 | Logit-Normal |
| 双模遮蔽 | $p_\mathrm{both} = 0$（默认关闭） | $\rho_\mathrm{tm}/2$ 置为 `mask_emb` | $\rho_\mathrm{cm}/2$ 不使用 | Logit-Normal |

默认超参数：$\rho_\mathrm{tm} = 0.30, \rho_\mathrm{cm} = 0.15$。

**关于双模遮蔽**：默认 $p_\mathrm{both} = 0$（关闭）；该行作为消融（§34.3）保留，用于检验"同时遮两个模态"是否带来额外收益，主训练不启用。

**关系不参与遮蔽**：关系作为全局参数（第 5 章），不存在"训练中未见"场景；关系文本 $\mathbf{c}_{r_k}$ 全程可用；关系嵌入 $\mathbf{R}$ 全程参与。模态遮蔽仅对节点施加。

### 9.3 遮蔽指示与 mask_emb

**文本遮蔽指示**：$m_i^\mathrm{text} \in \{0, 1\}^N$，$m_i^\mathrm{text} = 0$ 表示文本被遮蔽。遮蔽时 $\mathbf{c}_i$ 被替换为可学习向量 $\mathbf{c}_\mathrm{mask} \in \mathbb{R}^{d_c}$（**注意不是零向量**——零向量与真实零文本难以区分；可学习 `mask_emb` 让模型能识别"这里的文本是被遮蔽的"）：

$$\tilde{\mathbf{c}}_i = \begin{cases} \mathbf{c}_i & m_i^\mathrm{text} = 1 \\ \mathbf{c}_\mathrm{mask} & m_i^\mathrm{text} = 0 \end{cases}$$

**坐标遮蔽指示**：$m_i^\mathrm{coord} \in \{0, 1\}^N$，$m_i^\mathrm{coord} = 0$ 表示该节点的数据端坐标**不使用**。这意味着：

- 该节点的 $\mathbf{x}_{i,1}^\mathrm{data}$ 不被使用；
- 该节点全程 $\mathbf{x}_{i,t}$ 为**先验样本**（从 $p_\mathrm{prior}$ 独立采样，与 $t$ 无关）；
- 该节点的 FM 损失 $\mathcal{L}_X$ 被屏蔽（见定义 7.2 中的 $m_i^\mathrm{coord}$ 因子）。

### 9.4 两个遮蔽 bit 作为模型输入

遮蔽指示 $m_i^\mathrm{text}, m_i^\mathrm{coord}$ 作为额外的 token 特征输入到节点初始嵌入（详见 Part III 的定义 10.4）。主干通过这两个 bit 知晓"这个节点哪个模态可见"，从而对不同场景调整策略。

### 9.5 各训练模式下的损失激活表

为清晰起见，列出每种模式下各损失项的计算状态：

| 损失项 | 全模态 | 文本遮蔽 | 坐标遮蔽 |
|-------|-------|---------|---------|
| $\mathcal{L}_X$（节点 FM） | 全激活 | 全激活 | **部分激活**（遮蔽节点屏蔽） |
| $\mathcal{L}_\mathrm{ex}$（边存在性） | 全激活 | 全激活 | 全激活 |
| $\mathcal{L}_\mathrm{ty}$（边类型） | 全激活 | 全激活 | 全激活 |

关键点：**边损失始终计算**。即使文本或坐标被遮蔽，主干也必须从剩余信息（另一个模态 + 边上下文 $\mathbf{E}_t$）预测正确的边。这是遮蔽训练信号反传的主要路径。

### 9.6 文本遮蔽的 $t$ 分布为何偏向 1

文本遮蔽模式下 $t \sim \mathrm{Beta}(5, 1)$（偏向 1）。理由：

- **$t \approx 1$（数据端）时几何坐标几乎干净**：此时若文本被遮蔽，模型只能从几何学到的信息做边预测。如果几何编码了正确的语义（由 FM 训练所致），应能单独做好预测——**这是对"几何是否真正编码了语义"的检验**。
- **$t \approx 0$（先验端）时几何也是噪声**：若此时文本也遮蔽，几乎没有信号可用，训练信号稀疏、损失大。

偏向 1 的 $t$ 让文本遮蔽的训练信号聚焦于"几何自足性"的检验，这是此模式的核心目的。

坐标遮蔽模式 $t$ 分布不需要特殊偏向——因为被遮蔽节点的几何全程是噪声，与 $t$ 无关。

### 9.7 为什么没有显式的 InfoNCE 对齐损失

一个常见的做法是加入 InfoNCE 损失把节点潜嵌入 $\mathbf{h}_i^V$ 与文本 $\mathbf{c}_i$ 拉到一起（或类似的对比学习目标）。本文**不采用** InfoNCE，理由：

- 文本 $\mathbf{c}_i$ 已经 concat 进节点初始嵌入作为输入。InfoNCE 让输出 $\mathbf{h}_i^V$ 对齐输入 $\mathbf{c}_i$ 等价于鼓励主干退化为近恒等映射，反而损害几何信息的编码。
- 模态遮蔽下的互相重建（遮掉文本仍要预测边、遮掉坐标仍要预测边）**已经实现了"跨模态对齐"的效果**，且比 InfoNCE 更强——不是在嵌入空间拉近，而是要求"从 A 能推出 B 的下游信号"。
- 文本编码器冻结，InfoNCE 只能单向拉主干靠近文本空间，无法真正"对齐"。

因此模态遮蔽就是本文的对齐机制，不额外加 InfoNCE。

**关系侧例外**：关系数 $K$ 少，关系嵌入 $\mathbf{R}$ 是**可学习参数且全局共享**，语义一致性对下游归纳至关重要。可在关系侧加入轻量 InfoNCE（$\mathbf{r}_k \leftrightarrow \mathbf{c}_{r_k}$），作为可选增强。详细形式见第 19 章。

### 9.8 本章小结

- 模态遮蔽是预训练的数据增强策略，不是损失项，通过改变 batch 构造实现；
- 四种模式（全模态、文本遮蔽、坐标遮蔽、双模遮蔽）按概率分布选择；
- 每种模式有对应的 $t$ 分布，使训练信号与模式目的对齐；
- 边损失在所有模式下都激活，是遮蔽训练的主要监督通道；
- 模态遮蔽的效果包括防偷懒、支持归纳、隐式对齐，优于显式 InfoNCE。

至此 Part II 完整描述了流匹配框架。节点连续 FM（第 7 章）、边离散 FM（第 8 章）、模态遮蔽（第 9 章）三者共同定义了预训练的生成目标与监督信号。Part III 将给出实现这些信号所需的模型架构——RieFormer。

---

*Part II 结束。*
# Part III. 模型架构：RieFormer

本部分详细给出 RiemannFM 的模型架构——**RieFormer**。其职责是接收噪声状态 $(\mathbf{X}_t, \mathbf{E}_t, \boldsymbol{\mu}_t)$、文本条件 $(\mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R})$、关系嵌入 $\mathbf{R}$ 和时间 $t$，输出节点切向量（供连续 FM）与 $K+1$ 维边 logits（供离散 FM）。

**架构概览**（自底向上）：
- **输入编码层**（第 10 章）：把各类对象编码为 RieFormer 可处理的初始 hidden。
- **RieFormer 块**（第 11-16 章）：$L$ 层堆叠，每层包含节点自注意力、关系自注意力、边自更新、节点-关系交叉、节点-边交叉、文本条件注入、前馈等子模块。
- **预测头**（第 17 章）：节点切向量头 + 边 $K+1$ 维 logistic 头。

**架构图预览**（完整数据流见第 18 章）：

```
输入
  ↓
[输入编码层]   →  H^{V,(0)}, H^{R,(0)}, H^{E,(0)}
  ↓
[RieFormer 块 × L]
  每层:
    ├─ A_V  节点流形自注意力
    ├─ A_R  关系自注意力
    ├─ C    边流自更新
    ├─ D_VR 节点-关系交叉
    ├─ D_VE 节点-边交叉
    ├─ E_V  节点文本条件注入
    ├─ E_R  关系文本条件注入
    └─ FFN  (节点/关系/边各自)
  ↓
H^{V,(L)}, H^{R,(L)}, H^{E,(L)}
  ↓
[预测头]
  ├─ 节点切向量头  →  v̂^V_i ∈ T_{x_{i,t}}M
  └─ 边预测头     →  ℓ̂^ex_{ij},  ℓ̂^(k)_{ij}
```

---

## 第 10 章 输入编码层

### 10.1 概览

输入编码层把每类对象转化为 RieFormer 主干的初始 hidden：

| 对象 | 输入 | 输出 hidden | 维度 |
|------|------|------------|------|
| 节点 | $\mathbf{x}_{i,t}, \mathbf{c}_i, \mathbf{p}_i, m_i$ | $\mathbf{h}_i^{V,(0)}$ | $d_v$ |
| 关系 | $\mathbf{c}_{r_k}, \mathbf{r}_k$ | $\mathbf{h}_k^{R,(0)}$ | $d_r$ |
| 边 | $\mathbf{E}_{t,ij}, \mu_{t,ij}, \mathbf{C}_\mathcal{R}, \mathbf{R}$ | $\mathbf{h}_{ij}^{E,(0)}$ | $d_b$ |

时间 $t$ 通过时间嵌入 $\mathbf{t}_\mathrm{emb}$ 注入。

**维度新约定**：为避免与流形欧氏分量维度 $d_e$ 视觉混淆，本文将**边 hidden 维度**记为 $d_b$（"bond" 意）。$d_b$ 是超参，默认设 $d_b = d_v$ 但允许独立调整。

### 10.2 坐标投影

**定义 10.1（坐标投影）。** $\pi: \mathcal{M} \to \mathbb{R}^{D_\pi}$：

$$\pi(\mathbf{x}) = \Big[\mathbf{x}^\mathbb{H}_{1:d_h} \,\big\|\, \mathrm{LN}_\mathbb{S}(\mathbf{x}^\mathbb{S}) \,\big\|\, \mathrm{LN}_\mathbb{R}(\mathbf{x}^\mathbb{R})\Big] \in \mathbb{R}^{D_\pi}$$

其中 $D_\pi = d_h + (d_s + 1) + d_e$（丢弃 Lorentz 时轴 $x_0^\mathbb{H}$）。

**设计细节**：

- $\mathbf{x}^\mathbb{H}_{1:d_h}$：保留空间分量，丢弃时轴 $x_0^\mathbb{H}$。后者由空间分量与 $\kappa_h$ 决定（$x_0 = \sqrt{\|\mathbf{x}^\mathbb{H}_{1:}\|^2 + 1/|\kappa_h|}$），作为 MLP 输入会注入 batch-wide DC 偏置。
- $\mathrm{LN}_\mathbb{S}, \mathrm{LN}_\mathbb{R}$：分别作用于球面与欧氏分量的独立 LayerNorm，剥离锚点附近的尺度 DC。
- **不对 $\mathbf{x}^\mathbb{H}_{1:d_h}$ 施加 LN**：以保留 Lorentz 几何结构（LN 会混淆签名 $(-,+,\ldots,+)$）。

**代价与补偿**：$\pi$ 截断了曲率 $\kappa_h, \kappa_s$ 到 encoder 入口的梯度通路。通过 ATH-Norm（定义 11.1）将 $[\kappa_h, \kappa_s]$ 作为 FiLM 条件注入每一层以回补。

### 10.3 随机游走位置编码

**定义 10.2（RWPE）。** $\mathbf{p}_i \in \mathbb{R}^{d_\mathrm{pe}}$ 为节点 $i$ 的 $d_\mathrm{pe}$-步随机游走位置编码：

$$\mathbf{p}_i = \big[(\tilde{\mathbf{A}})_{ii},\, (\tilde{\mathbf{A}}^2)_{ii},\, \ldots,\, (\tilde{\mathbf{A}}^{d_\mathrm{pe}})_{ii}\big]$$

其中 $\tilde{\mathbf{A}} = \mathbf{D}^{-1}\mathbf{A}$ 为行归一化邻接矩阵。邻接矩阵 $\mathbf{A}$ 由当前 $\mathbf{E}_t$（考虑掩码）对 $K$ 维求和并二值化得到。

**作用**：对文本遮蔽节点（$m_i^\mathrm{text} = 0$，其文本替换为共享 `mask_emb`），若几何侧 $\mathbf{x}_{i,t}$ 又相似，主干无法区分同构位置上的不同实体，损失退化到 $\log M$。RWPE 是 identity-free 的结构签名（仅依赖邻接），打破该符号对称性而不泄漏 entity 身份。

### 10.4 时间嵌入

**定义 10.3（时间嵌入）。** 设 $d_t \in \mathbb{Z}_{>0}$ 为偶数。正弦位置编码：

$$\boldsymbol{\psi}(t) = \big[\sin(\omega_1 t), \ldots, \sin(\omega_{d_t/2} t), \cos(\omega_1 t), \ldots, \cos(\omega_{d_t/2} t)\big] \in \mathbb{R}^{d_t}$$

其中 $\omega_l = 10000^{-2l/d_t}$，$l \in [d_t/2]$。经两层 MLP 投影：

$$\mathbf{t}_\mathrm{emb} = \mathbf{W}_2^t\, \mathrm{SiLU}\!\big(\mathbf{W}_1^t \boldsymbol{\psi}(t) + \mathbf{b}_1^t\big) + \mathbf{b}_2^t \in \mathbb{R}^{d_t}$$

两层 MLP 提供更丰富的时间条件表达能力。

### 10.5 节点初始嵌入

**定义 10.4（节点初始嵌入）。**

$$\mathbf{h}_i^{V,(0)} = \mathrm{MLP}_\mathrm{node}\!\Big(\big[\pi(\mathbf{x}_{i,t}) \,\big\|\, \tilde{\mathbf{c}}_i \,\big\|\, \mathbf{p}_i \,\big\|\, m_i \,\big\|\, m_i^\mathrm{text} \,\big\|\, m_i^\mathrm{coord}\big]\Big) + \mathbf{W}_\mathrm{tp}^V \mathbf{t}_\mathrm{emb} \in \mathbb{R}^{d_v}$$

其中：
- $\pi(\mathbf{x}_{i,t}) \in \mathbb{R}^{D_\pi}$：坐标投影；
- $\tilde{\mathbf{c}}_i \in \mathbb{R}^{d_c}$：文本条件（若遮蔽则为 `mask_emb`，见 §9.3）；
- $\mathbf{p}_i \in \mathbb{R}^{d_\mathrm{pe}}$：RWPE；
- $m_i, m_i^\mathrm{text}, m_i^\mathrm{coord}$：真实节点 bit、文本遮蔽 bit、坐标遮蔽 bit；
- $\mathbf{W}_\mathrm{tp}^V \in \mathbb{R}^{d_v \times d_t}$：时间投影矩阵。

输入维度：$D_\pi + d_c + d_\mathrm{pe} + 3$；输出维度：$d_v$。

### 10.6 关系初始嵌入

**定义 10.5（关系初始嵌入）。**

$$\mathbf{h}_k^{R,(0)} = \mathrm{MLP}_\mathrm{rel}\!\Big(\big[\mathbf{r}_k \,\big\|\, \mathbf{c}_{r_k}\big]\Big) + \mathbf{W}_\mathrm{tp}^R \mathbf{t}_\mathrm{emb} \in \mathbb{R}^{d_r}$$

其中：
- $\mathbf{r}_k \in \mathbb{R}^{d_r}$：关系可学习嵌入（定义 5.1）；
- $\mathbf{c}_{r_k} \in \mathbb{R}^{d_c}$：关系文本；
- $\mathbf{W}_\mathrm{tp}^R \in \mathbb{R}^{d_r \times d_t}$：时间投影矩阵。

输入维度：$d_r + d_c$；输出维度：$d_r$。

**说明**：关系不参与模态遮蔽（§9.2），故无遮蔽 bit；关系不在图上有拓扑位置，故无 RWPE；关系无"虚关系"概念，故无 $m_k$ 指示。关系初始嵌入比节点简洁。

### 10.7 边初始嵌入

**定义 10.6（边初始嵌入）。**

$$\mathbf{h}_{ij}^{E,(0)} = \mathrm{MLP}_\mathrm{edge}\!\Big(\big[\mathbf{E}_{t,ij}\mathbf{R} \,\big\|\, \mathbf{E}_{t,ij}\mathbf{C}_\mathcal{R} \,\big\|\, \mu_{t,ij}\big]\Big) \in \mathbb{R}^{d_b}$$

其中：
- $\mathbf{E}_{t,ij}\mathbf{R} \in \mathbb{R}^{d_r}$：激活关系的嵌入之和（$\mathbf{E}_{t,ij} \in \{0,1\}^K$ 作为多热指示，与关系嵌入矩阵 $\mathbf{R}$ 的矩阵乘积）；
- $\mathbf{E}_{t,ij}\mathbf{C}_\mathcal{R} \in \mathbb{R}^{d_c}$：激活关系的文本嵌入之和；
- $\mu_{t,ij} \in \{0, 1\}$：掩码 bit。

输入维度：$d_r + d_c + 1$；输出维度：$d_b$。

**掩码位置的行为**：若 $\mu_{t,ij} = 1$（被掩码），则 $\mathbf{E}_{t,ij} = \mathbf{0}_K$（约定，见 §8.1），故 $\mathbf{E}_{t,ij}\mathbf{R} = \mathbf{0}, \mathbf{E}_{t,ij}\mathbf{C}_\mathcal{R} = \mathbf{0}$；但 $\mu_{t,ij} = 1$ 本身被送入 MLP，让其知晓"这里是被掩码"。

**时间嵌入**：边初始嵌入**不直接加** $\mathbf{t}_\mathrm{emb}$——因为 $\mathbf{E}_{t,ij}$ 本身随 $t$ 变化，已隐含时间信息。时间条件通过后续 ATH-Norm 逐层注入。

### 10.8 本章小结

至此得到三类 hidden：

- 节点：$\mathbf{H}^{V,(0)} \in \mathbb{R}^{N \times d_v}$
- 关系：$\mathbf{H}^{R,(0)} \in \mathbb{R}^{K \times d_r}$
- 边：$\mathbf{H}^{E,(0)} \in \mathbb{R}^{N \times N \times d_b}$

以及全局时间嵌入 $\mathbf{t}_\mathrm{emb} \in \mathbb{R}^{d_t}$。这些送入 RieFormer 主干。

---

## 第 11 章 RieFormer 块结构

### 11.1 块的整体结构

RieFormer 由 $L$ 个同构的块堆叠而成。第 $l$ 层（$l \in [L]$）接收上层输出 $(\mathbf{H}^{V,(l-1)}, \mathbf{H}^{R,(l-1)}, \mathbf{H}^{E,(l-1)})$，输出 $(\mathbf{H}^{V,(l)}, \mathbf{H}^{R,(l)}, \mathbf{H}^{E,(l)})$。每个块按顺序执行以下子模块（Pre-Norm 残差）：

1. **[A_V]** 节点流形自注意力（第 12 章）
2. **[A_R]** 关系自注意力（第 13 章）
3. **[C]** 边流自更新（第 14 章）
4. **[D_VR]** 节点-关系双向交叉（第 15 章）
5. **[D_VE]** 节点-边双向交叉（第 15 章）
6. **[E_V]** 节点文本条件注入（第 16 章）
7. **[E_R]** 关系文本条件注入（第 16 章）
8. **[FFN_V / FFN_R / FFN_E]** 三类 token 各自前馈残差

### 11.2 ATH-Norm：自适应时间条件归一化

**定义 11.1（ATH-Norm）。** 设 $d \in \mathbb{Z}_{>0}$ 为特征维度，$d_\mathrm{cond} \in \mathbb{Z}_{\geq 0}$ 为可选辅助条件维度（如曲率标量 $[\kappa_h, \kappa_s]$，$d_\mathrm{cond} = 0$ 时该通道关闭）。对输入 $\mathbf{h} \in \mathbb{R}^d$、时间嵌入 $\mathbf{t}_\mathrm{emb} \in \mathbb{R}^{d_t}$ 和可选条件 $\mathbf{c}_\mathrm{cond} \in \mathbb{R}^{d_\mathrm{cond}}$：

$$\mathrm{ATH}\text{-}\mathrm{Norm}(\mathbf{h}, \mathbf{t}_\mathrm{emb}, \mathbf{c}_\mathrm{cond}) = \boldsymbol{\gamma} \odot \mathrm{LN}(\mathbf{h}) + \boldsymbol{\beta}$$

其中 $\mathrm{LN}$ 为无仿射参数的 LayerNorm（仅做均值方差归一化），FiLM 式仿射参数由 $\mathbf{t}_\mathrm{emb}$ 与 $\mathbf{c}_\mathrm{cond}$ 生成：

$$[\boldsymbol{\gamma} \,\|\, \boldsymbol{\beta}] = \mathbf{W}_a \big[\mathbf{t}_\mathrm{emb} \,\|\, \mathbf{c}_\mathrm{cond}\big] + \mathbf{b}_a$$

**初始化**：$\mathbf{W}_a$ 初始化为 $\mathbf{0}$，$\mathbf{b}_a$ 前 $d$ 维初始化为 $1$、后 $d$ 维为 $0$，保证训练开始时 ATH-Norm 退化为恒等 LayerNorm。

**用途**：所有子模块的 Pre-Norm 位置均使用 ATH-Norm，使每层的归一化参数**随时间 $t$ 自适应**。

### 11.3 FFN 块

**定义 11.2（FFN）。** 扩张因子 $r_\mathrm{ffn}$（默认 4），激活 SiLU，Dropout 防过拟合。对节点、关系、边三类 token 各有独立 FFN：

$$\mathrm{FFN}_*(\mathbf{h}) = \mathbf{W}_2^{*}\, \mathrm{Drop}\!\big(\mathrm{SiLU}(\mathbf{W}_1^{*} \mathbf{h} + \mathbf{b}_1^{*})\big) + \mathbf{b}_2^{*}, \quad * \in \{V, R, E\}$$

维度：$\mathbf{W}_1^V \in \mathbb{R}^{r_\mathrm{ffn} d_v \times d_v}, \mathbf{W}_2^V \in \mathbb{R}^{d_v \times r_\mathrm{ffn} d_v}$（节点），类似地对关系（用 $d_r$）、边（用 $d_b$）。

---

## 第 12 章 子模块 A_V：节点流形感知自注意力

### 12.1 Manifold RoPE

**定义 12.1（Manifold RoPE 角度）。** 对第 $s$ 个头（$s \in [n_h]$），频率 $\omega_l^{(s)} = 10000^{-2l/d_\mathrm{head}}$（$d_\mathrm{head} = d_v / n_h$，$l \in [d_\mathrm{head}/2]$）。对节点对 $(i, j)$：

$$\theta_{ij, l}^{(s)} = \omega_l^{(s)} \cdot d_\mathcal{M}(\mathbf{x}_{i,t}, \mathbf{x}_{j,t})$$

即流形距离作为"相对位置"代入标准 RoPE 角度公式。

**定义 12.2（Manifold RoPE 旋转矩阵）。** 旋转矩阵 $\boldsymbol{\Omega}(\boldsymbol{\theta}_{ij}^{(s)}) \in \mathbb{R}^{d_\mathrm{head} \times d_\mathrm{head}}$ 为块对角矩阵，由 $d_\mathrm{head}/2$ 个 $2\times 2$ 旋转块组成：

$$\boldsymbol{\Omega}(\boldsymbol{\theta}_{ij}^{(s)}) = \mathrm{diag}\!\left(\begin{pmatrix} \cos\theta_{ij,1}^{(s)} & -\sin\theta_{ij,1}^{(s)} \\ \sin\theta_{ij,1}^{(s)} & \cos\theta_{ij,1}^{(s)} \end{pmatrix}, \ldots, \begin{pmatrix} \cos\theta_{ij,d_\mathrm{head}/2}^{(s)} & -\sin\theta_{ij,d_\mathrm{head}/2}^{(s)} \\ \sin\theta_{ij,d_\mathrm{head}/2}^{(s)} & \cos\theta_{ij,d_\mathrm{head}/2}^{(s)} \end{pmatrix}\right)$$

**作用**：把流形距离**编码为 query-key 的相对旋转**，注意力对距离敏感且尊重流形几何。

### 12.2 Geodesic Kernel

**定义 12.3（Geodesic Kernel）。** 对第 $s$ 个注意力头：

$$\kappa^{(s)}(\mathbf{x}_{i,t}, \mathbf{x}_{j,t}) = w_\mathbb{H}^{(s)} \kappa_\mathbb{H}(\mathbf{x}_{i,t}^\mathbb{H}, \mathbf{x}_{j,t}^\mathbb{H}) + w_\mathbb{S}^{(s)} \kappa_\mathbb{S}(\mathbf{x}_{i,t}^\mathbb{S}, \mathbf{x}_{j,t}^\mathbb{S}) + w_\mathbb{R}^{(s)} \kappa_\mathbb{R}(\mathbf{x}_{i,t}^\mathbb{R}, \mathbf{x}_{j,t}^\mathbb{R})$$

其中：
- $\kappa_\mathbb{H}(\mathbf{a}, \mathbf{b}) = -d_\mathbb{H}(\mathbf{a}, \mathbf{b})$
- $\kappa_\mathbb{S}(\mathbf{a}, \mathbf{b}) = \kappa_s \cdot \mathbf{a}^\top \mathbf{b}$
- $\kappa_\mathbb{R}(\mathbf{a}, \mathbf{b}) = -\|\mathbf{a} - \mathbf{b}\|_2^2$

$w_\mathbb{H}^{(s)}, w_\mathbb{S}^{(s)}, w_\mathbb{R}^{(s)} \in \mathbb{R}$ 为每头独立的可学习权重。

**作用**：Geodesic Kernel 作为 attention 偏置的加性项，直接注入"几何近则注意力强"的先验，减轻主干对几何结构学习的压力。

### 12.3 边偏置

边信息通过边偏置项进入节点注意力：

$$\text{EdgeBias}_{ij}^{(l),(s)} = \mathbf{w}_b^{(s)\top} \mathbf{h}_{ij}^{E,(l-1)}$$

其中 $\mathbf{w}_b^{(s)} \in \mathbb{R}^{d_b}$ 为每头独立的边偏置权重。这让节点注意力直接利用当前层边 hidden 的信息。

### 12.4 完整注意力计算

**定义 12.4（节点流形感知自注意力）。** Pre-Norm 风格：

$$\bar{\mathbf{h}}_i^V = \mathrm{ATH}\text{-}\mathrm{Norm}(\mathbf{h}_i^{V,(l-1)}, \mathbf{t}_\mathrm{emb}, [\kappa_h, \kappa_s])$$

对第 $s$ 个头的 query/key/value 投影：

$$\mathbf{q}_i^{(s)} = \mathbf{W}_Q^{(s)} \bar{\mathbf{h}}_i^V,\quad \mathbf{k}_j^{(s)} = \mathbf{W}_K^{(s)} \bar{\mathbf{h}}_j^V,\quad \mathbf{v}_j^{(s)} = \mathbf{W}_V^{(s)} \bar{\mathbf{h}}_j^V$$

注意力分数：

$$a_{ij}^{(s)} = \frac{(\boldsymbol{\Omega}(\boldsymbol{\theta}_{ij}^{(s)}) \mathbf{q}_i^{(s)})^\top \mathbf{k}_j^{(s)}}{\sqrt{d_\mathrm{head}}} + \beta^{(s)} \kappa^{(s)}(\mathbf{x}_{i,t}, \mathbf{x}_{j,t}) + \mathbf{w}_b^{(s)\top} \mathbf{h}_{ij}^{E,(l-1)}$$

含三项：RoPE 调制的 Q-K 内积、Geodesic Kernel 偏置、边偏置。$\beta^{(s)} \in \mathbb{R}$ 为可学习缩放。

注意力权重与输出：

$$\alpha_{ij}^{(s)} = \mathrm{softmax}_j(a_{ij}^{(s)}), \quad \mathbf{o}_i^{(s)} = \sum_{j=1}^{N} m_j \alpha_{ij}^{(s)} \mathbf{v}_j^{(s)}$$

（虚节点通过 $m_j$ 屏蔽，防止关注到 padding。）

多头拼接与残差：

$$\mathrm{MHA}_i^V = \mathbf{W}_O^V \big[\mathbf{o}_i^{(1)} \,\|\, \cdots \,\|\, \mathbf{o}_i^{(n_h)}\big]$$

$$\tilde{\mathbf{h}}_i^V = \mathbf{h}_i^{V,(l-1)} + \mathrm{MHA}_i^V$$

---

## 第 13 章 子模块 A_R：关系自注意力

### 13.1 设计动机

关系自注意力让关系 token 之间互相交互，建模关系间的代数结构（如 `father_of` 与 `parent_of` 的包含关系、`spouse_of` 的对称性、`father_of` 与 `mother_of` 的互斥性）。

**简化设计**：相比节点自注意力，关系自注意力**只保留 Geodesic Kernel 的思想（即相似度作为 attention bias），去掉 Manifold RoPE**。理由：

- 关系没有流形坐标（第 5 章），无测地距离可用；
- 使用关系嵌入的欧氏距离或余弦相似度作为"关系相似度"是更直接的选择；
- 关系之间不存在"边"（关系-关系间没有标签），无边偏置。

### 13.2 关系相似度偏置

**定义 13.1（关系相似度偏置）。** 对第 $s$ 个注意力头：

$$\kappa^{R,(s)}_{kk'} = w^{R,(s)} \cdot \cos(\mathbf{r}_k, \mathbf{r}_{k'})$$

其中 $\cos(\cdot, \cdot)$ 为余弦相似度，$w^{R,(s)} \in \mathbb{R}$ 为每头独立可学习权重。

**含义**：关系嵌入在欧氏空间中越接近（语义相近），attention bias 越大。

**为何使用 $\mathbf{r}_k$ 而非 $\mathbf{h}_k^{R,(l-1)}$**：保持跨层 bias 的一致性——所有层都基于**同一套绝对相似度先验**做关系间 attention 调制，独立于层深；这与 Geodesic Kernel 用绝对坐标（而非 hidden）的思路一致，作为稳定的几何/语义锚点。

### 13.3 关系自注意力

**定义 13.2（关系自注意力）。**

$$\bar{\mathbf{h}}_k^R = \mathrm{ATH}\text{-}\mathrm{Norm}(\mathbf{h}_k^{R,(l-1)}, \mathbf{t}_\mathrm{emb})$$

Q/K/V 投影：$\mathbf{q}_k^{R,(s)} = \mathbf{W}_Q^{R,(s)} \bar{\mathbf{h}}_k^R$ 等。

注意力分数（无 RoPE、无边偏置、含相似度偏置）：

$$a_{kk'}^{R,(s)} = \frac{\mathbf{q}_k^{R,(s)\top} \mathbf{k}_{k'}^{R,(s)}}{\sqrt{d_\mathrm{head}^R}} + \kappa^{R,(s)}_{kk'}$$

其中 $d_\mathrm{head}^R = d_r / n_h$。后续 softmax、多头拼接、残差连接与节点类似。

---

## 第 14 章 子模块 C：边流自更新

### 14.1 动机

边 hidden $\mathbf{h}_{ij}^{E,(l)}$ 随层深需要演化，以迭代精化对"位置 $(i,j)$ 边信号"的编码。子模块 C 负责这一更新。

更新信号来源：
- 当前节点 hidden $\mathbf{h}_i^{V,(l)}, \mathbf{h}_j^{V,(l)}$（节点对的当前表示）；
- 当前节点坐标的几何关系 $\log_{\mathbf{x}_{i,t}}(\mathbf{x}_{j,t})$（几何方向和距离）；
- 当前边 hidden $\mathbf{h}_{ij}^{E,(l-1)}$（已有边信号）；
- 当前关系 hidden $\mathbf{H}^{R,(l)}$（关系上下文）。

### 14.2 边流自更新公式

**定义 14.1（边流自更新）。**

$$\bar{\mathbf{h}}_{ij}^E = \mathrm{ATH}\text{-}\mathrm{Norm}(\mathbf{h}_{ij}^{E,(l-1)}, \mathbf{t}_\mathrm{emb})$$

$$\Delta \mathbf{h}_{ij}^E = \mathrm{MLP}_C\!\Big(\big[\mathbf{h}_i^{V,(l)} \,\|\, \mathbf{h}_j^{V,(l)} \,\|\, \pi^T(\log_{\mathbf{x}_{i,t}}(\mathbf{x}_{j,t})) \,\|\, \bar{\mathbf{h}}_{ij}^E\big]\Big)$$

$$\mathbf{h}_{ij}^{E,(l)} \leftarrow \mathbf{h}_{ij}^{E,(l-1)} + \Delta \mathbf{h}_{ij}^E$$

**输入维度**：$d_v + d_v + D_\pi + d_b$。其中 $\pi^T: T_{\mathbf{x}}\mathcal{M} \to \mathbb{R}^{D_\pi}$ 为切向量版的坐标投影——与定义 10.1 的 $\pi$ 类似：丢弃 Lorentz 时轴的切分量（双曲切向量自由度为 $d_h$），对球面、欧氏切分量分别做 LN。这保证 encoder 入口的切向量与流形点走同一规范化通道。

**节点更新在前**：子模块 C 的输入使用**当前层节点 hidden** $\mathbf{h}_i^{V,(l)}$（已被 [A_V] 更新过），而非上一层 $\mathbf{h}_i^{V,(l-1)}$。这让边更新利用最新的节点表示。

### 14.3 为什么不直接用关系嵌入

子模块 C 的公式中未显式使用 $\mathbf{h}_k^R$。关系信息通过以下路径间接进入边：

- 边**初始嵌入** $\mathbf{h}_{ij}^{E,(0)}$ 已含 $\mathbf{E}_{t,ij}\mathbf{R}$ 和 $\mathbf{E}_{t,ij}\mathbf{C}_\mathcal{R}$（定义 10.6），即边知晓当前激活的关系类型。
- 后续层通过**子模块 D_VE**（节点-边交叉，第 15 章）和节点-关系交叉（D_VR）间接影响边。

若需要更显式的关系-边耦合，可在 $\Delta \mathbf{h}_{ij}^E$ 的 MLP 输入中加入 $\mathbf{E}_{t,ij} \mathbf{H}^{R,(l)}$ 项（激活关系的当前 hidden 之和）。作为架构增强的可选项。

---

## 第 15 章 子模块 D：双向交叉交互

### 15.1 D_VR：节点-关系双向交叉

**动机**：节点需要知晓"与自己相关的关系有哪些"，关系需要知晓"使用此关系的节点有哪些"。双向交叉注意力提供这两路信号。

**定义 15.1（节点-关系双向交叉）。**

节点侧读取关系：

$$\bar{\mathbf{h}}_i^V = \mathrm{ATH}\text{-}\mathrm{Norm}(\mathbf{h}_i^{V,(l)})$$

$$\mathbf{h}_i^{V,(l)} \leftarrow \mathbf{h}_i^{V,(l)} + \mathrm{CrossAttn}_{V \leftarrow R}(\bar{\mathbf{h}}_i^V,\, \mathbf{H}^{R,(l)})$$

关系侧读取节点：

$$\bar{\mathbf{h}}_k^R = \mathrm{ATH}\text{-}\mathrm{Norm}(\mathbf{h}_k^{R,(l)})$$

$$\mathbf{h}_k^{R,(l)} \leftarrow \mathbf{h}_k^{R,(l)} + \mathrm{CrossAttn}_{R \leftarrow V}(\bar{\mathbf{h}}_k^R,\, \mathbf{H}^{V,(l)},\, \mathbf{m})$$

其中 $\mathrm{CrossAttn}$ 为标准 Q-K-V 交叉注意力（查询为第一参数，键值来自第二参数；第三参数是 mask）。关系读节点时通过 $\mathbf{m}$ 屏蔽虚节点。

### 15.2 D_VE：节点-边双向交叉

**动机**：节点需要知晓"与自己相连的边是什么类型"，边需要"与两端节点的最新表示对齐"。

**定义 15.2（节点-边双向交叉）。**

节点侧从边聚合（对每个节点 $i$，读取所有以 $i$ 为端点的边）：

$$\mathbf{h}_i^{V,(l)} \leftarrow \mathbf{h}_i^{V,(l)} + \mathrm{CrossAttn}_{V \leftarrow E}\!\Big(\bar{\mathbf{h}}_i^V,\, \{\mathbf{h}_{ij}^{E,(l)}\}_{j \in [N]} \cup \{\mathbf{h}_{ji}^{E,(l)}\}_{j \in [N]},\, \mathbf{m}\Big)$$

边侧从节点对更新：

$$\mathbf{h}_{ij}^{E,(l)} \leftarrow \mathbf{h}_{ij}^{E,(l)} + \mathrm{MLP}_{E \leftarrow V}\!\big([\mathbf{h}_i^{V,(l)} \,\|\, \mathbf{h}_j^{V,(l)}]\big)$$

边侧更新实际上是一个从节点对到边的直接 MLP 投影——因为每条边只连接 **固定的两个节点**，无需做 attention（attention 在这里退化为直接连接）。

### 15.3 计算成本讨论

D_VE 中节点读取边的 cross-attention 的复杂度为 $O(N \cdot N \cdot d_v) = O(N^2 d_v)$。对大子图（$N = 100$）尚可；对超大子图需稀疏化策略（见第 37 章局限讨论）。

---

## 第 16 章 子模块 E：文本条件注入

### 16.1 动机

文本条件 $\mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}$ 已经在初始嵌入阶段通过 concat 进入主干。但随层深加深，文本锚点可能被主干更新"稀释"。子模块 E 在每层重新注入文本条件，确保文本信号始终被模型可访问。这与 Diffusion Transformer（DiT）、Stable Diffusion、Perceiver 等架构的设计一致。

### 16.2 E_V：节点文本条件注入

**定义 16.1（节点文本条件注入）。**

$$\bar{\mathbf{h}}_i^V = \mathrm{ATH}\text{-}\mathrm{Norm}(\mathbf{h}_i^{V,(l)})$$

$$\mathbf{h}_i^{V,(l)} \leftarrow \mathbf{h}_i^{V,(l)} + \mathrm{CrossAttn}_{V \leftarrow C_V}\!\big(\bar{\mathbf{h}}_i^V,\, \tilde{\mathbf{C}}_\mathcal{V}\big)$$

其中 $\tilde{\mathbf{C}}_\mathcal{V}$ 为经模态遮蔽处理后的节点文本矩阵（遮蔽位置的文本行已替换为 `mask_emb`）。这个 cross-attention 让每个节点 token 从完整的节点文本矩阵中查询补充信息。

### 16.3 E_R：关系文本条件注入

**定义 16.2（关系文本条件注入）。**

$$\bar{\mathbf{h}}_k^R = \mathrm{ATH}\text{-}\mathrm{Norm}(\mathbf{h}_k^{R,(l)})$$

$$\mathbf{h}_k^{R,(l)} \leftarrow \mathbf{h}_k^{R,(l)} + \mathrm{CrossAttn}_{R \leftarrow C_R}\!\big(\bar{\mathbf{h}}_k^R,\, \mathbf{C}_\mathcal{R}\big)$$

### 16.4 前馈残差

每个块的最后，三类 token 各自过 FFN 残差：

$$\mathbf{h}_i^{V,(l)} \leftarrow \mathbf{h}_i^{V,(l)} + \mathrm{FFN}_V(\mathrm{ATH}\text{-}\mathrm{Norm}(\mathbf{h}_i^{V,(l)}))$$

$$\mathbf{h}_k^{R,(l)} \leftarrow \mathbf{h}_k^{R,(l)} + \mathrm{FFN}_R(\mathrm{ATH}\text{-}\mathrm{Norm}(\mathbf{h}_k^{R,(l)}))$$

$$\mathbf{h}_{ij}^{E,(l)} \leftarrow \mathbf{h}_{ij}^{E,(l)} + \mathrm{FFN}_E(\mathrm{ATH}\text{-}\mathrm{Norm}(\mathbf{h}_{ij}^{E,(l)}))$$

---

## 第 17 章 预测头

### 17.1 节点切向量头

**定义 17.1（节点切向量头）。** 对每个节点 $i$：

$$\mathbf{u}_i = \mathrm{MLP}_V^{\mathrm{head}}(\mathbf{h}_i^{V,(L)}) \in \mathbb{R}^D$$

$$\hat{\mathbf{v}}_i^V = \mathrm{Proj}_{\mathbf{x}_{i,t}}(\mathbf{u}_i) \in T_{\mathbf{x}_{i,t}}\mathcal{M}$$

其中 $\mathrm{MLP}_V^\mathrm{head}$ 为两层 MLP（SiLU 激活），输出维度与 ambient 空间维度 $D = (d_h + 1) + (d_s + 1) + d_e$ 一致。

**输出约束**：通过 $\mathrm{Proj}_{\mathbf{x}_{i,t}}$（定义 4.7）保证 $\hat{\mathbf{v}}_i^V \in T_{\mathbf{x}_{i,t}}\mathcal{M}$，满足后续 $\exp$ 映射与黎曼范数的几何合法性要求。

### 17.2 边预测头

**定义 17.2（边 $K+1$ 维预测头）。** 对每对 $(i, j)$：

**存在性 logit**：

$$\hat{\ell}_{ij}^\mathrm{ex} = \mathbf{w}_\mathrm{ex}^\top \mathbf{h}_{ij}^{E,(L)} + b_\mathrm{ex}$$

其中 $\mathbf{w}_\mathrm{ex} \in \mathbb{R}^{d_b}, b_\mathrm{ex} \in \mathbb{R}$。

**类型 logits（双线性形式，支持归纳）**：

$$\hat{\ell}_{ij}^{(k)} = (\mathbf{h}_{ij}^{E,(L)})^\top \mathbf{W}_\mathrm{type} \mathbf{r}_k + b_k, \quad k \in [K]$$

其中 $\mathbf{W}_\mathrm{type} \in \mathbb{R}^{d_b \times d_r}$，$\{b_k\}_{k=1}^K$ 为每类独立的偏置（默认，对应伪代码中的 `b_type: shape (K,)`）；亦可退化为全局共享标量 $b_\mathrm{type}$ 以进一步减参（低频关系场景）。

**归纳性**：新关系只要有 $\mathbf{r}_k$ 就能直接打分，无需修改打分头结构。

**实用化简**：若 $d_b = d_r$，可令 $\mathbf{W}_\mathrm{type} = \mathbf{I}$（即单纯内积），参数更少。

### 17.3 训练与推理接口

在训练时：
- $\hat{\mathbf{v}}_i^V$ 送入 $\mathcal{L}_X$（定义 7.2）；
- $\hat{\ell}_{ij}^\mathrm{ex}, \hat{\ell}_{ij}^{(k)}$ 送入 $\mathcal{L}_\mathrm{ex}, \mathcal{L}_\mathrm{ty}$（定义 8.2, 8.3）。

在推理时：
- 节点 ODE 积分：$\mathbf{x}_{i,t+\Delta t} = \exp_{\mathbf{x}_{i,t}}(\Delta t \cdot \hat{\mathbf{v}}_i^V)$；
- 边解码：按选择策略对被掩码位置采样 $(e_{ij}, \mathbf{E}_{ij}^{(k)})$（详见第 25 章）。

---

## 第 18 章 完整前向流程总览

### 18.1 数据流 ASCII 图（训练/推理统一）

```
═══════════════════════════════════════════════════════════════════════════════
                    RieFormer 完整数据流（单次前向，时间 t）
═══════════════════════════════════════════════════════════════════════════════

┌────────────────────── 输入 ────────────────────────────────┐
│  X_t ∈ M^N       节点坐标                                   │
│  E_t ∈ {0,1}^{N×N×K}, μ_t ∈ {0,1}^{N×N}   边状态            │
│  C_V ∈ R^{N×d_c}, C_R ∈ R^{K×d_c}        文本(冻结)         │
│  R ∈ R^{K×d_r}                            关系嵌入(可学习)   │
│  m ∈ {0,1}^N, m_i^text, m_i^coord         节点掩码与遮蔽     │
│  t ∈ [0,1]                                时间               │
└────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────── 输入编码层 ────────────────────────────┐
│                                                            │
│  时间嵌入: t → ψ(t) → MLP → t_emb ∈ R^{d_t}                │
│                                                            │
│  节点初始: H^{V,(0)}[i] =                                   │
│    MLP_node([π(x_{i,t}) || c̃_i || p_i || m_i               │
│              || m_i^text || m_i^coord])                    │
│    + W_tp^V · t_emb       ∈ R^{d_v}                        │
│                                                            │
│  关系初始: H^{R,(0)}[k] =                                   │
│    MLP_rel([r_k || c_{r_k}]) + W_tp^R · t_emb  ∈ R^{d_r}   │
│                                                            │
│  边初始:   H^{E,(0)}[i,j] =                                 │
│    MLP_edge([E_{t,ij}·R || E_{t,ij}·C_R || μ_{t,ij}])      │
│                             ∈ R^{d_b}                      │
└────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────── RieFormer 主干 × L 层 ─────────────────────────┐
│                                                            │
│  每层块（Pre-Norm + ATH-Norm 残差）:                        │
│                                                            │
│   [A_V]  节点流形自注意力(Manifold RoPE + Geodesic         │
│          Kernel + 边偏置)                                  │
│          H^V ← H^V + MHA_V(ATH-Norm(H^V), X_t, H^E)        │
│                                                            │
│   [A_R]  关系自注意力(相似度偏置)                           │
│          H^R ← H^R + MHA_R(ATH-Norm(H^R), R)               │
│                                                            │
│   [C]    边流自更新                                         │
│          ΔH^E = MLP_C([H^V_i || H^V_j || log_{x_i}(x_j)    │
│                       || H^E_{ij}])                        │
│          H^E ← H^E + ΔH^E                                  │
│                                                            │
│   [D_VR] 节点-关系双向交叉                                  │
│          H^V ← H^V + CA_{V←R}(H^V, H^R)                    │
│          H^R ← H^R + CA_{R←V}(H^R, H^V, m)                 │
│                                                            │
│   [D_VE] 节点-边双向交叉                                    │
│          H^V ← H^V + CA_{V←E}(H^V, H^E, m)                 │
│          H^E ← H^E + MLP_{E←V}([H^V_i || H^V_j])           │
│                                                            │
│   [E_V]  节点文本条件注入                                   │
│          H^V ← H^V + CA_{V←C_V}(H^V, C̃_V)                  │
│                                                            │
│   [E_R]  关系文本条件注入                                   │
│          H^R ← H^R + CA_{R←C_R}(H^R, C_R)                  │
│                                                            │
│   [FFN]  三类 token 各自前馈残差                            │
│                                                            │
└────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────── 预测头 ─────────────────────────────────┐
│                                                            │
│  节点切向量头:                                              │
│    u_i = MLP_V^head(H^{V,(L)}_i)   ∈ R^D                   │
│    v̂_i^V = Proj_{x_{i,t}}(u_i)    ∈ T_{x_{i,t}}M          │
│                                                            │
│  边预测头:                                                  │
│    ℓ̂^ex_{ij} = w_ex^T · H^{E,(L)}_{ij} + b_ex             │
│    ℓ̂^(k)_{ij} = H^{E,(L)}_{ij}^T · W_type · r_k + b_k     │
│                                                            │
└────────────────────────────────────────────────────────────┘
                               │
                               ▼
                   [训练损失 / 推理 ODE-解码]

═══════════════════════════════════════════════════════════════════════════════
```

### 18.2 各张量形状速查

| 张量 | 形状 | 说明 |
|------|------|------|
| $\mathbf{X}_t$ | $(N, D)$ | 节点坐标，$D = d_h + d_s + d_e + 2$ |
| $\mathbf{E}_t$ | $(N, N, K)$ | 边类型 |
| $\boldsymbol{\mu}_t$ | $(N, N)$ | 边掩码 |
| $\mathbf{C}_\mathcal{V}$ | $(N, d_c)$ | 节点文本 |
| $\mathbf{C}_\mathcal{R}$ | $(K, d_c)$ | 关系文本 |
| $\mathbf{R}$ | $(K, d_r)$ | 关系嵌入（参数） |
| $\mathbf{m}, \mathbf{m}^\mathrm{text}, \mathbf{m}^\mathrm{coord}$ | $(N,)$ | 节点 bit |
| $\mathbf{t}_\mathrm{emb}$ | $(d_t,)$ | 时间嵌入 |
| $\mathbf{H}^{V,(l)}$ | $(N, d_v)$ | 节点 hidden |
| $\mathbf{H}^{R,(l)}$ | $(K, d_r)$ | 关系 hidden |
| $\mathbf{H}^{E,(l)}$ | $(N, N, d_b)$ | 边 hidden |
| $\hat{\mathbf{v}}^V$ | $(N, D)$ | 节点切向量预测 |
| $\hat{\boldsymbol{\ell}}^\mathrm{ex}$ | $(N, N)$ | 存在性 logits |
| $\hat{\boldsymbol{\ell}}^\mathrm{type}$ | $(N, N, K)$ | 类型 logits |

### 18.3 参数可训练性分类

| 参数/对象 | 可训练 | 备注 |
|----------|-------|------|
| 文本编码器 $\phi_\mathrm{text}$ | ❄️ 冻结 | Qwen3-Embedding 预训练 |
| 节点文本 $\mathbf{C}_\mathcal{V}$ | ❄️ | 由 $\phi_\mathrm{text}$ 确定性计算 |
| 关系文本 $\mathbf{C}_\mathcal{R}$ | ❄️ | 同上 |
| 关系嵌入 $\mathbf{R}$ | 🔥 训练 | 可学习参数 |
| 输入编码 MLP | 🔥 | $\mathrm{MLP}_\mathrm{node}, \mathrm{MLP}_\mathrm{rel}, \mathrm{MLP}_\mathrm{edge}$ |
| 时间嵌入 MLP | 🔥 | $\mathbf{W}_1^t, \mathbf{W}_2^t$ 等 |
| RieFormer 所有子模块参数 | 🔥 | Q/K/V、Geodesic Kernel 权重等 |
| 预测头 | 🔥 | $\mathrm{MLP}_V^\mathrm{head}, \mathbf{w}_\mathrm{ex}, \mathbf{W}_\mathrm{type}$ 等 |
| 曲率 $\kappa_h, \kappa_s$ | 可选 🔥/❄️ | 默认固定；消融可学习 |
| `mask_emb` $\mathbf{c}_\mathrm{mask}$ | 🔥 | 文本遮蔽用 |

---

*Part III 结束。*
# Part IV. 训练

本部分给出 RiemannFM 预训练的完整算法：损失函数（第 19 章）、优化协议（第 20 章）、关系嵌入初始化（第 21 章）、完整训练流程（第 22 章）。

---

## 第 19 章 损失函数完整定义

### 19.1 总览

预训练总损失由四项组成（可选的 InfoNCE 对齐为轻量正则）：

$$\mathcal{L} = \lambda_X \mathcal{L}_X + \lambda_\mathrm{ex} \mathcal{L}_\mathrm{ex} + \lambda_\mathrm{ty} \mathcal{L}_\mathrm{ty} + \lambda_\mathrm{align}^R \mathcal{L}_\mathrm{align}^R$$

其中前三项为主损失（节点 FM + 边离散 FM），第四项为可选正则。**注意没有 $\mathcal{L}_R$**（因关系不参与 FM，见第 5 章）。

### 19.2 节点 FM 损失（重述）

$$\mathcal{L}_X = \mathbb{E}_{t, \mathbf{X}_0, \mathbf{X}_1}\Bigg[\frac{1}{Z_X} \sum_{i=1}^{N} m_i m_i^\mathrm{coord} \cdot \big\|\hat{\mathbf{v}}_i^V(t) - \mathbf{v}_i^\star(t)\big\|_{T_{\mathbf{x}_{i,t}}\mathcal{M}}^2\Bigg]$$

具体定义见 §7.4。目标场 $\mathbf{v}_i^\star(t) = \log_{\mathbf{x}_{i,t}}(\mathbf{x}_{i,1}) / (1 - t + \epsilon)$。

### 19.3 边存在性损失（重述）

$$\mathcal{L}_\mathrm{ex} = \mathbb{E}_{t, \ldots}\Bigg[\frac{1}{Z_\mathrm{ex}} \sum_{(i,j)} m_i m_j \mu_{t,ij} \cdot \mathrm{BCE}\!\big(\hat{\ell}_{ij}^\mathrm{ex}, e_{1,ij}\big)\Bigg]$$

具体定义见定义 8.2。**仅在掩码位置**计算。

### 19.4 边类型损失（重述）

$$\mathcal{L}_\mathrm{ty} = \mathbb{E}_{t, \ldots}\Bigg[\frac{1}{Z_\mathrm{ty}} \sum_{(i,j)} m_i m_j \mu_{t,ij} e_{1,ij} \cdot \sum_{k=1}^{K} \mathrm{BCE}\!\big(\hat{\ell}_{ij}^{(k)}, \mathbf{E}_{1,ij}^{(k)}\big)\Bigg]$$

具体定义见定义 8.3。**门控 + 掩码位置**双重限制。

### 19.5 关系-文本对齐损失（可选）

**定义 19.1（关系对齐 InfoNCE）。** 设 $\mathbf{W}_p \in \mathbb{R}^{d_p \times d_r}$ 和 $\mathbf{W}_p^c \in \mathbb{R}^{d_p \times d_c}$ 为对齐投影矩阵，$d_p$ 为对齐空间维度，$\tau > 0$ 为温度。定义

$$\mathbf{z}_k^R = \mathbf{W}_p \mathbf{r}_k, \quad \mathbf{z}_k^C = \mathbf{W}_p^c \mathbf{c}_{r_k}$$

对称 InfoNCE 损失：

$$\mathcal{L}_\mathrm{align}^R = -\frac{1}{2K}\sum_{k=1}^{K}\Bigg[\log\frac{\exp(\mathrm{sim}(\mathbf{z}_k^R, \mathbf{z}_k^C)/\tau)}{\sum_{k'} \exp(\mathrm{sim}(\mathbf{z}_k^R, \mathbf{z}_{k'}^C)/\tau)} + \log\frac{\exp(\mathrm{sim}(\mathbf{z}_k^C, \mathbf{z}_k^R)/\tau)}{\sum_{k'} \exp(\mathrm{sim}(\mathbf{z}_k^C, \mathbf{z}_{k'}^R)/\tau)}\Bigg]$$

其中 $\mathrm{sim}(\cdot, \cdot)$ 为余弦相似度。

**用途**：拉近关系嵌入与文本嵌入在对齐空间的语义一致性，支持跨子图一致性与归纳新关系。

**权重**：$\lambda_\mathrm{align}^R \in [0.01, 0.05]$（小权重，作为正则，不主导训练）。

### 19.6 损失权重默认值

| 损失 | 符号 | 默认权重 |
|------|------|---------|
| 节点 FM | $\lambda_X$ | 1.0 |
| 边存在性 | $\lambda_\mathrm{ex}$ | 1.0 |
| 边类型 | $\lambda_\mathrm{ty}$ | 0.5 |
| 关系对齐（可选） | $\lambda_\mathrm{align}^R$ | 0.02 |

**调参提示**：
- $\lambda_\mathrm{ty}$ 默认小于 $\lambda_\mathrm{ex}$：因为类型损失已被存在性门控，其有效样本数较少，相对权重不应过大；
- 节点 FM 与边存在性相对权重 1:1 平衡连续与离散信号；
- 若训练早期边损失收敛快、节点 FM 慢，可临时提升 $\lambda_X$（如 2.0）。

---

## 第 20 章 优化协议

### 20.1 主干参数：AdamW

主干网络（RieFormer 各层、预测头、输入编码 MLP 等）使用标准 **AdamW** 优化器：

- 学习率：$\mathrm{lr}_\mathrm{main} = 1 \times 10^{-4}$
- $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$
- 权重衰减：$\mathrm{wd} = 0.01$（不衰减 LayerNorm 与 bias）

**学习率调度**：线性 warmup（5% 步数）+ 余弦退火至 $\mathrm{lr}_\mathrm{main} / 10$。

### 20.2 关系嵌入 $\mathbf{R}$ 的优化

**欧氏优化**：$\mathbf{R}$ 作为欧氏可学习参数（第 5 章），使用标准 AdamW，无流形约束。

**学习率分组**：$\mathbf{R}$ 的梯度来自多个损失（$\mathcal{L}_\mathrm{ex}, \mathcal{L}_\mathrm{ty}, \mathcal{L}_\mathrm{align}^R$），累积信号较强。建议学习率比主干小 3-5 倍：

$$\mathrm{lr}_R = \mathrm{lr}_\mathrm{main} / 3 = 3.3 \times 10^{-5}$$

PyTorch 实现通过 param_group 指定：

```python
optimizer = AdamW([
    {'params': main_params, 'lr': 1e-4},
    {'params': [R], 'lr': 3.3e-5}
], weight_decay=0.01)
```

### 20.3 曲率参数 $\kappa_h, \kappa_s$ 的可选学习

默认固定曲率（$\kappa_h = -1, \kappa_s = 1$）。消融模式下可设为可学习参数（需注意符号约束：$\kappa_h < 0, \kappa_s > 0$）。若学习：

- 用 $\log|\kappa_h|, \log\kappa_s$ 作为 unconstrained 参数，取指数获得实际值，保证符号；
- 学习率：$\mathrm{lr}_\kappa = \mathrm{lr}_\mathrm{main} / 10 = 10^{-5}$；
- 权重衰减：不加。

### 20.4 梯度裁剪

**全局梯度裁剪**：对所有参数的梯度总范数裁剪到 $\mathrm{clip} = 1.0$：

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1, \frac{\mathrm{clip}}{\|\mathbf{g}\|_2}\right)$$

**作用**：防止 RieFormer 主干在早期训练中因 FM 损失、离散 FM 损失同时不稳定而梯度爆炸。

### 20.5 混合精度与数值稳定性

- **计算精度**：bfloat16（bf16）训练，主干前向与 loss 反传；
- **流形运算精度**：$\exp_\mathbf{x}, \log_\mathbf{x}, \mathrm{Proj}$ 等几何原语用 fp32 计算，避免 $\sinh, \arccosh$ 等在 bf16 下的数值问题；
- **$\mathrm{arccosh}, \arccos$ 的输入裁剪**：在计算双曲距离前裁剪 $\kappa_h \langle \mathbf{a}, \mathbf{b}\rangle_\mathrm{L}$ 到 $[1 + 10^{-7}, \infty)$，球面类似。

---

## 第 21 章 关系嵌入的初始化

### 21.1 动机

$\mathbf{R} \in \mathbb{R}^{K \times d_r}$ 从随机初始化开始训练会导致：

- 早期关系间代数结构混乱，边类型预测头难以学习；
- 关系-文本对齐的正信号稀疏（$\mathbf{c}_{r_k}$ 提供的语义结构被随机 $\mathbf{r}_k$ 掩盖）；
- 收敛慢。

采用**文本驱动初始化**显著缓解上述问题。

### 21.2 文本驱动初始化

**定义 21.1（关系嵌入初始化）。** 设 $\mathbf{W}_\mathrm{init} \in \mathbb{R}^{d_r \times d_c}$ 为初始化投影矩阵。初始化：

$$\mathbf{r}_k^{(0)} = \mathbf{W}_\mathrm{init} \mathbf{c}_{r_k} + \boldsymbol{\epsilon}_k, \quad \boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{0}, \sigma_\mathrm{init}^2 \mathbf{I})$$

其中 $\sigma_\mathrm{init}^2$ 为小扰动（如 $\sigma_\mathrm{init} = 0.01$），避免完全由文本决定造成初始化塌缩。

**$\mathbf{W}_\mathrm{init}$ 的选项**：

**选项 A（冻结 PCA）**：对 $\{\mathbf{c}_{r_k}\}_{k=1}^K$ 做 PCA，取前 $d_r$ 主成分方向作为 $\mathbf{W}_\mathrm{init}$。优点：无需训练、确定性、可解释。缺点：忽略边结构信息。

**选项 B（启动阶段训练的 MLP）**：见 §21.3 启动阶段流程。优点：初始化同时利用文本和图结构信号。缺点：需要额外训练阶段。

**选项 C（随机矩阵，作为 baseline）**：$\mathbf{W}_\mathrm{init}$ 高斯随机。优点：最简单。缺点：失去文本先验。

**默认推荐**：选项 A（PCA）。简单有效，在多数任务上接近选项 B。

### 21.3 启动阶段流程（可选增强）

若使用选项 B 的启动初始化，流程为：

**Phase 1（启动阶段，前 5% 步数）**：
- $\mathbf{R}$ 冻结，不更新；
- 只更新主干参数，让主干先学会"在给定 $\mathbf{R}$ 的条件下做边预测"；
- 同时收集每条训练边 $(i, r_k, j)$ 对应的节点对中点信息，累积每个关系的"使用上下文"。

**Phase 2（初始化重置）**：启动阶段结束时，按如下算法初始化 $\mathbf{R}$：

```
for k = 1, ..., K:
    ctx_k = mean over all edges (i, r_k, j) in train data of
              (x_i + x_j) / 2  # 节点对中点（欧氏近似或锚点切空间）
    r_k = W_init · c_{r_k} + α · ctx_k_projected  # 文本 + 上下文混合
```

**Phase 3（联合训练）**：解冻 $\mathbf{R}$，进入标准训练。

**代价**：额外实现复杂度。对于 Wikidata5M 规模的数据，选项 A（PCA）通常已足够，Phase 2 的复杂初始化收益有限。

### 21.4 初始化后的验证

初始化正确性可通过如下检查：

- **语义相近检查**：语义相近的关系（如 `father_of` 与 `mother_of`）应初始时距离较近；
- **语义无关检查**：语义无关的关系（如 `father_of` 与 `capital_of`）应初始时距离较远；
- **嵌入方差检查**：$\mathrm{Var}(\mathbf{R})$ 应与主干参数方差同量级，避免过小（训练不动）或过大（梯度爆炸）。

---

## 第 22 章 训练算法

### 22.1 完整 step 伪代码

```python
def train_step(batch, model, optimizer, step_idx):
    G = batch  # 子图 (V, E, C_V, C_R, m)
    N, K = G.num_nodes, G.num_relations
    
    # === Phase 0: 模态遮蔽模式选择 ===
    mode = sample_modality_mask_mode()
    # 根据 mode 设置 m_text, m_coord, 并可能调整 p_t
    m_text  = (torch.rand(N) > rho_tm).float() if mode == 'text_mask'  else torch.ones(N)
    m_coord = (torch.rand(N) > rho_cm).float() if mode == 'coord_mask' else torch.ones(N)

    if mode == 'text_mask':
        t = torch.distributions.Beta(5, 1).sample()
    else:
        # Logit-Normal
        u = torch.randn(())
        t = torch.sigmoid(u)
    
    # === Phase 1: 先验采样 ===
    X_0 = sample_prior_M(N)  # 节点先验 (M^N)
    X_1 = G.node_coords  # 节点数据
    
    # === Phase 2: 节点测地线插值 ===
    Xt = exp_map(X_0, t * log_map(X_0, X_1))
    v_star_X = log_map(Xt, X_1) / (1 - t + 1e-4)
    
    # === Phase 3: 边掩码采样 ===
    alpha_t = 1 - torch.cos(torch.pi * t / 2)  # 余弦调度
    mu_t = torch.bernoulli(torch.full((N, N), 1 - alpha_t))  # 1=掩码
    E_t = torch.where(mu_t.unsqueeze(-1) == 1,
                      torch.zeros_like(G.E_1),
                      G.E_1)
    
    # === Phase 4: 模态遮蔽应用 ===
    C_V_tilde = apply_text_mask(G.C_V, m_text, model.mask_emb)
    # 坐标遮蔽: 遮蔽节点的 Xt 不从 X_1 插值, 全用先验
    Xt = apply_coord_mask(Xt, m_coord, X_0)
    
    # === Phase 5: 主干前向 ===
    H_V, H_R, H_E = model.backbone(
        Xt, E_t, mu_t, model.R, C_V_tilde, G.C_R,
        G.m, m_text, m_coord, t
    )
    
    # === Phase 6: 预测头 ===
    u_V = model.MLP_V_head(H_V)
    v_hat = proj_tangent(u_V, Xt)  # 切空间投影
    
    ell_ex = torch.einsum('ijb,b->ij', H_E, model.w_ex) + model.b_ex
    # model.b_type: shape (K,), broadcast to (N, N, K)
    ell_type = torch.einsum('ijb,bd,kd->ijk', H_E, model.W_type, model.R) \
             + model.b_type
    
    # === Phase 7: 损失计算 ===
    L_X = riemannian_mse(v_hat, v_star_X, 
                         mask=G.m * m_coord)
    L_ex = bce(ell_ex, (G.E_1.sum(-1) > 0).float(),
               mask=G.m[:, None] * G.m[None, :] * mu_t)
    L_ty = bce(ell_type, G.E_1,
               mask=G.m[:, None, None] * G.m[None, :, None]
                    * mu_t[..., None]
                    * (G.E_1.sum(-1, keepdim=True) > 0).float())
    
    if model.use_align:
        L_align_R = infonce_loss(
            model.W_p @ model.R.T, model.W_p_c @ G.C_R.T, tau=0.1
        )
    else:
        L_align_R = 0.0
    
    L = (lambda_X * L_X + lambda_ex * L_ex + 
         lambda_ty * L_ty + lambda_align * L_align_R)
    
    # === Phase 8: 反向传播 ===
    L.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    return L.item()
```

### 22.2 Batch 构造与子图采样

训练 batch 由多个子图组成。子图采样策略：

**策略 A（邻接 k-hop 展开）**：从 $\mathcal{K}$ 随机选一个种子节点 $v$，取其 $k$-hop 邻域并限制大小不超过 $N_\mathrm{max}$。适合保留局部密集结构。

**策略 B（随机游走）**：从种子 $v$ 出发随机游走 $L$ 步，访问节点构成子图。适合探索更广的拓扑。

**策略 C（度数采样 + k-hop）**：按度数分布采样种子（偏好高度节点），再 $k$-hop 展开。适合采样到重要子图。

**默认**：策略 A（$k = 2$，$N_\mathrm{max} = 64$）。

### 22.3 虚节点填充

若采样到的子图 $|\mathcal{V}_\mathcal{G}| < N_\mathrm{max}$，按定义 3.7 填充虚节点至 $N_\mathrm{max}$。

### 22.4 Batch 级别的细节

- **Batch size**：典型 32-128 子图；由 GPU 显存决定。
- **梯度累积**：若单次 forward 显存不够，使用梯度累积（accumulation steps 4-8）。
- **分布式训练**：数据并行（DDP），$\mathbf{R}$ 作为参数自动同步。

---

# Part V. 推理与生成

本部分给出 RiemannFM 的推理算法：总览（第 23 章）、节点 ODE 积分（第 24 章）、边离散解码（第 25 章）。不同下游任务使用不同的推理配置，详见 Part VI。

---

## 第 23 章 推理算法总览

### 23.1 统一推理循环

推理从先验端开始，沿时间 $t$ 从 0 推进到 1，每步同步更新节点坐标（ODE 积分）与边（掩码解码）。

**统一推理伪代码**：

```python
def inference(model, G, num_steps=50, decode_strategy='confidence'):
    N, K = G.num_nodes, G.num_relations
    
    # 初始化：先验端
    X = sample_prior_M(N)  # 节点先验
    mu = torch.ones(N, N)  # 全掩码
    E = torch.zeros(N, N, K)  # 占位
    
    dt = 1.0 / num_steps
    
    for n in range(num_steps):
        t_n = n * dt
        t_next = (n + 1) * dt
        
        # 主干前向
        H_V, H_R, H_E = model.backbone(
            X, E, mu, model.R, G.C_V, G.C_R, G.m, t_n
        )
        
        # === 节点 ODE 步 ===
        u = model.MLP_V_head(H_V)
        v_hat = proj_tangent(u, X)
        X = exp_map(X, dt * v_hat)
        X = retract(X)  # 数值修复
        
        # === 边解码步 ===
        ell_ex = edge_head_ex(H_E)
        ell_type = edge_head_type(H_E, model.R)
        
        alpha_now = 1 - torch.cos(torch.pi * t_n / 2)
        alpha_next = 1 - torch.cos(torch.pi * t_next / 2)
        n_decode = int((alpha_next - alpha_now) * N * N)
        
        S = select_positions(ell_ex, mu, n_decode, decode_strategy)
        
        for (i, j) in S:
            e = torch.bernoulli(torch.sigmoid(ell_ex[i, j]))
            if e == 0:
                E[i, j] = 0
            else:
                E[i, j, :] = torch.bernoulli(torch.sigmoid(ell_type[i, j]))
                # 保证至少一类激活
                if E[i, j].sum() == 0:
                    k_star = torch.argmax(ell_type[i, j])
                    E[i, j, k_star] = 1
            mu[i, j] = 0
    
    return X, E
```

### 23.2 步数 $T$ 的选择

| 场景 | 典型 $T$ | 备注 |
|------|---------|------|
| 无条件子图生成 | 50-100 | 质量优先 |
| 条件子图补全 | 20-50 | 大部分已知，少数待补 |
| KGC 单步预测（RP/LP） | 1 | 直接读 logits |
| 异常检测打分 | 1-5 | 似然评估 |

$T$ 可在推理时调整，无需重训——这是 FM 相对 GAN 的优势。

### 23.3 采样器选择

**默认：欧拉法**。一阶 ODE 求解，每步一次主干前向，实现简单。

**高阶可选：黎曼 Heun 法**（二阶，每步两次前向）：

```
预测: X_pred = exp_{X_t}(dt · v̂(X_t, t))
修正: v̂_next = v̂(X_pred, t+dt)
更新: X_{t+dt} = exp_{X_t}((dt/2) · (v̂ + PT(v̂_next)))
```

其中 PT 为沿测地线的平行移动（在本方法中用近似——直接把切向量当作 ambient 向量相加，实证影响小）。

**推荐**：欧拉法 + $T = 50$ 是性能/质量平衡点。

---

## 第 24 章 节点 ODE 积分

### 24.1 欧拉法单步更新

时间 $t_n \to t_{n+1}$，步长 $\Delta t = t_{n+1} - t_n$。对每个真实节点 $i$（$m_i = 1$）：

$$\mathbf{x}_{i, t_{n+1}} = \exp_{\mathbf{x}_{i, t_n}}\!\big(\Delta t \cdot \hat{\mathbf{v}}_i^V(t_n)\big)$$

其中 $\hat{\mathbf{v}}_i^V(t_n)$ 由主干与切向量头在 $(\mathbf{X}_{t_n}, \mathbf{E}_{t_n}, t_n)$ 处前向得到。

**虚节点**（$m_i = 0$）：直接保持 $\mathbf{x}_i = \mathbf{x}_\varnothing$，不更新。

### 24.2 数值修复

每步欧拉更新后，调用 $\mathrm{Retract}$（定义 4.8）：

$$\mathbf{x}_{i, t_{n+1}} \leftarrow \mathrm{Retract}(\mathbf{x}_{i, t_{n+1}})$$

这修复浮点累积导致的轻微流形约束违反。

### 24.3 $\exp_\mathbf{x}$ 的数值稳定性

**双曲分量小切向量**：当 $\|\mathbf{v}^\mathbb{H}\|_\mathrm{L} \to 0$ 时，$\sinh(\eta)/\eta \to 1$，代码需特殊处理：

```python
def exp_H(x_H, v_H, kappa_h):
    norm_v = torch.sqrt(torch.clamp(lorentz_inner(v_H, v_H), min=0))
    eta = torch.sqrt(abs(kappa_h)) * norm_v
    # 小 eta 情形：sinh(eta)/eta ≈ 1 + eta²/6
    small = eta < 1e-5
    coeff = torch.where(small, 
                        torch.ones_like(eta),
                        torch.sinh(eta) / eta)
    return torch.cosh(eta) * x_H + coeff / torch.sqrt(abs(kappa_h)) * v_H
```

### 24.4 生成关系坐标？

**无需更新关系**：关系 $\mathbf{R}$ 是静态参数，推理时直接使用训练好的值。

---

## 第 25 章 边的离散解码

### 25.1 解码循环

每步选择一批位置解码。从全掩码开始（$\boldsymbol{\mu}_0 = \mathbf{1}$），最终至全解码（$\boldsymbol{\mu}_T = \mathbf{0}$）。本步要解码的位置数由调度决定：

$$n_\mathrm{decode}(n) = \lfloor(\alpha(t_{n+1}) - \alpha(t_n)) \cdot N^2\rfloor$$

其中 $\alpha$ 为前向调度函数（§8.2）。

### 25.2 位置选择策略

**策略 A（置信度优先，MaskGIT 风格）** — 推荐

对所有仍掩码位置 $(i, j)$ 计算置信度：

$$c_{ij} = \max\!\big(\sigma(\hat{\ell}_{ij}^\mathrm{ex}),\, 1 - \sigma(\hat{\ell}_{ij}^\mathrm{ex})\big)$$

（即"二元决定最自信的一方"的概率。）按 $c_{ij}$ 降序选 top $n_\mathrm{decode}$ 位置。

**优点**：先确定高置信位置，后续解码基于更多已知信息。  
**文献依据**：Chang et al. 2022（MaskGIT），此策略显著优于随机解码。

**策略 B（随机）**

从掩码位置均匀随机采样 $n_\mathrm{decode}$ 个。简单无偏，作为 baseline。

**策略 C（Gumbel 温度退火）**

在 $c_{ij}$ 上加 Gumbel 噪声，按退火温度采样。初期探索、后期收敛。

**默认**：策略 A（置信度优先）。

### 25.3 选中位置的值采样

对选中位置 $(i, j)$：

```
e_ij = Bernoulli(σ(ℓ̂^ex_ij))  # 存在性
if e_ij == 0:
    E_{ij} = 0^K  # 无边
else:
    for k in 1..K:
        E_{ij}^{(k)} = Bernoulli(σ(ℓ̂^{(k)}_ij))
    # 保证至少一类激活
    if E_{ij}.sum() == 0:
        k_star = argmax_k ℓ̂^{(k)}_ij
        E_{ij}^{(k_star)} = 1
μ_{ij} = 0  # 解码完成
```

**温度可调**：推理时可引入温度 $\tau$：$p = \sigma(\hat{\ell} / \tau)$。低温（$\tau < 1$）更尖锐；高温（$\tau > 1$）更随机。默认 $\tau = 1$。

**top-k 采样**：对类型预测可改用 top-k（取概率最高的 $k$ 个类型）或 top-p（核采样）。默认独立 Bernoulli。

### 25.4 训练-推理 mismatch 的实证影响

训练时掩码位置 i.i.d. 采样；推理时依赖已解码结构。理论上存在 mismatch，但扩散/FM 文献（SEDD、MaskGIT、Discrete Flow Matching）实证表明该 mismatch 在实践中可忽略。本方法不做额外对齐技巧。

### 25.5 并行推理成本

每步主干前向一次（主要成本）+ 掩码解码（轻量）。总成本 $T$ 次前向，与标准扩散/FM 推理同量级。

对极大子图（$N \gg 100$），主干前向中节点-边交叉（§15.3）的 $O(N^2)$ 成本成为瓶颈。第 37 章讨论稀疏化策略。

---

*Part IV-V 结束。*
# Part VI. 下游任务与微调

本部分给出 RiemannFM 的下游任务适配：KGC（第 27 章）、条件子图生成（第 28 章）、图异常检测（第 29 章）、微调策略（第 30 章）。总览见第 26 章。

---

## 第 26 章 下游任务总览

### 26.1 四类任务的统一框架

RiemannFM 预训练得到的生成式模型可通过**推理配置的调整**适配四类下游任务，无需改动主干架构：

| 任务 | 推理配置 | 所需能力 |
|------|---------|---------|
| RP（关系预测）$(h, ?, t)$ | 单步、$t=1$、读类型 logits | 条件分布 $p(k \mid h, t)$ |
| LP（连接预测）$(h, r, ?)$ | 单步、$t=1$、读存在性 + 类型 logits | 条件分布 $p(v \mid h, r)$ |
| 条件子图生成 | 多步、inpainting | 条件生成 |
| 图异常检测 | 单步或少步、打分 | 似然评估 |

### 26.2 预训练-微调接口

**零样本推理**（直接用预训练模型）：所有四类任务都可直接进行零样本推理，不需微调。

**微调**：在任务特定数据上进一步训练，损失函数可做任务特定调整（详见各章）。微调时：

- 通常**全参数微调**：预训练模型的所有参数都可更新（包括主干和预测头）；
- 可选 **LoRA**：只训练低秩适配矩阵，冻结主干；
- 可选**冻结主干**：仅训练预测头（快速适配，适合小数据）。

### 26.3 任务间的兼容性

不同下游任务可**联合微调**（multi-task fine-tuning），利用同一主干服务多任务。这是"基础模型"的核心价值——一次预训练，多种适配。

---

## 第 27 章 知识图谱补全（KGC）

### 27.1 关系预测 RP：$(h, ?, t)$

**任务定义**：给定头实体 $v_h$ 和尾实体 $v_t$，预测最可能的关系类型 $r_k$。

**推理协议**：

1. 构造以 $v_h, v_t$ 为核心的子图（如 2-hop 邻域）；
2. 确保 $(v_h, v_t)$ 位置未被观察（强制 $\mu_{ht} = 1$，或等价地在输入中置 $\mathbf{E}_{ht} = \mathbf{0}$）；
3. 设 $t = 1$（或训练中常用的中间 $t$，如 $t = 0.9$，取决于 ablation）；
4. 主干单次前向，读 $\hat{\ell}_{ht}^{(k)}$，$k \in [K]$；
5. $\hat{k} = \arg\max_k \hat{\ell}_{ht}^{(k)}$，或取 top-$N$ 作为候选。

**评估指标**：
- **MRR**（Mean Reciprocal Rank）：$\mathrm{MRR} = \frac{1}{|\mathcal{T}|}\sum_{(h, k, t) \in \mathcal{T}} \frac{1}{\mathrm{rank}(k)}$
- **Hits@K**（Hit at K）：真实关系是否在预测前 $K$。常用 $K = 1, 3, 10$。

### 27.2 连接预测 LP：$(h, r, ?)$

**任务定义**：给定头 $v_h$ 和关系 $r_k$，预测最可能的尾 $v_t$。

**推理协议**：

**(a) 子图内 LP**（transductive，候选 $t$ 限于当前子图）：
1. 对子图内每个候选 $v_t$，计算 $\hat{\ell}_{ht}^\mathrm{ex} \cdot \sigma(\hat{\ell}_{ht}^{(k)})$；
2. 按得分排序取 top-$N$。

**(b) 全图 LP**（inductive / KG-level，候选 $t$ 为全体 $\mathcal{V}$）：
1. 对每个候选实体 $v \in \mathcal{V}$，构造以 $v_h$ 和 $v$ 为核心的子图；
2. 对每个子图做推理；
3. 聚合所有候选的打分。

**(c) 快速 LP**（用规范坐标）：在预训练后，对每个实体 $v$ 计算其"规范坐标"$\bar{\mathbf{x}}_v$（通过主干对只包含 $v$ 的单节点子图的固定点），用 $\bar{\mathbf{x}}_v$ 作为快速打分的实体表示。

**评估指标**：与 RP 相同（MRR, Hits@K）。

### 27.3 KGC 微调损失

**关系预测微调**：基础损失为多分类交叉熵，可加辅助 FM 损失稳定。

$$\mathcal{L}_\mathrm{RP\_finetune} = -\sum_{(h, k^*, t)} \log \frac{\exp(\hat{\ell}_{ht}^{(k^*)})}{\sum_{k'} \exp(\hat{\ell}_{ht}^{(k')})} + \lambda_\mathrm{aux} (\mathcal{L}_X + \mathcal{L}_\mathrm{ex})$$

**连接预测微调**：类似，但对所有候选尾实体做 softmax。

### 27.4 归纳 KGC

**场景**：测试时出现训练未见的实体（cold-start entity）。新实体只有文本 $\mathbf{c}_{v_\mathrm{new}}$，无学到的"规范坐标"。

**处理**：利用预训练的模态遮蔽（第 9 章）——训练时模型已见过"坐标遮蔽"场景，推理时对新实体设 $m_i^\mathrm{coord} = 0$，坐标取先验样本，让模型从文本推理。

**评估基准**：如 FB15k-237 的 inductive splits（v1-v4）、WN18RR inductive 等。

---

## 第 28 章 条件子图生成

### 28.1 条件设定

给定部分已知结构作为 anchor，生成未知部分：

- **条件节点集** $\mathcal{V}_\mathrm{cond} \subseteq [N]$：这些节点的坐标 $\mathbf{x}_i$ 已知且固定；
- **条件边集** $\mathcal{E}_\mathrm{cond} \subseteq [N]^2$：这些位置的 $\mathbf{E}_{ij}$ 已知且固定；
- 其余节点、边由模型生成。

特例：
- $\mathcal{V}_\mathrm{cond} = \emptyset, \mathcal{E}_\mathrm{cond} = \emptyset$：无条件生成（所有节点/边从先验生成）；
- $\mathcal{V}_\mathrm{cond} = [N], \mathcal{E}_\mathrm{cond} = \emptyset$：已知所有节点，生成所有边；
- $\mathcal{V}_\mathrm{cond} \neq \emptyset, \mathcal{E}_\mathrm{cond} = \mathcal{E}_\mathrm{all} \setminus \{\text{少数}\}$：子图补全（inpainting）。

### 28.2 Inpainting 式推理

在统一推理循环中修改：

**条件节点**：在每步 ODE 更新中**强制把条件位置置回训练时的测地线插值**。推荐做法（与训练路径一致，避免 train-test mismatch）：

```python
# 方案 B（推荐）：条件节点沿训练测地线前进
X = exp_map(X, dt * v_hat)  # 非条件节点正常积分
Xt_cond = exp_map(X0[cond_nodes],
                  t_next * log_map(X0[cond_nodes], G.X_1[cond_nodes]))
X[cond_nodes] = Xt_cond
```

简化的"替换式 inpainting"（MaskGIT/DDIM 风格，每步直接置为数据端）亦可使用，适合最终只读取 $t=1$ 结果的场景：

```python
# 方案 A（替换式，简洁但 train-test mismatch 更大）
X = exp_map(X, dt * v_hat)
X[cond_nodes] = G.X_1[cond_nodes]
```

默认使用方案 B；若实现简便性优先或生成质量无明显差异，可回退方案 A。

**条件边**：在边解码循环中**跳过固定位置**：

```python
S = select_positions(ell_ex, mu, n_decode)
for (i, j) in S:
    if (i, j) in cond_edges:
        continue  # 跳过, 保持真实值
    # 否则正常解码
```

### 28.3 从文本描述生成子图

极端情形：$\mathcal{V}_\mathrm{cond}$ 的节点只有**文本描述**，无坐标。这是 "text-to-graph" 生成。

**推理**：
- 节点初始化为先验采样（类似普通生成）；
- 整个推理过程中**这些节点的文本始终输入主干**，引导生成方向；
- 预训练中的坐标遮蔽（第 9 章）模式为此场景提供了训练信号。

### 28.4 生成质量评估

- **边恢复精度**：给定节点，生成边的 F1 / 准确率；
- **结构统计一致性**：生成子图的度分布、聚类系数等与真实数据的 KL 散度；
- **语义合理性**（人工评估）：生成子图的边类型与节点语义是否合理。

---

## 第 29 章 图异常检测

### 29.1 边级异常

**任务**：对已有边 $(h, r_k, t)$，判断其是否异常。

**打分**：

$$s_\mathrm{edge}(h, r_k, t) = 1 - \sigma(\hat{\ell}_{ht}^{(k)})$$

直接用预训练模型单次推理（$t = 1$ 或 $t = 0.9$）得到 $\hat{\ell}$，低概率边为异常。

**阈值**：可根据训练数据边的分数分布决定异常阈值（如低于 10% 分位数）。

### 29.2 子图级异常

**任务**：判断整个子图 $\mathcal{G}$ 是否异常。

**打分**：基于重构误差。

```python
def subgraph_anomaly_score(G, num_perturbations=10):
    total_loss = 0
    for _ in range(num_perturbations):
        t = 0.5  # 中间时刻
        X_t = interpolate(X_0, G.X_1, t)  # X_0 为先验
        mu_t = torch.bernoulli(0.5, shape=(N, N))
        E_t = mask_edges(G.E_1, mu_t)
        
        # 前向，计算损失
        L = compute_losses(model, X_t, E_t, mu_t, G.X_1, G.E_1)
        total_loss += L
    
    return total_loss / num_perturbations
```

高损失子图为异常。

### 29.3 无监督评估

- **AUC-ROC**：区分异常 vs 正常子图；
- **Precision@K**：Top-K 异常中的真正异常比例；
- **对比基线**：DOMINANT（Ding et al. 2019）、CoLA（Liu et al. 2021）等。

---

## 第 30 章 微调策略汇总

### 30.1 全参数微调 vs LoRA

| 策略 | 适用 | 更新参数 |
|------|------|---------|
| 全参数 | 数据充足、任务与预训练差异大 | 所有 |
| LoRA | 数据少、任务与预训练接近 | 低秩适配矩阵（典型 $r=8-32$） |
| 只训预测头 | 最小数据 | 预测头 |

### 30.2 任务特定的损失加权

| 任务 | 建议权重 |
|------|---------|
| RP/LP | 主损失: KGC 交叉熵; $\lambda_\mathrm{aux} = 0.1$ |
| 条件生成 | 保持预训练损失比例 |
| 异常检测 | 只用 $\mathcal{L}_X + \mathcal{L}_\mathrm{ex} + \mathcal{L}_\mathrm{ty}$，不加 KGC 头 |

### 30.3 Checkpoint 选择

微调起点：**预训练最后一个 epoch**（默认）或**预训练最后 5 个 epoch 的 EMA**（更稳）。

---

# Part VII. 实验

本部分给出 RiemannFM 的实验设置与结果报告。此处为论文规划的实验框架，具体数值待实验完成后填入。

---

## 第 31 章 数据集与预处理

### 31.1 预训练数据：Wikidata5M

- **规模**：约 5M 实体，825 种关系，21M 边；
- **文本**：每实体有 label 与 description，每关系有 label；
- **预处理**：移除低频关系（少于 100 条边），实体文本用 Qwen3-Embedding 预编码缓存；
- **子图采样**：2-hop 邻域，$N_\mathrm{max} = 64$。

### 31.2 KGC 基准

- **FB15k-237**：14541 实体，237 关系，272115 训练边；
- **WN18RR**：40943 实体，11 关系，86835 训练边；
- **ogbl-wikikg2**（大规模）：2.5M 实体，535 关系，16M 训练边。

### 31.3 归纳 KGC 基准

- **FB15k-237 inductive**（Teru et al. 2020）：v1-v4 四个归纳分割，训练 / 测试实体不重叠；
- **WN18RR inductive**：类似。

### 31.4 生成与异常检测基准

- **子图生成**：随机采样 Wikidata5M 子图，mask 部分用于 inpainting；
- **异常检测**：注入合成异常（如随机替换关系类型）到正常子图，评估检测能力。

---

## 第 32 章 实现与超参数

### 32.1 默认超参数表

| 类别 | 超参数 | 默认值 |
|------|-------|-------|
| 流形 | $d_h, d_s, d_e$ | 16, 16, 32 |
| 流形 | $\kappa_h, \kappa_s$ | $-1, 1$（可学习时初始化为此） |
| 文本 | $d_c$ | 768（Qwen3） |
| 模型 | $d_v, d_r, d_b$ | 384 |
| 模型 | $L$（层数） | 6 |
| 模型 | $n_h$（头数） | 8 |
| 模型 | $d_\mathrm{pe}$ | 16 |
| 模型 | $d_t$ | 128 |
| 模型 | $r_\mathrm{ffn}$ | 4 |
| 对齐 | $d_p$（若用） | 64 |
| 训练 | Batch size | 64（子图） |
| 训练 | 学习率（主干） | $1 \times 10^{-4}$ |
| 训练 | 学习率（$\mathbf{R}$） | $3.3 \times 10^{-5}$ |
| 训练 | Weight decay | 0.01 |
| 训练 | Warmup / 总步数 | 5% / 100k |
| 损失 | $\lambda_X, \lambda_\mathrm{ex}, \lambda_\mathrm{ty}$ | 1.0, 1.0, 0.5 |
| 损失 | $\lambda_\mathrm{align}^R$ | 0.02 |
| 遮蔽 | $\rho_\mathrm{tm}, \rho_\mathrm{cm}$ | 0.30, 0.15 |
| 推理 | $T$（生成步数） | 50 |
| 推理 | 解码策略 | confidence |

### 32.2 硬件与训练规模

- **GPU**：8 × A100 80GB；
- **训练时长**：约 3 天（Wikidata5M 子集，100k 步）；
- **框架**：PyTorch 2.0，DDP，bfloat16。

---

## 第 33 章 主结果

*（待实验完成后填入具体数值。本节规划如下对比表：）*

### 33.1 KGC 对比

在 FB15k-237, WN18RR, ogbl-wikikg2 上与 TransE、RotatE、CompGCN、NBFNet 等基线对比 MRR、Hits@1、Hits@3、Hits@10。

### 33.2 归纳 KGC 对比

在 FB15k-237-v1 到 v4 上与 GraIL、NBFNet、RMPI 等归纳 KGC 基线对比。

### 33.3 条件子图生成对比

与 DiGress、GraphGPT 等图扩散模型对比生成质量（度分布 MMD、F1 恢复率等）。

### 33.4 异常检测对比

与 DOMINANT、CoLA 等对比 AUC-ROC、Precision@K。

---

## 第 34 章 消融研究

关键消融方向：

### 34.1 乘积流形分量消融

- 仅 $\mathbb{H}$（纯双曲）
- 仅 $\mathbb{S}$（纯球面）
- 仅 $\mathbb{R}$（纯欧氏）
- $\mathbb{H} \times \mathbb{R}$（双曲+欧氏）
- $\mathbb{S} \times \mathbb{R}$（球面+欧氏）
- $\mathbb{H} \times \mathbb{S} \times \mathbb{R}$（本文默认）

### 34.2 联合 FM vs 独立训练

- 联合：节点 FM + 边 FM 共享主干（默认）
- 独立：分别训练节点 FM 和边预测模型
- 预期联合 > 独立

### 34.3 模态遮蔽策略

- 无遮蔽：$p_\mathrm{full} = 1$
- 仅文本遮蔽
- 仅坐标遮蔽
- 组合（默认）

### 34.4 关系全局坐标 vs 纯文本驱动

- 默认：$\mathbf{R}$ 可学习 + 文本条件
- 纯文本：$\mathbf{R}$ 确定性投影于文本（无可学习）

### 34.5 几何感知模块消融

- 去掉 Manifold RoPE
- 去掉 Geodesic Kernel
- 去掉边偏置

### 34.6 关系自注意力 A_R 消融

- 有 / 无 A_R
- 有 / 无相似度偏置

### 34.7 时间分布 $p_t$

- Uniform
- Logit-Normal（默认）
- Beta(1, 3)
- Beta(3, 1)

### 34.8 推理步数 $T$

- $T = 10, 20, 50, 100$
- 解码策略：random vs confidence

---

## 第 35 章 分析与可视化

### 35.1 关系几何结构分析

可视化学到的关系嵌入 $\mathbf{R}$ 在二维 UMAP 投影中的聚类，观察是否与关系类别（如"family"、"location"等）一致。

### 35.2 节点几何与文本语义的对齐度量

计算节点坐标与文本嵌入的相关性（CKA、相互信息等），对比有/无模态遮蔽训练的差异。

### 35.3 生成轨迹可视化

绘制 $\mathbf{X}_t$ 沿 $t \in [0, 1]$ 的演化过程，观察从先验（混沌）到数据（聚类）的流动。

### 35.4 注意力模式分析

检查 RieFormer 各层的 attention 权重，验证 Geodesic Kernel 与边偏置的实际作用。

### 35.5 错误案例分析

挑选典型失败案例（如 MRR 低的查询），分析模型行为。

---

# Part VIII. 讨论与扩展

---

## 第 36 章 与相关工作的比较

### 36.1 与图扩散模型

**DiGress**（Vignac et al. 2023）：离散图扩散，只做无条件生成。RiemannFM 不同之处：
- 文本条件 + 几何流形
- 统一支持 KGC、生成、异常检测

**GraphGPT**（Zhao et al. 2023）：autoregressive 图生成。RiemannFM 不同：
- 并行生成（非自回归）
- 几何结构

### 36.2 与 KGE 方法

**TransE, RotatE**（Bordes et al. 2013, Sun et al. 2019）：纯判别模型，无生成能力。
**MuRP, RotH**（Balazevic et al. 2019, Chami et al. 2020）：双曲 KGE，关系作为几何变换。

RiemannFM 不同：
- 文本条件；
- 同时支持生成与补全；
- 乘积流形（MuRP/RotH 仅双曲）；
- 关系作为欧氏嵌入 + 可学习 $\mathbf{W}_\mathrm{type}$（而非流形变换）。

### 36.3 与 Riemannian Flow Matching

**Chen & Lipman 2024**：黎曼流形上的流匹配。RiemannFM 在此基础上：
- 乘积流形（而非单一流形）；
- 与离散 FM 联合（文本条件图生成）；
- 应用于知识图谱。

### 36.4 与图基础模型

**GraphGPT, OFA, GFM**（Chen et al. 2023 等）：多任务图基础模型，基于监督学习。RiemannFM 不同：
- 基于流匹配（生成式）；
- 几何流形表示。

### 36.5 与文本-图联合模型

**KEPLER, KG-BERT**：KG + 语言模型。RiemannFM 不同：
- 不训练 LM，只使用冻结文本嵌入；
- 几何表示 + 生成能力。

---

## 第 37 章 局限与未来工作

### 37.1 局限

**(1) $N^2$ 复杂度**：边 hidden 与节点-边交叉的 $O(N^2)$ 计算对大子图限制严格（$N \leq 200$ 可行，超过则慢）。稀疏化是一个方向。

**(2) 关系数 $K$ 大时的可学习 $\mathbf{R}$**：当 $K \gg 10^4$（如 Wikidata 完整 8000+ 关系），$\mathbf{R}$ 参数量与优化复杂度增大。

**(3) 推理步数成本**：多步 FM 比单步方法慢。

**(4) 文本编码器的语言限制**：依赖预训练英文/多语言编码器，对低资源语言支持有限。

### 37.2 未来工作

**时间动态 KG**：将时间戳作为节点或边的额外属性，建模 KG 随时间演化。

**多模态节点**：节点除文本外，有图像/视频等多模态特征。

**与 LLM 的对齐**：探索让 LLM 使用 RiemannFM 生成的子图作为事实支持（retrieval-augmented generation）。

**大规模稀疏图生成**：稀疏 attention、分层生成等，支持 $N > 1000$ 的大子图。

---

# 附录

## 附录 A：符号表

### A.1 集合与索引

| 符号 | 含义 |
|------|------|
| $\mathcal{V}$ | 实体集 |
| $\mathcal{R}$ | 关系类型集 |
| $\mathcal{K}$ | 知识图谱 $= (\mathcal{V}, \mathcal{E})$ |
| $\mathcal{E}$ | 事实集 $\subseteq \mathcal{V} \times \mathcal{R} \times \mathcal{V}$ |
| $\mathcal{G}$ | 子图 |
| $\mathcal{V}_\mathcal{G}, \mathcal{E}_\mathcal{G}$ | 子图节点集、边集 |
| $N$ | 子图节点数上界（虚填充后） |
| $K$ | 关系类型数 $= |\mathcal{R}|$ |
| $[n]$ | $\{1, \ldots, n\}$ |

### A.2 流形与几何对象

| 符号 | 含义 |
|------|------|
| $\mathcal{M}$ | 乘积流形 $\mathbb{H} \times \mathbb{S} \times \mathbb{R}$ |
| $\mathbb{H}^{d_h}_{\kappa_h}$ | 双曲空间（Lorentz 模型） |
| $\mathbb{S}^{d_s}_{\kappa_s}$ | 球面 |
| $\mathbb{R}^{d_e}$ | 欧氏空间 |
| $\langle \cdot, \cdot \rangle_\mathrm{L}$ | Lorentz 内积 |
| $d_\mathcal{M}$ | 乘积流形测地距离 |
| $T_\mathbf{x}\mathcal{M}$ | $\mathbf{x}$ 处切空间 |
| $\log_\mathbf{x}, \exp_\mathbf{x}$ | 对数、指数映射 |
| $\mathrm{Proj}_\mathbf{x}$ | 切空间正交投影 |
| $\mathrm{Retract}$ | 流形约束投影 |
| $\boldsymbol{\Omega}(\boldsymbol{\theta})$ | Manifold RoPE 旋转矩阵 |
| $\kappa^{(s)}$ | Geodesic Kernel |
| $\mathbf{x}_\varnothing$ | 流形锚点 |

### A.3 网络张量

| 符号 | 含义 | 形状 |
|------|------|------|
| $\mathbf{X}, \mathbf{X}_t$ | 节点坐标 | $(N, D)$ |
| $\mathbf{E}, \mathbf{E}_t$ | 边类型张量 | $(N, N, K)$ |
| $\boldsymbol{\mu}_t$ | 边掩码 | $(N, N)$ |
| $\mathbf{R}$ | 关系嵌入 | $(K, d_r)$ |
| $\mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}$ | 节点/关系文本 | $(N, d_c), (K, d_c)$ |
| $\mathbf{m}$ | 真实节点掩码 | $(N,)$ |
| $m_i^\mathrm{text}, m_i^\mathrm{coord}$ | 节点模态遮蔽 bit | $(N,)$ |
| $\mathbf{H}^V, \mathbf{H}^R, \mathbf{H}^E$ | 三类 hidden | $(N, d_v), (K, d_r), (N, N, d_b)$ |
| $\mathbf{t}_\mathrm{emb}$ | 时间嵌入 | $(d_t,)$ |
| $\hat{\mathbf{v}}^V$ | 节点切向量预测 | $(N, D)$ |
| $\hat{\boldsymbol{\ell}}^\mathrm{ex}, \hat{\boldsymbol{\ell}}^\mathrm{type}$ | 边 logits | $(N, N), (N, N, K)$ |

### A.4 损失与超参数

| 符号 | 含义 |
|------|------|
| $\mathcal{L}_X$ | 节点 FM 损失 |
| $\mathcal{L}_\mathrm{ex}, \mathcal{L}_\mathrm{ty}$ | 边存在性、类型损失 |
| $\mathcal{L}_\mathrm{align}^R$ | 关系-文本对齐 |
| $\lambda_X, \lambda_\mathrm{ex}, \lambda_\mathrm{ty}, \lambda_\mathrm{align}^R$ | 损失权重 |
| $t \in [0, 1]$ | 流匹配时间 |
| $p_t$ | $t$ 采样分布 |
| $\alpha(t)$ | 边掩码调度函数 |
| $\rho_\mathrm{tm}, \rho_\mathrm{cm}$ | 文本/坐标遮蔽比例 |
| $\mathbf{v}_i^\star(t)$ | 目标向量场 |
| $p_\mathrm{prior}$ | 节点先验分布 |

---

## 附录 B：几何运算的数值实现

### B.1 双曲映射的稳定实现

```python
def exp_H(x_H, v_H, kappa_h):
    """双曲指数映射，数值稳定版"""
    inner_vv = lorentz_inner(v_H, v_H)
    norm_v = torch.sqrt(torch.clamp(inner_vv, min=1e-10))
    sqrt_k = torch.sqrt(abs(kappa_h))
    eta = sqrt_k * norm_v
    small = eta < 1e-5
    sinh_over_eta = torch.where(
        small,
        torch.ones_like(eta) + eta**2 / 6,  # Taylor
        torch.sinh(eta) / eta
    )
    return torch.cosh(eta) * x_H + sinh_over_eta / sqrt_k * v_H

def log_H(x_H, y_H, kappa_h):
    """双曲对数映射"""
    inner = lorentz_inner(x_H, y_H)
    # 裁剪保证 arccosh 域
    arg = torch.clamp(kappa_h * inner, min=1 + 1e-7)
    d = torch.arccosh(arg) / torch.sqrt(abs(kappa_h))
    u = y_H - kappa_h * inner.unsqueeze(-1) * x_H
    u_norm = torch.sqrt(torch.clamp(abs(lorentz_inner(u, u)), min=1e-10))
    small = u_norm < 1e-7
    coeff = torch.where(small, torch.zeros_like(d), d / u_norm)
    return coeff.unsqueeze(-1) * u
```

### B.2 球面映射的稳定实现

```python
def exp_S(x_S, v_S, kappa_s):
    """球面指数映射"""
    norm_v = torch.sqrt(torch.clamp((v_S * v_S).sum(-1), min=1e-10))
    sqrt_k = torch.sqrt(kappa_s)
    xi = sqrt_k * norm_v
    small = xi < 1e-5
    sin_over_xi = torch.where(
        small,
        torch.ones_like(xi) - xi**2 / 6,
        torch.sin(xi) / xi
    )
    return torch.cos(xi).unsqueeze(-1) * x_S + \
           (sin_over_xi / sqrt_k).unsqueeze(-1) * v_S

def log_S(x_S, y_S, kappa_s):
    """球面对数映射"""
    inner = (x_S * y_S).sum(-1)
    arg = torch.clamp(kappa_s * inner, -1 + 1e-7, 1 - 1e-7)
    d = torch.arccos(arg) / torch.sqrt(kappa_s)
    w = y_S - kappa_s * inner.unsqueeze(-1) * x_S
    w_norm = torch.sqrt(torch.clamp((w * w).sum(-1), min=1e-10))
    small = w_norm < 1e-7
    coeff = torch.where(small, torch.zeros_like(d), d / w_norm)
    return coeff.unsqueeze(-1) * w
```

### B.3 Retract 的实现

```python
def retract_M(z, kappa_h, kappa_s, d_h, d_s):
    """乘积流形约束投影"""
    z_H = z[..., :d_h+1]
    z_S = z[..., d_h+1:d_h+1+d_s+1]
    z_R = z[..., d_h+1+d_s+1:]
    
    # Hyperbolic
    inner_H = lorentz_inner(z_H, z_H)
    scale_H = torch.sqrt(torch.clamp(kappa_h * inner_H, min=1e-10))
    z_H_new = z_H / scale_H.unsqueeze(-1)
    
    # Spherical
    norm_S = torch.sqrt(torch.clamp((z_S * z_S).sum(-1), min=1e-10))
    z_S_new = z_S / (torch.sqrt(kappa_s) * norm_S).unsqueeze(-1)
    
    # Euclidean (identity)
    return torch.cat([z_H_new, z_S_new, z_R], dim=-1)
```

---

## 附录 C：命题与证明（要点）

### C.1 CFM 定理在乘积流形上的推广

**命题**：设 $\mathcal{M} = \mathcal{M}_1 \times \mathcal{M}_2$，$p_0, p_1$ 为 $\mathcal{M}$ 上分布，条件路径 $\gamma_t$ 沿各分量独立测地线。则黎曼 CFM 损失在乘积流形上分解：

$$\mathcal{L}_\mathrm{RCFM} = \mathcal{L}_\mathrm{RCFM}^{(1)} + \mathcal{L}_\mathrm{RCFM}^{(2)}$$

*证明要点*：黎曼范数在乘积切空间 $T_\mathbf{x}\mathcal{M} = T_{\mathbf{x}_1}\mathcal{M}_1 \oplus T_{\mathbf{x}_2}\mathcal{M}_2$ 上分解为各分量范数之和。条件路径各分量独立，目标场亦分解。两者结合得分解式。∎

### C.2 $\mathbf{v}^\star(t) = \log_{\mathbf{x}_t}(\mathbf{x}_1) / (1 - t)$ 的等价性

**命题**：对测地线 $\gamma(t) = \exp_{\mathbf{x}_0}(t \log_{\mathbf{x}_0}(\mathbf{x}_1))$，其速度 $\dot\gamma(t) = \log_{\mathbf{x}_t}(\mathbf{x}_1) / (1 - t)$。

*证明要点*：测地线 $\gamma$ 的范数恒等于 $d(\mathbf{x}_0, \mathbf{x}_1)$（等速性）。从 $\mathbf{x}_t$ 到 $\mathbf{x}_1$ 的剩余测地段长度 $= (1-t) d(\mathbf{x}_0, \mathbf{x}_1)$，方向由 $\log_{\mathbf{x}_t}(\mathbf{x}_1)$ 给出，范数等于 $(1-t) d(\mathbf{x}_0, \mathbf{x}_1)$。故 $\log_{\mathbf{x}_t}(\mathbf{x}_1) = (1-t) \dot\gamma(t)$，即 $\dot\gamma(t) = \log_{\mathbf{x}_t}(\mathbf{x}_1) / (1-t)$。∎

### C.3 $K+1$ 维分解的概率因子化

**命题**：$\mathbf{E}_{ij} \in \{0, 1\}^K$ 的分布可分解为存在性 $e_{ij}$ 与条件类型分布：

$$\Pr[\mathbf{E}_{ij} = \mathbf{b}] = \Pr[e_{ij} = \mathbb{1}[\mathbf{b} \neq 0]] \cdot \Pr[\mathbf{E}_{ij} = \mathbf{b} \mid e_{ij} = \mathbb{1}[\mathbf{b} \neq 0]]$$

*证明要点*：$e_{ij}$ 是 $\mathbf{E}_{ij}$ 的确定性函数，故联合分布可如此分解。若 $e_{ij} = 0$，条件分布退化为 $\delta_{\mathbf{0}_K}$（无边）；若 $e_{ij} = 1$，条件分布为 $\{0, 1\}^K \setminus \{\mathbf{0}_K\}$ 上的某分布，本文建模为独立 Bernoulli（截尾）。∎

### C.4 切空间投影的正交性

见第 4 章命题 4.2。

---

## 附录 D：完整伪代码

### D.1 训练 step 完整算法

见第 22 章 §22.1。

### D.2 推理算法

见第 23 章 §23.1。

### D.3 关系嵌入初始化

```python
def init_R_pca(C_R, d_r):
    """选项 A：PCA 初始化"""
    U, S, V = torch.pca_lowrank(C_R, q=d_r)
    W_init = V.T  # d_r × d_c
    R_init = C_R @ W_init.T  # K × d_r
    R_init = R_init + 0.01 * torch.randn_like(R_init)
    return R_init.requires_grad_(True)
```

---

## 附录 E：超参数完整表与训练曲线

*（待实验完成后填入）*

---

## 附录 F：额外实验结果

*（待实验完成后填入）*

- F.1 不同文本编码器的影响（Qwen3 / E5 / SBERT）
- F.2 曲率 $\kappa$ 学习的可视化
- F.3 扩展数据集结果

---