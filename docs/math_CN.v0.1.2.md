# RiemannFM 数学形式化

**RiemannFM: Geometry- and Text-Aware Flow Matching on Product Manifolds with Joint Continuous-Discrete  Dynamics for Knowledge Graph Generation — 完整数学定义**

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
- **有向性**：$\mathbf{E}$ 作为有向张量不强制对称，一般地 $\mathbf{E}_{ij} \neq \mathbf{E}_{ji}$；
- **近似对称子集**：存在语义对称关系子集 $\mathcal{R}_{\mathrm{sym}} \subseteq \mathcal{R}$（如 spouse、sibling、shares border with、twin city），在数据中对任意 $r_k \in \mathcal{R}_{\mathrm{sym}}$ 有 $\mathbf{E}_{ij}^{(k)} \approx \mathbf{E}_{ji}^{(k)}$（实测 Wikidata5M 上反向边覆盖率 $\sim 90\text{--}99\%$，其余为标注缺失）；
- **多重性**：$\|\mathbf{E}_{ij}\|_1$ 可大于 $1$；
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

### 3.6 训练样本与数据集

**定义 3.9（训练样本）。** 一个训练样本即子图的五元组编码：
$$\left(\mathbf{X} \in \mathcal{M}^N,\; \mathbf{E} \in \{0,1\}^{N \times N \times K},\; \mathbf{C}_\mathcal{V} \in \mathbb{R}^{N \times d_c},\; \mathbf{C}_\mathcal{R} \in \mathbb{R}^{K \times d_c},\; \mathbf{m} \in \{0,1\}^N\right)$$

**定义 3.10（训练数据集）。** $\mathcal{D} = \{\mathcal{G}_1, \mathcal{G}_2, \ldots, \mathcal{G}_{|\mathcal{D}|}\}$ 为从知识图谱 $\mathcal{K}$ 中采样的子图集合。

### 5.1 总体定义

**定义 5.1（RieFormer）。** RieFormer 为参数为 $\theta$ 的映射：
$$f_\theta(\mathbf{X}_t, \mathbf{S}_t, t, \mathbf{C}_\mathcal{V}, \mathbf{C}_\mathcal{R}, \mathbf{m}) = (\hat{\mathbf{V}}, \hat{\mathbf{P}})$$

输入：$\mathbf{X}_t \in \mathcal{M}^N$，$\mathbf{S}_t \in \{\texttt{M}, 0, 1\}^{N \times N \times K}$，$t \in [0,1]$，$\mathbf{C}_\mathcal{V} \in \mathbb{R}^{N \times d_c}$，$\mathbf{C}_\mathcal{R} \in \mathbb{R}^{K \times d_c}$，$\mathbf{m} \in \{0,1\}^N$。

输出：

- $\hat{\mathbf{v}}_i \in T_{\mathbf{x}_{t,i}}\mathcal{M}$，节点 $i$ 的切向量预测，$i \in [N]$；
- $\hat{\mathbf{P}} \in [0,1]^{N \times N \times K}$，$\hat{\mathbf{P}}_{ij}^{(k)}$ 为边类型概率预测。

记 $\mathbf{X}_t = (\mathbf{x}_{t,1}, \ldots, \mathbf{x}_{t,N}) \in \mathcal{M}^N$，其中 $\mathbf{x}_{t,i} \in \mathcal{M}$ 为节点 $i$ 在时间 $t$ 的流形坐标，通过测地线插值从噪声 $\mathbf{x}_{0,i}$ 到数据 $\mathbf{x}_{1,i}$ 得到。$\mathbf{S}_t$ 为三态边状态张量，$\mathbf{s}_{t,ij}^{(k)} \in \{\texttt{M}, 0, 1\}$ 分别表示"未揭示"、"显式无边"、"揭示为边"；详细动力学见第 6 节。

$\theta$ 包含所有可学习参数，含曲率 $\kappa_h, \kappa_s$。

**架构语义约定。**

- $\mathbf{x}_{t,i} \in \mathcal{M}$、$\mathbf{s}_{t,ij} \in \{\texttt{M}, 0, 1\}^K$ 为**几何/离散状态**（state），在 $L$ 层 RieFormer 中保持不变，仅作为 attention 偏置与归一化条件被反复读取；
- $\mathbf{h}_i^{V,(l)} \in \mathbb{R}^{d_v}$、$\mathbf{h}_{ij}^{E,(l)} \in \mathbb{R}^{d_{e'}}$ 为**隐藏嵌入**（latent），完全生活在 Euclidean 空间，承担层间信息流。

几何信息仅通过三个接口与 $\mathbf{h}$ 对接：(i) 入口 $\pi$ 的实际实现式（定义 5.3）；(ii) attention 的 M-RoPE / Geodesic Kernel（定义 5.8–5.9）与 ATH-Norm 的曲率 FiLM（定义 5.11）；(iii) 出口的切空间投影（定义 5.18）。因此多头注意力本质为 Euclidean 算子，与流形保结构目标不冲突——流形约束不施加于 $\mathbf{h}$，而是体现在 attention 的 inductive bias 与最终 $\hat{\mathbf{v}} \in T_{\mathbf{x}_t}\mathcal{M}$ 的投影闭合上。

### 5.2 输入编码层

**记号**：$[\mathbf{a} \| \mathbf{b}]$ 表示向量拼接（concatenation）。

**定义 5.2（文本投影）。** 预训练文本编码器的输出维度 $d_c$ 随编码器选择而变化（如本工作 Qwen3-Embedding $d_c = 768$）。为解耦模型与文本编码器，引入共享线性投影 $\mathbf{W}_{\mathrm{c}} \in \mathbb{R}^{d_p \times d_c}$ 将文本条件映射到模型内部维度 $d_p$：

$$\bar{\mathbf{C}}_\mathcal{V} = \mathbf{C}_\mathcal{V} \mathbf{W}_{\mathrm{c}}^\top \in \mathbb{R}^{N \times d_p}, \qquad \bar{\mathbf{C}}_\mathcal{R} = \mathbf{C}_\mathcal{R} \mathbf{W}_{\mathrm{c}}^\top \in \mathbb{R}^{K \times d_p}$$

节点文本与关系文本**共享同一投影矩阵**。后续所有公式中 $\bar{\mathbf{c}}_i \in \mathbb{R}^{d_p}$ 和 $\bar{\mathbf{c}}_{r_k} \in \mathbb{R}^{d_p}$ 分别指投影后的节点和关系文本条件向量。

**定义 5.3（坐标投影）。** $\pi: \mathcal{M} \to \mathbb{R}^D$：
$$\pi(\mathbf{x}) = [\mathbf{x}^{\mathbb{H}} \| \mathbf{x}^{\mathbb{S}} \| \mathbf{x}^{\mathbb{R}}]$$

然而，在实际实现中，我们采用如下方法：$\pi: \mathcal{M} \to \mathbb{R}^{D-\mathbb{1}}$：
$$\pi(\mathbf{x}) = \Big[\,\mathbf{x}^{\mathbb{H}}_{1:d_h}\ \Big\|\ \mathrm{LN}_{\mathbb{S}}(\mathbf{x}^{\mathbb{S}})\ \Big\|\ \mathrm{LN}_{\mathbb{R}}(\mathbf{x}^{\mathbb{R}})\,\Big]$$

其中：

- $\mathbf{x}^{\mathbb{H}}_{1:d_h}$ 表示丢弃 Lorentz 时轴 $x_0$，仅保留空间分量。理由：$x_0 = \sqrt{\|\mathbf{x}^{\mathbb{H}}_{1:}\|^2 + 1/|\kappa_h|}$ 是空间坐标与 $\kappa_h$ 的确定性函数，小‑tangent 区批间方差近零，作为 MLP 输入会注入 batch‑wide DC 偏置；
- $\mathrm{LN}_{\mathbb{S}}, \mathrm{LN}_{\mathbb{R}}$ 分别为作用于 $\mathbb{S}, \mathbb{R}$ 分量的独立 LayerNorm，用以剥离 anchor 锚点（$s_0 \approx 1/\sqrt{\kappa_s}$、$\mathbf{0}_{d_e}$）附近的尺度 DC；
- 不对 $\mathbf{x}^{\mathbb{H}}_{1:d_h}$ 施加 LN，以保留 Lorentz 几何结构（LN 会混淆签名 $(-,+,\ldots,+)$）。

**设计理由。** 入口 DC 会经 MLP 放大为投影头的 rank‑1 塌缩（$L_{\mathrm{align}}$ 冻结在 $\log M$），本变换是消除该吸引子的最小入口修复。

**代价与补偿。** $\pi_{\mathrm{enc}}$ 截断了曲率 $\kappa_h, \kappa_s$ 到 encoder 入口的梯度通路，由 ATH‑Norm（定义 5.11）将 $[\kappa_h, \kappa_s]$ 作为 FiLM 条件注入每一层以回补。

**定义 5.4（位置编码，RWPE）。** $\mathbf{p}_i \in \mathbb{R}^{d_{\mathrm{pe}}}$ 为节点 $i$ 的 $k$-步随机游走位置编码（RWPE, Dwivedi et al. 2022）：

$$\mathbf{p}_i = \big[(\tilde{\mathbf{A}})_{ii},\, (\tilde{\mathbf{A}}^2)_{ii},\, \ldots,\, (\tilde{\mathbf{A}}^k)_{ii}\big],$$

其中 $\tilde{\mathbf{A}} = \mathbf{D}^{-1}\mathbf{A}$ 为行归一化邻接矩阵，邻接 $\mathbf{A}$ 由三态边状态 $\mathbf{S}_t$ 折叠得到：

$$\mathbf{A}_{ij} = \mathbb{1}\!\left[\exists\, k \in [K] : \mathbf{s}_{t,ij}^{(k)} = 1\right]$$

即只有"已揭示为 $1$"的 $(i, j, k)$ 才贡献邻接，$\texttt{M}$（未揭示）与 $0$（显式无边）同等视作无连接。$d_{\mathrm{pe}} = k$ 为 RWPE 步数（超参）。

**RWPE 的作用。** 对 MASK_C 节点（其文本侧被共享 `mask_emb` 替换），若几何侧 $\mathbf{x}$ 又相似，backbone 无法区分同构位置上的不同 entity，$L_{\mathrm{mask\_c}}$ 退化到 $\log M$。RWPE 是 identity-free 的结构签名（仅依赖邻接），打破该符号对称性而不泄漏 entity 身份。

**定义 5.5（时间嵌入）。** 设 $d_t \in \mathbb{Z}_{>0}$ 为偶数，正弦位置编码采用块拼接形式：
$$\boldsymbol{\psi}(t) = [\sin(\omega_1 t),\, \ldots,\, \sin(\omega_{d_t/2} t),\, \cos(\omega_1 t),\, \ldots,\, \cos(\omega_{d_t/2} t)] \in \mathbb{R}^{d_t}$$
其中 $\omega_l = 10000^{-2l/d_t}$，$l \in [d_t/2]$。经两层 MLP 投影：
$$\mathbf{t}_{\mathrm{emb}} = \mathbf{W}_2\,\sigma\!\left(\mathbf{W}_1 \boldsymbol{\psi}(t) + \mathbf{b}_1\right) + \mathbf{b}_2 \in \mathbb{R}^{d_t}$$
其中 $\mathbf{W}_1 \in \mathbb{R}^{d_t \times d_t}$，$\mathbf{W}_2 \in \mathbb{R}^{d_t \times d_t}$，$\sigma(\cdot) = \mathrm{SiLU}$。两层 MLP 相比单层线性提供更丰富的时间条件表达能力。

**定义 5.6（节点初始嵌入）。**
$$\mathbf{h}_i^{V,(0)} = \mathrm{MLP}_{\mathrm{node}}\!\left([\pi(\mathbf{x}_{t,i}) \| \bar{\mathbf{c}}_i \| \mathbf{p}_i \| m_i]\right) + \mathbf{W}_{\mathrm{tp}}\,\mathbf{t}_{\mathrm{emb}} \in \mathbb{R}^{d_v}$$

其中 $\pi$ 取定义 5.3 的实际实现式，$\mathbf{p}_i$ 为定义 5.4 的 RWPE，$\mathbf{t}_{\mathrm{emb}}$ 为定义 5.5 的时间嵌入。输入维度为 $(D - \mathbb{1}[d_h > 0]) + d_p + d_{\mathrm{pe}} + 1$，$d_v$ 为节点隐藏维度，$\mathbf{W}_{\mathrm{tp}} \in \mathbb{R}^{d_v \times d_t}$ 为时间投影矩阵。

此时间投影与 ATH-Norm（定义 5.11）的逐层时间条件互补：前者在输入层提供全局时间锚点，后者在每层自适应调节归一化参数。
