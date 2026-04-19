# 数学符号表（Sections 1–4）

本文整理 [math_CN.v0.1.1.md](math_CN.v0.1.1.md) Section 1–4 中出现的数学符号及其含义。

## 1. 基础集合与索引约定

| 符号 | 含义 |
|------|------|
| $\mathcal{V} = \{v_1,\ldots,v_{\|\mathcal{V}\|}\}$ | 实体集，$v_i$ 为第 $i$ 个实体 |
| $\mathcal{R} = \{r_1,\ldots,r_K\}$ | 关系类型集 |
| $K \triangleq \|\mathcal{R}\|$ | 关系类型总数 |
| $\mathcal{K} = (\mathcal{V},\mathcal{E})$ | 知识图谱 |
| $\mathcal{E} \subseteq \mathcal{V}\times\mathcal{R}\times\mathcal{V}$ | 事实（三元组）集 |
| $(v_i, r_k, v_j)$ | 一条有向事实：$v_i$ 通过 $r_k$ 连接到 $v_j$ |
| $\mathcal{R}_{\mathrm{sym}}\subseteq\mathcal{R}$ | 语义对称关系子集 |

## 2. 子图形式化

| 符号 | 含义 |
|------|------|
| $[n] \triangleq \{1,2,\ldots,n\}$ | 前 $n$ 个正整数的索引集 |
| $\mathcal{G} = (\mathcal{V}_\mathcal{G}, \mathcal{E}_\mathcal{G})$ | 知识图谱的诱导子图 |
| $\mathcal{V}_\mathcal{G},\ \mathcal{E}_\mathcal{G}$ | 子图的节点集 / 事实集 |
| $N = \|\mathcal{V}_\mathcal{G}\|$ | 子图节点数 |
| $i,j \in [N]$ | 子图内局部节点编号 |

## 3. 子图矩阵表示

### 3.1 乘积流形

| 符号 | 含义 |
|------|------|
| $\langle\mathbf{a},\mathbf{b}\rangle_{\mathrm{L}} = -a_0b_0 + \sum_{l\geq 1} a_l b_l$ | Lorentz 内积 |
| $\mathcal{M} = \mathbb{H}^{d_h}_{\kappa_h}\times\mathbb{S}^{d_s}_{\kappa_s}\times\mathbb{R}^{d_e}$ | 乘积黎曼流形 |
| $\mathbb{H}^{d_h}_{\kappa_h}$ | $d_h$ 维双曲空间（Lorentz 模型） |
| $\mathbb{S}^{d_s}_{\kappa_s}$ | $d_s$ 维球面 |
| $\mathbb{R}^{d_e}$ | $d_e$ 维欧氏空间 |
| $\kappa_h < 0,\ \kappa_s > 0$ | 双曲 / 球面分量的曲率 |
| $d_h, d_s, d_e$ | 三分量的内蕴维度 |
| $D \triangleq (d_h+1)+(d_s+1)+d_e$ | 总环境空间维度 |
| $\mathbf{x}^{\mathbb{H}}, \mathbf{x}^{\mathbb{S}}, \mathbf{x}^{\mathbb{R}}$ | 点 $\mathbf{x}\in\mathcal{M}$ 的三个子空间分量 |

### 3.2–3.4 子图五元组分量

| 符号 | 含义 |
|------|------|
| $\mathbf{X} = (\mathbf{x}_1,\ldots,\mathbf{x}_N)\in\mathcal{M}^N$ | 节点流形坐标矩阵 |
| $\mathbf{x}_i = (\mathbf{x}_i^{\mathbb{H}},\mathbf{x}_i^{\mathbb{S}},\mathbf{x}_i^{\mathbb{R}})$ | 第 $i$ 个节点的坐标 |
| $\mathbf{E}\in\{0,1\}^{N\times N\times K}$ | 边类型张量 |
| $\mathbf{E}_{ij}^{(k)} = \mathbb{1}[(v_i,r_k,v_j)\in\mathcal{E}_\mathcal{G}]$ | $i\to j$ 是否存在关系 $r_k$ |
| $\mathbf{E}_{ij}\in\{0,1\}^K$ | $i\to j$ 的多热关系向量 |
| $\|\mathbf{E}_{ij}\|_1$ | $i\to j$ 的关系类型数 |
| $\bar d = \tfrac{1}{N}\sum_i \|\{j:\mathbf{E}_{ij}\neq\mathbf{0}\}\|$ | 平均出度 |
| $\Sigma,\ \Sigma^*$ | 字符表及其有限字符串集 |
| $\circ$ | 字符串拼接 |
| $\phi_{\mathrm{text}}:\Sigma^*\to\mathbb{R}^{d_c}$ | 预训练文本编码器 |
| $d_c$ | 文本编码维度 |
| $\mathbf{c}_{v_i}=\mathbf{c}_i\in\mathbb{R}^{d_c}$ | 节点 $v_i$ 的文本条件向量 |
| $\mathbf{c}_{r_k}\in\mathbb{R}^{d_c}$ | 关系 $r_k$ 的文本条件向量 |
| $\mathrm{label}_\cdot,\ \mathrm{desc}_\cdot$ | 节点 / 关系的文本标签与描述 |
| $\mathbf{C}_\mathcal{V}\in\mathbb{R}^{N\times d_c}$ | 子图级节点文本矩阵 |
| $\mathbf{C}_\mathcal{R}\in\mathbb{R}^{K\times d_c}$ | 全局关系文本矩阵 |

### 3.5–3.6 填充、掩码、数据集

| 符号 | 含义 |
|------|------|
| $N_{\max}$ | 子图节点数上界（预设超参） |
| $\mathbf{x}_\varnothing$ | 虚节点锚点坐标 |
| $\mathbf{x}_\varnothing^{\mathbb{H}}=(1/\sqrt{\|\kappa_h\|},0,\ldots)$ | 双曲分量的锚点 |
| $\mathbf{x}_\varnothing^{\mathbb{S}}=(1/\sqrt{\kappa_s},0,\ldots)$ | 球面分量的锚点 |
| $\mathbf{x}_\varnothing^{\mathbb{R}}=\mathbf{0}_{d_e}$ | 欧氏分量的锚点 |
| $\mathbf{m}\in\{0,1\}^{N_{\max}}$ | 节点掩码（$1$ 真 / $0$ 虚） |
| $m_i$ | 第 $i$ 个节点的掩码值 |
| $\mathcal{D}=\{\mathcal{G}_1,\ldots,\mathcal{G}_{\|\mathcal{D}\|}\}$ | 训练子图数据集 |
| $\mathbf{0}_K,\ \mathbf{0}_{d_c},\ \mathbf{0}_{d_e}$ | 相应维度的零向量 |

## 4. 流形上的基本运算

### 4.1 测地距离

| 符号 | 含义 |
|------|------|
| $d_{\mathbb{H}}(\mathbf{a},\mathbf{b}) = \tfrac{1}{\sqrt{\|\kappa_h\|}}\operatorname{arccosh}(\kappa_h\langle\mathbf{a},\mathbf{b}\rangle_{\mathrm{L}})$ | 双曲测地距离 |
| $d_{\mathbb{S}}(\mathbf{a},\mathbf{b}) = \tfrac{1}{\sqrt{\kappa_s}}\arccos(\kappa_s\,\mathbf{a}^\top\mathbf{b})$ | 球面测地距离 |
| $d_{\mathbb{R}}(\mathbf{a},\mathbf{b}) = \|\mathbf{a}-\mathbf{b}\|_2$ | 欧氏距离 |
| $d_\mathcal{M}(\mathbf{x},\mathbf{y}) = \sqrt{d_{\mathbb{H}}^2+d_{\mathbb{S}}^2+d_{\mathbb{R}}^2}$ | 乘积流形测地距离（分量 $\ell_2$ 组合） |

### 4.2 切空间

| 符号 | 含义 |
|------|------|
| $T_{\mathbf{x}^{\mathbb{H}}}\mathbb{H}$ | 双曲切空间：$\{\mathbf{v}:\langle\mathbf{v},\mathbf{x}^{\mathbb{H}}\rangle_{\mathrm{L}}=0\}$ |
| $T_{\mathbf{x}^{\mathbb{S}}}\mathbb{S}$ | 球面切空间：$\{\mathbf{v}:\mathbf{v}^\top\mathbf{x}^{\mathbb{S}}=0\}$ |
| $T_{\mathbf{x}^{\mathbb{R}}}\mathbb{R}^{d_e}=\mathbb{R}^{d_e}$ | 欧氏切空间（即自身） |
| $T_\mathbf{x}\mathcal{M}$ | 乘积切空间 |
| $\mathbf{v}=(\mathbf{v}^{\mathbb{H}},\mathbf{v}^{\mathbb{S}},\mathbf{v}^{\mathbb{R}})$ | 乘积切向量 |

### 4.3–4.4 对数 / 指数映射

| 符号 | 含义 |
|------|------|
| $\log_\mathbf{x}(\mathbf{y}):\mathcal{M}\to T_\mathbf{x}\mathcal{M}$ | 对数映射（把 $\mathbf{y}$ 拉回到 $\mathbf{x}$ 的切空间） |
| $\mathbf{u}=\mathbf{y}^{\mathbb{H}}-\kappa_h\langle\mathbf{x}^{\mathbb{H}},\mathbf{y}^{\mathbb{H}}\rangle_{\mathrm{L}}\mathbf{x}^{\mathbb{H}}$ | 双曲 log 的中间投影向量 |
| $\mathbf{w}=\mathbf{y}^{\mathbb{S}}-\kappa_s(\mathbf{x}^{\mathbb{S}\top}\mathbf{y}^{\mathbb{S}})\mathbf{x}^{\mathbb{S}}$ | 球面 log 的中间投影向量 |
| $\|\mathbf{u}\|_{\mathrm{L}}=\sqrt{\|\langle\mathbf{u},\mathbf{u}\rangle_{\mathrm{L}}\|}$ | Lorentz 范数 |
| $\exp_\mathbf{x}(\mathbf{v}):T_\mathbf{x}\mathcal{M}\to\mathcal{M}$ | 指数映射（沿切向量推回流形） |
| $\cosh,\sinh,\cos,\sin$ | 双曲 / 球面指数映射中的三角/双曲函数 |
| $\mathrm{id}_\mathcal{M},\mathrm{id}_{T_\mathbf{x}\mathcal{M}}$ | 流形 / 切空间上的恒等映射 |
| injectivity radius | 互逆域；球面为 $\pi/\sqrt{\kappa_s}$，双曲/欧氏为 $\infty$ |

### 4.5 切空间黎曼范数

| 符号 | 含义 |
|------|------|
| $\|\mathbf{v}\|_{T_\mathbf{x}\mathcal{M}} = \sqrt{\|\mathbf{v}^{\mathbb{H}}\|_{\mathrm{L}}^2+\|\mathbf{v}^{\mathbb{S}}\|_2^2+\|\mathbf{v}^{\mathbb{R}}\|_2^2}$ | 乘积切空间的黎曼范数 |

## 通用记号

| 符号 | 含义 |
|------|------|
| $\triangleq$ | 定义为 |
| $\mathbb{1}[\cdot]$ | 指示函数 |
| $\|\cdot\|_1,\ \|\cdot\|_2$ | $\ell_1$ / $\ell_2$ 范数 |
| $\mathbb{Z}_{>0},\ \mathbb{R}_{\geq 0}$ | 正整数集 / 非负实数集 |
