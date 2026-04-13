# RiemannFM Mathematical Formalization

**RiemannFM: Geometry- and Text-Aware Flow Matching on Product Manifolds with Joint Continuous-Discrete Dynamics for Knowledge Graph Generation — Complete Mathematical Definitions**

---

## Table of Contents

- [RiemannFM Mathematical Formalization](#riemannfm-mathematical-formalization)
  - [Table of Contents](#table-of-contents)
  - [1. Base Sets and Indexing Conventions](#1-base-sets-and-indexing-conventions)
  - [2. Formal Definition of Subgraphs](#2-formal-definition-of-subgraphs)
  - [3. Matrix Representation of Subgraphs](#3-matrix-representation-of-subgraphs)
    - [3.1 Embedding Space: Product Manifold](#31-embedding-space-product-manifold)
    - [3.2 Node Coordinates](#32-node-coordinates)
    - [3.3 Edge Type Tensor](#33-edge-type-tensor)
    - [3.4 Text Conditioning](#34-text-conditioning)
    - [3.5 Virtual Node Padding](#35-virtual-node-padding)
    - [3.6 Training Samples and Dataset](#36-training-samples-and-dataset)
  - [4. Basic Operations on Manifolds](#4-basic-operations-on-manifolds)
    - [4.1 Geodesic Distance](#41-geodesic-distance)
    - [4.2 Tangent Space](#42-tangent-space)
    - [4.3 Logarithmic Map](#43-logarithmic-map)
    - [4.4 Exponential Map](#44-exponential-map)
    - [4.5 Tangent Space Riemannian Norm](#45-tangent-space-riemannian-norm)
  - [5. Model Architecture: RieFormer](#5-model-architecture-rieformer)
    - [5.1 Overall Definition](#51-overall-definition)
    - [5.2 Input Encoding Layer](#52-input-encoding-layer)
    - [5.3 RieFormer Block](#53-rieformer-block)
      - [5.3.1 Submodule A: Manifold-Aware Multi-Head Attention](#531-submodule-a-manifold-aware-multi-head-attention)
      - [5.3.1 Submodule A: Manifold-Aware Multi-Head Attention](#531-submodule-a-manifold-aware-multi-head-attention-1)
      - [5.3.3 Submodule C: Edge Stream Self-Update](#533-submodule-c-edge-stream-self-update)
      - [5.3.4 Submodule D: Bidirectional Cross-Interaction](#534-submodule-d-bidirectional-cross-interaction)
      - [5.3.5 Submodule E: Text Conditioning Injection](#535-submodule-e-text-conditioning-injection)
    - [5.4 Output Projection Layer](#54-output-projection-layer)
      - [5.4.1 Vector Field Output Head](#541-vector-field-output-head)
      - [5.4.2 Edge Type Output Head](#542-edge-type-output-head)
    - [5.5 Permutation Equivariance](#55-permutation-equivariance)
  - [6. Joint Discrete-Continuous Flow Matching](#6-joint-discrete-continuous-flow-matching)
    - [6.1 Noise Prior Distribution](#61-noise-prior-distribution)
    - [6.2 Forward Interpolation Process](#62-forward-interpolation-process)
    - [6.3 Conditional Targets](#63-conditional-targets)
    - [6.4 Training Loss Function](#64-training-loss-function)
    - [6.5 Training and Inference Algorithms](#65-training-and-inference-algorithms)
  - [7. Downstream Task Fine-Tuning](#7-downstream-task-fine-tuning)
    - [7.1 Unified Masking Framework](#71-unified-masking-framework)
    - [7.2 Task 1: Knowledge Graph Completion](#72-task-1-knowledge-graph-completion)
    - [7.3 Task 2: Text-Conditioned Subgraph Generation](#73-task-2-text-conditioned-subgraph-generation)
    - [7.4 Task 3: Graph Anomaly Detection](#74-task-3-graph-anomaly-detection)
  - [Symbol Quick Reference](#symbol-quick-reference)
    - [Base Sets and Indices](#base-sets-and-indices)
    - [Manifolds and Geometry](#manifolds-and-geometry)
    - [Subgraph Representation](#subgraph-representation)
    - [Model Architecture (RieFormer)](#model-architecture-rieformer)
    - [Flow Matching](#flow-matching)
    - [Loss Functions](#loss-functions)
    - [Downstream Tasks](#downstream-tasks)

---

## 1. Base Sets and Indexing Conventions

**Definition 1.1 (Base Sets).**

- **Entity set**: $\mathcal{V} = \{v_1, v_2, \ldots, v_{|\mathcal{V}|}\}$.
- **Relation type set**: $\mathcal{R} = \{r_1, r_2, \ldots, r_K\}$, where $K \triangleq |\mathcal{R}|$.
- **Knowledge graph**: $\mathcal{K} = (\mathcal{V}, \mathcal{E})$, where the fact set $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{R} \times \mathcal{V}$. A fact $(v_i, r_k, v_j) \in \mathcal{E}$ denotes that entity $v_i$ is directionally connected to entity $v_j$ via relation $r_k$.

**Self-loops**: $i = j$ is allowed, i.e., $(v_i, r_k, v_i) \in \mathcal{E}$ is valid.

**Multi-relational edges**: The same ordered entity pair $(v_i, v_j)$ may be associated with multiple relation types, i.e., $(v_i, r_a, v_j) \in \mathcal{E}$ and $(v_i, r_b, v_j) \in \mathcal{E}$ may coexist with $a \neq b$.

**Inverse relations**: Semantically inverse relations (e.g., "father" and "child") exist as independent elements in $\mathcal{R}$; no artificial inverse relations are added.

---

## 2. Formal Definition of Subgraphs

**Notation**: For any positive integer $n$, let $[n] \triangleq \{1, 2, \ldots, n\}$.

**Definition 2.1 (Subgraph).** A subgraph $\mathcal{G} = (\mathcal{V}_{\mathcal{G}}, \mathcal{E}_{\mathcal{G}})$ is an induced subgraph of $\mathcal{K}$:
- $\mathcal{V}_{\mathcal{G}} \subseteq \mathcal{V}$, $|\mathcal{V}_{\mathcal{G}}| = N$;
- $\mathcal{E}_{\mathcal{G}} = \{(v_i, r_k, v_j) \in \mathcal{E} \mid v_i, v_j \in \mathcal{V}_{\mathcal{G}}\}$.

Nodes in $\mathcal{V}_{\mathcal{G}}$ are re-indexed as $[N]$. Hereafter, $i, j \in [N]$ refer to local indices within the subgraph.

---

## 3. Matrix Representation of Subgraphs

A subgraph $\mathcal{G}$ is encoded as a 5-tuple $(\mathbf{X}, \mathbf{E}, \mathbf{C}_{\mathcal{V}}, \mathbf{C}_{\mathcal{R}}, \mathbf{m})$, corresponding to node manifold coordinates, edge type tensor, node text conditioning matrix, relation text conditioning matrix, and node mask, respectively. Each component is defined in the following subsections.

### 3.1 Embedding Space: Product Manifold

**Definition 3.1 (Lorentz Inner Product).** For $\mathbf{a}, \mathbf{b} \in \mathbb{R}^{d_h+1}$:
$$\langle \mathbf{a}, \mathbf{b} \rangle_{\mathrm{L}} = -a_0 b_0 + \sum_{l=1}^{d_h} a_l b_l$$

**Definition 3.2 (Product Manifold).**
$$\mathcal{M} = \mathbb{H}^{d_h}_{\kappa_h} \times \mathbb{S}^{d_s}_{\kappa_s} \times \mathbb{R}^{d_e}$$

where:
- $\mathbb{H}^{d_h}_{\kappa_h} = \{\mathbf{z} \in \mathbb{R}^{d_h+1} \mid \langle \mathbf{z}, \mathbf{z} \rangle_{\mathrm{L}} = 1/\kappa_h,\; z_0 > 0\}$, the $d_h$-dimensional hyperbolic space (Lorentz model) with curvature $\kappa_h < 0$;
- $\mathbb{S}^{d_s}_{\kappa_s} = \{\mathbf{z} \in \mathbb{R}^{d_s+1} \mid \|\mathbf{z}\|_2^2 = 1/\kappa_s\}$, the $d_s$-dimensional sphere with curvature $\kappa_s > 0$;
- $\mathbb{R}^{d_e}$ is the $d_e$-dimensional Euclidean space.

Total ambient space dimension $D \triangleq (d_h + 1) + (d_s + 1) + d_e$.

**Component notation**: For $\mathbf{x} \in \mathcal{M}$, denote $\mathbf{x}^{\mathbb{H}} \in \mathbb{R}^{d_h+1}$, $\mathbf{x}^{\mathbb{S}} \in \mathbb{R}^{d_s+1}$, $\mathbf{x}^{\mathbb{R}} \in \mathbb{R}^{d_e}$ as the three subspace components.

### 3.2 Node Coordinates

**Definition 3.3 (Node Coordinates).**
$$\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N) \in \mathcal{M}^N$$
where $\mathbf{x}_i = (\mathbf{x}_i^{\mathbb{H}},\, \mathbf{x}_i^{\mathbb{S}},\, \mathbf{x}_i^{\mathbb{R}}) \in \mathcal{M}$, $i \in [N]$.

### 3.3 Edge Type Tensor

**Definition 3.4 (Edge Type Tensor).**
$$\mathbf{E} \in \{0,1\}^{N \times N \times K}$$
$$\mathbf{E}_{ij}^{(k)} = \mathbb{1}[(v_i, r_k, v_j) \in \mathcal{E}_{\mathcal{G}}]$$

For fixed $(i, j)$, $\mathbf{E}_{ij} \in \{0,1\}^K$ is a multi-hot vector: $\mathbf{E}_{ij} = \mathbf{0}_K$ indicates no edge from $i$ to $j$, and $\|\mathbf{E}_{ij}\|_1 \in \{0, 1, \ldots, K\}$ gives the number of relation types from $i$ to $j$.

**Properties**:
- **Asymmetry**: In general, $\mathbf{E}_{ij} \neq \mathbf{E}_{ji}$;
- **Multiplicity**: $\|\mathbf{E}_{ij}\|_1$ may be greater than $1$.
- **Sparsity**: Average out-degree $\bar{d} = \frac{1}{N}\sum_{i=1}^{N} |\{j \in [N] : \mathbf{E}_{ij} \neq \mathbf{0}_K\}| \ll N$.

### 3.4 Text Conditioning

**Notation**: Let $\Sigma$ be the character alphabet, $\Sigma^*$ the set of finite strings over $\Sigma$, and $\circ$ the string concatenation operator.

**Definition 3.5 (Text Conditioning Vector).** Let $\phi_{\mathrm{text}}: \Sigma^* \to \mathbb{R}^{d_c}$ be a pretrained text encoder. For each node $v_i$:
$$\mathbf{c}_{v_i} = \phi_{\mathrm{text}}(\mathrm{label}_{v_i} \circ \mathrm{desc}_{v_i}) \in \mathbb{R}^{d_c}$$

For each relation type $r_k$:
$$\mathbf{c}_{r_k} = \phi_{\mathrm{text}}(\mathrm{label}_{r_k} \circ \mathrm{desc}_{r_k}) \in \mathbb{R}^{d_c}$$

Hereafter we use the shorthand $\mathbf{c}_i \triangleq \mathbf{c}_{v_i}$.

**Definition 3.6 (Text Conditioning Matrices).**
- Node text matrix: $\mathbf{C}_{\mathcal{V}} = (\mathbf{c}_1, \ldots, \mathbf{c}_N) \in \mathbb{R}^{N \times d_c}$ (subgraph-level);
- Relation text matrix: $\mathbf{C}_{\mathcal{R}} = (\mathbf{c}_{r_1}, \ldots, \mathbf{c}_{r_K}) \in \mathbb{R}^{K \times d_c}$ (global, shared across all subgraphs).

### 3.5 Virtual Node Padding

**Definition 3.7 (Virtual Node Padding).** Let $N_{\max} \in \mathbb{Z}_{>0}$ be the upper bound on subgraph node count (a preset hyperparameter). For a subgraph with $|\mathcal{V}_{\mathcal{G}}| \leq N_{\max}$, pad $N_{\max} - |\mathcal{V}_{\mathcal{G}}|$ virtual nodes to make the total node count $N_{\max}$:

- Virtual node coordinates: $\mathbf{x}_i = \mathbf{x}_\varnothing \in \mathcal{M}$, where the anchor components are
$$\mathbf{x}_\varnothing^{\mathbb{H}} = \left(\frac{1}{\sqrt{|\kappa_h|}},\, 0,\, \ldots,\, 0\right) \in \mathbb{R}^{d_h+1}, \quad \mathbf{x}_\varnothing^{\mathbb{S}} = \left(\frac{1}{\sqrt{\kappa_s}},\, 0,\, \ldots,\, 0\right) \in \mathbb{R}^{d_s+1}, \quad \mathbf{x}_\varnothing^{\mathbb{R}} = \mathbf{0}_{d_e}$$
- Virtual node edges: $\forall j \in [N_{\max}]$, $\mathbf{E}_{ij} = \mathbf{E}_{ji} = \mathbf{0}_K$;
- Virtual node text: $\mathbf{c}_i = \mathbf{0}_{d_c}$.

One can verify that $\mathbf{x}_\varnothing^{\mathbb{H}}$ satisfies $\langle \mathbf{x}_\varnothing^{\mathbb{H}}, \mathbf{x}_\varnothing^{\mathbb{H}} \rangle_{\mathrm{L}} = -1/|\kappa_h| = 1/\kappa_h$ with a positive first component; and $\mathbf{x}_\varnothing^{\mathbb{S}}$ satisfies $\|\mathbf{x}_\varnothing^{\mathbb{S}}\|_2^2 = 1/\kappa_s$.

**Definition 3.8 (Node Mask).**
$$\mathbf{m} \in \{0,1\}^{N_{\max}}, \quad m_i = \begin{cases} 1 & \text{real node} \\ 0 & \text{virtual node} \end{cases}$$

satisfying $\sum_{i=1}^{N_{\max}} m_i = |\mathcal{V}_{\mathcal{G}}|$.

**Convention**: From this point onward, $N \triangleq N_{\max}$. In all subsequent formulas, $i, j \in [N]$ includes both real and virtual nodes; virtual nodes are identified and masked via $\mathbf{m}$.

### 3.6 Training Samples and Dataset

**Definition 3.9 (Training Sample).** A training sample is the 5-tuple encoding of a subgraph:
$$\left(\mathbf{X} \in \mathcal{M}^N,\; \mathbf{E} \in \{0,1\}^{N \times N \times K},\; \mathbf{C}_{\mathcal{V}} \in \mathbb{R}^{N \times d_c},\; \mathbf{C}_{\mathcal{R}} \in \mathbb{R}^{K \times d_c},\; \mathbf{m} \in \{0,1\}^N\right)$$

**Definition 3.10 (Training Dataset).** $\mathcal{D} = \{\mathcal{G}_1, \mathcal{G}_2, \ldots, \mathcal{G}_{|\mathcal{D}|}\}$ is the collection of subgraphs sampled from the knowledge graph $\mathcal{K}$.

---

## 4. Basic Operations on Manifolds

### 4.1 Geodesic Distance

**Definition 4.1 (Component Geodesic Distances).**

- Hyperbolic distance $d_{\mathbb{H}}: \mathbb{H}^{d_h}_{\kappa_h} \times \mathbb{H}^{d_h}_{\kappa_h} \to \mathbb{R}_{\geq 0}$:
$$d_{\mathbb{H}}(\mathbf{a}, \mathbf{b}) = \frac{1}{\sqrt{|\kappa_h|}} \operatorname{arccosh}\!\left(\kappa_h \cdot \langle \mathbf{a}, \mathbf{b} \rangle_{\mathrm{L}}\right)$$
where $\kappa_h \cdot \langle \mathbf{a}, \mathbf{b} \rangle_{\mathrm{L}} \geq 1$ holds for all $\mathbf{a}, \mathbf{b} \in \mathbb{H}^{d_h}_{\kappa_h}$.

- Spherical distance $d_{\mathbb{S}}: \mathbb{S}^{d_s}_{\kappa_s} \times \mathbb{S}^{d_s}_{\kappa_s} \to \mathbb{R}_{\geq 0}$:
$$d_{\mathbb{S}}(\mathbf{a}, \mathbf{b}) = \frac{1}{\sqrt{\kappa_s}} \arccos\!\left(\kappa_s \cdot \mathbf{a}^\top \mathbf{b}\right)$$
where $\kappa_s \cdot \mathbf{a}^\top \mathbf{b} \in [-1, 1]$ is guaranteed by the Cauchy-Schwarz inequality.

- Euclidean distance $d_{\mathbb{R}}: \mathbb{R}^{d_e} \times \mathbb{R}^{d_e} \to \mathbb{R}_{\geq 0}$:
$$d_{\mathbb{R}}(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2$$

**Definition 4.2 (Product Manifold Geodesic Distance).** $d_{\mathcal{M}}: \mathcal{M} \times \mathcal{M} \to \mathbb{R}_{\geq 0}$:
$$d_{\mathcal{M}}(\mathbf{x}, \mathbf{y}) = \sqrt{d_{\mathbb{H}}(\mathbf{x}^{\mathbb{H}}, \mathbf{y}^{\mathbb{H}})^2 + d_{\mathbb{S}}(\mathbf{x}^{\mathbb{S}}, \mathbf{y}^{\mathbb{S}})^2 + d_{\mathbb{R}}(\mathbf{x}^{\mathbb{R}}, \mathbf{y}^{\mathbb{R}})^2}$$

$d_{\mathcal{M}}$ is a metric on $\mathcal{M}$ (the $\ell_2$ combination of component distances preserves the metric axioms).

### 4.2 Tangent Space

**Definition 4.3 (Tangent Space).** For $\mathbf{x} \in \mathcal{M}$, the component tangent spaces are:
- $T_{\mathbf{x}^{\mathbb{H}}}\mathbb{H} = \{\mathbf{v} \in \mathbb{R}^{d_h+1} \mid \langle \mathbf{v}, \mathbf{x}^{\mathbb{H}} \rangle_{\mathrm{L}} = 0\}$
- $T_{\mathbf{x}^{\mathbb{S}}}\mathbb{S} = \{\mathbf{v} \in \mathbb{R}^{d_s+1} \mid \mathbf{v}^\top \mathbf{x}^{\mathbb{S}} = 0\}$
- $T_{\mathbf{x}^{\mathbb{R}}}\mathbb{R}^{d_e} = \mathbb{R}^{d_e}$

Product tangent space: $T_\mathbf{x}\mathcal{M} = T_{\mathbf{x}^{\mathbb{H}}}\mathbb{H} \times T_{\mathbf{x}^{\mathbb{S}}}\mathbb{S} \times \mathbb{R}^{d_e}$. A tangent vector is denoted $\mathbf{v} = (\mathbf{v}^{\mathbb{H}}, \mathbf{v}^{\mathbb{S}}, \mathbf{v}^{\mathbb{R}}) \in T_\mathbf{x}\mathcal{M}$.

### 4.3 Logarithmic Map

The logarithmic map $\log_\mathbf{x}(\mathbf{y})$ pulls back a point $\mathbf{y}$ on the manifold to the tangent space at $\mathbf{x}$. It answers the question: "Starting from $\mathbf{x}$, in which direction and how far must one travel to reach $\mathbf{y}$?" The returned tangent vector encodes both direction and distance.

**Definition 4.4 (Logarithmic Map).** $\log_\mathbf{x}: \mathcal{M} \to T_\mathbf{x}\mathcal{M}$, with components:

- Hyperbolic component: Let $\mathbf{u} = \mathbf{y}^{\mathbb{H}} - \kappa_h \langle \mathbf{x}^{\mathbb{H}}, \mathbf{y}^{\mathbb{H}} \rangle_{\mathrm{L}} \cdot \mathbf{x}^{\mathbb{H}}$, then
$$\log_{\mathbf{x}^{\mathbb{H}}}(\mathbf{y}^{\mathbb{H}}) = \frac{d_{\mathbb{H}}(\mathbf{x}^{\mathbb{H}}, \mathbf{y}^{\mathbb{H}})}{\|\mathbf{u}\|_{\mathrm{L}}} \cdot \mathbf{u}$$
where $\|\mathbf{u}\|_{\mathrm{L}} = \sqrt{|\langle \mathbf{u}, \mathbf{u} \rangle_{\mathrm{L}}|}$. One can verify $\langle \mathbf{u}, \mathbf{x}^{\mathbb{H}} \rangle_{\mathrm{L}} = 0$, i.e., $\mathbf{u} \in T_{\mathbf{x}^{\mathbb{H}}}\mathbb{H}$.

- Spherical component: Let $\mathbf{w} = \mathbf{y}^{\mathbb{S}} - \kappa_s (\mathbf{x}^{\mathbb{S}\top} \mathbf{y}^{\mathbb{S}}) \cdot \mathbf{x}^{\mathbb{S}}$, then
$$\log_{\mathbf{x}^{\mathbb{S}}}(\mathbf{y}^{\mathbb{S}}) = \frac{d_{\mathbb{S}}(\mathbf{x}^{\mathbb{S}}, \mathbf{y}^{\mathbb{S}})}{\|\mathbf{w}\|_2} \cdot \mathbf{w}$$
One can verify $\mathbf{w}^\top \mathbf{x}^{\mathbb{S}} = 0$, i.e., $\mathbf{w} \in T_{\mathbf{x}^{\mathbb{S}}}\mathbb{S}$.

- Euclidean component:
$$\log_{\mathbf{x}^{\mathbb{R}}}(\mathbf{y}^{\mathbb{R}}) = \mathbf{y}^{\mathbb{R}} - \mathbf{x}^{\mathbb{R}}$$

Product: $\log_\mathbf{x}(\mathbf{y}) = (\log_{\mathbf{x}^{\mathbb{H}}}(\mathbf{y}^{\mathbb{H}}),\, \log_{\mathbf{x}^{\mathbb{S}}}(\mathbf{y}^{\mathbb{S}}),\, \log_{\mathbf{x}^{\mathbb{R}}}(\mathbf{y}^{\mathbb{R}}))$.

**Convention**: When $\mathbf{y} = \mathbf{x}$, $\log_\mathbf{x}(\mathbf{x}) = \mathbf{0} \in T_\mathbf{x}\mathcal{M}$.

### 4.4 Exponential Map

The exponential map $\exp_\mathbf{x}(\mathbf{v})$ pushes a tangent vector $\mathbf{v}$ back onto the manifold. It answers the question: "Starting from $\mathbf{x}$, following the direction of tangent vector $\mathbf{v}$ for a geodesic distance of $\|\mathbf{v}\|$, which point on the manifold is reached?"

**Definition 4.5 (Exponential Map).** $\exp_\mathbf{x}: T_\mathbf{x}\mathcal{M} \to \mathcal{M}$, with components:

- Hyperbolic component:
$$\exp_{\mathbf{x}^{\mathbb{H}}}(\mathbf{v}^{\mathbb{H}}) = \cosh(\sqrt{|\kappa_h|}\,\|\mathbf{v}^{\mathbb{H}}\|_{\mathrm{L}})\,\mathbf{x}^{\mathbb{H}} + \frac{\sinh(\sqrt{|\kappa_h|}\,\|\mathbf{v}^{\mathbb{H}}\|_{\mathrm{L}})}{\sqrt{|\kappa_h|}\,\|\mathbf{v}^{\mathbb{H}}\|_{\mathrm{L}}}\,\mathbf{v}^{\mathbb{H}}$$

- Spherical component:
$$\exp_{\mathbf{x}^{\mathbb{S}}}(\mathbf{v}^{\mathbb{S}}) = \cos(\sqrt{\kappa_s}\,\|\mathbf{v}^{\mathbb{S}}\|_2)\,\mathbf{x}^{\mathbb{S}} + \frac{\sin(\sqrt{\kappa_s}\,\|\mathbf{v}^{\mathbb{S}}\|_2)}{\sqrt{\kappa_s}\,\|\mathbf{v}^{\mathbb{S}}\|_2}\,\mathbf{v}^{\mathbb{S}}$$

- Euclidean component:
$$\exp_{\mathbf{x}^{\mathbb{R}}}(\mathbf{v}^{\mathbb{R}}) = \mathbf{x}^{\mathbb{R}} + \mathbf{v}^{\mathbb{R}}$$

Product: $\exp_\mathbf{x}(\mathbf{v}) = (\exp_{\mathbf{x}^{\mathbb{H}}}(\mathbf{v}^{\mathbb{H}}),\, \exp_{\mathbf{x}^{\mathbb{S}}}(\mathbf{v}^{\mathbb{S}}),\, \exp_{\mathbf{x}^{\mathbb{R}}}(\mathbf{v}^{\mathbb{R}}))$.

**Convention**: When $\mathbf{v} = \mathbf{0}$, $\exp_\mathbf{x}(\mathbf{0}) = \mathbf{x}$ (in each component, $\sinh(0)/0$ and $\sin(0)/0$ are taken as the limit value $1$).

**Proposition 4.1 (Inverse Property).** Within the injectivity radius (hyperbolic and Euclidean: $\infty$; spherical: $\pi/\sqrt{\kappa_s}$), $\exp_\mathbf{x} \circ \log_\mathbf{x} = \mathrm{id}_{\mathcal{M}}$ and $\log_\mathbf{x} \circ \exp_\mathbf{x} = \mathrm{id}_{T_\mathbf{x}\mathcal{M}}$.

### 4.5 Tangent Space Riemannian Norm

**Definition 4.6 (Tangent Space Riemannian Norm).** For $\mathbf{v} = (\mathbf{v}^{\mathbb{H}}, \mathbf{v}^{\mathbb{S}}, \mathbf{v}^{\mathbb{R}}) \in T_\mathbf{x}\mathcal{M}$:
$$\|\mathbf{v}\|_{T_\mathbf{x}\mathcal{M}} = \sqrt{\|\mathbf{v}^{\mathbb{H}}\|_{\mathrm{L}}^2 + \|\mathbf{v}^{\mathbb{S}}\|_2^2 + \|\mathbf{v}^{\mathbb{R}}\|_2^2}$$

---

## 5. Model Architecture: RieFormer

### 5.1 Overall Definition

**Definition 5.1 (RieFormer).** RieFormer is a mapping with parameters $\theta$:
$$f_\theta(\mathbf{X}_t, \mathbf{E}_t, t, \mathbf{C}_{\mathcal{V}}, \mathbf{C}_{\mathcal{R}}, \mathbf{m}) = (\hat{\mathbf{V}}, \hat{\mathbf{P}})$$

Inputs: $\mathbf{X}_t \in \mathcal{M}^N$, $\mathbf{E}_t \in \{0,1\}^{N \times N \times K}$, $t \in [0,1]$, $\mathbf{C}_{\mathcal{V}} \in \mathbb{R}^{N \times d_c}$, $\mathbf{C}_{\mathcal{R}} \in \mathbb{R}^{K \times d_c}$, $\mathbf{m} \in \{0,1\}^N$.

Outputs:
- $\hat{\mathbf{v}}_i \in T_{\mathbf{x}_{t,i}}\mathcal{M}$, the predicted tangent vector for node $i$, $i \in [N]$;
- $\hat{\mathbf{P}} \in [0,1]^{N \times N \times K}$, where $\hat{\mathbf{P}}_{ij}^{(k)}$ is the predicted edge type probability.

Let $\mathbf{X}_t = (\mathbf{x}_{t,1}, \ldots, \mathbf{x}_{t,N}) \in \mathcal{M}^N$, where $\mathbf{x}_{t,i} \in \mathcal{M}$ is the manifold coordinate of node $i$ at time $t$, obtained via geodesic interpolation from noise $\mathbf{x}_{0,i}$ to data $\mathbf{x}_{1,i}$.

$\theta$ includes all learnable parameters, including curvatures $\kappa_h, \kappa_s$.

### 5.2 Input Encoding Layer

**Notation**: $[\mathbf{a} \| \mathbf{b}]$ denotes vector concatenation.

**Definition 5.2 (Coordinate Projection).** $\pi: \mathcal{M} \to \mathbb{R}^D$:
$$\pi(\mathbf{x}) = [\mathbf{x}^{\mathbb{H}} \| \mathbf{x}^{\mathbb{S}} \| \mathbf{x}^{\mathbb{R}}]$$

**Definition 5.3 (Node Initial Embedding).**
$$\mathbf{h}_i^{V,(0)} = \mathrm{MLP}_{\mathrm{node}}\!\left([\pi(\mathbf{x}_{t,i}) \| \mathbf{c}_i \| m_i]\right) \in \mathbb{R}^{d_v}$$
where the input dimension is $D + d_c + 1$, and $d_v$ is the node hidden dimension.

**Definition 5.4 (Edge Initial Embedding).**
$$\mathbf{h}_{ij}^{E,(0)} = \mathrm{MLP}_{\mathrm{edge}}\!\left([\mathbf{E}_{t,ij}\mathbf{W}_{\mathrm{rel}} \| \mathbf{E}_{t,ij}\mathbf{C}_{\mathcal{R}}]\right) \in \mathbb{R}^{d_{e'}}$$
where $\mathbf{W}_{\mathrm{rel}} \in \mathbb{R}^{K \times d_r}$ is a learnable relation embedding matrix, $\mathbf{E}_{t,ij}\mathbf{W}_{\mathrm{rel}} \in \mathbb{R}^{d_r}$ is the sum of active relation type embeddings, and $\mathbf{E}_{t,ij}\mathbf{C}_{\mathcal{R}} \in \mathbb{R}^{d_c}$ is the sum of active relation type text embeddings. Input dimension is $d_r + d_c$.

**Definition 5.5 (Time Embedding).** Let $d_t \in \mathbb{Z}_{>0}$ be even; sinusoidal positional encoding:
$$\boldsymbol{\psi}(t) = [\sin(\omega_1 t),\, \cos(\omega_1 t),\, \ldots,\, \sin(\omega_{d_t/2} t),\, \cos(\omega_{d_t/2} t)] \in \mathbb{R}^{d_t}$$
where $\omega_l = 10000^{-2l/d_t}$, $l \in [d_t/2]$. Linear projection:
$$\mathbf{t}_{\mathrm{emb}} = \mathbf{W}_t \boldsymbol{\psi}(t) + \mathbf{b}_t \in \mathbb{R}^{d_v}$$
where $\mathbf{W}_t \in \mathbb{R}^{d_v \times d_t}$, $\mathbf{b}_t \in \mathbb{R}^{d_v}$.

### 5.3 RieFormer Block

RieFormer consists of $L \in \mathbb{Z}_{>0}$ identically structured blocks stacked together. Layer $l$ ($l \in [L]$) receives node embeddings $\mathbf{H}^{V,(l-1)} \in \mathbb{R}^{N \times d_v}$ and edge embeddings $\mathbf{H}^{E,(l-1)} \in \mathbb{R}^{N \times N \times d_{e'}}$, and outputs $\mathbf{H}^{V,(l)} \in \mathbb{R}^{N \times d_v}$ and $\mathbf{H}^{E,(l)} \in \mathbb{R}^{N \times N \times d_{e'}}$.

Each block executes five submodules in the following order:

- A. Manifold-Aware Multi-Head Attention (Manifold RoPE + Geodesic Kernel) → updates node embeddings
- B. ATH-Norm → normalizes node embeddings
- C. Edge Stream Self-Update → updates edge embeddings
- D. Bidirectional Cross-Interaction → mutual injection between node and edge embeddings
- E. Text Conditioning Injection → injects text semantics via cross-attention

#### 5.3.1 Submodule A: Manifold-Aware Multi-Head Attention

#### 5.3.1 Submodule A: Manifold-Aware Multi-Head Attention

Let $n_h \in \mathbb{Z}_{>0}$ be the number of attention heads, $d_{\mathrm{head}} = d_v / n_h$ (requiring $n_h \mid d_v$ and $2 \mid d_{\mathrm{head}}$).

**Definition 5.6 (Manifold RoPE).** For the $s$-th attention head ($s \in [n_h]$), frequencies $\omega_l^{(s)} = 10000^{-2l/d_{\mathrm{head}}}$, $l \in [d_{\mathrm{head}}/2]$. For a node pair $(i, j)$, define the angle:
$$\theta_{ij,l}^{(s)} = \omega_l^{(s)} \cdot d_{\mathcal{M}}(\mathbf{x}_{t,i}, \mathbf{x}_{t,j})$$

The rotation matrix $\mathbf{R}(\boldsymbol{\theta}_{ij}^{(s)}) \in \mathbb{R}^{d_{\mathrm{head}} \times d_{\mathrm{head}}}$ is a block-diagonal matrix composed of $d_{\mathrm{head}}/2$ rotation blocks of size $2 \times 2$:
$$\mathbf{R}(\boldsymbol{\theta}_{ij}^{(s)}) = \mathrm{diag}\!\left(\begin{pmatrix} \cos\theta_{ij,1}^{(s)} & -\sin\theta_{ij,1}^{(s)} \\ \sin\theta_{ij,1}^{(s)} & \cos\theta_{ij,1}^{(s)} \end{pmatrix}, \ldots, \begin{pmatrix} \cos\theta_{ij,d_{\mathrm{head}}/2}^{(s)} & -\sin\theta_{ij,d_{\mathrm{head}}/2}^{(s)} \\ \sin\theta_{ij,d_{\mathrm{head}}/2}^{(s)} & \cos\theta_{ij,d_{\mathrm{head}}/2}^{(s)} \end{pmatrix}\right)$$

**Definition 5.7 (Geodesic Kernel).** For the $s$-th attention head:
$$\kappa^{(s)}(\mathbf{x}_{t,i}, \mathbf{x}_{t,j}) = w_{\mathbb{H}}^{(s)} \kappa_{\mathbb{H}}(\mathbf{x}_{t,i}^{\mathbb{H}}, \mathbf{x}_{t,j}^{\mathbb{H}}) + w_{\mathbb{S}}^{(s)} \kappa_{\mathbb{S}}(\mathbf{x}_{t,i}^{\mathbb{S}}, \mathbf{x}_{t,j}^{\mathbb{S}}) + w_{\mathbb{R}}^{(s)} \kappa_{\mathbb{R}}(\mathbf{x}_{t,i}^{\mathbb{R}}, \mathbf{x}_{t,j}^{\mathbb{R}})$$
where:
- $\kappa_{\mathbb{H}}(\mathbf{a}, \mathbf{b}) = -d_{\mathbb{H}}(\mathbf{a}, \mathbf{b})$
- $\kappa_{\mathbb{S}}(\mathbf{a}, \mathbf{b}) = \kappa_s \cdot \mathbf{a}^\top \mathbf{b}$
- $\kappa_{\mathbb{R}}(\mathbf{a}, \mathbf{b}) = -\|\mathbf{a} - \mathbf{b}\|_2^2$

$w_{\mathbb{H}}^{(s)}, w_{\mathbb{S}}^{(s)}, w_{\mathbb{R}}^{(s)} \in \mathbb{R}$ are per-head learnable weights.

**Definition 5.8 (Manifold-Aware Attention).** Query/key/value projections ($s \in [n_h]$):
$$\mathbf{q}_i^{(s)} = \mathbf{W}_Q^{(s)} \mathbf{h}_i^{V,(l-1)},\quad \mathbf{k}_j^{(s)} = \mathbf{W}_K^{(s)} \mathbf{h}_j^{V,(l-1)},\quad \mathbf{v}_j^{(s)} = \mathbf{W}_V^{(s)} \mathbf{h}_j^{V,(l-1)} \in \mathbb{R}^{d_{\mathrm{head}}}$$
where $\mathbf{W}_Q^{(s)}, \mathbf{W}_K^{(s)}, \mathbf{W}_V^{(s)} \in \mathbb{R}^{d_{\mathrm{head}} \times d_v}$.

Attention score:
$$a_{ij}^{(s)} = \frac{(\mathbf{R}(\boldsymbol{\theta}_{ij}^{(s)})\mathbf{q}_i^{(s)})^\top \mathbf{k}_j^{(s)}}{\sqrt{d_{\mathrm{head}}}} + \beta^{(s)} \kappa^{(s)}(\mathbf{x}_{t,i}, \mathbf{x}_{t,j}) + \mathbf{w}_b^{(s)\top} \mathbf{h}_{ij}^{E,(l-1)}$$
where $\beta^{(s)} \in \mathbb{R}$ is a learnable scaling coefficient, and $\mathbf{w}_b^{(s)} \in \mathbb{R}^{d_{e'}}$ is the edge bias weight.

Attention weights: $\alpha_{ij}^{(s)} = \mathrm{softmax}_j(a_{ij}^{(s)})$.

Euclidean aggregation:
$$\mathbf{o}_i^{(s)} = \sum_{j=1}^N \alpha_{ij}^{(s)} \cdot \mathbf{v}_j^{(s)} \in \mathbb{R}^{d_{\mathrm{head}}}$$

Multi-head concatenation and projection:
$$\mathrm{MHA}_i = \mathbf{W}_O [\mathbf{o}_i^{(1)} \| \cdots \| \mathbf{o}_i^{(n_h)}] \in \mathbb{R}^{d_v}$$
where $\mathbf{W}_O \in \mathbb{R}^{d_v \times d_v}$.

Residual connection: $\tilde{\mathbf{h}}_i^V = \mathbf{h}_i^{V,(l-1)} + \mathrm{MHA}_i$.

#### 5.3.3 Submodule C: Edge Stream Self-Update

Edge embeddings are independently updated via factorized attention: for edge $(i, j)$, information is aggregated separately from the other outgoing edges of head node $i$ and the other incoming edges of tail node $j$.

**Definition 5.11 (Factorized Edge Attention).** For edge $(i, j)$:

Head-side aggregation (other outgoing edges of $i$):
$$\mathbf{g}_{ij}^{\mathrm{head}} = \sum_{p \in [N] \setminus \{j\}} \gamma_{ip \to ij}^{\mathrm{head}} \cdot \mathbf{W}_{\mathrm{Ev}}^{\mathrm{head}} \mathbf{h}_{ip}^{E,(l-1)}$$

Tail-side aggregation (other incoming edges of $j$):
$$\mathbf{g}_{ij}^{\mathrm{tail}} = \sum_{p \in [N] \setminus \{i\}} \gamma_{pj \to ij}^{\mathrm{tail}} \cdot \mathbf{W}_{\mathrm{Ev}}^{\mathrm{tail}} \mathbf{h}_{pj}^{E,(l-1)}$$

Attention weights:
$$\gamma_{ip \to ij}^{\mathrm{head}} = \mathrm{softmax}_{p}\!\left(\frac{(\mathbf{W}_{\mathrm{Eq}}^{\mathrm{head}} \mathbf{h}_{ij}^{E,(l-1)})^\top (\mathbf{W}_{\mathrm{Ek}}^{\mathrm{head}} \mathbf{h}_{ip}^{E,(l-1)})}{\sqrt{d_{e'}}}\right)$$
The tail-side $\gamma_{pj \to ij}^{\mathrm{tail}}$ is defined analogously, using independent weight matrices $\mathbf{W}_{\mathrm{Eq}}^{\mathrm{tail}}, \mathbf{W}_{\mathrm{Ek}}^{\mathrm{tail}}$.

All weight matrices $\mathbf{W}_{\mathrm{Eq}}^{\mathrm{head}}, \mathbf{W}_{\mathrm{Ek}}^{\mathrm{head}}, \mathbf{W}_{\mathrm{Ev}}^{\mathrm{head}}, \mathbf{W}_{\mathrm{Eq}}^{\mathrm{tail}}, \mathbf{W}_{\mathrm{Ek}}^{\mathrm{tail}}, \mathbf{W}_{\mathrm{Ev}}^{\mathrm{tail}} \in \mathbb{R}^{d_{e'} \times d_{e'}}$.

Residual update:
$$\tilde{\mathbf{h}}_{ij}^E = \mathbf{h}_{ij}^{E,(l-1)} + \mathrm{MLP}_{\mathrm{E\text{-}up}}\!\left([\mathbf{h}_{ij}^{E,(l-1)} \| \mathbf{g}_{ij}^{\mathrm{head}} \| \mathbf{g}_{ij}^{\mathrm{tail}}]\right)$$
where $\mathrm{MLP}_{\mathrm{E\text{-}up}}: \mathbb{R}^{3d_{e'}} \to \mathbb{R}^{d_{e'}}$.

Followed by LayerNorm: $\bar{\mathbf{h}}_{ij}^E = \mathrm{LN}(\tilde{\mathbf{h}}_{ij}^E)$.

#### 5.3.4 Submodule D: Bidirectional Cross-Interaction

**Definition 5.12 (Edge→Node Aggregation).** For node $i$, aggregate its associated edge embeddings:
$$\hat{\mathbf{h}}_i^V = \bar{\mathbf{h}}_i^V + \mathrm{MLP}_{E \to V}\!\left(\sum_{j=1}^N \alpha_{ij}^{E \to V} \cdot \mathbf{W}_{\mathrm{Ev2n}} \bar{\mathbf{h}}_{ij}^E\right)$$
where $\mathbf{W}_{\mathrm{Ev2n}} \in \mathbb{R}^{d_v \times d_{e'}}$, $\mathrm{MLP}_{E \to V}: \mathbb{R}^{d_v} \to \mathbb{R}^{d_v}$. Attention weights:
$$\alpha_{ij}^{E \to V} = \mathrm{softmax}_j\!\left(\frac{(\mathbf{W}_Q^{E \to V} \bar{\mathbf{h}}_i^V)^\top (\mathbf{W}_K^{E \to V} \bar{\mathbf{h}}_{ij}^E)}{\sqrt{d_v}}\right)$$
where $\mathbf{W}_Q^{E \to V} \in \mathbb{R}^{d_v \times d_v}$, $\mathbf{W}_K^{E \to V} \in \mathbb{R}^{d_v \times d_{e'}}$.

**Definition 5.13 (Node→Edge Injection).** For edge $(i, j)$:
$$\hat{\mathbf{h}}_{ij}^E = \bar{\mathbf{h}}_{ij}^E + \mathrm{MLP}_{V \to E}\!\left([\hat{\mathbf{h}}_i^V \| \hat{\mathbf{h}}_j^V \| \hat{\mathbf{h}}_i^V \odot \hat{\mathbf{h}}_j^V]\right)$$
where $\mathrm{MLP}_{V \to E}: \mathbb{R}^{3d_v} \to \mathbb{R}^{d_{e'}}$.

#### 5.3.5 Submodule E: Text Conditioning Injection

**Definition 5.14 (Text Cross-Attention).** For node $i$, using the node embedding as query and text conditioning as key/value:
$$\mathbf{q}_i^{\mathrm{text}} = \mathbf{W}_Q^{\mathrm{text}} \hat{\mathbf{h}}_i^V, \quad \mathbf{k}_j^{\mathrm{text}} = \mathbf{W}_K^{\mathrm{text}} \mathbf{c}_j, \quad \mathbf{v}_j^{\mathrm{text}} = \mathbf{W}_V^{\mathrm{text}} \mathbf{c}_j$$
where $\mathbf{W}_Q^{\mathrm{text}} \in \mathbb{R}^{d_v \times d_v}$, $\mathbf{W}_K^{\mathrm{text}}, \mathbf{W}_V^{\mathrm{text}} \in \mathbb{R}^{d_v \times d_c}$.

Cross-attention:
$$\mathrm{CrossAttn}_i = \sum_{j=1}^N \mathrm{softmax}_j\!\left(\frac{\mathbf{q}_i^{\mathrm{text}\top} \mathbf{k}_j^{\mathrm{text}}}{\sqrt{d_v}}\right) \cdot \mathbf{v}_j^{\mathrm{text}}$$

**Definition 5.15 (Layer $l$ Output).**
- Node: $\mathbf{h}_i^{V,(l)} = \mathrm{LN}(\hat{\mathbf{h}}_i^V + \mathrm{CrossAttn}_i)$
- Edge: $\mathbf{h}_{ij}^{E,(l)} = \hat{\mathbf{h}}_{ij}^E$

### 5.4 Output Projection Layer

#### 5.4.1 Vector Field Output Head

**Definition 5.16 (Tangent Space Projection).** For any $\hat{\mathbf{u}} \in \mathbb{R}^D$, split it in reverse order of the coordinate projection $\pi$ into $\hat{\mathbf{u}}^{\mathbb{H}} \in \mathbb{R}^{d_h+1}$, $\hat{\mathbf{u}}^{\mathbb{S}} \in \mathbb{R}^{d_s+1}$, $\hat{\mathbf{u}}^{\mathbb{R}} \in \mathbb{R}^{d_e}$. Project onto $T_{\mathbf{x}_t}\mathcal{M}$:
$$\hat{\mathbf{v}}^{\mathbb{H}} = \hat{\mathbf{u}}^{\mathbb{H}} - \kappa_h \langle \hat{\mathbf{u}}^{\mathbb{H}}, \mathbf{x}_t^{\mathbb{H}} \rangle_{\mathrm{L}} \cdot \mathbf{x}_t^{\mathbb{H}} \in T_{\mathbf{x}_t^{\mathbb{H}}}\mathbb{H}$$
$$\hat{\mathbf{v}}^{\mathbb{S}} = \hat{\mathbf{u}}^{\mathbb{S}} - \kappa_s (\mathbf{x}_t^{\mathbb{S}\top} \hat{\mathbf{u}}^{\mathbb{S}}) \cdot \mathbf{x}_t^{\mathbb{S}} \in T_{\mathbf{x}_t^{\mathbb{S}}}\mathbb{S}$$
$$\hat{\mathbf{v}}^{\mathbb{R}} = \hat{\mathbf{u}}^{\mathbb{R}} \in \mathbb{R}^{d_e}$$

**Definition 5.17 (Vector Field Prediction).** For node $i$:
$$\hat{\mathbf{u}}_i = \mathrm{MLP}_{\mathrm{vec}}(\mathbf{h}_i^{V,(L)}) \in \mathbb{R}^D$$
where $\mathrm{MLP}_{\mathrm{vec}}: \mathbb{R}^{d_v} \to \mathbb{R}^D$. After projection via Definition 5.16, we obtain $\hat{\mathbf{v}}_i = (\hat{\mathbf{v}}_i^{\mathbb{H}}, \hat{\mathbf{v}}_i^{\mathbb{S}}, \hat{\mathbf{v}}_i^{\mathbb{R}}) \in T_{\mathbf{x}_{t,i}}\mathcal{M}$.

#### 5.4.2 Edge Type Output Head

**Definition 5.18 (Relation Interaction Layer).** For edge $(i, j)$ and each relation type $k \in [K]$, construct the candidate relation feature:
$$\mathbf{r}_{ij}^{(k)} = \mathrm{MLP}_{\mathrm{rel\text{-}proj}}\!\left([\mathbf{h}_{ij}^{E,(L)} \| \mathbf{c}_{r_k}]\right) \in \mathbb{R}^{d_{e'}}$$
where $\mathrm{MLP}_{\mathrm{rel\text{-}proj}}: \mathbb{R}^{d_{e'} + d_c} \to \mathbb{R}^{d_{e'}}$.

The $K$ candidate features $\{\mathbf{r}_{ij}^{(1)}, \ldots, \mathbf{r}_{ij}^{(K)}\}$ interact via a relation Transformer (self-attention over the relation dimension) to produce $\tilde{\mathbf{r}}_{ij}^{(k)} \in \mathbb{R}^{d_{e'}}$, $k \in [K]$.

**Definition 5.19 (Edge Type Probability).**
$$\hat{\mathbf{P}}_{ij}^{(k)} = \sigma(\mathbf{w}_{\mathrm{cls}}^\top \tilde{\mathbf{r}}_{ij}^{(k)} + b_{\mathrm{cls}}) \in [0,1]$$
where $\sigma(\cdot)$ is the sigmoid function, $\mathbf{w}_{\mathrm{cls}} \in \mathbb{R}^{d_{e'}}$, $b_{\mathrm{cls}} \in \mathbb{R}$. Each relation type is predicted independently (corresponding to multi-hot labels).

### 5.5 Permutation Equivariance

**Proposition 5.1 (Permutation Equivariance).** For any permutation matrix $\boldsymbol{\Pi} \in \{0,1\}^{N \times N}$:
$$f_\theta(\boldsymbol{\Pi}\mathbf{X}_t, \boldsymbol{\Pi}\mathbf{E}_t\boldsymbol{\Pi}^\top, t, \boldsymbol{\Pi}\mathbf{C}_{\mathcal{V}}, \mathbf{C}_{\mathcal{R}}, \boldsymbol{\Pi}\mathbf{m}) = (\boldsymbol{\Pi}\hat{\mathbf{V}}, \boldsymbol{\Pi}\hat{\mathbf{P}}\boldsymbol{\Pi}^\top)$$

where $\mathbf{C}_{\mathcal{R}}$ and $t$ are unaffected by the permutation.

---

## 6. Joint Discrete-Continuous Flow Matching

**Convention**: $t = 0$ corresponds to the noise end, $t = 1$ corresponds to the data end. $\mathbf{X}_1, \mathbf{E}_1$ are training data (from subgraphs), and $\mathbf{X}_0, \mathbf{E}_0$ are noise samples.

### 6.1 Noise Prior Distribution

**Definition 6.1 (Continuous Noise Prior).** For each node $i \in [N]$, each component is sampled independently:
- Hyperbolic component: $\mathbf{x}_{0,i}^{\mathbb{H}} \sim \mathrm{Uniform}(B_{\mathbb{H}}(\mathbf{x}_\varnothing^{\mathbb{H}}, R_{\mathbb{H}}))$, i.e., uniformly sampled with respect to the Riemannian volume measure within a geodesic ball centered at the anchor $\mathbf{x}_\varnothing^{\mathbb{H}}$ with geodesic radius $R_{\mathbb{H}} > 0$;
- Spherical component: $\mathbf{x}_{0,i}^{\mathbb{S}} \sim \mathrm{Uniform}(\mathbb{S}^{d_s}_{\kappa_s})$, i.e., the uniform distribution with respect to the spherical volume measure;
- Euclidean component: $\mathbf{x}_{0,i}^{\mathbb{R}} \sim \mathcal{N}(\mathbf{0}, \sigma_0^2 \mathbf{I}_{d_e})$.

where $R_{\mathbb{H}} > 0$ and $\sigma_0 > 0$ are hyperparameters. The continuous noise prior is denoted $p_0^{\mathcal{M}}$.

**Definition 6.2 (Discrete Noise Prior).** For each pair $(i, j) \in [N]^2$ and each relation type $k \in [K]$, independently sample:
$$\mathbf{E}_{0,ij}^{(k)} \sim \mathrm{Bernoulli}(\rho_k)$$

where the marginal frequency is:
$$\rho_k = \frac{\sum_{\mathcal{G} \in \mathcal{D}} \sum_{i=1}^N \sum_{j=1}^N \mathbf{E}_{1,ij}^{(k)}}{|\mathcal{D}| \cdot N^2}$$

i.e., the average occurrence frequency of relation type $r_k$ in the training set (virtual node edges are zero and do not affect the numerator).

### 6.2 Forward Interpolation Process

**Definition 6.3 (Continuous Interpolation).** For each node $i \in [N]$, interpolate along the geodesic between $\mathbf{x}_{0,i}$ (noise) and $\mathbf{x}_{1,i}$ (data):
$$\mathbf{x}_{t,i} = \exp_{\mathbf{x}_{0,i}}\!\left(t \cdot \log_{\mathbf{x}_{0,i}}(\mathbf{x}_{1,i})\right) \in \mathcal{M}, \quad t \in [0, 1]$$

Boundary conditions: $\mathbf{x}_{0,i} = \mathbf{x}_{0,i}$ (noise), $\mathbf{x}_{1,i} = \mathbf{x}_{1,i}$ (data).

**Definition 6.4 (Discrete Interpolation).** For each pair $(i, j) \in [N]^2$, independently sample a mask:
$$z_{ij} \sim \mathrm{Bernoulli}(t)$$
$$\mathbf{E}_{t,ij} = z_{ij} \cdot \mathbf{E}_{1,ij} + (1 - z_{ij}) \cdot \mathbf{E}_{0,ij}$$

All $K$ relation types for the same pair $(i, j)$ share the same $z_{ij}$, i.e., they are taken entirely from the data end or the noise end. Different directions $z_{ij}$ and $z_{ji}$ are sampled independently.

### 6.3 Conditional Targets

**Definition 6.5 (Conditional Vector Field Target).** For each node $i \in [N]$:
$$\mathbf{u}_{t,i} = \frac{1}{1-t}\log_{\mathbf{x}_{t,i}}(\mathbf{x}_{1,i}) \in T_{\mathbf{x}_{t,i}}\mathcal{M}$$

This is the tangent vector pointing from the current position $\mathbf{x}_{t,i}$ toward the data point $\mathbf{x}_{1,i}$, scaled by the remaining time $1 - t$ to form a velocity: following this velocity for time $1 - t$ exactly reaches $\mathbf{x}_{1,i}$. During training, $t \sim \mathrm{Uniform}(0, 1 - \epsilon_t)$, where $\epsilon_t > 0$ is a small constant to avoid the singularity as $t \to 1$.

**Definition 6.6 (Edge Type Target).** For each pair $(i, j) \in [N]^2$ and each relation type $k \in [K]$:
$$\mathbf{E}_{1,ij}^{(k)} \in \{0,1\}$$
i.e., the ground-truth edge type label from the data end.

### 6.4 Training Loss Function

**Definition 6.7 (Manifold Vector Field Loss).**
$$\mathcal{L}_{\mathrm{cont}} = \frac{1}{N}\sum_{i=1}^N m_i \cdot \|\hat{\mathbf{v}}_i - \mathbf{u}_{t,i}\|_{T_{\mathbf{x}_{t,i}}\mathcal{M}}^2$$

where $\|\cdot\|_{T_{\mathbf{x}_{t,i}}\mathcal{M}}$ is the tangent space Riemannian norm from Definition 4.6. Virtual nodes ($m_i = 0$) do not contribute to the loss.

**Definition 6.8 (Edge Type Loss).** Let $\mathcal{S} \subseteq [N]^2$ be the sampled set of edge pairs.
$$\mathcal{L}_{\mathrm{disc}} = \frac{1}{|\mathcal{S}|}\sum_{(i,j) \in \mathcal{S}}\sum_{k=1}^K \mathrm{BCE}_{\mathrm{w}}(\hat{\mathbf{P}}_{ij}^{(k)}, \mathbf{E}_{1,ij}^{(k)})$$

where the weighted binary cross-entropy is:
$$\mathrm{BCE}_{\mathrm{w}}(\hat{p}, y) = -\left[w_k^+ \cdot y \cdot \log(\hat{p} + \epsilon_p) + (1 - y) \cdot \log(1 - \hat{p} + \epsilon_p)\right]$$

The positive sample weight $w_k^+ = \min\!\left(\frac{1 - \rho_k}{\rho_k},\, w_{\max}\right)$ corrects for class imbalance, $w_{\max} > 0$ is the truncation upper bound, and $\epsilon_p > 0$ is a numerical stability constant.

**Remark 6.8a (Softmax-CE tightening for single-hot KGs).** If the dataset satisfies $\max_{(i,j)} \sum_{k=1}^K \mathbf{E}_{1,ij}^{(k)} = 1$ (at most one relation per pair, e.g. wikidata_5m), the per-channel BCE in Definition 6.8 decouples into $K$ independent binary classifiers: on every positive pair only 1 channel is positive while the other $K - 1$ are easy-negatives, diluting the gradient by $O(1/K)$ and pinning $\mathcal{L}_{\mathrm{disc}}$ near zero for large $K$ (wikidata_5m has $K \approx 822$). An equivalent tightened form applies $K$-way softmax-CE on positive pairs while retaining zero-target BCE on negatives to preserve the "no edge" signal:
$$\mathcal{L}_{\mathrm{disc}}^{\mathrm{sm}} = \frac{1}{|\mathcal{S}^+|}\sum_{(i,j) \in \mathcal{S}^+}\!\Bigl(-\sum_{k=1}^K \tilde{\mathbf{E}}_{1,ij}^{(k)} \log \mathrm{softmax}(\hat{\mathbf{P}}_{ij})^{(k)}\Bigr) + \frac{1}{|\mathcal{S}^-|}\sum_{(i,j) \in \mathcal{S}^-}\sum_{k=1}^K \mathrm{softplus}(\hat{\mathbf{P}}_{ij}^{(k)})$$
where $\tilde{\mathbf{E}}_{1,ij} = \mathbf{E}_{1,ij} / \sum_k \mathbf{E}_{1,ij}^{(k)}$ normalises positive-pair labels to a probability distribution (one-hot under single-hot, uniform split under multi-hot) and $\hat{\mathbf{P}}_{ij}$ denotes predicted logits. The flag `edge_loss_mode: softmax_ce` selects this formulation; datasets with non-trivial multi-hot fractions (e.g. fb15k_237 at ~9.5%) should keep the default `bce` to avoid truncating multi-hot labels.

**Definition 6.9 (Graph-Text Contrastive Loss).** Project backbone hidden states into the alignment space. Using raw data-end manifold coordinates $\mathbf{x}_{1,i}$ directly causes degenerate contrastive loss: entity embeddings initialise near the manifold origin and remain directionally collapsed (cosine similarity $> 0.99$) during early training, making InfoNCE unable to distinguish nodes (loss $\equiv \log M$). We therefore use the final hidden state $\mathbf{h}_i$ from the RieFormer backbone as the graph-side input — $\mathbf{h}_i$ integrates noised coordinates, edge types, and text cross-attention, providing sufficient inter-sample variance from the start:
$$\mathbf{g}_i = \mathrm{MLP}_{\mathrm{proj}}(\mathbf{h}_i) \in \mathbb{R}^{d_\mathrm{a}}$$
where $\mathbf{h}_i \in \mathbb{R}^{d_{\mathrm{node}}}$ is the node hidden state output by the backbone (Definition 5.1), $\mathrm{MLP}_{\mathrm{proj}}: \mathbb{R}^{d_{\mathrm{node}}} \to \mathbb{R}^{d_\mathrm{a}}$, and $d_\mathrm{a}$ is the alignment space dimension.

Cosine similarity: $\mathrm{sim}(\mathbf{g}_i, \mathbf{c}_j) = \frac{\mathbf{g}_i^\top \mathbf{c}_j}{\|\mathbf{g}_i\|_2 \|\mathbf{c}_j\|_2}$

Let $\mathcal{B} \subseteq [N]$ be the set of node indices in the mini-batch; the symmetric contrastive loss is:
$$\mathcal{L}_{\mathrm{align}} = \frac{1}{2}\left(\mathcal{L}_{\mathrm{align}}^{g \to c} + \mathcal{L}_{\mathrm{align}}^{c \to g}\right)$$
where:
$$\mathcal{L}_{\mathrm{align}}^{g \to c} = -\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \log \frac{\exp(\mathrm{sim}(\mathbf{g}_i, \mathbf{c}_i) / \tau)}{\sum_{j \in \mathcal{B}} \exp(\mathrm{sim}(\mathbf{g}_i, \mathbf{c}_j) / \tau)}$$
$\mathcal{L}_{\mathrm{align}}^{c \to g}$ is defined symmetrically. $\tau > 0$ is the temperature hyperparameter.

**Definition 6.9a (Node Three-Way Partition).** During pretraining, for each subgraph the real-node index set $\mathcal{V}_{\mathrm{real}} \subseteq [N]$ is split into three disjoint subsets at ratios $(p_c, p_x)$:

$$\mathcal{V}_{\mathrm{real}} = \mathcal{U} \sqcup \mathcal{M}_c \sqcup \mathcal{M}_x$$

with $|\mathcal{M}_c| \approx p_c |\mathcal{V}_{\mathrm{real}}|$ (**semantic-mask** subset), $|\mathcal{M}_x| \approx p_x |\mathcal{V}_{\mathrm{real}}|$ (**geometric-mask** subset), and $|\mathcal{U}| \geq 1$ (at least one REAL anchor).  Virtual nodes form $\mathcal{V}_{\mathrm{virt}} = [N] \setminus \mathcal{V}_{\mathrm{real}}$.

**Per-node state**:

| Subset | Geometry $\mathbf{x}_{t,i}$ | Text $\bar{\mathbf{c}}_i$ | Time label $t_i$ | Contributes to |
|------|-----|-----|-----|-----|
| $\mathcal{U}$ (REAL) | Geodesic interpolation (flow) | Real $\bar{\mathbf{c}}_i$ | Subgraph batch-$t$ | $\mathcal{L}_{\mathrm{cont}}$, $\mathcal{L}_{\mathrm{align}}$ |
| $\mathcal{M}_c$ (semantic) | $\mathbf{x}_{1,i}$ held fixed | Learnable $\mathbf{e}_{\mathrm{mask}} \in \mathbb{R}^{d_c}$ | $t_i = 1$ | $\mathcal{L}_{\mathrm{mask}\_c}$ |
| $\mathcal{M}_x$ (geometric) | $\mathbf{x}_{0,i}$ (pure noise) | Real $\bar{\mathbf{c}}_i$ | $t_i = 0$ | $\mathcal{L}_{\mathrm{mask}\_x}$ |

Each masked node loses **one** modality while the other serves as an anchor, preventing collapse of the backbone representation $\mathbf{h}_i$.

**Definition 6.10a (Semantic Mask Identification Loss $\mathcal{L}_{\mathrm{mask}\_c}$).** For $i \in \mathcal{M}_c$, geometry is preserved ($t_i = 1 \Rightarrow \mathbf{x}_{t,i} = \mathbf{x}_{1,i}$) and text is replaced by a single learnable vector $\mathbf{e}_{\mathrm{mask}} \in \mathbb{R}^{d_c}$.  Using the original text $\bar{\mathbf{c}}_i$ as the contrastive target, the projection head $\mathrm{MLP}_{\mathrm{mask}\_c}: \mathbb{R}^{d_v} \to \mathbb{R}^{d_c}$ computes:

$$\mathbf{p}_i = \mathrm{MLP}_{\mathrm{mask}\_c}(\mathbf{h}_i) \in \mathbb{R}^{d_c}$$

Symmetric InfoNCE over in-batch $\mathcal{M}_c$ nodes:

$$\mathcal{L}_{\mathrm{mask}\_c} = \frac{1}{2}\!\left(\mathcal{L}_{\mathrm{mask}\_c}^{p \to c} + \mathcal{L}_{\mathrm{mask}\_c}^{c \to p}\right), \quad \mathcal{L}_{\mathrm{mask}\_c}^{p \to c} = -\frac{1}{|\mathcal{M}_c|}\sum_{i \in \mathcal{M}_c} \log \frac{\exp(\mathrm{sim}(\mathbf{p}_i, \bar{\mathbf{c}}_i)/\tau_{\mathrm{mask}\_c})}{\sum_{j \in \mathcal{M}_c} \exp(\mathrm{sim}(\mathbf{p}_i, \bar{\mathbf{c}}_j)/\tau_{\mathrm{mask}\_c})}$$

Set to $0$ when $|\mathcal{M}_c| < 2$.  Mirrors the T2G / GAD downstream direction (identify entity from its manifold position).

**Definition 6.10b (Geometric Mask Reconstruction Loss $\mathcal{L}_{\mathrm{mask}\_x}$).** For $i \in \mathcal{M}_x$, $t_i = 0$ forces $\mathbf{x}_{t,i} = \mathbf{x}_{0,i}$ (pure noise) while the text $\bar{\mathbf{c}}_i$ remains real.  The vector-field target is Definition 6.5 at $t=0$:

$$\mathbf{u}_{0,i} = \log_{\mathbf{x}_{0,i}}(\mathbf{x}_{1,i})$$

The loss reuses the Riemannian tangent-space MSE of Definition 6.7 restricted to $\mathcal{M}_x$:

$$\mathcal{L}_{\mathrm{mask}\_x} = \frac{1}{|\mathcal{M}_x|} \sum_{i \in \mathcal{M}_x} \left\| \hat{\mathbf{v}}_i - \mathbf{u}_{0,i} \right\|^2_{T_{\mathbf{x}_{0,i}}\mathcal{M}}$$

Set to $0$ when $|\mathcal{M}_x| = 0$.  Shares the backbone vector-field output $\hat{\mathbf{V}}$ with $\mathcal{L}_{\mathrm{cont}}$ — no additional prediction head.  Mirrors the KGC $(h, r, ?)$ downstream direction (locate an entity on the manifold given its text).

**Participation rules** (strictly aligned with the partition):

| Loss | $\mathcal{U}$ | $\mathcal{M}_c$ | $\mathcal{M}_x$ | Virtual |
|------|:---:|:---:|:---:|:---:|
| $\mathcal{L}_{\mathrm{cont}}$ | ✓ | ✗ ($\mathbf{x}_t$ frozen at $\mathbf{x}_1$) | ✗ (handled by $\mathcal{L}_{\mathrm{mask}\_x}$) | ✗ |
| $\mathcal{L}_{\mathrm{disc}}$ | ✓ | ✓ | ✓ | ✗ |
| $\mathcal{L}_{\mathrm{align}}$ | ✓ | ✗ (text replaced) | ✗ ($\mathbf{h}$ quality low) | ✗ |
| $\mathcal{L}_{\mathrm{mask}\_c}$ | ✗ | ✓ | ✗ | ✗ |
| $\mathcal{L}_{\mathrm{mask}\_x}$ | ✗ | ✗ | ✓ | ✗ |

**Definition 6.10 (Total Training Loss).**
$$\mathcal{L} = \mathcal{L}_{\mathrm{cont}} + \lambda\,\mathcal{L}_{\mathrm{disc}} + \mu\,\mathcal{L}_{\mathrm{align}} + \nu_c\,\mathcal{L}_{\mathrm{mask}\_c} + \nu_x\,\mathcal{L}_{\mathrm{mask}\_x}$$
where $\lambda, \mu, \nu_c, \nu_x \geq 0$ are loss weight hyperparameters.  Setting $\nu_c = \nu_x = 0$ recovers the original unmasked objective.

### 6.5 Training and Inference Algorithms

**Algorithm 1: RiemannFM Single Training Step**

**Input**: Training subgraph $(\mathbf{X}_1, \mathbf{E}_1, \mathbf{C}_{\mathcal{V}}, \mathbf{C}_{\mathcal{R}}, \mathbf{m})$, parameters $\theta$
**Output**: Loss $\mathcal{L}$

1. Collator emits the node three-way partition $\{\mathcal{U}, \mathcal{M}_c, \mathcal{M}_x\}$ and the per-node time label $t_{\mathrm{node}} \in \mathbb{R}^{B \times N}$ (Definition 6.9a): $0$ on $\mathcal{M}_x$, $1$ on $\mathcal{M}_c$, placeholder on $\mathcal{U}$
2. Sample the subgraph-level scalar $t \sim \mathrm{Uniform}(0, 1 - \epsilon_t)$ and fill the placeholder positions of $t_{\mathrm{node}}$ via broadcasting
3. For $i \in \mathcal{M}_c$: replace the input text with $\mathbf{e}_{\mathrm{mask}}$ (geometry retained at $\mathbf{x}_{1,i}$ via $t_i = 1$)
4. For $i \in [N]$: sample $\mathbf{x}_{0,i} \sim p_0^{\mathcal{M}}$ (Definition 6.1)
5. For $(i, j) \in [N]^2$, $k \in [K]$: sample $\mathbf{E}_{0,ij}^{(k)} \sim \mathrm{Bernoulli}(\rho_k)$ (Definition 6.2)
6. For $i \in [N]$: $\mathbf{x}_{t,i} = \exp_{\mathbf{x}_{0,i}}(t_i \cdot \log_{\mathbf{x}_{0,i}}(\mathbf{x}_{1,i}))$ (Definition 6.3, per-node interpolation with $t_{\mathrm{node}}$)
7. For $(i, j) \in [N]^2$: sample $z_{ij} \sim \mathrm{Bernoulli}(t)$, $\mathbf{E}_{t,ij} = z_{ij} \mathbf{E}_{1,ij} + (1 - z_{ij}) \mathbf{E}_{0,ij}$ (Definition 6.4, edge flow uses the subgraph scalar $t$)
8. $(\hat{\mathbf{V}}, \hat{\mathbf{P}}, \mathbf{H}) = f_\theta(\mathbf{X}_t, \mathbf{E}_t, t_{\mathrm{node}}, \mathbf{C}_{\mathcal{V}}, \mathbf{C}_{\mathcal{R}}, \mathbf{m})$ (Definition 5.1)
9. For $i \in [N]$: $\mathbf{u}_{t,i} = \frac{1}{1-t_i}\log_{\mathbf{x}_{t,i}}(\mathbf{x}_{1,i})$ (Definition 6.5)
10. Compute $\mathcal{L} = \mathcal{L}_{\mathrm{cont}} + \lambda\,\mathcal{L}_{\mathrm{disc}} + \mu\,\mathcal{L}_{\mathrm{align}} + \nu_c\,\mathcal{L}_{\mathrm{mask}\_c} + \nu_x\,\mathcal{L}_{\mathrm{mask}\_x}$ (Definition 6.10)
11. Riemannian Adam update for $\theta$: standard Adam for Euclidean parameters, Riemannian gradient for curvature parameters
12. Curvature projection: $\kappa_h \leftarrow \min(\kappa_h, -\epsilon_\kappa)$, $\kappa_s \leftarrow \max(\kappa_s, \epsilon_\kappa)$

where $\epsilon_t > 0$, $\epsilon_\kappa > 0$ are small constants.

**Algorithm 2: RiemannFM Inference**

**Input**: Text conditioning $\mathbf{C}_{\mathcal{V}}, \mathbf{C}_{\mathcal{R}}$, number of steps $T \in \mathbb{Z}_{>0}$, step size $\Delta t = 1/T$
**Output**: $(\mathbf{X}_1, \mathbf{E}_1, \mathbf{m}_{\mathrm{pred}})$

1. Sample $\mathbf{X}_0, \mathbf{E}_0$ (Definitions 6.1, 6.2)
2. **for** $s = 0, 1, \ldots, T-1$ **do**:
   - $t = s \cdot \Delta t$
   - $(\hat{\mathbf{V}}, \hat{\mathbf{P}}) = f_\theta(\mathbf{X}_t, \mathbf{E}_t, t, \mathbf{C}_{\mathcal{V}}, \mathbf{C}_{\mathcal{R}}, \mathbf{m})$
   - For $i \in [N]$: $\mathbf{x}_{t+\Delta t, i} = \exp_{\mathbf{x}_{t,i}}(\Delta t \cdot \hat{\mathbf{v}}_i)$
   - For $(i, j) \in [N]^2$, $k \in [K]$: with probability $p_{\mathrm{flip}}(t, \Delta t) = \min\!\left(\frac{\Delta t}{1-t},\, 1\right)$ update $\mathbf{E}_{t+\Delta t, ij}^{(k)} = \mathbb{1}[\hat{\mathbf{P}}_{ij}^{(k)} > 0.5]$, otherwise keep $\mathbf{E}_{t, ij}^{(k)}$ unchanged
3. Virtual node removal: $m_{\mathrm{pred},i} = \mathbb{1}[d_{\mathcal{M}}(\mathbf{x}_{1,i}, \mathbf{x}_\varnothing) \geq \epsilon_{\mathrm{null}}]$
4. Edge binarization: $\mathbf{E}_{1,ij}^{(k)} \leftarrow \mathbb{1}[\hat{\mathbf{P}}_{ij}^{(k)} > p_{\mathrm{thresh}}]$

where $\epsilon_{\mathrm{null}} > 0$ is the virtual node detection threshold and $p_{\mathrm{thresh}} \in (0, 1)$ is the edge binarization threshold.

---

## 7. Downstream Task Fine-Tuning

### 7.1 Unified Masking Framework

**Definition 7.1 (Task Masks).**
- Node task mask $\mathbf{m}^{\mathrm{task}} \in \{0, 1\}^N$: $m_i^{\mathrm{task}} = 1$ indicates to-be-predicted, $m_i^{\mathrm{task}} = 0$ indicates known condition.
- Edge task mask $\mathbf{M}^{\mathrm{task}} \in \{0, 1\}^{N \times N}$: $M_{ij}^{\mathrm{task}} = 1$ indicates to-be-predicted, $M_{ij}^{\mathrm{task}} = 0$ indicates known condition.

**Definition 7.2 (Conditioned Noisy Input).** To-be-predicted parts are interpolated normally per Section 6; known parts retain the ground-truth data-end values:
$$\mathbf{x}_{t,i}^{\mathrm{cond}} = \begin{cases} \exp_{\mathbf{x}_{0,i}}(t \cdot \log_{\mathbf{x}_{0,i}}(\mathbf{x}_{1,i})) & m_i^{\mathrm{task}} = 1 \\ \mathbf{x}_{1,i} & m_i^{\mathrm{task}} = 0 \end{cases}$$

$$\mathbf{E}_{t,ij}^{\mathrm{cond}} = \begin{cases} z_{ij} \cdot \mathbf{E}_{1,ij} + (1 - z_{ij}) \cdot \mathbf{E}_{0,ij} & M_{ij}^{\mathrm{task}} = 1 \\ \mathbf{E}_{1,ij} & M_{ij}^{\mathrm{task}} = 0 \end{cases}$$

**Definition 7.3 (Conditioned Fine-Tuning Loss).** The loss is computed only on the to-be-predicted parts:
$$\mathcal{L}_{\mathrm{cont}}^{\mathrm{task}} = \frac{1}{\sum_i m_i^{\mathrm{task}} \cdot m_i} \sum_{i=1}^N m_i^{\mathrm{task}} \cdot m_i \cdot \|\hat{\mathbf{v}}_i - \mathbf{u}_{t,i}\|_{T_{\mathbf{x}_{t,i}}\mathcal{M}}^2$$

$$\mathcal{L}_{\mathrm{disc}}^{\mathrm{task}} = \frac{1}{\sum_{i,j} M_{ij}^{\mathrm{task}}} \sum_{(i,j)} M_{ij}^{\mathrm{task}} \sum_{k=1}^K \mathrm{BCE}_{\mathrm{w}}(\hat{\mathbf{P}}_{ij}^{(k)}, \mathbf{E}_{1,ij}^{(k)})$$

Note that $\mathcal{L}_{\mathrm{cont}}^{\mathrm{task}}$ requires both $m_i^{\mathrm{task}} = 1$ (to-be-predicted) and $m_i = 1$ (real node, not virtual).

### 7.2 Task 1: Knowledge Graph Completion

**Masking strategy.** Given a training subgraph, construct task masks as follows:

- Edge mask: For each existing edge $(i, j)$ (i.e., $\mathbf{E}_{1,ij} \neq \mathbf{0}_K$), independently set $M_{ij}^{\mathrm{task}} = 1$ with probability $p_{\mathrm{mask}}^E$;
- Node mask: For each real node $i$ ($m_i = 1$), independently set $m_i^{\mathrm{task}} = 1$ with probability $p_{\mathrm{mask}}^V$, simultaneously masking all associated edges: $\forall j \in [N]$, $M_{ij}^{\mathrm{task}} = M_{ji}^{\mathrm{task}} = 1$;
- Remaining positions: $m_i^{\mathrm{task}} = 0$, $M_{ij}^{\mathrm{task}} = 0$.

where $p_{\mathrm{mask}}^E, p_{\mathrm{mask}}^V \in (0, 1)$ are hyperparameters.

**Fine-tuning loss.**
$$\mathcal{L}_{\mathrm{KGC}} = \mathcal{L}_{\mathrm{cont}}^{\mathrm{task}} + \lambda_{\mathrm{KGC}}\,\mathcal{L}_{\mathrm{disc}}^{\mathrm{task}}$$
where $\lambda_{\mathrm{KGC}} > 0$ is a weight hyperparameter.

**Evaluation.** For a query fact $(v_i, r_k, ?)$, rank all candidate entities $v_j$ by the model output $\hat{\mathbf{P}}_{ij}^{(k)}$. Metrics:
- $\mathrm{MRR} = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \frac{1}{\mathrm{rank}_q}$, where $\mathcal{Q}$ is the test query set;
- $\mathrm{Hits}@n = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \mathbb{1}[\mathrm{rank}_q \leq n]$, $n \in \{1, 3, 10\}$.

### 7.3 Task 2: Text-Conditioned Subgraph Generation

**Problem setting.** Given a text query $q \in \Sigma^*$, generate a subgraph $\mathcal{G}$ that semantically matches the query.

**Masking strategy.** Full masking:
$$m_i^{\mathrm{task}} = 1 \quad \forall i \in [N], \qquad M_{ij}^{\mathrm{task}} = 1 \quad \forall (i, j) \in [N]^2$$

**Text conditioning replacement.** Replace all node text conditioning with the query embedding:
$$\mathbf{c}_q = \phi_{\mathrm{text}}(q) \in \mathbb{R}^{d_c}, \qquad \mathbf{C}_{\mathcal{V}} = \mathbf{1}_N \mathbf{c}_q^\top \in \mathbb{R}^{N \times d_c}$$

where $\mathbf{1}_N \in \mathbb{R}^N$ is the all-ones vector. $\mathbf{C}_{\mathcal{R}}$ remains unchanged.

**Fine-tuning loss.** Since all parts are masked, the conditioned loss degenerates to the pretraining loss form:
$$\mathcal{L}_{\mathrm{T2G}} = \mathcal{L}_{\mathrm{cont}} + \lambda_{\mathrm{T2G}}\,\mathcal{L}_{\mathrm{disc}} + \mu_{\mathrm{T2G}}\,\mathcal{L}_{\mathrm{align}}$$
where $\lambda_{\mathrm{T2G}}, \mu_{\mathrm{T2G}} > 0$ are weight hyperparameters. The contrastive loss $\mathcal{L}_{\mathrm{align}}$ encourages the generated node coordinates to be close to the query text $\mathbf{c}_q$ in the alignment space.

**Evaluation.** Align the generated subgraph with the ground-truth subgraph via maximum matching, then compute:
- **Node F1**: Based on matched node text similarity, using a BERTScore threshold to determine successful matches;
- **Relation F1**: On matched node pairs, compare the predicted edge type set with the ground-truth edge type set using F1;
- **BERTScore**: Average BERTScore between node text descriptions in the generated and ground-truth subgraphs.

### 7.4 Task 3: Graph Anomaly Detection

**Problem setting.** Given a subgraph $\mathcal{G}$, detect anomalous edges and anomalous nodes. This task directly uses the pretrained model without fine-tuning.

**Anomaly score construction.** For the subgraph to be inspected, set $\mathbf{X}_1 = \mathbf{X}$, $\mathbf{E}_1 = \mathbf{E}$ as the data end, sample noise $\mathbf{X}_0, \mathbf{E}_0$ per Definitions 6.1–6.2, and construct noisy inputs $\mathbf{X}_t, \mathbf{E}_t$ per Definitions 6.3–6.4.

**Definition 7.4 (Edge Anomaly Score).** Let $\mathcal{T} = \{t_1, \ldots, t_{|\mathcal{T}|}\} \subset (0, 1)$ be a preselected set of time steps (e.g., a uniform grid). For each existing edge $(i, j, k)$ ($\mathbf{E}_{1,ij}^{(k)} = 1$):
$$S_{ij}^{(k)} = 1 - \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \hat{\mathbf{P}}_{ij}^{(k)}\!\left(f_\theta(\mathbf{X}_t, \mathbf{E}_t, t, \mathbf{C}_{\mathcal{V}}, \mathbf{C}_{\mathcal{R}}, \mathbf{m})\right)$$

Intuition: The pretrained model assigns high probability $\hat{\mathbf{P}}_{ij}^{(k)} \approx 1$ to normal facts, yielding anomaly score $S_{ij}^{(k)} \approx 0$; for anomalous facts, it assigns low probability, yielding anomaly scores close to $1$. Averaging over multiple time steps improves robustness.

**Definition 7.5 (Node Anomaly Score).** Let $\mathcal{N}_i = \{j \in [N] : \mathbf{E}_{1,ij} \neq \mathbf{0}_K\}$ be the set of outgoing-edge neighbors of node $i$.
$$S_i = \frac{1}{|\mathcal{N}_i|}\sum_{j \in \mathcal{N}_i} \max_{k:\,\mathbf{E}_{1,ij}^{(k)}=1} S_{ij}^{(k)}$$

That is, a node's anomaly level is the average anomaly score of the most suspicious relation among all its outgoing edges.

**Evaluation.** AUROC, AP (Average Precision).

---

## Symbol Quick Reference

### Base Sets and Indices

| Symbol | Type | Domain | Meaning |
|--------|------|--------|---------|
| $\mathcal{V}$ | Set | — | Entity set |
| $\mathcal{R}$ | Set | — | Relation type set |
| $K$ | Scalar | $\mathbb{Z}_{>0}$ | Total number of relation types, $K \triangleq |\mathcal{R}|$ |
| $\mathcal{K}$ | Graph | — | Knowledge graph $(\mathcal{V}, \mathcal{E})$ |
| $\mathcal{E}$ | Set | $\subseteq \mathcal{V} \times \mathcal{R} \times \mathcal{V}$ | Fact set |
| $\mathcal{G}$ | Graph | — | Subgraph $(\mathcal{V}_{\mathcal{G}}, \mathcal{E}_{\mathcal{G}})$ |
| $N$ | Scalar | $\mathbb{Z}_{>0}$ | Subgraph node count (after padding, $N = N_{\max}$) |
| $N_{\max}$ | Scalar | $\mathbb{Z}_{>0}$ | Upper bound on subgraph node count |
| $[n]$ | Set | — | $\{1, 2, \ldots, n\}$ |
| $\mathcal{D}$ | Set | — | Training dataset (collection of subgraphs) |

### Manifolds and Geometry

| Symbol | Type | Domain | Meaning |
|--------|------|--------|---------|
| $\mathcal{M}$ | Manifold | — | Product manifold $\mathbb{H}^{d_h}_{\kappa_h} \times \mathbb{S}^{d_s}_{\kappa_s} \times \mathbb{R}^{d_e}$ |
| $d_h, d_s, d_e$ | Scalar | $\mathbb{Z}_{>0}$ | Intrinsic dimensions of hyperbolic, spherical, Euclidean subspaces |
| $D$ | Scalar | $\mathbb{Z}_{>0}$ | Total ambient dimension, $D = (d_h+1)+(d_s+1)+d_e$ |
| $\kappa_h$ | Scalar | $\mathbb{R}_{<0}$ | Hyperbolic space curvature (learnable) |
| $\kappa_s$ | Scalar | $\mathbb{R}_{>0}$ | Spherical curvature (learnable) |
| $\langle \cdot, \cdot \rangle_{\mathrm{L}}$ | Operation | $\mathbb{R}$ | Lorentz inner product |
| $T_\mathbf{x}\mathcal{M}$ | Space | — | Tangent space at $\mathbf{x}$ |
| $\exp_\mathbf{x}$ | Map | $T_\mathbf{x}\mathcal{M} \to \mathcal{M}$ | Exponential map |
| $\log_\mathbf{x}$ | Map | $\mathcal{M} \to T_\mathbf{x}\mathcal{M}$ | Logarithmic map |
| $d_{\mathcal{M}}$ | Function | $\mathcal{M} \times \mathcal{M} \to \mathbb{R}_{\geq 0}$ | Product manifold geodesic distance |
| $d_{\mathbb{H}}, d_{\mathbb{S}}, d_{\mathbb{R}}$ | Function | — | Component subspace geodesic distances |
| $\pi$ | Map | $\mathcal{M} \to \mathbb{R}^D$ | Coordinate projection |
| $\mathbf{x}_\varnothing$ | Vector | $\mathcal{M}$ | Virtual node anchor |

### Subgraph Representation

| Symbol | Type | Domain | Meaning |
|--------|------|--------|---------|
| $\mathbf{x}_i$ | Vector | $\mathcal{M}$ | Manifold coordinate of node $i$ |
| $\mathbf{x}_i^{\mathbb{H}}, \mathbf{x}_i^{\mathbb{S}}, \mathbf{x}_i^{\mathbb{R}}$ | Vector | $\mathbb{R}^{d_h+1}, \mathbb{R}^{d_s+1}, \mathbb{R}^{d_e}$ | Subspace components |
| $\mathbf{X}$ | Tuple | $\mathcal{M}^N$ | Node coordinate sequence |
| $\mathbf{E}$ | Tensor | $\{0,1\}^{N \times N \times K}$ | Edge type tensor |
| $\mathbf{E}_{ij}$ | Vector | $\{0,1\}^K$ | Multi-hot edge type from node $i$ to $j$ |
| $\mathbf{C}_{\mathcal{V}}$ | Matrix | $\mathbb{R}^{N \times d_c}$ | Node text conditioning matrix |
| $\mathbf{C}_{\mathcal{R}}$ | Matrix | $\mathbb{R}^{K \times d_c}$ | Relation text conditioning matrix |
| $\mathbf{c}_i$ | Vector | $\mathbb{R}^{d_c}$ | Text conditioning of node $i$ ($\mathbf{c}_i \triangleq \mathbf{c}_{v_i}$) |
| $\mathbf{c}_{r_k}$ | Vector | $\mathbb{R}^{d_c}$ | Text conditioning of relation $r_k$ |
| $\mathbf{m}$ | Vector | $\{0,1\}^N$ | Virtual node mask |
| $\phi_{\mathrm{text}}$ | Map | $\Sigma^* \to \mathbb{R}^{d_c}$ | Pretrained text encoder |
| $d_c$ | Scalar | $\mathbb{Z}_{>0}$ | Text encoder output dimension |

### Model Architecture (RieFormer)

| Symbol | Type | Domain | Meaning |
|--------|------|--------|---------|
| $f_\theta$ | Map | — | RieFormer model |
| $\theta$ | — | — | All learnable parameters (including $\kappa_h, \kappa_s$) |
| $L$ | Scalar | $\mathbb{Z}_{>0}$ | Number of RieFormer layers |
| $d_v$ | Scalar | $\mathbb{Z}_{>0}$ | Node hidden dimension |
| $d_{e'}$ | Scalar | $\mathbb{Z}_{>0}$ | Edge hidden dimension |
| $d_r$ | Scalar | $\mathbb{Z}_{>0}$ | Relation embedding dimension |
| $d_t$ | Scalar | $\mathbb{Z}_{>0}$ (even) | Time embedding dimension |
| $d_{\mathrm{head}}$ | Scalar | $\mathbb{Z}_{>0}$ (even) | Attention head dimension, $d_{\mathrm{head}} = d_v / n_h$ |
| $d_\mathrm{a}$ | Scalar | $\mathbb{Z}_{>0}$ | Alignment space dimension |
| $n_h$ | Scalar | $\mathbb{Z}_{>0}$ | Number of attention heads |
| $\mathbf{h}_i^{V,(l)}$ | Vector | $\mathbb{R}^{d_v}$ | Layer $l$ embedding of node $i$ |
| $\mathbf{h}_{ij}^{E,(l)}$ | Vector | $\mathbb{R}^{d_{e'}}$ | Layer $l$ embedding of edge $(i,j)$ |
| $\mathbf{t}_{\mathrm{emb}}$ | Vector | $\mathbb{R}^{d_v}$ | Time embedding |
| $\hat{\mathbf{v}}_i$ | Vector | $T_{\mathbf{x}_{t,i}}\mathcal{M}$ | Model-predicted tangent vector |
| $\hat{\mathbf{P}}_{ij}^{(k)}$ | Scalar | $[0,1]$ | Edge type probability prediction |
| $\kappa^{(s)}$ | Function | $\mathcal{M} \times \mathcal{M} \to \mathbb{R}$ | Geodesic kernel for head $s$ |
| $\hat{\rho}_i$ | Scalar | $\mathbb{R}_{\geq 0}$ | Hierarchical depth estimate |
| $\odot$ | Operation | — | Element-wise product (Hadamard product) |
| $\|$ | Operation | — | Vector concatenation |

### Flow Matching

| Symbol | Type | Domain | Meaning |
|--------|------|--------|---------|
| $t$ | Scalar | $[0,1]$ | Flow matching time step ($0$=noise, $1$=data) |
| $\mathbf{X}_0, \mathbf{E}_0$ | — | — | Noise-end samples |
| $\mathbf{X}_1, \mathbf{E}_1$ | — | — | Data-end (training subgraph) |
| $\mathbf{x}_{t,i}$ | Vector | $\mathcal{M}$ | Interpolated coordinate of node $i$ at time $t$ |
| $\mathbf{E}_{t,ij}$ | Vector | $\{0,1\}^K$ | Interpolated edge type of $(i,j)$ at time $t$ |
| $z_{ij}$ | Scalar | $\{0,1\}$ | Edge interpolation mask, $z_{ij} \sim \mathrm{Bernoulli}(t)$ |
| $\mathbf{u}_{t,i}$ | Vector | $T_{\mathbf{x}_{t,i}}\mathcal{M}$ | Conditional vector field target |
| $\rho_k$ | Scalar | $[0,1]$ | Marginal frequency of relation $r_k$ |
| $p_0^{\mathcal{M}}$ | Distribution | — | Continuous noise prior |
| $R_{\mathbb{H}}$ | Scalar | $\mathbb{R}_{>0}$ | Hyperbolic uniform distribution truncation radius |
| $\sigma_0$ | Scalar | $\mathbb{R}_{>0}$ | Euclidean noise standard deviation |
| $T$ | Scalar | $\mathbb{Z}_{>0}$ | Number of inference steps |
| $\Delta t$ | Scalar | $(0,1]$ | Inference step size, $\Delta t = 1/T$ |
| $p_{\mathrm{flip}}$ | Function | $[0,1]$ | Edge update flip probability |

### Loss Functions

| Symbol | Type | Domain | Meaning |
|--------|------|--------|---------|
| $\mathcal{L}_{\mathrm{cont}}$ | Scalar | $\mathbb{R}_{\geq 0}$ | Manifold vector field loss |
| $\mathcal{L}_{\mathrm{disc}}$ | Scalar | $\mathbb{R}_{\geq 0}$ | Edge type loss |
| $\mathcal{L}_{\mathrm{align}}$ | Scalar | $\mathbb{R}_{\geq 0}$ | Graph-text contrastive loss |
| $\mathcal{L}$ | Scalar | $\mathbb{R}_{\geq 0}$ | Total training loss |
| $\lambda, \mu$ | Scalar | $\mathbb{R}_{>0}$ | Loss weights |
| $\tau$ | Scalar | $\mathbb{R}_{>0}$ | Contrastive loss temperature |
| $w_k^+$ | Scalar | $\mathbb{R}_{>0}$ | Positive sample weight for relation $r_k$ |
| $w_{\max}$ | Scalar | $\mathbb{R}_{>0}$ | Positive sample weight truncation upper bound |
| $\mathcal{S}$ | Set | $\subseteq [N]^2$ | Edge loss sampling set |
| $\mathcal{B}$ | Set | $\subseteq [N]$ | Contrastive loss mini-batch indices |
| $\epsilon_t, \epsilon_p, \epsilon_\kappa$ | Scalar | $\mathbb{R}_{>0}$ | Numerical stability/truncation constants |

### Downstream Tasks

| Symbol | Type | Domain | Meaning |
|--------|------|--------|---------|
| $\mathbf{m}^{\mathrm{task}}$ | Vector | $\{0,1\}^N$ | Node task mask |
| $\mathbf{M}^{\mathrm{task}}$ | Matrix | $\{0,1\}^{N \times N}$ | Edge task mask |
| $\mathcal{L}_{\mathrm{cont}}^{\mathrm{task}}, \mathcal{L}_{\mathrm{disc}}^{\mathrm{task}}$ | Scalar | $\mathbb{R}_{\geq 0}$ | Conditioned fine-tuning loss |
| $p_{\mathrm{mask}}^E, p_{\mathrm{mask}}^V$ | Scalar | $(0,1)$ | KGC masking probabilities |
| $\mathbf{c}_q$ | Vector | $\mathbb{R}^{d_c}$ | T2G query text embedding |
| $S_{ij}^{(k)}$ | Scalar | $[0,1]$ | Edge anomaly score |
| $S_i$ | Scalar | $[0,1]$ | Node anomaly score |
| $\mathcal{T}$ | Set | $\subset (0,1)$ | GAD preselected time step set |
| $\mathcal{N}_i$ | Set | $\subseteq [N]$ | Outgoing-edge neighbor set of node $i$ |
| $\epsilon_{\mathrm{null}}$ | Scalar | $\mathbb{R}_{>0}$ | Virtual node detection threshold |
| $p_{\mathrm{thresh}}$ | Scalar | $(0,1)$ | Edge binarization threshold |
