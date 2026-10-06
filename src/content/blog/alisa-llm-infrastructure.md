---
title: "Alisa’s Book of LLMs：大模型 Infra 核心章节"
description: "围绕 Transformer 计算核算、推理、数值精度、GPU 与分布式并行整理的中文学习材料。"
date: "2026-10-06"
tags: ["翻译", "Transformer", "大模型推理", "分布式训练"]
sourceURL: "https://alisawuffles.notion.site/alisa-s-book-of-llms"
sourceAuthor: "Alisa Liu"
translationScope: "依据已有中文译稿重新编排与整理；保留原文来源，包含整理者的解释。"
---

## 从计算图理解大模型系统

读大模型系统资料时，模型结构、显存估算、推理优化和分布式训练经常分散在不同章节。本文依据已有中文译稿，将 Alisa’s Book of LLMs 中的 Infra 内容整理为六个部分：先确定 Transformer 的计算与存储需求，再讨论推理、缩放规律、GPU、数值精度和并行策略。

阅读时可以抓住三个问题：每个算子处理什么形状的张量，哪些中间结果需要保留，以及数据在设备之间如何移动。公式、示例代码和图表用于解释这些问题；文中的工程补充属于整理说明，不等同于原书的逐句翻译。

<span id="ch1-modern-transformer"></span>

## 1 现代 Transformer 架构与计算核算

 **核心概览：** 本节以使用 RMSNorm、RoPE、GQA 与 SwiGLU 的稠密 Decoder-Only Transformer 为例。不同模型可能替换其中的模块，后续公式需要结合具体结构使用。阅读顺序是张量形状、前向计算、参数量、激活值和显存预算。

<span id="sec-arch-overview"></span>

### 1.1 架构定义与符号对照表

现代 Transformer 模型超参数与张量维度标准符号对照表：

| 维度 / 描述 (Dimension Description) | 符号 (Symbol) | 常见取值示例 |
|----|----|----|
|  **number of sequences in the batch** （批次中的序列数量） | $B$ | 批大小（Batch Size） |
|  **number of layers** （模型层数） | $L$ | 堆叠的 Transformer Block 数量（如 32、80） |
|  **sequence length (number of tokens to generate)** （生成长度） | $T$ | 自回归生成的 Token 数（单步 Decode 时 $T=1$） |
|  **sequence length (provided context)** （上下文输入长度） | $S$ | Prefill 阶段的上下文 Token 数（如 4096、8192 等） |
|  **vocab size** （词表大小） | $V$ | 词表容量（如 32,000、128,256、151,936） |
|  **hidden dimension** （隐藏层维度 / 模型维度 $d_{\text{model}}$） | $D$ | 如 4096 (8B), 8192 (70B) |
|  **head dimension** （注意力头维度） | $H$ | 通常固定为 128，满足 $D = N \cdot H$ |
|  **number of query heads, $N \cdot H = D$** （Query 注意力头数） | $N$ | 如 32 (8B), 64 (70B) |
|  **number of key/value heads, $K < N$ in GQA** （KV 注意力头数） | $K$ | GQA 下 $K < N$（如 8 个 KV 头）；标准 MHA 下 $K = N$ |
|  **group size in GQA = $N // K$** （GQA 每组大小） | $G$ | 每个 KV 头服务的 Query 头数（例如 $32 // 8 = 4$） |
|  **MLP hidden dimension, generally $F = 4D$** （前馈隐层维度） | $F$ | 传统标准为 $4D$；在 SwiGLU 下通常设为 $\approx \frac{8}{3}D$ |

![示意图 1：Tokens Embedding W_e: (V, D) Transformer Layer (× L) RMSNorm GQA Attention + RoPE 旋转位置编码 + RMSNorm SwiGLU FFN Gate × Up → Down + Final RMSNo](/translations/alisa-llm-infrastructure/diagram-1.svg)

图 1.1：现代主流 Decoder-Only Transformer 完整张量计算流架构图

<span id="sec-layer-forward"></span>

### 1.2 逐层前向传播推导

 **1. Token Embedding（词嵌入层）：**

- 词嵌入矩阵 $\mathbf{W}_e \in \mathbb{R}^{V \times D}$，初始隐藏状态 $\mathbf{X}^{(0)} \in \mathbb{R}^{B \times S \times D}$：

$$
\mathbf{X}^{(0)} = \mathbf{W}_e[\text{tokens}]
$$

 **2. 循环层（Layer Loop，针对每层 $\ell \in [0, \dots, L-1]$）：**

- **RMSNorm** ：将 $\mathbf{X}^{(\ell)}$ 的每个元素除以 $\mathbf{X}^{(\ell)}$ 的均方根 RMS（使隐层状态具有单位 RMS），然后乘以可学习的重新缩放参数 $\gamma$：

  $$
  \bar{\mathbf{X}}^{(\ell)} = \frac{\mathbf{X}^{(\ell)}}{\operatorname{RMS}(\mathbf{X}^{(\ell)}) + \epsilon} \odot \gamma_{\text{attn}}^{(\ell)}, \quad \operatorname{RMS}(\mathbf{X}) = \sqrt{\frac{1}{D}\sum_{i=1}^D x_i^2}
  $$

- **线性投影** ：每个头使用 $\mathbf{W}_Q^{(\ell)} \in \mathbb{R}^{D \times D}$，$\mathbf{W}_K \in \mathbb{R}^{D \times KH}$，$\mathbf{W}_V^{(\ell)} \in \mathbb{R}^{D \times KH}$（其中 $H = D/N$）将 $\mathbf{X}^{(\ell)}$ 投影到头的低维子空间：

  $$
  \mathbf{Q} = \bar{\mathbf{X}}\mathbf{W}_Q \in \mathbb{R}^{B \times T \times D}, \quad \mathbf{K} = \bar{\mathbf{X}}\mathbf{W}_K \in \mathbb{R}^{B \times S \times (K H)}, \quad \mathbf{V} = \bar{\mathbf{X}}\mathbf{W}_V \in \mathbb{R}^{B \times S \times (K H)}
  $$

  - *\[可选 QK-Norm\]*：对 Query 和 Key 向量应用 RMSNorm，以控制进入点积运算的向量模长。
- **Reshape 暴露头维度** ：将通道维度展开 $D \to N \times H$ 以及 $K \cdot H \to K \times H$，然后转置序列长度维度（$S$ 或 $T$）和头维度（$N$ 或 $K$）：

  $$
  \mathbf{Q} \in \mathbb{R}^{B \times T \times D} \to \mathbb{R}^{B \times N \times T \times H}
  $$

  $$
  \mathbf{K} \in \mathbb{R}^{B \times S \times (K\cdot H)} \to \mathbb{R}^{B \times K \times S \times H}
  $$

  $$
  \mathbf{V} \in \mathbb{R}^{B \times S \times (K\cdot H)} \to \mathbb{R}^{B \times K \times S \times H}
  $$

- **针对 GQA 展开 $K, V$** ：将 $K, V$ 沿组维度重复展开以对齐 Query 头数：

  $$
  \mathbf{K} \in \mathbb{R}^{B \times K \times S \times H} \to \mathbb{R}^{B \times N \times S \times H}
  $$

  $$
  \mathbf{V} \in \mathbb{R}^{B \times K \times S \times H} \to \mathbb{R}^{B \times N \times S \times H}
  $$

- **应用 RoPE** ：在每个位置 $m$ 通过旋转矩阵 $\mathbf{R}_m$ 旋转 Query 向量 $\mathbf{q}_m \in \mathbb{R}^H$（或 Key 向量 $\mathbf{k}_m$）：
  - 对于维度对 $i$（对应 $\mathbf{q}_m$ 中的索引对 $(2i, 2i+1)$），旋转角度为 $m\theta_i$，其中 $\theta_i = \Theta^{-\frac{2i}{H}}$；
  - 超参数 $\Theta$ 控制基底旋转频率，$H$ 为头维度：

  $$
  \mathbf{R}_m = \begin{bmatrix} \ddots & & \\ & \mathbf{R}_m^{(i)} & \\ & & \ddots \end{bmatrix} \in \mathbb{R}^{H \times H} \quad \text{其中} \quad \mathbf{R}_m^{(i)} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}
  $$

  $$
  \mathbf{q}_m \leftarrow \mathbf{R}_m \mathbf{q}_m, \quad \mathbf{k}_m \leftarrow \mathbf{R}_m \mathbf{k}_m
  $$

- **计算注意力分数** ：
  - 除以头维度平方根 $\sqrt{H}$，否则点积的幅度会随 $\sqrt{H}$ 增大（Softmax 输入过大 $\to$ 分布极度尖锐 $\to$ 梯度难以更新）：

  $$
  \mathbf{A} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{H}} \in \mathbb{R}^{B \times N \times T \times S}
  $$

  - 应用因果掩码（Causal Mask）：

  $$
  \mathbf{A}_{ij} \leftarrow \begin{cases} \mathbf{A}_{ij} & \text{若 } j \le i \\ -\infty & \text{若 } j > i \end{cases}
  $$

  - 应用 Softmax：

  $$
  \mathbf{A} = \operatorname{softmax}(\mathbf{A}) = \frac{\exp(\mathbf{A}_{ij})}{\sum_{k=1}^S \exp(\mathbf{A}_{ik})}
  $$

- **加权聚合与输出投影** ：从 Value 的加权和中获取注意力输出：

  $$
  \mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{B \times N \times T \times H} \to \mathbb{R}^{B \times T \times D}
  $$

   应用输出投影 $\mathbf{W}_O^{(\ell)} \in \mathbb{R}^{D \times D}$ 混合不同头的输出：

  $$
  \mathbf{O}_{\text{proj}} = \mathbf{O}\mathbf{W}_O
  $$

   残差连接：

  $$
  \mathbf{X}^{(\ell)} \leftarrow \mathbf{X}^{(\ell)} + \mathbf{O}_{\text{proj}}
  $$

- **前馈网络（Feed Forward Network, FFN）** ：
  - RMSNorm 归一化：

  $$
  \bar{\mathbf{X}}^{(\ell)} = \frac{\mathbf{X}^{(\ell)}}{\operatorname{RMS}(\mathbf{X}^{(\ell)}) + \epsilon} \odot \gamma_{\text{ffn}}^{(\ell)}
  $$

  - 使用 $\mathbf{W}_{\text{up}}^{(\ell)} \in \mathbb{R}^{D \times F}$ 和 $\mathbf{W}_{\text{gate}}^{(\ell)} \in \mathbb{R}^{D \times F}$ 进行门控与升维投影 \[升维扩张\]：

  $$
  \mathbf{U} = \bar{\mathbf{X}}\mathbf{W}_{\text{up}}, \quad \mathbf{G} = \bar{\mathbf{X}}\mathbf{W}_{\text{gate}}
  $$

  - SwiGLU 激活：

  $$
  \operatorname{Swish}(\mathbf{G}) = \mathbf{G} \odot \sigma(\mathbf{G}) = \mathbf{G} \odot \frac{1}{1 + e^{-\mathbf{G}}}
  $$

  $$
  \mathbf{H} = \operatorname{Swish}(\mathbf{G}) \odot \mathbf{U} \in \mathbb{R}^{B \times T \times F}
  $$

  - 使用 $\mathbf{W}_{\text{down}}^{(\ell)} \in \mathbb{R}^{F \times D}$ 进行降维投影：

  $$
  \mathbf{F} = \mathbf{H}\mathbf{W}_{\text{down}}
  $$

  - 残差连接：

  $$
  \mathbf{X}^{(\ell+1)} = \mathbf{X}^{(\ell)} + \mathbf{F}
  $$

 **3. 最终层归一化与反词嵌入（Final Layer Norm & Unembedding）：**

- 最终归一化：

$$
\mathbf{X}_{\text{final}} = \frac{\mathbf{X}^{(L)}}{\operatorname{RMS}(\mathbf{X}^{(L)}) + \epsilon} \odot \gamma_{\text{final}}
$$

- 使用 $\mathbf{W}_u \in \mathbb{R}^{D \times V}$ 投影到词表维度以获得最终 Logits：

$$
\mathbf{Z} = \mathbf{X}_{\text{final}}\mathbf{W}_u \in \mathbb{R}^{B \times T \times V}
$$

 <span id="sec-impl-notes"></span>

### 1.3 核心实现细节与 PyTorch

- 使用 `scores.masked_fill(~mask, -torch.inf)` 构建 Softmax 前的注意力分数：
  - 约定传入的 `mask` 中，`True` 表示 **可以被关注** 的位置；
  - `tensor.masked_fill(mask, value)` 会在 `mask` 为 `True` 的对应位置用 `value` 填充。
- **RoPE 缓存预分配与实现** ：
  - 我们需要为每个（位置, 索引）对 $(m, i)$ 缓存 $\cos(m\theta_i)$ 和 $\sin(m\theta_i)$；
  - 这可以在模型初始化时预先完成：

``` python
positions = torch.arange(max_seq_len, device=device)  # shape (max_seq_len)
thetas = self.theta ** (-torch.arange(0, d_k, 2, device=device) / d_k)  # shape (d_k // 2)
angles = positions.unsqueeze(-1) * thetas.unsqueeze(0)
```

- 在工程实践中，为了避免进行大量 $2 \times 2$ 的小矩阵乘法，我们将旋转表达为点积运算：
  - 通过将最后的头维度 $H$ 重新塑形为 $(H/2, 2)$，提取出 $\mathbf{Q}, \mathbf{K}$ 的偶数与奇数索引：

``` python
x_pairs = x.reshape(*x.shape[:-1], -1, 2)
x_even = x_pairs[..., 0]
x_odd = x_pairs[..., 1]

# 计算旋转矩阵在所有偶数和奇数位置的结果
x_out_even = x_even * cos - x_odd * sin
x_out_odd = x_even * sin + x_odd * cos

# 通过将它们并排堆叠并展平来进行交织 (torch.stack 会增加一个新维度)
torch.stack([x_out_even, x_out_odd], dim=-1).flatten(start_dim=-2)
```

- **完整 Attention 模块数据流实现** ：
  - 需要 `.reshape()` 将 `d_model` ($D$) 展开为 `num_heads x head_dim` ($N \times H$)；
  - 需要 `qkv.unbind()` 分割 Queries、Keys、Values \[视具体实现而定\]；
  - 需要 `.transpose()` 交换 `num_heads` 与 `seq_len` 维度以进行注意力运算；
  - 获取 `output` 后，需要再次 `.transpose()` 与 `.reshape()` 恢复原始形状：

``` python
batch_size, seq_len, _ = x.shape

x_norm = self.norm(x)

qkv = self.qkv_proj(x_norm)  # (batch, seq_len, 3 * d_model)

qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
q, k, v = qkv.unbind(dim=2)  # (batch, seq_len, num_heads, head_dim)
q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
k = k.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
v = v.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
output = scaled_dot_product_attention(q, k, v, mask=causal_mask)  # (batch, num_heads, seq_len, head_dim)

output = output.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
output = output.reshape(batch, seq_len, d_model)  # (batch, seq_len, d_model)
output = self.out_proj(output)  # (batch, seq_len, d_model)

return x + output

def scaled_dot_product_attention(q, k, v, mask):
    """
    k, q: (batch_size, ..., seq_len, d_k)
    v: (batch_size, ..., seq_len, d_v)
    returns o: (batch_size, ..., seq_len, d_v)
    """
    d_k = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(~mask, -torch.inf)
    return softmax(scores, dim=-1) @ v
```

<span id="sec-accounting-params"></span>

### 1.4 模型参数量精确核算

- **Embedding 词嵌入层** ：$(V, D)$ 参数量为 $VD$；
- **Attention 注意力模块** ：$2D^2 + 2DKH \approx 4D^2$（标准多头注意力 MHA 中 $N = K$）：
  - $Q$ 投影为 $(D, D)$；
  - $K$ 投影为 $(D, KH)$；
  - $V$ 投影为 $(D, KH)$；
  - $O$ 输出投影为 $(D, D)$；
- **FFN 前馈网络模块** ：$3DF$
  - Up 投影矩阵为 $(D, F)$；
  - Gate 门控投影矩阵为 $(D, F)$；
  - Down 降维投影矩阵为 $(F, D)$；
- **LayerNorm / RMSNorm 归一化层** ：每层 $2D$（Pre-Attention 和 Pre-FFN 各有 $D$ 个参数，即针对维度 $D$ 中每个维度的 $\gamma$），外加最后的 Final Norm（$D$）；
- **Unembedding 反嵌入层** ：$(V, D)$ 参数量为 $VD$；
- **全模型总参数量** ：

  $$
  P_{\text{total}} = 2VD + L(4D^2 + 2D + 3DF) \approx 2VD + 12LD^2 \quad \left(\text{当 } F = \frac{8}{3}D \text{ 时}\right)
  $$

<span id="sec-accounting-activations"></span>

### 1.5 模型激活值显存核算

- **Attention 注意力激活值** ：$6BSD + BNS^2$
  - LayerNorm 输入：$(B, S, D)$；
  - LayerNorm 输出：$(B, S, D)$；
  - Q, K, V 投影输出：分别为 $(B, S, D)$，$(B, S, KH)$，$(B, S, KH)$；
  - Attention Scores 注意力分数：$(B, N, S, S)$；
  - Attention 输出：$(B, S, D)$；
- **FFN 前馈网络激活值** ：$2BSD + 2BSF \approx 8BSD$（当 $F = \frac{8}{3}D$ 时）
  - LayerNorm 输入：$(B, S, D)$；
  - Gate 和 Up 投影输出：各 $(B, S, F)$；
  - Down 降维投影输出：$(B, S, D)$；
- **单层总激活值（未开启 FlashAttention）** ：

  $$
  \text{Activations per layer} = 14BSD + BNS^2
  $$

<span id="sec-accounting-flops"></span>

### 1.6 前向与反向 FLOPs 理论推导

假设处于 Prefill 预填充阶段（因此 $S = T$）：

- **Attention 每层 FLOPs** ：$8BSD^2 + 4BS^2D$
  - $Q$ 投影：$(B, S, D) \times (D, D) \to 2BSD^2$ FLOPs；
  - $K$ 投影：$(B, S, D) \times (D, KH) \to 2BSDKH \approx 2BSD^2$（对于 $K = N$）；
  - $V$ 投影：$(B, S, D) \times (D, KH) \to 2BSDKH \approx 2BSD^2$（对于 $K = N$）；
  - $QK^\top$ 矩阵乘：$(B, N, S, H) \times (B, N, H, S) \to 2BNS^2H = 2BS^2D$（因为 $D = NH$）；
  - $A \times V$ 矩阵乘：$(B, N, S, S) \times (B, N, S, H) \to 2BS^2D$；
  - $O$ 输出投影：$(B, S, D) \times (D, D) \to 2BSD^2$ FLOPs；
- **FFN 每层 FLOPs** ：$6BSDF \approx 16BSD^2$（当 $F = \frac{8}{3}D$ 时）
  - Up 投影：$(B, S, D) \times (D, F) \to 2BSDF$；
  - Gate 投影：$(B, S, D) \times (D, F) \to 2BSDF$；
  - Down 投影：$(B, S, F) \times (F, D) \to 2BSDF$；
- **每层总计** ：$8BSD^2 + 4BS^2D + 16BSD^2 = 2BSD(12D + 2S)$；
- **Unembedding 反词嵌入层** ：$2BSDV$（$(B, S, D) \times (D, V) \to 2BSDV$）；
- **全模型完整前向传播总 FLOPs** ：

  $$
  \text{Forward FLOPs} = 2LBSD(12D + 2S) + 2BSDV \approx 2BSD(12LD + 2LS + V)
  $$

- **反向传播 FLOPs** ：
  - 普遍设定为 **前向传播的 2 倍** ；
  - 原因：需要同时计算关于参数和关于输入的梯度，每个梯度各需要一次矩阵乘；关于输入 $X$ 的梯度 $\partial \mathcal{L}/\partial X$ 正是传入前一层的梯度。

<span id="sec-accounting-inference-mem"></span>

### 1.7 推理显存深度分析

推理时的总显存占用公式：

$$
\text{Memory}_{\text{inference}} = \text{Model Weights} + \text{KV Cache} + \text{Peak Activations}
$$

- **模型参数量计算代码** ：

``` python
  num_params = sum(p.numel() for p in model.parameters())
  ```

- **KV Cache 尺寸公式** ：

  $$
  \text{KV Cache Size} = B \cdot S \cdot (K \cdot H) \cdot L \cdot 2
  $$

  - $B$ = Batch Size；
  - $S$ = 序列长度 Sequence Length；
  - $K$ = KV 注意力头数；
  - $H$ = 头维度 Head Dimension；
  - $L$ = 模型层数 Number of Layers；
  - 因子 2 分别对应 Key 与 Value。

- **激活值分析** ：
  - Prefill 阶段为 $O(BNS^2 + BSF)$；Decode 阶段为 $O(BNS + BF)$；
  - `torch.inference_mode()` 会在每层计算完毕后立即释放中间张量，因此显存开销仅取决于 **单层最大的峰值激活值** ；
  - 输入维度在 Prefill 为 $B \times S \times D$，在 Decode 为 $B \times T \times D$；
  - 在开启 FlashAttention 后，$S \times S$ 注意力大矩阵从不被实体化，Attention 激活值从 $O(S^2)$ 降低到 $O(S)$；
  - 峰值激活值细项：
    - $B \cdot S \cdot D$：层的输入张量；
    - $B \cdot S \cdot 3 \cdot D$：K, Q, V 投影向量；
    - $B \cdot N \cdot S^2$：注意力矩阵（不带 FlashAttention 时），对批次中每个样本、每个 Query 头都有一个 $S \times S$ 矩阵；
    - $B \cdot S \cdot F$：FFN 中间激活状态；
  - 在无 FA 时激活值随序列长度 $S$ 呈二次方增长，在有 FA 时呈线性增长；

- **系统瓶颈动态转移规律** ：
  - 在极小 Batch Size 和极短序列长度下， **静态权重占主导** ；
  - 在 Prefill 预填充阶段，长序列（大 $S$）下 **激活值中的 $S^2$ 注意力项占主导** ；
  - 在大 Batch Size（大 $B$）下， **KV Cache 与激活值同时爆发增长** 。

<span id="sec-accounting-train-mem"></span>

### 1.8 训练显存深度分析：训练四件套

训练时的总显存占用构成：

$$
\text{Memory}_{\text{train}} = \text{Model Weights} + \text{Optimizer States} + \text{Gradients} + \text{Activations}
$$

- **$P$ 个模型参数** ：
  - FP32 主权重 \[全精度或混合精度\] $\to 4P$ 字节；
  - BF16 前向/反向快速计算副本 \[混合精度\] $\to 2P$ 字节；
- **$2P$ 个优化器状态（一阶动量与二阶动量）** ：
  - AdamW 优化器状态必须保留在 FP32 \[混合精度\] $\to 8P$ 字节；
- **$P$ 个梯度** ：
  - 以 FP32 保存 $\to 4P$ 字节（即使在混合精度下，梯度使用 BF16 计算，但在累加时保留为 FP32）；
- **激活值（Activations）** ：
  - 往往是显存占用的最大主导部分：取决于 $B, S, D, L$；
  - 不开启 FlashAttention 时每层需要 $14BSD + BNS^2$ 激活值；
  - 开启 FlashAttention 后，第二项变为 $BNS$，激活值内存直接与 $BS$（总 Token 数量）成正比缩放；
  - 激活值在反向传播中用于计算参数梯度，可以通过 **梯度检查点（Gradient Checkpointing / 激活值重计算）** 牺牲额外 1 次前向计算时间来大幅消除。

<span id="sec-attention-variants"></span>

### 1.9 注意力机制变体：滑动窗口与稀疏注意力

- 标准注意力复杂度为 $O(n^2)$；
- 通过使用局部注意力，可使 KV Cache  **与序列长度完全解耦** ：
  - 一旦某个 Token 的历史位置滑出了当前设定的窗口范围，就可以直接将其 KV 丢弃释放；
- **滑动窗口注意力（Sliding Window Attention）** ：
  - 每个 Token 仅仅关注其前序的最近 $W$ 个 Token，因此计算量从 $O(n^2)$ 降低到 $O(nW)$；
  - $n$ 个 Token 每个做 $O(W)$ 的计算（关注 $W$ 个 Keys/Values）；
- **稀疏注意力（Sparse Attention）** ；
- **交织注意力（Interleave Attention）** ：在网络中将局部注意力层与全局注意力层进行交替穿插（interleave local attention with global attention，例如 Mistral / Gemma-2）。

![示意图 2：标准全自注意力 (Causal O(N²)) 滑动窗口注意力 (Window W=3, O(NW))](/translations/alisa-llm-infrastructure/diagram-2.svg)

图 1.3：全注意力因果三角掩码 vs 滑动窗口带状注意力掩码（窗口外历史 KV 可直接丢弃）

<span id="sec-rmsnorm-details"></span>

### 1.10 RMSNorm 深度原理解析

- 归一化通过防止梯度爆炸/消失来稳定训练；
  - 单一统一的缩放过于严苛：不同的特征通常需要不同的数值量级；
  - 参数 $\gamma$ 将归一化的稳定性与逐维度量级变化能力进行了结合；
- $\gamma \in \mathbb{R}^D$ 是学习到的逐维度重缩放参数：
  - 归一化步骤强制隐藏状态具有单位 RMS，但这破坏了模型所学习到的任何尺度信息；
- $\gamma$ 重新赋予了网络对特征幅度的逐维度控制权：
  - $\gamma_i > 1 \implies$ 放大（amplify）维度 $i$；
  - $\gamma_i < 1 \implies$ 抑制（suppress）维度 $i$；
  - $\gamma_i \approx 0 \implies$ 关闭杀死（kill）维度 $i$。

<span id="sec-swiglu-details"></span>

### 1.11 SwiGLU FFN 门控机制

- 在 SwiGLU 中，$\mathbf{G}$ 与 $\mathbf{U}$ 均贡献特征内容：
  - $\mathbf{U}$ 提供了一条学习到的特征表示分支，而 $\mathbf{G}$ 提供了另一条特征表示分支（并通过其自身的置信度进行自门控）；

<span id="sec-rope-math"></span>

### 1.12 RoPE 旋转位置编码数学证明与复数本质

- **RoPE 仅旋转 Query 和 Key 向量，不旋转 Value 向量** ：
  - 位置信息只需要影响 Token 之间「谁应该关注谁」，而不需要影响在注意力通道中实际传递的信息内容；
- 对于处于位置 $m$ 的 Query 向量 $\mathbf{q}$ 和处于位置 $n$ 的 Key 向量 $\mathbf{k}$，我们希望内积 $\mathbf{q} \cdot \mathbf{k}$  **仅取决于相对位置 $m - n$** ：
  - 我们希望找到映射函数 $f$，使得内积 $\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle$ 是一个仅以相对形式编码信息的函数 $g$，例如：

  $$
  \langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, n - m)
  $$

  - RoPE 正是该问题的一种优雅解法：

  $$
  f(\mathbf{x}, m) = \mathbf{R}_{m\theta}\mathbf{x}, \quad g(\mathbf{q}, \mathbf{k}, n - m) = \mathbf{q}^\top \mathbf{R}_{(n - m)\theta} \mathbf{k}
  $$

  - **严密代数证明** ：

  $$
  \begin{aligned}
  \langle \mathbf{R}_{m\theta}\mathbf{q}, \mathbf{R}_{n\theta}\mathbf{k} \rangle
  &= (\mathbf{R}_{m\theta}\mathbf{q})^\top \mathbf{R}_{n\theta}\mathbf{k} \\
  &= \mathbf{q}^\top \mathbf{R}_{m\theta}^\top \mathbf{R}_{n\theta}\mathbf{k} \\
  &= \mathbf{q}^\top \mathbf{R}_{-m\theta}\mathbf{R}_{n\theta}\mathbf{k} \quad \left(\text{因为正交矩阵 } \mathbf{R}_\alpha^\top = \mathbf{R}_{-\alpha}\right) \\
  &= \mathbf{q}^\top \mathbf{R}_{(n - m)\theta}\mathbf{k} \quad \left(\text{因为旋转复合 } \mathbf{R}_\alpha \mathbf{R}_\beta = \mathbf{R}_{\alpha+\beta}\right)
  \end{aligned}
  $$

- 旋转使相对位置信息进入内积；不能据此断言任意 Query / Key 的内积都会随距离单调衰减；
- 对于二维向量 $\mathbf{x} = [x_1, x_2]$，旋转角度 $\theta$ 的变换为：

  $$
  \mathbf{R}_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
  $$

   对于位置 $m$ 与头维度索引 $i$，旋转角度为 $m\theta_i$：

  $$
  \mathbf{R}_{m\theta}\mathbf{x} = \begin{bmatrix} x_1 \cos(m\theta_i) - x_2 \sin(m\theta_i) \\ x_1 \sin(m\theta_i) + x_2 \cos(m\theta_i) \end{bmatrix}
  $$

- 对于 $H$ 维嵌入向量，将其划分为 $H/2$ 个二维对，并以不同频率对每一对独立应用旋转：
  - $\Theta$ 是 RoPE 的唯一超参数，通常取 10,000（在 HuggingFace 中称为 `rotary_base`）：
    - 它定义了模型能够原生分辨的最长相对距离；
    - 当 $m\theta_i = 2\pi$ 时完成一次完整旋转 $\implies m = \frac{2\pi}{\theta_i}$；
    - 最慢（最小）的 $\theta_i$ 为 $\Theta^{-1} \implies m = 2\pi\Theta$；
    - 因此最慢的维度对在经历  **$2\pi\Theta$ 个位置（即约 62,831 个 Token）** 后才完成一整圈旋转。
  - 对于每个维度对 $i$，$\theta_i$ 公式为：

  $$
  \theta_i = \Theta^{-2i/H}
  $$

  - 这提供了指数间隔（与原始 Transformer 的正弦绝对位置编码完全相同的设计）：
    - 频率呈对数均匀分布（log uniform spread）$\to$ 有效覆盖不同尺度的距离；
  - **低频对（大 $i \to$ 小 $\theta_i$）** ：旋转极慢 $\to$ 相邻位置间变化极微小 $\to$ 编码全局长距离宏观信息；
  - **高频对（小 $i \to$ 大 $\theta_i$）** ：旋转极快 $\to$ 相邻位置差异巨大 $\to$ 赋予强大的局部判别能力。
- 单位置处的完整旋转矩阵是 **分块对角矩阵（Block-Diagonal）** ，每个 $2 \times 2$ 块处理一个维度对：

  $$
  \mathbf{R}_m = \begin{bmatrix} \ddots & & \\ & \mathbf{R}_m^{(i)} & \\ & & \ddots \end{bmatrix}
  $$

- **复数形式等价重构** ：
  - 将每个实数对 $[x_{2i}, x_{2i+1}]$ 视为一个复数 $z = x_{2i} + i x_{2i+1}$；
  - 旋转角度 $\theta$ 等价于复数乘法乘以 $e^{i\theta}$（欧拉公式 Euler's Theorem：$e^{i\theta} = \cos\theta + i\sin\theta$）：

  $$
  \begin{aligned}
  z e^{i\theta} &= (a + bi)(\cos\theta + i\sin\theta) \\
  &= (a\cos\theta - b\sin\theta) + i(a\sin\theta + b\cos\theta)
  \end{aligned}
  $$

  - 这与在复平面中应用旋转矩阵是完全等价的：

  $$
  \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix}
  $$

- 在工程实践中，为了最大化 GPU 吞吐量，并不显式构建全零的稀疏旋转大矩阵，而是直接进行 **逐元素乘法累加（Element-wise Multiplication）** ：

  $$
  \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \odot \begin{bmatrix} \cos(m\theta) \\ \cos(m\theta) \end{bmatrix} + \begin{bmatrix} -x_2 \\ x_1 \end{bmatrix} \odot \begin{bmatrix} \sin(m\theta) \\ \sin(m\theta) \end{bmatrix}
  $$

<span id="ch2-inference"></span>

## 2 大模型推理系统

<span id="sec-latency-throughput"></span>

### 2.1 延迟与吞吐量

- **延迟（Latency）** ：完成单个请求所需的时间，以秒为单位衡量；
- **吞吐量（Throughput）** ：在所有请求中，单位时间内能够处理的 Token 数量（或请求数），以 Tokens/Second 为单位衡量。

<span id="sec-batching-packing"></span>

### 2.2 批处理与打包技术演进

- **传统批处理（Traditional Batching）** ：收集 $N$ 个请求，组合在一起处理，等待所有序列全部完成，然后才收集下一批：
  - 如果一个序列生成 500 个 Token，而另一个只生成 10 个 Token，短序列就必须空转等待；
- **连续批处理（Continuous Batching）** ：一旦某个序列生成结束，立刻在其空位上插入新的请求，无需等待整个批次全部结束：
  - 批次始终保持打满状态；
- **选择性批处理（Selective Batching）** ：巧妙地混合处于 Prefill 阶段与 Generation（Decode）阶段的序列：
  - 核心思想：Prefill 阶段是计算密集型（Compute-heavy），而 Decode 阶段是访存带宽受限（Memory-bound）；
- **序列打包（Sequence Packing）** ：将样本沿长度拼接直到达到最大序列长度，使用注意力掩码防止样本间交叉污染；
- **Token 预算批处理（Token-Budget Batching）** ：将样本分批，使得每个批次内的总 Token 数量（在该批次填充到内部最大序列长度后）不超过设定的预算：
  - 通常在模型微调（Finetuning）中采用；
  - 非常合理，因为 GPU 显存占用直接由 `batch_size x sequence_length` 决定。

<span id="sec-speculative-decoding"></span>

### 2.3 投机解码原理与无损数学证明

- **投机解码** 利用了「Prefill 比自回归生成更快」的事实；
- 从草稿模型（Draft Model）$q$ 生成 $K$ 个 Token；
- 使用目标模型（Target Model）$p$ 评估这些 Token：
  - 以如下概率依序接受每个草稿 Token $x$：

  $$
  \min\left(1, \frac{p(x)}{q(x)}\right)
  $$

  - 若 $p(x) > q(x)$，则 **必然接受** 它；
  - 若发生拒绝，截断后续草稿，从重新归一化后的修正分布中采样：

    $$
    \max(0, p(x) - q(x))
    $$

     从第一个被拒绝的 Token 开始采样；
  - **始终能从 Teacher 目标模型免费获得一个 Token** ：因为在对草稿 Token 打分时，已经顺带算出了下一个 Token 的 Logits。

 **严密全概率公式等价性证明：**

$$
P(\text{输出 Token } x) = P(\text{草稿生成 } x) \times P(\text{接受 } x) + P(\text{采样 Token 被拒绝}) \times P(\text{重采样选出 } x)
$$

- **情况 1（第一项）：$x$ 直接从草稿中被接受** ：

  $$
  q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))
  $$

- **情况 2（第二项）：$x$ 在草稿被拒绝后被选中** ：
  - 草稿生成 $x'$ 且被拒绝的概率：
    - 若 $p(x') > q(x')$ 则为 0；
    - 否则为：

    $$
    q(x') \cdot \left(1 - \frac{p(x')}{q(x')}\right) = q(x') - p(x')
    $$

  - 因此拒绝草稿的总概率为：

  $$
  \begin{aligned}
  P(\text{拒绝}) &= \sum_{x'} \max(0, q(x') - p(x')) \\
  &= \sum_{x'} \max(0, p(x') - q(x'))
  \end{aligned}
  $$

  - 第二步推导成立是因为概率分布 $p$ 和 $q$ 全域求和均为 1；
  - 发生拒绝时，从归一化分布中抽样的概率为：

  $$
  \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}
  $$

  - 因此拒绝后获得 $x$ 的总概率为：

  $$
  \sum_{x'} \max(0, p(x') - q(x')) \cdot \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))} = \max(0, p(x) - q(x))
  $$

- **合并两种情况** ：

  $$
  \min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x)
  $$

<span id="sec-kv-cache"></span>

### 2.4 KV Cache 机制与 PyTorch 实现

- 带缓存的单 Token 前向传播：
  - Cache 张量形状应为 `(batch, num_heads, max_seq_len, head_dim)`；
  - 工程实践中，我们将 Cache 保存在 Self-Attention 模块内部的 `self.kv_cache` 中。

``` python
# 根据 max_seq_len 预先分配缓存
kv_cache = [
    {
        'k': torch.zeros(batch, num_heads, max_seq_len, head_dim, device='cuda', dtype=torch.float16),
        'v': torch.zeros(batch, num_heads, max_seq_len, head_dim, device='cuda', dtype=torch.float16),
    }
    for _ in range(num_layers)
]

# 在注意力运算中，使用 cache[:, :, :position+1, :] 作为 keys / values

def forward_with_cache(model, new_token, kv_cache, position):
    """
    不再处理完整的序列，而是仅处理当前的新 Token，
    并直接复用先前所有位置缓存的 K, V。
    """
    # 仅嵌入新 Token
    x = model.embed(new_token)  # (batch, 1, d_model)

    for layer_idx, layer in enumerate(model.layers):
        q, k, v = layer.qkv_proj(x).chunk(3, dim=-1)

        # 更新缓存
        kv_cache[layer_idx]['k'][:, :, position, :] = k.squeeze(2)
        kv_cache[layer_idx]['v'][:, :, position, :] = v.squeeze(2)

        # 对全部已缓存位置执行注意力
        k_full = kv_cache[layer_idx]['k'][:, :, :position+1, :]
        v_full = kv_cache[layer_idx]['v'][:, :, :position+1, :]

        x = attention(q, k_full, v_full)
        x = layer.ffn(x)

    return model.lm_head(x)
```

<span id="sec-reducing-kv-cache"></span>

### 2.5 降低 KV Cache 尺寸的策略

 **1. 降低 KV Cache 维度：**

- 在标准 **多头注意力（MHA）** 中，KV Cache 大小为：`num_layers × num_heads × seq_len × head_dim × 2 (K 和 V)`；
- 在 **多查询注意力（MQA）** 中，所有注意力头共享相同的 K 和 V，但每个头拥有独立的 Q：
  - KV Cache 大小缩小了 `num_heads` 倍；
  - 推理速度大幅加快，显存效率显著提升；
- 在 **分组查询注意力（GQA）** 中，将注意力头划分为若干组，组内共享 K 和 V：
  - MHA 与 MQA 之间的折中平衡方案；
- MQA 和 GQA 在不同头之间共享了 KV，因此会牺牲一部分头级别的表征能力；
- **多头潜在注意力（MLA）** 由 DeepSeek-V2 提出：
  - 不再在 KV Cache 中存储形状为 `(seq_len, num_heads × head_dim)` 的 Keys 和 Values，而是仅缓存一个极小的潜在向量（Latent Vector），形状为 `(seq_len, latent_dim)`；
  - 然后在每个 Decode 解码步中，再将潜在向量重新投影放大回完整尺寸；
- MLA 保持了每个头独立的 KV 表征，但将它们压缩到了一个共享的潜在空间中：
  - 在推理时增加了一些额外的计算开销（从潜在向量到 KV 的线性投影）；
  - MLA 无法直接兼容标准 RoPE（因而 DeepSeek 采用了解耦 RoPE 方案）；
- **跨层注意力（Cross-Layer Attention）** ：在网络不同层之间共享 KV Cache。

 **2. 局部注意力（Local Attention）：**

- 标准注意力是 $O(n^2)$，使用局部注意力可以使 KV Cache  **完全独立于整个序列长度** ；
- 一旦某个 Token 滑出了当前设定的窗口，就可以直接丢弃其缓存。

<span id="sec-sampling-strategies"></span>

### 2.6 采样策略工业实现

``` python
def sample(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature

    if top_k is not None:
        values, indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, indices, values)

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率高于阈值的 Token
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        indices_to_remove = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

<span id="sec-flash-attention"></span>

### 2.7 Flash Attention 核心原理与 Tiling 切分

- **标准注意力机制的内存瓶颈** ：
  - 显存问题在于中间矩阵 `attn_weights` 形状为 `(batch, num_heads, seq_len, seq_len)`；
  - 在 `seq_len` = 8192、32 个头、FP16 精度下，仅该矩阵就需要 $8192^2 \times 32 \times 2 \text{ 字节} \approx 4\text{ GB}$；

``` python
import torch
import torch.nn.functional as F
import math

def standard_attention(q, k, v):
    # q, k, v 形状均为: (batch, num_heads, seq_len, head_dim)
    scale = math.sqrt(q.size(-1))

    # 实例化完整的 N × N 注意力矩阵
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, num_heads, seq_len, seq_len)
    attn_weights = F.softmax(attn_weights, dim=-1)

    output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
    return output
```

- **为什么标准注意力是访存带宽受限（Memory-bound）的？**
  - $N \times N$ 注意力矩阵先被写入 GPU 主存 HBM，然后读回（用于计算 Softmax），再次写入 HBM，最后再次从 HBM 读出（与 $V$ 进行矩阵乘）；
- **Flash Attention**  利用在线 Softmax 技巧在分块中计算注意力，将所有中间计算结果保存在极速片上  **SRAM**  中，而不是将完整的注意力矩阵写回 GPU 主显存：
  - 将注意力的激活值显存占用从 $O(n^2)$ 降低到 $O(n)$；
  - 运行速度显著加快，因为显存带宽不再成为瓶颈；
  - 在推理时可通过 HuggingFace 的 `model.to_bettertransformer()` 或在 `from_pretrained()` 中设置 `attn_implementation="flash_attention_2"` 开启；

``` python
def flash_attention(q, k, v):
    # 输入形状相同: (batch, num_heads, seq_len, head_dim)
    # PyTorch 自动选择最佳后端 (Flash Attention, memory-efficient 或标准 math)
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return output
```

- FlashAttention 从不实体化完整的 $N \times N$ 矩阵，而是将其切分为能够完整容纳在 SRAM 中的小块（Tiles）进行计算；
- **概念上的计算 5 步流程** ：
  1.  加载 $Q$ 的一个分块（例如 64 行）；
  2.  加载 $K$ 和 $V$ 的一个分块（例如 64 列）；
  3.  在 SRAM 内部计算该分块的注意力分数，应用增量 Softmax，并乘以对应的 $V$ 分块——全部在 SRAM 内部完成；
  4.  仅将最终累积的输出写回 HBM；
  5.  对所有分块循环重复上述过程。
- 由于每一个 `(batch, head)` 组合是完全独立的，因此系统拥有 `batch × num_heads` 个并行 Worker 同时在 Q, K, V 的 `m × head_dim` 分块上工作（其中 `m` 为分块大小）；
- **Flash Attention 是一种精确计算方法，而非近似算法** （数学上严格等价）；
- `is_causal=True` 掩码标志被高效融合进底层 CUDA Kernel 中，无需在物理显存中实例化 Mask 掩码矩阵。

<span id="ch3-scaling-laws"></span>

## 3 缩放定律与算力分配

<span id="sec-mup"></span>

### 3.1 最大更新参数化 ($\mu P$, Maximal Update Parameterization)

- **核心问题** ：在小尺度模型上搜索找到的最佳超参数，无法直接泛化到更大尺度的模型上；
- **标准参数化** ：当网络宽度变化时，不同层参数的更新幅度会出现不一致；
- **$\mu P$** ：通过按层调整参数初始化方差和学习率，使得 **更新幅度相对于权重本身的相对比例在不同网络宽度之间保持恒定** ；
- $\mu P$ 主要调整的是 **宽度缩放（Width Scaling）** 。

<span id="sec-scaling-compute"></span>

### 3.2 算力-学习率与算力-Loss 拟合公式

 **拟合学习率与总算力预算：**

$$
\text{LR}(C) = \beta C^{-\alpha} \implies \log \text{LR}(C) = \log \beta - \alpha \log C
$$

 **拟合训练损失与总算力预算：**

- 公式中必须包含不可约损失项 $\mathcal{L}_\infty$（代表数据本身的不可压缩信息熵），否则当算力 $C \to \infty$ 时，损失 $\mathcal{L} \to 0$：

$$
\mathcal{L}(C) = \mathcal{L}_\infty + \beta C^{-\alpha} \implies \log(\mathcal{L}(C) - \mathcal{L}_\infty) = \log \beta - \alpha \log C
$$

 <span id="sec-least-squares"></span>

### 3.3 最小二乘法闭式解与非线性迭代

- 如何拟合上述方程？使用 **最小二乘法（Least Squares）** ；
- 最小二乘法的优化目标是最小化残差平方和：

  $$
  S = \sum_i (y_i - f(x_i))^2
  $$

  - 分为普通（或线性）最小二乘法（Ordinary / Linear Least Squares）与非线性最小二乘法（Non-linear Least Squares）；
- 线性最小二乘法拥有 **解析闭式解（Closed-Form Solution）** ：
  - 对于方程 $y = \mathbf{X}\beta$，闭式解为：

  $$
  \beta = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top y
  $$

- 非线性最小二乘法通过 **迭代微调优化（Iterative Refinement）** 求解。

<span id="ch4-gpus"></span>

## 4 GPU 硬件体系结构

<span id="sec-gpu-memory-hierarchy"></span>

### 4.1 HBM 与片上 SRAM 内存层级

- **高带宽显存（High Bandwidth Memory, HBM）** 是 GPU 的主显存：
  - 从 GPU 算力核心的角度看，它属于慢速内存（Slow Memory）；
  - A100 的容量为 40GB 或 80GB；
- **静态随机存取内存（Static RAM, SRAM）** 是片上高速小型缓存：
  - 单张 A100 GPU 上的 SRAM 总量仅约为  **~20MB** 。

<span id="ch5-precision"></span>

## 5 数值精度与量化系统

<span id="sec-mixed-precision"></span>

### 5.1 混合精度训练机制与下溢消除

- **混合精度（Mixed Precision）训练流程** ：
  - 主权重（Master Weights）保留在 FP32；
  - 为前向和反向传播制作权重的 BF16 计算副本；
  - 激活值在 BF16 下计算；
  - 梯度在 BF16 下计算，但最终 **累加到 FP32 中** ；
    - 这避免了将微小梯度加到巨大权重上引起的精度截断问题；
    - BF16 在接近 0 的区域拥有更高精度，因此它可以表示像 $0.0001$ 这样微小的梯度，但无法精确表示更新后的权重值 $1.0001$；
    - 我们只需要让累加后的梯度落在 FP32 中即可，单个梯度相对于权重本身往往很小；
    - 每个参数上的 `.grad` 张量与该参数保持相同的数据类型，因此单个梯度会被转换为 FP32。
- **直觉理解（Intuition）** ：
  - 矩阵乘法对舍入误差和舍入噪声具有高度容忍度，因此在前向/反向中使用 BF16 完全可行；
  - 保留 FP32 主权重有助于累积较小更新；是否使用独立主权重取决于具体训练实现。
- **激活值比权重更难被量化** 。

<span id="sec-precision-types"></span>

### 5.2 精度选项全景对比

- **FP32** ：全精度（4 字节）；
- **FP16** ：半精度显存（2 字节）；
- **BF16** ：显存大小与 FP16 相同（2 字节），但具备更好的数值稳定性（拥有与 FP32 相同的超大动态范围）；
- **INT8** ：仅占 FP32 四分之一显存（1 字节），需要进行量化；
- **INT4** ：甚至更小（0.5 字节）。

<span id="sec-memmap-load"></span>

### 5.3 数据加载 memmap 与模型 BF16/INT8 加载

- **数据加载** ：
  - 使用 `memmap`（内存映射）可以避免一次性将整个巨型数据集加载进物理内存中；
- **在 BF16 下加载模型** ：
  - 使用 HuggingFace `.from_pretrained()` 加载模型时：
    - `torch_dtype=torch.bfloat16` 是 BF16 的推荐首选（权重和激活值均为 BF16）；
    - `load_in_8bit=True` 用于量化，调用 `LLM.int8()`：
      - 权重以 INT8 保存，激活值以 FP16 保存；
      - 通过权重压缩带来显存节省；
      - 由于激活值保留在更高精度，因此并不等同于全 INT8 计算；
  - `model.half()` 或 `model.to(torch.bfloat16)` 可将模型转换为 FP16 / BF16：

``` python
model = MyModel()
model.load_state_dict(torch.load('model.pt'))
model = model.half()
```

- 上面的 `model.half()` 将浮点参数转换为 FP16；若需要 BF16，应使用 `model.to(torch.bfloat16)`。算子的实际执行精度还取决于实现；
- 是否使用 BF16，应结合硬件支持、模型数值表现与精度评估决定。

<span id="sec-autocast-bnb"></span>

### 5.4 自动混合精度与 bitsandbytes 量化实战

- `torch.autocast()` 执行自动混合精度：
  - 权重保持在 FP32，算子选择性地使用 BF16 或 FP32；
  - 它管理每个算子的运行精度，但并不管理主权重；
  - 比直接将整个模型以 BF16 加载占用更多显存，但潜在上更加稳定；
  - 矩阵乘法走 FP16（容忍低精度），Softmax 走 FP32（需要数值稳定性），LayerNorm 走 FP32（归约求和需要高精度）；
  - 当模型处于 FP32 且我们无法轻易转换它，或者在纯 BF16 下观察到数值问题时非常有用；
  - `dtype` 参数指定了「低精度」的类型：

``` python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(x)
```

- 要使用 `bitsandbytes`，将 `nn.Linear` 替换为 `bnb.nn.Linear8bitLt` 或 `bnb.nn.Linear4bit`：

``` python
def replace_linear_with_8bit(model):
    """递归将所有 nn.Linear 替换为 bnb.nn.Linear8bitLt"""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            # 创建量化替代层
            new_layer = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
            )
            # 拷贝权重 (当移动到 CUDA 时会自动触发量化)
            new_layer.weight = bnb.nn.Int8Params(
                child.weight.data,
                requires_grad=False,
            )
            if child.bias is not None:
                new_layer.bias = nn.Parameter(child.bias.data)

            setattr(model, name, new_layer)
        else:
            # 递归遍历子模块
            replace_linear_with_8bit(child)

    return model

# 使用范例
model = MyTransformer()
model.load_state_dict(torch.load('model.pt'))
model = replace_linear_with_8bit(model)
model = model.to('cuda')  # 在此步完成量化
```

<span id="ch6-parallelism"></span>

## 6 分布式并行系统

<span id="sec-5d-parallelism"></span>

### 6.1 5D 并行体系总览与强扩展性

- **数据并行（Data Parallelism）** 跨设备切分 Batch，而 **模型并行（Model Parallelism）** 跨设备切分单次前向传播的计算；
  - FSDP 为了内存效率在各 Rank 间分片切分模型参数，但每个 Rank 在参数即将被使用前通过 All-Gather 临时重构它，因此每个 Rank 仍然计算完整的前向传播；
- **数据并行的局限性** ：
  - 要求设备数 $M < B$，这不一定是好事，因为我们不希望总 Batch Size $B$ 超过「临界批次大小（Critical Batch Size）」；
  - 单模型可能依然无法塞入单张设备（即使是 ZeRO-3 也无法降低每张设备上的单卡激活值显存占用）；
- **强扩展性（Strong Scaling）** ：增加用于训练的芯片数量能够带来吞吐量（FLOPs/Second）的同比例线性增长；
- **扩展维度的权衡** ：DP 扩展吞吐量，TP / PP 扩展模型显存容量，SP 扩展激活值显存容量；
- **5D 并行维度** ：
  - **数据并行 (DP)** ：跨设备切分数据 Batch；
  - **张量并行 (TP)** ：跨设备切分单个层内的权重矩阵；
  - **流水线并行 (PP)** ：跨设备切分模型的不同层 / 阶段；
  - **序列并行 (SP)** ：跨设备切分输入序列长度维度；
  - **专家并行 (EP)** ：跨设备分布式调度 MoE 模型中的不同专家。

<span id="sec-collectives"></span>

### 6.2 核心集合通信原语与前后向对偶性

- **Broadcast（广播，一对多，相同数据）** ：一个 GPU 持有数据，并向所有其他 GPU 发送一份完全相同的副本；
- **All-Gather（全收集，多对多）** ：每个 GPU 持有一块数据分片，所有 GPU 最终都获得完整的数据集合：
  - 沿某个轴 **消除切分分片（Removes Sharding）** ：

  $$
  \operatorname{AllGather}_Y : \mathbf{A}[I, J_Y] \to \mathbf{A}[I, J]
  $$

- **Reduce-Scatter（规约分散）** ：每个 GPU 持有未归约的数据，通过规约聚合结合，最终结果切分分散在各个 GPU 上：
  - 与 All-Gather 非常相似，但不再是单纯保留每个分片，而是将它们求和叠加；
  - 沿某个轴 **增加切分分片（Adds Sharding）** ：

  $$
  \operatorname{ReduceScatter}_{Y, J} : \mathbf{A}[I, J]\{U_Y\} \to \mathbf{A}[I, J_Y]
  $$

- **All-Reduce（全规约）** ：每个 GPU 持有未归约的数据，通过规约求和结合，每个 GPU 最终都获得全局求和结果：

  $$
  \operatorname{AllReduce}_Y \mathbf{A}[I, J]\{U_Y\} \to \mathbf{A}[I, J]
  $$

- **Ring All-Reduce** ：Reduce-Scatter + All-Gather（每个进程仅与两个相邻邻居通信）：
  - 在初始阶段，每个 GPU 持有未归约的数据；
  - Reduce-Scatter：通过规约结合数据，每个 GPU 获得一个规约后的子集分片（完成了所有的算术运算，没有任何多余的冗余拷贝）；
  - All-Gather：每个 GPU 获取所有规约后子集的完整集合（完成了所有的拷贝通信，没有任何多余的算术运算）；
- **极度重要的通信标度定律** ：
  - **对于 All-Gather、Reduce-Scatter 和 All-Reduce，可以先用数据量与有效带宽估算主要传输成本；实际耗时还受参与设备数、算法轮次、通信延迟和拓扑影响。**
- **Reduce-Scatter 与 All-Gather 互为反向传播对偶** ：
  - 前向 All-Gather $\to$ 反向必然是 Reduce-Scatter：
    - All-Gather 在前向中向每台设备广播同一个 Chunk，各卡参与不同的下游计算；
    - 在上游向外分支处，梯度按链式法则相加（若 $x = a + b$，则 $\partial f/\partial x = \partial f/\partial a \cdot \partial a/\partial x + \partial f/\partial b \cdot \partial b/\partial x$）；
    - 这正好是 Reduce-Scatter：将所有上游梯度汇总求和到该 Chunk 的起源设备；
    - 前向扩散分支（Fan-out）$\to$ 反向求和聚合（Sum）；
  - 前向 Reduce-Scatter $\to$ 反向必然是 All-Gather：
    - Reduce-Scatter 在前向中将多个输入相加为一个 Chunk；
    - 在反向传播中，求和节点将上游梯度拷贝复制给每一个被加数；
    - 这正好是 All-Gather：将每个 Chunk 的梯度广播复制回其贡献者；
    - 前向求和聚合（Sum）$\to$ 反向扩散分支（Fan-out）；
  - **这意味着 All-Reduce 的反向传播是另一个完全相同的 All-Reduce。**

<span id="sec-mesh-notation"></span>

### 6.3 设备网格切分符号与 4 种分布式矩阵乘

- **网格切分符号（Partitioning Notation）** ：
  - 设备网格拥有坐标轴 $(X, Y)$，矩阵 $\mathbf{A}$ 拥有坐标轴 $(I, J)$；
  - $I_X$：将 $\mathbf{A}$ 的行沿着设备网格的列（沿 $X$ 网格轴）进行切分；
  - $I_Y$：将 $\mathbf{A}$ 的行沿着设备网格的行（沿 $Y$ 网格轴）进行切分；
  - $J_Y$：将 $\mathbf{A}$ 的列沿着设备网格的行进行切分；
  - $J_X$：将 $\mathbf{A}$ 的列沿着设备网格的列进行切分；
  - $I_{XY}$：将 $\mathbf{A}$ 的行在展平后的整个 $XY$ 网格所有设备上切分；
  - $I$：不切分 $\mathbf{A}$ 的行；
  - **某个网格维度未出现，意味着数据在该维度上完整复制（Replicated）** （例如 $Y$ 未出现：表示每一列包含相同的数据）。
- 矩阵乘法的极好特性：当矩阵乘数写为分块形式时，乘积可直接用分块矩阵乘法表达；
- **4 种矩阵乘法切分案例** ：
  - **Case 1：两个矩阵均没有切分收缩维度（Contracting Dimension）** ：

    $$
    \mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \to \mathbf{C}[I_X, K_Y]
    $$

    - **完全不需要任何通信** ；
    - 在本地执行局部块矩阵乘法；
    - 输出自然以期望的形式被切分；
  - **Case 2：矩阵 $\mathbf{A}$ 或 $\mathbf{B}$ 之一切分了收缩维度** ：

    $$
    \mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \to \mathbf{C}[I, K]
    $$

    - 先通过 All-Gather 收集 $\mathbf{A}$ 的各个分片，使每台设备持有完整副本，再与 $\mathbf{B}$ 相乘：

    $$
    \operatorname{AllGather}_X[I, J_X] \to \mathbf{A}[I, J]
    $$

    $$
    \mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \to \mathbf{C}[I, K]
    $$

  - **Case 3：矩阵 $\mathbf{A}$ 和 $\mathbf{B}$ 均切分了收缩维度** ：

    $$
    \mathbf{A}[I, J_X] \cdot \mathbf{B}[J_X, K] \to \mathbf{C}[I, K]
    $$

    - 矩阵乘法可以直接进行，但每台设备计算出的只是期望乘积的局部部分和（Partial Sum）；
    - 沿 $X$ 维度的每台设备持有不同的部分和（记作 $\mathbf{C}[I, K]\{U_X\}$，表示沿 $X$ 网格轴未归约）；
    - 通过跨 $X$ 轴执行一次 All-Reduce 完成最终求和：

    $$
    \operatorname{AllReduce}_X \mathbf{C}[I, K]\{U_X\} \to \mathbf{C}[I, K]
    $$

    - 结果使得每台设备拥有完全相同的完整求和值；
  - **Case 4：矩阵 $\mathbf{A}$ 和 $\mathbf{B}$ 的非收缩维度沿着同一个轴被切分** 。

<span id="sec-data-parallelism"></span>

### 6.4 数据并行体系：原生 DDP 到 ZeRO-1/2/3

 **原生数据并行（Naive DDP）：**

- 将大小为 $B$ 的批次内的样本切分到 $M$ 台设备上，并交换梯度；
- **执行步骤** ：
  1.  在本地 Micro-batch 上运行前向传播；
  2.  计算所有参数的梯度；
  3.  在所有 GPU 之间 *All-Reduce* 梯度，使每张卡获得平均梯度；
  4.  每张 GPU 独立更新所有参数（每张 GPU 在此步做完全相同的工作）；
- 当模型能够放入单张设备时，应始终首选此方案；
- **通信仅发生在反向传播阶段** ；
- 允许通过增加设备数量任意扩展批大小。

 **ZeRO Stage 1：切分优化器状态（Shard Optimizer States）：**

- 每台设备仅持有 Adam 优化器向量的 $1/M$，并且仅更新它所维护优化器状态的那部分参数；
- **执行步骤** ：
  1.  在本地 Micro-batch 上运行前向传播；
  2.  计算所有参数的梯度；
  3.  *Reduce-Scatter* 梯度：每张 GPU 获得属于自己参数子集的归约梯度；
  4.  每张 GPU 使用其维护的优化器状态子集更新对应的参数子集；
  5.  *All-Gather* 收集更新后的全部参数；
- **关键收益** ：因为 1 次 All-Reduce 的通信代价与 Reduce-Scatter + All-Gather 完全相同， **因此该阶段完全没有引入任何额外的通信开销，获得了纯粹免费的显存收益（Free Memory Wins）。**

 **ZeRO Stage 2：切分优化器状态 + 梯度（Shard Optimizer States + Gradients）：**

- 同时切分梯度，每台设备仅保留 $1/M$ 的梯度；
- **执行步骤** ：
  1.  在本地 Micro-batch 上运行前向传播；
  2.  每当某层的梯度计算完成，立即对其执行 *Reduce-Scatter*，每张 GPU 仅保留属于自己的梯度分片，并立即释放其余梯度显存；
  3.  每张 GPU 使用其维护的优化器状态更新参数子集；
  4.  *All-Gather* 收集更新后的参数。

 **ZeRO Stage 3：切分优化器状态 + 梯度 + 模型参数（FSDP）：**

- 同时切分模型参数，每台设备仅持有 $1/N$ 的模型参数；
- 在前向计算每层前*即时 All-Gather* 获取该层参数，计算完毕后立即丢弃；
- 能够训练单卡完全无法装下的超大模型；
- **通信代价分析** ：在前向计算每层前即时 All-Gather 获取该层参数，计算完毕后立即丢弃；可训练单卡完全装不下的超大模型；Stage 3 还需要考虑前向与反向的参数聚合、是否重新分片以及通信重叠，不能笼统认为三个阶段通信成本完全相同。

<span id="sec-pipeline-parallelism"></span>

### 6.5 流水线并行 (PP) 与 1F1B 调度

- **朴素模型并行的缺陷** ：如果简单地把不同层移到不同 GPU，在每层计算前把隐层状态传过去，这非常低效，完全无法提升系统的吞吐量；
- **原生异步流水线并行** ：通过 `Pipe` 将批次拆分为微批次（Micro-batches）：

``` python
from torch.distributed.pipeline.sync import Pipe

model = nn.Sequential(
    nn.Linear(512, 512).to('cuda:0'),
    nn.ReLU().to('cuda:0'),
    nn.Linear(512, 512).to('cuda:1'),
    nn.ReLU().to('cuda:1'),
)

model = Pipe(model, chunks=8)  # 将 Batch 划分为 8 个 micro-batches
output = model(x)
```

<span id="sec-tensor-parallelism"></span>

### 6.6 Megatron-LM 张量并行 (TP) 与工业实现

- **FSDP 与张量并行可以极高效地协同组合** ：
  - 切分 Batch 维度 $B$ 降低了 All-Gather 的通信张量体积（sharding batch dimension B reduces the size of all-gathers）；
  - 切分 FFN 隐层维度 $F$ 降低了 FSDP 跨卡通信的开销；

- **MLP 层的张量并行设计（两层线性映射中间夹激活）** ：

  - 采用  **列并行（Column Parallel，针对 $W^{\text{up}}$） $\to$ 逐元素激活 $\to$ 行并行（Row Parallel，针对 $W^{\text{down}}$）** ；
  - 第一层权重按列切分，每台设备计算自己的切片并在本地独立应用非线性激活；
  - 第二层权重按行切分，每台设备计算出一个局部部分结果，最后仅执行一次 *All-Reduce* 求和；
  - **核心代数原理（The Key Insight）** ：

``` python
  y = h @ W2
    = [h_0, h_1] @ [W2_0]
                   [W2_1]
    = h_0 @ W2_0 + h_1 @ W2_1
    = y_0 + y_1
  ```

  - 任何其他切分方式都会强制在中间插入一次多余的集合通信：
    - 对于矩阵乘法 $Y = XW$：
    - 对 $W$ 采取列切分意味着每台设备计算出输出的一部分 Slice，产出自然切分的输出；
    - 对 $W$ 采取行切分意味着每台设备计算出全量输出的一个 Partial Sum 部分和，需要归约求和才能完成；
  - 对于 SwiGLU 结构的 MLP：对 $W_{\text{gate}}$ 和 $W_{\text{up}}$ 同时执行列切分，保持逐元素点积在 $F$ 维度切分，最后对 $W_{\text{down}}$ 执行行切分并在末尾执行一次 All-Reduce。

- **Attention 注意力层的张量并行设计** ：
  - 每台设备处理注意力头的子集（因为各个注意力头相互完全独立）；
  - 在末尾仅执行一次 All-Reduce（与 MLP 结构完全对齐）；
  - **注意力头相互完全独立，因此按头切分在整个注意力计算过程中不需要任何通信。**
  - 结构为：列并行（QKV 投影） $\to$ 本地注意力计算 $\to$ 行并行（输出投影）。

``` python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.out_features_per_rank = out_features // world_size
        self.rank = rank

        self.linear = nn.Linear(
            in_features,
            self.out_features_per_rank,
            bias=False,
            device=f"cuda:{rank}"
        )

    def forward(self, x):
        return self.linear(x)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.in_features_per_rank = in_features // world_size
        self.rank = rank

        self.linear = nn.Linear(
            self.in_features_per_rank,
            out_features,
            bias=False,
            device=f"cuda:{rank}"
        )

    def forward(self, x):
        # x 为局部切片: (batch, in_features_per_rank)
        # 每个 GPU 计算出局部部分结果
        partial = self.linear(x)

        # All-Reduce 在所有 GPU 间相加聚合局部结果
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)

        return partial

class TensorParallelMLP(nn.Module):
    """
    列并行 -> 激活 -> 行并行: 整层仅需 1 次 All-Reduce 通信
    """
    def __init__(self, d_model, d_ff, world_size, rank):
        super().__init__()
        assert d_model % world_size == d_ff % world_size == 0
        self.fc1 = ColumnParallelLinear(d_model, d_ff, world_size, rank)
        self.fc2 = RowParallelLinear(d_ff, d_model, world_size, rank)

    def forward(self, x):
        x = self.fc1(x)      # (batch, seq, d_ff // world_size)
        x = nn.functional.silu(x)
        x = self.fc2(x)      # (batch, seq, d_model), 已完成 all-reduced
        return x

class TensorParallelAttention(nn.Module):
    def __init__(self, d_model, num_heads, world_size, rank):
        super().__init__()
        assert num_heads % world_size == 0
        self.rank = rank

        # 每个 GPU 处理头的一个子集
        # QKV 投影仅针对本地负责的头
        self.qkv = ColumnParallelLinear(
            d_model,
            3 * d_model,
            world_size,
            rank
        )
        self.out_proj = RowParallelLinear(
            d_model,
            d_model,
            world_size,
            rank
        )

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # 投影到本地局部 Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads_per_rank, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # 在本地头子集上执行注意力运算
        q = q.transpose(1, 2)  # (batch, local_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, -1)

        # 输出投影并执行 All-Reduce
        return self.out_proj(attn_out)
```
