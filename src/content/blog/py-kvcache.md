---
title: "py-kvcache：基于 NVMe SSD 的外部 KV 缓存"
description: "论文 §5、§6.4–6.6、§7–8 选章中译，保留原始图表、实验条件与公式，梳理缓存加载的关键路径。"
date: "2026-10-06"
tags: ["翻译", "论文", "KV Cache", "NVMe", "vLLM"]
sourceURL: "https://arxiv.org/abs/2609.11744v1"
sourceAuthor: "Joseph Kanichai、Tiziano De Matteis、Animesh Trivedi"
translationScope: "论文选章译文整理：§5、§6.4–6.6、§7–8；中文初稿整理于 2026-09-24。"
---

## 外部缓存的收益取决于关键路径

将 KV Cache 放到 CPU 或 NVMe SSD，可以扩展缓存容量，但命中缓存并不自动意味着请求更快。读取、暂存、布局转换和传输如果进入关键路径，可能抵消省去 Prefill 的收益。

本文依据已有中文译稿，整理论文第 5 节、第 6.4–6.6 节及第 7–8 节。保留实验条件、图表与公式，按“现有方案的开销 → 实现策略 → 实验结果 → 适用范围”阅读。未收录的章节请参见原论文；以下实验中的“我们”指论文作者。

<span id="S5"></span>

## 5 现有 KV 缓存系统的特征分析

<span id="S5.p1"></span> <span id="S5.p1.1"></span>

为明确我们自己的外部 KV 缓存 py-kvcache（[§6](#S6)）的设计动机并指导其设计，我们对现有 KV 缓存实现进行特征分析，以找出延迟的来源以及设计中可以改进的部分。分析特别关注降低首 token 延迟（TTFT）的关键路径，以及拷贝与计算的重叠。在这些实验中，“文档长度”和“提示词长度”可以互换使用，均指一个请求中的 token 数量。

<span id="S5.p2"></span> <span id="S5.p2.1"></span>

除非另有说明，本节所有实验均在本地节点上使用 Llama 3.2 3B 进行。

<span id="S5.p3"></span> <span id="S5.p3.1"></span>

除非另有说明，Offload 指 vLLM 默认的、通过 KV Offload API 实现的 CPU 缓存。本节所有实验均关闭 GPU 前缀缓存，以单独考察 CPU 和磁盘的影响。实验期间，vLLM 正在积极修改这两种缓存接口，因此实验横跨多个 vLLM 版本。

<span id="S5.SS1"></span>

### 5.1 GPU ↔ CPU 传输路径

<span id="S5.SS1.p1"></span> <span id="S5.SS1.p1.1"></span>

第一个实验旨在了解：选择 KV Transfer API 还是 KV Offload API 是否会影响 TTFT，以及这种影响是否随文档长度而变化。我们将使用 KV Transfer API 的 LMCache 与使用 KV Offload API 的 vLLM 默认 CPU 缓存实现进行比较。在 vLLM [v0.16](https://github.com/vllm-project/vllm/releases/tag/v0.16.0) 上，我们使用长文档基准测试，覆盖一系列文档长度。每次运行先发送 $N$ 个长度为 $D$ 的不同提示词来填充缓存，再以随机顺序将每个提示词重复 $R$ 次，总计产生 $N+(N\times R)$ 个请求。第一阶段测量计算并存储 KV 数据的请求，第二阶段测量加载已有前缀的请求。所有请求都只生成一个输出 token，使测量结果主要由预填充或缓存加载决定，而不是由解码决定。

<span id="S5.SS1.p2"></span> <span id="S5.SS1.p2.1"></span>

[图 4](#S5.F4) 表明，与重新计算整个提示词相比，两种外部缓存都显著降低了缓存命中时的 TTFT。收益随文档长度增加而增大：从 1k token 时的 2.2 倍增至 80k token 时的 32.8 倍，因为被替代的重计算开销增长得比需要传输的 KV 数据量更快。两种接口之间的差距远小于它们各自与基线的差距。在 1k、40k 和 80k token 时，Offload 分别快 1.03、1.05 和 1.04 倍；在 10k token 时，LMCache 快 1.02 倍。这些差距的绝对值均不超过 33 ms。因此，在这一基准测试中，接口选择对缓存命中时 TTFT 的影响很小，因为两者都必须通过相同的 CPU → GPU 路径传输相同的 KV 数据，而这一传输主导了测量结果。这并不说明两种接口在一般情况下等价，只说明重复的长文档无法明显区分它们。因此，下一个实验转向更短、且大多为新请求的流量；在这种情况下，连接器开销在 TTFT 中所占的比例更大。

<span id="S5.F4"></span>

![图 4：长文档基准测试的 TTFT（vLLM v0.16，本地节点）。相对于重计算，两种外部缓存都将缓存命中时的 TTFT 降低了 2.2–32.8 倍。](/translations/py-kvcache/query_ttft_vs_doclen.png)

*图 4 ：长文档基准测试的 TTFT（vLLM v0.16，本地节点）。相对于重计算，两种外部缓存都将缓存命中时的 TTFT 降低了 2.2–32.8 倍。*

<span id="S5.SS1.p3"></span> <span id="S5.SS1.p3.1"></span>

第一个实验使用重复的长提示词，每次缓存命中都由大规模传输主导。因此，我们使用 ShareGPT，测试两种接口在更短、且大多为新请求的流量下是否也表现相当。我们使用相同的两种缓存实现，以每秒 32 个请求回放超过 10,000 个 ShareGPT 提示词。这些结果未绘图展示。LMCache 的 TTFT 为 206 ms，而 Offload 实现为 128 ms，相差 1.61 倍。由于 ShareGPT 的前缀复用很少，这个实验并不测量缓存命中的收益，而是展示连接器开销，尤其关注存储路径。即使大多数请求不加载可复用前缀，这种开销也会影响 TTFT；这说明两种接口确实存在差异，但差异出现在存储路径，而不是[图 4](#S5.F4) 所测量的缓存命中路径。

<span id="S5.SS1.SSSx1"></span>

#### 追踪 GPU ↔ CPU 传输路径

<span id="S5.SS1.SSSx1.p1"></span> <span id="S5.SS1.SSSx1.p1.1"></span>

为确定两个连接器的差异所在，我们对两条缓存路径插桩，并针对一个 40k token 的提示词，记录缓存加载、存储、GPU → CPU 与 GPU ← CPU 拷贝，以及模型前向计算。默认 Offload 实现使用 GPU 与 CPU 内存之间的异步 DMA 拷贝。在我们的追踪中，通过 `transfer_async` 将拷贝加入队列的开销可以忽略不计，而 LMCache 中对应的 `wait_for_save` 调用约耗时 400 μs。LMCache 还使用 CUDA 内核执行拷贝，这会消耗 GPU 计算资源。

<span id="S5.SS1.SSSx1.p2"></span> <span id="S5.SS1.SSSx1.p2.1"></span>

在整个基准测试中，Offload 的 GPU ↔ CPU 拷贝累计耗时 3.67 s，每个预热请求发起 21 次传输，每次平均耗时 9.08 ms。LMCache 在同一阶段累计耗时 4.13 s，但将每个请求拆成了 640 次传输，每次平均耗时 310 μs。Offload 的传输大小由 vLLM 的 `max_num_batched_tokens` 决定，而 LMCache 将同样的 KV 数据划分成更小的缓存块。Offload 较大的传输与后续前向计算交错执行，使相当一部分拷贝时间能够与模型执行重叠。因此，这个实验将传输粒度与执行重叠识别为 TTFT 差异的可能原因；但由于拷贝机制不同，仅凭累计耗时无法判断哪个系统的原始传输带宽更高。

<span id="S5.SS1.SSSx1.p3"></span> <span id="S5.SS1.SSSx1.p3.1"></span>

传输追踪显示，Offload 拷贝与后续前向计算存在重叠，但没有解释为什么会产生这种重叠。为确定原因，我们在改变 `max_num_batched_tokens` 的同时，追踪连续的 vLLM 引擎迭代。调度器以贪心方式填充每个执行批次，直到达到该上限，并在当前迭代之后立即提交下一次迭代。Offload 连接器将第 N 次迭代产生的 KV 数据加入队列，在第 N+1 次迭代期间执行相应的存储工作，使拷贝能够与后续前向计算重叠。LMCache 使用的 Transfer 路径没有以同样的方式推迟存储，而是在每次前向计算之后等待存储操作。这个实验确认，观察到的 TTFT 差异部分来自存储工作在引擎执行周期中的位置，而不只是拷贝本身的成本。

<span id="S5.SS1.SSSx1.p4"></span> <span id="S5.SS1.SSSx1.p4.1"></span>

为检验 TTFT 差异是否由 GPU ↔ CPU 的原始拷贝带宽造成，我们同时改变文档长度和 `max_num_batched_tokens`，并计算所测拷贝的有效带宽。这会改变 KV Offload API 发起的传输大小与次数，同时保持底层 GPU 和 CPU 不变。

<span id="S5.F5"></span>

![图 5：不同 max_num_batched_tokens 下的 GPU ↔ CPU 传输带宽。图例中的 1k–10k 表示 max_num_batched_tokens 的取值（vLLM v0.16，本地节点）。](/translations/py-kvcache/from_gpu_vs_doclen.png)

*图 5 ：不同 max_num_batched_tokens 下的 GPU ↔︎ CPU 传输带宽。图例中的 1k–10k 表示 max_num_batched_tokens 的取值（vLLM v0.16，本地节点）。*

<span id="S5.SS1.SSSx1.p5"></span> <span id="S5.SS1.SSSx1.p5.1"></span>

尽管拷贝机制和粒度不同，[图 5](#S5.F5) 显示两个系统都能维持约 24–26 GB/s 的 GPU ↔ CPU 带宽。一旦传输足够大，改变 `max_num_batched_tokens` 对大块数据传输带宽几乎没有影响。因此，我们认为 Offload 的 TTFT 优势主要来自发起更少的传输，以及让传输与模型执行异步并行，而不是以显著更高的原始带宽搬运数据。这促使我们在 py-kvcache 中保留 Offload 的异步执行模型，同时减少缓存路径其他环节的协调与 I/O 开销。

<span id="S5.SS2"></span>

### 5.2 NVMe SSD KV 缓存

<span id="S5.SS2.p1"></span> <span id="S5.SS2.p1.1"></span>

在考察 GPU ↔ CPU 传输路径之后，我们接着研究将 KV 数据存储到 NVMe SSD 时引入的额外成本。这些实验关注磁盘 KV 缓存在什么条件下有用，以及开销究竟来自存储设备、文件系统还是缓存连接器。

<span id="S5.F6"></span>

![图 6：TTFT 盈亏平衡边界（vLLM v0.22，Snellius 节点）。在各条曲线上方，加载缓存前缀比重计算更快。](/translations/py-kvcache/other-kv-pareto.png)

*图 6 ：TTFT 盈亏平衡边界（vLLM v0.22，Snellius 节点）。在各条曲线上方，加载缓存前缀比重计算更快。*

<span id="S5.SS2.p2"></span> <span id="S5.SS2.p2.1"></span>

第一个磁盘实验确定：何时加载缓存前缀比在 GPU 上重新计算它更快。这一点至关重要，因为 GPU 与磁盘性能差异很大，可能使结果偏向任何一方。我们使用 [§4.1](https://arxiv.org/html/2609.11744v1#S4.SS1) 所述的 Pareto 基准测试，在一系列文档大小下测量冷启动 TTFT（计算＋存储操作）与缓存命中 TTFT（加载操作）。然后在测量点之间插值，得到两种选择具有相同 TTFT 的边界。我们在边界紧邻的上方和下方运行额外配置，以验证这一边界。实验使用 llm-d 作为文件系统 KV 缓存；如 [§3.2](https://arxiv.org/html/2609.11744v1#S3.SS2) 所述，测试在我们支持直接 I/O 的分支上运行，以排除页缓存的影响。下面讨论的边界与 [§7](#S7) 使用同一个 [v0.22](https://github.com/vllm-project/vllm/releases/tag/v0.22.0) 分支测得。

<span id="S5.SS2.p3"></span> <span id="S5.SS2.p3.1"></span>

[图 6](#S5.F6) 给出的 Pareto 边界表明，缓存命中并不足以保证缓存有益。每个子图中的曲线标出了缓存命中 TTFT 与重计算相等时所需的最小可复用前缀比例；背景色则表示在各个文档大小和前缀比例下，预测该路径相对于重计算整个提示词的 TTFT 加速或减速。在我们的 Snellius 节点上，使用 Llama 3.2 3B 和 llm-d 时，8k token 的提示词需要 77.8% 的前缀复用才能达到盈亏平衡；而在 80k token 时，这个前缀比例降至提示词的 7.8%。边界会随 GPU、模型和 SSD 改变。更快的 GPU 能更快地重新计算前缀，而更大或更慢的模型则为加载缓存 KV 数据留出了更多时间。

<span id="S5.SS2.p4"></span> <span id="S5.SS2.p4.1"></span>

在[图 7](#S5.F7) 中，我们回答这样一个问题：存储平台（DRAM、SSD 等）需要多快，缓存才能胜过重计算？依据 [§4.1](https://arxiv.org/html/2609.11744v1#S4.SS1)，对于一篇包含 $D$ 个 token 的文档，我们从计算该文档所需的时间中减去缓存命中的非传输部分，剩下的就是搬运 KV 数据可用的时间预算：

<span id="S5.Ex3"></span>

|     |                         |     |
|-----|-------------------------|-----|
|     | $\mbox{max\_io}(D)=f(D)-\left(g(D)-t_{\mbox{\scriptsize copy}}(D)\right),$ |     |

<span id="S5.SS2.p4.2"></span>

其中，$g(D)$ 是缓存命中 TTFT，$t_{\mbox{\scriptsize copy}}(D)$ 是将 KV 数据搬入 GPU 所花费的时间；该值取自 [§4.1](https://arxiv.org/html/2609.11744v1#S4.SS1) 所述的性能分析追踪，是该路径上记录的传输事件时长的中位数。在 llm-d 中，这是从磁盘到 GPU 的时间，涵盖磁盘读取和 GPU 拷贝。用文档的 KV 字节数除以这一预算，就得到缓存路径必须维持的吞吐量；达到该吞吐量时，完整缓存的提示词才能与重计算耗时相等。Llama 3.2 3B 每个 token 存储 114,688 字节，因此随着提示词增长，所需吞吐量会急剧下降：从 1k token 时的 23.2 GB/s，降至 8k 时的 10.4 GB/s 和 80k 时的 3.5 GB/s。这是因为预填充成本的增长快于其产生的 KV 数据量。

<span id="S5.F7"></span>

![图 7：盈亏平衡所需的存储带宽（vLLM v0.22，Snellius 节点）。这是图 6 中 Pareto 边界的带宽表示。虚线表示我们的 SSD 实测的理想吞吐量。](/translations/py-kvcache/min_bandwidth.png)

*图 7 ：盈亏平衡所需的存储带宽（vLLM v0.22，Snellius 节点）。这是 图 6 中 Pareto 边界的带宽表示。虚线表示我们的 SSD 实测的理想吞吐量。*

<span id="S5.SS2.p5"></span> <span id="S5.SS2.p5.1"></span>

GPU ↔ CPU 拷贝能够维持 54–56 GB/s，因此在所有文档大小下都能满足要求。llm-d 在 1k token 时达到 9.0 GB/s，在更大的文档下达到 11.2–12.0 GB/s；因此，它在 1k 和 2k 时低于要求，从 8k 开始高于要求。参考线表示我们配置中各驱动器的理想吞吐量，以及达到盈亏平衡所需的吞吐量。实际含义是，存储带宽决定了最小文档大小。在大约 8k token 以下，预填充足够短，我们测量的任何存储设备都无法比 GPU 重建提示词更快地将其加载；而在中档驱动器上，这个阈值会移到 32k。因此，是否准入外部缓存必须取决于模型、GPU、SSD 和可复用前缀长度，不能将每次命中都视为有益。

<span id="S5.F8"></span>
<span id="S5.F8.sf1"></span>

![(a) I/O 引擎吞吐量。](/translations/py-kvcache/io_engine_request_size.png)

*(a) I/O 引擎吞吐量。*

<span id="S5.F8.sf2"></span>

![(b) 文件系统元数据吞吐量。](/translations/py-kvcache/fs-metadata.png)

*(b) 文件系统元数据吞吐量。*

<a href="#S5.F8">图 8</a>：存储微基准测试。


<span id="S5.SS2.p6"></span> <span id="S5.SS2.p6.1"></span>

Pareto 模型预测了在实测计算和传输成本下能够达到的最佳 TTFT。即使磁盘读取已接近 fio（v3.36）\[[3](https://arxiv.org/html/2609.11744v1#bib.bib15)\] 测得的原始吞吐量，最初的 llm-d 加载测量结果仍比这一预测更慢，波动也更大。为找出差距的来源，我们对 llm-d 文件系统连接器插桩，追踪其内存拷贝与块 I/O 请求。

<span id="S5.SS2.p7"></span> <span id="S5.SS2.p7.1"></span>

追踪显示，llm-d 针对每个缓存块分别拷贝并提交数据。更重要的是，它分配的最小暂存缓冲区比测试模型实际需要的 KV 数据更大。一个包含 4.4 GB 有效 KV 数据的请求，因此会导致约 10 GB 数据被拷贝并读取或写入。我们通过 llm-d issue [\#454](https://github.com/llm-d/llm-d-kv-cache/issues/454) 和 [\#389](https://github.com/llm-d/llm-d-kv-cache/issues/389)，向上游报告了这个分配问题以及一个独立的 vLLM 集成兼容性问题；分配缺陷在 [\#589](https://github.com/llm-d/llm-d-kv-cache/pull/589) 中得到修复。去除最小分配量后，在同一测试中，TTFT 降至比 LMCache 低约 50%。连接器读取可超过 12 GB/s，但写入仍在 2 GB/s 左右。因此，仅凭设备带宽无法完整预测缓存性能：暂存缓冲区策略和数据放大可能主导存储路径。这促使我们按实际需要分配暂存空间，并明确限制一个操作可以预留多少中间内存。

<span id="S5.SS2.p8"></span> <span id="S5.SS2.p8.1"></span>

在 SCBench 中观察到的另一个原生 vLLM 卸载问题，将结合[图 12](#S7.F12) 中的执行追踪讨论。

<span id="S5.SS2.SSSx1"></span>

#### I/O 引擎与请求大小

<span id="S5.SS2.SSSx1.p1"></span> <span id="S5.SS2.SSSx1.p1.1"></span>

接下来，我们测试更具扩展性的异步 I/O 引擎或更大的工作线程池，是否能改善本地节点上的 SSD KV 缓存。作为参考，fio 在 Kioxia 驱动器上测得 13.5 GB/s，在 Samsung PM9A3 上测得 6 GB/s；nvbandwidth 则在本地节点的 CPU 与 GPU 内存之间测得 25–26 GB/s，两者通过 PCIe 4.0 ×16 链路连接。我们的 Snellius 节点通过 PCIe 5.0 ×16 链路测得 54–56 GB/s。在 Snellius 上的独立存储基准测试中，io_uring 单线程达到每秒 338k 次操作，而测试的 POSIX 路径为每秒 13k 次；io_uring 在 16 个线程下达到每秒 259 万次操作。然而，替换 I/O 路径并未使 LLM 基准测试的 TTFT 得到可测量的改善。块追踪解释了这个结果：文件系统主要发出 512 KiB 请求，也有一些 1 MiB 请求。如[图 8(a)](#S5.F8.sf1) 所示，所测引擎在 4 KiB 请求下差异显著，但在单线程、1 MiB 请求下都趋近驱动器带宽上限。

<span id="S5.SS2.SSSx1.p2"></span> <span id="S5.SS2.SSSx1.p2.1"></span>

该图比较了 SPDK 用户态 NVMe 驱动、启用或关闭提交队列轮询的原生 io_uring、封装这两种后端的 xNVMe，以及 xNVMe 和 liburing 的 Python 绑定。单线程、4 KiB 请求时，各引擎表现接近；但到 16 个线程时，只有原生引擎吞吐量提高，Python 路径反而下降。我们将这一现象归因于 Python 运行时，而不是设备本身的特性。在 1 MiB 请求下，差异消失，所有引擎都集中在 fio 为这块驱动器测得的 6 GB/s 附近；唯一的例外是 Python xNVMe SPDK 绑定，它未能在这一请求大小下运行。

<span id="S5.SS2.SSSx1.p3"></span> <span id="S5.SS2.SSSx1.p3.1"></span>

因此，LLM 工作负载受限于带宽以及大规模传输的执行位置，而不是小 I/O 的操作速率。高小 I/O IOPS 和额外工作线程可以改善合成微基准测试，却未必改善端到端推理。这一发现促使我们使用少量异步工作线程和有界队列深度，而不是依靠大型线程池获得存储并行度。由于 KV 缓存文件远大于 1 MiB，Python 实现可以达到与原生实现相同的存储吞吐量（[§6](#S6)）。

<span id="S5.SS2.SSSx2"></span>

#### 文件系统元数据的扩展性

<span id="S5.SS2.SSSx2.p1"></span> <span id="S5.SS2.SSSx2.p1.1"></span>

由于 py-kvcache 计划使用普通文件系统，并由多个 vLLM 实例共享，我们测试缓存增长时，文件系统元数据是否会成为瓶颈。我们创建分别包含 10k、100k 和一百万个文件的工作目录，并分别测量 [§4.1](https://arxiv.org/html/2609.11744v1#S4.SS1) 定义的查找和发布事务。如[图 8(b)](#S5.F8.sf2) 汇总的结果所示，查找吞吐量基本保持不变，分别为每秒 121 万、119 万和 117 万次操作。发布操作分别达到每秒 122.7k、87.4k 和 104.5k 次操作。

<span id="S5.SS2.SSSx2.p2"></span> <span id="S5.SS2.SSSx2.p2.1"></span>

相比之下，插桩后的 vLLM 工作负载每秒仅发出约 170 次查找和 1,500 次发布。一个最多包含 64 个实例的独立连接器测试同样显示，在 CPU 容量成为限制因素之前，并未出现文件系统特有的性能崩塌。因此，我们认为分层文件系统布局能够支撑所评估服务工作负载要求的元数据操作速率。这并不意味着单次元数据操作没有成本，但说明在测试规模下，文件系统命名空间本身不太可能成为吞吐量瓶颈。

<span id="S5.SS2.SSSx3"></span>

#### 磁盘读取的执行位置

<span id="S5.SS2.SSSx3.p1"></span> <span id="S5.SS2.SSSx3.p1.1"></span>

最后，我们利用请求追踪，判断磁盘 I/O 是与有效工作重叠，还是仍停留在缓存命中的关键路径上。在现有磁盘路径中，请求必须先等待磁盘 → CPU 传输，随后才能执行余下的 CPU → GPU 传输。因此，即使 SSD 达到预期带宽，只要读取是在请求进入执行阶段之后才开始，其延迟仍会体现在 TTFT 中。这一观察促使我们研究：能否通过预加载，让排队请求的磁盘读取更早开始。

<span id="S5.SS2.SSSx4"></span>

#### 特征分析小结

<span id="S5.SS2.SSSx4.p1"></span> <span id="S5.SS2.SSSx4.p1.1"></span>

这些实验明确了 py-kvcache 设计需要满足的要求。缓存应避免加载或存储低于实测盈亏平衡点的前缀，并按有效 KV 数据量成比例地分配暂存内存。磁盘路径应面向大规模、受带宽限制的 I/O，并限制并发度，而不是试图通过大量线程和小请求来获得高 IOPS。普通的分层文件系统足以应对观察到的元数据负载；如果可能，磁盘读取应在请求执行前开始，避免存储延迟完全留在 TTFT 的关键路径上。下一节介绍 py-kvcache 如何实现这些要求。

<span id="S6"></span>

## 6 py-kvcache 的设计（节选）

<span id="S6.SS4"></span>

### 6.4 异步传输与有界暂存

<span id="S6.SS4.p1"></span> <span id="S6.SS4.p1.1"></span>

py-kvcache 以块为单位存储 KV 数据，每个块组合若干个 vLLM GPU 块，并对应一个文件。在我们实验采用的 256-token 块大小下，一个 Llama 3.2 3B 缓存块约占 28 MiB。现有文件系统后端使用阻塞工作线程池获得 I/O 并发，但在这样的对象大小下，这种方式没有收益。py-kvcache 改为使用一个 reactor 线程，提交多个未完成操作并收取其完成结果。存储并发度由配置的 I/O 深度控制，而不是由工作线程数量控制。

<span id="S6.SS4.p2"></span> <span id="S6.SS4.p2.1"></span>

可配置的软件 I/O 深度使我们能够为内存分配设定上限。单次加载或存储覆盖一个请求的整个前缀，通常跨越多个存储块。因此，py-kvcache 将其拆成每个存储块一个任务；每个任务占用固定 CPU 暂存池中的一个槽位，在 I/O 和 GPU 传输都完成后归还槽位。这样可以尽量减轻 CPU 内存压力，并避免在采用不同调度方式（例如延迟请求）时可能出现的死锁或类似问题。大于暂存池的请求会被增量处理，而不是为整个 KV 对象预留内存。加载、存储和预加载共用同一个池，避免各条路径独立预留各自需要的容量。暂存池由一次连续、对齐且页锁定的 CPU 内存分配构成。CPU 池还可以选择通过 LRU 或 ARC 替换策略保留已完成的槽位，使其在充当传输缓冲区的同时成为 DRAM 缓存。磁盘 I/O 与到 GPU 的 DMA 拷贝使用相同的 CPU 槽位，因而避免了 llm-d 中观察到的写放大等问题（[§5.2](#S5.SS2)）。

<span id="S6.SS4.p3"></span> <span id="S6.SS4.p3.1"></span>

传输按存储块（即每个文件）进行流水线处理，而不是作为单个操作执行。加载时，reactor 首先通过 io_uring 发起异步 `openat` 操作，数量不超过独立设置的前瞻深度。只有在 I/O 深度尚有余量、可以提交读取时，才获取槽位。磁盘读取完成后，其块映射立即加入 CPU → GPU 传输队列。在一次 reactor 迭代中就绪的所有加载映射，包括来自不同请求的映射，都会合并为一次 `swap_blocks_batch` 调用，在专用 CUDA 流上启动。DMA 进行期间，reactor 可以收取其他磁盘操作的完成结果，并提交更多读取。只有当对应 CUDA 事件报告完成后，暂存槽位才会被释放。[图 3](https://arxiv.org/html/2609.11744v1#S4.F3) 的追踪可以观察到这一流水线。在配置的 I/O 深度之外，池还预留额外的拷贝余量，避免 CUDA 传输占用槽位而使磁盘流水线断流。

<span id="S6.SS4.p4"></span> <span id="S6.SS4.p4.1"></span>

存储使用反向流水线。每个暂存槽位都有自己的 CUDA 流，该流等待该槽位的 CUDA 事件，并从 GPU 内存拷贝一个存储块，无需进行设备级全局同步。一旦某个槽位的 GPU → CPU 拷贝完成，reactor 就提交它的 I/O 写入。从拷贝启动直到收取写入完成结果，一次存储始终占用一个 I/O 深度预算名额。随后发布已完成的文件，并释放该槽位，或将其保留在可选的 DRAM 缓存中。结合 Offload 的延迟存储语义，这使某次引擎迭代的 GPU 拷贝和磁盘写入能够与后续迭代的工作重叠。不过，由于我们的 NVMe 驱动器性能很高，而且这种重叠也受到 `max_num_batched_tokens` 的隐式限制，我们在实验中从未观察到这种情况。

<span id="S6.SS5"></span>

### 6.5 预加载

<span id="S6.SS5.p1"></span> <span id="S6.SS5.p1.1"></span>

尽管异步 I/O 减少了传输引擎内部的阻塞，普通的按需读取仍要等到请求被选中执行后才开始。因此，py-kvcache 利用调度器信息，对排队请求进行预加载。管理器对等待请求进行有界前瞻，并在存储路径有可用容量时向工作进程发送预加载计划。在这些请求被调度之前，reactor 就开始将匹配前缀从磁盘搬入现有的 CPU 暂存池。如果某个请求随后开始执行，它会接续尚未完成的读取，或认领已暂存的块，使请求路径上只剩 CPU → GPU 的数据提升。这可以显著降低 TTFT。

<span id="S6.SS5.p2"></span> <span id="S6.SS5.p2.1"></span>

这需要对我们的 [vLLM 分支](https://github.com/t348575/vllm/tree/d6eadf416bb5234047760bf55d532f2f038cf697)做少量修改。原始 KV Offload API 只在调度器选中请求后才要求工作进程加载 KV 数据，而工作进程本身无法看到等待队列。我们为调度器增加了一个有界的 `on_preload_candidates` 回调，并将由此得到的预加载标识符和块元数据传递给工作进程；工作进程通过新增的连接器钩子启动读取。这些调度器信息保留在 vLLM 内部，由 KV Transfer API 与 KV Offload API 之间的转换层处理。之后的按需加载会识别并认领同一个暂存操作。没有这条调度器—工作进程通路，py-kvcache 就无法在请求获准执行前开始磁盘 I/O，完整的读取过程仍会停留在 TTFT 的关键路径上。

<span id="S6.F10"></span>
<span id="S6.F10.sf1"></span>

![(a) 仅磁盘时的查询 TTFT。](/translations/py-kvcache/surf_pure_disk_query_ttft_conc50.png)

*(a) 仅磁盘时的查询 TTFT。*

<span id="S6.F10.sf2"></span>

![(b) 磁盘＋DRAM 时的查询 TTFT。](/translations/py-kvcache/surf_disk_dram_query_ttft_conc50.png)

*(b) 磁盘＋DRAM 时的查询 TTFT。*

<a href="#S6.F10">图 10</a>：独立长文档评估（vLLM v0.22，Llama 3.2 3B，Snellius 节点）。


<span id="S6.SS5.p3"></span> <span id="S6.SS5.p3.1"></span>

预加载工作被有意置于按需流量之后。reactor 首先调度已就绪的按需读取操作，其次调度新的按需加载与存储操作，最后才调度推测性的预加载。它为前台工作预留与所配置 I/O 深度相同数量的暂存槽位；前台加载可以回收已保留的缓存槽位，也可以回收尚未被认领的预加载槽位。如果某个推测性文件已经打开或正在处理，此时出现了按需加载操作，reactor 会关闭该文件描述符并将预加载重新入队，避免它继续消耗读取带宽。多个预加载候选请求若请求相同前缀，会共享一次磁盘读取和一个引用计数暂存槽位；后续按需加载会加入尚未完成的操作，而不会重复发起读取。之所以引入这些规则，是因为我们观察到 vLLM 原生次级缓存层（[§3.3](https://arxiv.org/html/2609.11744v1#S3.SS3)）中的并发推测性提升会耗尽 CPU 内存，驱逐刚提升的块，并迫使被调度的请求重新计算这些块；[图 12](#S7.F12) 展示了对此进行的追踪。

<span id="S6.SS6"></span>

### 6.6 盈亏平衡

<span id="S6.SS6.p1"></span> <span id="S6.SS6.p1.1"></span>

最后，py-kvcache 不会把每个匹配的前缀都视为值得加载。[§4.1](https://arxiv.org/html/2609.11744v1#S4.SS1) 描述的 Pareto 边界求取流程作为离线校准，针对每种节点与模型组合运行一次，生成一个小文件，保存各来源层的盈亏平衡前缀长度。服务过程中不进行任何测量或拟合。查找时，管理器将可复用前缀与其来源层的阈值比较。如果预测加载成本高于重计算，管理器就拒绝加载，让 vLLM 重新计算前缀。因此，这道门控是在防止损失，而不是创造收益。在盈亏平衡点以下，提示词足够短，其 TTFT 的绝对值本就很小，所以拒绝加载可以避免无效的传输工作，但不会实质性改善延迟。它的价值在于避免外部缓存在 Bailian 轨迹这类工作负载上造成性能退化。

<span id="S7"></span>

## 7 py-kvcache 评估

<span id="S7.p1"></span> <span id="S7.p1.1"></span>

我们评估 py-kvcache 是否改善端到端服务性能、其设计中的哪些部分带来了观察到的改进，以及在真实的长上下文工作负载下，其行为是否仍然有效。我们首先使用受控的长文档工作负载，单独考察磁盘和预加载行为，然后使用 LongBench、SCBench 和 Bailian 轨迹评估完整系统。本节实验在本地节点和 Snellius 节点上进行，使用 Llama 3.2 3B 或 Qwen3 4B。本节所有测量均使用 vLLM [v0.22](https://github.com/vllm-project/vllm/releases/tag/v0.22.0)。这里不评估 llm-d，因为在本项工作进行期间，它被弃用，转而采用功能相同的 vLLM 原生文件系统层（[§3.2](https://arxiv.org/html/2609.11744v1#S3.SS2)）。

<span id="S7.SS1"></span>

### 7.1 独立前缀缓存测试

<span id="S7.SS1.SSSx1"></span>

#### 仅磁盘配置的比较

<span id="S7.SS1.SSSx1.p1"></span> <span id="S7.SS1.SSSx1.p1.1"></span>

我们首先考察：当所有可复用 KV 数据都必须从磁盘获取时，py-kvcache 是否能改善缓存命中延迟。在 Snellius 节点上，我们使用 50 个并发请求，将 LMCache 的磁盘后端与开启和关闭预加载的 py-kvcache 进行比较。GPU 前缀缓存和盈亏平衡门控均关闭，以使每个匹配的前缀都走磁盘路径。

<span id="S7.SS1.SSSx1.p2"></span> <span id="S7.SS1.SSSx1.p2.1"></span>

在测试的各个文档大小下，[图 10(a)](#S6.F10.sf1) 的查询 TTFT 结果显示，未开启预加载的 py-kvcache 始终比 LMCache 快约 1.5 倍，单独体现了异步传输引擎的收益。开启预加载后，相对 LMCache 的总加速比达到 2.0–2.5 倍；[§7.2](#S7.SS2) 将分离这两部分的贡献。

<span id="S7.F11"></span>

![图 11：Snellius 节点上的 LongBench、SCBench 工作负载，以及本地节点上的 Bailian 轨迹的查询 TTFT（vLLM v0.22，Qwen3 4B，Snellius 节点）。py-kvcache、LMCache 和原生 vLLM KV Offload 均开启 GPU 前缀缓存。](/translations/py-kvcache/trace_workloads_query_ttft.png)

*图 11 ：Snellius 节点上的 LongBench、SCBench 工作负载，以及本地节点上的 Bailian 轨迹的查询 TTFT（vLLM v0.22，Qwen3 4B，Snellius 节点）。py-kvcache、LMCache 和原生 vLLM KV Offload 均开启 GPU 前缀缓存。*

<span id="S7.SS1.SSSx2"></span>

#### 分层 KV 缓存

<span id="S7.SS1.SSSx2.p1"></span> <span id="S7.SS1.SSSx2.p1.1"></span>

接下来，我们比较同时具有 CPU DRAM 和磁盘的完整“类生产”配置。如[图 10(b)](#S6.F10.sf2) 所示，在 10k、40k 和 80k token 时，py-kvcache 分别比 LMCache 快 1.19、1.53 和 1.23 倍。在 40k 和 80k token 时，它与原生 vLLM 卸载实现的性能差距分别在 1.10 倍和 1.04 倍以内；在 10k 时，则比原生实现快 1.12 倍。相对于没有 CPU 层的相同 py-kvcache 配置，加入 DRAM 使 TTFT 在 10k、40k 和 80k token 时分别改善 1.30、1.18 和 1.08 倍。在 1k token 时，四种配置的结果彼此相差不超过 0.12 s，性能排序发生反转；这符合 [§5.2](#S5.SS2) 确定的盈亏平衡点以下的预期。因此，在外部缓存值得使用的文档大小下，py-kvcache 接近 vLLM 集成式原生实现的性能。

<span id="S7.F12"></span>

![图 12：SCBench 的 CPU 池占用与存储吞吐量（vLLM v0.22，Qwen3 4B，Snellius 节点）。无界的数据提升使原生 CPU 池持续满载，并从磁盘读取 3.4 TB；相比之下，py-kvcache 的有界暂存只读取 85 GB，并在 480 s 时完成所有请求的服务。](/translations/py-kvcache/pool_compare.png)

*图 12 ：SCBench 的 CPU 池占用与存储吞吐量（vLLM v0.22，Qwen3 4B，Snellius 节点）。无界的数据提升使原生 CPU 池持续满载，并从磁盘读取 3.4 TB；相比之下，py-kvcache 的有界暂存只读取 85 GB，并在 480 s 时完成所有请求的服务。*

<span id="S7.SS2"></span>

### 7.2 预加载的影响

<span id="S7.SS2.p1"></span> <span id="S7.SS2.p1.1"></span>

前面的结果比较的是完整系统，因此接下来我们使用相同的仅磁盘 py-kvcache 配置，分别开启和关闭预加载，以单独考察预加载机制。[图 10(a)](#S6.F10.sf1) 的查询 TTFT 结果显示，预加载在 40k token 时带来 1.66 倍改善，在 80k token 时带来 1.34 倍改善。80k token 下收益较小，是因为 CPU → GPU 传输显著大于 40k 时；这段时间主导了总 TTFT，实际上“掩盖”了预加载的收益。

<span id="S7.SS2.p2"></span> <span id="S7.SS2.p2.1"></span>

此外，我们运行了一个混合工作负载，包含 50 个长度为 80k token 的请求，其中 50% 复用缓存前缀，最大并发数为 8。在这次完整基准测试中，预加载将查询轮次的总墙钟时间从 73 s 降至 68 s，减少 6.8%。在同一次运行中，所有请求的平均单请求 TTFT 降低 0.4 s，而只看前缀复用请求时降低 1.3 s。将输出长度从 128 token 调整到 1k token，对预加载收益几乎没有影响。这符合预期，因为该机制影响的是预填充，而不是随后的解码阶段。

<span id="S7.SS2.p3"></span> <span id="S7.SS2.p3.1"></span>

追踪解释了改进来自哪里。没有预加载时，第一次文件读取在缓存查找开始后约 6 ms 才启动；之后，请求仍需等待磁盘 → CPU 传输，再等待 CPU → GPU 传输。预加载将相同的 I/O 启动成本提前到请求还在调度器中等待时支付。当请求进入执行阶段，磁盘阶段已经部分或全部完成。

<span id="S7.SS3"></span>

### 7.3 代表性的长上下文工作负载

<span id="S7.SS3.SSSx1"></span>

#### LongBench

<span id="S7.SS3.SSSx1.p1"></span> <span id="S7.SS3.SSSx1.p1.1"></span>

受控工作负载总是构造可复用前缀，因此接下来我们测试，在不那么规则的请求序列中，是否仍会出现同样的优势。我们在 Snellius 节点上回放 LongBench 的“多文档问答”和“代码仓库理解”两个领域。如[图 11](#S7.F11) 所示，py-kvcache 在两个领域都取得最低的平均查询 TTFT：相对于重计算快 6.02–7.43 倍，相对于 GPU 前缀缓存快 2.02–2.12 倍，相对于 LMCache 快 2.77–3.64 倍，相对于原生 vLLM KV Offload 实现快 1.22–1.79 倍。与受控基准测试不同，这些工作负载既包含不同长度的前缀链，又具有高并发。这表明，在可复用前缀分布不均匀、更贴近实际的工作负载中，预加载以及 py-kvcache 整体都非常有效。py-kvcache 带来的性能提升主要源自传输所处的执行位置。在这样的并发度下，请求会依次排队；py-kvcache 根据调度器等待列表进行预加载，而不是在缓存查找时才启动（[§6.5](#S6.SS5)）。因此，可复用前缀可以在对应请求仍处于等待状态时读入，关键路径上只剩 CPU → GPU 的提升。

<span id="S7.SS3.SSSx2"></span>

#### SCBench

<span id="S7.SS3.SSSx2.p1"></span> <span id="S7.SS3.SSSx2.p1.1"></span>

我们使用 SCBench KV 工作负载，以轮转顺序测试一个更大的多轮工作集。它与我们的 LongBench 工作负载类似，但代表 KV 缓存的最坏情况。如[图 11](#S7.F11) 所示，在 Snellius 节点上，py-kvcache 相对 GPU 前缀缓存取得 3.11 倍加速，相对原生 CPU＋磁盘卸载取得 2.48 倍加速，相对 LMCache 取得 1.26 倍加速。

<span id="S7.SS3.SSSx2.p2"></span> <span id="S7.SS3.SSSx2.p2.1"></span>

原生卸载结果意外地差，促使我们追踪数据提升、存储、按需加载、内存预留和驱逐。[图 12](#S7.F12) 上半部分是原生 vLLM 文件系统卸载，下半部分是 py-kvcache。阴影区域表示 CPU 池占用，曲线表示磁盘读写吞吐量。只要调度器查找在磁盘上找到 KV 数据，原生 vLLM 就开始一次提升。多次查找可能接连发生，导致多个提升操作同时预留大部分 CPU 缓存。随后，存储操作可能无法取得暂存内存，已完成的提升也可能立即被驱逐以腾出空间。当请求最终被调度时，其按需加载可能再次发现没有空闲 CPU 内存，只能回退到前缀重计算。原生实现的追踪反复锁住 CPU 池的大部分空间，运行持续超过 1,200 s。在这次追踪中，最终只有两个请求将提升后的数据从 CPU 传到了 GPU；磁盘总读取量达到 3.4 TB，而 py-kvcache 只读取 85 GB，两个系统各自的缓存总大小均为 465 GB。我们通过 [vLLM issue \#49902](https://github.com/vllm-project/vllm/issues/49902) 向上游报告了这一行为。该问题已被确认，维护者希望引入用于检测背压的策略或机制，相关讨论见 [\#50031](https://github.com/vllm-project/vllm/issues/50031) 和 [\#50014](https://github.com/vllm-project/vllm/pull/50014)。

<span id="S7.SS3.SSSx2.p3"></span> <span id="S7.SS3.SSSx2.p3.1"></span>

相比之下，py-kvcache 同一时刻只允许一个推测性预加载；当按需加载到达时，它会停止发出预加载工作。它还通过配置的 I/O 深度限制每次加载或存储所预留的内存。在这一配置下，即使完整上下文大得多，py-kvcache 也只需要 576 MB 空闲 CPU 内存。[图 12](#S7.F12) 下半部分显示，有界暂存避免了反复出现的占用尖峰，并使运行在约 480 s 后完成。这个实验证明，仅仅提前读取并不够；应当按推测性工作的本来性质对待它：它是推测性的，因此还必须限制规模，并让位于按需操作。

<span id="S7.SS4"></span>

### 7.4 Bailian 生产轨迹

<span id="S7.SS4.p1"></span> <span id="S7.SS4.p1.1"></span>

最后，我们回放阿里云百炼（Bailian）的 Coder、交互式（A）和 API 驱动（B）轨迹。如[图 11](#S7.F11) 所示，在本地节点（RTX 4000 Ada）上，py-kvcache 比 GPU 前缀缓存快 1.12–1.79 倍，比 LMCache 快 1.10–1.25 倍。它与原生 vLLM KV Offload 实现仍然接近；在这三种轨迹中，原生实现仅快 1.02–1.06 倍。这主要是因为相对于 LongBench 和 SCBench，该工作负载的提示词较短，前缀复用也较少。

<span id="S7.SS4.p2"></span> <span id="S7.SS4.p2.1"></span>

相同轨迹在 Snellius 节点上呈现不同结果。在我们的配置下，这些轨迹的平均请求长度低于为 Qwen3 4B 测得的 SSD 盈亏平衡点——6,203 token；而且 H100 更大的 GPU 内存保留了很大一部分工作集。因此，只使用 GPU 前缀缓存时的 TTFT 与 py-kvcache 相同，而原生 vLLM KV Offload 实现高出 0.2 s。外部缓存是否有用，既取决于工作负载的可复用前缀分布，也取决于 GPU 的计算与内存容量。这促使 py-kvcache 采用针对模型和硬件的盈亏平衡门控，而不是加载每个匹配前缀。

<span id="S7.SS5"></span>

### 7.5 评估小结

<span id="S7.SS5.p1"></span> <span id="S7.SS5.p1.1"></span>

评估表明，py-kvcache 同时改善了传输路径和磁盘读取的执行位置。在受控的仅磁盘基准测试中，80k token 时，其 TTFT 相对于 LMCache 减半。相对于没有前瞻的同一引擎，预加载进一步带来 1.34–1.66 倍改善。LongBench 和 SCBench 表明，这些收益可以延伸到链式和多轮上下文；SCBench 追踪则说明，在内存压力下，为什么必须限制暂存规模并优先处理按需工作。Bailian 轨迹展示了这些收益的边界：外部 KV 缓存在较小 GPU 上有帮助，但在 H100 上，许多前缀低于盈亏平衡点，应留在 GPU 内存中或重新计算。

<span id="S8"></span>

## 8 讨论

<span id="S8.p1"></span> <span id="S8.p1.1"></span>

结果表明，应将外部 KV 缓存理解为一种关键路径优化，而不只是内存层次结构的扩展。峰值存储带宽决定了 KV 数据能以多快的速度搬运，但不决定搬运何时开始、预留多少中间内存，或加载是否比重计算更快。在评估的各个系统中，这些调度和资源管理决策往往与底层存储介质同样重要。这一区别解释了为什么较慢的数据来源有时反而能产生更低的 TTFT。

<span id="S8.SS1"></span>

### 8.1 关键路径的重要性

<span id="S8.SS1.p1"></span> <span id="S8.SS1.p1.1"></span>

存储微基准测试进一步支持这一解释。如[图 8(a)](#S5.F8.sf1) 所示，测试的 I/O 引擎在 4 KiB 请求下差异显著，尤其是在增加线程时；但在单线程、1 MiB 请求下都趋近设备带宽上限。KV 缓存工作负载主要生成大请求，因此提高小 I/O 操作速率或替换 I/O 引擎本身，并未改善端到端 TTFT。对这一工作负载来说，异步 I/O 的有效作用并不是最大化 IOPS，而是维持足够多尚未完成的大操作，以利用 SSD 带宽。

<span id="S8.SS1.p2"></span> <span id="S8.SS1.p2.1"></span>

GPU 传输结果带来了类似结论。LMCache 与原生 KV Offload 实现在大块 GPU ↔ CPU 传输上达到相近带宽，但 Offload 路径发出的传输更少、更大；关键在于，它将存储工作隐藏在模型前向计算期间。因此，其优势来自传输粒度与重叠，而不是显著更快的物理拷贝路径。py-kvcache 保留了这一 Offload 执行模型，并将相同原则应用于存储读取。即使传输字节数不变，将读取提前也可以减少可见延迟。在受控实验中，相对于不开启预加载的同一 py-kvcache 引擎，预加载使 TTFT 改善 1.34–1.66 倍。这说明，优化工作在何时执行，可能比优化该工作单独执行时的速度更有价值。

<span id="S8.SS1.p3"></span> <span id="S8.SS1.p3.1"></span>

预加载将按需工作转化为推测性工作。只有当所选请求很可能被执行，且其排队时间足够长、能够完成有用的 I/O 时，预加载才有益。无限制的预加载器可能会为被延迟或从未调度的请求消耗存储带宽、暂存内存和 CPU ↔ GPU 传输能力。SCBench 实验展示了这种风险：多个原生提升操作同时预留了 CPU 层的大部分空间，挤出了刚提升的块，使后续按需操作没有足够内存。py-kvcache 只允许一个预加载，与按需传输共享缓冲区，并在按需加载到达时让路；这一策略是有意采取的保守设计。结果表明，成功的预加载需要准入控制和资源上限，而不只是更早提交。原生 vLLM Offload 根据 KV 块查找进行提升；与之不同，py-kvcache 可以访问调度器列表，因此能更明智地选择要预加载的请求。

<span id="S8.SS2"></span>

### 8.2 与现有系统比较

<span id="S8.SS2.p1"></span> <span id="S8.SS2.p1.1"></span>

LMCache 支持多种服务引擎、存储后端和分布式部署，而 py-kvcache 专门面向 vLLM 的 KV Offload 与共享文件系统存储 \[[7](https://arxiv.org/html/2609.11744v1#bib.bib4), [19](https://arxiv.org/html/2609.11744v1#bib.bib8)\]。在评估配置中，py-kvcache 得益于 Offload 粒度更大且推迟执行的传输；但这并不意味着 Offload 普遍更优，也不意味着它能替代 LMCache 更广泛的功能，以及对其他存储介质和分布式部署的广泛支持。原生 vLLM Offload 实现是最接近的比较对象，因为它使用相同 API，并集成 CPU 层和文件系统层 \[[35](https://arxiv.org/html/2609.11744v1#bib.bib7)\]。在我们的分层缓存测试中，py-kvcache 与 vLLM 原生 Offload 实现性能相近，在 LongBench 和 SCBench 上表现更好。SCBench 的结果反映出 py-kvcache 更保守的预加载策略，以及在苛刻的最坏条件下更稳健的设计。

<span id="S8.SS2.p2"></span> <span id="S8.SS2.p2.1"></span>

py-kvcache 和 llm-d 都使用分层、按哈希寻址的文件，以实现共享与原子发布 \[[18](https://arxiv.org/html/2609.11744v1#bib.bib9)\]。py-kvcache 使用受 I/O 深度限制的传输任务和共享暂存池，而 llm-d 使用各自独立的缓冲区。

<span id="S8.SS2.p3"></span> <span id="S8.SS2.p3.1"></span>

当工作集能放入显存时，GPU 前缀缓存仍是更好的选择，因为 vLLM 只需交换指针，无需任何传输或拷贝。

<span id="S8.SS3"></span>

### 8.3 何时应该使用外部缓存

<span id="S8.SS3.p1"></span> <span id="S8.SS3.p1.1"></span>

[图 6](#S5.F6) 中的 Pareto 边界说明，为什么仅凭缓存命中不足以作为策略信号。当请求具有较长的可复用前缀、可复用工作集超过 GPU 容量，而且请求等待足够久、能让传输与其他工作重叠时，外部复用最有吸引力。对于短前缀、低复用、更快的 GPU，或排队时间很少的轻负载系统，它的吸引力会降低。合适的阈值还会随模型架构、KV 数据类型、GPU 计算速率、PCIe 带宽、存储带宽和来源层变化。因此，本工作测得的数值阈值只属于特定配置。

<span id="S8.SS3.p2"></span> <span id="S8.SS3.p2.1"></span>

Bailian 轨迹展示了这一适用范围的两面。在本地节点上，外部缓存改善了 TTFT，因为较小的 RTX 4000 Ada 能保留的工作集较少，而且重计算相对昂贵。在 H100 系统上，更大的显存容量保留了更多前缀，平均请求又低于为 Qwen3 4B 测得的 SSD 盈亏平衡点。因此，即便存在可复用数据，外部传输的收益也很小。实用的多层策略应只在预测 CPU 或 SSD 的传输成本低于重计算时使用它们，并保留重计算作为有效的回退选择，而不是将其视为缓存失败或未命中。

<span id="S8.SS3.p3"></span> <span id="S8.SS3.p3.1"></span>

考虑盈亏平衡的决策还应包含工作预期所处的执行位置。仅基于前缀长度和设备带宽的静态阈值，无法区分按需读取与可能被排队时间隐藏的预加载。反过来，某个预加载单从传输时间看似乎有利，但当请求不太可能很快运行时，它也可能造成浪费。更完善的策略会综合前缀大小、来源层、当前暂存容量、各介质的实测传输速率，以及预期排队时间。本文评估的有界策略，是迈向更复杂调度器和缓存控制器的第一步。
