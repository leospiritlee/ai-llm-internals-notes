# 第7章 推理：一次生成在算什么

**一句话**：生成阶段每吐一个 token 都要回看全部历史，那些键值张量就是 `KV cache`，它 `Large` 且 `Dynamic`，旧系统因 `fragmentation and over-reservation` 浪费 `60% – 80%` 显存；`PagedAttention` 借 `virtual memory and paging in operating systems` 把它管起来，在 `without requiring any model architecture changes` 的前提下拿到 `24x higher throughput`。

**为什么需要它**：第 4 章留了一笔账：注意力把位置距离压成一步，代价是 `quadratic in sequence length`；第 6 章说明后训练改的是权重，不是计算形状。于是模型一旦部署，最贵的动作既不是训练也不是改提示词，而是 `autoregressive decoding process` 里反复发生的回看。`vLLM` 官方博客第一句判断就是 `the performance of LLM serving is bottlenecked by memory`，而不是算力。这句判断能解释很多线上现象：首字等半天后面却很顺、并发一上来就 OOM、加机器换不来线性吞吐。对读者来说，这一章是把「模型能力」与「服务质量」分开的第一处实证——同一份权重，换一套缓存管理，吞吐能差出一个数量级。综合报告口径把 `推理与效率层` 称为工程师最该投入的一层，也是这个理由。

**核心结构**：先看 `KV cache` 从哪来。博客原文：`In the autoregressive decoding process, all the input tokens to the LLM produce their attention key and value tensors, and these tensors are kept in GPU memory to generate next tokens`。也就是说，历史位置的键和值算过一次就存着，下一步不必重算。它由两个形容词定义：`Large`——`Takes up to 1.7GB for a single sequence in LLaMA-13B`；`Dynamic`——`Its size depends on the sequence length, which is highly variable and unpredictable`。两条合起来正好是传统连续内存管理最怕的组合：单个对象很大、长度又事先不知道，于是只能按最大长度预留，结果 `existing systems waste **60% – 80%** of memory due to fragmentation and over-reservation`。`PagedAttention` 的做法是 `partitions the KV cache of each sequence into blocks`，每块装 `the keys and values for a fixed number of tokens`，并允许 `storing continuous keys and values in non-contiguous memory space`；注意力核在计算时 `identifies and fetches these blocks efficiently`。这一层的产出不是新模型，而是显存利用率。

**关键设计**：**把缓存当虚拟内存管，是把「省下来的显存」直接换成并发，而不是只省了一笔资源**。博客给的类比非常完整：`one can think of blocks as pages, tokens as bytes, and sequences as processes`，`The contiguous *logical blocks* of a sequence are mapped to non-contiguous *physical blocks* via a block table`，并且 `The physical blocks are allocated on demand as new tokens are generated`。因为 `memory waste only happens in the last block of a sequence`，实际是 `near-optimal memory usage, with a mere waste of under 4%`。真正的产品收益写在下一句：`It allows the system to batch more sequences together, increase GPU utilization, and thereby significantly increase the throughput`。另一条同样重要的是共享：`multiple output sequences are generated from the same prompt`，此时 `the computation and memory for the prompt can be shared between the output sequences`；靠 `reference counts` 与 `Copy-on-Write` 保证安全，官方测到 `cutting their memory usage by up to 55%`（针对 `parallel sampling` 与 `beam search` 这类复杂采样）。把这两条与 `24x higher throughput` 放在一起，就能读出第 1 章那句判据的实证版本：能力没变，服务曲线变了。

**拆解**：

### 一次回答的两段体感不同

`vLLM` 博客把生成描述成逐 token 回看历史，工程上因此出现两段不同体感：开头要把整段输入一次性算完并建立 `KV cache`，之后每步只算新位置、再从缓存里取历史。综合报告口径把这两段对应的指标列成 `TTFT`（首 token 时间）、`ITL/TPOT`（token 间延迟）、吞吐（`tok/s`）、`p50/p95` 延迟、`每千 token 成本` 与 `GPU 显存占用`。这套指标是二手汇总给出的工程口径，但和第 4、7 章的因果完全对得上：首字慢是输入侧一次性成本高，后面顺是缓存命中，而长输入会同时抬高 `TTFT` 并撑大 `KV cache`。第 12 章会把它们变成准入表上的可勾选行。

### 为什么加机器不一定换吞吐

如果显存被 `over-reservation` 吃掉，增加请求只会更早撞上 OOM，而不是把 GPU 喂满。`PagedAttention` 的 `block table` 把 `logical blocks` 与 `physical blocks` 解耦，物理块 `allocated on demand`，于是同一张卡能 `batch more sequences together`。`vLLM` 博客同时给了两个入口：`vLLM can be used for both offline inference and online serving`，离线用 `pip install vllm` 后 `from vllm import LLM`，在线则是 `start an OpenAI API-compatible server`；逐字示例里有 `outputs = llm.generate(prompts)` 与 `lmsys/vicuna-7b-v1.3`。注意官方特意强调 `without requiring any model architecture changes`——服务引擎与模型架构是两个可独立更换的部件，这也是第 6 章末尾「现象 → 层」映射的一个硬证据。

### 缓存共享为什么值得单列

`parallel sampling` 与 `beam search` 的共同点是多条输出共享同一段提示词。传统做法要为每条序列各存一份提示词的键值，博客给的解法是 `different sequences in PagedAttention can share the blocks by mapping their logical blocks to the same physical block`，并用 `reference counts` 加 `Copy-on-Write` 防止写入互相污染——这与操作系统进程共享物理页是同一套思路。`PageAttention's memory sharing greatly reduces the memory overhead of complex sampling algorithms` 之后给的量化是 `up to 55%`。对做产品的人，这条比吞吐倍数更有指导性：如果你的场景要一次出多个候选再挑，缓存共享策略会直接改变能开几路并发。

### 这一章不解决什么

`PagedAttention` 管的是已有键值怎么放，不改变注意力的计算量，所以 `quadratic in sequence length` 依然在；第 8 章的 `FlashAttention` 处理的是 `IO-aware` 的搬运次数，`GQA` 处理的是键值头数量，二者与分页是互补而非替代。综合报告口径还列了 `KV 量化`、`跨层 KV 共享`、`token 淘汰`、`Speculative Decoding` 与 `continuous batching` 等一整排手段，以及 `GGUF`、`AWQ`、`GPTQ`、`FP8` 与 `4-bit` 这类量化口径；这些都是二手编排清单，本书只把有 A 级原句支撑的分页机制讲成结论。同理，`PagedAttention` 也不解决事实错误与无出处，那是第 9 章的事。

**今天怎么做**：把你线上最贵的一类请求按「输入长度 / 输出长度 / 并发数」三栏记一次，再看 `TTFT` 与 `p95` 分别跟着哪一栏走。**延伸，非语料原文**：如果两栏都跟着输入长度走，就先别调解码参数，去查缓存与并发；这个排查顺序是编排者给的练习，不是任何来源的表述。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `KV cache` 是可有可无的优化——博客说它 `Large` 且 `Dynamic`，`1.7GB` 只是一条序列；以为 `60% – 80%` 是模型本身吃掉的——原文归因是 `fragmentation and over-reservation`；以为 `PagedAttention` 改了模型——官方写 `without requiring any model architecture changes`；以为 `24x higher throughput` 是通用倍数——它是 `vLLM` 官方博客在自家对比下的 `up to` 口径；以为 `Copy-on-Write` 只是省内存技巧——它同时是 `parallel sampling` 安全共享的前提；还有把 `TTFT`、`ITL/TPOT`、`p50/p95` 当成一手官方定义，那是综合报告口径列出的工程指标清单。

**记住这一句**：生成是逐 token 回看历史，`KV cache` 又大又变长；`PagedAttention` 用 `block table` 把逻辑块映射到按需分配的物理块，把省下的显存换成 `batch more sequences together`，而权重一行没改。

</div>
