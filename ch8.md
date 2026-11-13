# 第8章 长上下文：窗口越大越好的错觉

**一句话**：`FlashAttention` 说 `Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length`，`GQA` 与 `Mixture of Experts (MoE)` 分别从键值头与激活参数上减负，而综合报告口径还提醒 `lost in the middle`——窗口变长不等于信息被用上。

**为什么需要它**：第 7 章把显存账算清了，但还有两笔账没算：一次注意力的读写次数，以及窗口里的信息到底有没有被注意到。两者常常被混成一句「上下文越长越贵」，结果是选型时只盯窗口数字。`FlashAttention` 的一手摘要给了一个更准的切入点：`Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup`——降 FLOPs 不等于降墙上时间。作者补的 `a missing principle is making attention algorithms IO-aware`，把瓶颈从算术搬到了 `reads and writes between levels of GPU memory`。这一章要建立的判断是：长上下文是三条独立约束的叠加，不是单一参数；窗口数字只是其中最表面的一条。综合报告口径把 `128K` 称为基线、`1M` 称为新的竞争焦点，并指出实际耗时收益远小于 FLOPs 降幅——这是二手趋势判断，本书不把它当官方结论。

**核心结构**：三条约束各管一段。第一条是计算与内存随长度平方增长，`quadratic in sequence length` 来自 `FlashAttention` 摘要；`Approximate attention methods` 试图用质量换复杂度，`often do not achieve wall-clock speedup`。第二条是显存搬运与键值头数量：`FlashAttention` 是 `an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM`，并 `requires fewer HBM accesses than standard attention`、`is optimal for a range of SRAM sizes`；`GQA` 那边，`Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference`，但 `MQA can lead to quality degradation`，于是 `grouped-query attention (GQA)` 取 `an intermediate (more than one, less than number of query heads) number of key-value heads`。第三条是信息利用：综合报告口径写 `注意力对序列长度是二次复杂度`，`100K token 的成本接近 10K 的百倍`，而且存在 `lost in the middle` 现象，即开头结尾的信息被关注得更多，因此 `长上下文 ≠ 更好的效果`。

**关键设计**：**`exact attention` 四个字，是 `FlashAttention` 与近似方法之间最重要的分界**。摘要没有去逼近注意力，而是 `making attention algorithms IO-aware -- accounting for reads and writes between levels of GPU memory`，用 `tiling` 减少 `GPU high bandwidth memory (HBM)` 与 `GPU on-chip SRAM` 之间的读写；结果是精确结果、更少的 `HBM accesses`，以及一串具体收益：`15% end-to-end wall-clock speedup on BERT-large`、`3× speedup on GPT-2`、`2.4× speedup on long-range arena`。更关键的是它打开了原本做不到的长度：`FlashAttention and block-sparse FlashAttention enable longer context in Transformers`，并给出 `the first Transformers to achieve better-than-chance performance on the Path-X challenge`（`16K, 61.4% accuracy`）与 `Path-256`（`64K, 63.1% accuracy`）。`GQA` 的设计同样值得记：`propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute`，结论是 `uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA`——它解决的是「不想为了快而另训一个模型」。至于 `Mixture of Experts (MoE)`，`Switch Transformer` 摘要给的是 `a sparsely-activated model -- with outrageous numbers of parameters -- but a constant computational cost`，并坦承旧有障碍是 `complexity, communication costs and training instability`，其结果是 `up to 7x increases in pre-training speed`、可用 `bfloat16` 训练、以及 `trillion parameter models` 与 `4x speedup over the T5-XXL model`。三条线都不改「长度平方」这条根本约束，但都改了它的实际账单。

**拆解**：

### 平方成本究竟花在哪

第 4 章说每个新位置都要和全部历史算相关度，第 7 章说历史键值必须常驻显存。二者叠起来就是长序列的双重代价：算术量随长度平方，缓存也随长度线性增长并占用 `GPU 显存占用`。`FlashAttention` 摘要的措辞是 `the time and memory complexity of self-attention are quadratic in sequence length`。这里容易漏掉一点：它同时是时间复杂度与内存复杂度，所以只优化其中一个方向往往看不到端到端收益——这正是 `often do not achieve wall-clock speedup` 那句在说的事。综合报告口径列 `MLA`、`GQA`、`稀疏/线性注意力` 与 `KV 量化`、`token 淘汰` 时，把它们都归在 `推理与效率层`，也是因为它们的共同目标都是压这份账单。

### IO-aware 为什么反而更快

在 GPU 上，把一块数据从 `GPU high bandwidth memory (HBM)` 搬到 `GPU on-chip SRAM` 往往比在片上做几次乘加更贵。`FlashAttention` 的 `an IO-aware exact attention algorithm that uses tiling` 就是把计算切成能放进 SRAM 的块，尽量让数据在片上被反复用，从而 `reduce the number of memory reads/writes`。摘要还给了理论支撑：`it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes`。这条思路的价值超出注意力本身：它示范了「先量瓶颈在哪一级存储，再决定优化什么」的方法，而不是默认算力就是瓶颈。第 7 章 `PagedAttention` 的 `non-contiguous memory space` 与这里 `tiling` 是同一类系统思维，只是作用在缓存分配与内核计算两处。

### GQA 与 MoE 各减哪一种「大」

`grouped-query attention (GQA)` 减的是解码时键值头的数量：`Multi-query attention (MQA)` 用 `a single key-value head` 换速度，代价是 `quality degradation`，GQA 折中到 `an intermediate (more than one, less than number of query heads) number of key-value heads`，并用 `5% of original pre-training compute` 把已有 `multi-head language model checkpoints` 升训过来。`Mixture of Experts (MoE)` 减的则是每 token 的激活参数：`selects different parameters for each incoming example`，于是 `a constant computational cost`。综合报告口径提醒 `总参数` 与每 token 实际计算量是两个不同概念，是选型时最容易误判的地方——这句是二手转述，但与 `Switch Transformer` 的摘要方向一致。两者都不承诺长文本效果更好，只承诺同样长度下更省。

### 窗口里的信息不是均匀被用

综合报告口径把 `lost in the middle` 列为长上下文的第三条约束：开头结尾的信息被关注得更多，中段容易被忽略，因此 `长上下文 ≠ 更好的效果`。这与前两条是不同性质的问题——前面是成本，这里是有效性；把 128K 塞满既可能更贵，也可能更差。同一份报告在应用层给的解法是 `上下文裁剪与摘要`、`结构化输出` 与把 `RAG` 做成 `分块 → 嵌入 → 索引 → 检索 → 重排 → 带引用生成`，即主动决定哪些内容进窗口，而不是指望窗口无限大。第 9 章会看到，`RAG` 原点论文早在 2020 年就把 `providing provenance for their decisions` 与 `updating their world knowledge remain open research problems` 列为出发点，两者其实是同一个问题的两面：知识不该全押在参数里，也不该全堆进上下文。

**今天怎么做**：挑一条长提示词，把同一事实分别放在开头、中段、结尾各问一次，记录三次回答是否一致。**延伸，非语料原文**：再逐步缩短中段内容看 `TTFT` 与答案稳定性各自怎么变，把两条曲线画在一起；这个对照练习是编排者设计的，不是任何来源的表述。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `Approximate attention methods` 一定更快——摘要写 `often do not achieve wall-clock speedup`；以为 `FlashAttention` 是近似算法——它是 `an IO-aware exact attention algorithm`；以为 `GQA` 等于 `Multi-query attention (MQA)`——前者是 `an intermediate (more than one, less than number of query heads) number of key-value heads`，后者 `only uses a single key-value head` 且可能 `quality degradation`；以为 `MoE` 的 `trillion parameter models` 意味着每 token 要算万亿参数——摘要明写 `a constant computational cost`；以为窗口越大效果越好——综合报告口径同时给了 `100K token 的成本接近 10K 的百倍` 与 `lost in the middle`；还有把 `128K`、`1M`、`MLA` 当一手结论，它们只出现在二手汇总里。

**记住这一句**：长上下文是平方计算、显存搬运与 `lost in the middle` 三条约束的叠加；`FlashAttention` 用 `IO-aware` 的 `tiling` 省读写、`GQA` 省键值头、`MoE` 省激活参数，但没有一条承诺中段信息因此被看见。

</div>
