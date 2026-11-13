# 第4章 架构：Transformer 与注意力

**一句话**：`Attention Is All You Need` 做的事只有一件——用 `based solely on attention mechanisms` 换掉 `dispensing with recurrence and convolutions entirely`，把序列里任意两个位置的距离压成一步；换来的是 `more parallelizable` 和显著更短的训练时间，也埋下了第 8 章那笔 `quadratic in sequence length` 的账。

**为什么需要它**：第 3 章把文字变成了 `input_ids`，第 2 章说清训练循环一直是前向、损失、`反向传播`、更新。缺的那一块是：这些 id 怎样互相看见？一手摘要开头就点明旧范式的形状——`The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration`，最好的模型还要再 `connect the encoder and decoder through an attention mechanism`。也就是说，注意力原本只是循环网络旁边的补丁，2017 年那篇把补丁提拔成主干。对读者而言这一章的价值不在复述公式，而在建立一个可迁移的判断：模型能不能并行、长序列贵在哪里、`KV cache` 为什么存在，全部由「位置之间怎么连」这一处决定。综合报告口径把主干收敛写作 `Decoder-only Transformer` 加 `RMSNorm`、`SwiGLU`、`RoPE`，并称 2026 年这批前沿开放模型里已没有人再用 `LayerNorm` 或学习式位置编码；这是二手转述，本书只当编排背景，不写成论文结论。

**核心结构**：把一次前向拆成四段就够用。第一段是表示，`embedding` 把 `input_ids` 映射成向量，`位置编码` 补上顺序信息——注意力本身对顺序不敏感，不补就分不清「猫追狗」和「狗追猫」。第二段是注意力：每个位置拿自己的查询去和所有位置的键算相关度，用 `softmax` 归一后对值加权求和，于是 `self-attention` 让任意两位置一步直达。第三段是多头与堆叠，`multi-head attention` 把表示分到若干子空间并行看不同关系，再层层叠上去；`FFN` 在每层里做逐位置的非线性变换。第四段是输出头，把最后一个位置的向量投回词表，得到下一个 token 的分布。`Hugging Face` 课程导言页把这一层的掌握标准写成三条并列目标：`About the Transformer architecture`、`How to distinguish between encoder, decoder, and encoder-decoder architectures and use cases`、`Fine-tuning a pretrained model`，并说明 `Chapters 1 to 4` 就是为此设计的。第 2 章那两条产品路线，在这里有了结构解释：`BERT` 用 `both left and right context` 看全句，`GPT-3` 用 `autoregressive language model` 只看左边。

**关键设计**：**去掉循环不是为了 elegance，而是为了让训练能并行，代价是把长度变成了平方项**。一手摘要把收益写得很具体：`superior in quality while being more parallelizable and requiring significantly less time to train`；在 `WMT 2014 English-to-German` 上拿到 `28.4 BLEU`、比当时包括集成在内的最好结果高出 `over 2 BLEU`，在 `WMT 2014 English-to-French` 上以 `3.5 days on eight GPUs` 取得 `41.8` 的新单模型最好成绩，且只是文献中最好模型训练成本的 `a small fraction`。摘要还用 `English constituency parsing` 证明这套结构不只在翻译上成立。但同一篇留下了一个后来十年都在还的债：`FlashAttention` 的摘要开头直接写 `Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length`。注意这里的两笔账不是一回事——训练时的并行收益来自「不必等上一步」，推理时的平方成本来自「每个新 token 都要回看全部历史」；第 7 章的 `KV cache` 正是为后者做的缓存，第 8 章的 `FlashAttention` 与 `GQA` 则分别处理内存读写与键值头数量。把这三处连起来看，才不会以为「并行」等于「不贵」。

**拆解**：

### 编码器、解码器与编码器—解码器

一手摘要里的 `encoder-decoder configuration` 是翻译这类「整句进、整句出」任务的形状：编码器读全句，解码器逐词生成，中间的 `attention mechanism` 负责对齐。`BERT` 走的是纯编码器一侧，摘要写 `designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers`，下游只需 `fine-tuned with just one additional output layer`；`GPT-3` 走纯解码器一侧，`autoregressive language model` 且 `without any gradient updates or fine-tuning`，任务与示例全靠 `text interaction with the model` 指定。课程导言页把三类结构的区分列为明确学习目标，正是因为这三类不是新旧版本，而是三种不同的可见性约束。第 9 章会看到 `RAG` 原点用的仍是 `a pre-trained seq2seq model`，说明 2020 年那套编码器—解码器在知识密集任务上还有位置。

### 一次注意力在算什么

不需要推导矩阵，只需要记住三个角色。查询是当前位置发出的「我想找什么」，键是每个位置挂出的「我有什么」，值是被取走的实际内容；`Q/K/V` 是综合报告口径给 `阶段 2` 列出的最低直觉项之一，同页还要求能说清 `tokenization`、`embedding`、多头注意力、`位置编码`、解码策略、`KV cache` 为何存在与上下文窗口为何有代价，并明确不要求论文级数学推导。这一步最容易被忽略的是掩码：解码器必须遮住未来位置，否则训练时就能抄答案。掩码一加，「看全句」和「只看左边」就从两种模型变成了同一个算子上的两种遮法，第 2 章那条路线分界由此收进一行配置。

### 为什么并行的另一面是显存

`more parallelizable` 说的是同一层内所有位置可以同时算，不再有循环依赖。但生成是逐 token 的：算完第 t 个，第 t+1 个又要和前面所有位置重新算一遍相关度。工程上的直接后果是，历史位置的键和值必须留在显存里，这就是第 7 章 `KV cache` 的由来。综合报告口径把 `推理与效率层` 列为工程师最该投入的一层，理由正在这里：它不改变模型能力，却同时决定 `TTFT`、`p95` 与 `每千 token 成本`。`FlashAttention` 给出的另一条线索是，长序列上真正的瓶颈常不是算力而是 `IO-aware` 所指向的显存读写，`tiling` 减少的是 `GPU high bandwidth memory (HBM)` 与 `GPU on-chip SRAM` 之间的搬运次数。本章只需记住因果链，具体机制放在第 8 章。

### 架构层到预训练层的接口

`Attention Is All You Need` 报的是翻译与句法任务上的质量，它并没有回答「模型为什么会回答问题」。这个问题要到 `Language Models are Few-Shot Learners` 才有转机：规模上到 `175 billion parameters` 后，`few-shot setting` 里不给梯度更新也能做新任务。换句话说，架构提供了可并行、可堆叠的计算骨架，能力要靠第 5 章的目标与数据灌进去，行为要靠第 6 章的后训练掰过来。综合报告口径把架构层与 `训练与对齐层` 分成两层，理由就是这个：前者决定你能做出什么选择，后者决定权重里到底有什么。把「换个结构」当成「换个能力」，是 reading 论文时最常见的错位。

**今天怎么做**：挑一个你能加载的小模型，打印一次前向里 `attention` 相关张量的形状，再单独记下序列长度翻倍时形状怎么变。**延伸，非语料原文**：把「形状变化」和账单上的延迟变化放在一张纸上对照，这个观察练习是编排者设计的，不是任何来源的表述。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `Transformer` 等于 `GPT-3`——前者是 2017 年的 `encoder-decoder configuration`，后者是 2020 年的 `autoregressive language model`；以为 `dispensing with recurrence and convolutions entirely` 意味着不再需要顺序信息——`位置编码` 正是补这一课，综合报告口径里 `RoPE` 已是主干默认；以为 `more parallelizable` 等于推理也便宜——`quadratic in sequence length` 与 `KV cache` 都来自同一个逐 token 回看动作；以为 `BERT` 被淘汰了——`How to distinguish between encoder, decoder, and encoder-decoder architectures and use cases` 至今仍是课程目标，第 9 章的 `seq2seq` 也还在用；还有把综合报告口径里的 `RMSNorm`、`SwiGLU`、`MLA`、`GQA` 当成一手论文结论，它们只是二手汇总里的编排背景。

**记住这一句**：注意力把位置之间的距离压成一步，换来了并行与更短的训练，也把长度写成了平方成本；架构只给骨架，能力来自第 5 章，行为来自第 6 章。

</div>
