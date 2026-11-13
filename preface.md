## 序

上一本《AIAgent开发指南》讲的是「怎么让一次运行可信」，起点是一次工具调用，终点是能不能放量。这本往回退一层：那次调用里，模型内部到底发生了什么？为什么一段中文进去会变成一串数字，为什么窗口开到很长反而更慢更贵，为什么同样的权重换个解码策略就像换了个模型，以及为什么一个在 `MMLU` 上很好看的分数，跟你的线上服务能不能用，几乎是两件事。

先说清语料从哪来。对本地资料盘做只读扫描，书籍类文件 5319 份（排除 node_modules、coverage、src 等代码目录），严格关键词命中 8 份，其中真正有文字层、可作教材的只有 4 份：《深度学习》《深度学习：智能时代的核心驱动力量》《深度学习的数学》《深入浅出 强化学习原理入门》，合计 117.9 万字。这四本讲的是 `神经网络`、`梯度`、`反向传播`、`损失函数`、`卷积` 与 `强化学习`，没有一本涉及 `Transformer`、预训练或对齐——它们是很好的底座，但不是主流大模型这条栈。所以语料分三级：A 级是 2026-11-12 网络检索抓取的一手原文，包括 16 篇 arXiv 论文摘要、`Hugging Face` 的 `Tokenizers` 章与 `tokenizer_summary` 文档、`vLLM` 官方博客、`OWASP` 的 `2025 Top 10 Risk & Mitigations for LLMs and Gen AI Apps` 类目名，逐字保留；B 级是一份多源综合研究报告，提供六层栈与学习路径这类编排骨架；C 级就是那四本本地教材全文。合并约 122 万字。

三级口径不调和，也不替它们调和。最典型的一处：`Scaling Laws for Neural Language Models`（2020）说「更大的模型样本效率更高，因此算力最优的做法是训很大的模型、用相对不多的数据、并在收敛前显著提前停止」；`Training Compute-Optimal Large Language Models`（2022）说「当前大模型被严重训练不足，模型规模与训练 token 数应该等比例放大」。两篇都是一手摘要，结论方向相反。本书按各自年份照录，在第 5 章把分歧摆出来，不选边。同理，B 级报告里的比例、型号参数、阶段周数与趋势判断，一律写明「综合报告口径」，绝不写成论文或官方文档的结论；报告自述那句「具体模型参数、上下文长度与价格请以官方 model card 为准」也照录。

贯穿全书的一条判据是：**模型能力来自训练，服务质量来自系统，两者不是同一条曲线**。`Attention Is All You Need` 讲的是一个架构怎样用 `based solely on attention mechanisms` 取代循环；`Training language models to follow instructions with human feedback` 讲的是同一批权重怎样因为 `fine-tuning with human feedback` 而变得可用；`PagedAttention` 那篇博客讲的是权重一行没改、吞吐却差出 `24x higher throughput`。把这三件事放在一张图里，才知道该去改提示词、改权重，还是改服务引擎。

## 一条 token 的旅程

```text
┌──────────────────────────────────────────────────┐
│ 一条 token 的旅程：从一段文字到一次可信生成       │
│ 一 变数字   tokenizer > input_ids > embedding      │
│ 二 进架构   attention > 多头 > FFN > 层堆叠       │
│ 三 得能力   预训练 > SFT > RLHF / DPO > 对齐税    │
│ 四 会生成   decode > KV cache > PagedAttention    │
│ 五 接世界   RAG > 多模态 > 长上下文 > 工具        │
│ 六 敢用     MMLU > OWASP > TTFT > p95 > 灰度      │
└──────────────────────────────────────────────────┘
```

六段是依赖，不是并列的知识点。没搞懂 `tokenization` 就去看注意力，会以为模型在读字；没搞懂 `KV cache` 就去调并发，会以为加机器总能换吞吐；没搞懂 `pre-training` 与 `SFT` 的分工，会拿 `微调` 去灌本该用 `RAG` 灌的新事实；没建评测集就换模型，只能靠「感觉好像更聪明了」验收。顺序反过来的典型症状是：本地跑一次很惊艳，接上真实流量就同时暴露延迟、成本、幻觉三个问题，而这三个问题分别长在第 7、9、12 章。

## 十二章速查表

| 章 | 主题 | 一句话 |
|----|------|--------|
| 01 | 全局地图：一条 token 从文字到回答 | 六层栈把「模型长什么样」与「服务怎么交付」分开，`应用/AI 工程师` 不必先学预训练 |
| 02 | 底座：从梯度下降到语言模型 | `神经网络`、`梯度`、`反向传播`、`损失函数` 是 C 级教材的地基；`BERT` 与 `GPT-3` 在此分道 |
| 03 | 分词：文本怎么变成数字 | `BPE`、`WordPiece`、`Unigram`、`SentencePiece` 都是在词表大小与序列长度之间换 |
| 04 | 架构：Transformer 与注意力 | `based solely on attention mechanisms`、`dispensing with recurrence and convolutions entirely` 是原点 |
| 05 | 预训练：目标、规模与两版缩放口径 | `power-law` 之外还有一场 2020 对 2022 的口径冲突，本书照录不调和 |
| 06 | 后训练：SFT、RLHF、DPO 与微调边界 | `InstructGPT` 的 `1.3B` 打赢 `175B`，`DPO` 用 `simple classification loss` 替掉奖励模型 |
| 07 | 推理：一次生成在算什么 | `KV cache` 是 `Large` 且 `Dynamic` 的，`PagedAttention` 把它当虚拟内存管 |
| 08 | 长上下文：窗口越大越好的错觉 | `quadratic in sequence length`、`FlashAttention`、`GQA`、`lost in the middle` 四条约束 |
| 09 | 知识接入：RAG 与检索 | `parametric and non-parametric memory` 的分工，决定事实该进权重还是进索引 |
| 10 | 推理能力：CoT 与可验证奖励 | 从 `eight chain of thought exemplars` 到 `pure reinforcement learning (RL)` |
| 11 | 多模态：从 CLIP 到同一条序列 | `predicting which caption goes with which image` 把视觉拉进语言的空间 |
| 12 | 评测、安全与选型 | `MMLU` 测的是学术广度，`OWASP` 十类测的是会不会出事，`TTFT` 测的是用户等不等得起 |

## 遇到这些问题，去哪章找

- **「同一段中文，为什么不同模型收费不一样」** → 第 03 章（`subword tokenization`、`Byte-level BPE`、`50,257`）
- **「模型明明认识这两个词，为什么算不对」** → 第 03 章（`annoyingly`、`##er`、`tokenize()`）
- **「中文要不要先分词」** → 第 03 章（`SentencePiece`、`Chinese and Japanese`、`"▁"`）
- **「为什么长文本慢得不成比例」** → 第 04、08 章（`quadratic in sequence length`、`FlashAttention`）
- **「参数越多一定越好吗」** → 第 05 章（`power-law`、`significantly undertrained`、`70B parameters`）
- **「该多大数据量，看到两种说法」** → 第 05 章（`Larger models are significantly more sample-efficient` 对 `for every doubling of model size`）
- **「基座模型答得很绕，怎么让它听话」** → 第 06 章（`labeler demonstrations of the desired model behavior`、`reinforcement learning from human feedback`）
- **「什么时候该微调，什么时候该检索」** → 第 06、09 章（`LoRA`、`store factual knowledge in their parameters`）
- **「奖励模型太难训，有没有更省事的对齐」** → 第 06 章（`DPO`、`simple classification loss`）
- **「首字等半天，后面却很顺」** → 第 07 章（`KV cache`、`PagedAttention`、`batch more sequences together`）
- **「并发一上来就 OOM」** → 第 07 章（`60% – 80%`、`fragmentation and over-reservation`、`1.7GB`）
- **「窗口开到 128K，效果反而变差」** → 第 08 章（`lost in the middle`、`GQA`）
- **「知识变了，要不要重新训」** → 第 09 章（`updating their world knowledge remain open research problems`、`dense vector index of Wikipedia`）
- **「答案没有出处，不敢给客户看」** → 第 09 章（`providing provenance for their decisions`）
- **「数学题为什么让它先写步骤会更准」** → 第 10 章（`a series of intermediate reasoning steps`、`GSM8K`）
- **「图怎么进语言模型」** → 第 11 章（`400 million (image, text) pairs`、`zero-shot transfer`）
- **「榜单第一，为什么线上还是不行」** → 第 12 章（`57 tasks`、`lopsided performance`、`TTFT`）
- **「会不会被人一句话套走系统提示词」** → 第 12 章（`LLM01:2025 Prompt Injection`、`LLM07:2025 System Prompt Leakage`）

## 二十个自测问题

1. `Models can only process numbers` 这句之后，文本到数字具体分哪两步？
2. 为什么 `word-based tokenizer` 会逼出 `[UNK]`，而 `Character-based tokenizers` 又会把序列拉长？
3. `BPE` 与 `WordPiece` 选合并对的规则差在哪？
4. `Byte-level BPE` 为什么用 `256 byte values` 就能不再需要 `"<unk>"`？
5. `Unigram` 与 `BPE` 在确定性上的差别是什么？
6. `Attention Is All You Need` 去掉了什么，换来了 `more parallelizable`？
7. `encoder-decoder configuration` 与 `BERT` 的 `both left and right context` 有什么关系？
8. `GPT-3` 的 `few-shot setting` 为什么是 `without any gradient updates or fine-tuning`？
9. `cross-entropy loss` 随规模呈什么形状，跨了几个数量级？
10. 2020 年那篇为什么主张 `stopping significantly before convergence`？
11. `Chinchilla` 用多少参数、多少倍数据，赢过 `Gopher (280B)`？
12. `InstructGPT` 的 `1.3B parameter InstructGPT model` 凭什么被偏好于 `175B GPT-3`？
13. `RLHF is a complex and often unstable procedure` 之后，`DPO` 用哪一样东西替掉了它？
14. `LoRA` 冻结了什么、注入了什么，为什么 `no additional inference latency`？
15. `KV cache` 的两个形容词是什么，分别带来什么工程后果？
16. `PagedAttention` 借了操作系统的哪个概念，`block table` 映射的是哪两种块？
17. `FlashAttention` 说的 `IO-aware` 是在省什么，为什么说它是 `exact attention`？
18. `RAG` 里的 `parametric` 与 `non-parametric memory` 各指什么？
19. `DeepSeek-R1` 那篇为什么强调 `obviating the need for human-labeled reasoning trajectories`？
20. `MMLU` 的 `57 tasks` 之外，还应该看哪些工程口径指标？

## 一个提醒

这本册子有明确边界。综合报告口径里出现的 2026 年前沿型号名称、总参数与激活参数、上下文长度与名次，都是二手转述或行业估算，本书不把它们当作官方事实引用，也不做型号横评；**明确不写**任何需要以厂商 model card 为准的价格、版本与榜单排名。缩放定律两篇的分歧按年份照录，不替来源调和，也不挑一个当「正确答案」。架构演进只写到有一手摘要支撑的 `FlashAttention`、`GQA`、`Mixture of Experts (MoE)`、`Switch Transformer` 为止。

另一个边界是分工：本书不重复《AIAgent开发指南》里的工具调用协议、循环终止条件、上下文工程与上线准入。那本从一次工具调用出发，这本从一次分词出发；两本合起来，才是「模型里发生了什么」加「模型外要补什么」。

**延伸，非语料原文**：读的时候手边放一个能跑的小模型，每读完一章就做一件可以观察的小事，比如打印一次 `input_ids`、换一次解码参数、把同一段提示词加长一倍看延迟变化。这些练习是编排者给的建议，不是任何来源的表述。