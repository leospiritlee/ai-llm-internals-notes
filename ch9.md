# 第9章 知识接入：RAG 与检索

**一句话**：`RAG` 把 `store factual knowledge in their parameters` 的权重记忆，和 `a dense vector index of Wikipedia` 这样的非参数记忆并起来，用 `a pre-trained neural retriever` 取证据，从而同时回应 `providing provenance for their decisions` 与 `updating their world knowledge remain open research problems`。

**为什么需要它**：第 6 章留了一道分界——`注入新事实用 RAG，改变风格/格式/领域术语才用微调`；第 8 章又说明把材料全堆进上下文既贵又未必有效。两处都指向同一个设计：事实放在可查、可换、可溯源的地方，让权重只负责语言与推理。`Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks` 的一手摘要把动因写得比任何工程口号都清楚：大预训练模型确实 `store factual knowledge in their parameters` 且微调后在下游很好，但 `their ability to access and precisely manipulate knowledge is still limited`，在知识密集任务上 `their performance lags behind task-specific architectures`；此外 `providing provenance for their decisions and updating their world knowledge remain open research problems`。换句话说，RAG 不是「怕模型不知道」的补丁，而是针对「查不准、改不动、说不清出处」三件事的结构性回答。综合报告口径把 `阶段 4` 称为 `投入回报最高` 的一段，也是二手说法。

**核心结构**：论文的定义句是 `We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever`。两种记忆的分工因此非常明确：`parametric` 指参数里学到的语言与泛化能力，`non-parametric memory` 指可独立更新的外部索引。论文还比较了两种配方：`one which conditions on the same retrieved passages across the whole generated sequence`，以及 `the other can use different passages per token`——同一份证据贯穿全文，还是每个 token 各取各的。它承接的先前工作是 `Pre-trained models with a differentiable access mechanism to explicit non-parametric memory`，但指出 `have so far been only investigated for extractive downstream tasks`，所以本文给的是 `a general-purpose fine-tuning recipe`。结果侧：`set the state-of-the-art on three open domain QA tasks`，生成上 `generate more specific, diverse and factual language`。这篇 `Accepted at NeurIPS 2020`，与第 4 章的 `seq2seq` 路线一脉相承。

**关键设计**：**可溯源与可更新，是「知识该放哪」这道题真正的判据，而不是模型记不记得住**。摘要把这两个 open problem 与 `access and precisely manipulate knowledge` 并列，等于说：就算参数里存着事实，只要不能指出出处、不能随世界变化而更新，知识密集任务就仍会输给 `task-specific architectures`。`dense vector index of Wikipedia` 之所以关键，是因为索引是独立于权重的对象——换语料不必重训，`a pre-trained neural retriever` 负责把问题映射到该取哪几段。生成侧的收益也写得很具体：`more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline`，注意对照项是纯参数基线，不是「比大模型聪明」。把这条与第 5、6 章连起来：`Scaling Laws` 说规模决定能力上限，`InstructGPT` 说后训练决定行为，`LoRA` 说适配能便宜，但没有一条能替代「事实从哪来、怎么证明」；这就是第 1 章六层地图里 `应用层` 不可替代的原因。综合报告口径把今天的工程链路写成 `分块` → `嵌入` → `向量库` → `检索` → `重排` → `带引用生成`，并称 `混合检索（BM25 + 向量）` 加 `reranker` 是生产基线；这是二手编排，本书用它说明位置，不把它写成论文结论。

**拆解**：

### 参数记忆与非参数记忆

`parametric and non-parametric memory` 的组合不是把两份答案拼起来，而是把「怎么说话」与「依据什么说」拆开。参数侧是 `a pre-trained seq2seq model`，负责语言组织；非参数侧是 `dense vector index of Wikipedia`，负责提供可替换的证据。论文在摘要里特意点出先前 `differentiable access mechanism` 的工作 `only investigated for extractive downstream tasks`，也就是说，早期做法只能从文章里抽片段，而 RAG 要的是 `for language generation`。这个区别落到线上很清楚：抽取式系统给出的是原文句子，生成式系统要能综合多段证据并说出自己的话，同时还得 `providing provenance for their decisions`。

### 两种配方与检索粒度

`same retrieved passages across the whole generated sequence` 是整篇共用一组证据，实现简单、出处好对齐；`different passages per token` 允许生成过程中换证据，适合需要多跳或长输出的任务，但溯源链更复杂。这个取舍后来在工程里演化成 `分块`、`重排` 与 `带引用生成` 的粒度问题。综合报告口径给的生产基线是 `混合检索（BM25 + 向量）` 加 `reranker`，并把 `cross-encoder` 列为重排手段，还提到 `RAG` 已从 `top-k + 祈祷` 演进到 `Agentic RAG`；这些都是二手汇总里的编排，本书只用来标明「检索不是一次 top-k 就完事」这个方向。第 12 章会把检索质量单独列为评测维度：`faithfulness`、`answer relevancy`、`context precision/recall`。

### 什么时候该检索、什么时候该微调

第 6 章给过判断顺序，这里补上证据侧的理由：如果需求是「知识变了要跟着变」，`updating their world knowledge remain open research problems` 说明重训不是好答案；如果需求是「答案要给客户看」，`providing provenance for their decisions` 说明必须有外部证据；如果需求只是「输出格式统一、术语一致」，`LoRA` 那类 `freezes the pre-trained model weights` 的适配更对路。综合报告口径在 `阶段 6` 要求「能讲清 `何时微调、何时提示、何时检索` 就是达标」，并让实操与基座模型做对照评测——这句话是二手学习路径，但和一手摘要的三条 open problem 并不冲突。反过来，若知识体量小且稳定、又不需要出处，硬搭一套索引只是增加失败点；「要不要搭」永远先于「怎么搭好」。

### 检索链路自己也会出事

把知识搬到外部，风险也从权重搬到了索引与输入。`OWASP` 的十类里 `LLM08:2025 Vector and Embedding Weaknesses` 专列向量与嵌入弱点，`LLM01:2025 Prompt Injection` 则可能借被检索回来的文档混进指令——这意味着被检索内容属于不可信输入，不能因为它「在知识库里」就当成可信指令。综合报告口径在 `生产与安全层` 把 `数据泄露`、`Prompt 注入` 与 `工具过度授权` 并列，也提示要在架构设计时一并考虑。第 12 章会把这十类展开成准入表；这里只需先记住一条：`RAG` 解决了出处与更新，但没有自动解决注入与权限，二者属于不同层。

**今天怎么做**：挑一条你答不上出处的问题，分别用「只改提示词」与「挂一份带来源的检索」各做一次，把两次回答连同引用一起存下来。**延伸，非语料原文**：再故意把知识库里的一条事实改掉，看系统能否只换索引就跟着变；这个验证动作是编排者给的练习，不是任何来源的表述。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `RAG` 是给小模型补知识——摘要对照的是 `a state-of-the-art parametric-only seq2seq baseline`，问题在 `access and precisely manipulate knowledge`；以为 `non-parametric memory` 就是向量数据库——论文写的是 `a dense vector index of Wikipedia, accessed with a pre-trained neural retriever`，检索器与索引是两件东西；以为 `different passages per token` 一定更好——它只是两种被比较的 RAG 配方之一；以为有了检索就不需要评测——综合报告口径把 `faithfulness` 与 `context precision/recall` 单列为 `RAG 专项`；以为 `Agentic RAG`、`混合检索` 是论文原文——它们出自二手汇总；还有以为索引里的文本天然可信，`LLM08:2025 Vector and Embedding Weaknesses` 与 `LLM01:2025 Prompt Injection` 正是反例。

**记住这一句**：`parametric and non-parametric memory` 的分工，决定事实该进权重还是进索引；`RAG` 真正回答的是 `providing provenance for their decisions` 与 `updating their world knowledge remain open research problems`，而出处、更新与不可信输入是三件要分开处理的事。

</div>
