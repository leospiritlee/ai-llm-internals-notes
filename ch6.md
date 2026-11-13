# 第6章 后训练：SFT、RLHF、DPO 与微调边界

**一句话**：`InstructGPT` 用 `labeler demonstrations of the desired model behavior` 加 `a dataset of rankings of model outputs` 让 `1.3B parameter InstructGPT model` 被偏好于 `175B GPT-3`；`DPO` 又把 `RLHF is a complex and often unstable procedure` 换成 `simple classification loss`；`LoRA` 则说明适配下游时 `freezes the pre-trained model weights` 也能不掉质量。

**为什么需要它**：第 5 章结尾那句 `not aligned with their users` 是这一章的全部起点。基座模型被训练成续写器，你给它一句问话，它最自然的反应是续写更多问话。让它「听话」不是提示词技巧能替代的事，而是要在权重里再走一遍训练。`Training language models to follow instructions with human feedback` 给出的路径是三段：先收人工写的示范做监督微调，再收输出排序做 `reinforcement learning from human feedback`，最后得到 `InstructGPT`。这一步改变了产品形态——同一个底座，后训练之前是补全接口，之后才是对话接口。综合报告口径把它排进四段式流水线的 `SFT` 与对齐两段，并注明 `RLHF（需 Reward Model + PPO）→ DPO（免 Reward Model，更稳定）→ GRPO / DAPO（免 Critic）→ RLVR / 可验证奖励` 的演进；这是二手编排，本书只在有 A 级摘要支撑的 `SFT`、`RLHF`、`DPO` 上做结论。

**核心结构**：把后训练拆成三条独立的线，别混成一条。第一条是行为对齐：`fine-tuning with human feedback` 的总名下，先用 `a dataset of labeler demonstrations of the desired model behavior` 做 `supervised learning` 微调 `GPT-3`，再用 `a dataset of rankings of model outputs` 做 `reinforcement learning from human feedback`，产出 `InstructGPT`；摘要给的结果是 `outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters`，同时 `improvements in truthfulness and reductions in toxic output generation`，且 `minimal performance regressions on public NLP datasets`。第二条是简化对齐：`DPO` 把 `first fitting a reward model that reflects the human preferences` 这一步 `enables extraction of the corresponding optimal policy in closed form`，于是 `solve the standard RLHF problem with only a simple classification loss`；其自述特性是 `stable, performant, and computationally lightweight`，并 `eliminating the need for sampling from the LM during fine-tuning`，实验上 `fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations`。第三条是任务适配：`LoRA` 面对 `full fine-tuning, which retrains all model parameters, becomes less feasible`，做法是 `freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture`。三条线管的分别是「听不听话」「对齐流程重不重」「换个任务要付多少钱」。

**关键设计**：**用 100x 更少的参数赢下人工偏好，说明「可用」主要不是规模属性，而是训练目标属性**。`InstructGPT` 摘要那句 `despite having 100x fewer parameters` 是全章最该被记住的证据，它把第 5 章「规模是主变量」的适用范围切清楚了：规模决定能力上限，但「是否按用户意图作答」由后训练的目标决定。同一篇也诚实地写了 `Even though InstructGPT still makes simple mistakes`，没有把对齐写成万能药。第二处设计判断在 `DPO`：它并不否认 RLHF 的框架，而是 `introduce a new parameterization of the reward model in RLHF`，从而把两阶段流程压成 `simple classification loss`；这解释了为什么 `eliminating the need for sampling from the LM during fine-tuning` 与省掉 `performing significant hyperparameter tuning` 会同时出现在同一句里。第三处在 `LoRA`：`Compared to GPT-3 175B fine-tuned with Adam`，可训练参数 `reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times`，且 `unlike adapters, no additional inference latency`——注意「不加推理延迟」是它区别于适配器类方法的关键，不是所有参数高效方法都有这条。把三处放在一起，后训练的选型就变成了三个独立问题：要不要对齐、用什么代价对齐、要不要为领域再训一份权重。

**拆解**：

### 三段式里的每段各补什么

第一段示范数据解决「输出长什么样」：`labeler-written prompts and prompts submitted through the OpenAI API` 起头，再由人工写出期望行为，`fine-tune GPT-3 using supervised learning`。第二段排序数据解决「多个候选里哪个更好」：`a dataset of rankings of model outputs` 训练出偏好信号，`further fine-tune this supervised model using reinforcement learning from human feedback`；这一步需要的正是第 2 章 C 级教材里的 `强化学习` 底座——奖励不必可微，策略仍可被优化。第三段是评测口径：摘要特意说 `In human evaluations on our prompt distribution`，也就是结论只在该提示分布上成立。综合报告口径把这两段称为 `SFT` 与 `RLHF`，并给出 `对齐税` 的 `10–20%` 能力回退估算；本书照录它是二手转述，不把它写成论文测得的数字。

### DPO 到底省掉了什么

`RLHF is a complex and often unstable procedure` 之后，摘要把复杂度来源列得很清楚：`first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model`。`DPO` 的改动是 `a new parameterization of the reward model in RLHF`，使 `extraction of the corresponding optimal policy in closed form` 成为可能，于是 `only a simple classification loss` 就够。省掉的不只是一个模型，还有 `sampling from the LM during fine-tuning` 与 `significant hyperparameter tuning`，以及由此带来的不稳定。摘要同时给了边界：`DPO can fine-tune LMs to align with human preferences as well as or better than existing methods`，在 `summarization and single-turn dialogue` 上是 `matches or improves response quality`，没有宣称全面超越。把这种「省了什么、没承诺什么」读出来，比记住缩写更有用。

### LoRA 与「何时微调」的分界

`LoRA` 的动机句很具体：`deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive`。它 `freezes the pre-trained model weights`，只 `injects trainable rank decomposition matrices into each layer of the Transformer architecture`，并在 `rank-deficiency in language model adaptation` 上给了实证解释。效果口径是 `on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3`，同时 `a higher training throughput` 且 `no additional inference latency`。但「能不能便宜地微调」与「该不该微调」是两件事：综合报告口径在 `阶段 6` 直接写了判断顺序——`注入新事实用 RAG，改变风格/格式/领域术语才用微调`，实操是 `Hugging Face PEFT + TRL`，并要求与基座模型做对照评测。第 9 章会从 `RAG` 一侧把这条分界讲完：事实是易变的、需要出处的，压进权重既难更新也难溯源。

### 后训练到推理的接口

对齐改变的是权重，不是推理的计算形状。无论 `SFT`、`RLHF` 还是 `DPO` 产出的模型，生成时仍是逐 token 回看历史，仍需 `KV cache`，仍受 `quadratic in sequence length` 约束。综合报告口径把 `训练与对齐层` 与 `推理与效率层` 明确分成两层，正是为了挡住一种常见错位：线上答得慢、答得贵，就想去「再微调一下」。第 7 章会用 `PagedAttention` 的 `without requiring any model architecture changes` 给出反例——`24x higher throughput` 是在权重一行没改的前提下拿到的。同理，答案事实错、没出处，该去第 9 章查检索；答非所问才回这一章。把「现象 → 层」的映射记牢，是第 1 章六层地图第一次真正派上用场。

**今天怎么做**：把你最近一次想微调的需求写成两行——要注入的是「新事实」还是「风格与格式」，再对照 `注入新事实用 RAG，改变风格/格式/领域术语才用微调` 判一次。**延伸，非语料原文**：如果落在前者，先去搭一个最小检索再回来重判；这条动作顺序是编排者给的练习，不是任何来源的表述。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `InstructGPT` 赢在参数更多——摘要明写 `despite having 100x fewer parameters`，且结论限定在 `our prompt distribution`；以为对齐能消除错误——同一篇写了 `Even though InstructGPT still makes simple mistakes`；以为 `DPO` 否定了 `RLHF`——它改的是 `a new parameterization of the reward model in RLHF`；以为 `LoRA` 等于量化或蒸馏——它 `freezes the pre-trained model weights` 并注入 `trainable rank decomposition matrices`，与 `量化`、`剪枝` 是不同动作；以为 `no additional inference latency` 是所有适配器方法的共性——摘要特意用 `unlike adapters` 把这点单挑出来；还有把 `GRPO`、`RLVR`、`对齐税` 当一手结论，它们只出现在综合报告口径里。

**记住这一句**：规模决定能力上限，后训练决定权好不好用；SFT 教形状、RLHF 与 DPO 教偏好、LoRA 降适配成本，而事实性缺口该交给第 9 章的检索，不是再训一次权重。

</div>
