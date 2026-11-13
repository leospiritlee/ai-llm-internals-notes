# 第2章 底座：从梯度下降到语言模型

**一句话**：`Transformer` 不是凭空出现的，它站在 `神经网络`、`梯度`、`反向传播`、`损失函数` 这套优化机器上；语言模型只是把「下一个单位是什么」变成了一个可微的训练目标，而 `BERT` 与 `GPT-3` 在这个目标上走向了两侧。

**为什么需要它**：本地四本教材里，`神经网络` 出现 1329 次、`梯度` 1076 次、`反向传播` 321 次、`卷积` 728 次、`强化学习` 481 次，却没有 `Transformer` 与预训练。这不是缺陷，而是年代：它们训练的是「给一个输入、收敛到一个判断」的机器。要读懂第 5 章为什么 `cross-entropy loss` 会随规模呈 `power-law`，必须先接受一件事——训练永远是同一套循环：前向算出预测，用 `损失函数` 量出差距，靠 `反向传播` 得到 `梯度`，再用 `学习率` 挪动 `权重`。换模型结构，换的只是前向那一步怎么算。

**核心结构**：先把底座拆成四段。第一段是表示：输入被编码成数字，`词向量` 这类稠密表示让语义相近的东西在空间里也相近。第二段是计算：一层层线性变换加 `激活函数`，`卷积` 是其中一种针对局部结构的先验。第三段是目标：`损失函数` 把「错得多离谱」压成一个标量，`交叉熵` 常用于分类。第四段是求解：`反向传播` 把误差逐层分摊回 `参数`，`梯度` 指出下降方向，`学习率`、`批量` 与 `迭代` 决定走多快走多久；`过拟合` 与 `正则化` 处理的是「在训练集上漂亮、在新数据上崩掉」。这四段在第 5 章会原封不动地出现在 `pre-training` 里，只是数据从标注样本换成了海量文本，目标换成了预测下一个 token。

**关键设计**：**同一个目标，两种读上下文的方式，分出了两条产品路线**。`BERT` 的摘要写得很清楚：`deep bidirectional representations`，靠 `unlabeled text` 里的 `both left and right context` 预训练，下游只需 `one additional output layer` 微调，就能在 `GLUE`、`MultiNLI`、`SQuAD v1.1` 上刷新结果；这是「先学表示、再接任务头」的判别式路线。`Language Models are Few-Shot Learners` 则是另一头：`GPT-3` 是 `autoregressive language model`，`175 billion parameters`，在 `few-shot setting` 里 `without any gradient updates or fine-tuning`，任务与示例全靠 `text interaction with the model` 指定。前者把能力压进表示、靠微调适配；后者把能力压进规模、靠上下文适配。`Hugging Face` 课程导言页把这条分界列为学习目标原句：`How to distinguish between encoder, decoder, and encoder-decoder architectures and use cases`。第 6 章会看到，正因为 `GPT-3` 这条路线不做梯度更新，「让它听话」才需要另发明一整套后训练。

**拆解**：

### 四本 C 级教材各补哪一段

《深度学习的数学》对应第一段与第二段，把 `梯度`、`损失函数`、`卷积` 与 `激活函数` 的算术摊开；《深度学习》与《深度学习：智能时代的核心驱动力量》覆盖网络结构、训练流程与 `过拟合`，是「优化机器」的整体图；《深入浅出 强化学习原理入门》单讲 `强化学习`，对应第 6 章的 `reinforcement learning from human feedback` 与第 10 章 `DeepSeek-R1` 的 `pure reinforcement learning (RL)`。把强化学习放在这里不是凑数：第 6 章的 `a dataset of rankings of model outputs` 之所以能变成训练信号，靠的正是「奖励不必可微、但策略可以被优化」这条思路；没有这一章的底座，`PPO-based RLHF` 与 `DPO` 的差别就只剩名词。

### 从「学一个判断」到「学一个分布」

C 级教材里的典型任务是给定输入、输出一个标签，损失衡量预测与标签的差距。语言模型换的是目标形状：综合报告口径把预训练写成 `next-token prediction`，A 级 `Scaling Laws for Neural Language Models` 把被优化的量写成 `cross-entropy loss`。两者说的是同一件事——每一步都在问「词表里下一个是谁」，用 `softmax` 把打分变成分布，再用交叉熵惩罚把概率分给了错误答案。这个改写影响深远：标签不需要人标，语料本身就是监督信号，于是规模第一次成了可以无限加的主变量，也就有了第 5 章 `power-law` 与 `seven orders of magnitude` 那条经验规律。

### 为什么规模会变成主角

`Language Models are Few-Shot Learners` 的摘要把逻辑写得很直白：先前范式是 `pre-training` 加 `fine-tuning`，虽然架构 `task-agnostic`，仍需要成千上万条任务专用样本；而人可以只靠几个例子或一句指令做新任务。于是作者把「规模」当成变量去试，结论是 scaling up language models greatly improves `task-agnostic, few-shot performance`，甚至能追平当时的微调最好结果，并且 `3-digit arithmetic` 这类任务也能在 `few-shot` 里出现。注意摘要同时留了反面：`few-shot learning still struggles` 的数据集、以及在大规模网页语料上训练带来的方法学问题都照写了。第 5 章讲缩放定律，第 6 章讲怎么补「听不懂意图」，都是从这两句留白长出来的。

### 底座到第 3 章的接口

`Hugging Face` 的 `Tokenizers` 章第一句就是 `Models can only process numbers`，第二句 `translate text into data that can be processed by the model`。也就是说，第 2 章那台优化机器再完整，也必须先有人把「字」变成「数」。`词向量` 是老答案——先切词、再给每个词一个向量；第 3 章的新答案是 `subword tokenization`，先把词拆成可复用的片段，再查表。这一步不是预处理细节：它决定了序列长度、词表规模、`embedding` 矩阵大小，进而决定第 7 章 `KV cache` 的体量和第 8 章长上下文的成本。把这条链记住，后面每一章的「为什么贵」都能回答得更准。

**今天怎么做**：找一张纸，把「前向 → 损失 → 反向传播 → 更新」画成四个框，再在上面标出你正在用的模型哪一步被替换过。**延伸，非语料原文**：标不出来的那一格，就是你需要回头补的那一章；这个自查动作是编排者设计的学习建议，不是任何来源的表述。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `Transformer` 推翻了梯度下降——它替换的是前向计算里的循环，训练循环一点没变；以为 `BERT` 是 `GPT-3` 的旧版本——两者是 `both left and right context` 与 `autoregressive language model` 两条不同目标；以为 `few-shot` 也在更新权重——摘要明写 `without any gradient updates or fine-tuning`；以为 `175 billion parameters` 是「效果好的原因」——`Training Compute-Optimal Large Language Models` 紧接着就指出大模型 `significantly undertrained`，第 5 章会正面处理这个反转。

**记住这一句**：模型结构换的是前向怎么算，训练永远是损失、梯度与更新；语言模型把「下一个是谁」变成可微目标，才让规模第一次成了主变量。

</div>
