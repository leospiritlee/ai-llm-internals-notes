# 第5章 预训练：目标、规模与两版缩放口径

**一句话**：预训练优化的是 `cross-entropy loss` 上的下一个 token，`Scaling Laws for Neural Language Models` 说它随规模呈 `power-law` 且主张大模型配相对少的数据、`stopping significantly before convergence`，两年后的 `Training Compute-Optimal Large Language Models` 说 `significantly undertrained`、要求 `for every doubling of model size` 把 token 数也翻倍；本书按年份照录，不选边。

**为什么需要它**：第 2 章末尾留了一个问题：把「下一个是谁」变成可微目标之后，标签不再需要人标，语料本身就是监督信号，于是规模第一次成了可以不断加大的主变量。规模一变成主变量，「给定算力该怎么分」就成了必须回答的工程问题。这一问题有两篇一手摘要给出方向相反的答案，且都是各自年份里被广泛引用的口径。多数科普会替读者调和成一个「后来修正了前者」的顺叙故事，但那不是来源自己的表述。本章把两份摘要并排放，正是为了让读者养成一个在整条 AI 栈里都有用的习惯：看到一个「规律」，先问它的自变量、因变量与实验范围各是什么。综合报告口径把 `next-token prediction` 列为四段式流水线第一段，并称「规模与数据质量决定上限」——这是二手转述，只作编排框架。

**核心结构**：先把 2020 年那篇的观察拆成四条。被优化的量是 `cross-entropy loss`；它 `scales as a power-law with model size, dataset size, and the amount of compute used for training`；有些趋势 `spanning more than seven orders of magnitude`；`Other architectural details such as network width or depth have minimal effects within a wide range`。摘要还说 `Simple equations govern the dependence of overfitting on model/dataset size`，并由这些关系 `determine the optimal allocation of a fixed compute budget`。结论句是 `Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data`。再看 2022 年那篇：它 `investigate the optimal model size and number of tokens`，训练了 `over 400 language models`、范围 `70 million to over 16 billion parameters` 与 `5 to 500 billion tokens`，结论是 `the model size and the number of training tokens should be scaled equally`。两篇的差别不在数学，而在「当时的大模型处在曲线哪一段」。

**关键设计**：**同一条 `power-law`，两次不同的算力分配结论，分歧的根源是「已有模型被训练到什么程度」**。2022 年摘要把原因写得毫不含糊：`current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant`。验证方式是训一个「预测出来的算力最优模型」`Chinchilla`，用与 `Gopher` 相同的算力预算，但 `70B parameters` 与 `4× more more data`；结果 `Chinchilla uniformly and significantly outperforms Gopher (280B)`、`GPT-3 (175B)`、`Jurassic-1 (178B)` 与 `Megatron-Turing NLG (530B)`，并在 `MMLU` 上达到 `67.5%`、比 `Gopher` 高出 `greater than a 7% improvement`。摘要还补了一个常被漏掉的下游收益：`This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage`。把两篇并起来读能得到一条不冲突的工程含义：算力预算固定时，模型大小与数据量必须一起算；但 2020 年那句 `stopping significantly before convergence` 仍按它自己的年份照录，本书不把它改写成「已被证伪」或「其实等价」。front 里那句「两篇都是一手摘要，结论方向相反」，是全书口径纪律的样板。

**拆解**：

### 被优化的量到底是什么

`cross-entropy loss` 衡量的是模型给「真正下一个 token」分了多少概率；用 `softmax` 把打分变成分布，再用 `交叉熵` 惩罚分错的位置。这一步从第 2 章那台优化机器原封不动搬来，只换了目标形状：不再有人工标签，每条文本自己就是样本与答案。C 级教材里的 `过拟合`、`正则化`、`批量`、`迭代` 在这里全部继续有效，`Scaling Laws` 摘要甚至专门写了 `the dependence of overfitting on model/dataset size`。理解这一点能挡住一类常见误判：把预训练想成「把知识写进参数」的加载过程，而忽略它是一次实实在在的、有损失有 `梯度` 有更新的训练；第 9 章讲 `store factual knowledge in their parameters` 时，这句话才有落点。

### 幂律为什么让预算可算

`power-law` 的价值不在「越大越好」这句口号，而在它把损失写成可外推的函数：给定算力，就能在模型大小与数据量之间求一个分配点。2020 年摘要说这些关系 `allow us to determine the optimal allocation of a fixed compute budget`，并指出宽深这类细节 `have minimal effects within a wide range`——这正是「规模成为主变量」的一手依据。但要注意 `some trends spanning more than seven orders of magnitude` 说的是观察范围，不是说任何项目都能外推七个数量级。综合报告口径把缩放定律列给 `算法/预训练工程师` 路线，而 `应用/AI 工程师` 只需知道因果；这条分界与第 1 章一致：绝大多数读者不需要复现 `compute-optimal training`，只需要明白为什么厂商会同时谈参数量和训练 token 数。

### 两版口径各自的自变量

2020 年那篇在「算力固定、数据可少」的前提下得出 `training very large models on a relatively modest amount of data`；2022 年那篇把前提换成「近年的做法是放大模型却 `keeping the amount of training data constant`」，于是得出 `for every doubling of model size the number of training tokens should also be doubled`。同一个 `fixed compute budget`，一个把预算更多投向参数，一个要求参数与数据等比例。把两份摘要的实验范围并排看：`over 400 language models`、`5 to 500 billion tokens` 是 2022 年那篇给出的覆盖；2020 年那篇则强调趋势跨 `seven orders of magnitude`。本书的处理方式是不替来源调和、不挑一个当正确答案，也不把 B 级报告里的前沿型号参数拿来当裁判——报告自己就写了「具体模型参数、上下文长度与价格请以官方 model card 为准」。

### 从预训练到后训练的那道缺口

`Language Models are Few-Shot Learners` 的摘要已经埋了伏笔：`GPT-3` 在 `few-shot setting` 里能做很多任务，但作者同时 `identify some datasets where GPT-3's few-shot learning still struggles`，也指出在大规模网页语料上训练带来的 `methodological issues`。更根本的一句来自两年后的对齐论文开头：`Making language models bigger does not inherently make them better at following a user's intent`，模型可能 `generate outputs that are untruthful, toxic, or simply not helpful to the user`，也就是 `these models are not aligned with their users`。预训练给的是续写能力，不是听懂意图的能力；这道缺口不是靠再加数据或再加参数补的，第 6 章整章都在讲它怎么被补上。综合报告口径把「对齐税」列为后训练的现实动因，同样只作二手背景。

**今天怎么做**：看到任何「某规模效果更好」的说法时，把三样东西抄下来——算力预算、参数量、训练 token 数，缺一样就先别用它做选型。**延伸，非语料原文**：再用这三栏去回看自己收藏过的模型介绍，标出哪些只给了参数量；这个核对动作是编排者给的练习，不是任何来源的表述。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `power-law` 是「参数越多越好」的同义反复——摘要里同时有 `dataset size`、`the amount of compute used for training` 与 `overfitting`；以为 2022 年那篇推翻了 2020 年那篇——本书照录不调和，也不把方向相反改写成顺叙修正；以为 `70B parameters` 打赢 `280B` 说明小模型更省——摘要强调的是 same compute budget 下的重新分配，且有 `4× more more data`；以为 `MMLU` 上的 `67.5%` 可以直接当产品验收分——第 12 章会用 `lopsided performance` 与 `frequently do not know when they are wrong` 说明它测的是什么；还有把 `对齐税` 的 `10–20%` 当一手结论，那是综合报告口径里的二手转述。

**记住这一句**：预训练把「下一个是谁」变成可微目标，让规模成了主变量；两版缩放口径的分歧在算力怎么分，本书按年份照录，而预训练给出的是续写能力，不是听懂意图。

</div>
