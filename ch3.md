# 第3章 分词：文本怎么变成数字

**一句话**：`tokenization` 是在词表大小与序列长度之间做交换，`word-based` 太长、`Character-based` 太碎，主流答案是 `subword tokenization`——高频词整留、低频词拆成有意义片段，最后再用 `Byte-level BPE` 把未登录字符这个洞彻底堵上。

**为什么需要它**：`Hugging Face` 的 `Tokenizers` 章开门见山：`Models can only process numbers`，分词器 `translate text into data that can be processed by the model`；目标是找到最有意义、同时尽可能小的表示。这条约束解释了一件常被忽略的事：你付的 token 费、占的上下文窗口、以及 `KV cache` 的显存，全都在这一段被决定。同一句中文，不同模型的 `input_ids` 长度可能差出一截；同一个生僻词被拆成三段，注意力就要多看三个位置。第 7、8 章所有关于「长」的成本，源头都在这里。

**核心结构**：官方章按三条路线排。第一条 `word-based`，用空格等规则切词，好处是简单直观，坏处是英语有 `500,000 words`，要全覆盖就得维护同样多的 id，且 `dog` 与 `dogs` 在模型眼里毫无关系；词表外的词只能落成 `[UNK]` 或 `"<unk>"`，官方说分词器大量吐出这种 token 是个坏信号，因为你正在丢信息。第二条 `Character-based tokenizers`，词表小、几乎没有 out-of-vocabulary，但一个词轻松变成 `10 or more tokens`，且单个字符语义稀薄；官方特别补了一句，中文里每个字携带的信息比拉丁字母多，所以这个代价因语言而异。第三条 `subword tokenization`，原则原句是 `Subword tokenization algorithms rely on the principle that frequently used words should not be split into smaller subwords`，例子是 `annoyingly` 拆成 `annoying` 与 `ly`，从而实现 `relatively good coverage with small vocabularies`。

**关键设计**：**三种子词算法只在「怎么选合并对」上分家，而字节级 BPE 才真正解决了覆盖问题**。`Hugging Face` 的 `tokenizer_summary` 文档写明 `Transformers support three subword tokenization algorithms: Byte pair encoding (BPE), Unigram, and WordPiece`。`BPE` 从 `base vocabulary` 的字符出发，`iteratively merges the most frequent adjacent pair`，`merge rules` 一直学到 `target vocabulary size`；文档给的实例是 GPT 的 `40,478`、`50,000 merges`。`WordPiece` 则 `merges pairs that maximize the likelihood of the training data`，用 `score("u", "g") = frequency("ug") / (frequency("u") × frequency("g"))` 衡量一次合并有多「出乎意料」，官方文档的对照很精彩：`BPE simply merges whichever pair appears the most. WordPiece measures how *informative* each merge is.`。`Unigram` 反过来，从一大堆候选出发，`Unigram removes the tokens with the lowest loss increase`、通常砍掉 `bottom 10-20%`，因此 `deterministic and based on merge rules` 属于 BPE，而 Unigram 是 `probabilistic and can sample different tokenizations during training`。最后 `Byte-level BPE` 用 `256 byte values` 作基底，`GPT-2` 的 `50,257` 来自 `256 byte tokens + 50,000 merges + special end-of-text token`，从而 `ensuring every word can be tokenized without the "<unk>" token`。

**拆解**：

### 一步调用背后的三段动作

官方把 `encoding` 定义为 `two-step process: the tokenization, followed by the conversion to input IDs`。第一步由 `tokenize()` 完成，示例 `['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']` 里 `transformer` 被切成 `transform` 与 `##er`，`##er` 这个前缀就是在标记「我不是独立词」。第二步由 `convert_tokens_to_ids()` 查 `from_pretrained()` 一起下载的 vocabulary，得到 `[7993, 170, 11303, 1200, 2443, 1110, 3014]`。反方向是 `decode()`，官方特别说明它不只把 id 换回 token，还会把属于同一个词的片段重新拼起来，这在生成场景里至关重要。一次 `tokenizer("Using a Transformer network is simple")` 的输出同时给出 `input_ids`、`token_type_ids` 与 `attention_mask` 三个字段——后者决定了哪些位置参与注意力，是第 4 章的直接输入。

### 词表大小不是越大越好

`Word-level tokenization` 的代价被文档写成一句因果：`Vocabulary size becomes extremely large`，于是 `The resulting embedding matrix is enormous, increasing memory and compute`；`Character-level tokenization` 则是 `But sequences become much longer`。两边都贵，只是贵在不同地方：前者撑大 `embedding` 矩阵，后者拉长序列，而第 4 章会看到注意力的 `quadratic in sequence length`。所以子词切分不是折中妥协，而是同时压住两个方向的成本。`Byte-level BPE` 更进一步，用固定 `256 byte values` 把基底锁死，把预算全留给 `merge rules`；这也解释了为什么同一份中文文本在不同模型上 token 数不同——合并规则是各自语料训出来的，不是通用常量。

### 中文与多语言为什么需要 SentencePiece

标准 `BPE` 与 `Unigram` 都假设 `whitespace separates words`，而 `SentencePiece` 文档直说这对 `Chinese and Japanese` 不成立。它的解法是 `SentencePiece treats the input text as a raw byte or character stream`，把空格本身用 `"▁"` 收进词表，解码时拼接所有 token 再把 `"▁"` 换回空格。这带来一个工程后果：同一套流程不必为中文单独写预分词规则，也就避免了「上游分词器与模型训练时不一致」这类最难查的线上偏差。官方 `Tokenizers` 章在列举算法时也把 `SentencePiece or Unigram, as used in several multilingual models` 与 `Byte-level BPE, as used in GPT-2`、`WordPiece, as used in BERT` 并列，恰好对应第 2 章那两条路线在输入层的分岔。

### 分词错误长什么样

三类症状值得单独记。第一类是 `[UNK]` 泛滥，官方说这意味着 `you're losing information along the way`；第二类是子词切得不合语义，`annoyingly` 如果没被拆成 `annoying` 与 `ly`，模型就少了一层可复用的组合线索；第三类是中文被按字符或按奇怪片段切开，导致 `tokenize()` 输出的序列长度与预期不符。它们在线上都不表现为「报错」，而表现为「同样的话这次答得差一点」「同样一批请求这次贵了一截」。所以第 12 章讲工程口径指标时，`每千 token 成本` 与 `p95` 的基线必须固定同一个 `AutoTokenizer` 版本，否则对比没有意义。

**今天怎么做**：把你线上最高频的十条提示词分别喂给 `AutoTokenizer`，打印 `input_ids` 的长度与 `tokenize()` 的分片，找出被拆得最碎的那一条。**延伸，非语料原文**：再把它改写一版、观察长度变化，把结果记成一页对照；这个练习是编排者给的建议，不是任何来源的表述。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 tokenizer 只是「预处理」——它决定词表、序列长度与 `embedding` 矩阵，直接连着成本与显存；以为中文一定要先分词——`SentencePiece` 正是为不用空格的语言设计的；以为 `BPE` 与 `WordPiece` 差不多——一个 `iteratively merges the most frequent adjacent pair`，一个 `maximize the likelihood of the training data`；以为 `Byte-level BPE` 还会有 `"<unk>"`——官方文档写的是 `ensuring every word can be tokenized without the "<unk>" token`；还有把 `decode()` 当成简单的 id 反查，忽略官方明说它 `groups together the tokens that were part of the same words`。

**记住这一句**：分词是在词表大小与序列长度之间做交换，高频词整留、低频词拆开；`Byte-level BPE` 用 `256 byte values` 堵住未登录字符，而这个长度会一路传导到 `KV cache` 的显存里。

</div>
