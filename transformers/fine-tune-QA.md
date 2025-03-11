# Hugging Face Transformers 微调语言模型-问答任务

我们已经学会使用 Pipeline 加载支持问答任务的预训练模型，本教程代码将展示如何微调训练一个支持问答任务的模型。

**注意：微调后的模型仍然是通过提取上下文的子串来回答问题的，而不是生成新的文本。**

### 模型执行问答效果示例

![Widget inference representing the QA task](docs/images/question_answering.png)


```python
# 根据你使用的模型和GPU资源情况，调整以下关键参数
squad_v2 = True
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
```

## 下载数据集

在本教程中，我们将使用[斯坦福问答数据集(SQuAD）](https://rajpurkar.github.io/SQuAD-explorer/)。

### SQuAD 数据集

**斯坦福问答数据集(SQuAD)** 是一个阅读理解数据集，由众包工作者在一系列维基百科文章上提出问题组成。每个问题的答案都是相应阅读段落中的文本片段或范围，或者该问题可能无法回答。

SQuAD2.0将SQuAD1.1中的10万个问题与由众包工作者对抗性地撰写的5万多个无法回答的问题相结合，使其看起来与可回答的问题类似。要在SQuAD2.0上表现良好，系统不仅必须在可能时回答问题，还必须确定段落中没有支持任何答案，并放弃回答。


```python
from datasets import load_dataset
```


```python
datasets = load_dataset("squad_v2" if squad_v2 else "squad")
```

The `datasets` object itself is [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which contains one key for the training, validation and test set.


```python
datasets
```




    DatasetDict({
        train: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 130319
        })
        validation: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 11873
        })
    })



#### 对比数据集

相比快速入门使用的 Yelp 评论数据集，我们可以看到 SQuAD 训练和测试集都新增了用于上下文、问题以及问题答案的列：

**YelpReviewFull Dataset：**

```json

DatasetDict({
    train: Dataset({
        features: ['label', 'text'],
        num_rows: 650000
    })
    test: Dataset({
        features: ['label', 'text'],
        num_rows: 50000
    })
})
```


```python
datasets["train"][0]
```




    {'id': '5733be284776f41900661182',
     'title': 'University_of_Notre_Dame',
     'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
     'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
     'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}




```python
datasets["train"][333]
```




    {'id': '56d4d0c32ccc5a1400d83250',
     'title': 'Beyoncé',
     'context': 'Beyoncé is believed to have first started a relationship with Jay Z after a collaboration on "\'03 Bonnie & Clyde", which appeared on his seventh album The Blueprint 2: The Gift & The Curse (2002). Beyoncé appeared as Jay Z\'s girlfriend in the music video for the song, which would further fuel speculation of their relationship. On April 4, 2008, Beyoncé and Jay Z were married without publicity. As of April 2014, the couple have sold a combined 300 million records together. The couple are known for their private relationship, although they have appeared to become more relaxed in recent years. Beyoncé suffered a miscarriage in 2010 or 2011, describing it as "the saddest thing" she had ever endured. She returned to the studio and wrote music in order to cope with the loss. In April 2011, Beyoncé and Jay Z traveled to Paris in order to shoot the album cover for her 4, and unexpectedly became pregnant in Paris.',
     'question': 'How many records combined have Beyoncé and Jay Z sold?',
     'answers': {'text': ['300 million'], 'answer_start': [447]}}



#### 从上下文中组织回复内容

我们可以看到答案是通过它们在文本中的起始位置（这里是第515个字符）以及它们的完整文本表示的，这是上面提到的上下文的子字符串。


```python
from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    display(HTML(df.to_html()))
```


```python
show_random_elements(datasets["train"])
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>context</th>
      <th>question</th>
      <th>answers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>57267a20708984140094c764</td>
      <td>Department_store</td>
      <td>All major cities have their distinctive local department stores, which anchored the downtown shopping district until the arrival of the malls in the 1960s. Washington, for example, after 1887 had Woodward &amp; Lothrop and Garfinckel's starting in 1905. Garfield's went bankrupt in 1990, as did Woodward &amp; Lothrop in 1994. Baltimore had four major department stores: Hutzler's was the prestige leader, followed by Hecht's, Hochschild's and Stewart's. They all operated branches in the suburbs, but all closed in the late twentieth century. By 2015, most locally owned department stores around the country had been consolidated into larger chains, or had closed down entirely.</td>
      <td>In what year did Garfield's go bankrupt?</td>
      <td>{'text': ['1990'], 'answer_start': [278]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5ad122d9645df0001a2d0eca</td>
      <td>Labour_Party_(UK)</td>
      <td>Support for the LRC was boosted by the 1901 Taff Vale Case, a dispute between strikers and a railway company that ended with the union being ordered to pay £23,000 damages for a strike. The judgement effectively made strikes illegal since employers could recoup the cost of lost business from the unions. The apparent acquiescence of the Conservative Government of Arthur Balfour to industrial and business interests (traditionally the allies of the Liberal Party in opposition to the Conservative's landed interests) intensified support for the LRC against a government that appeared to have little concern for the industrial proletariat and its problems.</td>
      <td>What hurt support for the LRC?</td>
      <td>{'text': [], 'answer_start': []}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5ace3e6532bba1001ae4a021</td>
      <td>Avicenna</td>
      <td>According to his autobiography, Avicenna had memorised the entire Quran by the age of 10. He learned Indian arithmetic from an Indian greengrocer,ءMahmoud Massahi and he began to learn more from a wandering scholar who gained a livelihood by curing the sick and teaching the young. He also studied Fiqh (Islamic jurisprudence) under the Sunni Hanafi scholar Ismail al-Zahid. Avicenna was taught some extent of philosophy books such as Introduction (Isagoge)'s Porphyry (philosopher), Euclid's Elements, Ptolemy's Almagest by an unpopular philosopher, Abu Abdullah Nateli, who claimed philosophizing.</td>
      <td>When did Avicenna start studying the Quran?</td>
      <td>{'text': [], 'answer_start': []}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5725ce2e271a42140099d20d</td>
      <td>Hellenistic_period</td>
      <td>The Odrysian Kingdom was a union of Thracian tribes under the kings of the powerful Odrysian tribe centered around the region of Thrace. Various parts of Thrace were under Macedonian rule under Philip II of Macedon, Alexander the Great, Lysimachus, Ptolemy II, and Philip V but were also often ruled by their own kings. The Thracians and Agrianes were widely used by Alexander as peltasts and light cavalry, forming about one fifth of his army. The Diadochi also used Thracian mercenaries in their armies and they were also used as colonists. The Odrysians used Greek as the language of administration and of the nobility. The nobility also adopted Greek fashions in dress, ornament and military equipment, spreading it to the other tribes. Thracian kings were among the first to be Hellenized.</td>
      <td>Which kings wre among the first to be Hellenized?</td>
      <td>{'text': ['Thracian'], 'answer_start': [741]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56d97744dc89441400fdb4cb</td>
      <td>2008_Summer_Olympics_torch_relay</td>
      <td>A Macau resident was arrested on April 26 for posting a message on cyberctm.com encouraging people to disrupt the relay. Both orchidbbs.com and cyberctm.com Internet forums were shut down from May 2 to 4. This fueled speculation that the shutdowns were targeting speeches against the relay. The head of the Bureau of Telecommunications Regulation has denied that the shutdowns of the websites were politically motivated. About 2,200 police were deployed on the streets, there were no interruptions.</td>
      <td>In addition to cyberctm.com, what other website was shut down for two days?</td>
      <td>{'text': ['orchidbbs.com'], 'answer_start': [126]}</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5a68bac68476ee001a58a7cc</td>
      <td>Genocide</td>
      <td>In 2007 the European Court of Human Rights (ECHR), noted in its judgement on Jorgic v. Germany case that in 1992 the majority of legal scholars took the narrow view that "intent to destroy" in the CPPCG meant the intended physical-biological destruction of the protected group and that this was still the majority opinion. But the ECHR also noted that a minority took a broader view and did not consider biological-physical destruction was necessary as the intent to destroy a national, racial, religious or ethnic group was enough to qualify as genocide.</td>
      <td>What former case did the European Court of Human Rights draw on in 2002 to further refine qualifiers of genocide?</td>
      <td>{'text': [], 'answer_start': []}</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5a6fd6388abb0b001a675f9b</td>
      <td>Alsace</td>
      <td>This situation prevailed until 1639, when most of Alsace was conquered by France so as to keep it out of the hands of the Spanish Habsburgs, who wanted a clear road to their valuable and rebellious possessions in the Spanish Netherlands. Beset by enemies and seeking to gain a free hand in Hungary, the Habsburgs sold their Sundgau territory (mostly in Upper Alsace) to France in 1646, which had occupied it, for the sum of 1.2 million Thalers. When hostilities were concluded in 1648 with the Treaty of Westphalia, most of Alsace was recognized as part of France, although some towns remained independent. The treaty stipulations regarding Alsace were complex; although the French king gained sovereignty, existing rights and customs of the inhabitants were largely preserved. France continued to maintain its customs border along the Vosges mountains where it had been, leaving Alsace more economically oriented to neighbouring German-speaking lands. The German language remained in use in local administration, in schools, and at the (Lutheran) University of Strasbourg, which continued to draw students from other German-speaking lands. The 1685 Edict of Fontainebleau, by which the French king ordered the suppression of French Protestantism, was not applied in Alsace. France did endeavour to promote Catholicism; Strasbourg Cathedral, for example, which had been Lutheran from 1524 to 1681, was returned to the Catholic Church. However, compared to the rest of France, Alsace enjoyed a climate of religious tolerance.</td>
      <td>When was Strasbourg Cathedral built?</td>
      <td>{'text': [], 'answer_start': []}</td>
    </tr>
    <tr>
      <th>7</th>
      <td>56dfbd5e231d4119001abd5b</td>
      <td>Internet_service_provider</td>
      <td>ISPs provide Internet access, employing a range of technologies to connect users to their network. Available technologies have ranged from computer modems with acoustic couplers to telephone lines, to television cable (CATV), wireless Ethernet (wi-fi), and fiber optics.</td>
      <td>what was an earlier technology used to connect to the internet?</td>
      <td>{'text': ['telephone lines'], 'answer_start': [182]}</td>
    </tr>
    <tr>
      <th>8</th>
      <td>56d2726659d6e41400145ffa</td>
      <td>Buddhism</td>
      <td>During the period of Late Mahayana Buddhism, four major types of thought developed: Madhyamaka, Yogacara, Tathagatagarbha, and Buddhist Logic as the last and most recent. In India, the two main philosophical schools of the Mahayana were the Madhyamaka and the later Yogacara. According to Dan Lusthaus, Madhyamaka and Yogacara have a great deal in common, and the commonality stems from early Buddhism. There were no great Indian teachers associated with tathagatagarbha thought.</td>
      <td>In India the two main philosophical schools of the Mahayana were Madhyamaka and what else?</td>
      <td>{'text': ['Yogacara'], 'answer_start': [96]}</td>
    </tr>
    <tr>
      <th>9</th>
      <td>56f8f65b9b226e1400dd1202</td>
      <td>Near_East</td>
      <td>The use of the term Middle East as a region of international affairs apparently began in British and American diplomatic circles quite independently of each other over concern for the security of the same country: Iran, then known to the west as Persia. In 1900 Thomas Edward Gordon published an article, The Problem of the Middle East, which began:</td>
      <td>Where did the use of the term Middle East as a region of international affairs begin?</td>
      <td>{'text': ['in British and American diplomatic circles'], 'answer_start': [86]}</td>
    </tr>
  </tbody>
</table>


## 预处理数据

**Tokenizer（分词器）就像一位「语言拆解专家」**，专门帮计算机理解人类文字。它的核心作用可以用三步说清楚：

---

### 1️⃣ **拆解文本**  
把句子拆成 **模型认识的片段**（词或子词）。  
例如：  
`"我爱自然语言处理"` → `["我", "爱", "自然", "语言", "处理"]`  
（英文如 `"Hugging Face"` → `["Hug", "##ging", "Face"]`）

---

### 2️⃣ **添加「暗号」**  
插入模型需要的**特殊标记**，比如：  
- **`[CLS]`**：开头标记（BERT用）  
- **`[SEP]`**：分隔标记（区分句子）  
```python
"你好吗？" → ["[CLS]", "你", "好", "吗", "？", "[SEP]"]
```

---

### 3️⃣ **转成密码数字**  
把每个词换成**模型词汇表里的ID号**，类似密码本：  
```python
["[CLS]", "你", "好", "吗"] → [101, 872, 1962, 3221, 102]
```

---

### 🌰 **实际效果示例**  
你输入：`"今天厦门天气如何？"`  
Tokenizer处理后输出：  
```python
{
  "input_ids": [101, 791, 1921, 1762, 1377, 1442, 3221, 102],
  "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1]  # 标记哪些是有效内容
}
```
模型看到这些数字就能分析语义，生成回答啦！

---

### 🤖 **不同模型的差异**  
- **BERT类**：拆词较细，加`[CLS]`/`[SEP]`  
- **GPT类**：按字节拆分，加`<|endoftext|>`  
- **多语言模型**：支持中/英/日等混合拆分  

一句话总结：**Tokenizer就是把人类语言「翻译」成AI能懂的数字密码！** 😊


```python
from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

**AutoTokenizer 就像「万能适配器」**  
——你只需要告诉它用哪个AI模型（比如BERT、GPT-3），它就会自动匹配对应的文字翻译规则。

举个栗子🌰：  
- 你想用 **BERT** 模型 → 它自动加载BERT的分词规则  
  ```python
  tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
  ```
- 你想用 **GPT** 模型 → 它自动切换成GPT的分词方式  
  ```python
  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  ```

**好处**：不用记不同模型的分词器名字（比如`BertTokenizer`、`GPT2Tokenizer`），一个`AutoTokenizer`通吃所有模型，就像万能充电器一样方便！

---

### 对比示例（手动 vs 自动）
| 方式          | 手动选择分词器                   | AutoTokenizer                  |
|---------------|----------------------------------|---------------------------------|
| **BERT模型**  | `from transformers import BertTokenizer`<br>`tokenizer = BertTokenizer.from_pretrained("bert-base")` | `AutoTokenizer.from_pretrained("bert-base")` |
| **GPT模型**   | `from transformers import GPT2Tokenizer`<br>`tokenizer = GPT2Tokenizer.from_pretrained("gpt2")` | `AutoTokenizer.from_pretrained("gpt2")` |

---

⚠️ **注意**：名字要对（比如`bert-base-chinese`不能写成`bert-chinese`），否则这个万能充电器也会找不到插口~

以下断言确保我们的 Tokenizers 使用的是 FastTokenizer（Rust 实现，速度和功能性上有一定优势）。


```python
import transformers
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
# 直接打印判断结果
print("是否是快速版分词器:", isinstance(tokenizer, transformers.PreTrainedTokenizerFast))

# 或更详细的输出
if isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
    print("✅ Tokenizer 是快速版 (PreTrainedTokenizerFast)")
else:
    print("❌ Tokenizer 是普通版 (PreTrainedTokenizer)")

```

    是否是快速版分词器: True
    ✅ Tokenizer 是快速版 (PreTrainedTokenizerFast)


**PreTrainedTokenizer 就像个「文字翻译官」**，专门帮 AI 模型和人类文字打交道。  

举个栗子🌰：  
你想问 AI "厦门今天热吗？"  
➡️ **翻译官的工作**：  
1. 把这句话切成小块：`["厦门", "今天", "热", "吗"]`  
2. 偷偷加暗号：`[开头暗号] 厦门 今天 热 吗 [结尾暗号]`  
3. 转成密码数字：`[101, 2345, 567, 8910, 102]`  

然后 AI 就能看懂这些数字密码，给出回答啦！  
（反过来也会把 AI 的数字密码翻译成人类文字给你看）

`assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)`

这个步骤是**可选的安全检查**，主要为了确保你加载的是**快速版分词器（PreTrainedTokenizerFast）**，而不是旧版的慢速分词器（PreTrainedTokenizer）。不检查也能运行，但可能会遇到以下问题：

---

### 🤔 **为什么要区分 Fast 和普通版？**
| 特性                | PreTrainedTokenizerFast（快速版）          | PreTrainedTokenizer（普通版）       |
|---------------------|--------------------------------------------|-------------------------------------|
| **底层实现**         | Rust语言编写（速度快）                     | Python实现（速度慢）                 |
| **批处理支持**       | ✅ 原生支持（如`batch_encode_plus`）        | ❌ 需手动循环处理                    |
| **特殊标记处理**     | 自动管理（如填充、截断）                   | 需手动配置                          |
| **典型场景**         | 生产环境、大数据处理                        | 教学或兼容旧代码                     |

---

### 💥 **不检查可能带来的问题**
1. **性能下降**：处理1000条文本时，快速版可能比普通版快**5-10倍**。
2. **功能缺失**：普通版可能缺少某些API（如`decode`的`skip_special_tokens`参数）。
3. **意外错误**：某些库（如Datasets）默认要求快速版分词器。

---

### 🌰 **实际案例**
假设你的`model_checkpoint`意外指向了一个没有快速版的模型：
```python
model_checkpoint = "some-old-model"  # 假设该模型只有普通版分词器
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# 此时 tokenizer 是 PreTrainedTokenizer 而非 Fast 版
# 后续调用 batch_encode_plus 可能报错！
```

通过`assert`检查，可以**提前发现问题**，避免后续代码崩溃。

---

### 🔧 **替代方案（如果不做断言）**
1. **直接使用**：如果确定模型有快速版，可以跳过检查。
2. **降级处理**：捕获异常并改用普通版逻辑：
```python
if not isinstance(tokenizer, PreTrainedTokenizerFast):
    print("警告：使用慢速分词器，性能可能受影响！")
    # 手动处理普通版的限制
```

---

总结：这个断言是**防御性编程**的体现，确保代码在性能和功能上按预期运行。对于关键项目建议保留，个人实验可跳过。

您可以在大模型表上查看哪种类型的模型具有可用的快速标记器，哪种类型没有。

您可以直接在两个句子上调用此标记器（一个用于答案，一个用于上下文）：


```python
tokenizer("What is your name?", "My name is Sylvain.")
```




    {'input_ids': [101, 2054, 2003, 2115, 2171, 1029, 102, 2026, 2171, 2003, 25353, 22144, 2378, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}




```python
 tokenizer("How are you?")
```




    {'input_ids': [101, 2129, 2024, 2017, 1029, 102], 'attention_mask': [1, 1, 1, 1, 1, 1]}



### Tokenizer 进阶操作

在问答预处理中的一个特定问题是如何处理非常长的文档。

在其他任务中，当文档的长度超过模型最大句子长度时，我们通常会截断它们，但在这里，删除上下文的一部分可能会导致我们丢失正在寻找的答案。

为了解决这个问题，我们允许数据集中的一个（长）示例生成多个输入特征，每个特征的长度都小于模型的最大长度（或我们设置的超参数）。


```python
# The maximum length of a feature (question and context)
max_length = 384 
# The authorized overlap between two part of the context when splitting it is needed.
doc_stride = 128 
```

---

### **为何设置 `max_length=384`？**
1. **模型限制**  
   BERT等模型最大支持 **512 tokens**，需为以下内容留空间：  
   - **问题本身**（约20-30 tokens）  
   - **特殊标记**（如 `[CLS]`、`[SEP]`，占3-5 tokens）  
   - **答案位置**（避免被截断）

2. **经验比例**  
   可用上下文长度 ≈ 总长的 **75%**（512×0.75≈384），平衡覆盖率和计算效率。

3. **分块优化**  
   结合 `doc_stride=128`（重叠量），确保答案在至少一个分块中完整出现。

---

### **实际案例**  
- **输入**：问题（20 tokens）+ 上下文（500 tokens）  
- **处理**：  
  1. 分块1：问题 + 上下文0-363  
  2. 分块2：问题 + 上下文236-500（与分块1重叠128 tokens）  
- **结果**：即使答案在360-400区间，也能被分块2覆盖。

---

### **调整建议**
- **短文本任务**：直接设为512  
- **超长文档**：可降低到256（需更多分块）  
- **支持更长模型**：如支持1024，可设为768  

一句话总结：**384是平衡模型限制、答案完整性和计算效率的经验值。**

`doc_stride=128` 的原理与 `max_length=384` 类似，但关注点不同。以下是简洁清晰的解释：

---

### **为何设置 `doc_stride=128`？**
1. **核心目的**  
   **避免答案被切割在分块边界**。通过设置分块间的重叠区域，确保即使答案位于分块边缘，也能被至少一个完整分块覆盖。

2. **经验公式**  
   `doc_stride` ≈ `max_length` 的 **1/3~1/4**（如 `384/3≈128`），平衡：
   - **计算效率**（分块越少越好）
   - **答案覆盖率**（重叠越多越安全）

---

### **分块逻辑示例**
- **参数**：
  - `max_length=384`（总长度）
  - 问题长度 = 20 tokens
  - 可用上下文长度 = `384 - 20 - 3（特殊标记）≈ 361 tokens`
  - `doc_stride=128`
- **分块步长** = `361 - 128 = 233 tokens`

| 分块 | 起始位置 | 结束位置 | 覆盖的上下文范围 |
|------|----------|----------|------------------|
| 1    | 0        | 360      | tokens 0-360     |
| 2    | 233      | 593      | tokens 233-593   |
| 3    | 466      | 826      | tokens 466-826   |

- **假设答案在 tokens 350-370**：
  - 分块1：覆盖到360 → 答案部分截断（350-360保留）
  - 分块2：从233开始 → 完整覆盖答案（350-370）

---

### **关键影响**
| `doc_stride` 值 | 优点               | 缺点                 |
|-----------------|--------------------|----------------------|
| **较小（如64）** | 答案覆盖率↑        | 分块数量↑，计算量↑   |
| **较大（如192）**| 分块数量↓，速度↑   | 漏答风险↑            |

---

### **调整建议**
- **短答案任务**（如实体抽取）：`doc_stride=64~128`
- **长答案任务**（如段落总结）：`doc_stride=128~256`

一句话总结：**`doc_stride=128` 是经验性参数，通过分块重叠平衡效率与答案完整性。**

假设我们有以下参数：
- **`max_length = 10`**（每个片段最多包含10个字符）
- **`doc_stride = 4`**（相邻片段重叠4个字符）

---

### **切割过程**
原始文本：`ABCDEFGHIJKLMN`（假设每个字母代表一个token）

1. **第一个片段**：  
   - 取前10个字符 → `ABCDEFGHIJ`（A到J）
   - 结束位置：第10个字符（J）

2. **第二个片段**：  
   - 起始位置 = 前片段的起始位置 + (`max_length - doc_stride`) = 0 + (10 - 4) = 6  
     （即从第7个字符开始，对应字母 `G`）
   - 实际字符：`GHIJKLMN`（G到N，共8个字符，不足10个则保留）
   - 重叠部分：`GHIJ`（与前一片段的后4个字符重叠）

---

### **图示切割效果**
```
原始文本： A B C D E F G H I J K L M N
片段1：    [A B C D E F G H I J]          → 长度10
片段2：            [G H I J K L M N]      → 起始位置6，重叠4个字符
```

---

### **为什么需要重叠？**
假设答案在 `H I J K` 区域：
- **无重叠**：可能被截断在片段1末尾或片段2开头
- **有重叠**：确保答案完整包含在至少一个片段中

---

### **实际问答中的参数**
当 `max_length=384` 且 `doc_stride=128` 时，逻辑完全一致，只是数值更大。这种滑动窗口切割是处理长文本问答的常用策略！ 😊

#### 超出最大长度的文本数据处理

下面，我们从训练集中找出一个超过最大长度（384）的文本：


```python
for i, example in enumerate(datasets["train"]):
    if len(tokenizer(example["question"], example["context"])["input_ids"]) > 384:
        break
# 挑选出来超过384（最大长度）的数据样例
example = datasets["train"][i]
```


```python
len(tokenizer(example["question"], example["context"])["input_ids"])
```




    437



#### 截断上下文不保留超出部分


```python
len(tokenizer(example["question"],
              example["context"],
              max_length=max_length,
              truncation="only_second")["input_ids"])
```




    384



#### 关于截断的策略

- 直接截断超出部分: truncation=`only_second`
- 仅截断上下文（context），保留问题（question）：`return_overflowing_tokens=True` & 设置`stride`



```python
tokenized_example = tokenizer(
    example["question"],
    example["context"],
    max_length=max_length,
    truncation="only_second",
    return_overflowing_tokens=True,
    stride=doc_stride
)
```

使用此策略截断后，Tokenizer 将返回多个 `input_ids` 列表。


```python
[len(x) for x in tokenized_example["input_ids"]]
```




    [384, 192]



解码两个输入特征，可以看到重叠的部分：


```python
for x in tokenized_example["input_ids"][:2]:
    print(tokenizer.decode(x))
```

    [CLS] beyonce got married in 2008 to whom? [SEP] on april 4, 2008, beyonce married jay z. she publicly revealed their marriage in a video montage at the listening party for her third studio album, i am... sasha fierce, in manhattan's sony club on october 22, 2008. i am... sasha fierce was released on november 18, 2008 in the united states. the album formally introduces beyonce's alter ego sasha fierce, conceived during the making of her 2003 single " crazy in love ", selling 482, 000 copies in its first week, debuting atop the billboard 200, and giving beyonce her third consecutive number - one album in the us. the album featured the number - one song " single ladies ( put a ring on it ) " and the top - five songs " if i were a boy " and " halo ". achieving the accomplishment of becoming her longest - running hot 100 single in her career, " halo "'s success in the us helped beyonce attain more top - ten singles on the list than any other woman during the 2000s. it also included the successful " sweet dreams ", and singles " diva ", " ego ", " broken - hearted girl " and " video phone ". the music video for " single ladies " has been parodied and imitated around the world, spawning the " first major dance craze " of the internet age according to the toronto star. the video has won several awards, including best video at the 2009 mtv europe music awards, the 2009 scottish mobo awards, and the 2009 bet awards. at the 2009 mtv video music awards, the video was nominated for nine awards, ultimately winning three including video of the year. its failure to win the best female video category, which went to american country pop singer taylor swift's " you belong with me ", led to kanye west interrupting the ceremony and beyonce [SEP]
    [CLS] beyonce got married in 2008 to whom? [SEP] single ladies " has been parodied and imitated around the world, spawning the " first major dance craze " of the internet age according to the toronto star. the video has won several awards, including best video at the 2009 mtv europe music awards, the 2009 scottish mobo awards, and the 2009 bet awards. at the 2009 mtv video music awards, the video was nominated for nine awards, ultimately winning three including video of the year. its failure to win the best female video category, which went to american country pop singer taylor swift's " you belong with me ", led to kanye west interrupting the ceremony and beyonce improvising a re - presentation of swift's award during her own acceptance speech. in march 2009, beyonce embarked on the i am... world tour, her second headlining worldwide concert tour, consisting of 108 shows, grossing $ 119. 5 million. [SEP]


#### 使用 offsets_mapping 获取原始的 input_ids

设置 `return_offsets_mapping=True`，将使得截断分割生成的多个 input_ids 列表中的 token，通过映射保留原始文本的 input_ids。

如下所示：第一个标记（[CLS]）的起始和结束字符都是（0, 0），因为它不对应问题/答案的任何部分，然后第二个标记与问题(question)的字符0到3相同.


```python
tokenized_example = tokenizer(
    example["question"],            # 第一个参数：问题文本
    example["context"],             # 第二个参数：上下文文本
    max_length=max_length,          # 最大输入长度（如384）
    truncation="only_second",       # 关键参数1：截断策略
    return_overflowing_tokens=True, # 关键参数2：返回分块结果
    return_offsets_mapping=True,    # 关键参数3：返回字符级位置映射
    return_token_type_ids=True,     # 显式要求返回 token_type_ids
    stride=doc_stride               # 分块滑动步长（如128）
)
print(tokenized_example["offset_mapping"][0][:100])
```

    [(0, 0), (0, 7), (8, 11), (12, 19), (20, 22), (23, 27), (28, 30), (31, 35), (35, 36), (0, 0), (0, 2), (3, 8), (9, 10), (10, 11), (12, 16), (16, 17), (18, 25), (26, 33), (34, 37), (38, 39), (39, 40), (41, 44), (45, 53), (54, 62), (63, 68), (69, 77), (78, 80), (81, 82), (83, 88), (89, 93), (93, 96), (97, 99), (100, 103), (104, 113), (114, 119), (120, 123), (124, 127), (128, 133), (134, 140), (141, 146), (146, 147), (148, 149), (150, 152), (152, 153), (153, 154), (154, 155), (156, 161), (162, 168), (168, 169), (170, 172), (173, 182), (182, 183), (183, 184), (185, 189), (190, 194), (195, 197), (198, 205), (206, 208), (208, 209), (210, 214), (214, 215), (216, 217), (218, 220), (220, 221), (221, 222), (222, 223), (224, 229), (230, 236), (237, 240), (241, 249), (250, 252), (253, 261), (262, 264), (264, 265), (266, 270), (271, 273), (274, 277), (278, 284), (285, 291), (291, 292), (293, 296), (297, 302), (303, 311), (312, 322), (323, 330), (330, 331), (331, 332), (333, 338), (339, 342), (343, 348), (349, 355), (355, 356), (357, 366), (367, 373), (374, 377), (378, 384), (385, 387), (388, 391), (392, 396), (397, 403)]


---

### **参数详解**
#### 1. `truncation="only_second"`
- **作用**：**只截断第二个参数（上下文）**，保持第一个参数（问题）完整
- **场景**：当 `问题+上下文` 总长度超过 `max_length` 时，优先保留问题完整性
- **示例**：
  ```python
  # 输入：问题长度20，上下文长度400 → 总长度420 > 384
  # 处理：截断上下文为 384-20-3（特殊标记）= 361 tokens
  ```

#### 2. `return_overflowing_tokens=True`
- **作用**：**返回分块后的多个输入特征**（当输入过长时自动分割）
- **输出字段**：`overflow_to_sample_mapping`（分块对应原始样本的索引）
- **分块逻辑**：
  - 将长上下文按 `max_length - 问题长度` 切割
  - 相邻分块重叠 `stride` tokens（确保答案不被切割）

#### 3. `return_offsets_mapping=True`
- **作用**：**返回每个 token 在原始文本中的字符位置**（起始和结束索引）
- **输出字段**：`offset_mapping`（列表的列表，每个元素是 `(start, end)` 元组）
- **关键用途**：将模型预测的 token 位置映射回原始文本（如定位答案）

---

### **`offset_mapping` 示例解析**
```python
# 假设打印结果前5个元素：
[(0, 0), (0, 3), (4, 7), (8, 11), (12, 15), ...]

# 对应含义：
# [CLS]  What    is    your   name?  [SEP] ...
# (0,0) (0,3) (4,7) (8,11) (12,15)   ...
```
- **特殊标记**：`[CLS]`、`[SEP]` 等无对应文本 → `(0, 0)`
- **问题部分**：字符索引从问题文本的起始位置计算
- **上下文部分**：字符索引从上下文文本的起始位置计算（需注意问题文本长度）

---

### **参数协同作用**
| 参数组合                        | 实际效果                                                                 |
|---------------------------------|------------------------------------------------------------------------|
| `truncation="only_second"` + `return_overflowing_tokens=True` | 将长上下文切割为多个分块，每个分块包含完整问题和部分上下文 |
| `return_offsets_mapping=True`   | 提供分块中每个 token 在原始文本中的位置，用于答案位置映射               |

---

### **应用场景**
1. **训练阶段**：将答案的字符位置转换为分块内的 token 位置
2. **推理阶段**：将模型预测的 token 位置反向映射到原始上下文
3. **数据验证**：检查分块是否覆盖正确答案的原始位置

---

通过这三个参数，实现了长文本问答任务中 **输入分块处理** 和 **位置精确映射** 的核心需求。

因此，我们可以使用这个映射来找到答案在给定特征中的起始和结束标记的位置。

我们只需区分偏移的哪些部分对应于问题，哪些部分对应于上下文。


```python
first_token_id = tokenized_example["input_ids"][0][1]
offsets = tokenized_example["offset_mapping"][0][1]
print(first_token_id)
print(offsets)
print(tokenizer.convert_ids_to_tokens([first_token_id])[0], example["question"][offsets[0]:offsets[1]])
```

    20773
    (0, 7)
    beyonce Beyonce



```python
second_token_id = tokenized_example["input_ids"][0][2]
offsets = tokenized_example["offset_mapping"][0][2]
print(tokenizer.convert_ids_to_tokens([second_token_id])[0], example["question"][offsets[0]:offsets[1]])
```

    got got



```python
first_token_id = tokenized_example["input_ids"][0][7]
offsets = tokenized_example["offset_mapping"][0][7]
print(first_token_id)
print(offsets)
print(tokenizer.convert_ids_to_tokens([first_token_id])[0], example["question"][offsets[0]:offsets[1]])
```

    3183
    (31, 35)
    whom whom



```python
first_token_id = tokenized_example["input_ids"][0][10]
offsets = tokenized_example["offset_mapping"][0][10]
print(first_token_id)
print(offsets)
print(tokenizer.convert_ids_to_tokens([first_token_id])[0], example["context"][offsets[0]:offsets[1]])
```

    2006
    (0, 2)
    on On



```python
# 遍历每个分块
for chunk_idx in range(len(tokenized_example["input_ids"])):
    print(f"\n=== 分块 {chunk_idx} ===")
    
    # 获取当前分块的数据
    input_ids = tokenized_example["input_ids"][chunk_idx]
    offset_mapping = tokenized_example["offset_mapping"][chunk_idx]
    token_type_ids = tokenized_example["token_type_ids"][chunk_idx]

    # 遍历分块内的每个 token
    for token_idx, (token_id, offset, token_type) in enumerate(zip(input_ids, offset_mapping, token_type_ids)):
        # 根据 token_type 选择来源文本
        if token_type == 0:
            source_text = example["question"]
        else:
            source_text = example["context"]

        # 关键修复点：分解 offset 元组为 start 和 end
        start = offset[0]  # 起始字符位置
        end = offset[1]    # 结束字符位置
        print(start, end)
        original_text = source_text[start:end]
        
        # 转换 token_id 为可读文本
        token_str = tokenizer.convert_ids_to_tokens([token_id])[0]  # 取列表第一个元素
        
        # 打印结果
        print(f"Token {token_idx}: {token_str} → {original_text}")
```

    
    === 分块 0 ===
    0 0
    Token 0: [CLS] → 
    0 7
    Token 1: beyonce → Beyonce
    8 11
    Token 2: got → got
    12 19
    Token 3: married → married
    20 22
    Token 4: in → in
    23 27
    Token 5: 2008 → 2008
    28 30
    Token 6: to → to
    31 35
    Token 7: whom → whom
    35 36
    Token 8: ? → ?
    0 0
    Token 9: [SEP] → 
    0 2
    Token 10: on → On
    3 8
    Token 11: april → April
    9 10
    Token 12: 4 → 4
    10 11
    Token 13: , → ,
    12 16
    Token 14: 2008 → 2008
    16 17
    Token 15: , → ,
    18 25
    Token 16: beyonce → Beyoncé
    26 33
    Token 17: married → married
    34 37
    Token 18: jay → Jay
    38 39
    Token 19: z → Z
    39 40
    Token 20: . → .
    41 44
    Token 21: she → She
    45 53
    Token 22: publicly → publicly
    54 62
    Token 23: revealed → revealed
    63 68
    Token 24: their → their
    69 77
    Token 25: marriage → marriage
    78 80
    Token 26: in → in
    81 82
    Token 27: a → a
    83 88
    Token 28: video → video
    89 93
    Token 29: mont → mont
    93 96
    Token 30: ##age → age
    97 99
    Token 31: at → at
    100 103
    Token 32: the → the
    104 113
    Token 33: listening → listening
    114 119
    Token 34: party → party
    120 123
    Token 35: for → for
    124 127
    Token 36: her → her
    128 133
    Token 37: third → third
    134 140
    Token 38: studio → studio
    141 146
    Token 39: album → album
    146 147
    Token 40: , → ,
    148 149
    Token 41: i → I
    150 152
    Token 42: am → Am
    152 153
    Token 43: . → .
    153 154
    Token 44: . → .
    154 155
    Token 45: . → .
    156 161
    Token 46: sasha → Sasha
    162 168
    Token 47: fierce → Fierce
    168 169
    Token 48: , → ,
    170 172
    Token 49: in → in
    173 182
    Token 50: manhattan → Manhattan
    182 183
    Token 51: ' → '
    183 184
    Token 52: s → s
    185 189
    Token 53: sony → Sony
    190 194
    Token 54: club → Club
    195 197
    Token 55: on → on
    198 205
    Token 56: october → October
    206 208
    Token 57: 22 → 22
    208 209
    Token 58: , → ,
    210 214
    Token 59: 2008 → 2008
    214 215
    Token 60: . → .
    216 217
    Token 61: i → I
    218 220
    Token 62: am → Am
    220 221
    Token 63: . → .
    221 222
    Token 64: . → .
    222 223
    Token 65: . → .
    224 229
    Token 66: sasha → Sasha
    230 236
    Token 67: fierce → Fierce
    237 240
    Token 68: was → was
    241 249
    Token 69: released → released
    250 252
    Token 70: on → on
    253 261
    Token 71: november → November
    262 264
    Token 72: 18 → 18
    264 265
    Token 73: , → ,
    266 270
    Token 74: 2008 → 2008
    271 273
    Token 75: in → in
    274 277
    Token 76: the → the
    278 284
    Token 77: united → United
    285 291
    Token 78: states → States
    291 292
    Token 79: . → .
    293 296
    Token 80: the → The
    297 302
    Token 81: album → album
    303 311
    Token 82: formally → formally
    312 322
    Token 83: introduces → introduces
    323 330
    Token 84: beyonce → Beyoncé
    330 331
    Token 85: ' → '
    331 332
    Token 86: s → s
    333 338
    Token 87: alter → alter
    339 342
    Token 88: ego → ego
    343 348
    Token 89: sasha → Sasha
    349 355
    Token 90: fierce → Fierce
    355 356
    Token 91: , → ,
    357 366
    Token 92: conceived → conceived
    367 373
    Token 93: during → during
    374 377
    Token 94: the → the
    378 384
    Token 95: making → making
    385 387
    Token 96: of → of
    388 391
    Token 97: her → her
    392 396
    Token 98: 2003 → 2003
    397 403
    Token 99: single → single
    404 405
    Token 100: " → "
    405 410
    Token 101: crazy → Crazy
    411 413
    Token 102: in → in
    414 418
    Token 103: love → Love
    418 419
    Token 104: " → "
    419 420
    Token 105: , → ,
    421 428
    Token 106: selling → selling
    429 431
    Token 107: 48 → 48
    431 432
    Token 108: ##2 → 2
    432 433
    Token 109: , → ,
    433 436
    Token 110: 000 → 000
    437 443
    Token 111: copies → copies
    444 446
    Token 112: in → in
    447 450
    Token 113: its → its
    451 456
    Token 114: first → first
    457 461
    Token 115: week → week
    461 462
    Token 116: , → ,
    463 471
    Token 117: debuting → debuting
    472 476
    Token 118: atop → atop
    477 480
    Token 119: the → the
    481 490
    Token 120: billboard → Billboard
    491 494
    Token 121: 200 → 200
    494 495
    Token 122: , → ,
    496 499
    Token 123: and → and
    500 506
    Token 124: giving → giving
    507 514
    Token 125: beyonce → Beyoncé
    515 518
    Token 126: her → her
    519 524
    Token 127: third → third
    525 536
    Token 128: consecutive → consecutive
    537 543
    Token 129: number → number
    543 544
    Token 130: - → -
    544 547
    Token 131: one → one
    548 553
    Token 132: album → album
    554 556
    Token 133: in → in
    557 560
    Token 134: the → the
    561 563
    Token 135: us → US
    563 564
    Token 136: . → .
    565 568
    Token 137: the → The
    569 574
    Token 138: album → album
    575 583
    Token 139: featured → featured
    584 587
    Token 140: the → the
    588 594
    Token 141: number → number
    594 595
    Token 142: - → -
    595 598
    Token 143: one → one
    599 603
    Token 144: song → song
    604 605
    Token 145: " → "
    605 611
    Token 146: single → Single
    612 618
    Token 147: ladies → Ladies
    619 620
    Token 148: ( → (
    620 623
    Token 149: put → Put
    624 625
    Token 150: a → a
    626 630
    Token 151: ring → Ring
    631 633
    Token 152: on → on
    634 636
    Token 153: it → It
    636 637
    Token 154: ) → )
    637 638
    Token 155: " → "
    639 642
    Token 156: and → and
    643 646
    Token 157: the → the
    647 650
    Token 158: top → top
    650 651
    Token 159: - → -
    651 655
    Token 160: five → five
    656 661
    Token 161: songs → songs
    662 663
    Token 162: " → "
    663 665
    Token 163: if → If
    666 667
    Token 164: i → I
    668 672
    Token 165: were → Were
    673 674
    Token 166: a → a
    675 678
    Token 167: boy → Boy
    678 679
    Token 168: " → "
    680 683
    Token 169: and → and
    684 685
    Token 170: " → "
    685 689
    Token 171: halo → Halo
    689 690
    Token 172: " → "
    690 691
    Token 173: . → .
    692 701
    Token 174: achieving → Achieving
    702 705
    Token 175: the → the
    706 720
    Token 176: accomplishment → accomplishment
    721 723
    Token 177: of → of
    724 732
    Token 178: becoming → becoming
    733 736
    Token 179: her → her
    737 744
    Token 180: longest → longest
    744 745
    Token 181: - → -
    745 752
    Token 182: running → running
    753 756
    Token 183: hot → Hot
    757 760
    Token 184: 100 → 100
    761 767
    Token 185: single → single
    768 770
    Token 186: in → in
    771 774
    Token 187: her → her
    775 781
    Token 188: career → career
    781 782
    Token 189: , → ,
    783 784
    Token 190: " → "
    784 788
    Token 191: halo → Halo
    788 789
    Token 192: " → "
    789 790
    Token 193: ' → '
    790 791
    Token 194: s → s
    792 799
    Token 195: success → success
    800 802
    Token 196: in → in
    803 806
    Token 197: the → the
    807 809
    Token 198: us → US
    810 816
    Token 199: helped → helped
    817 824
    Token 200: beyonce → Beyoncé
    825 831
    Token 201: attain → attain
    832 836
    Token 202: more → more
    837 840
    Token 203: top → top
    840 841
    Token 204: - → -
    841 844
    Token 205: ten → ten
    845 852
    Token 206: singles → singles
    853 855
    Token 207: on → on
    856 859
    Token 208: the → the
    860 864
    Token 209: list → list
    865 869
    Token 210: than → than
    870 873
    Token 211: any → any
    874 879
    Token 212: other → other
    880 885
    Token 213: woman → woman
    886 892
    Token 214: during → during
    893 896
    Token 215: the → the
    897 902
    Token 216: 2000s → 2000s
    902 903
    Token 217: . → .
    904 906
    Token 218: it → It
    907 911
    Token 219: also → also
    912 920
    Token 220: included → included
    921 924
    Token 221: the → the
    925 935
    Token 222: successful → successful
    936 937
    Token 223: " → "
    937 942
    Token 224: sweet → Sweet
    943 949
    Token 225: dreams → Dreams
    949 950
    Token 226: " → "
    950 951
    Token 227: , → ,
    952 955
    Token 228: and → and
    956 963
    Token 229: singles → singles
    964 965
    Token 230: " → "
    965 969
    Token 231: diva → Diva
    969 970
    Token 232: " → "
    970 971
    Token 233: , → ,
    972 973
    Token 234: " → "
    973 976
    Token 235: ego → Ego
    976 977
    Token 236: " → "
    977 978
    Token 237: , → ,
    979 980
    Token 238: " → "
    980 986
    Token 239: broken → Broken
    986 987
    Token 240: - → -
    987 994
    Token 241: hearted → Hearted
    995 999
    Token 242: girl → Girl
    999 1000
    Token 243: " → "
    1001 1004
    Token 244: and → and
    1005 1006
    Token 245: " → "
    1006 1011
    Token 246: video → Video
    1012 1017
    Token 247: phone → Phone
    1017 1018
    Token 248: " → "
    1018 1019
    Token 249: . → .
    1020 1023
    Token 250: the → The
    1024 1029
    Token 251: music → music
    1030 1035
    Token 252: video → video
    1036 1039
    Token 253: for → for
    1040 1041
    Token 254: " → "
    1041 1047
    Token 255: single → Single
    1048 1054
    Token 256: ladies → Ladies
    1054 1055
    Token 257: " → "
    1056 1059
    Token 258: has → has
    1060 1064
    Token 259: been → been
    1065 1068
    Token 260: par → par
    1068 1070
    Token 261: ##od → od
    1070 1073
    Token 262: ##ied → ied
    1074 1077
    Token 263: and → and
    1078 1080
    Token 264: im → im
    1080 1086
    Token 265: ##itated → itated
    1087 1093
    Token 266: around → around
    1094 1097
    Token 267: the → the
    1098 1103
    Token 268: world → world
    1103 1104
    Token 269: , → ,
    1105 1113
    Token 270: spawning → spawning
    1114 1117
    Token 271: the → the
    1118 1119
    Token 272: " → "
    1119 1124
    Token 273: first → first
    1125 1130
    Token 274: major → major
    1131 1136
    Token 275: dance → dance
    1137 1139
    Token 276: cr → cr
    1139 1141
    Token 277: ##az → az
    1141 1142
    Token 278: ##e → e
    1142 1143
    Token 279: " → "
    1144 1146
    Token 280: of → of
    1147 1150
    Token 281: the → the
    1151 1159
    Token 282: internet → Internet
    1160 1163
    Token 283: age → age
    1164 1173
    Token 284: according → according
    1174 1176
    Token 285: to → to
    1177 1180
    Token 286: the → the
    1181 1188
    Token 287: toronto → Toronto
    1189 1193
    Token 288: star → Star
    1193 1194
    Token 289: . → .
    1195 1198
    Token 290: the → The
    1199 1204
    Token 291: video → video
    1205 1208
    Token 292: has → has
    1209 1212
    Token 293: won → won
    1213 1220
    Token 294: several → several
    1221 1227
    Token 295: awards → awards
    1227 1228
    Token 296: , → ,
    1229 1238
    Token 297: including → including
    1239 1243
    Token 298: best → Best
    1244 1249
    Token 299: video → Video
    1250 1252
    Token 300: at → at
    1253 1256
    Token 301: the → the
    1257 1261
    Token 302: 2009 → 2009
    1262 1265
    Token 303: mtv → MTV
    1266 1272
    Token 304: europe → Europe
    1273 1278
    Token 305: music → Music
    1279 1285
    Token 306: awards → Awards
    1285 1286
    Token 307: , → ,
    1287 1290
    Token 308: the → the
    1291 1295
    Token 309: 2009 → 2009
    1296 1304
    Token 310: scottish → Scottish
    1305 1308
    Token 311: mob → MOB
    1308 1309
    Token 312: ##o → O
    1310 1316
    Token 313: awards → Awards
    1316 1317
    Token 314: , → ,
    1318 1321
    Token 315: and → and
    1322 1325
    Token 316: the → the
    1326 1330
    Token 317: 2009 → 2009
    1331 1334
    Token 318: bet → BET
    1335 1341
    Token 319: awards → Awards
    1341 1342
    Token 320: . → .
    1343 1345
    Token 321: at → At
    1346 1349
    Token 322: the → the
    1350 1354
    Token 323: 2009 → 2009
    1355 1358
    Token 324: mtv → MTV
    1359 1364
    Token 325: video → Video
    1365 1370
    Token 326: music → Music
    1371 1377
    Token 327: awards → Awards
    1377 1378
    Token 328: , → ,
    1379 1382
    Token 329: the → the
    1383 1388
    Token 330: video → video
    1389 1392
    Token 331: was → was
    1393 1402
    Token 332: nominated → nominated
    1403 1406
    Token 333: for → for
    1407 1411
    Token 334: nine → nine
    1412 1418
    Token 335: awards → awards
    1418 1419
    Token 336: , → ,
    1420 1430
    Token 337: ultimately → ultimately
    1431 1438
    Token 338: winning → winning
    1439 1444
    Token 339: three → three
    1445 1454
    Token 340: including → including
    1455 1460
    Token 341: video → Video
    1461 1463
    Token 342: of → of
    1464 1467
    Token 343: the → the
    1468 1472
    Token 344: year → Year
    1472 1473
    Token 345: . → .
    1474 1477
    Token 346: its → Its
    1478 1485
    Token 347: failure → failure
    1486 1488
    Token 348: to → to
    1489 1492
    Token 349: win → win
    1493 1496
    Token 350: the → the
    1497 1501
    Token 351: best → Best
    1502 1508
    Token 352: female → Female
    1509 1514
    Token 353: video → Video
    1515 1523
    Token 354: category → category
    1523 1524
    Token 355: , → ,
    1525 1530
    Token 356: which → which
    1531 1535
    Token 357: went → went
    1536 1538
    Token 358: to → to
    1539 1547
    Token 359: american → American
    1548 1555
    Token 360: country → country
    1556 1559
    Token 361: pop → pop
    1560 1566
    Token 362: singer → singer
    1567 1573
    Token 363: taylor → Taylor
    1574 1579
    Token 364: swift → Swift
    1579 1580
    Token 365: ' → '
    1580 1581
    Token 366: s → s
    1582 1583
    Token 367: " → "
    1583 1586
    Token 368: you → You
    1587 1593
    Token 369: belong → Belong
    1594 1598
    Token 370: with → with
    1599 1601
    Token 371: me → Me
    1601 1602
    Token 372: " → "
    1602 1603
    Token 373: , → ,
    1604 1607
    Token 374: led → led
    1608 1610
    Token 375: to → to
    1611 1616
    Token 376: kanye → Kanye
    1617 1621
    Token 377: west → West
    1622 1634
    Token 378: interrupting → interrupting
    1635 1638
    Token 379: the → the
    1639 1647
    Token 380: ceremony → ceremony
    1648 1651
    Token 381: and → and
    1652 1659
    Token 382: beyonce → Beyoncé
    0 0
    Token 383: [SEP] → 
    
    === 分块 1 ===
    0 0
    Token 0: [CLS] → 
    0 7
    Token 1: beyonce → Beyonce
    8 11
    Token 2: got → got
    12 19
    Token 3: married → married
    20 22
    Token 4: in → in
    23 27
    Token 5: 2008 → 2008
    28 30
    Token 6: to → to
    31 35
    Token 7: whom → whom
    35 36
    Token 8: ? → ?
    0 0
    Token 9: [SEP] → 
    1041 1047
    Token 10: single → Single
    1048 1054
    Token 11: ladies → Ladies
    1054 1055
    Token 12: " → "
    1056 1059
    Token 13: has → has
    1060 1064
    Token 14: been → been
    1065 1068
    Token 15: par → par
    1068 1070
    Token 16: ##od → od
    1070 1073
    Token 17: ##ied → ied
    1074 1077
    Token 18: and → and
    1078 1080
    Token 19: im → im
    1080 1086
    Token 20: ##itated → itated
    1087 1093
    Token 21: around → around
    1094 1097
    Token 22: the → the
    1098 1103
    Token 23: world → world
    1103 1104
    Token 24: , → ,
    1105 1113
    Token 25: spawning → spawning
    1114 1117
    Token 26: the → the
    1118 1119
    Token 27: " → "
    1119 1124
    Token 28: first → first
    1125 1130
    Token 29: major → major
    1131 1136
    Token 30: dance → dance
    1137 1139
    Token 31: cr → cr
    1139 1141
    Token 32: ##az → az
    1141 1142
    Token 33: ##e → e
    1142 1143
    Token 34: " → "
    1144 1146
    Token 35: of → of
    1147 1150
    Token 36: the → the
    1151 1159
    Token 37: internet → Internet
    1160 1163
    Token 38: age → age
    1164 1173
    Token 39: according → according
    1174 1176
    Token 40: to → to
    1177 1180
    Token 41: the → the
    1181 1188
    Token 42: toronto → Toronto
    1189 1193
    Token 43: star → Star
    1193 1194
    Token 44: . → .
    1195 1198
    Token 45: the → The
    1199 1204
    Token 46: video → video
    1205 1208
    Token 47: has → has
    1209 1212
    Token 48: won → won
    1213 1220
    Token 49: several → several
    1221 1227
    Token 50: awards → awards
    1227 1228
    Token 51: , → ,
    1229 1238
    Token 52: including → including
    1239 1243
    Token 53: best → Best
    1244 1249
    Token 54: video → Video
    1250 1252
    Token 55: at → at
    1253 1256
    Token 56: the → the
    1257 1261
    Token 57: 2009 → 2009
    1262 1265
    Token 58: mtv → MTV
    1266 1272
    Token 59: europe → Europe
    1273 1278
    Token 60: music → Music
    1279 1285
    Token 61: awards → Awards
    1285 1286
    Token 62: , → ,
    1287 1290
    Token 63: the → the
    1291 1295
    Token 64: 2009 → 2009
    1296 1304
    Token 65: scottish → Scottish
    1305 1308
    Token 66: mob → MOB
    1308 1309
    Token 67: ##o → O
    1310 1316
    Token 68: awards → Awards
    1316 1317
    Token 69: , → ,
    1318 1321
    Token 70: and → and
    1322 1325
    Token 71: the → the
    1326 1330
    Token 72: 2009 → 2009
    1331 1334
    Token 73: bet → BET
    1335 1341
    Token 74: awards → Awards
    1341 1342
    Token 75: . → .
    1343 1345
    Token 76: at → At
    1346 1349
    Token 77: the → the
    1350 1354
    Token 78: 2009 → 2009
    1355 1358
    Token 79: mtv → MTV
    1359 1364
    Token 80: video → Video
    1365 1370
    Token 81: music → Music
    1371 1377
    Token 82: awards → Awards
    1377 1378
    Token 83: , → ,
    1379 1382
    Token 84: the → the
    1383 1388
    Token 85: video → video
    1389 1392
    Token 86: was → was
    1393 1402
    Token 87: nominated → nominated
    1403 1406
    Token 88: for → for
    1407 1411
    Token 89: nine → nine
    1412 1418
    Token 90: awards → awards
    1418 1419
    Token 91: , → ,
    1420 1430
    Token 92: ultimately → ultimately
    1431 1438
    Token 93: winning → winning
    1439 1444
    Token 94: three → three
    1445 1454
    Token 95: including → including
    1455 1460
    Token 96: video → Video
    1461 1463
    Token 97: of → of
    1464 1467
    Token 98: the → the
    1468 1472
    Token 99: year → Year
    1472 1473
    Token 100: . → .
    1474 1477
    Token 101: its → Its
    1478 1485
    Token 102: failure → failure
    1486 1488
    Token 103: to → to
    1489 1492
    Token 104: win → win
    1493 1496
    Token 105: the → the
    1497 1501
    Token 106: best → Best
    1502 1508
    Token 107: female → Female
    1509 1514
    Token 108: video → Video
    1515 1523
    Token 109: category → category
    1523 1524
    Token 110: , → ,
    1525 1530
    Token 111: which → which
    1531 1535
    Token 112: went → went
    1536 1538
    Token 113: to → to
    1539 1547
    Token 114: american → American
    1548 1555
    Token 115: country → country
    1556 1559
    Token 116: pop → pop
    1560 1566
    Token 117: singer → singer
    1567 1573
    Token 118: taylor → Taylor
    1574 1579
    Token 119: swift → Swift
    1579 1580
    Token 120: ' → '
    1580 1581
    Token 121: s → s
    1582 1583
    Token 122: " → "
    1583 1586
    Token 123: you → You
    1587 1593
    Token 124: belong → Belong
    1594 1598
    Token 125: with → with
    1599 1601
    Token 126: me → Me
    1601 1602
    Token 127: " → "
    1602 1603
    Token 128: , → ,
    1604 1607
    Token 129: led → led
    1608 1610
    Token 130: to → to
    1611 1616
    Token 131: kanye → Kanye
    1617 1621
    Token 132: west → West
    1622 1634
    Token 133: interrupting → interrupting
    1635 1638
    Token 134: the → the
    1639 1647
    Token 135: ceremony → ceremony
    1648 1651
    Token 136: and → and
    1652 1659
    Token 137: beyonce → Beyoncé
    1660 1663
    Token 138: imp → imp
    1663 1666
    Token 139: ##rov → rov
    1666 1671
    Token 140: ##ising → ising
    1672 1673
    Token 141: a → a
    1674 1676
    Token 142: re → re
    1676 1677
    Token 143: - → -
    1677 1689
    Token 144: presentation → presentation
    1690 1692
    Token 145: of → of
    1693 1698
    Token 146: swift → Swift
    1698 1699
    Token 147: ' → '
    1699 1700
    Token 148: s → s
    1701 1706
    Token 149: award → award
    1707 1713
    Token 150: during → during
    1714 1717
    Token 151: her → her
    1718 1721
    Token 152: own → own
    1722 1732
    Token 153: acceptance → acceptance
    1733 1739
    Token 154: speech → speech
    1739 1740
    Token 155: . → .
    1741 1743
    Token 156: in → In
    1744 1749
    Token 157: march → March
    1750 1754
    Token 158: 2009 → 2009
    1754 1755
    Token 159: , → ,
    1756 1763
    Token 160: beyonce → Beyoncé
    1764 1772
    Token 161: embarked → embarked
    1773 1775
    Token 162: on → on
    1776 1779
    Token 163: the → the
    1780 1781
    Token 164: i → I
    1782 1784
    Token 165: am → Am
    1784 1785
    Token 166: . → .
    1785 1786
    Token 167: . → .
    1786 1787
    Token 168: . → .
    1788 1793
    Token 169: world → World
    1794 1798
    Token 170: tour → Tour
    1798 1799
    Token 171: , → ,
    1800 1803
    Token 172: her → her
    1804 1810
    Token 173: second → second
    1811 1821
    Token 174: headlining → headlining
    1822 1831
    Token 175: worldwide → worldwide
    1832 1839
    Token 176: concert → concert
    1840 1844
    Token 177: tour → tour
    1844 1845
    Token 178: , → ,
    1846 1856
    Token 179: consisting → consisting
    1857 1859
    Token 180: of → of
    1860 1863
    Token 181: 108 → 108
    1864 1869
    Token 182: shows → shows
    1869 1870
    Token 183: , → ,
    1871 1879
    Token 184: grossing → grossing
    1880 1881
    Token 185: $ → $
    1881 1884
    Token 186: 119 → 119
    1884 1885
    Token 187: . → .
    1885 1886
    Token 188: 5 → 5
    1887 1894
    Token 189: million → million
    1894 1895
    Token 190: . → .
    0 0
    Token 191: [SEP] → 


用最简单的比喻解释这段代码：

**1. 分块（切书）**  
- 就像一本厚书拆成几本小册子，每本最多512页（模型一次读不完长文本）

**2. 文字变数字（加密）**  
- 把每个字变成数字密码，比如 "贝"→100，"爷"→101  
- `input_ids` 就是这些密码组成的列表：[100, 101, ...]

**3. 记位置（书签）**  
- `offset_mapping` 记录每个密码在原文的位置，比如 (0,2) 表示前两个字

**4. 区分问题和答案（贴标签）**  
- `token_type_ids=0` 表示文字来自问题（如 "贝爷哪年结婚？"）  
- `token_type_ids=1` 表示文字来自答案（如 "2000年..."）

**5. 找对应文字（解密）**  
- 用密码本把数字转回文字  
- 根据位置标签，从问题或答案文本截取对应文字

**就像这样：**  
密码 `100` → 查密码本 → 是"贝" → 在问题第0-2个位置 → 截取"贝爷"

整个过程让计算机像人类一样：先看问题，再快速翻书找答案位置。


```python
example["question"]
```




    'Beyonce got married in 2008 to whom?'




```python
example["context"]
```




    'On April 4, 2008, Beyoncé married Jay Z. She publicly revealed their marriage in a video montage at the listening party for her third studio album, I Am... Sasha Fierce, in Manhattan\'s Sony Club on October 22, 2008. I Am... Sasha Fierce was released on November 18, 2008 in the United States. The album formally introduces Beyoncé\'s alter ego Sasha Fierce, conceived during the making of her 2003 single "Crazy in Love", selling 482,000 copies in its first week, debuting atop the Billboard 200, and giving Beyoncé her third consecutive number-one album in the US. The album featured the number-one song "Single Ladies (Put a Ring on It)" and the top-five songs "If I Were a Boy" and "Halo". Achieving the accomplishment of becoming her longest-running Hot 100 single in her career, "Halo"\'s success in the US helped Beyoncé attain more top-ten singles on the list than any other woman during the 2000s. It also included the successful "Sweet Dreams", and singles "Diva", "Ego", "Broken-Hearted Girl" and "Video Phone". The music video for "Single Ladies" has been parodied and imitated around the world, spawning the "first major dance craze" of the Internet age according to the Toronto Star. The video has won several awards, including Best Video at the 2009 MTV Europe Music Awards, the 2009 Scottish MOBO Awards, and the 2009 BET Awards. At the 2009 MTV Video Music Awards, the video was nominated for nine awards, ultimately winning three including Video of the Year. Its failure to win the Best Female Video category, which went to American country pop singer Taylor Swift\'s "You Belong with Me", led to Kanye West interrupting the ceremony and Beyoncé improvising a re-presentation of Swift\'s award during her own acceptance speech. In March 2009, Beyoncé embarked on the I Am... World Tour, her second headlining worldwide concert tour, consisting of 108 shows, grossing $119.5 million.'



借助`tokenized_example`的`sequence_ids`方法，我们可以方便的区分token的来源编号：

- 对于特殊标记：返回None，
- 对于正文Token：返回句子编号（从0开始编号）。

综上，现在我们可以很方便的在一个输入特征中找到答案的起始和结束 Token。


```python
sequence_ids = tokenized_example.sequence_ids()
print(sequence_ids)
```

    [None, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]



```python
answers = example["answers"]
start_char = answers["answer_start"][0]
end_char = start_char + len(answers["text"][0])

# 当前span在文本中的起始标记索引。
token_start_index = 0
while sequence_ids[token_start_index] != 1:
    token_start_index += 1

# 当前span在文本中的结束标记索引。
token_end_index = len(tokenized_example["input_ids"][0]) - 1
while sequence_ids[token_end_index] != 1:
    token_end_index -= 1

# 检测答案是否超出span范围（如果超出范围，该特征将以CLS标记索引标记）。
offsets = tokenized_example["offset_mapping"][0]
if (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
    # 将token_start_index和token_end_index移动到答案的两端。
    # 注意：如果答案是最后一个单词，我们可以移到最后一个标记之后（边界情况）。
    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
        token_start_index += 1
    start_position = token_start_index - 1
    while offsets[token_end_index][1] >= end_char:
        token_end_index -= 1
    end_position = token_end_index + 1
    print(start_position, end_position)
else:
    print("答案不在此特征中。")

```

    18 19


打印检查是否准确找到了起始位置：


```python
# 通过查找 offset mapping 位置，解码 context 中的答案 
print(tokenizer.decode(tokenized_example["input_ids"][0][start_position: end_position+1]))
# 直接打印 数据集中的标准答案（answer["text"])
print(answers["text"][0])
```

    jay z
    Jay Z


#### 关于填充的策略

- 对于没有超过最大长度的文本，填充补齐长度。
- 对于需要左侧填充的模型，交换 question 和 context 顺序


```python
pad_on_right = tokenizer.padding_side == "right"
```

### 整合以上所有预处理步骤

让我们将所有内容整合到一个函数中，并将其应用到训练集。

针对不可回答的情况（上下文过长，答案在另一个特征中），我们为开始和结束位置都设置了cls索引。

如果allow_impossible_answers标志为False，我们还可以简单地从训练集中丢弃这些示例。


```python
def prepare_train_features(examples):
    # 一些问题的左侧可能有很多空白字符，这对我们没有用，而且会导致上下文的截断失败
    # （标记化的问题将占用大量空间）。因此，我们删除左侧的空白字符。
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # 使用截断和填充对我们的示例进行标记化，但保留溢出部分，使用步幅（stride）。
    # 当上下文很长时，这会导致一个示例可能提供多个特征，其中每个特征的上下文都与前一个特征的上下文有一些重叠。
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 由于一个示例可能给我们提供多个特征（如果它具有很长的上下文），我们需要一个从特征到其对应示例的映射。这个键就提供了这个映射关系。
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # 偏移映射将为我们提供从令牌到原始上下文中的字符位置的映射。这将帮助我们计算开始位置和结束位置。
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 让我们为这些示例进行标记！
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 我们将使用 CLS 特殊 token 的索引来标记不可能的答案。
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # 获取与该示例对应的序列（以了解上下文和问题是什么）。
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 一个示例可以提供多个跨度，这是包含此文本跨度的示例的索引。
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # 如果没有给出答案，则将cls_index设置为答案。
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # 答案在文本中的开始和结束字符索引。
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # 当前跨度在文本中的开始令牌索引。
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # 当前跨度在文本中的结束令牌索引。
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # 检测答案是否超出跨度（在这种情况下，该特征的标签将使用CLS索引）。
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # 否则，将token_start_index和token_end_index移到答案的两端。
                # 注意：如果答案是最后一个单词（边缘情况），我们可以在最后一个偏移之后继续。
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples
```

#### datasets.map 的进阶使用

使用 `datasets.map` 方法将 `prepare_train_features` 应用于所有训练、验证和测试数据：

- batched: 批量处理数据。
- remove_columns: 因为预处理更改了样本的数量，所以在应用它时需要删除旧列。
- load_from_cache_file：是否使用datasets库的自动缓存

datasets 库针对大规模数据，实现了高效缓存机制，能够自动检测传递给 map 的函数是否已更改（因此需要不使用缓存数据）。如果在调用 map 时设置 `load_from_cache_file=False`，可以强制重新应用预处理。


```python
tokenized_datasets = datasets.map(prepare_train_features,
                                  batched=True,
                                  remove_columns=datasets["train"].column_names)
```


    Map:   0%|          | 0/87599 [00:00<?, ? examples/s]



    Map:   0%|          | 0/10570 [00:00<?, ? examples/s]


## 微调模型

现在我们的数据已经准备好用于训练，我们可以下载预训练模型并进行微调。

由于我们的任务是问答，我们使用 `AutoModelForQuestionAnswering` 类。(对比 Yelp 评论打分使用的是 `AutoModelForSequenceClassification` 类）

警告通知我们正在丢弃一些权重（`vocab_transform` 和 `vocab_layer_norm` 层），并随机初始化其他一些权重（`pre_classifier` 和 `classifier` 层）。在微调模型情况下是绝对正常的，因为我们正在删除用于预训练模型的掩码语言建模任务的头部，并用一个新的头部替换它，对于这个新头部，我们没有预训练的权重，所以库会警告我们在用它进行推理之前应该对这个模型进行微调，而这正是我们要做的事情。


```python
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

    Some weights of DistilBertForQuestionAnswering were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['qa_outputs.weight', 'qa_outputs.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


#### 训练超参数（TrainingArguments）


```python
batch_size=64
model_dir = f"models/{model_checkpoint}-finetuned-squad"

args = TrainingArguments(
    output_dir=model_dir,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
)
```

#### Data Collator（数据整理器）

数据整理器将训练数据整理为批次数据，用于模型训练时的批次处理。本教程使用默认的 `default_data_collator`。



```python
from transformers import default_data_collator

data_collator = default_data_collator
```

### 实例化训练器（Trainer）

为了减少训练时间（需要大量算力支持），我们不在本教程的训练模型过程中计算模型评估指标。

而是训练完成后，再独立进行模型评估。


```python
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

    Detected kernel version 4.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.


#### GPU 使用情况

训练数据与模型配置：

- SQUAD v1.1
- model_checkpoint = "distilbert-base-uncased"
- batch_size = 64

NVIDIA GPU 使用情况：

```shell
Every 1.0s: nvidia-smi                                                   Wed Dec 20 15:39:57 2023

Wed Dec 20 15:39:57 2023
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:00:0D.0 Off |                    0 |
| N/A   67C    P0              67W /  70W |  14617MiB / 15360MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A     16384      C   /root/miniconda3/bin/python               14612MiB |
+---------------------------------------------------------------------------------------+
```


```python
trainer.train()
```



    <div>

      <progress value='4152' max='4152' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [4152/4152 2:23:19, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>1.491100</td>
      <td>1.249441</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.108800</td>
      <td>1.161671</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.975700</td>
      <td>1.158766</td>
    </tr>
  </tbody>
</table><p>


    Checkpoint destination directory models/distilbert-base-uncased-finetuned-squad/checkpoint-500 already exists and is non-empty.Saving will proceed but saved results may be invalid.
    Checkpoint destination directory models/distilbert-base-uncased-finetuned-squad/checkpoint-1000 already exists and is non-empty.Saving will proceed but saved results may be invalid.
    Checkpoint destination directory models/distilbert-base-uncased-finetuned-squad/checkpoint-1500 already exists and is non-empty.Saving will proceed but saved results may be invalid.
    Checkpoint destination directory models/distilbert-base-uncased-finetuned-squad/checkpoint-2000 already exists and is non-empty.Saving will proceed but saved results may be invalid.
    Checkpoint destination directory models/distilbert-base-uncased-finetuned-squad/checkpoint-2500 already exists and is non-empty.Saving will proceed but saved results may be invalid.
    Checkpoint destination directory models/distilbert-base-uncased-finetuned-squad/checkpoint-3000 already exists and is non-empty.Saving will proceed but saved results may be invalid.
    Checkpoint destination directory models/distilbert-base-uncased-finetuned-squad/checkpoint-3500 already exists and is non-empty.Saving will proceed but saved results may be invalid.
    Checkpoint destination directory models/distilbert-base-uncased-finetuned-squad/checkpoint-4000 already exists and is non-empty.Saving will proceed but saved results may be invalid.





    TrainOutput(global_step=4152, training_loss=1.3038662743246854, metrics={'train_runtime': 8602.4737, 'train_samples_per_second': 30.872, 'train_steps_per_second': 0.483, 'total_flos': 2.602335381127373e+16, 'train_loss': 1.3038662743246854, 'epoch': 3.0})



### 训练完成后，第一时间保存模型权重文件。


```python
model_to_save = trainer.save_model(model_dir)
```

## 模型评估

**评估模型输出需要一些额外的处理：将模型的预测映射回上下文的部分。**

模型直接输出的是预测答案的`起始位置`和`结束位置`的**logits**


```python
import torch

for batch in trainer.get_eval_dataloader():
    break
batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
with torch.no_grad():
    output = trainer.model(**batch)
output.keys()
```




    odict_keys(['loss', 'start_logits', 'end_logits'])



模型的输出是一个类似字典的对象，其中包含损失（因为我们提供了标签），以及起始和结束logits。我们不需要损失来进行预测，让我们看一下logits：


```python
output.start_logits.shape, output.end_logits.shape
```




    (torch.Size([64, 384]), torch.Size([64, 384]))




```python
output.start_logits.argmax(dim=-1), output.end_logits.argmax(dim=-1)
```




    (tensor([ 46,  57,  78,  43, 118, 108,  72,  35, 108,  34,  73,  41,  80,  91,
             156,  35,  83,  91,  80,  58,  77,  31,  42,  53,  41,  35,  42,  77,
              11,  44,  27, 133,  66,  40,  87,  44,  43,  41, 127,  26,  28,  33,
              87, 127,  95,  25,  43, 132,  42,  29,  44,  46,  24,  44,  65,  58,
              81,  14,  59,  72,  25,  36,  57,  43], device='cuda:0'),
     tensor([ 47,  58,  81,  44, 118, 109,  75,  37, 109,  36,  76,  42,  83,  94,
             158,  35,  83,  94,  83,  60,  80,  31,  43,  54,  42,  35,  43,  80,
              13,  45,  28, 133,  66,  41,  89,  45,  44,  42, 127,  27,  30,  34,
              32, 127,  97,  26,  44, 132,  43,  30,  45,  47,  25,  45,  65,  59,
              81,  14,  60,  72,  25,  36,  58,  43], device='cuda:0'))



#### 如何从模型输出的位置 logit 组合成答案

我们有每个特征和每个标记的logit。在每个特征中为每个标记预测答案最明显的方法是，将起始logits的最大索引作为起始位置，将结束logits的最大索引作为结束位置。

在许多情况下这种方式效果很好，但是如果此预测给出了不可能的结果该怎么办？比如：起始位置可能大于结束位置，或者指向问题中的文本片段而不是答案。在这种情况下，我们可能希望查看第二好的预测，看它是否给出了一个可能的答案，并选择它。

选择第二好的答案并不像选择最佳答案那么容易：
- 它是起始logits中第二佳索引与结束logits中最佳索引吗？
- 还是起始logits中最佳索引与结束logits中第二佳索引？
- 如果第二好的答案也不可能，那么对于第三好的答案，情况会更加棘手。

为了对答案进行分类，
1. 将使用通过添加起始和结束logits获得的分数
1. 设计一个名为`n_best_size`的超参数，限制不对所有可能的答案进行排序。
1. 我们将选择起始和结束logits中的最佳索引，并收集这些预测的所有答案。
1. 在检查每一个是否有效后，我们将按照其分数对它们进行排序，并保留最佳的答案。

以下是我们如何在批次中的第一个特征上执行此操作的示例：


```python
n_best_size = 20
```


```python
import numpy as np

start_logits = output.start_logits[0].cpu().numpy()
end_logits = output.end_logits[0].cpu().numpy()

# 获取最佳的起始和结束位置的索引：
start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

valid_answers = []

# 遍历起始位置和结束位置的索引组合
for start_index in start_indexes:
    for end_index in end_indexes:
        if start_index <= end_index:  # 需要进一步测试以检查答案是否在上下文中
            valid_answers.append(
                {
                    "score": start_logits[start_index] + end_logits[end_index],
                    "text": ""  # 我们需要找到一种方法来获取与上下文中答案对应的原始子字符串
                }
            )

```


然后，我们可以根据它们的得分对`valid_answers`进行排序，并仅保留最佳答案。唯一剩下的问题是如何检查给定的跨度是否在上下文中（而不是问题中），以及如何获取其中的文本。为此，我们需要向我们的验证特征添加两个内容：

- 生成该特征的示例的ID（因为每个示例可以生成多个特征，如前所示）；
- 偏移映射，它将为我们提供从标记索引到上下文中字符位置的映射。

这就是为什么我们将使用以下函数稍微不同于`prepare_train_features`来重新处理验证集：


```python
def prepare_validation_features(examples):
    # 一些问题的左侧有很多空白，这些空白并不有用且会导致上下文截断失败（分词后的问题会占用很多空间）。
    # 因此我们移除这些左侧空白
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # 使用截断和可能的填充对我们的示例进行分词，但使用步长保留溢出的令牌。这导致一个长上下文的示例可能产生
    # 几个特征，每个特征的上下文都会稍微与前一个特征的上下文重叠。
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 由于一个示例在上下文很长时可能会产生几个特征，我们需要一个从特征映射到其对应示例的映射。这个键就是为了这个目的。
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # 我们保留产生这个特征的示例ID，并且会存储偏移映射。
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # 获取与该示例对应的序列（以了解哪些是上下文，哪些是问题）。
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # 一个示例可以产生几个文本段，这里是包含该文本段的示例的索引。
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # 将不属于上下文的偏移映射设置为None，以便容易确定一个令牌位置是否属于上下文。
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

```

将`prepare_validation_features`应用到整个验证集：


```python
validation_features = datasets["validation"].map(
    prepare_validation_features,
    batched=True,
    remove_columns=datasets["validation"].column_names
)
```


    Map:   0%|          | 0/10570 [00:00<?, ? examples/s]


Now we can grab the predictions for all features by using the `Trainer.predict` method:


```python
raw_predictions = trainer.predict(validation_features)
```





`Trainer`会隐藏模型不使用的列（在这里是`example_id`和`offset_mapping`，我们需要它们进行后处理），所以我们需要将它们重新设置回来：


```python
validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
```

现在，我们可以改进之前的测试：

由于在偏移映射中，当它对应于问题的一部分时，我们将其设置为None，因此可以轻松检查答案是否完全在上下文中。我们还可以从考虑中排除非常长的答案（可以调整的超参数）。

展开说下具体实现：
- 首先从模型输出中获取起始和结束的逻辑值（logits），这些值表明答案在文本中可能开始和结束的位置。
- 然后，它使用偏移映射（offset_mapping）来找到这些逻辑值在原始文本中的具体位置。
- 接下来，代码遍历可能的开始和结束索引组合，排除那些不在上下文范围内或长度不合适的答案。
- 对于有效的答案，它计算出一个分数（基于开始和结束逻辑值的和），并将答案及其分数存储起来。
- 最后，它根据分数对答案进行排序，并返回得分最高的几个答案。


```python
max_answer_length = 30
```


```python
start_logits = output.start_logits[0].cpu().numpy()
end_logits = output.end_logits[0].cpu().numpy()
offset_mapping = validation_features[0]["offset_mapping"]

# 第一个特征来自第一个示例。对于更一般的情况，我们需要将example_id匹配到一个示例索引
context = datasets["validation"][0]["context"]

# 收集最佳开始/结束逻辑的索引：
start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
valid_answers = []
for start_index in start_indexes:
    for end_index in end_indexes:
        # 不考虑超出范围的答案，原因是索引超出范围或对应于输入ID的部分不在上下文中。
        if (
            start_index >= len(offset_mapping)
            or end_index >= len(offset_mapping)
            or offset_mapping[start_index] is None
            or offset_mapping[end_index] is None
        ):
            continue
        # 不考虑长度小于0或大于max_answer_length的答案。
        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
            continue
        if start_index <= end_index: # 我们需要细化这个测试，以检查答案是否在上下文中
            start_char = offset_mapping[start_index][0]
            end_char = offset_mapping[end_index][1]
            valid_answers.append(
                {
                    "score": start_logits[start_index] + end_logits[end_index],
                    "text": context[start_char: end_char]
                }
            )

valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]
valid_answers

```




    [{'score': 15.986347, 'text': 'Denver Broncos'},
     {'score': 14.585561,
      'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 13.152991, 'text': 'Carolina Panthers'},
     {'score': 12.38233, 'text': 'Broncos'},
     {'score': 10.981544,
      'text': 'Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 10.852013,
      'text': 'American Football Conference (AFC) champion Denver Broncos'},
     {'score': 10.635618,
      'text': 'The American Football Conference (AFC) champion Denver Broncos'},
     {'score': 10.283654, 'text': 'Denver'},
     {'score': 9.451225,
      'text': 'American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 9.234833,
      'text': 'The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 8.7582445,
      'text': 'Denver Broncos defeated the National Football Conference'},
     {'score': 8.187819,
      'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina'},
     {'score': 8.134832, 'text': 'Panthers'},
     {'score': 8.092252,
      'text': 'Denver Broncos defeated the National Football Conference (NFC)'},
     {'score': 7.7162285,
      'text': 'the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 7.595868,
      'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10'},
     {'score': 7.382572,
      'text': 'National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 7.320059,
      'text': 'Denver Broncos defeated the National Football Conference (NFC'},
     {'score': 6.755249, 'text': 'Carolina'},
     {'score': 6.728976, 'text': 'champion Denver Broncos'}]



打印比较模型输出和标准答案（Ground-truth）是否一致:


```python
datasets["validation"][0]["answers"]
```




    {'text': ['Denver Broncos', 'Denver Broncos', 'Denver Broncos'],
     'answer_start': [177, 177, 177]}



**模型最高概率的输出与标准答案一致**

正如上面的代码所示，这在第一个特征上很容易，因为我们知道它来自第一个示例。

对于其他特征，我们需要建立一个示例与其对应特征的映射关系。

此外，由于一个示例可以生成多个特征，我们需要将由给定示例生成的所有特征中的所有答案汇集在一起，然后选择最佳答案。

下面的代码构建了一个示例索引到其对应特征索引的映射关系：


```python
import collections

examples = datasets["validation"]
features = validation_features

example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
features_per_example = collections.defaultdict(list)
for i, feature in enumerate(features):
    features_per_example[example_id_to_index[feature["example_id"]]].append(i)
```

当`squad_v2 = True`时，有一定概率出现不可能的答案（impossible answer)。

上面的代码仅保留在上下文中的答案，我们还需要获取不可能答案的分数（其起始和结束索引对应于CLS标记的索引）。

当一个示例生成多个特征时，我们必须在所有特征中的不可能答案都预测出现不可能答案时（因为一个特征可能之所以能够预测出不可能答案，是因为答案不在它可以访问的上下文部分），这就是为什么一个示例中不可能答案的分数是该示例生成的每个特征中的不可能答案的分数的最小值。


```python
from tqdm.auto import tqdm

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # 构建一个从示例到其对应特征的映射。
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # 我们需要填充的字典。
    predictions = collections.OrderedDict()

    # 日志记录。
    print(f"正在后处理 {len(examples)} 个示例的预测，这些预测分散在 {len(features)} 个特征中。")

    # 遍历所有示例！
    for example_index, example in enumerate(tqdm(examples)):
        # 这些是与当前示例关联的特征的索引。
        feature_indices = features_per_example[example_index]

        min_null_score = None # 仅在squad_v2为True时使用。
        valid_answers = []
        
        context = example["context"]
        # 遍历与当前示例关联的所有特征。
        for feature_index in feature_indices:
            # 我们获取模型对这个特征的预测。
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # 这将允许我们将logits中的某些位置映射到原始上下文中的文本跨度。
            offset_mapping = features[feature_index]["offset_mapping"]

            # 更新最小空预测。
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # 浏览所有的最佳开始和结束logits，为 `n_best_size` 个最佳选择。
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 不考虑超出范围的答案，原因是索引超出范围或对应于输入ID的部分不在上下文中。
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # 不考虑长度小于0或大于max_answer_length的答案。
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # 在极少数情况下我们没有一个非空预测，我们创建一个假预测以避免失败。
            best_answer = {"text": "", "score": 0.0}
        
        # 选择我们的最终答案：最佳答案或空答案（仅适用于squad_v2）
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions

```

在原始结果上应用后处理问答结果：


```python
final_predictions = postprocess_qa_predictions(datasets["validation"], validation_features, raw_predictions.predictions)
```

    正在后处理 10570 个示例的预测，这些预测分散在 10784 个特征中。



      0%|          | 0/10570 [00:00<?, ?it/s]


使用 `datasets.load_metric` 中加载 `SQuAD v2` 的评估指标


```python
from datasets import load_metric

metric = load_metric("squad_v2" if squad_v2 else "squad")
```

    /tmp/ipykernel_20254/2330875496.py:3: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
      metric = load_metric("squad_v2" if squad_v2 else "squad")
    /root/miniconda3/lib/python3.11/site-packages/datasets/load.py:752: FutureWarning: The repository for squad contains custom code which must be executed to correctly load the metric. You can inspect the repository content at https://raw.githubusercontent.com/huggingface/datasets/2.16.1/metrics/squad/squad.py
    You can avoid this message in future by passing the argument `trust_remote_code=True`.
    Passing `trust_remote_code=True` will be mandatory to load this metric from the next major release of `datasets`.
      warnings.warn(



    Downloading builder script:   0%|          | 0.00/1.72k [00:00<?, ?B/s]



    Downloading extra modules:   0%|          | 0.00/1.11k [00:00<?, ?B/s]


接下来，我们可以调用上面定义的函数进行评估。

只需稍微调整一下预测和标签的格式，因为它期望的是一系列字典而不是一个大字典。

在使用`squad_v2`数据集时，我们还需要设置`no_answer_probability`参数（我们在这里将其设置为0.0，因为如果我们选择了答案，我们已经将答案设置为空）。


```python
if squad_v2:
    formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
else:
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
metric.compute(predictions=formatted_predictions, references=references)
```




    {'exact_match': 74.88174077578051, 'f1': 83.6359321422016}




```python

```

### Homework：加载本地保存的模型，进行评估和再训练更高的 F1 Score


```python
trained_model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
```


```python
trained_trainer = Trainer(
    trained_model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

    Detected kernel version 4.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.



```python

```
