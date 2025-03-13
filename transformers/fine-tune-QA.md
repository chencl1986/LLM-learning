# Hugging Face Transformers 微调语言模型-问答任务

我们已经学会使用 Pipeline 加载支持问答任务的预训练模型，本教程代码将展示如何微调训练一个支持问答任务的模型。

**注意：微调后的模型仍然是通过提取上下文的子串来回答问题的，而不是生成新的文本。**

### 模型执行问答效果示例

![Widget inference representing the QA task](docs/images/question_answering.png)


```python
# 根据你使用的模型和GPU资源情况，调整以下关键参数
squad_v2 = False
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


    Downloading readme: 0.00B [00:00, ?B/s]



    Downloading data:   0%|          | 0.00/14.5M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/1.82M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/87599 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/10570 [00:00<?, ? examples/s]


The `datasets` object itself is [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which contains one key for the training, validation and test set.


```python
datasets
```




    DatasetDict({
        train: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 87599
        })
        validation: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 10570
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




    {'id': '56d443ef2ccc5a1400d830db',
     'title': 'Beyoncé',
     'context': 'Beyoncé attended St. Mary\'s Elementary School in Fredericksburg, Texas, where she enrolled in dance classes. Her singing talent was discovered when dance instructor Darlette Johnson began humming a song and she finished it, able to hit the high-pitched notes. Beyoncé\'s interest in music and performing continued after winning a school talent show at age seven, singing John Lennon\'s "Imagine" to beat 15/16-year-olds. In fall of 1990, Beyoncé enrolled in Parker Elementary School, a music magnet school in Houston, where she would perform with the school\'s choir. She also attended the High School for the Performing and Visual Arts and later Alief Elsik High School. Beyoncé was also a member of the choir at St. John\'s United Methodist Church as a soloist for two years.',
     'question': "What city was Beyoncé's elementary school located in?",
     'answers': {'text': ['Fredericksburg'], 'answer_start': [49]}}



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
      <td>5731ab21b9d445190005e44f</td>
      <td>Religion_in_ancient_Rome</td>
      <td>The meaning and origin of many archaic festivals baffled even Rome's intellectual elite, but the more obscure they were, the greater the opportunity for reinvention and reinterpretation — a fact lost neither on Augustus in his program of religious reform, which often cloaked autocratic innovation, nor on his only rival as mythmaker of the era, Ovid. In his Fasti, a long-form poem covering Roman holidays from January to June, Ovid presents a unique look at Roman antiquarian lore, popular customs, and religious practice that is by turns imaginative, entertaining, high-minded, and scurrilous; not a priestly account, despite the speaker's pose as a vates or inspired poet-prophet, but a work of description, imagination and poetic etymology that reflects the broad humor and burlesque spirit of such venerable festivals as the Saturnalia, Consualia, and feast of Anna Perenna on the Ides of March, where Ovid treats the assassination of the newly deified Julius Caesar as utterly incidental to the festivities among the Roman people. But official calendars preserved from different times and places also show a flexibility in omitting or expanding events, indicating that there was no single static and authoritative calendar of required observances. In the later Empire under Christian rule, the new Christian festivals were incorporated into the existing framework of the Roman calendar, alongside at least some of the traditional festivals.</td>
      <td>What poet wrote a long poem describing Roman religious holidays?</td>
      <td>{'text': ['Ovid'], 'answer_start': [346]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56e08b457aa994140058e5e3</td>
      <td>Hydrogen</td>
      <td>Hydrogen forms a vast array of compounds with carbon called the hydrocarbons, and an even vaster array with heteroatoms that, because of their general association with living things, are called organic compounds. The study of their properties is known as organic chemistry and their study in the context of living organisms is known as biochemistry. By some definitions, "organic" compounds are only required to contain carbon. However, most of them also contain hydrogen, and because it is the carbon-hydrogen bond which gives this class of compounds most of its particular chemical characteristics, carbon-hydrogen bonds are required in some definitions of the word "organic" in chemistry. Millions of hydrocarbons are known, and they are usually formed by complicated synthetic pathways, which seldom involve elementary hydrogen.</td>
      <td>What is the form of hydrogen and carbon called?</td>
      <td>{'text': ['hydrocarbons'], 'answer_start': [64]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56cef65baab44d1400b88d36</td>
      <td>Spectre_(2015_film)</td>
      <td>Christopher Orr, writing in The Atlantic, also criticised the film, saying that Spectre "backslides on virtually every [aspect]". Lawrence Toppman of The Charlotte Observer called Craig's performance "Bored, James Bored." Alyssa Rosenberg, writing for The Washington Post, stated that the film turned into "a disappointingly conventional Bond film."</td>
      <td>What adjective did Lawrence Toppman use to describe Craig's portrayal of James Bond?</td>
      <td>{'text': ['Bored'], 'answer_start': [201]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>571a30bb10f8ca1400304f53</td>
      <td>Seattle</td>
      <td>King County Metro provides frequent stop bus service within the city and surrounding county, as well as a South Lake Union Streetcar line between the South Lake Union neighborhood and Westlake Center in downtown. Seattle is one of the few cities in North America whose bus fleet includes electric trolleybuses. Sound Transit currently provides an express bus service within the metropolitan area; two Sounder commuter rail lines between the suburbs and downtown; its Central Link light rail line, which opened in 2009, between downtown and Sea-Tac Airport gives the city its first rapid transit line that has intermediate stops within the city limits. Washington State Ferries, which manages the largest network of ferries in the United States and third largest in the world, connects Seattle to Bainbridge and Vashon Islands in Puget Sound and to Bremerton and Southworth on the Kitsap Peninsula.</td>
      <td>To what two islands does the ferry service connect?</td>
      <td>{'text': ['Bainbridge and Vashon'], 'answer_start': [796]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>570d2cb4fed7b91900d45cb5</td>
      <td>Macintosh</td>
      <td>In 1998, after the return of Steve Jobs, Apple consolidated its multiple consumer-level desktop models into the all-in-one iMac G3, which became a commercial success and revitalized the brand. Since their transition to Intel processors in 2006, the complete lineup is entirely based on said processors and associated systems. Its current lineup comprises three desktops (the all-in-one iMac, entry-level Mac mini, and the Mac Pro tower graphics workstation), and four laptops (the MacBook, MacBook Air, MacBook Pro, and MacBook Pro with Retina display). Its Xserve server was discontinued in 2011 in favor of the Mac Mini and Mac Pro.</td>
      <td>What took the place of Mac's Xserve server?</td>
      <td>{'text': ['Mac Mini and Mac Pro'], 'answer_start': [613]}</td>
    </tr>
    <tr>
      <th>5</th>
      <td>570af6876b8089140040f646</td>
      <td>Videoconferencing</td>
      <td>Technological developments by videoconferencing developers in the 2010s have extended the capabilities of video conferencing systems beyond the boardroom for use with hand-held mobile devices that combine the use of video, audio and on-screen drawing capabilities broadcasting in real-time over secure networks, independent of location. Mobile collaboration systems now allow multiple people in previously unreachable locations, such as workers on an off-shore oil rig, the ability to view and discuss issues with colleagues thousands of miles away. Traditional videoconferencing system manufacturers have begun providing mobile applications as well, such as those that allow for live and still image streaming.</td>
      <td>What is one example of an application that videoconferencing manufacturers have begun to offer?</td>
      <td>{'text': ['still image streaming'], 'answer_start': [689]}</td>
    </tr>
    <tr>
      <th>6</th>
      <td>56e82d0100c9c71400d775eb</td>
      <td>Dialect</td>
      <td>Italy is home to a vast array of native regional minority languages, most of which are Romance-based and have their own local variants. These regional languages are often referred to colloquially or in non-linguistic circles as Italian "dialects," or dialetti (standard Italian for "dialects"). However, the majority of the regional languages in Italy are in fact not actually "dialects" of standard Italian in the strict linguistic sense, as they are not derived from modern standard Italian but instead evolved locally from Vulgar Latin independent of standard Italian, with little to no influence from what is now known as "standard Italian." They are therefore better classified as individual languages rather than "dialects."</td>
      <td>What are Italian dialects termed in the Italian language?</td>
      <td>{'text': ['dialetti'], 'answer_start': [251]}</td>
    </tr>
    <tr>
      <th>7</th>
      <td>56e147e6cd28a01900c6772b</td>
      <td>Universal_Studios</td>
      <td>The Universal Film Manufacturing Company was incorporated in New York on April 30, 1912. Laemmle, who emerged as president in July 1912, was the primary figure in the partnership with Dintenfass, Baumann, Kessel, Powers, Swanson, Horsley, and Brulatour. Eventually all would be bought out by Laemmle. The new Universal studio was a vertically integrated company, with movie production, distribution and exhibition venues all linked in the same corporate entity, the central element of the Studio system era.</td>
      <td>Along with exhibition and distribution, what business did the Universal Film Manufacturing Company engage in?</td>
      <td>{'text': ['movie production'], 'answer_start': [368]}</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5731933a05b4da19006bd2d0</td>
      <td>Steven_Spielberg</td>
      <td>Spielberg's next film, Schindler's List, was based on the true story of Oskar Schindler, a man who risked his life to save 1,100 Jews from the Holocaust. Schindler's List earned Spielberg his first Academy Award for Best Director (it also won Best Picture). With the film a huge success at the box office, Spielberg used the profits to set up the Shoah Foundation, a non-profit organization that archives filmed testimony of Holocaust survivors. In 1997, the American Film Institute listed it among the 10 Greatest American Films ever Made (#9) which moved up to (#8) when the list was remade in 2007.</td>
      <td>Whose life was 'Schindler's List' based on?</td>
      <td>{'text': ['Oskar Schindler'], 'answer_start': [72]}</td>
    </tr>
    <tr>
      <th>9</th>
      <td>56de93f94396321400ee2a36</td>
      <td>Arnold_Schwarzenegger</td>
      <td>In 1985, Schwarzenegger appeared in "Stop the Madness", an anti-drug music video sponsored by the Reagan administration. He first came to wide public notice as a Republican during the 1988 presidential election, accompanying then-Vice President George H.W. Bush at a campaign rally.</td>
      <td>In what presidential election year did Schwarzenegger make a name for himself as a prominent Republican?</td>
      <td>{'text': ['1988'], 'answer_start': [184]}</td>
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

    /root/miniconda3/envs/peft/lib/python3.10/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(


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




    396



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




    [384, 157]



解码两个输入特征，可以看到重叠的部分：


```python
for x in tokenized_example["input_ids"][:2]:
    print(tokenizer.decode(x))
```

    [CLS] how many wins does the notre dame men's basketball team have? [SEP] the men's basketball team has over 1, 600 wins, one of only 12 schools who have reached that mark, and have appeared in 28 ncaa tournaments. former player austin carr holds the record for most points scored in a single game of the tournament with 61. although the team has never won the ncaa tournament, they were named by the helms athletic foundation as national champions twice. the team has orchestrated a number of upsets of number one ranked teams, the most notable of which was ending ucla's record 88 - game winning streak in 1974. the team has beaten an additional eight number - one teams, and those nine wins rank second, to ucla's 10, all - time in wins against the top team. the team plays in newly renovated purcell pavilion ( within the edmund p. joyce center ), which reopened for the beginning of the 2009 – 2010 season. the team is coached by mike brey, who, as of the 2014 – 15 season, his fifteenth at notre dame, has achieved a 332 - 165 record. in 2009 they were invited to the nit, where they advanced to the semifinals but were beaten by penn state who went on and beat baylor in the championship. the 2010 – 11 team concluded its regular season ranked number seven in the country, with a record of 25 – 5, brey's fifth straight 20 - win season, and a second - place finish in the big east. during the 2014 - 15 season, the team went 32 - 6 and won the acc conference tournament, later advancing to the elite 8, where the fighting irish lost on a missed buzzer - beater against then undefeated kentucky. led by nba draft picks jerian grant and pat connaughton, the fighting irish beat the eventual national champion duke blue devils twice during the season. the 32 wins were [SEP]
    [CLS] how many wins does the notre dame men's basketball team have? [SEP] championship. the 2010 – 11 team concluded its regular season ranked number seven in the country, with a record of 25 – 5, brey's fifth straight 20 - win season, and a second - place finish in the big east. during the 2014 - 15 season, the team went 32 - 6 and won the acc conference tournament, later advancing to the elite 8, where the fighting irish lost on a missed buzzer - beater against then undefeated kentucky. led by nba draft picks jerian grant and pat connaughton, the fighting irish beat the eventual national champion duke blue devils twice during the season. the 32 wins were the most by the fighting irish team since 1908 - 09. [SEP]


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

    [(0, 0), (0, 3), (4, 8), (9, 13), (14, 18), (19, 22), (23, 28), (29, 33), (34, 37), (37, 38), (38, 39), (40, 50), (51, 55), (56, 60), (60, 61), (0, 0), (0, 3), (4, 7), (7, 8), (8, 9), (10, 20), (21, 25), (26, 29), (30, 34), (35, 36), (36, 37), (37, 40), (41, 45), (45, 46), (47, 50), (51, 53), (54, 58), (59, 61), (62, 69), (70, 73), (74, 78), (79, 86), (87, 91), (92, 96), (96, 97), (98, 101), (102, 106), (107, 115), (116, 118), (119, 121), (122, 126), (127, 138), (138, 139), (140, 146), (147, 153), (154, 160), (161, 165), (166, 171), (172, 175), (176, 182), (183, 186), (187, 191), (192, 198), (199, 205), (206, 208), (209, 210), (211, 217), (218, 222), (223, 225), (226, 229), (230, 240), (241, 245), (246, 248), (248, 249), (250, 258), (259, 262), (263, 267), (268, 271), (272, 277), (278, 281), (282, 285), (286, 290), (291, 301), (301, 302), (303, 307), (308, 312), (313, 318), (319, 321), (322, 325), (326, 330), (330, 331), (332, 340), (341, 351), (352, 354), (355, 363), (364, 373), (374, 379), (379, 380), (381, 384), (385, 389), (390, 393), (394, 406), (407, 408), (409, 415), (416, 418)]


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

    2129
    (0, 3)
    how How



```python
second_token_id = tokenized_example["input_ids"][0][2]
offsets = tokenized_example["offset_mapping"][0][2]
print(tokenizer.convert_ids_to_tokens([second_token_id])[0], example["question"][offsets[0]:offsets[1]])
```

    many many



```python
first_token_id = tokenized_example["input_ids"][0][7]
offsets = tokenized_example["offset_mapping"][0][7]
print(first_token_id)
print(offsets)
print(tokenizer.convert_ids_to_tokens([first_token_id])[0], example["question"][offsets[0]:offsets[1]])
```

    8214
    (29, 33)
    dame Dame



```python
first_token_id = tokenized_example["input_ids"][0][10]
offsets = tokenized_example["offset_mapping"][0][10]
print(first_token_id)
print(offsets)
print(tokenizer.convert_ids_to_tokens([first_token_id])[0], example["context"][offsets[0]:offsets[1]])
```

    1055
    (38, 39)
    s 0



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
    0 3
    Token 1: how → How
    4 8
    Token 2: many → many
    9 13
    Token 3: wins → wins
    14 18
    Token 4: does → does
    19 22
    Token 5: the → the
    23 28
    Token 6: notre → Notre
    29 33
    Token 7: dame → Dame
    34 37
    Token 8: men → men
    37 38
    Token 9: ' → '
    38 39
    Token 10: s → s
    40 50
    Token 11: basketball → basketball
    51 55
    Token 12: team → team
    56 60
    Token 13: have → have
    60 61
    Token 14: ? → ?
    0 0
    Token 15: [SEP] → 
    0 3
    Token 16: the → The
    4 7
    Token 17: men → men
    7 8
    Token 18: ' → '
    8 9
    Token 19: s → s
    10 20
    Token 20: basketball → basketball
    21 25
    Token 21: team → team
    26 29
    Token 22: has → has
    30 34
    Token 23: over → over
    35 36
    Token 24: 1 → 1
    36 37
    Token 25: , → ,
    37 40
    Token 26: 600 → 600
    41 45
    Token 27: wins → wins
    45 46
    Token 28: , → ,
    47 50
    Token 29: one → one
    51 53
    Token 30: of → of
    54 58
    Token 31: only → only
    59 61
    Token 32: 12 → 12
    62 69
    Token 33: schools → schools
    70 73
    Token 34: who → who
    74 78
    Token 35: have → have
    79 86
    Token 36: reached → reached
    87 91
    Token 37: that → that
    92 96
    Token 38: mark → mark
    96 97
    Token 39: , → ,
    98 101
    Token 40: and → and
    102 106
    Token 41: have → have
    107 115
    Token 42: appeared → appeared
    116 118
    Token 43: in → in
    119 121
    Token 44: 28 → 28
    122 126
    Token 45: ncaa → NCAA
    127 138
    Token 46: tournaments → tournaments
    138 139
    Token 47: . → .
    140 146
    Token 48: former → Former
    147 153
    Token 49: player → player
    154 160
    Token 50: austin → Austin
    161 165
    Token 51: carr → Carr
    166 171
    Token 52: holds → holds
    172 175
    Token 53: the → the
    176 182
    Token 54: record → record
    183 186
    Token 55: for → for
    187 191
    Token 56: most → most
    192 198
    Token 57: points → points
    199 205
    Token 58: scored → scored
    206 208
    Token 59: in → in
    209 210
    Token 60: a → a
    211 217
    Token 61: single → single
    218 222
    Token 62: game → game
    223 225
    Token 63: of → of
    226 229
    Token 64: the → the
    230 240
    Token 65: tournament → tournament
    241 245
    Token 66: with → with
    246 248
    Token 67: 61 → 61
    248 249
    Token 68: . → .
    250 258
    Token 69: although → Although
    259 262
    Token 70: the → the
    263 267
    Token 71: team → team
    268 271
    Token 72: has → has
    272 277
    Token 73: never → never
    278 281
    Token 74: won → won
    282 285
    Token 75: the → the
    286 290
    Token 76: ncaa → NCAA
    291 301
    Token 77: tournament → Tournament
    301 302
    Token 78: , → ,
    303 307
    Token 79: they → they
    308 312
    Token 80: were → were
    313 318
    Token 81: named → named
    319 321
    Token 82: by → by
    322 325
    Token 83: the → the
    326 330
    Token 84: helm → Helm
    330 331
    Token 85: ##s → s
    332 340
    Token 86: athletic → Athletic
    341 351
    Token 87: foundation → Foundation
    352 354
    Token 88: as → as
    355 363
    Token 89: national → national
    364 373
    Token 90: champions → champions
    374 379
    Token 91: twice → twice
    379 380
    Token 92: . → .
    381 384
    Token 93: the → The
    385 389
    Token 94: team → team
    390 393
    Token 95: has → has
    394 406
    Token 96: orchestrated → orchestrated
    407 408
    Token 97: a → a
    409 415
    Token 98: number → number
    416 418
    Token 99: of → of
    419 424
    Token 100: upset → upset
    424 425
    Token 101: ##s → s
    426 428
    Token 102: of → of
    429 435
    Token 103: number → number
    436 439
    Token 104: one → one
    440 446
    Token 105: ranked → ranked
    447 452
    Token 106: teams → teams
    452 453
    Token 107: , → ,
    454 457
    Token 108: the → the
    458 462
    Token 109: most → most
    463 470
    Token 110: notable → notable
    471 473
    Token 111: of → of
    474 479
    Token 112: which → which
    480 483
    Token 113: was → was
    484 490
    Token 114: ending → ending
    491 495
    Token 115: ucla → UCLA
    495 496
    Token 116: ' → '
    496 497
    Token 117: s → s
    498 504
    Token 118: record → record
    505 507
    Token 119: 88 → 88
    507 508
    Token 120: - → -
    508 512
    Token 121: game → game
    513 520
    Token 122: winning → winning
    521 527
    Token 123: streak → streak
    528 530
    Token 124: in → in
    531 535
    Token 125: 1974 → 1974
    535 536
    Token 126: . → .
    537 540
    Token 127: the → The
    541 545
    Token 128: team → team
    546 549
    Token 129: has → has
    550 556
    Token 130: beaten → beaten
    557 559
    Token 131: an → an
    560 570
    Token 132: additional → additional
    571 576
    Token 133: eight → eight
    577 583
    Token 134: number → number
    583 584
    Token 135: - → -
    584 587
    Token 136: one → one
    588 593
    Token 137: teams → teams
    593 594
    Token 138: , → ,
    595 598
    Token 139: and → and
    599 604
    Token 140: those → those
    605 609
    Token 141: nine → nine
    610 614
    Token 142: wins → wins
    615 619
    Token 143: rank → rank
    620 626
    Token 144: second → second
    626 627
    Token 145: , → ,
    628 630
    Token 146: to → to
    631 635
    Token 147: ucla → UCLA
    635 636
    Token 148: ' → '
    636 637
    Token 149: s → s
    638 640
    Token 150: 10 → 10
    640 641
    Token 151: , → ,
    642 645
    Token 152: all → all
    645 646
    Token 153: - → -
    646 650
    Token 154: time → time
    651 653
    Token 155: in → in
    654 658
    Token 156: wins → wins
    659 666
    Token 157: against → against
    667 670
    Token 158: the → the
    671 674
    Token 159: top → top
    675 679
    Token 160: team → team
    679 680
    Token 161: . → .
    681 684
    Token 162: the → The
    685 689
    Token 163: team → team
    690 695
    Token 164: plays → plays
    696 698
    Token 165: in → in
    699 704
    Token 166: newly → newly
    705 714
    Token 167: renovated → renovated
    715 722
    Token 168: purcell → Purcell
    723 731
    Token 169: pavilion → Pavilion
    732 733
    Token 170: ( → (
    733 739
    Token 171: within → within
    740 743
    Token 172: the → the
    744 750
    Token 173: edmund → Edmund
    751 752
    Token 174: p → P
    752 753
    Token 175: . → .
    754 759
    Token 176: joyce → Joyce
    760 766
    Token 177: center → Center
    766 767
    Token 178: ) → )
    767 768
    Token 179: , → ,
    769 774
    Token 180: which → which
    775 783
    Token 181: reopened → reopened
    784 787
    Token 182: for → for
    788 791
    Token 183: the → the
    792 801
    Token 184: beginning → beginning
    802 804
    Token 185: of → of
    805 808
    Token 186: the → the
    809 813
    Token 187: 2009 → 2009
    813 814
    Token 188: – → –
    814 818
    Token 189: 2010 → 2010
    819 825
    Token 190: season → season
    825 826
    Token 191: . → .
    827 830
    Token 192: the → The
    831 835
    Token 193: team → team
    836 838
    Token 194: is → is
    839 846
    Token 195: coached → coached
    847 849
    Token 196: by → by
    850 854
    Token 197: mike → Mike
    855 857
    Token 198: br → Br
    857 859
    Token 199: ##ey → ey
    859 860
    Token 200: , → ,
    861 864
    Token 201: who → who
    864 865
    Token 202: , → ,
    866 868
    Token 203: as → as
    869 871
    Token 204: of → of
    872 875
    Token 205: the → the
    876 880
    Token 206: 2014 → 2014
    880 881
    Token 207: – → –
    881 883
    Token 208: 15 → 15
    884 890
    Token 209: season → season
    890 891
    Token 210: , → ,
    892 895
    Token 211: his → his
    896 905
    Token 212: fifteenth → fifteenth
    906 908
    Token 213: at → at
    909 914
    Token 214: notre → Notre
    915 919
    Token 215: dame → Dame
    919 920
    Token 216: , → ,
    921 924
    Token 217: has → has
    925 933
    Token 218: achieved → achieved
    934 935
    Token 219: a → a
    936 939
    Token 220: 332 → 332
    939 940
    Token 221: - → -
    940 943
    Token 222: 165 → 165
    944 950
    Token 223: record → record
    950 951
    Token 224: . → .
    952 954
    Token 225: in → In
    955 959
    Token 226: 2009 → 2009
    960 964
    Token 227: they → they
    965 969
    Token 228: were → were
    970 977
    Token 229: invited → invited
    978 980
    Token 230: to → to
    981 984
    Token 231: the → the
    985 987
    Token 232: ni → NI
    987 988
    Token 233: ##t → T
    988 989
    Token 234: , → ,
    990 995
    Token 235: where → where
    996 1000
    Token 236: they → they
    1001 1009
    Token 237: advanced → advanced
    1010 1012
    Token 238: to → to
    1013 1016
    Token 239: the → the
    1017 1027
    Token 240: semifinals → semifinals
    1028 1031
    Token 241: but → but
    1032 1036
    Token 242: were → were
    1037 1043
    Token 243: beaten → beaten
    1044 1046
    Token 244: by → by
    1047 1051
    Token 245: penn → Penn
    1052 1057
    Token 246: state → State
    1058 1061
    Token 247: who → who
    1062 1066
    Token 248: went → went
    1067 1069
    Token 249: on → on
    1070 1073
    Token 250: and → and
    1074 1078
    Token 251: beat → beat
    1079 1085
    Token 252: baylor → Baylor
    1086 1088
    Token 253: in → in
    1089 1092
    Token 254: the → the
    1093 1105
    Token 255: championship → championship
    1105 1106
    Token 256: . → .
    1107 1110
    Token 257: the → The
    1111 1115
    Token 258: 2010 → 2010
    1115 1116
    Token 259: – → –
    1116 1118
    Token 260: 11 → 11
    1119 1123
    Token 261: team → team
    1124 1133
    Token 262: concluded → concluded
    1134 1137
    Token 263: its → its
    1138 1145
    Token 264: regular → regular
    1146 1152
    Token 265: season → season
    1153 1159
    Token 266: ranked → ranked
    1160 1166
    Token 267: number → number
    1167 1172
    Token 268: seven → seven
    1173 1175
    Token 269: in → in
    1176 1179
    Token 270: the → the
    1180 1187
    Token 271: country → country
    1187 1188
    Token 272: , → ,
    1189 1193
    Token 273: with → with
    1194 1195
    Token 274: a → a
    1196 1202
    Token 275: record → record
    1203 1205
    Token 276: of → of
    1206 1208
    Token 277: 25 → 25
    1208 1209
    Token 278: – → –
    1209 1210
    Token 279: 5 → 5
    1210 1211
    Token 280: , → ,
    1212 1214
    Token 281: br → Br
    1214 1216
    Token 282: ##ey → ey
    1216 1217
    Token 283: ' → '
    1217 1218
    Token 284: s → s
    1219 1224
    Token 285: fifth → fifth
    1225 1233
    Token 286: straight → straight
    1234 1236
    Token 287: 20 → 20
    1236 1237
    Token 288: - → -
    1237 1240
    Token 289: win → win
    1241 1247
    Token 290: season → season
    1247 1248
    Token 291: , → ,
    1249 1252
    Token 292: and → and
    1253 1254
    Token 293: a → a
    1255 1261
    Token 294: second → second
    1261 1262
    Token 295: - → -
    1262 1267
    Token 296: place → place
    1268 1274
    Token 297: finish → finish
    1275 1277
    Token 298: in → in
    1278 1281
    Token 299: the → the
    1282 1285
    Token 300: big → Big
    1286 1290
    Token 301: east → East
    1290 1291
    Token 302: . → .
    1292 1298
    Token 303: during → During
    1299 1302
    Token 304: the → the
    1303 1307
    Token 305: 2014 → 2014
    1307 1308
    Token 306: - → -
    1308 1310
    Token 307: 15 → 15
    1311 1317
    Token 308: season → season
    1317 1318
    Token 309: , → ,
    1319 1322
    Token 310: the → the
    1323 1327
    Token 311: team → team
    1328 1332
    Token 312: went → went
    1333 1335
    Token 313: 32 → 32
    1335 1336
    Token 314: - → -
    1336 1337
    Token 315: 6 → 6
    1338 1341
    Token 316: and → and
    1342 1345
    Token 317: won → won
    1346 1349
    Token 318: the → the
    1350 1353
    Token 319: acc → ACC
    1354 1364
    Token 320: conference → conference
    1365 1375
    Token 321: tournament → tournament
    1375 1376
    Token 322: , → ,
    1377 1382
    Token 323: later → later
    1383 1392
    Token 324: advancing → advancing
    1393 1395
    Token 325: to → to
    1396 1399
    Token 326: the → the
    1400 1405
    Token 327: elite → Elite
    1406 1407
    Token 328: 8 → 8
    1407 1408
    Token 329: , → ,
    1409 1414
    Token 330: where → where
    1415 1418
    Token 331: the → the
    1419 1427
    Token 332: fighting → Fighting
    1428 1433
    Token 333: irish → Irish
    1434 1438
    Token 334: lost → lost
    1439 1441
    Token 335: on → on
    1442 1443
    Token 336: a → a
    1444 1450
    Token 337: missed → missed
    1451 1455
    Token 338: buzz → buzz
    1455 1457
    Token 339: ##er → er
    1457 1458
    Token 340: - → -
    1458 1462
    Token 341: beat → beat
    1462 1464
    Token 342: ##er → er
    1465 1472
    Token 343: against → against
    1473 1477
    Token 344: then → then
    1478 1488
    Token 345: undefeated → undefeated
    1489 1497
    Token 346: kentucky → Kentucky
    1497 1498
    Token 347: . → .
    1499 1502
    Token 348: led → Led
    1503 1505
    Token 349: by → by
    1506 1509
    Token 350: nba → NBA
    1510 1515
    Token 351: draft → draft
    1516 1521
    Token 352: picks → picks
    1522 1524
    Token 353: je → Je
    1524 1528
    Token 354: ##rian → rian
    1529 1534
    Token 355: grant → Grant
    1535 1538
    Token 356: and → and
    1539 1542
    Token 357: pat → Pat
    1543 1546
    Token 358: con → Con
    1546 1548
    Token 359: ##na → na
    1548 1552
    Token 360: ##ught → ught
    1552 1554
    Token 361: ##on → on
    1554 1555
    Token 362: , → ,
    1556 1559
    Token 363: the → the
    1560 1568
    Token 364: fighting → Fighting
    1569 1574
    Token 365: irish → Irish
    1575 1579
    Token 366: beat → beat
    1580 1583
    Token 367: the → the
    1584 1592
    Token 368: eventual → eventual
    1593 1601
    Token 369: national → national
    1602 1610
    Token 370: champion → champion
    1611 1615
    Token 371: duke → Duke
    1616 1620
    Token 372: blue → Blue
    1621 1627
    Token 373: devils → Devils
    1628 1633
    Token 374: twice → twice
    1634 1640
    Token 375: during → during
    1641 1644
    Token 376: the → the
    1645 1651
    Token 377: season → season
    1651 1652
    Token 378: . → .
    1653 1656
    Token 379: the → The
    1657 1659
    Token 380: 32 → 32
    1660 1664
    Token 381: wins → wins
    1665 1669
    Token 382: were → were
    0 0
    Token 383: [SEP] → 
    
    === 分块 1 ===
    0 0
    Token 0: [CLS] → 
    0 3
    Token 1: how → How
    4 8
    Token 2: many → many
    9 13
    Token 3: wins → wins
    14 18
    Token 4: does → does
    19 22
    Token 5: the → the
    23 28
    Token 6: notre → Notre
    29 33
    Token 7: dame → Dame
    34 37
    Token 8: men → men
    37 38
    Token 9: ' → '
    38 39
    Token 10: s → s
    40 50
    Token 11: basketball → basketball
    51 55
    Token 12: team → team
    56 60
    Token 13: have → have
    60 61
    Token 14: ? → ?
    0 0
    Token 15: [SEP] → 
    1093 1105
    Token 16: championship → championship
    1105 1106
    Token 17: . → .
    1107 1110
    Token 18: the → The
    1111 1115
    Token 19: 2010 → 2010
    1115 1116
    Token 20: – → –
    1116 1118
    Token 21: 11 → 11
    1119 1123
    Token 22: team → team
    1124 1133
    Token 23: concluded → concluded
    1134 1137
    Token 24: its → its
    1138 1145
    Token 25: regular → regular
    1146 1152
    Token 26: season → season
    1153 1159
    Token 27: ranked → ranked
    1160 1166
    Token 28: number → number
    1167 1172
    Token 29: seven → seven
    1173 1175
    Token 30: in → in
    1176 1179
    Token 31: the → the
    1180 1187
    Token 32: country → country
    1187 1188
    Token 33: , → ,
    1189 1193
    Token 34: with → with
    1194 1195
    Token 35: a → a
    1196 1202
    Token 36: record → record
    1203 1205
    Token 37: of → of
    1206 1208
    Token 38: 25 → 25
    1208 1209
    Token 39: – → –
    1209 1210
    Token 40: 5 → 5
    1210 1211
    Token 41: , → ,
    1212 1214
    Token 42: br → Br
    1214 1216
    Token 43: ##ey → ey
    1216 1217
    Token 44: ' → '
    1217 1218
    Token 45: s → s
    1219 1224
    Token 46: fifth → fifth
    1225 1233
    Token 47: straight → straight
    1234 1236
    Token 48: 20 → 20
    1236 1237
    Token 49: - → -
    1237 1240
    Token 50: win → win
    1241 1247
    Token 51: season → season
    1247 1248
    Token 52: , → ,
    1249 1252
    Token 53: and → and
    1253 1254
    Token 54: a → a
    1255 1261
    Token 55: second → second
    1261 1262
    Token 56: - → -
    1262 1267
    Token 57: place → place
    1268 1274
    Token 58: finish → finish
    1275 1277
    Token 59: in → in
    1278 1281
    Token 60: the → the
    1282 1285
    Token 61: big → Big
    1286 1290
    Token 62: east → East
    1290 1291
    Token 63: . → .
    1292 1298
    Token 64: during → During
    1299 1302
    Token 65: the → the
    1303 1307
    Token 66: 2014 → 2014
    1307 1308
    Token 67: - → -
    1308 1310
    Token 68: 15 → 15
    1311 1317
    Token 69: season → season
    1317 1318
    Token 70: , → ,
    1319 1322
    Token 71: the → the
    1323 1327
    Token 72: team → team
    1328 1332
    Token 73: went → went
    1333 1335
    Token 74: 32 → 32
    1335 1336
    Token 75: - → -
    1336 1337
    Token 76: 6 → 6
    1338 1341
    Token 77: and → and
    1342 1345
    Token 78: won → won
    1346 1349
    Token 79: the → the
    1350 1353
    Token 80: acc → ACC
    1354 1364
    Token 81: conference → conference
    1365 1375
    Token 82: tournament → tournament
    1375 1376
    Token 83: , → ,
    1377 1382
    Token 84: later → later
    1383 1392
    Token 85: advancing → advancing
    1393 1395
    Token 86: to → to
    1396 1399
    Token 87: the → the
    1400 1405
    Token 88: elite → Elite
    1406 1407
    Token 89: 8 → 8
    1407 1408
    Token 90: , → ,
    1409 1414
    Token 91: where → where
    1415 1418
    Token 92: the → the
    1419 1427
    Token 93: fighting → Fighting
    1428 1433
    Token 94: irish → Irish
    1434 1438
    Token 95: lost → lost
    1439 1441
    Token 96: on → on
    1442 1443
    Token 97: a → a
    1444 1450
    Token 98: missed → missed
    1451 1455
    Token 99: buzz → buzz
    1455 1457
    Token 100: ##er → er
    1457 1458
    Token 101: - → -
    1458 1462
    Token 102: beat → beat
    1462 1464
    Token 103: ##er → er
    1465 1472
    Token 104: against → against
    1473 1477
    Token 105: then → then
    1478 1488
    Token 106: undefeated → undefeated
    1489 1497
    Token 107: kentucky → Kentucky
    1497 1498
    Token 108: . → .
    1499 1502
    Token 109: led → Led
    1503 1505
    Token 110: by → by
    1506 1509
    Token 111: nba → NBA
    1510 1515
    Token 112: draft → draft
    1516 1521
    Token 113: picks → picks
    1522 1524
    Token 114: je → Je
    1524 1528
    Token 115: ##rian → rian
    1529 1534
    Token 116: grant → Grant
    1535 1538
    Token 117: and → and
    1539 1542
    Token 118: pat → Pat
    1543 1546
    Token 119: con → Con
    1546 1548
    Token 120: ##na → na
    1548 1552
    Token 121: ##ught → ught
    1552 1554
    Token 122: ##on → on
    1554 1555
    Token 123: , → ,
    1556 1559
    Token 124: the → the
    1560 1568
    Token 125: fighting → Fighting
    1569 1574
    Token 126: irish → Irish
    1575 1579
    Token 127: beat → beat
    1580 1583
    Token 128: the → the
    1584 1592
    Token 129: eventual → eventual
    1593 1601
    Token 130: national → national
    1602 1610
    Token 131: champion → champion
    1611 1615
    Token 132: duke → Duke
    1616 1620
    Token 133: blue → Blue
    1621 1627
    Token 134: devils → Devils
    1628 1633
    Token 135: twice → twice
    1634 1640
    Token 136: during → during
    1641 1644
    Token 137: the → the
    1645 1651
    Token 138: season → season
    1651 1652
    Token 139: . → .
    1653 1656
    Token 140: the → The
    1657 1659
    Token 141: 32 → 32
    1660 1664
    Token 142: wins → wins
    1665 1669
    Token 143: were → were
    1670 1673
    Token 144: the → the
    1674 1678
    Token 145: most → most
    1679 1681
    Token 146: by → by
    1682 1685
    Token 147: the → the
    1686 1694
    Token 148: fighting → Fighting
    1695 1700
    Token 149: irish → Irish
    1701 1705
    Token 150: team → team
    1706 1711
    Token 151: since → since
    1712 1716
    Token 152: 1908 → 1908
    1716 1717
    Token 153: - → -
    1717 1719
    Token 154: 09 → 09
    1719 1720
    Token 155: . → .
    0 0
    Token 156: [SEP] → 


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




    "How many wins does the Notre Dame men's basketball team have?"




```python
example["context"]
```




    "The men's basketball team has over 1,600 wins, one of only 12 schools who have reached that mark, and have appeared in 28 NCAA tournaments. Former player Austin Carr holds the record for most points scored in a single game of the tournament with 61. Although the team has never won the NCAA Tournament, they were named by the Helms Athletic Foundation as national champions twice. The team has orchestrated a number of upsets of number one ranked teams, the most notable of which was ending UCLA's record 88-game winning streak in 1974. The team has beaten an additional eight number-one teams, and those nine wins rank second, to UCLA's 10, all-time in wins against the top team. The team plays in newly renovated Purcell Pavilion (within the Edmund P. Joyce Center), which reopened for the beginning of the 2009–2010 season. The team is coached by Mike Brey, who, as of the 2014–15 season, his fifteenth at Notre Dame, has achieved a 332-165 record. In 2009 they were invited to the NIT, where they advanced to the semifinals but were beaten by Penn State who went on and beat Baylor in the championship. The 2010–11 team concluded its regular season ranked number seven in the country, with a record of 25–5, Brey's fifth straight 20-win season, and a second-place finish in the Big East. During the 2014-15 season, the team went 32-6 and won the ACC conference tournament, later advancing to the Elite 8, where the Fighting Irish lost on a missed buzzer-beater against then undefeated Kentucky. Led by NBA draft picks Jerian Grant and Pat Connaughton, the Fighting Irish beat the eventual national champion Duke Blue Devils twice during the season. The 32 wins were the most by the Fighting Irish team since 1908-09."



借助`tokenized_example`的`sequence_ids`方法，我们可以方便的区分token的来源编号：

- 对于特殊标记：返回None，
- 对于正文Token：返回句子编号（从0开始编号）。

综上，现在我们可以很方便的在一个输入特征中找到答案的起始和结束 Token。


```python
sequence_ids = tokenized_example.sequence_ids()
print(sequence_ids)
```

    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]


 **`sequence_ids`**：

---

### **类比场景**
想象你在玩一个**双色荧光笔标记**的游戏：
- **黄色**：标记问题（比如："贝爷哪年结婚？"）
- **蓝色**：标记书中的答案段落（比如书里写："贝爷2008年结婚..."）
- **红色**：标记特殊符号（比如书的封面、章节分隔页）

`sequence_ids` 就是一个**颜色编号列表**，告诉你每个字属于哪部分。

---

### **三种标记规则**
1. **`None` → 红色标记**  
   - 对应特殊符号：`[CLS]`（开头标志）、`[SEP]`（分隔符）
   - 例：`[CLS]` → `None`

2. **`0` → 黄色标记**  
   - 所有来自**问题**的文字  
   - 例："贝爷"、"哪年" → `0`

3. **`1` → 蓝色标记**  
   - 所有来自**书本文档**的文字  
   - 例："2008年"、"结婚" → `1`

---

### **实际效果示例**
假设问题和文档组合后：
```
[CLS] 贝爷哪年结婚？ [SEP] 贝爷2008年与Jay-Z结婚... [SEP]
```

对应的 `sequence_ids` 就像这样：
```
[ None, 0,0,0,0, None, 1,1,1,1,1, None ]
```
可视化标记：
```
红色 [CLS] → 黄黄黄黄 → 红色 [SEP] → 蓝蓝蓝蓝蓝 → 红色 [SEP]
```

---

### **核心用途**
1. **快速定位答案范围**  
   ```python
   # 找到文档部分的起止位置
   start = sequence_ids.index(1)                 # 第一个蓝色标记的位置
   end = len(sequence_ids) - sequence_ids[::-1].index(1) - 1  # 最后一个蓝色标记
   ```

2. **过滤无效内容**  
   ```python
   # 只处理文档部分的文字
   if sequence_ids[i] == 1:
       print("这是书里的内容！")
   ```

3. **处理长文本分块**  
   - 当文档太长时，自动分成多块，每块都有自己的 `sequence_ids`
   - 例：分块1的蓝色标记对应文档前半部分，分块2对应后半部分

---

### **为什么需要它？**
就像读书时用荧光笔划重点：
- **黄色**：明确问题（知道要找什么）
- **蓝色**：快速锁定答案区域（不用读完整本书）
- **红色**：忽略无关的封面/分隔页

这让模型像人类一样：先看问题，再快速翻书找答案位置，而不是傻傻通读全文。


```python
# 检查分块数量
num_chunks = len(tokenized_example["input_ids"])
print(f"生成分块数: {num_chunks}")

# 遍历每个分块
for chunk_idx in range(num_chunks):
    print(f"\n=== 分块 {chunk_idx} ===")
    
    # 正确获取当前分块的数据
    chunk_input_ids = tokenized_example["input_ids"][chunk_idx]
    chunk_sequence_ids = tokenized_example.sequence_ids(chunk_idx)  # 关键修复点
    
    # 打印关键信息
    print(f"Token数量: {len(chunk_input_ids)}")
    print(f"sequence_ids结构: {chunk_sequence_ids[:20]}...")  # 打印前20个元素
    
    # 检查问题部分是否完整
    question_segment = [i for i, sid in enumerate(chunk_sequence_ids) if sid == 0]
    print(f"问题部分覆盖的token位置: {question_segment[:5]}...")

```

    生成分块数: 2
    
    === 分块 0 ===
    Token数量: 384
    sequence_ids结构: [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1]...
    问题部分覆盖的token位置: [1, 2, 3, 4, 5]...
    
    === 分块 1 ===
    Token数量: 157
    sequence_ids结构: [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1]...
    问题部分覆盖的token位置: [1, 2, 3, 4, 5]...



```python
answers = example["answers"]
start_char = answers["answer_start"][0]
end_char = start_char + len(answers["text"][0])

print(answers)
print(start_char)
print(end_char)
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

    {'text': ['over 1,600'], 'answer_start': [30]}
    30
    40
    23 26


打印检查是否准确找到了起始位置：


```python
# 通过查找 offset mapping 位置，解码 context 中的答案 
print(tokenizer.decode(tokenized_example["input_ids"][0][start_position: end_position+1]))
# 直接打印 数据集中的标准答案（answer["text"])
print(answers["text"][0])
```

    over 1, 600
    over 1,600


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

---

### **功能目标**
这个函数就像一位 **数据加工厂的流水线工人**，负责把原始问答数据改造成适合模型理解的格式。主要解决两个问题：
1. **长文本切割**：当答案文章太长时，切成多个短块（类似将长视频分段）
2. **答案定位**：在每个短块中标注答案的位置（类似视频剪辑时标记精彩片段的起止时间）

---

### **核心处理步骤**

#### **1. 清理问题文字（去左空格）**
- **问题**：用户提问可能包含多余空格，例如 `"   Beyonce哪年结婚？"`
- **处理**：去掉左边的空格 → `"Beyonce哪年结婚？"`
- **原因**：防止空格占用分词名额，导致正文被过度截断

#### **2. 文本分块处理**
- **操作**：将长文章切成多个小块（每块最长 `max_length`，块间重叠 `stride`）
- **示例**：
  ```
  原文章：段落1...段落2...段落3...（总长超过max_length）
  分块1：段落1...段落2（前半）
  分块2：段落2（后半）...段落3
  ```

#### **3. 记录分块关系**
- **overflow_to_sample_mapping**：记录每个分块属于哪个原始样本  
  （类似快递分箱时在每箱贴原订单号）
- **offset_mapping**：记录每个分词对应的原始字符位置  
  （类似每块积木对应原图纸的位置）

#### **4. 处理无答案情况**
- **场景**：当答案不在当前分块中（例如答案在另一个分块里）
- **标记**：将答案位置设为 `[CLS]` 的位置（模型看到这个就知道当前块无答案）

#### **5. 精确定位答案**
- **步骤**：
  1. **确定答案字符范围**：`start_char` 到 `end_char`
  2. **找到分块的上下文部分**（跳过问题和特殊标记）
  3. **检查答案是否在本分块**：
     - 是 → 调整到精确的分词位置
     - 否 → 标记为 `[CLS]`

---

### **实际案例演示**
**输入数据**：
```python
{
    "question": "Beyonce哪年结婚？",
    "context": "Beyonce于2008年与Jay-Z结婚...（长文本）",
    "answers": {"text": ["2008年"], "answer_start": }
}
```

**处理过程**：
1. **分块**：将长 `context` 分成两个块
2. **块1处理**：
   - 发现答案 `2008年` 在块1中
   - 标注起始位置为 `token 6`，结束位置为 `token 7`
3. **块2处理**：
   - 块2不包含答案 → 标注为 `[CLS]`

**输出特征**：
```python
{
    "input_ids": [101, 2345, 3456, ..., 102],  # 分块后的token
    "start_positions": 6, 
    "end_positions": 7
}
```

---

### **参数控制行为**
| 参数 | 作用 | 类比解释 |
|------|------|----------|
| `max_length=384` | 每块最大长度 | 每段视频最长5分钟 |
| `stride=128` | 分块间重叠长度 | 两段视频间重叠30秒防止漏内容 |
| `pad_on_right=True` | 问题在右/左填充 | 字幕在视频左下方还是右下方 |

---

### **总结**
这个函数就像一位智能剪辑师：
1. **切分长视频**（分块处理）
2. **标记关键片段**（答案定位）
3. **处理特殊情况**（无答案时打标记）

最终输出模型可以直接学习的标准化数据格式，是训练高质量问答模型的关键预处理步骤！ 🚀

#### datasets.map 的进阶使用

使用 `datasets.map` 方法将 `prepare_train_features` 应用于所有训练、验证和测试数据：

- batched: 批量处理数据。
- remove_columns: 因为预处理更改了样本的数量，所以在应用它时需要删除旧列。
- load_from_cache_file：是否使用datasets库的自动缓存

datasets 库针对大规模数据，实现了高效缓存机制，能够自动检测传递给 map 的函数是否已更改（因此需要不使用缓存数据）。如果在调用 map 时设置 `load_from_cache_file=False`，可以强制重新应用预处理。

---

### **核心流程类比**
想象你经营一个 **大型快递分拣中心**，需要处理三种包裹（训练集、验证集、测试集）。`datasets.map` 就是你的 **自动化分拣流水线**，`prepare_train_features` 是你定制的 **智能分拣规则**。

---

### **分拣线参数解析**
```python
tokenized_datasets = datasets.map(
    prepare_train_features,  # 你的智能分拣规则
    batched=True,            # 整箱处理（而不是单件）
    remove_columns=原始包裹标签  # 撕掉旧标签
)
```

#### 1. **`batched=True` → 整箱处理模式**
- **传统方式**：工人逐个检查包裹（单条数据处理）
- **高效模式**：整箱倒进机器，同时处理数百个包裹（批量处理）
- **优势**：速度提升 10-100 倍，特别适合 GPU 并行计算

#### 2. **`remove_columns` → 清除旧标签**
- **原因**：经过分拣后，包裹形状改变（数据列变化）
- **操作**：
  - 原始标签：发件人、收件人（`question`, `context` 等）
  - 新标签：目的地代码、重量分级（`input_ids`, `attention_mask` 等）
- **示例**：就像快递重新包装后，需要去掉旧面单

#### 3. **缓存机制 → 智能暂存区**
- **自动检测**：如果分拣规则没变，直接使用暂存区处理好的包裹
- **强制刷新**：`load_from_cache_file=False` 就像要求「不管有没有旧包裹，全部重新分拣」
- **优势**：节省 70% 以上时间，避免重复劳动

---

### **完整工作流程**
1. **收包裹**：三种类型包裹进入流水线（训练/验证/测试集）
2. **规则应用**：
   - 智能切割大包裹（长文本分块）
   - 贴上精准目的地标签（答案位置标记）
   - 丢弃破损包裹（无效样本）
3. **输出结果**：
   - 标准化快递箱（模型可读的 `input_ids` 等）
   - 精准物流标签（`start_positions`, `end_positions`）

---

### **技术细节对应**
| 快递场景 | 数据处理 |
|---------|----------|
| 包裹类型区分 | 保持训练/验证/测试集结构 |
| 分拣机器人 | `prepare_train_features` 函数 |
| 整箱处理 | 批量矩阵运算 |
| 暂存区 | Hugging Face 的缓存文件（通常存于 ~/.cache/huggingface/datasets）|

---

### **为什么需要这样设计？**
1. **效率优先**：如同快递行业追求每日百万件处理量，深度学习的核心就是 **大规模数据吞吐**
2. **资源管理**：缓存机制像双十一的预售包装，提前完成部分工作减轻高峰压力
3. **质量管控**：`remove_columns` 确保不会把生鲜和普通包裹混淆（防止数据污染）

通过这套系统，你的模型就像高效的物流网络，能快速准确地将「问题包裹」送达「答案目的地」！🚚✨


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

---

### 一、任务与模型类的对应关系
Hugging Face Transformers 库为不同任务提供了专用类，就像选择不同的工具：

| 任务类型                  | 对应模型类                          | 示例场景                     |
|--------------------------|-----------------------------------|----------------------------|
| 文本分类                  | `AutoModelForSequenceClassification` | 情感分析、评分预测          |
| 问答任务                  | `AutoModelForQuestionAnswering`     | SQuAD 问答、阅读理解        |
| 文本生成                  | `AutoModelForCausalLM`              | 故事续写、对话生成          |
| 掩码语言建模              | `AutoModelForMaskedLM`              | BERT 式填空任务             |
| 序列到序列                | `AutoModelForSeq2SeqLM`             | 翻译、摘要生成              |
| 标记分类                  | `AutoModelForTokenClassification`   | 命名实体识别、词性标注      |
| 多选任务                  | `AutoModelForMultipleChoice`        | 多选题回答                  |

---

### 二、Transformers 库的主要模型类
以下是常用的模型类（以 **BERT** 架构为例，其他模型类似）：

#### 1. 基础模型
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")  # 通用特征提取
```

#### 2. 任务专用模型
```python
# 文本分类（如情感分析）
AutoModelForSequenceClassification.from_pretrained(...)

# 问答任务（如SQuAD）
AutoModelForQuestionAnswering.from_pretrained(...)

# 文本生成（如GPT风格）
AutoModelForCausalLM.from_pretrained(...)

# 序列到序列（如BART/T5）
AutoModelForSeq2SeqLM.from_pretrained(...)

# 标记级分类（如NER）
AutoModelForTokenClassification.from_pretrained(...)
```

---

### 三、处理自定义任务的三种方案
如果你的任务没有现成类，可以通过以下方法解决：

#### 方案 1：改造现有模型（推荐）
```python
from transformers import AutoModel

# 加载基础模型
model = AutoModel.from_pretrained("bert-base-uncased")

# 添加自定义头部
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = model
        self.custom_head = nn.Linear(768, 3)  # 假设你的任务需要3类输出
        
    def forward(self, inputs):
        outputs = self.bert(**inputs)
        pooled = outputs.last_hidden_state[:,0]  # 取CLS标记
        return self.custom_head(pooled)
```

#### 方案 2：继承并扩展
```python
from transformers import BertPreTrainedModel, BertModel

class MyCustomModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.my_layer = nn.Linear(config.hidden_size, 5)  # 自定义输出维度

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        return self.my_layer(sequence_output[:,0])  # 使用CLS标记
```

#### 方案 3：使用 `AutoModelWithHeads`
（需安装 `adapters` 库）
```python
from transformers.adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("bert-base-uncased")
model.add_classification_head("my_task", num_labels=3)  # 添加分类头
```

---

### 四、关于警告信息的解释
当你看到类似这样的警告：
```
Some weights were not used... (vocab_transform, vocab_layer_norm)
You should probably TRAIN this model...
```
这是 **正常现象**！因为：
1. 预训练模型的原始头部（如MLM头部）被移除
2. 新的任务头部（如分类器）需要重新训练
3. 库在提醒你需要微调后才能用于推理

---

### 五、学习资源推荐
1. [官方任务指南](https://huggingface.co/docs/transformers/task_summary)
2. [自定义模型教程](https://huggingface.co/docs/transformers/custom_models)
3. [社区论坛](https://discuss.huggingface.co/)（遇到问题时优先搜索）

通过灵活组合这些方法，你可以应对任何自定义任务需求！🚀


```python
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

    Some weights of DistilBertForQuestionAnswering were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['qa_outputs.bias', 'qa_outputs.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


#### TensorBoard


```python
from tensorboard import version
print("TensorBoard 版本:", version.VERSION)
```

    TensorBoard 版本: 2.19.0



```python
%load_ext tensorboard

# 指定日志目录和端口（注意这里的端口要与检测的8001一致）
log_dir = "your_logs_directory"  # 替换为实际的日志目录路径
%tensorboard --logdir $log_dir --port 8001 --bind_all
```

    The tensorboard extension is already loaded. To reload it, use:
      %reload_ext tensorboard



    Reusing TensorBoard on port 8001 (pid 1319472), started 17:47:30 ago. (Use '!kill 1319472' to kill it.)




<iframe id="tensorboard-frame-6c031199972a8469" width="100%" height="800" frameborder="0">
</iframe>
<script>
  (function() {
    const frame = document.getElementById("tensorboard-frame-6c031199972a8469");
    const url = new URL("/", window.location);
    const port = 8001;
    if (port) {
      url.port = port;
    }
    frame.src = url;
  })();
</script>




```python
import socket
# 创建TCP套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 尝试连接本地8001端口（非阻塞方式）
result = sock.connect_ex(('localhost', 8001))
# 断言验证（0表示端口开放）
assert result == 0, "TensorBoard 端口 8001 未开启！"
```


```python
# 有问题，暂时不用，先训练

# import evaluate

# # 加载F1指标（支持分类任务的micro/macro/weighted）
# squad_metric = evaluate.load("/root/projects/LLM-learning/evaluate/squad_v2.py" if squad_v2 else "/root/projects/LLM-learning/evaluate/squad.py")

# # 定义计算函数（处理模型输出）
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)  # 分类任务取最大概率类别
#     return squad_metric.compute(
#         predictions=predictions, 
#         references=labels,
#         average="macro"  # "micro"（全局统计）、"macro"（类别平均）、"weighted"（加权平均）
#     )

```

#### 训练超参数（TrainingArguments）


```python
batch_size=64
model_dir = f"models/{model_checkpoint}-finetuned-squad"

args = TrainingArguments(
    output_dir=model_dir,  # 模型/日志保存路径
    evaluation_strategy = "epoch",  # 每个epoch后评估（可选"steps"按步评估）
    learning_rate=2e-5,  # 经典微调学习率（预训练模型的典型学习率范围：1e-5~5e-5）
    per_device_train_batch_size=batch_size,  # 每个GPU的训练批次（总batch_size = 该值 * GPU数量）
    per_device_eval_batch_size=batch_size,   # 每个GPU的评估批次（可大于训练batch_size）
    num_train_epochs=3,  # 训练轮次（SQuAD等中型数据集常用2-5轮）
    weight_decay=0.01,  # L2正则化强度（防止过拟合，常用0.01-0.1）
    fp16=True,  # 启用FP16混合精度
    # save_strategy="epoch",       # 每个epoch保存检查点
    # load_best_model_at_end=True, # 训练结束加载最佳模型
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

    Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.


#### GPU 使用情况

训练数据与模型配置：

- SQUAD v1.1
- model_checkpoint = "distilbert-base-uncased"
- batch_size = 64

NVIDIA GPU 使用情况：

```shell
Every 5.0s: nvidia-smi                                                                                                                                 deepseek-r1-t4-test: Wed Mar 12 17:52:30 2025

Wed Mar 12 17:52:30 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.216.03             Driver Version: 535.216.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:07.0 Off |                    0 |
| N/A   53C    P0              63W /  70W |  10945MiB / 15360MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   1271234      C   /root/miniconda3/envs/peft/bin/python     10942MiB |
+---------------------------------------------------------------------------------------+
```


```python
trainer.train()
```



    <div>

      <progress value='4152' max='4152' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [4152/4152 49:18, Epoch 3/3]
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
      <td>1.498200</td>
      <td>1.273643</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.121800</td>
      <td>1.185791</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.985200</td>
      <td>1.167942</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=4152, training_loss=1.3124258081807336, metrics={'train_runtime': 2959.0362, 'train_samples_per_second': 89.749, 'train_steps_per_second': 1.403, 'total_flos': 2.602335381127373e+16, 'train_loss': 1.3124258081807336, 'epoch': 3.0})



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
              11,  44,  27, 133,  66,  40,  87,  44,  43,  83, 127,  26,  28,  33,
              87, 127,  95,  25,  43, 132,  42,  29,  44,  46,  24,  44,  65,  58,
              81,  14,  59,  72,  25,  36,  55,  43], device='cuda:0'),
     tensor([ 47,  58,  81,  44, 118, 109,  75,  37, 109,  36,  76,  42,  83,  94,
             158,  35,  83,  94,  83,  60,  80,  31,  43,  54,  42,  35,  43,  80,
              13,  45,  28, 133,  66,  41,  89,  45,  44,  85, 127,  27,  30,  34,
              89, 127,  97,  26,  44, 132,  43,  30,  45,  47,  25,  45,  65,  59,
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




    [{'score': 15.2265625, 'text': 'Denver Broncos'},
     {'score': 13.082031,
      'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 11.640625, 'text': 'Carolina Panthers'},
     {'score': 11.4296875, 'text': 'Broncos'},
     {'score': 11.277344,
      'text': 'American Football Conference (AFC) champion Denver Broncos'},
     {'score': 10.154297,
      'text': 'The American Football Conference (AFC) champion Denver Broncos'},
     {'score': 10.123047, 'text': 'Denver'},
     {'score': 9.285156,
      'text': 'Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 9.1328125,
      'text': 'American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 8.009766,
      'text': 'The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 8.008057,
      'text': 'Denver Broncos defeated the National Football Conference'},
     {'score': 7.694336,
      'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina'},
     {'score': 7.317871,
      'text': 'Denver Broncos defeated the National Football Conference (NFC'},
     {'score': 7.251465,
      'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.'},
     {'score': 7.1833496,
      'text': 'Denver Broncos defeated the National Football Conference (NFC)'},
     {'score': 6.987793, 'text': 'AFC) champion Denver Broncos'},
     {'score': 6.864746, 'text': 'champion Denver Broncos'},
     {'score': 6.614258,
      'text': 'National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 6.426758, 'text': 'Panthers'},
     {'score': 6.2529297, 'text': 'Carolina'}]



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

metric = load_metric("squad_v2" if squad_v2 else "squad", trust_remote_code=True)
```


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




    {'exact_match': 74.33301797540209, 'f1': 83.26051790761488}




```python

```

### 加载本地保存的模型，进行评估和再训练更高的 F1 Score


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

    Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.



```python
trained_trainer.train()
```



    <div>

      <progress value='4152' max='4152' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [4152/4152 49:39, Epoch 3/3]
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
      <td>0.846900</td>
      <td>1.212233</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.713500</td>
      <td>1.232366</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.686100</td>
      <td>1.247295</td>
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





    TrainOutput(global_step=4152, training_loss=0.7533242697890324, metrics={'train_runtime': 2980.1413, 'train_samples_per_second': 89.114, 'train_steps_per_second': 1.393, 'total_flos': 2.602335381127373e+16, 'train_loss': 0.7533242697890324, 'epoch': 3.0})




```python
model_to_save_2 = trained_trainer.save_model(f"{model_dir}-2")
```


```python
import torch

for batch in trained_trainer.get_eval_dataloader():
    break
batch = {k: v.to(trained_trainer.args.device) for k, v in batch.items()}
with torch.no_grad():
    output = trained_trainer.model(**batch)
output.keys()
```




    odict_keys(['loss', 'start_logits', 'end_logits'])




```python
output.start_logits.shape, output.end_logits.shape
```




    (torch.Size([64, 384]), torch.Size([64, 384]))




```python
output.start_logits.argmax(dim=-1), output.end_logits.argmax(dim=-1)
```




    (tensor([ 46,  57,  78,  54, 118, 107,  72,  35, 107,  34,  73,  52,  80,  91,
             156,  35,  83,  91,  80,  58,  77,  31,  42,  53,  52,  35,  53,  77,
              11,  44,  27, 133,  66,  40,  87,  44,  43,  41, 127,  26,  28,  33,
              87, 127,  95,  25,  43, 132,  42,  29,  44,  46,  24,  44,  65,  58,
              81,  14,  59,  72,  25,  36,  57,  43], device='cuda:0'),
     tensor([ 47,  58,  81,  44, 118, 110,  75,  37, 110,  36,  76,  42,  83,  94,
             158,  35,  83,  94,  83,  60,  80,  31,  43,  54,  42,  35,  43,  91,
              13,  45,  28, 133,  66,  41,  89,  45,  44,  42, 127,  27,  30,  34,
              90, 127,  97,  26,  44, 132,  43,  30,  45,  47,  25,  45,  65,  59,
              81,  14,  60,  72,  25,  36,  58,  43], device='cuda:0'))




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


```python
validation_features = datasets["validation"].map(
    prepare_validation_features,
    batched=True,
    remove_columns=datasets["validation"].column_names
)
```


```python
raw_predictions = trained_trainer.predict(validation_features)
```






```python
validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
```


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




    [{'score': 18.515625, 'text': 'Denver Broncos'},
     {'score': 15.769531,
      'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 14.910156, 'text': 'Broncos'},
     {'score': 13.9296875, 'text': 'Carolina Panthers'},
     {'score': 13.019531,
      'text': 'The American Football Conference (AFC) champion Denver Broncos'},
     {'score': 12.859375,
      'text': 'American Football Conference (AFC) champion Denver Broncos'},
     {'score': 12.537109, 'text': 'Denver'},
     {'score': 12.1640625,
      'text': 'Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 10.9296875,
      'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.'},
     {'score': 10.2734375,
      'text': 'The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 10.113281,
      'text': 'American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
     {'score': 9.332031,
      'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina'},
     {'score': 9.089844,
      'text': 'Carolina Panthers 24–10 to earn their third Super Bowl title.'},
     {'score': 8.664551, 'text': 'AFC) champion Denver Broncos'},
     {'score': 8.66333,
      'text': 'Denver Broncos defeated the National Football Conference'},
     {'score': 8.26416,
      'text': 'Denver Broncos defeated the National Football Conference (NFC)'},
     {'score': 7.742676, 'text': 'Panthers'},
     {'score': 7.600586,
      'text': 'Denver Broncos defeated the National Football Conference (NFC'},
     {'score': 7.4921875, 'text': 'Carolina'},
     {'score': 7.3242188,
      'text': 'Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.'}]




```python
datasets["validation"][0]["answers"]
```




    {'text': ['Denver Broncos', 'Denver Broncos', 'Denver Broncos'],
     'answer_start': [177, 177, 177]}




```python
import collections

examples = datasets["validation"]
features = validation_features

example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
features_per_example = collections.defaultdict(list)
for i, feature in enumerate(features):
    features_per_example[example_id_to_index[feature["example_id"]]].append(i)
```


```python
final_predictions = postprocess_qa_predictions(datasets["validation"], validation_features, raw_predictions.predictions)
```

    正在后处理 10570 个示例的预测，这些预测分散在 10784 个特征中。



      0%|          | 0/10570 [00:00<?, ?it/s]



```python
from datasets import load_metric

metric = load_metric("squad_v2" if squad_v2 else "squad", trust_remote_code=True)
```


```python
if squad_v2:
    formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
else:
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
metric.compute(predictions=formatted_predictions, references=references)
```




    {'exact_match': 74.91958372753075, 'f1': 83.83155050300782}




```python

```
