# Hugging Face Transformers 微调训练入门

本示例将介绍基于 Transformers 实现模型微调训练的主要流程，包括：
- 数据集下载
- 数据预处理
- 训练超参数配置
- 训练评估指标设置
- 训练器基本介绍
- 实战训练
- 模型保存

## YelpReviewFull 数据集

**Hugging Face 数据集：[ YelpReviewFull ](https://huggingface.co/datasets/yelp_review_full)**

### 数据集摘要

Yelp评论数据集包括来自Yelp的评论。它是从Yelp Dataset Challenge 2015数据中提取的。

### 支持的任务和排行榜
文本分类、情感分类：该数据集主要用于文本分类：给定文本，预测情感。

### 语言
这些评论主要以英语编写。

### 数据集结构

#### 数据实例
一个典型的数据点包括文本和相应的标签。

来自YelpReviewFull测试集的示例如下：

```json
{
    'label': 0,
    'text': 'I got \'new\' tires from them and within two weeks got a flat. I took my car to a local mechanic to see if i could get the hole patched, but they said the reason I had a flat was because the previous patch had blown - WAIT, WHAT? I just got the tire and never needed to have it patched? This was supposed to be a new tire. \\nI took the tire over to Flynn\'s and they told me that someone punctured my tire, then tried to patch it. So there are resentful tire slashers? I find that very unlikely. After arguing with the guy and telling him that his logic was far fetched he said he\'d give me a new tire \\"this time\\". \\nI will never go back to Flynn\'s b/c of the way this guy treated me and the simple fact that they gave me a used tire!'
}
```

#### 数据字段

- 'text': 评论文本使用双引号（"）转义，任何内部双引号都通过2个双引号（""）转义。换行符使用反斜杠后跟一个 "n" 字符转义，即 "\n"。
- 'label': 对应于评论的分数（介于1和5之间）。

#### 数据拆分

Yelp评论完整星级数据集是通过随机选取每个1到5星评论的130,000个训练样本和10,000个测试样本构建的。总共有650,000个训练样本和50,000个测试样本。

## 下载数据集


```python
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
```


    Downloading readme: 0.00B [00:00, ?B/s]



    Downloading data:   0%|          | 0.00/299M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/23.5M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/650000 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/50000 [00:00<?, ? examples/s]



```python
dataset
```




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




```python
dataset["train"][333]
```




    {'label': 3,
     'text': "All in favor of a deep dish pizza say I!.......IIIIIII,  ok now that i have that out of my system. This place is such a great hangout/eat-in spot. I hadn't been here and years and some friends invited us out for the evening. I was so glad they were paying cause  I was low on funds at the time.\\n\\nWe arrived on a friday night and of course it was busy there. We waited about 10 minutes to get a table which wasn't bad considering the crowd. We looked over the menu and they have so many great choices. Pizza, pasta, appetizers, seafood, burgers, salads and sandwiches. \\n\\nAfter ordering two mango lemonades that were wayyyyy over sweetened we ordered our food. We both are going gluten free which is tough but UNO's gave us a nice selection of dishes to choose from. Plus! They make a thin crust gluten free pizza which taste great. My hubby ordered the mediterrean thin crust because he loves kalamata olives and I ordered the Guac-alicious burger with a Caesar side salad. My salad came out pretty quick which was nice but it had a little too much dressing on it. I didn't complain, it still tasted great.\\n\\nI'm not into red meat so I tried to order a black bean burger or get chicken instead of beef, but the ran out of black bean and they couldn't get the chicken so i just ordered it anyway. The burger was piled really high with all the toppings including guacamole and it was very creamy but i couldn't get over the taste of the burger because it just didn't have any flavor. Very saddening. I ended up just eating the veggies and discarding the meat. I snacked on some of my hubbies pizza even though it was only a small amount. \\n\\nWe came here twice in one week. The second time we ordered the 9-grain deep dish with mushrooms, parmesan and a garlic white sauce. Was this pizza amazing or what?? I will probably always eat this pizza whenever I come. \\n\\nOnly down side is slow service. It took 20 minutes for our pizza to come out and my hubbies was a little over cooked. He got the numero uno which was ok but mine had way more flavor!!"}




```python
dataset["train"][666]
```




    {'label': 2,
     'text': 'Just ate there, right next to GameStop & Google, has 3 small booths, & ordered the pepper steak w/ onion ($10.95). Food is fast fresh & hot, but mine had too much onion & not enough steak. At the end of the meal I was just eating onions with rice, though I hear this is healthy for you. Counter lady was cordial, but didn\'t reply when customers told her, \\"Have a nice day\\" #awkward. I know that English isn\'t her first language but she needs to catch on that people are wishing her well. Wasn\'t stuffed full either despite having eaten a large plate (I usually get this feeling eating Asian). This is basically a nice place to go for lunch that won\'t ruin your appetite for dinner. (Side note: Food is very clean. Brushed my teeth an hour before w/ Tom\'s of Maine fluoride-free peppermint & still had minty fresh breath an hour after eating)'}




```python
import random
import pandas as pd
import datasets
from IPython.display import display, HTML
```


```python
def show_random_elements(dataset, num_examples=15):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))
```


```python
show_random_elements(dataset["train"])
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1 star</td>
      <td>We've been coming here for 15+ yrs and we USED to love this place. The owners were great and all the servers were nice. Well we went back today, a Saturday afternoon and we were the only people in there, it was dead. We thought that was pretty rare and odd, well we ended up figuring out it's because of their new servers. From the moment we got there the lady was a total witch. Every single thing she did was done with attitude, not once did I see her even try and put on a fake smile at least. We received our food and it was eh, nothing great at all. Not what it used to be or compared to their other locations.  We were so fed up with her that when we were leaving we decided to ask her if she knew what guest/customer service is and she rolled her eyes and just sipped on her drink the whole time. Her name is V or B, so if you get her, I guarantee you will not be satisfied by her service. We're never coming back here.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5 stars</td>
      <td>I have eaten at several TdB's and the service and food is always amazing and the the one in LV definitely surpassed my expectations. Although we did have reservations, they over shot our reservations by 20-25 minutes. Order the pitcher of Sangria nad set back and enjoy!!</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4 stars</td>
      <td>**UPDATE**\n\nOkay, after staying at the Alladin/ Planet Hollywood I am going to beg Ballys to take us back!!!  I went back there my last day and everything was sunshine compared to the OTHER hotel. I will just have to be super sweet and bat my eyelashes and wink at the receptionist (whether guy or girl) and beg for the newer cleaner rooms next time we are there. My guy and I figured that the people at the OTHER HOTEL are so rude and arent ready to jump to keep you happy because they arent UNIONIZED. I am giving this hotel an upgrade to 4 stars. LUV YA BALLYS!!!! PS... They do have comfortable beds &amp; room service is also good. \n\n\nThis is one of the worst hotels. If you are the new side the rooms are a bit better. But still not up to par. We go there at least 3 times a year because of business and it sucks that the organization can only use this place. Most of the time the check in people are rude.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5 stars</td>
      <td>joe the server was amazing we had a blast he was soo funny and the food was so tasty its a great place for a birthday  we r visiting from california and best service ever he was hilarious and I would go everyday if I could lol</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1 star</td>
      <td>Worst experience ever. They messed up my order twice. Tacos were soggy and hardly had anyway meat. Customer service was horrible to. They were rude and had no customer service skills.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5 stars</td>
      <td>I love Pin Kaow it's hands down the best Thai place for locals off the strip. The staff is always friendly and seems to be a very clean place. I originally went to the location on Eastern for years. Both places are the same, both great. I'm a huge curry fan so I order basically the same thing every visit.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1 star</td>
      <td>Whole Paycheck prices!  What more is there to say!  A small cart will be over $400!\nWayne Gorsek</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2 star</td>
      <td>My friend and I decided to get a Pedicure and came upon this salon.  When we arrived, there weren't any other customers there. They do have a very large selection of different color nail polish to choose from.  We found it a little odd that one of the techs was wearing gloves, and the other was not.  They do spend a lot of time, and do have good attention to detail. My tech accidently cut me a little bit, which he was apologetic for. Before he started the nail polish, he squirted something over my toes (rubbing alcohol perhaps?) which just about made me scream when it hit my cut. When it came to the foot massage/leg rub, it was not pleasant at all. The tech put an exfoliating scrub on my legs, and rubbed my legs for what seemed like forever.  It wasn't really a massage, it felt like they were trying to rub multiple layers of skin off my legs! \nWhen it was all done, I was pleased with the way that my feet looked, but it was not a relaxing experience. I will likely not go there again.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2 star</td>
      <td>Looks very impressive from the outside, but I was disappointed on the interior of the main part. Very blah. Go down to look at the heart of the guy that started it. Creepy!\n\nAlso, we were approached by people outside asking for money. One guy had a whole story about his wife having cancer and how he needed help. I wanted to tell him to go to the church and ask for help. I think he was just looking to scam tourists.\n\nThere are much better churches to visit in the area including Christ Church Cathedral.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5 stars</td>
      <td>Our favorite neighborhood breakfast place.  Chicken on the Coup - is amazing.  Love the stuffing and gravy - and then mix in the eggs - fantastic.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3 stars</td>
      <td>The YardHouse chain of restaurants are generally consistent in terms of food and service. This location is a little unique in that it's located in a very high end mall. The location isn't as large as the Costa Mesa location, but seems larger than many. Service is pretty good if they are busy. If the are not busy, they somehow have really slow, inattentive service. \n\nNot bad, but there's fierce competition for $$ in this area and this is an average experience.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5 stars</td>
      <td>The 5 stars is for a must-do Wednesday 1/2 price martini night.  The lengthy menu will keep you coming back for more.  I didn't catch the pour, I'm sure it is bottom shelf but when you mix it you don't lose.  The servings are generous and not watered down like you get at a lot of places when they have martini specials.\n\nMy husband and I simply sat at the bar.  When the weather cools will certainly opt for outdoor seating.  The staff and other patrons were friendly and quite happy to get their weekly Therapy!</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4 stars</td>
      <td>I'm a big fan of local pizza joints.  I've gotten pizza, wings, fish &amp; chips, olive lover's salad, and bread sticks from this place twice.  \n\nThe pizza is very good,  wings are good, but they were premature baby chick wings.\n\nSalad was good, fish &amp; chips were okay.  \n\nBread sticks were good too.  \n\nFirst time the woman on the phone was pleasant and delivery was awesome!\n\nSecond time, woman on the phone was helpful, but pick - up was interestingly rude.  Don't know what was going on, but that's about it.</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4 stars</td>
      <td>We have gone to this Friday's several times. It's adequate. The food is decent. The atmosphere is fun. The Irish Car Bombs are excellent. :) The service can be hit or miss, but apparently the managers are reading these Yelps cuz the service has gotten considerably better. \n\nWhat makes this restaurant stand out for us is Kara. The last couple times we've come to Fridays, we were served by Kara. She is a FANTASTIC server. She is friendly, professional, and gets our orders right no matter how many in our party. She doesn't forget things or make you feel like you are bothering her. She is GLAD to have you as a guest. I realize that might be what a server is SUPPOSED to do, but you guys know as well as I do that's not always the case, especially with chains. \n\nJust like with any chain, your service can be hit or miss, but give this Friday's a chance and you might just be surprised at the improvement.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2 star</td>
      <td>I've never had banh mi that I thought would break my teeth while eating it.  Not to mention this particular lee's was very frugal of the meat toppings.</td>
    </tr>
  </tbody>
</table>


## 预处理数据

下载数据集到本地后，使用 Tokenizer 来处理文本，对于长度不等的输入数据，可以使用填充（padding）和截断（truncation）策略来处理。

Datasets 的 `map` 方法，支持一次性在整个数据集上应用预处理函数。

下面使用填充到最大长度的策略，处理整个数据集：

1. **为什么要处理文本长度pip show evaluate？**  
   就像衣服有尺码，神经网络模型也有固定的"输入尺寸"。比如BERT模型最多"吃"512个单词片段。太长的文本会被截断（切掉尾巴），太短的会补零（相当于给衣服加填充物）。

2. **分词器在做什么？**  
   把文字转换成数字密码（如"你好"→[101, 2345])，同时：
   • 自动加特殊符号：比如[CLS]开头、[SEP]分隔
   • 记录哪些是真实内容（attention_mask里1表示真实，0是填充的）

3. **map方法的神奇之处**  
   这个操作就像流水线作业，把整个数据集批量送进处理函数。假设数据集有1万条文本，用`batched=True`参数，可能分100批次处理（每批100条），效率比逐条处理高得多。

4. **处理后的数据结构**  
   每个样本会变成包含多个数组的字典：
   ```python
   {
     'input_ids': [101, 2345, 1032, 0, 0],  # 数字化的文本
     'token_type_ids': [0,0,..],            # 区分句子（用于问答任务）
     'attention_mask': [1,1,..0,0]          # 标记有效内容位置
   }
   ```

举个生活化的例子：
原始句子："我爱吃披萨" → 处理后会变成类似：
```
[CLS] 我 爱 吃 披萨 [PAD] [PAD] [PAD]...
对应的数字：[101, 2769, 3342, 1563, 5643, 0, 0, 0...]
注意力的遮罩：[1,1,1,1,1,0,0,0...]
```
其中：
• [CLS]是BERT要求的起始符号
• [PAD]是填充的占位符（实际用0表示）
• 注意力遮罩告诉模型哪些位置需要关注


```python
# 从transformers库导入自动分词器
from transformers import AutoTokenizer

# 加载预训练的分词器（这里用的是BERT的区分大小写版本）
# [2,4](@ref)：Hugging Face的Tokenizer支持填充和截断策略
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# 定义一个处理数据集的函数
def tokenize_function(examples):
    # 对文本进行分词，并应用两个重要策略：
    # 1. padding="max_length"：将所有文本填充到模型允许的最大长度（如512）
    # 2. truncation=True：超过最大长度的部分会被截断
    # [2,4](@ref)：这是Hugging Face推荐的标准化处理方式
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 将处理函数应用到整个数据集（支持批量处理加速）
# batched=True表示一次性处理多个样本，比逐条处理快10倍以上
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

    /root/miniconda3/envs/peft/lib/python3.10/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(



    tokenizer_config.json:   0%|          | 0.00/49.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/339 [00:00<?, ?B/s]



    vocab.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]



    Map:   0%|          | 0/650000 [00:00<?, ? examples/s]



    Map:   0%|          | 0/50000 [00:00<?, ? examples/s]



```python
print(tokenized_datasets.cache_files)
```

    {'train': [{'filename': '/root/.cache/huggingface/datasets/yelp_review_full/yelp_review_full/0.0.0/c1f9ee939b7d05667af864ee1cb066393154bf85/cache-42c6b839c042ef53.arrow'}], 'test': [{'filename': '/root/.cache/huggingface/datasets/yelp_review_full/yelp_review_full/0.0.0/c1f9ee939b7d05667af864ee1cb066393154bf85/cache-b616e165245db566.arrow'}]}



```python
# 随机展示处理后的样本（假设show_random_elements是自定义的检查函数）
# 通过这个可以查看处理后的数据结构，例如：
# {
#   'input_ids': [101, 2345, 1032, ..., 0, 0], 
#   'attention_mask': [1,1,..1,0,0]
# }
show_random_elements(tokenized_datasets["train"], num_examples=1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>text</th>
      <th>input_ids</th>
      <th>token_type_ids</th>
      <th>attention_mask</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1 star</td>
      <td>Went there to celebrate with some friends and I was excited to try some Mexican food - something I'd been missing since I moved from San Diego to Pittsburgh.  Since there was a long wait we got margaritas and waited for the tables.  I got a strawberry margarita, but the flavor was unnecessary as all I could taste was cheap, nasty tequila.  Gross.  To mitigate the taste and try not to be sloshed before dinner we asked for some chips and salsa while we waited.  When we finally got some (15 minutes later) I was buzzed and disappointed.  The chips and salsa were worse than even the store brand from the supermarket.  My opinion, if a \"Mexican food\" restaurant can't do the good chips and salsa and margaritas, I don't even want to try anything else.  We paid our tab and left before our table was even ready.  No thank you.</td>
      <td>[101, 23158, 1204, 1175, 1106, 8294, 1114, 1199, 2053, 1105, 146, 1108, 7215, 1106, 2222, 1199, 4112, 2094, 118, 1380, 146, 112, 173, 1151, 3764, 1290, 146, 1427, 1121, 1727, 4494, 1106, 5610, 119, 1967, 1175, 1108, 170, 1263, 3074, 1195, 1400, 12477, 18971, 15662, 1116, 1105, 3932, 1111, 1103, 7072, 119, 146, 1400, 170, 15235, 6614, 12477, 18971, 15662, 117, 1133, 1103, 16852, 1108, 14924, 1112, 1155, 146, 1180, 5080, 1108, 10928, 117, 13392, 21359, 21005, 119, 15161, 119, 1706, 26410, 25342, 1103, 5080, 1105, 2222, 1136, 1106, 1129, 188, 8867, 8961, 1196, 4014, 1195, 1455, 1111, 1199, 13228, ...]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]</td>
    </tr>
  </tbody>
</table>


Wed Mar  5 15:34:53 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.216.03             Driver Version: 535.216.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:07.0 Off |                    0 |
| N/A   38C    P0              26W /  70W |    985MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A    103559      C   /root/miniconda3/envs/peft/bin/python       982MiB |
+---------------------------------------------------------------------------------------+

根据提供的日志和硬件监控信息，当前系统状态可从以下角度分析：

---

### **一、文件下载与模型加载**
1. **Tokenizer 相关文件处理**  
   • `tokenizer_config.json`（已完成100%下载）：该文件定义了分词器的配置参数（如是否区分大小写、特殊标记映射路径等）。例如，`do_lower_case=True` 表示输入文本会被统一转为小写。
   • `vocab.txt`（下载中）：词汇表文件，包含所有标记及其唯一索引，用于将文本转化为模型可识别的数字序列。例如，`[CLS]`可能对应索引0，`[SEP]`对应索引1。
   • `tokenizer.json`（下载中）：包含分词器的完整配置和模型类型（如BPE、WordPiece），是分词器的核心文件。

2. **模型配置文件加载**  
   • `config.json`（下载中）：定义模型架构参数，如隐藏层维度（`hidden_size`）、注意力头数（`num_attention_heads`）、层数（`num_hidden_layers`）等。例如，`hidden_size=768`表示每层有768个神经元。

3. **进度解读**  
   • `Map: 35%` 可能表示模型权重正在从文件映射到内存，或分词器初始化完成35%。

---

### **二、GPU资源占用**
• **Tesla T4使用情况**  
  • **显存占用**：985MiB/15360MiB，占比约6.4%，显示当前任务对GPU压力较低。
  • **进程信息**：Python进程（PID 103559）正在运行，可能与模型推理或训练相关。例如，加载模型权重（如`model.safetensors`）或执行前向计算。
  • **计算模式**：`Compute M.`显示为`Default`，表明未启用特定计算模式（如MIG多实例GPU）。

---

### **三、综合行为推断**
当前系统可能正在执行以下操作之一：
1. **模型初始化**  
   • 通过Hugging Face的`from_pretrained()`方法加载预训练模型，自动下载配置文件（如`config.json`）和分词器文件。
   • 示例代码类似：
     ```python
     from transformers import AutoTokenizer, AutoModel
     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
     model = AutoModel.from_pretrained("bert-base-uncased").cuda()
     ```

2. **文本预处理**  
   • 使用分词器将输入文本（如用户提问）转换为Token ID序列，需依赖`vocab.txt`和`tokenizer.json`。

3. **轻量级推理任务**  
   • 显存占用较低可能表明任务规模较小（如短文本分类或问答），未涉及全量训练。

---

### **四、潜在风险与优化建议**
• **显存利用率低**：Tesla T4的显存使用率不足10%，可考虑批量处理任务或启用混合精度训练（`fp16`/`bf16`）以提升吞吐量。
• **下载速度限制**：`5.64kB/s`的下载速率可能受网络带宽影响，建议检查代理设置或切换至本地缓存模型。

---

**总结**：系统正在加载一个基于Transformer架构的预训练模型（如BERT或GPT），完成分词器和模型配置的初始化，并利用GPU执行轻量级计算任务。

### 数据抽样

使用 1000 个数据样本，在 BERT 上演示小规模训练（基于 Pytorch Trainer）

`shuffle()`函数会随机重新排列列的值。如果您希望对用于洗牌数据集的算法有更多控制，可以在此函数中指定generator参数来使用不同的numpy.random.Generator。


```python
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

## 微调训练配置

### 加载 BERT 模型

警告通知我们正在丢弃一些权重（`vocab_transform` 和 `vocab_layer_norm` 层），并随机初始化其他一些权重（`pre_classifier` 和 `classifier` 层）。在微调模型情况下是绝对正常的，因为我们正在删除用于预训练模型的掩码语言建模任务的头部，并用一个新的头部替换它，对于这个新头部，我们没有预训练的权重，所以库会警告我们在用它进行推理之前应该对这个模型进行微调，而这正是我们要做的事情。


```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


### 训练超参数（TrainingArguments）

完整配置参数与默认值：https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/trainer#transformers.TrainingArguments

源代码定义：https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/training_args.py#L161

**最重要配置：模型权重保存路径(output_dir)**


```python
from transformers import TrainingArguments

model_dir = "models/bert-base-cased-finetune-yelp"

# logging_steps 默认值为500，根据我们的训练数据和步长，将其设置为100
training_args = TrainingArguments(output_dir=model_dir,
                                  per_device_train_batch_size=16,
                                  num_train_epochs=5,
                                  logging_steps=100)
```


```python
# 完整的超参数配置
print(training_args)
```

    TrainingArguments(
    _n_gpu=1,
    adafactor=False,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    auto_find_batch_size=False,
    bf16=False,
    bf16_full_eval=False,
    data_seed=None,
    dataloader_drop_last=False,
    dataloader_num_workers=0,
    dataloader_persistent_workers=False,
    dataloader_pin_memory=True,
    ddp_backend=None,
    ddp_broadcast_buffers=None,
    ddp_bucket_cap_mb=None,
    ddp_find_unused_parameters=None,
    ddp_timeout=1800,
    debug=[],
    deepspeed=None,
    disable_tqdm=False,
    dispatch_batches=None,
    do_eval=False,
    do_predict=False,
    do_train=False,
    eval_accumulation_steps=None,
    eval_delay=0,
    eval_steps=None,
    evaluation_strategy=IntervalStrategy.NO,
    fp16=False,
    fp16_backend=auto,
    fp16_full_eval=False,
    fp16_opt_level=O1,
    fsdp=[],
    fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},
    fsdp_min_num_params=0,
    fsdp_transformer_layer_cls_to_wrap=None,
    full_determinism=False,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs=None,
    greater_is_better=None,
    group_by_length=False,
    half_precision_backend=auto,
    hub_always_push=False,
    hub_model_id=None,
    hub_private_repo=False,
    hub_strategy=HubStrategy.EVERY_SAVE,
    hub_token=<HUB_TOKEN>,
    ignore_data_skip=False,
    include_inputs_for_metrics=False,
    include_num_input_tokens_seen=False,
    include_tokens_per_second=False,
    jit_mode_eval=False,
    label_names=None,
    label_smoothing_factor=0.0,
    learning_rate=5e-05,
    length_column_name=length,
    load_best_model_at_end=False,
    local_rank=0,
    log_level=passive,
    log_level_replica=warning,
    log_on_each_node=True,
    logging_dir=models/bert-base-cased-finetune-yelp/runs/Jan24_01-36-25_ecs-4325,
    logging_first_step=False,
    logging_nan_inf_filter=True,
    logging_steps=100,
    logging_strategy=IntervalStrategy.STEPS,
    lr_scheduler_kwargs={},
    lr_scheduler_type=SchedulerType.LINEAR,
    max_grad_norm=1.0,
    max_steps=-1,
    metric_for_best_model=None,
    mp_parameters=,
    neftune_noise_alpha=None,
    no_cuda=False,
    num_train_epochs=5,
    optim=OptimizerNames.ADAMW_TORCH,
    optim_args=None,
    output_dir=models/bert-base-cased-finetune-yelp,
    overwrite_output_dir=False,
    past_index=-1,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=16,
    prediction_loss_only=False,
    push_to_hub=False,
    push_to_hub_model_id=None,
    push_to_hub_organization=None,
    push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
    ray_scope=last,
    remove_unused_columns=True,
    report_to=[],
    resume_from_checkpoint=None,
    run_name=models/bert-base-cased-finetune-yelp,
    save_on_each_node=False,
    save_only_model=False,
    save_safetensors=True,
    save_steps=500,
    save_strategy=IntervalStrategy.STEPS,
    save_total_limit=None,
    seed=42,
    skip_memory_metrics=True,
    split_batches=False,
    tf32=None,
    torch_compile=False,
    torch_compile_backend=None,
    torch_compile_mode=None,
    torchdynamo=None,
    tpu_metrics_debug=False,
    tpu_num_cores=None,
    use_cpu=False,
    use_ipex=False,
    use_legacy_prediction_loop=False,
    use_mps_device=False,
    warmup_ratio=0.0,
    warmup_steps=0,
    weight_decay=0.0,
    )


### 训练过程中的指标评估（Evaluate)

**[Hugging Face Evaluate 库](https://huggingface.co/docs/evaluate/index)** 支持使用一行代码，获得数十种不同领域（自然语言处理、计算机视觉、强化学习等）的评估方法。 当前支持 **完整评估指标：https://huggingface.co/evaluate-metric**

训练器（Trainer）在训练过程中不会自动评估模型性能。因此，我们需要向训练器传递一个函数来计算和报告指标。 

Evaluate库提供了一个简单的准确率函数，您可以使用`evaluate.load`函数加载


```python
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
```


接着，调用 `compute` 函数来计算预测的准确率。

在将预测传递给 compute 函数之前，我们需要将 logits 转换为预测值（**所有Transformers 模型都返回 logits**）。


```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

#### 训练过程指标监控

通常，为了监控训练过程中的评估指标变化，我们可以在`TrainingArguments`指定`evaluation_strategy`参数，以便在 epoch 结束时报告评估指标。


```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir=model_dir,
                                  evaluation_strategy="epoch", 
                                  per_device_train_batch_size=16,
                                  num_train_epochs=3,
                                  logging_steps=30)
```

## 开始训练

### 实例化训练器（Trainer）

`kernel version` 版本问题：暂不影响本示例代码运行


```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```

    Detected kernel version 4.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.


## 使用 nvidia-smi 查看 GPU 使用

为了实时查看GPU使用情况，可以使用 `watch` 指令实现轮询：`watch -n 1 nvidia-smi`:

```shell
Every 1.0s: nvidia-smi                                                   Wed Dec 20 14:37:41 2023

Wed Dec 20 14:37:41 2023
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:00:0D.0 Off |                    0 |
| N/A   64C    P0              69W /  70W |   6665MiB / 15360MiB |     98%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A     18395      C   /root/miniconda3/bin/python                6660MiB |
+---------------------------------------------------------------------------------------+
```


```python
trainer.train()
```



    <div>

      <progress value='189' max='189' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [189/189 05:39, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>1.242100</td>
      <td>1.090886</td>
      <td>0.526000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.901400</td>
      <td>0.960115</td>
      <td>0.591000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.638200</td>
      <td>0.978361</td>
      <td>0.592000</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=189, training_loss=0.9693943861300353, metrics={'train_runtime': 341.7098, 'train_samples_per_second': 8.779, 'train_steps_per_second': 0.553, 'total_flos': 789354427392000.0, 'train_loss': 0.9693943861300353, 'epoch': 3.0})




```python
small_test_dataset = tokenized_datasets["test"].shuffle(seed=64).select(range(100))
```


```python
trainer.evaluate(small_test_dataset)
```



<div>

  <progress value='13' max='13' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [13/13 00:02]
</div>






    {'eval_loss': 1.0753791332244873,
     'eval_accuracy': 0.52,
     'eval_runtime': 2.9889,
     'eval_samples_per_second': 33.457,
     'eval_steps_per_second': 4.349,
     'epoch': 3.0}



### 保存模型和训练状态

- 使用 `trainer.save_model` 方法保存模型，后续可以通过 from_pretrained() 方法重新加载
- 使用 `trainer.save_state` 方法保存训练状态


```python
trainer.save_model(model_dir)
```


```python

```


```python
trainer.save_state()
```


```python

```


```python
# trainer.model.save_pretrained("./")
```

## Homework: 使用完整的 YelpReviewFull 数据集训练，看 Acc 最高能到多少


```python

```
