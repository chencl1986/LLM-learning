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
      <td>My first experience at \"Sugar Factory\" was the one at Paris Hotel! Remembered great service and good food!\n\nMy visit this time was because of a Groupon deal - $30 for $62 worth of dinner/drinks.\n\nWe made reservations but was running a little late. When I came up to the hostess and told her that, she pretty much just threw me at the worst possible seat outside. My mistake for assuming that I should've been asked if I wanted indoors or outdoors, or if I wanted to wait for indoor seats. \n\nTook about 10 mins for us to see our waiter, Justin. He took our water order, for we were waiting for our friend, and told us about their half-off special for the night on drinks.\nTook another 10-15 mins to ACTUALLY get our water. \n\nBecause we were waiting for our friend still, we decided to order a couple things - we got a $29 White Gummi Goblet, my $18 3-Piece Chicken Fingers and Onion Rings and my friend's $8 4-Piece, Cracker-Sized Bruschetta.\n\nOur goblet arrives.. As they're suppose to, they bring the cup filled with ice and they poor the drink in front of you. Justin pours the drink .. Then DEMANDS for my friend to sip it because it was going to over flow.. Like DEMANDED, as in told her \"hurry up, sip it because it's going to spill all over the table\" ... W T F?!\n\nOur food comes shortly after the uncomfortable incident, it was good and all but just not worth what we paid for..\n\nWe had ordered another goblet (Berry Bliss) and for what they're worth, they are definitely good drinks. Although they were sweet and fruity, they pack the alcohol punch that comes creepin' after a while of drinking it!\n\nSomewhere along that time period Justin had made really awkward, not so funny jokes.. It made us feel so uncomfortable and he did it in no flirty, nor professional manner!\n\nAs closing time approaches, Justin had asked for my phone for the Groupon voucher, he took it so our discount can be applied to the bill. He came back, dropped the check, and literally ran off .. \nSooooo, where's my phone?!?!? Seriously took a few minutes to see him.. When he finally noticed I was eyeing him down, I asked \"so where's my phone?!\".\nHe ran towards his manager, dropped off my phone and AGAIN awkwardly, unprofessionally joked about his manager looking through my pictures.. True or not, I shouldn't have had to ask for my phone! Nor did he have to make that stupid joke! \nI've seriously have had it with him that night and wonder how and why he works there.\n\nTo end the night, we saw on our bill that gratuity was included... For THREE people ?!? Because we had our Groupon deal??? I honestly didn't understand but I would've really liked for our \"funny\" waiter, Justin, to have explained things..\n\nFor the record, I don't think I'd be coming back ... If I do ..... Nvm yeah def not, thanks to Justin!</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3 stars</td>
      <td>Cool little taco shop in a questionable part of town.  we went early and the place was dead. The waitress was very nice and helpful even going in to the kitchen to get us samples of the meats they offer.  The salsa bar was huge with lots of interesting choices, nothing rocked my world but the cilantro cream was tasty.  we ordered a variety of items, my least favorite being the quesadilla, just tasted like fried greasy tortilla couldn't get past it to eat more than a bite or two.  the tacos were good I had the shark and hubby had the pork it was his favorite. we had a carne asada burrito as well and the meat was tasty.  It was good not great but I think we will be back to try it again.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3 stars</td>
      <td>Burgers.  Great!!  Those chips they serve not as much.  Our portion  unedible  greasy thru and thru. And mentioned to server  she sent manager over. Very nice guy brought us a fresh order and they were not much better.  The server also mixed up the placement of the burger on the table and we had no idea til half way thru when my mom commented the burger was spicy. \nHad the Monty. Great burger nice portion. My niece had the buffalo and the taste was great.  Would be a 4 but for $12 burger the whole meal should be good.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1 star</td>
      <td>I have never written a bad review before - however, I cannot not write this.  I am still shocked at the treatment I received from Gary, supposedly a service oriented nail 'professional.'    He is rude (to say the least), arrogant (without reason), his Groupon is a lie (he doesn't even do Gel nails).  He does NOT even deserve to be in business.  Again, I have NEVER felt so strongly about such horrible customer service.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3 stars</td>
      <td>Best pizza in Vegas! Vegas--you're in trouble if that's true. It's not bad pizza, I just think that the election is rigged. The owners are smart and humor runs high in their marketing ploys. Gimmicks like getting a discount on certain day if you're name is Mark are pretty brilliant. \n\nThe service is also pretty on point and upbeat. The place itself is designed to feel homey like a pizza joint should and succeeds. I am especially fond of the mammoth booths. Extraneous seating is always a winner in my book.  \n\nThe special on my day is that you got free Nuke fries with your order. The spicy Nuke fries are o.k., but I think I would be upset if I paid for them. Gesture appreciated though. Metro has four Eastside specialties on their menu--so my friend and I went with the Four Corners, which gives you two slices of each Eastsides. \n\nWith exciting toppings like eggplant and ricotta in the mix--I thought the Four Corners was going to be a knockout, but the uppercut never came. I got rabbit punched instead. The idea is good topping-wise on these pizzas, but the ingredients all feel very commercial--like straight out of a can or factory. \n\nAlso, the Large pizza is pretty large--so only order thick crust if you are starving or if you want to take some with you. The Four Corners hit the spot as far as my pizza craving, but surely there is better pizza in town. That being said, the pizza hounds have been let loose and the hunt begins!</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4 stars</td>
      <td>Having moved from Utah, where we first found this place, I was very sad to think we'd never be back to it. We were thrilled when one opened up so close to us. And we were not disappointed with the quality. \n\nI usually get a combo with soup and sandwich, though I've recently switched to getting just a regular soup. I never finish the whole sandwich and it feels like a waste to leave half of it there. \nMy husband's favorite sandwich is gone now, replaced by a similar-but-not-the-same option.  Not bad though. \nMy favorite soup by far is the chicken enchilada chili, but I've liked every other one I've tried as well. My parents also love the place. \nThe chocolate covered strawberries are a wonderful touch. I've gotten two desserts so far and they were pretty good. Love the beverage options with the syrups - sprite with cherry and blackberry is my favorite. \nI do wish the bread given with the soups seemed a little more fresh though.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1 star</td>
      <td>I recently moved to the area and have been looking for a nail place close by. Let me just preface this by saying I am the type of customer who never goes into a business requesting extended services close to closing time. That being said, I drove up at 3:55pm on a Sunday (their posted closing time is 5pm) I had planned a mani/pedi but with only an hour I thought I'd guage how many services to get once I walked in.\n\nThe place is pretty and well decorated. It smelled nice and I was promptly greeted as I walked in. The place was empty except for one customer getting a manicure and they were already painting her nails so it was safe to assume she was almost done with her manicure. \n\nThe lady who greeted me asked what I would like. In the interest of time I decided to only get a pedicure and told her I would like a pedicure. She looked at the clock, grimaced and said \"oh, sorry...you wouldnt have enough time for a pedicure before closing\"\n\nI was floored. An hour and 10 mins in an empty salon isn't enough time for a pedicure? ? ? Did they send their staff home early? Were they trying to close early? Not sure...\n\nBecause of the reviews I was expecting a great experience. Not the case! \n\nI went across the street (Albertson's shopping plaza) to Nice Nails and got a mani/pedi and some much needed relaxation :-)\n\nNot sure if I'll give nail room another try, didn't feel like they wanted my business....</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2 star</td>
      <td>$3.95 for a soda?  I should have known right there they weren't interested in locals (yes, I'm familiar with the prices at other local restaurants this is, after all, my neighborhood).  My brisket sandwich had so much fat that I gave up and took the rest home for the dog.  Too bad, because I was looking forward to having a high quality BBQ restaurant, nearby.  I will watch to see if there are any substantial changes in their approach, before returning.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4 stars</td>
      <td>Love this park! it's a nice green space in the midle of downtown. every afternoon you will find runners jogging the path around the park, kids and parents playing on the playground, people playing volleyball on the sand court, and people walking their dogs. compared to where I used to live in Austin, dog owners here are VERY responsible. I rarely ever see someone who doesn't pick up after their pet. (note: the park does not have stations that provide doggy poop bags, you have to bring your own)\n\nonly reason this park doesn't get five stars is because there are quite a few homeless people who use the area for sleeping or hanging out. if you walk under the bridge toward the library you walk right into a HUGE crowd of homeless people hanging out outside the library. it can be a little intimidating.\n\nthere are two shaded ramadas in the park which you can reserve for parties or other events for a fee. the information for how to reserve one is posted on the ramadas themselves.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2 star</td>
      <td>My family arrived to celebrate my moms birthday! The hosting staff did a great job of seating us in a minimal amount of time. We had an emergency change in plans and they accommodated us quickly. They also accommodated the temperature of the room for our easily cooled friends. Everyone enjoyed their food. And the server was great UNTIL......two spiders crawled up onto our table right between me and my son. When I complained, I got some half assed excuse that they had just sprayed the night before from the manager. My husband, not wanting to look douchey and not like we were just trying to get a free meal, told them not to worry about it. But personally, I think they still should have done something about it. When you pay for a $500 meal, you don't want to deal with bugs. I was in the service industry for nearly 10 years, I understand the prejudice that goes on behind closed doors. I understand that they probably thought I was lying, but frankly, I couldn't care less. That's gross. I don't want to pay for a meal with bugs.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2 star</td>
      <td>I got a promotion in the mail for ridiculously low rates so I had to take them up on the offer. I kind of wish I hadn't. The hotel, or at least my room has seen better days. It was really worn down. Everything was just good enough nothing more nothing less. The towels were a little rough and the bed was a bit lumpy. \n\nThe casino is actually kind of small and limited, it seems this place lives off more from its nightlife. I don't think I would stay at the Palms again. There are so many other places in Vegas that you can get a whatever experience.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2 star</td>
      <td>I lived here from 2008-2010 and did genuinely enjoy my experience with the apartment itself.  The building is old construction which was nice because I didn't have an issue hearing my neighbors.  Also didn't have a big issue with the heat/cold as the place was well insulated.  The location is also great as you are close to Old Town and the 101.\n\nThe issues I did have were with the management company.  Rent is due by the 2nd and HAS to be paid online.  The office doesn't tell you that it has to be paid by 12 AM Central Standard time otherwise you will get a $50 late charge.  I incurred a late charge and asked the office to reverse it since I had paid rent on time but was told since it was paid at 12:05 AM on the 3rd CST, they would not reverse the charges.  \n\nAt first when I moved in there were around 4 washers/dryers for 40 units.  The laundry room was remodeled and the washer/dryers were reduced down to 2 units each.  The explanation was that the new models were \"higher efficiency\" and would be much faster but just because the wash cycle is faster doesn't mean your neighbor will be courteous and remove their laundry on time.  The worst part was I had someone steal my underwear on 3 separate occasions.  When I called the office to report it I was told \"maybe you should just stay with your laundry so that people don't steal your personal items.\"  It was only after the 3rd time that I filed a police report and found out that the SDPD had had SEVERAL complaints about this issue and the office chose to over look them.   \n\nI also had issues with people parking in my assigned space and even had a glove box full of \"nasty grams\" to put on the offenders windshield.  One time I called the # listed on the carport to have the car towed and was told I needed a special \"code\" to have them come out.  I called the office and asked for the code and was told they don't give that out to residents but I could call during office hours and they would be happy to help with my parking issue.  I explained it was happening at 11 PM and the office wasn't open but was told to just find an uncovered spot to park in.\n\nThe kicker was when I moved out.  I had lived in the apartment for 2 years and did my best on upkeep.  Once I moved out I got a bill saying I owed them $200 over other $200 refundable deposit I had put down when I moved in.  I asked for an itemized list of what I was charged for and found out I was charged $180 to repaint my 500 sq foot apartment which I found odd since there were no holes in the wall nor had I made any paint color changes.  Was also charged to replace the carpet which was funny because it wasn't new when I moved in, just shampooed.\n\nOh yeah, dumpster divers totally true.  Happened at least once a week</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3 stars</td>
      <td>Quick review\nI expected the wait for seating, but not the long wait for the food (it's only burgers).\nAfter the wait, still got my temp wrong  on my burger (more waiting)\nTaste was good but portion were small for the high price on Maui onion rings, fries and burgers.\nService was real good. \nI believe KGB (Kerry Gourmet burger) and Burger bar are much better burger places.\nStill a fan of Chef Ramsey.</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5 stars</td>
      <td>This is my favorite Asian restaurant in the Valley. This place is not much for ambiance, but it features wonderfully authentic Cantonese- Hong Kong cuisine at outrageously cheap prices, served by very friendly and helpful staff. The menu of Asian Cafe Express is extensive, but skip the first few pages (these list the \"Americanized Chinese\" dishes that most people are familiar with) and go directly to the pages with the items categorized as \"Hong Kong style\" or \"HK style.\" If you're not sure what to order, you can browse the huge pictures of popular dishes, on the wall. But it will be best to ask the waitpersons for suggestions -- they'll be more than happy to point out dishes in the menu that are favorites of the many regular customers there. Portions are more than ample, even for the soups: the smallest serving (\"medium\"  in the menu) of soup serves two generously.  And they have several varieties of congee, including my favorite: with pork and preserved egg -- comfort food like no other for this Asian ;-)\n\nFor those who use the light rail, this is a convenient dining destination because it is so close to the Mesa terminus of the light rail. Be prepared for a taste treat, and don't act too surprised when the bill arrives and you find that you don't pay much for such great food.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2 star</td>
      <td>Am I the ONLY person in the state who doesn't need to eat here?  Can someone tell me what I am missing?  I just don't get the hype.  The food is ok, nothing stands out as \"must come back for\".  I will keep coming, because my sister and mother are addicted, and I am sure there is something on the menu I can find that I will like.</td>
    </tr>
  </tbody>
</table>


## 预处理数据

下载数据集到本地后，使用 Tokenizer 来处理文本，对于长度不等的输入数据，可以使用填充（padding）和截断（truncation）策略来处理。

Datasets 的 `map` 方法，支持一次性在整个数据集上应用预处理函数。

下面使用填充到最大长度的策略，处理整个数据集：

1. **为什么要处理文本长度？**  
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



```python
print(tokenized_datasets.cache_files)
```

    {'train': [{'filename': '/root/.cache/huggingface/datasets/yelp_review_full/yelp_review_full/0.0.0/c1f9ee939b7d05667af864ee1cb066393154bf85/cache-42c6b839c042ef53.arrow'}], 'test': [{'filename': '/root/.cache/huggingface/datasets/yelp_review_full/yelp_review_full/0.0.0/c1f9ee939b7d05667af864ee1cb066393154bf85/cache-90992f974cd05082.arrow'}]}



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
      <td>5 stars</td>
      <td>I went to the Charleston location yesterday and the staff was a bit rude and didn't know their own product. Today I decided to go to the Ft. Apache and Trop location and the ladies were just darling and were great help. Ashley was wonderful and helpful. The staff went out of their way to help. One of the girls helped even tie the bowl part to the roof of my car since it wouldn't fit anywhere into the car. Thank you guys. You all ROCK :) I totally love my new papasan chair.</td>
      <td>[101, 146, 1355, 1106, 1103, 10874, 2450, 8128, 1105, 1103, 2546, 1108, 170, 2113, 14708, 1105, 1238, 112, 189, 1221, 1147, 1319, 3317, 119, 3570, 146, 1879, 1106, 1301, 1106, 1103, 143, 1204, 119, 16995, 1105, 157, 12736, 2450, 1105, 1103, 8564, 1127, 1198, 18556, 1105, 1127, 1632, 1494, 119, 9017, 1108, 7310, 1105, 14739, 119, 1109, 2546, 1355, 1149, 1104, 1147, 1236, 1106, 1494, 119, 1448, 1104, 1103, 2636, 2375, 1256, 5069, 1103, 7329, 1226, 1106, 1103, 3664, 1104, 1139, 1610, 1290, 1122, 2010, 112, 189, 4218, 5456, 1154, 1103, 1610, 119, 4514, 1128, 3713, 119, 1192, 1155, 155, ...]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]</td>
    </tr>
  </tbody>
</table>


以下是该表格中各个字段的详细解释，按NLP处理流程分阶段说明：

---

### ▋ 字段结构解析 (针对BERT类模型)

| 字段名称          | 示例值片段                     | 作用层级      | 技术细节                                                                 |
|-------------------|------------------------------|--------------|--------------------------------------------------------------------------|
| **label**         | "1 star"                     | 业务标签层    | 原始业务标签（此处展示为可读形式，实际训练需转换为数值如0-4对应1-5星）    |
| **text**          | 用户评论原文                 | 原始数据层    | 未处理的原始文本输入                                                     |
| **input_ids**     | [101, 23158, 1204,...]       | Token编码层   | 将文本转换为模型可识别的数字ID序列                                        |
| **token_type_ids**| [0, 0, 0,...]                | 句子分段层     | 标识token属于哪个句子（单句子任务全为0）                                  |
| **attention_mask**| [1, 1, 1,...]                | 注意力机制层   | 控制模型关注有效内容（1=有效token，0=填充位）                             |

---

### ▋ 关键技术点详解

#### 1. **label字段的特殊处理**
```python
# 实际训练时应转换为数值标签
label_mapping = {"1 star": 0, "2 stars": 1, ..., "5 stars": 4}
dataset = dataset.map(lambda x: {"label": label_mapping[x["label"]]})
```

#### 2. **input_ids的构造过程**
- **特殊标记说明**：
  - `101`: [CLS] 分类标记（BERT等模型的起始符）
  - `102`: [SEP] 分隔标记（此例未出现，因单句输入）
  - `0`: [PAD] 填充标记（此例未出现，因已用max_length填充）

#### 3. **token_type_ids的扩展应用**
```python
# 双句子任务时的典型结构（如QA）
tokenizer("How are you?", "I'm fine", return_token_type_ids=True)
# 输出：
# token_type_ids = [0,0,0,0,0, 1,1,1,1]
```

#### 4. **attention_mask的动态性**
```python
# 实际处理变长文本时的mask示例：
原始文本: "Hello world"
填充后: "Hello world [PAD] [PAD]"
attention_mask: [1,1,0,0]
```

---

### ▋ 数据处理流程可视化
```
原始文本
   ↓ (分词器处理)
[CLS] Went there to... [SEP] → 分词结果
   ↓ (词汇表映射)
101 23158 1204 ... 102 → input_ids
   ↓ (句子标识)
0   0     0    ... 0   → token_type_ids
   ↓ (有效标识)
1   1     1    ... 1   → attention_mask
```

---

### ▋ 最佳实践建议
1. **动态填充策略**：
```python
# 替代固定长度填充，提升效率
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer)
```

2. **验证字段一致性**：
```python
# 检查各字段长度是否匹配
assert len(input_ids) == len(token_type_ids) == len(attention_mask)
```

3. **解码验证**：
```python
# 反向验证编码正确性
decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
assert decoded_text == original_text
```

---

是否需要进一步了解如何将这些预处理后的数据输入模型进行训练？

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
# 从完整训练集中创建小型训练子集（1000条样本）
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
"""
执行步骤：
1. shuffle(seed=42): 先对训练集进行随机打乱（设置随机种子保证可复现性）
2. select(range(1000)): 选取前1000条打乱后的样本
作用：
- 创建小规模训练集，加速实验迭代
- 保持数据分布的随机性
- 固定随机种子保证每次运行结果一致
"""

# 从完整测试集中创建小型验证子集（1000条样本） 
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
"""
典型应用场景：
1. 快速验证模型是否能过拟合（用少量数据测试学习能力）
2. 资源有限时进行超参数调试
3. 原型开发阶段的快速实验
4. 教学演示场景（缩短训练时间）

注意事项（使用时需知）：
- 小样本可能无法代表完整数据分布
- 评估指标会有较大方差
- 正式训练时建议使用完整数据集
- 生产环境需要更严谨的验证集划分
"""

# 扩展：查看数据集结构示例
print(small_train_dataset)
# 输出示例：Dataset(features: ['input_ids', 'token_type_ids', 'attention_mask', 'label'], num_rows: 1000)

print(small_eval_dataset)
```

    Dataset({
        features: ['label', 'text', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 1000
    })
    Dataset({
        features: ['label', 'text', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 1000
    })


## 微调训练配置

### 加载 BERT 模型

警告通知我们正在丢弃一些权重（`vocab_transform` 和 `vocab_layer_norm` 层），并随机初始化其他一些权重（`pre_classifier` 和 `classifier` 层）。在微调模型情况下是绝对正常的，因为我们正在删除用于预训练模型的掩码语言建模任务的头部，并用一个新的头部替换它，对于这个新头部，我们没有预训练的权重，所以库会警告我们在用它进行推理之前应该对这个模型进行微调，而这正是我们要做的事情。


```python
from transformers import AutoModelForSequenceClassification

# 关键参数解析：
# "bert-base-cased" - 使用区分大小写的BERT基础版
# num_labels=5       - 五分类任务（对应Yelp的1-5星评分）
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", 
    num_labels=5
)
```

    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


### 训练超参数（TrainingArguments）

完整配置参数与默认值：https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/trainer#transformers.TrainingArguments

源代码定义：https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/training_args.py#L161

**最重要配置：模型权重保存路径(output_dir)**


```python
from transformers import TrainingArguments

# 💾 模型保存路径配置（最重要参数！所有训练产出物都会保存到这里）
model_dir = "models/bert-base-cased-finetune-yelp"  # 推荐使用模型名+任务名的目录结构

# 🎛️ 创建训练参数配置实例
training_args = TrainingArguments(
    # 必须参数
    output_dir=model_dir,                   # 模型/日志/检查点的保存根目录
                                          # 📂 目录将包含：
                                          #   |- config.json
                                          #   |- trainer_state.json
                                          #   |- checkpoint-100/...
    
    # ⚡ 训练效率参数
    per_device_train_batch_size=16,       # 每个GPU的批次大小（调整依据显存）
                                          # 💡 3080显卡建议值：16-32
                                          # ❗ 总批次大小 = 该值 * GPU数 * gradient_accumulation_steps
    
    # ⏱️ 训练时长控制
    num_train_epochs=5,                   # 训练总轮次（建议先用1轮试跑，再全量训练）
                                          # 🔄 1个epoch = 完整过一遍训练集
    
    # 📊 日志配置
    logging_steps=100,                    # 每训练100步记录一次日志（默认500）
                                          # 📈 设小值可更密集监控训练过程
                                          # 💻 控制台将输出：
                                          #    Step | Loss   | Learning Rate
                                          #    100  | 0.532  | 0.00005
)

# 扩展建议参数（根据需求添加）：
"""
# 🎯 优化器配置
learning_rate=5e-5,                     # BERT常用初始学习率（5e-5 ~ 3e-5）
weight_decay=0.01,                      # 权重衰减防过拟合

# 📉 学习率调度
warmup_steps=500,                       # 预热步数（从小学习率逐步上升）

# 💽 内存优化
fp16=True,                              # 启用混合精度训练（显存减半）
gradient_accumulation_steps=2,          # 梯度累积步数（模拟更大批次）

# 🧪 验证配置
evaluation_strategy="steps",             # 每eval_steps评估一次
eval_steps=200,                         # 评估频率
load_best_model_at_end=True,            # 训练结束时加载最佳模型
"""

```




    '\n# 🎯 优化器配置\nlearning_rate=5e-5,                     # BERT常用初始学习率（5e-5 ~ 3e-5）\nweight_decay=0.01,                      # 权重衰减防过拟合\n\n# 📉 学习率调度\nwarmup_steps=500,                       # 预热步数（从小学习率逐步上升）\n\n# 💽 内存优化\nfp16=True,                              # 启用混合精度训练（显存减半）\ngradient_accumulation_steps=2,          # 梯度累积步数（模拟更大批次）\n\n# 🧪 验证配置\nevaluation_strategy="steps",             # 每eval_steps评估一次\neval_steps=200,                         # 评估频率\nload_best_model_at_end=True,            # 训练结束时加载最佳模型\n'




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
    evaluation_strategy=no,
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
    hub_strategy=every_save,
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
    logging_dir=models/bert-base-cased-finetune-yelp/runs/Mar06_16-27-33_deepseek-r1-t4-test,
    logging_first_step=False,
    logging_nan_inf_filter=True,
    logging_steps=100,
    logging_strategy=steps,
    lr_scheduler_kwargs={},
    lr_scheduler_type=linear,
    max_grad_norm=1.0,
    max_steps=-1,
    metric_for_best_model=None,
    mp_parameters=,
    neftune_noise_alpha=None,
    no_cuda=False,
    num_train_epochs=5,
    optim=adamw_torch,
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
    save_strategy=steps,
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


以下是对 `TrainingArguments` 配置的通俗解释，按功能分类并添加注释：

---

### 🌟 **核心训练参数**
```python
output_dir="models/bert-base-cased-finetune-yelp"  # 模型保存路径（最重要！训练结果全存在这）
per_device_train_batch_size=16   # 每个GPU的批次大小（显存不足时调小此值）
num_train_epochs=5               # 训练总轮次（通常3-5轮足够微调）
learning_rate=5e-05              # 学习率（BERT常用5e-5, 太大容易震荡，太小收敛慢）
```

---

### 💻 **资源相关参数**
```python
fp16=False                       # 是否启用混合精度训练（True可省显存，需GPU支持）
gradient_accumulation_steps=1    # 梯度累积步数（模拟更大批次，显存不足时使用）
optim="adamw_torch"              # 优化器类型（Adam的改进版，适合深度学习）
```

---

### ⏱️ **训练过程控制**
```python
logging_steps=100                # 每100步打印一次日志（默认500，调小可更频繁监控）
save_steps=500                   # 每500步保存一次模型（频繁保存会占用磁盘）
evaluation_strategy="no"         # 评估策略（"no"不评估，"steps"按步评估，"epoch"每轮评估）
```

---

### 🛠️ **优化相关参数**
```python
weight_decay=0.0                 # 权重衰减系数（防过拟合，常用0.01）
warmup_steps=0                   # 预热步数（初始阶段用小学习率）
max_grad_norm=1.0                # 梯度裁剪阈值（防梯度爆炸）
```

---

### 📊 **日志与保存**
```python
logging_dir="models/.../runs/..." # TensorBoard日志路径
report_to=[]                     # 上报平台（例如["wandb"]接入可视化）
save_total_limit=None            # 最大保存检查点数（设为3只保留最新3个模型）
```

---

### 🔧 **其他实用参数**
```python
seed=42                          # 随机种子（固定后结果可复现）
disable_tqdm=False               # 是否禁用进度条（True时更简洁）
remove_unused_columns=True       # 自动删除模型不需要的列（节省内存）
```

---

### 🚀 **参数选择建议**
1. **学习率**：从 `5e-5` 开始尝试，观察损失变化
2. **批次大小**：在显存允许范围内尽量调大（如16→32）
3. **训练轮次**：用早停法（`EarlyStoppingCallback`）防过拟合
4. **混合精度**：设置 `fp16=True` 可减少30%显存占用
5. **多GPU支持**：无需修改代码，启动时加 `--nproc_per_node=GPU数量`

---

### ⚠️ **特别注意项**
```python
do_train=False  # 当前配置未启用训练！（需设为True才会开始训练）
do_eval=False   # 当前未启用验证！（需配合eval_dataset使用）
```

---

### 🔄 **完整训练启动示例**
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,       # 需要评估时添加
    compute_metrics=compute_metrics  # 自定义评估函数
)

# 开始训练（do_train=True时才生效）
trainer.train() 
```

---

通过合理配置这些参数，可以平衡训练速度、资源消耗和模型性能。建议先用小数据子集调试参数，再全量训练。

## 什么是学习率，什么是震荡？

### 📚 学习率（Learning Rate）通俗解释

**学习率就像「下山的步长」**  
想象你蒙着眼从山顶往下走，要找到最低点（模型的最优参数）。学习率就是你每步迈的幅度：
- **太大步（高学习率）**：容易一步跨过山谷，在对面的山坡来回跳（震荡）
- **太小步（低学习率）**：要走很久才能到底，甚至卡在半山腰（收敛慢）

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*kA38KJq9aeZkBW-0rH2yCg.gif)

---

### 💥 什么是震荡（Oscillation）？

**震荡就像「在山谷两边反复横跳」**  
当学习率太大时，参数更新会像这样：
1. 当前点：A（损失较高）
2. 计算梯度：指向谷底方向
3. 大步更新：直接跳到对面的B点
4. 再计算梯度：又指向另一个方向
5. 结果：在谷底两侧反复跳动，无法稳定到最低点

![](https://developers.google.com/static/machine-learning/crash-course/images/LearningRateTooLarge.svg)

---

### 🌰 具体案例对比
| 学习率 | 训练表现 | 损失曲线 | 适用场景 |
|-------|---------|---------|---------|
| 0.1   | 剧烈震荡 | 锯齿状波动 | ❌ 几乎不用 |
| 1e-3  | 偶尔震荡 | 波动下降 | 简单任务 |
| 5e-5  | 平稳下降 | 平滑收敛 | ✅ BERT微调 |
| 1e-6  | 缓慢下降 | 近乎水平 | 精细调优 |

---

### 🔧 如何避免震荡？
1. **学习率预热**：前1000步从小学习率逐步增大
2. **梯度裁剪**：限制单步更新幅度（`max_grad_norm=1.0`）
3. **自适应优化器**：使用AdamW而不是SGD
4. **监控损失曲线**：出现震荡时立即降低学习率

---

### 🛠️ BERT的实践经验
```python
# 安全的学习率范围建议
learning_rate = 5e-5  # 默认安全值（适合大部分情况）
learning_rate = 3e-5  # 更保守的选择（数据量小时）
learning_rate = 1e-4  # 高风险！需配合梯度裁剪使用
```

通过合理控制学习率，可以让模型既快速收敛，又不会「跑过头」。就像开车时找到合适的油门力度，既不会急刹急停，也不会龟速前进。

### 训练过程中的指标评估（Evaluate)

**[Hugging Face Evaluate 库](https://huggingface.co/docs/evaluate/index)** 支持使用一行代码，获得数十种不同领域（自然语言处理、计算机视觉、强化学习等）的评估方法。 当前支持 **完整评估指标：https://huggingface.co/evaluate-metric**

训练器（Trainer）在训练过程中不会自动评估模型性能。因此，我们需要向训练器传递一个函数来计算和报告指标。 

Evaluate库提供了一个简单的准确率函数，您可以使用`evaluate.load`函数加载


```python
import numpy as np
import evaluate
import os

# 确认环境变量设置
print("当前HF端点:", os.getenv('HF_ENDPOINT', '默认（未设置）'))  # 应该显示 https://hf-mirror.com

metric = evaluate.load("./accuracy.py")
```

    当前HF端点: https://hf-mirror.com



接着，调用 `compute` 函数来计算预测的准确率。

在将预测传递给 compute 函数之前，我们需要将 logits 转换为预测值（**所有Transformers 模型都返回 logits**）。


```python
def compute_metrics(eval_pred):
    """计算模型评估指标的核心函数
    
    Args:
        eval_pred (tuple): 包含模型输出logits和真实标签的元组
            logits (np.ndarray): 模型输出的未归一化概率分布，shape=(batch_size, num_classes)
            labels (np.ndarray): 真实标签，shape=(batch_size,)
    
    Returns:
        dict: 包含评估指标的字典，例如 {'accuracy': 0.85}
    
    Notes:
        - 适用于分类任务，多分类场景需确保axis参数正确
        - 多标签分类需改用sigmoid +阈值处理
    """
    # 解包模型输出和标签
    logits, labels = eval_pred  # 等价于 logits = eval_pred, labels = eval_pred
    
    # 将logits转换为预测类别（取概率最大的类别）
    predictions = np.argmax(logits, axis=-1)  # axis=-1表示最后一个维度（分类维度）
    # 示例：logits.shape=(32,5)→predictions.shape=(32,)
    
    # 调用评估指标计算（假设metric是accuracy）
    return metric.compute(
        predictions=predictions,  # 模型预测的类别索引
        references=labels         # 真实的类别索引
    )


# 典型应用场景示例
"""
输入样例：
eval_pred = (
    np.array([[1.2, -0.5], [0.3, 2.1]], dtype=np.float32),  # logits（batch_size=2, num_classes=2）
    np.array([0, 1], dtype=np.int32)                        # labels
)
输出结果：
{'accuracy': 0.5}  # 第0个样本预测正确，第1个预测错误（argmax([0.3,2.1])=1，但标签是1，实际应该正确？这里可能需要检查示例数据）
"""

# 扩展功能建议
"""
1. 多指标计算：
   f1 = evaluate.load("f1")
   return {
       "accuracy": metric.compute(...),
       "f1": f1.compute(...)
   }

2. 处理多维度输出（如NER）：
   predictions = np.argmax(logits, axis=-1)  # shape=(batch_size, seq_len)

3. 概率校准：
   probs = softmax(logits, axis=-1)
   return {"roc_auc": roc_auc_score(labels, probs[:,1])}
"""

```




    '\n1. 多指标计算：\n   f1 = evaluate.load("f1")\n   return {\n       "accuracy": metric.compute(...),\n       "f1": f1.compute(...)\n   }\n\n2. 处理多维度输出（如NER）：\n   predictions = np.argmax(logits, axis=-1)  # shape=(batch_size, seq_len)\n\n3. 概率校准：\n   probs = softmax(logits, axis=-1)\n   return {"roc_auc": roc_auc_score(labels, probs[:,1])}\n'



#### 训练过程指标监控

通常，为了监控训练过程中的评估指标变化，我们可以在`TrainingArguments`指定`evaluation_strategy`参数，以便在 epoch 结束时报告评估指标。


```python
from transformers import TrainingArguments, Trainer

# 🎛️ 训练参数配置
training_args = TrainingArguments(
    # 必须参数
    output_dir=model_dir,  # 模型保存路径（⚠️重要！会自动创建以下内容）
    # 📂 目录结构示例：
    #   ├── config.json
    #   ├── runs/ (TensorBoard日志)
    #   └── checkpoint-100/ (自动保存的检查点)
    
    # 🔄 验证策略
    evaluation_strategy="epoch",  # 每个epoch后验证（可选："steps"按步验证/"no"不验证）
    # eval_steps=500,           # 当evaluation_strategy="steps"时，每500步验证一次
    
    # 🚀 训练效率参数
    per_device_train_batch_size=16,  # 每个GPU的批次大小（调整依据显存）
    # 总批次大小 = 16 * GPU数量 * gradient_accumulation_steps
    
    # ⏱️ 训练时长控制
    num_train_epochs=3,           # 训练总轮次（推荐3-5轮用于微调）
    
    # 📊 日志与监控
    logging_steps=30,             # 每30训练步记录一次日志（默认500）
    # report_to="wandb",         # 集成可视化工具（可选："tensorboard"/"wandb"）
    
    # 💡 推荐添加的优化参数
    # learning_rate=5e-5,        # BERT常用学习率（默认5e-5）
    # fp16=True,                # 混合精度训练（节省30%显存）
    # gradient_accumulation_steps=2, # 梯度累积（模拟更大批次）
    # warmup_steps=500,         # 学习率预热步数
    # load_best_model_at_end=True, # 训练完成时加载最佳模型
)

# 典型应用场景示例
"""
# 初始化训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics  # 需要自定义评估函数
)

# 启动训练
trainer.train() 

# 保存最终模型
trainer.save_model("final_model")
"""

```




    '\n# 初始化训练器\ntrainer = Trainer(\n    model=model,\n    args=training_args,\n    train_dataset=train_dataset,\n    eval_dataset=val_dataset,\n    compute_metrics=compute_metrics  # 需要自定义评估函数\n)\n\n# 启动训练\ntrainer.train() \n\n# 保存最终模型\ntrainer.save_model("final_model")\n'



## 开始训练

### 实例化训练器（Trainer）

`kernel version` 版本问题：暂不影响本示例代码运行


```python
# 初始化训练器 (核心训练引擎)
trainer = Trainer(
    # 🧠 模型配置
    model=model,  # 要训练的模型实例（需继承自PreTrainedModel）
    # 示例：AutoModelForSequenceClassification.from_pretrained(...)
    
    # ⚙️ 训练参数配置
    args=training_args,  # 训练参数对象（包含batch_size/epochs等超参数）
    # 通过TrainingArguments类创建，控制训练全过程
    
    # 📦 数据配置
    train_dataset=small_train_dataset,  # 训练集（Dataset对象）
    # 建议格式：datasets.Dataset或torch.utils.data.Dataset
    eval_dataset=small_eval_dataset,    # 验证集（用于评估模型性能）
    # 数据预处理应在加载dataset前完成
    
    # 📊 评估指标配置
    compute_metrics=compute_metrics,  # 自定义评估指标计算函数
    # 该函数接收eval_pred(logits, labels)，返回指标字典
    # 示例：计算准确率/召回率等
    
    # 🔄 可选高级配置（按需添加）
    # data_collator=collate_fn,        # 自定义批次数据打包逻辑
    # callbacks=[EarlyStoppingCallback(...)], # 训练回调（如早停）
    # tokenizer=tokenizer,            # 用于数据预处理记录
)

# 典型训练流程
"""
1. 启动训练：
   trainer.train() 
   
2. 评估模型：
   eval_results = trainer.evaluate()
   
3. 保存最佳模型：
   trainer.save_model("best_model")
   
4. 查看训练日志：
   !tensorboard --logdir=models/bert-base-cased-finetune-yelp/runs/
"""

# 参数详解表
"""
| 参数                | 作用                                 | 典型值示例                     |
|---------------------|--------------------------------------|-------------------------------|
| model               | 定义模型架构                         | BertForSequenceClassification |
| args                | 控制训练超参数                       | TrainingArguments实例         |
| train_dataset       | 模型学习的训练数据                   | Dataset对象（1w条样本）       |
| eval_dataset        | 监控模型性能的验证数据               | Dataset对象（2k条样本）       |
| compute_metrics     | 定义评估指标计算方法                 | 返回{accuracy: 0.85}的函数    |
| data_collator       | 自定义数据打包逻辑（如动态填充）      | DataCollatorWithPadding       |
"""

```

    Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.





    '\n| 参数                | 作用                                 | 典型值示例                     |\n|---------------------|--------------------------------------|-------------------------------|\n| model               | 定义模型架构                         | BertForSequenceClassification |\n| args                | 控制训练超参数                       | TrainingArguments实例         |\n| train_dataset       | 模型学习的训练数据                   | Dataset对象（1w条样本）       |\n| eval_dataset        | 监控模型性能的验证数据               | Dataset对象（2k条样本）       |\n| compute_metrics     | 定义评估指标计算方法                 | 返回{accuracy: 0.85}的函数    |\n| data_collator       | 自定义数据打包逻辑（如动态填充）      | DataCollatorWithPadding       |\n'



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
      [189/189 05:38, Epoch 3/3]
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
      <td>1.449800</td>
      <td>1.203277</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.031600</td>
      <td>0.998879</td>
      <td>0.576000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.765300</td>
      <td>0.973219</td>
      <td>0.590000</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=189, training_loss=1.1220053264072962, metrics={'train_runtime': 341.0837, 'train_samples_per_second': 8.795, 'train_steps_per_second': 0.554, 'total_flos': 789354427392000.0, 'train_loss': 1.1220053264072962, 'epoch': 3.0})




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






    {'eval_loss': 1.0240269899368286,
     'eval_accuracy': 0.49,
     'eval_runtime': 2.9923,
     'eval_samples_per_second': 33.419,
     'eval_steps_per_second': 4.344,
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
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 👈 必须放在所有import之前
```


```python
# 加载完整数据集（已下载时自动使用缓存）
from datasets import load_dataset
dataset = load_dataset("yelp_review_full")
```


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
dataset["train"][666]
```




    {'label': 2,
     'text': 'Just ate there, right next to GameStop & Google, has 3 small booths, & ordered the pepper steak w/ onion ($10.95). Food is fast fresh & hot, but mine had too much onion & not enough steak. At the end of the meal I was just eating onions with rice, though I hear this is healthy for you. Counter lady was cordial, but didn\'t reply when customers told her, \\"Have a nice day\\" #awkward. I know that English isn\'t her first language but she needs to catch on that people are wishing her well. Wasn\'t stuffed full either despite having eaten a large plate (I usually get this feeling eating Asian). This is basically a nice place to go for lunch that won\'t ruin your appetite for dinner. (Side note: Food is very clean. Brushed my teeth an hour before w/ Tom\'s of Maine fluoride-free peppermint & still had minty fresh breath an hour after eating)'}




```python
# 预处理数据
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

    /root/miniconda3/envs/peft/lib/python3.10/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(



    Map:   0%|          | 0/50000 [00:00<?, ? examples/s]



```python
print("可用划分:", list(tokenized_datasets.keys()))
```

    可用划分: ['train', 'test']



```python
## 微调训练配置
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
### 训练超参数（TrainingArguments）
from transformers import TrainingArguments

model_dir = "models/bert-base-cased-finetune-yelp-full"

# # logging_steps 默认值为500，根据我们的训练数据和步长，将其设置为100
# training_args = TrainingArguments(output_dir=model_dir,
#                                   per_device_train_batch_size=16,
#                                   num_train_epochs=5,
#                                   logging_steps=100)
```


```python
### 训练过程中的指标评估（Evaluate)
import numpy as np
import evaluate

metric = evaluate.load("./accuracy.py")
```


```python
# 调用 `compute` 函数来计算预测的准确率。
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```


```python
from tensorboard import version
print("TensorBoard 版本:", version.VERSION)
```

    TensorBoard 版本: 2.19.0



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
#### 训练过程指标监控
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    # 输出目录：保存模型和日志的核心路径
    output_dir=model_dir,  

    report_to="tensorboard",          # 启用TensorBoard
    logging_dir="models/bert-base-cased-finetune-yelp-full/runs",  # 明确日志路径

    # 维持工作进程
    dataloader_persistent_workers=True,
    
    # 评估策略：每个epoch结束后在验证集评估（更耗时但结果准确）
    evaluation_strategy="epoch",  
    
    # 批次配置：物理batch_size=8，通过2次梯度累积等效于16
    per_device_train_batch_size=12,     # 适合T4等中等显存GPU
    gradient_accumulation_steps=2,     # 累计2个batch的梯度再更新参数，等效batch_size=24
    
    # 训练轮次：3轮在650k数据下可能略少（推荐5-10轮）
    num_train_epochs=8,  
    
    # 日志记录：每100步打印日志（约每100*8=800样本记录一次）
    logging_steps=200,  
    
    # 混合精度：开启FP16训练（需GPU支持）
    fp16=True,
    learning_rate=2e-5,          # 重要！BERT微调黄金学习率 # 从3e-5→2e-5，更稳定收敛
    weight_decay=0.05,           # 防止过拟合 # 从0.01→0.05，增强正则化
    warmup_ratio=0.1,            # 前10%步数用于学习率预热
    save_strategy="epoch",       # 每个epoch保存检查点
    load_best_model_at_end=True, # 训练结束加载最佳模型
    metric_for_best_model="accuracy",
    lr_scheduler_type="cosine",         # 新增：余弦退火调度
    # CPU并行优化配置（重点调整部分）
    dataloader_num_workers=4        # ← 根据8核设置为4（最佳实践：核心数的50%）
)
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
    dataloader_num_workers=4,
    dataloader_persistent_workers=True,
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
    do_eval=True,
    do_predict=False,
    do_train=False,
    eval_accumulation_steps=None,
    eval_delay=0,
    eval_steps=None,
    evaluation_strategy=epoch,
    fp16=True,
    fp16_backend=auto,
    fp16_full_eval=False,
    fp16_opt_level=O1,
    fsdp=[],
    fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},
    fsdp_min_num_params=0,
    fsdp_transformer_layer_cls_to_wrap=None,
    full_determinism=False,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs=None,
    greater_is_better=True,
    group_by_length=False,
    half_precision_backend=auto,
    hub_always_push=False,
    hub_model_id=None,
    hub_private_repo=False,
    hub_strategy=every_save,
    hub_token=<HUB_TOKEN>,
    ignore_data_skip=False,
    include_inputs_for_metrics=False,
    include_num_input_tokens_seen=False,
    include_tokens_per_second=False,
    jit_mode_eval=False,
    label_names=None,
    label_smoothing_factor=0.0,
    learning_rate=2e-05,
    length_column_name=length,
    load_best_model_at_end=True,
    local_rank=0,
    log_level=passive,
    log_level_replica=warning,
    log_on_each_node=True,
    logging_dir=models/bert-base-cased-finetune-yelp-full/runs,
    logging_first_step=False,
    logging_nan_inf_filter=True,
    logging_steps=200,
    logging_strategy=steps,
    lr_scheduler_kwargs={},
    lr_scheduler_type=cosine,
    max_grad_norm=1.0,
    max_steps=-1,
    metric_for_best_model=accuracy,
    mp_parameters=,
    neftune_noise_alpha=None,
    no_cuda=False,
    num_train_epochs=8,
    optim=adamw_torch,
    optim_args=None,
    output_dir=models/bert-base-cased-finetune-yelp-full,
    overwrite_output_dir=False,
    past_index=-1,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=12,
    prediction_loss_only=False,
    push_to_hub=False,
    push_to_hub_model_id=None,
    push_to_hub_organization=None,
    push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
    ray_scope=last,
    remove_unused_columns=True,
    report_to=['tensorboard'],
    resume_from_checkpoint=None,
    run_name=models/bert-base-cased-finetune-yelp-full,
    save_on_each_node=False,
    save_only_model=False,
    save_safetensors=True,
    save_steps=500,
    save_strategy=epoch,
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
    warmup_ratio=0.1,
    warmup_steps=0,
    weight_decay=0.05,
    )



```python
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # ← 新增这行

## 开始训练
### 实例化训练器（Trainer）
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)
```

    Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.



```python
trainer.train(resume_from_checkpoint=True)
```



    <div>

      <progress value='216664' max='216664' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [216664/216664 32:11:53, Epoch 7/8]
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
      <td>2</td>
      <td>0.633500</td>
      <td>0.731120</td>
      <td>0.686660</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.466700</td>
      <td>0.849862</td>
      <td>0.679700</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.273800</td>
      <td>1.191260</td>
      <td>0.671200</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.208900</td>
      <td>1.318907</td>
      <td>0.669620</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=216664, training_loss=0.2978812934675971, metrics={'train_runtime': 115915.1343, 'train_samples_per_second': 44.86, 'train_steps_per_second': 1.869, 'total_flos': 1.2306866400800532e+18, 'train_loss': 0.2978812934675971, 'epoch': 8.0})




```python
# 1. 创建小型测试集
small_test_dataset = tokenized_datasets["test"].shuffle(seed=64).select(range(100))
# 2. 使用小型测试集评估
small_results = trainer.evaluate(small_test_dataset)
print("\n📊 评估结果详情：")
for key, value in small_results.items():
    if isinstance(value, float):
        print(f"{key:25} → {value:.4f}")  # 浮点数保留4位小数
    else:
        print(f"{key:25} → {value}")
```



<div>

  <progress value='1860' max='13' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [13/13 02:13]
</div>



    
    📊 评估结果详情：
    eval_loss                 → 0.8131
    eval_accuracy             → 0.6700
    eval_runtime              → 0.9124
    eval_samples_per_second   → 109.5980
    eval_steps_per_second     → 14.2480
    epoch                     → 8.0000



```python
# 使用完整测试集评估
full_results = trainer.evaluate(tokenized_datasets["test"])
print("\n📊 评估结果详情：")
for key, value in full_results.items():
    if isinstance(value, float):
        print(f"{key:25} → {value:.4f}")  # 浮点数保留4位小数
    else:
        print(f"{key:25} → {value}")
```


```python
### 保存模型和训练状态
trainer.save_model(model_dir)
trainer.save_state()
```

## 中断训练并清理显存


```python
# 在中断后立即执行
trainer.save_model("manual_checkpoint")  # 保存模型权重
trainer.save_state()  # 保存优化器/调度器状态
```


```python
# 复制日志文件到安全位置
cp -r models/bert-base-cased-finetune-yelp-full/runs models/bert-base-cased-finetune-yelp-full/backup/

```


      Cell In[16], line 2
        cp -r models/bert-base-cased-finetune-yelp-full/runs models/bert-base-cased-finetune-yelp-full/backup/
              ^
    SyntaxError: invalid syntax




```python
import torch
from IPython import display

display.clear_output(wait=True)  # 清理输出
torch.cuda.empty_cache()  # 清空显存
```


```python
# 修改参数防止恢复后死锁
training_args.dataloader_persistent_workers = False
```


```python

```
