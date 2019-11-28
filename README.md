# PyClue

Python toolkit for Chinese Language Understanding Evaluation benchmark.

中文语言理解测评基准的Python工具包，快速测评代表性数据集、基准（预训练）模型，并针对自己的数据选择合适的基准（预训练）模型进行快速应用。

## 基准（预训练）模型

**现已支持以下模型：**

1. [BERT-base](https://github.com/google-research/bert)
2. [BERT-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)
3. [albert_xlarge](https://github.com/brightmart/albert_zh)
4. [albert_large](https://github.com/brightmart/albert_zh)
5. [albert_base](https://github.com/brightmart/albert_zh)
6. [albert_base_ext](https://github.com/brightmart/albert_zh)
7. [albert_small](https://github.com/brightmart/albert_zh)
8. [albert_tiny](https://github.com/brightmart/albert_zh)
9. [roberta](https://github.com/brightmart/roberta_zh)
10. [roberta_wwm_ext](https://github.com/ymcui/Chinese-BERT-wwm)
11. [roberta_wwm_ext_large](https://github.com/ymcui/Chinese-BERT-wwm)

即将加入：

1. [XLNet_mid](https://github.com/ymcui/Chinese-PreTrained-XLNet)
2. [ERNIE_base](https://github.com/PaddlePaddle/ERNIE)

## 支持任务类型

### 分类任务

**现已支持以下数据集：**

1. **LCQMC口语化描述的语义相似度任务 Semantic Similarity Task**

   输入是两个句子，输出是0或1。其中0代表语义不相似，1代表语义相似。

   ```
   数据量：训练集(238,766)，验证集(8,802)，测试集(12,500)
       例子： 
        1.聊天室都有哪些好的 [分隔符] 聊天室哪个好 [分隔符] 1
        2.飞行员没钱买房怎么办？ [分隔符] 父母没钱买房子 [分隔符] 0
   ```

2. **XNLI语言推断任务 Natural Language Inference**

   跨语言理解的数据集，给定一个前提和假设，判断这个假设与前提是否具有蕴涵、对立、中性关系。

   ```
   数据量：训练集(392,703)，验证集(2,491)，测试集(5,011)
       例子： 
        1.从 概念 上 看 , 奶油 收入 有 两 个 基本 方面 产品 和 地理 .[分隔符] 产品 和 地理 是 什么 使 奶油 抹 霜 工作 . [分隔符] neutral
        2.我们 的 一个 号码 会 非常 详细 地 执行 你 的 指示 [分隔符] 我 团队 的 一个 成员 将 非常 精确 地 执行 你 的 命令  [分隔符] entailment
       
       原始的XNLI覆盖15种语言（含低资源语言）。我们选取其中的中文，并将做格式转换，使得非常容易进入训练和测试阶段。
   ```

3. ##### TNEWS 今日头条中文新闻（短文本）分类 Short Text Classificaiton for News

   ```
   数据量：训练集(266,000)，验证集(57,000)，测试集(57,000)
       例子：
       6552431613437805063_!_102_!_news_entertainment_!_谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分_!_佟丽娅,网络谣言,快乐大本营,李浩菲,谢娜,观众们
       每行为一条数据，以_!_分割的个字段，从前往后分别是 新闻ID，分类code，分类名称，新闻字符串（仅含标题），新闻关键词
   ```

4. **INEWS 互联网情感分析任务 Sentiment Analysis for Internet News**

   ```
   数据量：训练集(5,356)，验证集(1,000)，测试集(1,000)     
       例子：
       1_!_00005a3efe934a19adc0b69b05faeae7_!_九江办好人民满意教育_!_近3年来，九江市紧紧围绕“人本教育、公平教育、优质教育、幸福教育”的目标，努力办好人民满意教育，促进了义务教育均衡发展，农村贫困地区办学条件改善。目前，该市特色教育学校有70所 ......
       每行为一条数据，以_!_分割的个字段，从前往后分别是情感类别，数据id，新闻标题，新闻内容
   ```

5. ##### BQ 智能客服问句匹配 Question Matching for Customer Service

   该数据集是自动问答系统语料，共有120,000对句子对，并标注了句子对相似度值，取值为0或1（0表示不相似，1表示相似）。数据中存在错别字、语法不规范等问题，但更加贴近工业场景。

   ```
   数据量：训练集(100,000)，验证集(10,000)，测试集(10,000)
       例子： 
        1.我存钱还不扣的 [分隔符] 借了每天都要还利息吗 [分隔符] 0
        2.为什么我的还没有额度 [分隔符] 为啥没有额度！！ [分隔符] 1
   ```

6. ##### THUCNEWS 长文本分类 Long Text classification

   该数据集共有4万多条中文新闻长文本标注数据，共14个类别: "体育":0, "娱乐":1, "家居":2, "彩票":3, "房产":4, "教育":5, "时尚":6, "时政":7, "星座":8, "游戏":9, "社会":10, "科技":11, "股票":12, "财经":13。

   ```
   数据量：训练集(33,437)，验证集(4,180)，测试集(4,180)
       例子： 
    11_!_科技_!_493337.txt_!_爱国者A-Touch MK3533高清播放器试用　　爱国者MP5简介:　　"爱国者"北京华旗资讯，作为国内知名数码产品制>造商。1993年创立于北京中关村，是一家致力于......
    每行为一条数据，以_!_分割的个字段，从前往后分别是 类别ID，类别名称，文本ID，文本内容。
   ```

7. ##### iFLYTEK 长文本分类 Long Text classification

   该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别："打车":0,"地图导航":1,"免费WIFI":2,"租车":3,….,"女性":115,"经营":116,"收款":117,"其他":118(分别用0-118表示)。

   ```
   数据量：训练集(12,133)，验证集(2,599)，测试集(2,600)
       例子： 
   17_!_休闲益智_!_玩家需控制一只酷似神龙大侠的熊猫人在科技感十足的未来城市中穿越打拼。感觉很山寨功夫熊猫，自由度非常高，可以做很多你想做的事情......
   每行为一条数据，以_!_分割字段，从前往后分别是 类别ID，类别名称，文本内容。
   ```

更多数据集即将加入，参考：https://github.com/CLUEbenchmark/CLUE

#### 快速测评Clue数据集

测评文件存放在`PyClue/examples/classifications`中，分别为`run_clue_task.py`（GPU/CPU版本）和`run_clue_task_tpu.py`（TPU版本）。

在`run_clue_task.py`中，需要调整的参数为：

```python
# 测评任务名 task_name:
#     支持: bq, xnli, lcqmc, inews, iflytek, thucnews, tnews
configs["task_name"] = "bq"

# 基准（预训练）模型名 pretrained_lm_name: 
#     如果该参数为None，需要指定`vocab_file`, `bert_config_file`和`init_checkpoint`三参数。
#     或者可以直接指定以下基准（预训练）模型：
#         bert, bert_wwm_ext, albert_xlarge, albert_large, albert_base, albert_base_ext, 
#         albert_small, albert_tiny, roberta, roberta_wwm_ext, roberta_wwm_ext_large
configs["pretrained_lm_name"] = "bert"

# 执行内容
configs["do_train"] = True
configs["do_eval"] = True
configs["do_predict"] = True

# 训练参数调整
configs["max_seq_length"] = 128
configs["train_batch_size"] = 32
configs["learning_rate"] = 2e-5
configs["warmup_proportion"] = 0.1
configs["num_train_epochs"] = 3.0
```

在`run_clue_task_tpu.py`中，使用tpu进行模型仅需增加以下内容：

```python
# tpu内容配置
configs["use_tpu"] = True
configs["tpu_name"] = "grpc://10.1.101.2:8470"
configs["num_tpu_cores"] = 8
```

关于`configs`的详细参数，参照`PyClue/PyClue/utils/classifier_utils/core.py`的`default_configs`：

```python
default_configs = {
    "task_name": None,
    "pretrained_lm_name": None,
    "do_train": False,
    "do_eval": False,
    "do_predict": False, 
    "data_dir": None,
    "output_dir": None,
    "vocab_file": None,
    "bert_config_file": None,
    "init_checkpoint": None,
    "do_lower_case": True,
    "max_seq_length": 128,
    "train_batch_size": 32,
    "eval_batch_size": 8,
    "predict_batch_size": 8,
    "learning_rate": 5e-5,
    "num_train_epochs": 3.0,
    "warmup_proportion": 0.1,
    "save_checkpoints_steps": 1000,
    "iterations_per_loop": 1000,
    "use_tpu": False,
    "tpu_name": None,
    "tpu_zone": None,
    "gcp_project": None,
    "master": None,
    "num_tpu_cores": 8
}
```

#### 应用于自定义数据集

执行文件存放在`PyClue/examples/classifications`中，分别为`run_user_task.py`（GPU/CPU版本）和`run_user_task_tpu.py`（TPU版本）。

在`run_user_task.py`中，需要调整的参数为：

```python
# 自定义任务名 task_name: 如果为None时，默认任务名为"user_defined_task"
configs["task_name"] = "" 

# 基准（预训练）模型名 pretrained_lm_name: 
#     如果该参数为None，需要指定`vocab_file`, `bert_config_file`和`init_checkpoint`三参数。
#     或者可以直接指定以下基准（预训练）模型：
#         bert, bert_wwm_ext, albert_xlarge, albert_large, albert_base, albert_base_ext, 
#         albert_small, albert_tiny, roberta, roberta_wwm_ext, roberta_wwm_ext_large
configs["pretrained_lm_name"] = None

# 执行内容
configs["do_train"] = True
configs["do_eval"] = True
configs["do_predict"] = True

# 数据目录 data_dir:
#     如果 `do_train` = True，数据目录中应至少包含train文件
#     如果 `do_eval` = True，数据目录中应至少包含dev文件
#     如果 `do_predict` = True，数据目录中应至少包含test文件
configs["data_dir"] = ""

# 数据配置项 data configs:
#     以下是一些案例说明
configs["labels"] = ["0", "1"]
# label_position, text_a_position , text_b_position & delimiter:
#     examples_1:
#         0_!_我想要回家_!_我准备回家
#         1_!_我想要回家_!_我准备吃饭
#     >> label_position = 0, text_a_position = 1, text_b_position = 2, delimiter = "_!_"
#     examples_2:
#         0_!_我很生气
#         1_!_我很开心
#     >> label_position = 0, text_a_position = 1, text_b_position = None, delimiter = "_!_"
configs["label_position"] = 0
configs["text_a_position"] = 1
configs["text_b_position"] = 2
configs["delimiter"] = "_!_"
# 是否丢弃首行 ignore_header:
#     是否丢弃每个文件的首行，影响文件：train, dev, test
configs["ignore_header"] = True
# 最小句长 min_seq_length:
#     是否丢弃句长小于`min_seq_length`的句子
configs["min_seq_length"] = 3
# 文件类型 file_type:
#     train, dev, test的文件类型，可以为"txt"或者"tsv"
configs["file_type"] = "txt"

# 输出路径 output_dir: 
#     保存训练后的模型，验证结果和tf_records类型的数据
configs["output_dir"] = ""

# 基准（预训练）模型组件
#     如果`pretrained_lm_name`不为None，以下组件会自动下载
configs["vocab_file"] = "vocab.txt"
configs["bert_config_file"] = "XXX_config.json"
configs["init_checkpoint"] = "XXX_model.ckpt"

# 训练参数
configs["max_seq_length"] = 128
configs["train_batch_size"] = 32
configs["learning_rate"] = 2e-5
configs["warmup_proportion"] = 0.1
configs["num_train_epochs"] = 3.0
```

在`run_user_task_tpu.py`中，使用tpu进行模型仅需增加以下内容：

```python
# tpu内容配置
configs["use_tpu"] = True
configs["tpu_name"] = "grpc://10.1.101.2:8470"
configs["num_tpu_cores"] = 8
```

关于自定义数据集的`configs`参数，在`PyClue/PyClue/utils/classifier_utils/core.py`的`default_configs`的基础上新增数据项参数，用例如下：

### 阅读理解任务

即将加入。

### 命名实体识别任务

即将加入。