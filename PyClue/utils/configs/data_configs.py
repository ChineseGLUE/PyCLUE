from ..classifier_utils.core import ClassificationProcessor


DATA_URLS = {
    "bq": "https://storage.googleapis.com/chineseglue/toolkitTasks/bq.zip",
    "xnli": "https://storage.googleapis.com/chineseglue/toolkitTasks/xnli.zip",
    "lcqmc": "https://storage.googleapis.com/chineseglue/toolkitTasks/lcqmc.zip",
    "inews": "https://storage.googleapis.com/chineseglue/toolkitTasks/inews.zip",
    "iflytek": "https://storage.googleapis.com/chineseglue/toolkitTasks/iflytek.zip",
    "thucnews": "https://storage.googleapis.com/chineseglue/toolkitTasks/thucnews.zip",
    "tnews": "https://storage.googleapis.com/chineseglue/toolkitTasks/tnews.zip",
    "jdcomment": "",
    "mrpc": "",
    "cola": ""
}


DATA_PROCESSORS = {
    "bq": ClassificationProcessor(
        labels = ["0", "1"],
        label_position = 2,
        text_a_position = 0,
        text_b_position = 1,
        file_type = "txt",
        delimiter = "_!_"
    ),
    "xnli": ClassificationProcessor(
        labels = ["0", "1", "2"],
        label_position = 2,
        text_a_position = 0,
        text_b_position = 1,
        file_type = "txt",
        delimiter = "_!_"
    ),
    "lcqmc": ClassificationProcessor(
        labels = ["0", "1"],
        label_position = 2,
        text_a_position = 0,
        text_b_position = 1,
        file_type = "txt",
        delimiter = "_!_"
    ),
    "inews": ClassificationProcessor(
        labels = ["0", "1", "2"],
        label_position = 0,
        text_a_position = 2,
        text_b_position = 3,
        file_type = "txt",
        delimiter = "_!_"
    ),
    "iflytek": ClassificationProcessor(
        labels = [str(i) for i in range(119)],
        label_position = 0,
        text_a_position = 1,
        text_b_position = None,
        file_type = "txt",
        delimiter = "_!_"
    ),
    "thucnews": ClassificationProcessor(
        labels = [str(i) for i in range(14)],
        label_position = 0,
        text_a_position = 3,
        text_b_position = None,
        file_type = "txt",
        delimiter = "_!_",
        min_seq_length = 3
    ),
    "tnews": ClassificationProcessor(
        labels = [str(100 + i) for i in range(17) if i != 5 and i != 11],
        label_position = 1,
        text_a_position = 3,
        text_b_position = None,
        file_type = "txt",
        delimiter = "_!_"
    ),
    "jdcomment": ClassificationProcessor(
        labels = ["1", "2", "3", "4", "5"],
        label_position = 0,
        text_a_position = 1,
        text_b_position = 2,
        file_type = "txt",
        delimiter = "_!_"
    ),
    "mrpc": ClassificationProcessor(
        labels = ["0", "1"],
        label_position = 0,
        text_a_position = 3,
        text_b_position = 4,
        file_type = "txt",
        delimiter = "_!_"
    ),
    "cola": ClassificationProcessor(
        labels = ["0", "1"],
        label_position = 1,
        text_a_position = 3,
        text_b_position = None,
        file_type = "txt",
        delimiter = "_!_"
    )
}
