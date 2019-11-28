from ..classifier_utils.core import ClassificationProcessor


DATA_URLS = {
    "bq": "https://github.com/liushaoweihua/chineseGLUE_datasets/raw/master/classification/bq.zip",
    "xnli": "https://github.com/liushaoweihua/chineseGLUE_datasets/raw/master/classification/xnli.zip",
    "lcqmc": "https://github.com/liushaoweihua/chineseGLUE_datasets/raw/master/classification/lcqmc.zip",
    "inews": "https://github.com/liushaoweihua/chineseGLUE_datasets/raw/master/classification/inews.zip",
    "iflytek": "https://github.com/liushaoweihua/chineseGLUE_datasets/raw/master/classification/iflytek.zip",
    "thucnews": "https://github.com/liushaoweihua/chineseGLUE_datasets/raw/master/classification/thucnews.zip",
    "tnews": "https://github.com/liushaoweihua/chineseGLUE_datasets/raw/master/classification/tnews.zip",
    "jdcomment": "https://github.com/liushaoweihua/chineseGLUE_datasets/raw/master/classification/jdcomment.zip",
    "mrpc": "https://github.com/liushaoweihua/chineseGLUE_datasets/raw/master/classification/mrpc.zip",
    "cola": "https://github.com/liushaoweihua/chineseGLUE_datasets/raw/master/classification/cola.zip"
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
