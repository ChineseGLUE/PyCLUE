import os
from PyCLUE.tasks.run_classifier import user_tasks, configs

# assign GPU devices or CPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# default configs: see PyCLUE.utils.classifier_utils.core
# below are some necessary paramters required in running this task

# task_name: default is "user_defined_task"
configs["task_name"] = "" 

# pretrained_lm_name: 
#     If None, should assign `vocab_file`, `bert_config_file`, `init_checkpoint`.
#     Or you can choose the following models:
#         bert, bert_wwm_ext, albert_xlarge, albert_large, albert_base, albert_base_ext, 
#         albert_small, albert_tiny, roberta, roberta_wwm_ext, roberta_wwm_ext_large
configs["pretrained_lm_name"] = None

# actions
configs["do_train"] = True
configs["do_eval"] = True
configs["do_predict"] = True

# data_dir: your own data path
#     If `do_train` = True, should contain at least train.txt
#     If `do_eval` = True, should contain at least dev.txt
#     If `do_predict` = True, should contain at least test.txt
configs["data_dir"] = ""

# data configs:
#     below are some examples
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
# ignore_header:
#     If to drop the first line of each file.
configs["ignore_header"] = True
# min_seq_length:
#     If to drop sequence that has length less than `min_seq_length`
configs["min_seq_length"] = 3
# file_type:
#     train, dev, test file type, can be "txt" or "tsv"
configs["file_type"] = "txt"

# output_dir: save trained model, evaluation results and tf_records data
configs["output_dir"] = ""

# your pretrained language model components
#     If `pretrained_lm_name` is not None, these components will auto installed.
configs["vocab_file"] = "vocab.txt"
configs["bert_config_file"] = "XXX_config.json"
configs["init_checkpoint"] = "XXX_model.ckpt"

configs["max_seq_length"] = 128
configs["train_batch_size"] = 32
configs["learning_rate"] = 2e-5
configs["warmup_proportion"] = 0.1
configs["num_train_epochs"] = 3.0


if __name__ == "__main__":
    user_tasks(configs)
