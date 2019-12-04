import os
from PyCLUE.tasks.run_classifier import clue_tasks, configs

# default configs: see PyCLUE.utils.classifier_utils.core
# below are some necessary paramters required in running this task

# task_name:
#     Support: 
#         chineseGLUE: bq, xnli, lcqmc, inews, thucnews, 
#         CLUE: afqmc, cmnli, copa, csl, iflytek, tnews, wsc
configs["task_name"] = "bq"

# pretrained_lm_name: 
#     If None, should assign `vocab_file`, `bert_config_file`, `init_checkpoint`.
#     Or you can choose the following models:
#         bert, bert_wwm_ext, albert_xlarge, albert_large, albert_base, albert_base_ext, 
#         albert_small, albert_tiny, roberta, roberta_wwm_ext, roberta_wwm_ext_large
configs["pretrained_lm_name"] = "bert"

# actions
configs["do_train"] = True
configs["do_eval"] = True
configs["do_predict"] = True

# train parameters
configs["max_seq_length"] = 128
configs["train_batch_size"] = 32
configs["learning_rate"] = 2e-5
configs["warmup_proportion"] = 0.1
configs["num_train_epochs"] = 3.0

# tpu configs
configs["use_tpu"] = True
configs["tpu_name"] = "grpc://10.1.101.2:8470"
configs["num_tpu_cores"] = 8


if __name__ == "__main__":
    clue_tasks(configs)
