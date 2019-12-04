PRETRAINED_LM_DICT = {
    "bert": "chinese_L-12_H-768_A-12",
    "bert_wwm_ext": "",
    "albert_xlarge": "",
    "albert_large": "",
    "albert_base": "",
    "albert_base_ext": "",
    "albert_small": "",
    "albert_tiny": "",
    "roberta": "",
    "roberta_wwm_ext": "",
    "roberta_wwm_ext_large": ""
}

PRETRAINED_LM_CONFIG = {
    "bert": "bert_config.json",
    "bert_wwm_ext": "bert_config.json",
    "albert_xlarge": "albert_config_xlarge.json",
    "albert_large": "albert_config_large.json",
    "albert_base": "albert_config_base.json",
    "albert_base_ext": "albert_config_base.json",
    "albert_small": "albert_config_small_google.json",
    "albert_tiny": "albert_config_tiny_g.json",
    "roberta": "bert_config_large.json",
    "roberta_wwm_ext": "bert_config.json",
    "roberta_wwm_ext_large": "bert_config.json"
    
}

PRETRAINED_LM_CKPT = {
    "bert": "bert_model.ckpt",
    "bert_wwm_ext": "bert_model.ckpt",
    "albert_xlarge": "albert_model.ckpt",
    "albert_large": "albert_model.ckpt",
    "albert_base": "albert_model.ckpt",
    "albert_base_ext": "albert_model.ckpt",
    "albert_small": "albert_model.ckpt",
    "albert_tiny": "albert_model.ckpt",
    "roberta": "roberta_zh_large_model.ckpt",
    "roberta_wwm_ext": "bert_model.ckpt",
    "roberta_wwm_ext_large": "bert_model.ckpt"
}

PRETRAINED_LM_URLS = {
    "bert": "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip",
    "bert_wwm_ext": "https://storage.googleapis.com/chineseglue/pretrain_models/chinese_wwm_ext_L-12_H-768_A-12.zip",
    "albert_xlarge": "https://storage.googleapis.com/albert_zh/albert_xlarge_zh_177k.zip",
    "albert_large": "https://storage.googleapis.com/albert_zh/albert_large_zh.zip",
    "albert_base": "https://storage.googleapis.com/albert_zh/albert_base_zh.zip",
    "albert_base_ext": "https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip",
    "albert_small": "https://storage.googleapis.com/albert_zh/albert_small_zh_google.zip",
    "albert_tiny": "https://storage.googleapis.com/albert_zh/albert_tiny_zh_google.zip",
    "roberta": "https://storage.googleapis.com/chineseglue/pretrain_models/roeberta_zh_L-24_H-1024_A-16.zip",
    "roberta_wwm_ext": "https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_ext_L-12_H-768_A-12.zip",
    "roberta_wwm_ext_large": "https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip"
}