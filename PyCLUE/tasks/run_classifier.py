# -*- coding: utf-8 -*-
# @Author: Liu Shaoweihua
# @Date:   2019-12-04


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from ..utils.classifier_utils.core import run_classifier, TaskConfigs, UserConfigs, ClassificationProcessor, PRETRAINED_LM_DICT
from ..utils.classifier_utils.core import default_configs as configs
from ..utils.file_utils import wget, unzip, rm, mkdir, rmdir, mv
from ..utils.configs.data_configs import DATA_URLS, DATA_PROCESSORS
from ..utils.configs.model_configs import PRETRAINED_LM_URLS


_CWD = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(_CWD, "../datasets"))
PRETRAINED_LM_DIR = os.path.abspath(os.path.join(_CWD, "../pretrained_lm"))


def clue_tasks(configs):
    ##########################################################################################################
    # download and unzip dataset and pretrained language model
    ##########################################################################################################
    configs = TaskConfigs(configs)
    if configs.task_name not in DATA_URLS:
        raise ValueError(
            "Not support task: %s" % configs.task_name)
    if configs.pretrained_lm_name not in PRETRAINED_LM_URLS:
        raise ValueError(
            "Not support pretrained language model: %s" % configs.pretrained_lm_name)
    processor = DATA_PROCESSORS.get(configs.task_name)

    if not os.path.exists(DATA_DIR):
        mkdir(DATA_DIR)
    if not os.path.exists(PRETRAINED_LM_DIR):
        mkdir(PRETRAINED_LM_DIR)
        
    data_dir = os.path.join(DATA_DIR, configs.task_name)
    pretrained_lm_dir = os.path.join(PRETRAINED_LM_DIR, configs.pretrained_lm_name)

    if not os.path.exists(data_dir):
        data_zip = wget(
            url=DATA_URLS.get(configs.task_name), 
            save_path=DATA_DIR, 
            rename=configs.task_name+".zip")
        unzip(file_path=data_zip)
        rm(data_zip)
        if not os.path.exists(data_dir):
            mkdir(data_dir)
            for item in os.listdir(DATA_DIR):
                if "train" in item:
                    mv(os.path.join(DATA_DIR, item), os.path.join(data_dir, item))
                if "test" in item:
                    mv(os.path.join(DATA_DIR, item), os.path.join(data_dir, item))
                if "dev" in item:
                    mv(os.path.join(DATA_DIR, item), os.path.join(data_dir, item))
                if "label" in item:
                    mv(os.path.join(DATA_DIR, item), os.path.join(data_dir, item))
        print("[saved]  data saved at: %s"
              % data_dir)
    else:
        print("[exists] data already exists: %s" 
              % data_dir)
    
    if not os.path.exists(pretrained_lm_dir):
        mkdir(pretrained_lm_dir)
        pretrained_lm_zip = wget(
            url=PRETRAINED_LM_URLS.get(configs.pretrained_lm_name), 
            save_path=pretrained_lm_dir, 
            rename=configs.pretrained_lm_name+".zip")
        unzip(file_path=pretrained_lm_zip)
        print("[saved]  pretrained language model saved at: %s"
              % os.path.join(pretrained_lm_dir, PRETRAINED_LM_DICT.get(configs.pretrained_lm_name)))
        rm(pretrained_lm_zip)
    else:
        print("[exists] pretrained language model already exists: %s" 
              % pretrained_lm_dir)
        
    ##########################################################################################################
    # run classifier
    ##########################################################################################################
    if not os.path.exists(configs.output_dir):
        os.makedirs(configs.output_dir)
    result_res = run_classifier(processor, configs)
    return result_res
    
    
def user_tasks(configs):
    ##########################################################################################################
    # download and unzip dataset and pretrained language model
    ##########################################################################################################
    configs = UserConfigs(configs)
    processor = ClassificationProcessor(configs.labels,
                                        configs.label_column,
                                        configs.text_a_column,
                                        configs.text_b_column,
                                        configs.ignore_header,
                                        configs.min_seq_length,
                                        configs.file_type,
                                        configs.delimiter)
    if configs.pretrained_lm_name != "user_defined_pretrained_lm":
        if configs.pretrained_lm_name not in PRETRAINED_LM_URLS:
            raise ValueError(
                "Not support pretrained language model: %s" % configs.pretrained_lm_name)
        if not os.path.exists(PRETRAINED_LM_DIR):
            mkdir(PRETRAINED_LM_DIR)
        pretrained_lm_dir = os.path.join(PRETRAINED_LM_DIR, configs.pretrained_lm_name)
        if not os.path.exists(pretrained_lm_dir):
            mkdir(pretrained_lm_dir)
            pretrained_lm_zip = wget(
                url=PRETRAINED_LM_URLS.get(configs.pretrained_lm_name), 
                save_path=pretrained_lm_dir, 
                rename=configs.pretrained_lm_name+".zip")
            unzip(file_path=pretrained_lm_zip)
            print("[saved]  pretrained language model saved at: %s"
                  % os.path.exists(os.path.join(pretrained_lm_dir, PRETRAINED_LM_DICT.get(configs.pretrained_lm_name))))
            rm(pretrained_lm_zip)
        else:
            print("[exists] pretrained language model already exists: %s" 
                  % pretrained_lm_dir)
    else:
        # TODO: should consider some other cases
        if "albert" in configs.init_checkpoint.lower():
            configs.pretrained_lm_name = "albert"
        else:
            configs.pretrained_lm_name = "bert"
    if not os.path.exists(configs.output_dir):
        os.makedirs(configs.output_dir)
    result_res = run_classifier(processor, configs)
    return result_res