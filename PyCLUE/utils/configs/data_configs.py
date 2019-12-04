# -*- coding: utf-8 -*-
# @Author: Liu Shaoweihua
# @Date:   2019-12-04

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..classifier_utils.core import ClassificationProcessor, InputExample
from ..classifier_utils import tokenization


class CmnliProcessor(ClassificationProcessor):
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if self.ignore_header:
            lines = lines[1:]
        if self.min_seq_length:
            lines = [line for line in lines if len(line) >= self.min_seq_length]
        for i, line in enumerate(lines):
            guid = "%s-%s" %(set_type, i)
            try:
                if set_type == "train":
                    label = tokenization.convert_to_unicode(line["gold_"+self.label_column])
                elif set_type == "dev":
                    label = tokenization.convert_to_unicode(line[self.label_column])
                elif set_type == "test":
                    label = self.labels[0]
                text_a = tokenization.convert_to_unicode(line[self.text_a_column])
                text_b = None if not self.text_b_column else tokenization.convert_to_unicode(line[self.text_b_column])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                )
            except Exception:
                print("### Error {}: {}".format(i, line))
        return examples

class CslProcessor(ClassificationProcessor):
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if self.ignore_header:
            lines = lines[1:]
        if self.min_seq_length:
            lines = [line for line in lines if len(line) >= self.min_seq_length]
        for i, line in enumerate(lines):
            guid = "%s-%s" %(set_type, i)
            try:
                label = tokenization.convert_to_unicode(line[self.label_column]) if set_type != "test" else self.labels[0]
                text_a = tokenization.convert_to_unicode(" ".join(line[self.text_a_column]))
                text_b = None if not self.text_b_column else tokenization.convert_to_unicode(line[self.text_b_column])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                )
            except Exception:
                print("### Error {}: {}".format(i, line))
        return examples

    
class WscProcessor(ClassificationProcessor):
    
    def _create_examples(self, lines, set_type):
        examples = []
        if self.ignore_header:
            lines = lines[1:]
        if self.min_seq_length:
            lines = [line for line in lines if len(line) >= self.min_seq_length]
        for i, line in enumerate(lines):
            guid = "%s-%s" %(set_type, i)
            try:
                label = tokenization.convert_to_unicode(line[self.label_column]) if set_type != "test" else self.labels[0]
                text_a = tokenization.convert_to_unicode(line[self.text_a_column])
                text_b = None if not self.text_b_column else tokenization.convert_to_unicode(line[self.text_b_column])
                text_a_list = list(text_a)
                target = line["target"]
                query = target["span1_text"]
                query_idx = target["span1_index"]
                pronoun = target["span2_text"]
                pronoun_idx = target["span2_index"]
                
                assert text_a[pronoun_idx:(pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
                assert text_a[query_idx: (query_idx + len(query))] == query, "query: {}".format(query)
                
                if pronoun_idx > query_idx:
                    text_a_list.insert(query_idx, "_")
                    text_a_list.insert(query_idx + len(query) + 1, "_")
                    text_a_list.insert(pronoun_idx + 2, "[")
                    text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
                else:
                    text_a_list.insert(pronoun_idx, "[")
                    text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
                    text_a_list.insert(query_idx + 2, "_")
                    text_a_list.insert(query_idx + len(query) + 2 + 1, "_")
                    
                text_a = "".join(text_a_list)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                )
            except Exception as e:
                print("### Error {}: {}".format(i, line))
        return examples
    

class CopaProcessor(ClassificationProcessor):
    
    def __init__(self, labels, label_column, ignore_header=False, min_seq_length=None, file_type="json", delimiter=None):
        self.language = "zh"
        self.labels = labels
        self.label_column = label_column
        self.ignore_header = ignore_header
        self.min_seq_length = min_seq_length
        self.file_type = file_type
        self.delimiter = delimiter
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if self.ignore_header:
            lines = lines[1:]
        if self.min_seq_length:
            lines = [line for line in lines if len(line) >= self.min_seq_length]
        for i, line in enumerate(lines):
            i = 2*i
            guid0 = "%s-%s" %(set_type, i)
            guid1 = "%s-%s" %(set_type, i+1)
            try:
                premise = tokenization.convert_to_unicode(line["premise"])
                choice0 = tokenization.convert_to_unicode(line["choice0"])
                label0 = tokenization.convert_to_unicode(str(1 if line[self.label_column] == 0 else 0)) if set_type != "test" else self.labels[0]
                choice1 = tokenization.convert_to_unicode(line["choice1"])
                label1 = tokenization.convert_to_unicode(str(0 if line[self.label_column] == 0 else 1)) if set_type != "test" else self.labels[0]
                if line["question"] == "effect":
                    text_a0 = premise
                    text_b0 = choice0
                    text_a1 = premise
                    text_b1 = choice1
                elif line["question"] == "cause":
                    text_a0 = choice0
                    text_b0 = premise
                    text_a1 = choice1
                    text_b1 = premise
                else:
                    raise Exception
                examples.append(
                    InputExample(guid=guid0, text_a=text_a0, text_b=text_b0, label=label0)
                )
                examples.append(
                    InputExample(guid=guid1, text_a=text_a1, text_b=text_b1, label=label1)
                )
            except Exception as e:
                print("### Error {}: {}".format(i, line))
        return examples
    

DATA_URLS = {
    # chineseGLUE txt Version
    "bq": "https://storage.googleapis.com/chineseglue/toolkitTasks/bq.zip", # 
    "xnli": "https://storage.googleapis.com/chineseglue/toolkitTasks/xnli.zip",
    "lcqmc": "https://storage.googleapis.com/chineseglue/toolkitTasks/lcqmc.zip",
    "inews": "https://storage.googleapis.com/chineseglue/toolkitTasks/inews.zip",
    "thucnews": "https://storage.googleapis.com/chineseglue/toolkitTasks/thucnews.zip",
    # CLUE json Version
    "afqmc": "https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip",
    "cmnli": "https://storage.googleapis.com/cluebenchmark/tasks/cmnli_public.zip",
    "copa": "https://storage.googleapis.com/cluebenchmark/tasks/copa_public.zip",
    "csl": "https://storage.googleapis.com/cluebenchmark/tasks/csl_public.zip",
    "iflytek": "https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip",
    "tnews": "https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip",
    "wsc": "https://storage.googleapis.com/cluebenchmark/tasks/wsc_public.zip"
}


DATA_PROCESSORS = {
    # chineseGLUE txt Version
    "bq": ClassificationProcessor(
        labels = ["0", "1"],
        label_column = 2,
        text_a_column = 0,
        text_b_column = 1,
        file_type = "txt",
        delimiter = "_!_"
    ),
    "xnli": ClassificationProcessor(
        labels = ["0", "1", "2"],
        label_column = 2,
        text_a_column = 0,
        text_b_column = 1,
        file_type = "txt",
        delimiter = "_!_"
    ),
    "lcqmc": ClassificationProcessor(
        labels = ["0", "1"],
        label_column = 2,
        text_a_column = 0,
        text_b_column = 1,
        file_type = "txt",
        delimiter = "_!_"
    ),
    "inews": ClassificationProcessor(
        labels = ["0", "1", "2"],
        label_column = 0,
        text_a_column = 2,
        text_b_column = 3,
        file_type = "txt",
        delimiter = "_!_"
    ),
    "thucnews": ClassificationProcessor(
        labels = [str(i) for i in range(14)],
        label_column = 0,
        text_a_column = 3,
        text_b_column = None,
        file_type = "txt",
        delimiter = "_!_",
        min_seq_length = 3
    ),
    # CLUE json Version
    "afqmc": ClassificationProcessor(
        labels = ["0", "1"],
        label_column = "label",
        text_a_column = "sentence1",
        text_b_column = "sentence2",
        file_type = "json",
        delimiter = None
    ),
    "iflytek": ClassificationProcessor(
        labels = [str(i) for i in range(119)],
        label_column = "label",
        text_a_column = "sentence",
        text_b_column = None,
        file_type = "json",
        delimiter = None
    ),
    "tnews": ClassificationProcessor(
        labels = [str(100 + i) for i in range(17) if i != 5 and i != 11],
        label_column = "label",
        text_a_column = "sentence",
        text_b_column = None,
        file_type = "json",
        delimiter = None
    ),
    "wsc": WscProcessor(
        labels = ["true", "false"],
        label_column = "label",
        text_a_column = "text",
        text_b_column = None,
        file_type = "json",
        delimiter = None
    ),
    "copa": CopaProcessor(
        labels = ["0", "1"],
        label_column = "label",
        file_type = "json",
        delimiter = None
    ),
    "csl": CslProcessor(
        labels = ["0", "1"],
        label_column = "label",
        text_a_column = "keyword",
        text_b_column = "abst",
        file_type = "json",
        delimiter = None
    ),
    "cmnli": CmnliProcessor(
        labels = ["contradiction", "entailment", "neutral"],
        label_column = "label",
        text_a_column = "sentence1",
        text_b_column = "sentence2",
        file_type = "json",
        delimiter = None
    )
}