# -*- coding: utf-8 -*-
# @Author: Liu Shaoweihua
# @Date:   2019-12-04

# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import json
import collections
import numpy as np
import tensorflow as tf
from . import tokenization, modeling
from . import optimization_finetuning as optimization
from ..configs.model_configs import PRETRAINED_LM_DICT, PRETRAINED_LM_CONFIG, PRETRAINED_LM_CKPT


_CWD = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(_CWD, "../../datasets"))
OUTPUT_DIR = os.path.abspath(os.path.join(_CWD, "../../task_outputs/classifications"))
PRETRAINED_LM_DIR = os.path.abspath(os.path.join(_CWD, "../../pretrained_lm"))


__all__ = [
    "TaskConfigs", "UserConfigs", "InputExample", "PaddingInputExample", "InputFeatures", 
    "DataProcessor", "ClassificationProcessor","convert_single_example", 
    "file_based_input_fn_builder", "create_model", "model_fn_builder", "run_classifier"
]

    
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
    "num_tpu_cores": 8,
    "verbose": 0
}


class TaskConfigs(object):
    
    def __init__(self, configs):
        self.task_name = configs.get("task_name").lower() or "user_defined_task"
        self.pretrained_lm_name = configs.get("pretrained_lm_name").lower() or "user_defined_pretrained_lm"
        self.do_train = configs.get("do_train")
        self.do_eval = configs.get("do_eval")
        self.do_predict = configs.get("do_predict")
        self.data_dir = configs.get("data_dir") or os.path.join(DATA_DIR, self.task_name)
        self.output_dir = configs.get("output_dir") or os.path.join(OUTPUT_DIR, self.task_name, self.pretrained_lm_name)
        self.vocab_file = configs.get("vocab_file") or os.path.join(PRETRAINED_LM_DIR, self.pretrained_lm_name, PRETRAINED_LM_DICT.get(self.pretrained_lm_name), "vocab.txt")
        self.bert_config_file = configs.get("bert_config_file") or os.path.join(PRETRAINED_LM_DIR, self.pretrained_lm_name, PRETRAINED_LM_DICT.get(self.pretrained_lm_name), PRETRAINED_LM_CONFIG.get(self.pretrained_lm_name))
        self.init_checkpoint = configs.get("init_checkpoint") or os.path.join(PRETRAINED_LM_DIR, self.pretrained_lm_name, PRETRAINED_LM_DICT.get(self.pretrained_lm_name), PRETRAINED_LM_CKPT.get(self.pretrained_lm_name))
        self.do_lower_case = configs.get("do_lower_case")
        self.max_seq_length = configs.get("max_seq_length")
        self.train_batch_size = configs.get("train_batch_size")
        self.eval_batch_size = configs.get("eval_batch_size")
        self.predict_batch_size = configs.get("predict_batch_size")
        self.learning_rate = configs.get("learning_rate")
        self.num_train_epochs = configs.get("num_train_epochs")
        self.warmup_proportion = configs.get("warmup_proportion")
        self.save_checkpoints_steps = configs.get("save_checkpoints_steps")
        self.iterations_per_loop = configs.get("iterations_per_loop")
        self.use_tpu = configs.get("use_tpu")
        self.tpu_name = configs.get("tpu_name")
        self.tpu_zone = configs.get("tpu_zone")
        self.gcp_project = configs.get("gcp_project")
        self.master = configs.get("master")
        self.num_tpu_cores = configs.get("num_tpu_cores")
        self.verbose = configs.get("verbose")
        

class UserConfigs(TaskConfigs):
    
    def __init__(self, configs):
        self.label_column = configs.get("label_column")
        self.text_a_column = configs.get("text_a_column")
        self.text_b_column = configs.get("text_b_column")
        self.delimiter = configs.get("delimiter")
        self.ignore_header = configs.get("ignore_header")
        self.min_seq_length = configs.get("min_seq_length")
        self.file_type = configs.get("file_type")
        super().__init__(configs)



class InputExample(object):
    """A single training/test example for simple sequence classification."""
    
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data"""
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example
        
        
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()
        
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()
        
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
        
    @classmethod
    def _read_file(cls, input_file, file_type, delimiter):
        """Reads files."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(line.strip())
            if file_type == "json":
                lines = [json.loads(item) for item in lines]
            else:
                lines = [item.split(delimiter) for item in lines]
            return lines


class ClassificationProcessor(DataProcessor):
    
    def __init__(self, labels, label_column, text_a_column, text_b_column=None, ignore_header=False, min_seq_length=None, file_type="json", delimiter=None):
        self.language = "zh"
        self.labels = labels
        self.label_column = label_column
        self.text_a_column = text_a_column
        self.text_b_column = text_b_column
        self.ignore_header = ignore_header
        self.min_seq_length = min_seq_length
        self.file_type = file_type
        self.delimiter = delimiter
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train."+self.file_type), self.file_type, delimiter=self.delimiter), "train"
        )
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "dev."+self.file_type), self.file_type, delimiter=self.delimiter), "dev"
        )
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test."+self.file_type), self.file_type, delimiter=self.delimiter), "test"
        )
    
    def get_labels(self):
        """See base class."""
        return self.labels
    
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
                text_a = tokenization.convert_to_unicode(line[self.text_a_column])
                text_b = None if not self.text_b_column else tokenization.convert_to_unicode(line[self.text_b_column])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                )
            except Exception:
                print("### Error {}: {}".format(i, line))
        return examples


def convert_single_example(
    ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    
    if isinstance(example, PaddingInputExample):
        return [InputFeatures(
            input_ids=[0]*max_seq_length,
            input_mask=[0]*max_seq_length,
            segment_ids=[0]*max_seq_length,
            label_id=0,
            is_real_example=False
        )]
    
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
        
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
            
    # The convention in BERT is:
    # (a) For sequence pairs:
    # tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    # type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    # tokens:   [CLS] the dog is hairy . [SEP]
    # type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    
    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # The mask has 1 for real tokens and 0 for padding tokens. Only real 
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" %(example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
        
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True
    )
    return feature
    

def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        
        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        writer.write(tf_example.SerializeToString())
        
    writer.close()


def file_based_input_fn_builder(
    input_file, seq_length, is_training, drop_remainder
):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64)
    }
    
    def _decode_record(record, name_to_features):
        """Decodes a record to a tensorflow example."""
        examples = tf.parse_single_example(record, name_to_features)

        # tf.train.Example only supports tf.int64, but the TPU only supports tf.int32
        # so cast all tf.int64 to tf.int32
        for name in list(examples.keys()):
            t = examples[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            examples[name] = t
        return examples
    
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        
        # For training, we want a lot of parallel reading and shuffling.
        # For evaluation, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
            
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder
            )
        )
        
        return d
    
    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            

def create_model(
    model_type, bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels, use_one_hot_embeddings):
    """Create a classification model."""
    if model_type.startswith("bert") or model_type.startswith("roberta"):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )
    elif model_type.startswith("albert"):
        model = modeling.AlBertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )
    
    # In this demo, we are doing a simple classification task on the entire segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output() instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value
    
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    
    output_bias = tf.get_variable(
        "output_bias", [num_labels], 
        initializer=tf.zeros_initializer())
    
    with tf.variable_scope("loss"):
        if model_type.startswith("albert"):
            try:
                ln_type = bert_config.ln_type
            except:
                ln_type = None
            if ln_type == "preln":
                # Add by brightmart, 10-06. If it is preln, we need to add an additional
                # layer: layer normalization as suggested in paper "ON LAYER NORMALIZATION
                # IN THE TRANSFORMER ARCHITECTURE"
                print("ln_type is preln. add LN layer.")
                output = layer_norm(output_layer)
            else:
                print("ln_type is postln or other, do nothing.")
            
        if is_training:
            # I.E., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, rate=0.1)
            
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs,
                                      axis=-1)  # todo 08-29 try temp-loss
    ###############bi_tempered_logistic_loss############################################################################
    # print("##cross entropy loss is used...."); tf.logging.info("##cross entropy loss is used....")
    # t1=0.9 #t1=0.90
    # t2=1.05 #t2=1.05
    # per_example_loss=bi_tempered_logistic_loss(log_probs,one_hot_labels,t1,t2,label_smoothing=0.1,num_iters=5) # TODO label_smoothing=0.0
    # tf.logging.info("per_example_loss:"+str(per_example_loss.shape))
    ##############bi_tempered_logistic_loss#############################################################################
        
        loss = tf.reduce_mean(per_example_loss)
        
        return loss, per_example_loss, logits, probabilities
    

def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def model_fn_builder(model_type, bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params): # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %(name, features[name].shape))
            
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
            
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        total_loss, per_example_loss, logits, probabilities = create_model(
            model_type, bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, num_labels, use_one_hot_embeddings)
        
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()
            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }
            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec
    
    return model_fn


def run_classifier(processor, configs):
    if configs.verbose == 0:
        tf.logging.set_verbosity(tf.logging.ERROR)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
    
    tokenization.validate_case_matches_checkpoint(configs.do_lower_case, configs.init_checkpoint)
    
    if not configs.do_train and not configs.do_eval and not configs.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
        
    bert_config = modeling.BertConfig.from_json_file(configs.bert_config_file)
    
    if configs.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (configs.max_seq_length, bert_config.max_position_embeddings))
        
    tf.gfile.MakeDirs(configs.output_dir)
    
    task_name = configs.task_name.lower()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=configs.vocab_file, do_lower_case=configs.do_lower_case)

    tpu_cluster_resolver = None
    if configs.use_tpu and configs.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            configs.tpu_name, zone=configs.tpu_zone, project=configs.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    # Cloud TPU: Invalid TPU configuration, ensure ClusterResolver is passed to tpu.
    print("[tpu]    tpu cluster resolver:", tpu_cluster_resolver)
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=configs.master,
        model_dir=configs.output_dir,
        save_checkpoints_steps=configs.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=configs.iterations_per_loop,
            num_shards=configs.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if configs.do_train:
        train_examples = processor.get_train_examples(configs.data_dir)
        print("[train]  length of total train_examples:", len(train_examples))
        num_train_steps = int(len(train_examples) / configs.train_batch_size * configs.num_train_epochs)
        num_warmup_steps = int(num_train_steps * configs.warmup_proportion)

    model_fn = model_fn_builder(
        model_type=configs.pretrained_lm_name,
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=configs.init_checkpoint,
        learning_rate=configs.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=configs.use_tpu,
        use_one_hot_embeddings=configs.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=configs.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=configs.train_batch_size,
        eval_batch_size=configs.eval_batch_size,
        predict_batch_size=configs.predict_batch_size)

    if configs.do_train:
        train_file = os.path.join(configs.output_dir, "train.tf_record")
        train_file_exists = os.path.exists(train_file)
        print("[train]  train file exists:", train_file_exists)
        print("[train]  train file path:", train_file)
        if not train_file_exists:  # if tf_record file not exist, convert from raw text file. # TODO
            file_based_convert_examples_to_features(
                train_examples, label_list, configs.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", configs.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=configs.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if configs.do_eval:
        # dev dataset
        dev_examples = processor.get_dev_examples(configs.data_dir)
        num_actual_dev_examples = len(dev_examples)
        if configs.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(dev_examples) % configs.eval_batch_size != 0:
                dev_examples.append(PaddingInputExample())
    
        eval_file = os.path.join(configs.output_dir, "dev.tf_record")
        file_based_convert_examples_to_features(
            dev_examples, label_list, configs.max_seq_length, tokenizer, eval_file)
    
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                     len(dev_examples), num_actual_dev_examples,
                     len(dev_examples) - num_actual_dev_examples)
        tf.logging.info("  Batch size = %d", configs.eval_batch_size)
    
        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if configs.use_tpu:
            assert len(dev_examples) % configs.eval_batch_size == 0
            eval_steps = int(len(dev_examples) // configs.eval_batch_size)
    
        eval_drop_remainder = True if configs.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=configs.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
    
        #######################################################################################################################
        # evaluate all checkpoints; you can use the checkpoint with the best dev accuarcy
        steps_and_files = []
        filenames = tf.gfile.ListDirectory(configs.output_dir)
        for filename in filenames:
            if filename.endswith(".index"):
                ckpt_name = filename[:-6]
                cur_filename = os.path.join(configs.output_dir, ckpt_name)
                global_step = int(cur_filename.split("-")[-1])
                tf.logging.info("Add {} to eval list.".format(cur_filename))
                steps_and_files.append([global_step, cur_filename])
        steps_and_files = sorted(steps_and_files, key=lambda x: x[0])
    
        output_eval_file = os.path.join(configs.output_dir, "dev_results.txt")
        print("[eval]   dev result saved at:", output_eval_file)
        tf.logging.info("dev_eval_file:" + output_eval_file)
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
                result = estimator.evaluate(input_fn=eval_input_fn,
                                            steps=eval_steps, checkpoint_path=filename)
    
                tf.logging.info("***** Eval results %s *****" % (filename))
                writer.write("***** Eval results %s *****\n" % (filename))
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        #######################################################################################################################
    
        # result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        #
        # output_eval_file = os.path.join(configs.output_dir, "dev_results.txt")
        # with tf.gfile.GFile(output_eval_file, "w") as writer:
        #  tf.logging.info("***** Eval results *****")
        #  for key in sorted(result.keys()):
        #    tf.logging.info("  %s = %s", key, str(result[key]))
        #    writer.write("%s = %s\n" % (key, str(result[key])))
    
        # test dataset
        test_examples = processor.get_test_examples(configs.data_dir)
        num_actual_test_examples = len(test_examples)
        if configs.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(test_examples) % configs.eval_batch_size != 0:
                test_examples.append(PaddingInputExample())
    
        eval_file = os.path.join(configs.output_dir, "test.tf_record")
        file_based_convert_examples_to_features(
            test_examples, label_list, configs.max_seq_length, tokenizer, eval_file)
    
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                     len(test_examples), num_actual_test_examples,
                     len(test_examples) - num_actual_test_examples)
        tf.logging.info("  Batch size = %d", configs.eval_batch_size)
    
        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if configs.use_tpu:
            assert len(test_examples) % configs.eval_batch_size == 0
            eval_steps = int(len(test_examples) // configs.eval_batch_size)
    
        eval_drop_remainder = True if configs.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=configs.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
    
        #######################################################################################################################
        # evaluate all checkpoints; you can use the checkpoint with the best dev accuarcy
        steps_and_files = []
        filenames = tf.gfile.ListDirectory(configs.output_dir)
        for filename in filenames:
            if filename.endswith(".index"):
                ckpt_name = filename[:-6]
                cur_filename = os.path.join(configs.output_dir, ckpt_name)
                global_step = int(cur_filename.split("-")[-1])
                tf.logging.info("Add {} to eval list.".format(cur_filename))
                steps_and_files.append([global_step, cur_filename])
        steps_and_files = sorted(steps_and_files, key=lambda x: x[0])
    
        output_eval_file = os.path.join(configs.output_dir, "test_results.txt")
        print("[test]   test result saved at:", output_eval_file)
        tf.logging.info("test_eval_file:" + output_eval_file)
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
                result = estimator.evaluate(input_fn=eval_input_fn,
                                            steps=eval_steps, checkpoint_path=filename)
    
                tf.logging.info("***** Eval results %s *****" % (filename))
                writer.write("***** Eval results %s *****\n" % (filename))
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        #######################################################################################################################
    #
    #     #result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    #     #
    #     #output_eval_file = os.path.join(configs.output_dir, "test_results.txt")
    #     # with tf.gfile.GFile(output_eval_file, "w") as writer:
    #     #  tf.logging.info("***** Eval results *****")
    #     #  for key in sorted(result.keys()):
    #     #    tf.logging.info("  %s = %s", key, str(result[key]))
    #     #    writer.write("%s = %s\n" % (key, str(result[key])))
    
    if configs.do_predict:
        predict_examples = processor.get_test_examples(configs.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if configs.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % configs.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())
    
        predict_file = os.path.join(configs.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(
            predict_examples, label_list, configs.max_seq_length, tokenizer, predict_file)
    
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                     len(predict_examples), num_actual_predict_examples,
                     len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", configs.predict_batch_size)
    
        predict_drop_remainder = True if configs.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=configs.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)
    
        result = estimator.predict(input_fn=predict_input_fn)
    
        output_predict_file = os.path.join(configs.output_dir, "test_results.tsv")
        print("[pred]   predict result saved at:", output_predict_file)
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            pred_labels = []
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                pred_label = np.argmax([item for item in probabilities])
                pred_labels.append(pred_label)
            output_lines = []
            for pred_data, pred_label in zip(test_examples, pred_labels):
                output_lines.append({"guid": pred_data.guid, "text_a": pred_data.text_a, "text_b": pred_data.text_b, "label": processor.labels[pred_label]})
            for item in output_lines:
                writer.write(json.dumps(item, ensure_ascii=False)+"\n")
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples
        
    dev_res, test_res = "", ""
    test_outputs = []
    
    dev_res_file = os.path.join(configs.output_dir, "dev_results.txt")
    test_res_file = os.path.join(configs.output_dir, "test_results.txt")
    test_output_file = os.path.join(configs.output_dir, "test_results.tsv")
    
    if configs.do_eval:
        with open(dev_res_file, "r") as f:
            dev_res = [item.strip().split(" = ") for item in f.readlines()[-4:]]
        with open(test_res_file, "r") as f:
            test_res = [item.strip().split(" = ") for item in f.readlines()[-4:]]
    
    if configs.do_predict:
        test_outputs = output_lines
        
    result_dict = {
        "dev_res": {item[0]:item[1] for item in dev_res},
        "test_res": {item[0]:item[1] for item in test_res},
        "test_outputs": test_outputs
    }
    
    return result_dict