import json
import csv
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn

import random
import time
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import scipy.stats as stats
from tqdm import tqdm, trange

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

import transformers
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertModel, BertForSequenceClassification, load_tf_weights_in_bert, BertTokenizer, get_linear_schedule_with_warmup)
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

import logging
logger = logging.getLogger(__name__)


class argument(object):
    def __init__(self):
        self.train = '/work/scratch/dw27ciju/DATA/SNLI/512_100/train.tsv'
        self.dev = '/work/scratch/dw27ciju/DATA/SNLI/512_100/dev.tsv'
        self.test = '/work/scratch/dw27ciju/DATA/SNLI/512_100/test.tsv'
        self.label_map = {"entailment": "Yes", "contradiction": "No", "neutral": "Maybe"}
        self.label_field = 'gold_label'
        self.bsz = 8
        self.num_train_epochs = 40
        self.pretrained_model_name = 'bert-base-cased'
        self.seed = 0
        self.learning_rate = 3e-5
        self.max_seq_length = 512
        self.accumulation_steps = 6


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


class InputFeatures(object):
  def __init__(self, input_ids, input_mask, segment_ids, label_id, masked_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.masked_id = masked_id


class BertForPrompt(nn.Module):
    def __init__(self, pretrained_model_name):
        super(BertForPrompt, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.cls = BertOnlyMLMHead(self.bert.config)

    def forward(self, input_ids, attention_mask, token_type_ids, label_word_id, masked_id):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        batch_mask_reps = batched_index_select(sequence_output, 1, masked_id.unsqueeze(-1)).squeeze(1)

        prediction_scores = self.cls(batch_mask_reps)

        return prediction_scores[:, label_word_id]


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def _create_examples(filename, set_type, class_type):
  """Creates examples for the training and dev sets."""
  examples = []
  with open(filename) as f:
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    lines = []
    for line in reader:
      lines.append(line)
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, line[0])
      text_a = line[7]
      text_b = line[8]
      label = line[-1]
      if class_type == '2_class':
        if label == 'contradiction' or label == 'entailment':
          examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
      elif class_type == 'all_class':
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
      else:
        if label == 'neutral':
          examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

  return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""
  label_list = ["contradiction", "entailment", "neutral"]
  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 10)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
          tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + tokenizer.tokenize("[MASK]. This is ") + tokens_b + ["[SEP]"]

    masked_id = tokens.index('[MASK]')
        
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    # if ex_index < 5:
    #   logger.info("*** Example ***")
    #   logger.info("guid: %s" % (example.guid))
    #   logger.info("tokens: %s" % " ".join(
    #       [str(x) for x in tokens]))
    #   logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #   logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #   logger.info(
    #       "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #   logger.info("label: %s (id = %d)" % (example.label, label_id))

    features.append(
        InputFeatures(input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                masked_id=masked_id))
  return features


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


def load_examples(args, tokenizer, split, class_type):

  if split == 'train':
    input_file = args.train
    set_type = "train"
  if split == 'evaluate':
    input_file = args.dev
    set_type = "dev"
  if split == 'test':
    input_file = args.test
    set_type = "test" 

  logger.info("Creating features from dataset file at %s", input_file)
  examples = _create_examples(input_file, set_type, class_type)
  features = convert_examples_to_features(examples, args.max_seq_length, tokenizer)
  
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
  all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
  all_masked_id = torch.tensor([f.masked_id for f in features], dtype=torch.long)
  
  dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_masked_id)

  return dataset


def create_optimizer(model):
  args = argument()
  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer
                  if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
      {'params': [p for n, p in param_optimizer
                  if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias= True)
  return optimizer


def information_entropy(predictions):
  """
  calculate the uncertainty when dealing with dataset in class 'NEUTRAL',
  Question is, the size of entropy is different from loss about precision in predicting [P]
  
  **sum normalized to 1**
  
  entropy = -p*log(p)
  """
  predict_value = F.softmax(predictions, dim = -1)

  log_value = torch.log(predict_value)
  loss_entropy = predict_value * log_value
  loss_entropy = torch.sum(loss_entropy, dim=-1)

  return -loss_entropy


def svm_classify(clf, entropy, label):
  """
  classifier used to classify two different classes, i.e. known and unknown class
  entropy, label: tensor
  return: fitted estimator
  """
  entropy_lst = entropy.tolist()
  entropy_lst = stats.zscore(entropy_lst)
  entropy_lst = np.array(entropy_lst).reshape(-1,1)
  label_lst = label.tolist()
  label_lst = np.array(label_lst).flatten()
  label_dict = {0: -1, 1: -1, 2: 1}

  label_idx = [label_dict[x] for x in label_lst]
  clf.fit(entropy_lst, label_idx)
  label_predict = clf.predict(entropy_lst)

  return clf, label_predict


def train(tokenizer ,label2ind_dict):
    
  args = argument()
  # convert_tf_checkpoint_to_pytorch(args.checkpoint_path, args.config_file, args.pytorch_dump_path)
  os.environ["CUDA_VISIBLE_DEVICES"] = '0'
  torch.backends.cudnn.benchmark = True

  seed_val = 42

  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)
  
  train_dataset_2_class = load_examples(args, tokenizer, 'train', '2_class')
  train_dataloader_2_class = DataLoader(train_dataset_2_class, shuffle = True, batch_size=args.bsz)

  train_dataset_1_class = load_examples(args, tokenizer, 'train', '1_class')
  train_dataset_1_class = [val for val in train_dataset_1_class for i in range(2)]
  train_dataloader_1_class = DataLoader(train_dataset_1_class, shuffle = True, batch_size=args.bsz) 

  print("len(train_dataset_2_class): ",len(train_dataset_2_class))
  

  model = BertForPrompt(args.pretrained_model_name)
  model.cuda()

  num_train_optimization_steps = len(train_dataset_2_class) * args.num_train_epochs
  optimizer = create_optimizer(model)
  scheduler = get_linear_schedule_with_warmup(optimizer, 1227, num_train_optimization_steps)

  loss_func = nn.CrossEntropyLoss()
  loss_total = []

  contradiction_id = tokenizer.convert_tokens_to_ids("Meanwhile")
  entailment_id = tokenizer.convert_tokens_to_ids("Finally")

  label_word_id = torch.tensor([contradiction_id, entailment_id])

  for epoch in range(args.num_train_epochs):
    model.train()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    
    start_time = time.time()
    # tqdm_bar = tqdm(range(args.accumulation_steps), desc="Training epoch{epoch}".format(epoch=epoch))
    clf = SVC(kernel='rbf', C=0.8)
    i = 0
    for batch1, batch2 in zip(train_dataloader_1_class, train_dataloader_2_class):
      input_ids_1_class = batch1[0].cuda()
      attention_mask_1_class = batch1[1].cuda()
      token_type_ids_1_class = batch1[2].cuda()
      label_1_class = batch1[3].cuda()
      masked_id_1_class = batch1[4].cuda()

      input_ids_2_class = batch2[0].cuda()
      attention_mask_2_class = batch2[1].cuda()
      token_type_ids_2_class = batch2[2].cuda()
      label_2_class = batch2[3].cuda()
      masked_id_2_class = batch2[4].cuda()
      label_word_id = label_word_id.cuda()

      predictions_1_class = model(input_ids_1_class, attention_mask_1_class, token_type_ids_1_class, label_word_id, masked_id_1_class)
      predictions_2_class = model(input_ids_2_class, attention_mask_2_class, token_type_ids_2_class, label_word_id, masked_id_2_class)

      entropy_loss_1_class = information_entropy(predictions_1_class)
      entropy_loss_2_class = information_entropy(predictions_2_class)

      entropy_loss = torch.mean(entropy_loss_1_class)
      loss = loss_func(predictions_2_class, label_2_class) - 0.3 * entropy_loss

      optimizer.zero_grad()

      loss.backward()
      optimizer.step()
      scheduler.step()
      
      loss_total.append(loss.detach().item())

      if i == 0:
        total_entropy = torch.cat((entropy_loss_1_class, entropy_loss_2_class))
        total_label = torch.cat((label_1_class, label_2_class))
        total_predictions = torch.cat((predictions_1_class, predictions_2_class), 0)
      else:
        total_entropy = torch.cat((total_entropy, entropy_loss_1_class))
        total_entropy = torch.cat((total_entropy, entropy_loss_2_class))
        total_label = torch.cat((total_label, label_1_class))
        total_label = torch.cat((total_label, label_2_class))
        total_predictions = torch.cat((total_predictions, predictions_1_class), 0)
        total_predictions = torch.cat((total_predictions, predictions_2_class), 0)
      
      i += 1

    label = total_label.data.cpu().numpy()

    clf, label_predict_unknown = svm_classify(clf, total_entropy, label)
    label_predict_known = torch.max(total_predictions.data, 1)[1].cpu().numpy()

    torch.cuda.empty_cache()

    label_predict = [None] * len(label_predict_known)
    for i in range(len(label_predict_known)):
      if label_predict_unknown[i] == 1:
        label_predict[i] = 2
      else:
        label_predict[i] = label_predict_known[i]

    acc_train = metrics.accuracy_score(label, label_predict)
    report_train = metrics.classification_report(label, label_predict, target_names=label2ind_dict.keys(), digits=4)
    confusion_train = metrics.confusion_matrix(label, label_predict)
    F1_avg = metrics.precision_score(label, label_predict, average='micro')
    print('Training Metric')
    print("F1_avg: %.4f" % F1_avg)
    print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))
    print("acc_train: %.4f" % acc_train)
    print(report_train)
    print(confusion_train)
    
    logger.info('Validation Metric')
    evaluation(model, tokenizer, label2ind_dict, clf, valid_or_test='test')

    time.sleep(1)
  

def evaluation(model, tokenizer, label2ind_dict, clf, valid_or_test="test"):
    args = argument()

    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    test_dataset = load_examples(args, tokenizer, 'test', 'all_class')
    test_dataloader = DataLoader(test_dataset, shuffle = True, batch_size=args.bsz)

    contradiction_id = tokenizer.convert_tokens_to_ids("Meanwhile")
    entailment_id = tokenizer.convert_tokens_to_ids("Finally")

    label_word_id = torch.tensor([contradiction_id, entailment_id])

    with torch.no_grad():
      for ind, batch in enumerate(test_dataloader):
          input_ids = batch[0].cuda()
          attention_mask = batch[1].cuda()
          token_type_ids = batch[2].cuda()
          label = batch[3].cuda()
          masked_id = batch[4].cuda()
          label_word_id = label_word_id.cuda()

          predictions = model(input_ids, attention_mask, token_type_ids, label_word_id, masked_id)
          entropy_loss = information_entropy(predictions)

          if ind == 0:
            total_entropy = entropy_loss
            total_label = label
            total_predictions = predictions
          else:
            total_entropy = torch.cat((total_entropy, entropy_loss))
            total_label = torch.cat((total_label, label))
            total_predictions = torch.cat((total_predictions, predictions), 0)

      label = total_label.data.cpu().numpy()

      total_entropy = total_entropy.data.cpu().numpy()
      total_entropy = stats.zscore(total_entropy)
      total_entropy = total_entropy.reshape(-1,1)

      label_predict_unknown = clf.predict(total_entropy)
      label_predict_known = torch.max(total_predictions.data, 1)[1].cpu().numpy()

      label_predict = [None] * len(label_predict_known)
      for i in range(len(label_predict_known)):
        if label_predict_unknown[i] == 1:
          label_predict[i] = 2
        else:
          label_predict[i] = label_predict_known[i]

    acc = metrics.accuracy_score(label, label_predict)
    if valid_or_test == "test":
        report = metrics.classification_report(label, label_predict, target_names=label2ind_dict.keys(), digits=4)
        confusion = metrics.confusion_matrix(label, label_predict)
        print('EVAL')
        print(report, '\n', confusion)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
label2ind_dict = {'contradiction': 0, 'entailment': 1, "neutral": 2}

train(tokenizer ,label2ind_dict)