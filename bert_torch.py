import os
import json
import glob
import argparse
import numpy as np
import torch
from torch import nn
import logging
import random
from tqdm import tqdm, trange
import time
from sklearn import metrics

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertModel, BertTokenizer, get_linear_schedule_with_warmup)
from torch.nn import SyncBatchNorm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BoolQExample(object):
  def __init__(self, guid, text_a, text_b=None, label=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class InputFeatures(object):
  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id

class argument(object):
  def __init__(self):
    self.model_name_or_path = 'bert-large-cased'
    self.do_train = True
    self.do_eval = True
    self.do_lower_case = True 
    self.boolq_train_data_path = '/work/scratch/dw27ciju/DATA/BoolQ_3L/train_full.json'
    self.boolq_dev_data_path = '/work/scratch/dw27ciju/DATA/BoolQ_3L/dev_full.json'
    self.boolq_test_data_path = '/work/scratch/dw27ciju/DATA/ACE-YNQA/ace_ynqa_full_complete.json'
    self.max_seq_length = 128
    self.learning_rate = 1e-6
    self.num_train_epochs = 3
    self.logging_steps = 500
    self.train_batch_size = 24
    self.eval_batch_size = 8
    self.predict_batch_size = 8
    self.output_dir = '/path/to/output-dir'
    self.config_name = ''
    self.tokenizer_name = ''
    self.gradient_accumulation_steps = 1
    self.weight_decay = 0.0
    self.adam_epsilon = 1e-6
    self.max_grad_norm = 5.0
    self.max_steps = -1
    self.warmup_proportion = 0.1
    self.save_steps = 1000
    self.eval_all_checkpoints = True
    self.no_cuda = True
    self.seed = 42

class MultiClass(nn.Module):
    """ text processed by bert model encode and get cls vector for classification
    """

    def __init__(self, model_config, num_classes=3):

      super(MultiClass, self).__init__()
      self.bert = BertModel.from_pretrained('bert-large-cased', config=model_config)
      self.num_classes = num_classes
      self.dropout = nn.Dropout(0.1)
      self.fc = nn.Linear(model_config.hidden_size, num_classes)

    def forward(self, batch_token, batch_segment, batch_attention_mask):

      out = self.bert(batch_token, attention_mask=batch_attention_mask,
                      token_type_ids=batch_segment,
                      output_hidden_states=False)
      
      pooled = out.pooler_output  # [batch, 768]
      out_fc = self.dropout(pooled)
      out_fc = self.fc(out_fc)

      return out_fc

def _create_examples(filename, set_type):
  """Creates examples for the training and dev sets."""
  examples = []
  with open(filename, encoding='utf-8') as f:
    for i, line in enumerate(f):
      data = json.loads(line)
      guid = "%s-%s" % (set_type, i)
      if data["answer"]==True:
          label = "Yes"
      elif data["answer"]==False:
          label = "No"
      else:
          label = data["answer"]
      examples.append(
          BoolQExample(
              guid=guid,
              text_a=data["passage"],
              text_b=data["question"],
              label=label))

  return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""
  label_list = ["Yes", "No", "no-answer"]
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
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
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
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

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
    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("tokens: %s" % " ".join(
          [str(x) for x in tokens]))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      logger.info(
          "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
      logger.info("label: %s (id = %d)" % (example.label, label_id))

    features.append(
        InputFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids,
                      label_id=label_id))
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

def load_examples(args, tokenizer, split, output_examples = False):

  if split == 'train':
    input_file = args.boolq_train_data_path
    set_type = "train"
  if split == 'evaluate':
    input_file = args.boolq_dev_data_path
    set_type = "dev"
  if split == 'test':
    input_file = args.boolq_test_data_path
    set_type = "test" 

  logger.info("Creating features from dataset file at %s", input_file)
  examples = _create_examples(input_file, set_type)
  features = convert_examples_to_features(examples, args.max_seq_length, tokenizer)
  
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
  all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

  if split == 'evaluate':
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  else:
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

  if output_examples:
    return dataset, examples, features
  return dataset


def evaluation(model, test_dataloader, loss_func, label2ind_dict, valid_or_test="test"):
    # model.load_state_dict(torch.load(save_path))

    model.eval()
    total_loss = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for ind, (token, mask, segment, label) in enumerate(test_dataloader):
        token = token.cuda()
        segment = segment.cuda()
        mask = mask.cuda()
        label = label.cuda()

        out = model(token, segment, mask)
        loss = loss_func(out, label)
        total_loss += loss.detach().item()

        label = label.data.cpu().numpy()
        predic = torch.max(out.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, label)
        predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if valid_or_test == "test":
        report = metrics.classification_report(labels_all, predict_all, target_names=label2ind_dict.keys(), digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        F1_avg = metrics.precision_score(labels_all, predict_all, average='micro')
        return acc, total_loss / len(test_dataloader), report, confusion, F1_avg
    return acc, total_loss / len(test_dataloader)

def train():
    label2ind_dict = {'Yes': 0, 'No': 1, 'no-answer': 2}
    args = argument()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.backends.cudnn.benchmark = True

    # load_data(os.path.join(data_dir, "cnews.train.txt"), label_dict)

    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    train_dataset = load_examples(args, tokenizer, 'train', output_examples=False) 
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=args.train_batch_size)
    print("len(train_dataset): ",len(train_dataset))
    

    test_dataset = load_examples(args, tokenizer, 'evaluate', output_examples=False)
    test_dataloader = DataLoader(test_dataset, shuffle = False, batch_size=args.predict_batch_size)
    

    model_config = BertConfig.from_pretrained('bert-large-cased')
    multi_classification_model = MultiClass(model_config, num_classes = 3)
    multi_classification_model.cuda()
    # multi_classification_model.load_state_dict(torch.load(config.save_path))

    num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
    param_optimizer = list(multi_classification_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias= True)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps * args.warmup_proportion), num_train_optimization_steps)
    loss_func = nn.CrossEntropyLoss()

    loss_total, top_acc = [], 0

    for epoch in range(num_train_epochs):
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)

        multi_classification_model.train()
        start_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="Training epoch{epoch}".format(epoch=epoch))

        for i, (token, mask, segment, label) in enumerate(tqdm_bar):
            # print(token, segment, mask, label)
            token = token.cuda()
            segment = segment.cuda()
            mask = mask.cuda()
            label = label.cuda()

            multi_classification_model.zero_grad()
            out = multi_classification_model(token, segment, mask)
            # print(out, label)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_total.append(loss.detach().item())
            label = label.data.cpu().numpy()
            predic = torch.max(out.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predic)

        acc_train = metrics.accuracy_score(labels_all, predict_all)
        # report_train = metrics.classification_report(labels_all, predict_all, target_names=label2ind_dict.keys(), digits=4)
        # confusion_train = metrics.confusion_matrix(labels_all, predict_all)
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))
        print(acc_train)
        # print(report_train, '\n', confusion_train)
        acc, loss, report, confusion, F1_avg = evaluation(multi_classification_model, test_dataloader, loss_func, label2ind_dict)
        print("Accuracy: %.4f Loss in test %.4f" % (acc, loss))
        if top_acc < acc:
            top_acc = acc
            # torch.save(multi_classification_model.state_dict(), config.save_path)
            print(report, '\n', confusion, F1_avg)

        time.sleep(1)

train()