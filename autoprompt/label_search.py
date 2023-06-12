import json
import csv
import logging
from pathlib import Path
import random
import numpy as np


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoConfig, AutoModelWithLMHead, AutoTokenizer, BertForMaskedLM)
from tqdm import tqdm

logger = logging.getLogger(__name__)
csv.field_size_limit(500 * 1024 * 1024)

class argument(object):
  def __init__(self):
    self.train = '/content/drive/MyDrive/BERT_QA/model/DATA/3-balance/SICK_TRAIN_ALL_S.tsv'
    self.dev = '/content/drive/MyDrive/BERT_QA/model/DATA/2-balance/SICK_DEV_US.tsv'
    self.template = '[CLS] {sentence_A} [P] [T] [T] [T] {sentence_B} [SEP]'
    self.label_map = '{"ENTAILMENT": 0, "CONTRADICTION": 1}'
    self.label_field = 'label'
    # self.template = '[CLS] {question} [P] [T] [T] [T] {passage} [SEP]'
    # self.label_map = '{"Yes": 0, "No": 1, "no-answer": 2}'
    # self.label_field = 'answer'
    self.bsz = 128
    self.iters = 80
    self.model_name = 'bert-base-cased'
    self.seed = 0
    self.limit = None
    self.num_cand = 10
    self.sentence_size = 512
    self.k = 50
    self.lr = 3e-4


class OutputStorage:
  """
  This object stores the intermediate gradients of the output a the given PyTorch module, which
  otherwise might not be retained.
  """
  def __init__(self, module):
    self._stored_output = None
    module.register_forward_hook(self.hook)

  def hook(self, module, input, output):
    self._stored_output = output

  def get(self):
    return self._stored_output


def encode_label(tokenizer, label, tokenize=False):
  """
  Helper function for encoding labels. Deals with the subtleties of handling multiple tokens.
  """
  if isinstance(label, str):
    if tokenize:
      # Ensure label is properly tokenized, and only retain first token
      # if it gets split into multiple tokens. TODO: Make sure this is
      # desired behavior.
      tokens = tokenizer.tokenize(label)
      if len(tokens) > 1:
        raise ValueError(f'Label "{label}" gets mapped to multiple tokens.')
      if tokens[0] == tokenizer.unk_token:
        raise ValueError(f'Label "{label}" gets mapped to unk.')
      label = tokens[0]
    encoded = torch.tensor(tokenizer.convert_tokens_to_ids([label])).unsqueeze(0)
  elif isinstance(label, list):
    encoded = torch.tensor(tokenizer.convert_tokens_to_ids(label)).unsqueeze(0)
  elif isinstance(label, int):
    encoded = torch.tensor([[label]])
  return encoded

class TriggerTemplatizer:
  """
  An object to facilitate creating transformers-friendly triggers inputs from a template.
  Parameters
  ==========
  template : str
      The template string, comprised of the following tokens:
          [T] to mark a trigger placeholder.
          [P] to mark a prediction placeholder.
          {fields} arbitrary fields instantiated from the dataset instances.
      For example a NLI template might look like:
          "[T] [T] [T] {premise} [P] {hypothesis}"
  tokenizer : PretrainedTokenizer
      A HuggingFace tokenizer. Must have special trigger and predict tokens.
  add_special_tokens : bool
      Whether or not to add special tokens when encoding. Default: False.
  """
  def __init__(self,
          template,
          config,
          tokenizer,
          label_field='label',
          label_map=None,
          tokenize_labels=False,
          add_special_tokens=False):
    if not hasattr(tokenizer, 'predict_token') or \
      not hasattr(tokenizer, 'trigger_token'):
      raise ValueError(
        'Tokenizer missing special trigger and predict tokens in vocab.'
        'Use `utils.add_special_tokens` to add them.'
      )
    self._template = template
    self._config = config
    self._tokenizer = tokenizer
    self._label_field = label_field
    self._label_map = label_map
    self._tokenize_labels = tokenize_labels
    self._add_special_tokens = add_special_tokens

  @property
  def num_trigger_tokens(self):
    return sum(token == '[T]' for token in self._template.split())

  def __call__(self, format_kwargs):
    # Format the template string
    format_kwargs = format_kwargs.copy()
    # if format_kwargs["answer"]==True:
    #   format_kwargs["answer"] = "Yes"
    # elif format_kwargs["answer"]==False:
    #   format_kwargs["answer"] = "No"
    # else:
    #   format_kwargs["answer"] = 'no-answer'
    label = format_kwargs.pop(self._label_field)
    # format_kwargs["sentence_A"], format_kwargs["sentence_B"] = _truncate_seq_pair(format_kwargs["sentence_A"], format_kwargs["sentence_B"], 512-3)
    text = self._template.format(**format_kwargs)
    if label is None:
      raise Exception(f'Bad data: {text}')

    # Have the tokenizer encode the text and process the output to:
    # - Create a trigger and predict mask
    # - Replace the predict token with a mask token
    model_inputs = self._tokenizer.encode_plus(
      text,
      add_special_tokens=self._add_special_tokens,
      return_tensors='pt'
    )
    input_ids = model_inputs['input_ids']
    trigger_mask = input_ids.eq(self._tokenizer.trigger_token_id)
    predict_mask = input_ids.eq(self._tokenizer.predict_token_id)
    input_ids[predict_mask] = self._tokenizer.mask_token_id

    model_inputs['trigger_mask'] = trigger_mask
    model_inputs['predict_mask'] = predict_mask

    # Encode the label(s)
    if self._label_map is not None:
      label = self._label_map[label]
    label_id = encode_label(
      tokenizer=self._tokenizer,
      label=label,
      tokenize=self._tokenize_labels
    )

    return model_inputs, label_id


def pad_squeeze_sequence(sequence, *args, **kwargs):
  """Squeezes fake batch dimension added by tokenizer before padding sequence."""
  return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)

class Collator:
  """
  Collates transformer outputs.
  """
  def __init__(self, pad_token_id=0):
    self._pad_token_id = pad_token_id


  def __call__(self, features):
    # Separate the list of inputs and labels
    model_inputs, labels = list(zip(*features))
    # Assume that all inputs have the same keys as the first
    proto_input = model_inputs[0]
    keys = list(proto_input.keys())
    padded_inputs = {}
    for key in keys:
      if key == 'input_ids':
        padding_value = self._pad_token_id
      else:
        padding_value = 0
      # NOTE: We need to squeeze to get rid of fake batch dim.
      sequence = [x[key] for x in model_inputs]
      padded = pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
      padded_inputs[key] = padded
    labels = pad_squeeze_sequence(labels, batch_first=True, padding_value=0)
    return padded_inputs, labels


def add_task_specific_tokens(tokenizer):
  tokenizer.add_special_tokens({
      'additional_special_tokens': ['[T]', '[P]', '[Y]']
  })
  tokenizer.trigger_token = '[T]'
  tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
  tokenizer.predict_token = '[P]'
  tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids('[P]')
  # NOTE: BERT and RoBERTa tokenizers work properly if [X] is not a special token...
  # tokenizer.lama_x = '[X]'
  # tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[X]')
  tokenizer.lama_y = '[Y]'
  tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[Y]')


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
      tokens_a = tokens_a[:-1]
    else:
      tokens_b = tokens_a[:-1]
  return tokens_a, tokens_b


def set_seed(seed: int):
  """Sets the relevant random seeds."""
  random.seed(seed)
  np.random.seed(seed)
  torch.random.manual_seed(seed)
  torch.cuda.manual_seed(seed)


def load_tsv(fname):
  with open(fname, 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
      if row['label'] == 'ENTAILMENT' or row['label'] == 'CONTRADICTION':
        yield row


def load_json(fname):
  with open(fname, 'r') as f:
    for line in f:
      yield json.loads(line)

def load_trigger_dataset(fname, templatizer, limit=None):
  loader = load_tsv(fname)
  instances = []

  for x in loader:
    try:
      model_inputs, label_id = templatizer(x)
    except ValueError as e:
      logger.warning('Encountered error "%s" when processing "%s".  Skipping.', e, x)
      continue
    else:
      instances.append((model_inputs, label_id))
  if limit:
    return random.sample(instances, limit)
  else:
    return instances

def load_pretrained(model_name):

  config = AutoConfig.from_pretrained(model_name)
  model = AutoModelWithLMHead.from_pretrained(model_name, config = config)
  model.eval()
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  add_task_specific_tokens(tokenizer)
  return config, model, tokenizer

def get_final_embeddings(model):
  if isinstance(model, BertForMaskedLM):
    return model.cls.predictions.transform
  else:
    raise NotImplementedError(f'{model} not currently supported')


def get_word_embeddings(model):
  if isinstance(model, BertForMaskedLM):
    return model.cls.predictions.decoder.weight
  else:
    raise NotImplementedError(f'{model} not currently supported')


def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
  """Replaces the trigger tokens in input_ids."""
  out = model_inputs.copy()
  input_ids = model_inputs['input_ids']
  trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
  try:
    filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
  except RuntimeError:
    filled = input_ids
  out['input_ids'] = filled
  return out

def main():
  args = argument()
  set_seed(args.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print('Loading model, tokenizer, etc.')
  config, model, tokenizer = load_pretrained(args.model_name)
  model.to(device)
  final_embeddings = get_final_embeddings(model)
  embedding_storage = OutputStorage(final_embeddings)
  word_embeddings = get_word_embeddings(model)

  label_map = json.loads(args.label_map)
  reverse_label_map = {y: x for x, y in label_map.items()}
  templatizer = TriggerTemplatizer(args.template, config, tokenizer, label_map = label_map, label_field = args.label_field, add_special_tokens = False)

  projection = torch.nn.Linear(config.hidden_size, len(label_map))
  projection.to(device)

  trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
  trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)

  print('Loading datasets')
  collator = Collator(pad_token_id=tokenizer.pad_token_id)
  train_dataset = load_trigger_dataset(args.train, templatizer)
  train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

  optimizer = torch.optim.Adam(projection.parameters(), lr=args.lr)

  scores = torch.matmul(projection.weight, word_embeddings.transpose(0, 1))
  scores = F.softmax(scores, dim=0)
  for i, row in enumerate(scores):
    _, top = row.topk(args.k)
    decoded = tokenizer.convert_ids_to_tokens(top)
    print(f"Top k for class {reverse_label_map[i]}: {', '.join(decoded)}")

  print('Training')
  for i in range(args.iters):
    epoch_loss = 0
    pbar = tqdm(train_loader)
    for model_inputs, labels in pbar:
      optimizer.zero_grad()
      model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
      labels = labels.to(device)
      trigger_mask = model_inputs.pop('trigger_mask')
      predict_mask = model_inputs.pop('predict_mask')
      model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
      with torch.no_grad():
        model(**model_inputs)
      embeddings = embedding_storage.get()
      predict_embeddings = embeddings.masked_select(predict_mask.unsqueeze(-1)).view(embeddings.size(0), -1)
      logits = projection(predict_embeddings)
      loss = F.cross_entropy(logits, labels.squeeze(-1))
      epoch_loss = epoch_loss + loss.item()
      loss.backward()
      optimizer.step()
      pbar.set_description(f'loss: {epoch_loss : 0.4f}')

    scores = torch.matmul(projection.weight, word_embeddings.transpose(0, 1))
    scores = F.softmax(scores, dim=0)
    for i, row in enumerate(scores):
      _, top = row.topk(args.k)
      decoded = tokenizer.convert_ids_to_tokens(top)
      print(f"Top k for class {reverse_label_map[i]}: {', '.join(decoded)}")
