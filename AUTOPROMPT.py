import time
import csv
import argparse
import json
import logging
from pathlib import Path
import random

import numpy as np
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import transformers
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)
csv.field_size_limit(500 * 1024 * 1024)

class argument(object):
    def __init__(self):
        self.train = '/content/drive/MyDrive/BERT_QA/model/DATA/3-balance/SICK_TRAIN_ALL_S.tsv'
        self.dev = '/content/drive/MyDrive/BERT_QA/model/DATA/3-balance/SICK_DEV_ALL_S.tsv'
        self.test = '/content/drive/MyDrive/BERT_QA/model/DATA/3-balance/SICK_TEST_ALL_S.tsv'
        self.template = '[CLS] {sentence_A} [P] [T] [T] [T] {sentence_B} [SEP]'
        self.label_map = '{"ENTAILMENT": ["bend", "influences", "##imus"], "CONTRADICTION": ["none", "Nobody", "neither"], "NEUTRAL": ["bend", "influences", "##imus", "none", "Nobody", "neither"]}'
        self.label_map_2_class = '{"ENTAILMENT": ["bend", "influences", "##imus"], "CONTRADICTION": ["none", "Nobody", "neither"]}'
        # self.label_map = '{"ENTAILMENT": ["Still", "Absolutely", "YES"], "CONTRADICTION": ["Later", "Worse", "neither"], "NEUTRAL": ["Still", "Absolutely", "YES", "Later", "Worse", "neither"]}'
        # self.label_map_2_class = '{"ENTAILMENT": ["Still", "Absolutely", "YES"], "CONTRADICTION": ["Later", "Worse", "neither"]}'
        self.svm_save_path = '/content/drive/MyDrive/BERT_QA/autotemplate/svm_save.kpl'
        self.label_field = 'label'
        self.bsz = 100
        self.eval_size = 64
        self.iters = 100
        self.accumulation_steps = 30
        self.model_name = 'bert-base-cased'
        self.seed = 0
        self.limit = None
        self.num_cand = 20
        self.sentence_size = 512


def pad_squeeze_sequence(sequence, *args, **kwargs):
    """Squeezes fake batch dimension added by tokenizer before padding sequence."""
    return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)


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



class Collator:
    """
    Collates transformer outputs.
    """
    def __init__(self, pad_token_id=0):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        # Separate the list of inputs and labels
        model_inputs, labels, ori_label = list(zip(*features))
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
        ori_label = list(ori_label)
        return padded_inputs, labels, ori_label


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
            add_special_tokens=False):
        if not hasattr(tokenizer, 'predict_token') or \
           not hasattr(tokenizer, 'trigger_token'):
            raise ValueError(
                'Tokenizer missing special trigger and predict tokens in vocab.'
                'Use `add_special_tokens` to add them.'
            )
        self._template = template
        self._config = config
        self._tokenizer = tokenizer
        self._label_field = label_field
        self._label_map = label_map
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
        #   format_kwargs["sentence1"], format_kwargs["sentence2"] = _truncate_seq_pair(format_kwargs["sentence1"], format_kwargs["sentence2"], 512-10)
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
            label_encoded = self._label_map[label]
        label_id = encode_label(
            tokenizer=self._tokenizer,
            label=label_encoded
        )

        return model_inputs, label_id, label


def add_task_specific_tokens(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[T]', '[P]', '[Y]']
    })
    tokenizer.trigger_token = '[T]'
    tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
    tokenizer.predict_token = '[P]'
    tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids('[P]')


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
              tokens_b = tokens_b[:-1]
        return tokens_a, tokens_b


def load_json(fname):
    with open(fname, 'r') as f:
        for line in f:
            yield json.loads(line)


def load_tsv(fname):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
          yield row


def load_trigger_dataset(fname, templatizer, limit=None):
    loader = load_tsv(fname)
    instances = []
    
    for x in loader:
        try:
            model_inputs, label_id, ori_label = templatizer(x)
        except ValueError as e:
            logger.warning('Encountered error "%s" when processing "%s".  Skipping.', e, x)
            continue
        else:
            instances.append((model_inputs, label_id, ori_label))
            
    if limit:
        return random.sample(instances, limit)
    else:
        return instances


def load_trigger_dataset_sep(fname, templatizer, limit=None):
    """
    seperately load dataset(CONTRADICTION & ENTAILMENT : NEUTRAL)
    """
    loader = load_tsv(fname)

    instances_1_class = []
    instances_2_class = []
    
    for x in loader:
        try:
            model_inputs, label_id, ori_label = templatizer(x)
        except ValueError as e:
            logger.warning('Encountered error "%s" when processing "%s".  Skipping.', e, x)
            continue
        else:
            if ori_label == 'NEUTRAL':
              instances_1_class.append((model_inputs, label_id, ori_label))
            else:
              instances_2_class.append((model_inputs, label_id, ori_label))            
    if limit:
        return random.sample(instances_1_class, limit)
    else:
        return (instances_1_class, instances_2_class)


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """
    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        predict_mask = model_inputs.pop('predict_mask')
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        logits = self._model(**model_inputs).logits
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        return predict_logits


class AccuracyFn:
    """
    Computing the accuracy when a label is mapped to multiple tokens is difficult in the current
    framework, since the data generator only gives us the token ids. To get around this we
    compare the target logp to the logp of all labels. If target logp is greater than all (but)
    one of the label logps we know we are accurate.
    """
    def __init__(self, tokenizer, label_map, device):
        self._all_label_ids = []
        self._pred_to_label = []
        logger.info(label_map)

        for label, label_tokens in label_map.items():
            self._all_label_ids.append(encode_label(tokenizer, label_tokens).to(device))
            self._pred_to_label.append(label)
        self._pred_to_label.append('NEUTRAL')
        logger.info(self._all_label_ids)

    def __call__(self, predict_logits, ori_label, svc, mode):
        bsz = predict_logits.size(0)

        label_ids = torch.cat((self._all_label_ids[0], self._all_label_ids[1]), 1)
        label_ids = label_ids.repeat(bsz, 1)

        # Get entropy mask from predict logits
        entropy = information_entropy(predict_logits, label_ids)

        entropy = entropy.tolist()
        entropy_char = []
        for i in range(len(entropy)):
          entropy_value = [entropy[i]]
          entropy_char.append(entropy_value)
        if mode == 'train':
          svc, label_idx = svm_classify(svc, entropy_char, ori_label)
          label_predict = svc.predict(entropy_char)
        else:
          label_predict = svc.predict(entropy_char)

        label_predict_total = [i+1 for i in label_predict]
        unknown_predictions = torch.tensor(label_predict_total).to("cuda")        
        unknown_mask = unknown_predictions.ge(1)

        # Get total log-probability for all labels
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.min(dim=-1)

        predictions = torch.where(unknown_mask, unknown_predictions, predictions)

        label_list = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
        label_dict = {"ENTAILMENT": 0, "CONTRADICTION": 1, "NEUTRAL": 2}

        label_idx_total = [label_dict[x] for x in ori_label]

        return label_idx_total, predictions.tolist(), svc


def load_pretrained(model_name):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    add_task_specific_tokens(tokenizer)
    return config, model, tokenizer


def svm_classify(svc, entropy, label):
  """
  classifier used to classify two different classes, i.e. known and unknown class
  entropy, label: tensor
  return: fitted estimator
  """
  label_list = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
  label_dict = {"ENTAILMENT": -1, "CONTRADICTION": -1, "NEUTRAL": 1}

  label_idx = [label_dict[x] for x in label]
  svc.fit(entropy, label_idx)

  return svc, label_idx


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed) 


def get_embeddings(model, config):
    """Returns the wordpiece embedding module."""
    base_model = getattr(model, config.model_type)
    embeddings = base_model.embeddings.word_embeddings
    return embeddings


def hotflip_attack(averaged_grad,
          embedding_matrix,
          increase_loss=False,
          num_candidates=1,
):
    """Returns the top candidate replacements. calculate Vcand"""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(embedding_matrix, averaged_grad)
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
            _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)
    return top_k_ids


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


def get_loss(predict_logits, label_ids):
    """
    The larger the target logp, the greater the probability of the corresponding target, and the smaller the -target logp
    so get_loss is -target_logp and aiming at reducing this value
    """
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    # label_ids = [102, 104, 106]
    target_logp = predict_logp.gather(-1, label_ids)
    target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
    target_logp = torch.logsumexp(target_logp, dim=-1)

    return -target_logp


def information_entropy(predict_logits, label_ids):
  """
  calculate the uncertainty when dealing with dataset in class 'NEUTRAL',
  Question is, the size of entropy is different from loss about precision in predicting [P]
  
  **sum normalized to 1**
  
  entropy = -p*log(p)
  """
  # loss_entropy = torch.empty((32, 6))
  predict_value = predict_logits.gather(-1, label_ids)
  target_value = F.softmax(predict_value, dim=-1)
  target_value = target_value - 1e32 * label_ids.eq(0)

  target_value_class_0 = torch.sum(target_value[:,:3], dim=-1).view(-1,1)
  target_value_class_1 = torch.sum(target_value[:,3:], dim=-1).view(-1,1)

  target_value = torch.cat((target_value_class_0, target_value_class_1), 1)
  log_value = torch.log(target_value)
  loss_entropy = target_value * log_value
  loss_entropy = torch.sum(loss_entropy, dim=-1)

  return -loss_entropy



def run_model():
    args = argument()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading model, tokenizer, etc.')
    config, model, tokenizer = load_pretrained(args.model_name)
    model.to(device)
    embeddings = get_embeddings(model, config)
    embedding_gradient = GradientStorage(embeddings)
    predictor = PredictWrapper(model)
    
    if args.label_map is not None:
        label_map = json.loads(args.label_map)
        label_map_2_class = json.loads(args.label_map_2_class)
        print(f"Label map: {label_map}")
    else:
        label_map = None
        print('No label map')

    templatizer = TriggerTemplatizer(
        args.template,
        config,
        tokenizer,
        label_map=label_map,
        label_field=args.label_field,
        add_special_tokens=False
    )


  # Obtain the initial trigger tokens and label mapping
    trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
    best_trigger_ids = trigger_ids.clone()

  # NOTE: Accuracy can only be computed if a fixed pool of labels is given, which currently
  # requires the label map to be specified. Since producing a label map may be cumbersome (e.g.,
  # for link prediction tasks), we just use (negative) loss as the evaluation metric in these cases.
    evaluation_fn = AccuracyFn(tokenizer, label_map_2_class, device)

    print('Loading datasets')
    collator = Collator(pad_token_id=tokenizer.pad_token_id)

    train_dataset_1_class, train_dataset_2_class = load_trigger_dataset_sep(args.train, templatizer, limit=args.limit)
    train_dataset_1_class = [val for val in train_dataset_1_class for i in range(2)]

    train_loader_1_class = DataLoader(train_dataset_1_class, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    train_loader_2_class = DataLoader(train_dataset_2_class, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    dev_dataset_1_class, dev_dataset_2_class = load_trigger_dataset_sep(args.test, templatizer, limit=args.limit)

    dev_dataset_1_class = [val for val in dev_dataset_1_class for i in range(2)]
    dev_dataset_2_class.extend(dev_dataset_1_class)

    dev_loader = DataLoader(dev_dataset_2_class, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    test_dataset = load_trigger_dataset(args.dev, templatizer, limit=args.limit)

    test_loader = DataLoader(test_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    best_dev_metric = -float('inf')
  # Measure elapsed time of trigger search
    start = time.time()

    for i in range(args.iters):
      print(f'Iteration: {i}')
      svc = SVC()
      print('Accumulating Gradient')
      model.zero_grad()

      pbar = tqdm(range(args.accumulation_steps))
      train_iter_1_class = iter(train_loader_1_class)
      train_iter_2_class = iter(train_loader_2_class)
      averaged_grad = None

      # Accumulate
      for step in pbar:        
        
        # Shuttle inputs to GPU
        try:
          model_inputs_1_class, labels_1_class, ori_label_1_class = next(train_iter_1_class)
          model_inputs_2_class, labels_2_class, ori_label_2_class = next(train_iter_2_class)
        except:
          logger.warning(
              'Insufficient data for number of accumulation steps. '
              'Effective batch size will be smaller than specified.'
          )
          break
            
        model_inputs_1_class = {k: v.to(device) for k, v in model_inputs_1_class.items()}
        model_inputs_2_class = {k: v.to(device) for k, v in model_inputs_2_class.items()}
        
        labels_1_class = labels_1_class.to(device)
        labels_2_class = labels_2_class.to(device)
        
        predict_logits_1_class = predictor(model_inputs_1_class, trigger_ids)
        predict_logits_2_class = predictor(model_inputs_2_class, trigger_ids)
        loss_entropy = information_entropy(predict_logits_1_class, labels_1_class).mean()

        # loss = loss1 + loss2

        loss = get_loss(predict_logits_2_class, labels_2_class).mean() - loss_entropy
        loss.backward()

        grad = embedding_gradient.get()

        bsz, _, emb_dim = grad.size()
        selection_mask = model_inputs_1_class['trigger_mask'].unsqueeze(-1)
        selection_mask_2 = model_inputs_2_class['trigger_mask'].unsqueeze(-1)
        
        grad = torch.masked_select(grad, selection_mask)
        grad = grad.view(bsz, templatizer.num_trigger_tokens, emb_dim)

        if averaged_grad is None:
            averaged_grad = grad.sum(dim=0) / args.accumulation_steps
        else:
            averaged_grad += grad.sum(dim=0) / args.accumulation_steps
      
      print('Evaluating Candidates')
      pbar = tqdm(range(args.accumulation_steps))
      train_iter = iter(dev_loader)

      token_to_flip = random.randrange(templatizer.num_trigger_tokens)
      candidates = hotflip_attack(averaged_grad[token_to_flip],
                      embeddings.weight,
                      increase_loss=False,
                      num_candidates=args.num_cand)
      print(candidates)
      current_score = 0
      candidate_scores = torch.zeros(args.num_cand, device=device)
      eval_score = 0
      for step in pbar:
        try:
          model_inputs, labels, ori_label = next(train_iter)
        except:
          logger.warning(
                  'Insufficient data for number of accumulation steps. '
                  'Effective batch size will be smaller than specified.'
          )
          break
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
          predict_logits = predictor(model_inputs, trigger_ids)

          label_idx_total, predictions,_ = evaluation_fn(predict_logits, ori_label, svc, 'train')
          eval_metric = f1_score(label_idx_total, predictions, average='macro')

        # Update current score
        current_score = eval_metric

        # NOTE: Instead of iterating over tokens to flip we randomly change just one each
        # time so the gradients don't get stale.
        for i, candidate in enumerate(candidates):
          temp_trigger = trigger_ids.clone()
          temp_trigger[:, token_to_flip] = candidate
          with torch.no_grad():
            predict_logits = predictor(model_inputs, temp_trigger)

            label_idx_total, predictions, svc = evaluation_fn(predict_logits, ori_label, svc, 'train')
            eval_metric = f1_score(label_idx_total, predictions, average='macro')

          candidate_scores[i] = eval_metric
          if eval_score < eval_metric:
            print('Save svc classifier.')
            report = classification_report(label_idx_total, predictions)
            confusion = confusion_matrix(label_idx_total, predictions)
            print(report, '\n', confusion)
            joblib.dump(svc, args.svm_save_path)
            eval_score = eval_metric

      if (candidate_scores > current_score).any():
          print('Better trigger detected.')
          best_candidate_score = candidate_scores.max()
          best_candidate_idx = candidate_scores.argmax()
          trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
          print(f'Train metric: {best_candidate_score: 0.4f}')
      else:
          print('No improvement detected. Skipping evaluation.')
          continue

      print('Evaluating')

      i = 0
      svc = joblib.load(args.svm_save_path)
      for model_inputs, labels, ori_label in tqdm(test_loader):
          model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
          labels = labels.to(device)
          with torch.no_grad():
              predict_logits = predictor(model_inputs, trigger_ids)
          label_idx, predictions,_ = evaluation_fn(predict_logits, ori_label, svc, 'test')
          if i == 0:
            label_idx_total = label_idx
            predict_all = predictions
          else:
            predict_all.extend(predictions)
            label_idx_total.extend(label_idx)
          i += 1

      dev_metric = f1_score(label_idx_total, predict_all, average='macro')
      report = classification_report(label_idx_total, predict_all)
      confusion = confusion_matrix(label_idx_total, predict_all)
      print(report, '\n', confusion)
      print(f'Trigger tokens: {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}')
      print(f'Dev metric: {dev_metric}')

      if dev_metric > best_dev_metric:
          print('Best performance so far')
          best_trigger_ids = trigger_ids.clone()
          best_dev_metric = dev_metric
    best_trigger_tokens = tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
    print(f'Best tokens: {best_trigger_tokens}')
    print(f'Best dev metric: {best_dev_metric}')


run_model()
