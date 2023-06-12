## NLP-YES-NO-IDK
### 1. Installing requirments:
~~~
pip install transformers
~~~
### 2. Download the data
The dataset for NLI task can download [here](https://drive.google.com/drive/folders/17FxNaIOj9RvkpD5ZS_PqiAQZlZxwuSiq)

There are a couple different datasets for NLI tasks so here are brief overviews of each:
- SICK-E: Unbiased dataset with 10k training instances.
- MNLI: Biased dataset with 433k training instances.
- SNLI: Biased dataset with 570k training instances.
## Code for AUTOPROMPT:
### Quick Overview of Templates
A prompt is constructed by mapping things like the original input and trigger tokens to a template that looks something like

`[CLS] {prem} [P] [T] [T] [T] {hyp}. [SEP]`

The example above is a template for generating NLI prompts with 3 trigger tokens where `{prem}` and `{hyp}` are placeholders for the subject NLI datasets. `[P]` denotes the placement of a special `[MASK]` token that will be used to "fill-in-the-blank" by the language model. Each trigger token in the set of trigger tokens that are shared across all prompts is denoted by `[T]`.

Depending on the language model (i.e. BERT or RoBERTa) you choose to generate prompts, the special tokens will be different. For BERT, stick `[CLS]` and `[SEP]` to each end of the template. For RoBERTa, use `<s>` and `</s>` instead.

```
python  -m autoprompt.create_trigger  --train SICK_TRAIN_ALL_S.tsv --dev SICK_DEV_ALL_S.tsv --template '[CLS] {sentence_A} [P] [T] [T] [T] {sentence_B} [SEP]'  --label-map '{"ENTAILMENT": ["bend", "influences", "##imus"], "CONTRADICTION": ["none", "Nobody", "neither"], "NEUTRAL": ["bend", "influences", "##imus", "none", "Nobody", "neither"]}' --bsz 100  --model-name bert-base
```
### Label Token Selections
```
python -m autoprompt.label_search --train ../data/SICK-E-balanced/3-balance/SICK_TRAIN_ALL_S.tsv --template '[CLS] {sentence_A} [P] [T] [T] [T] {sentence_B} [SEP]' --label-map '{"ENTAILMENT": 0, "CONTRADICTION": 1, "NEUTRAL": 2}' --iters 50 --model-name bert_base
```
## Code for prompt-based Fine-tuning:
<!-- ### Generation of Templates:
~~~
python -m prompt_fine_tuning.create_template
~~~
### Label Token Selection:
Similar to the process of automatic template search, we generate candidate label word mappings by running:
~~~
python -m prompt_fine_tuning.label_search
~~~ -->
### Run Fine-tuning Model:
To carry out experiments with multiple data splits, you can use the following codes:
~~~
python -m prompt_fine_tuning.prompt_based_fine_tuning
~~~
With the change of split data seed, we can get a more accurate results by caculating the average precision from each split dataset.
## Bibliography
```
@inproceedings{autoprompt:emnlp20,
  author = {Taylor Shin and Yasaman Razeghi and Robert L. Logan IV and Eric Wallace and Sameer Singh},
  title = { {AutoPrompt}: Eliciting Knowledge from Language Models with Automatically Generated Prompts },
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2020}
}

@inproceedings{gao2021making,
   title={Making Pre-trained Language Models Better Few-shot Learners},
   author={Gao, Tianyu and Fisch, Adam and Chen, Danqi},
   booktitle={Association for Computational Linguistics (ACL)},
   year={2021}
}
```
