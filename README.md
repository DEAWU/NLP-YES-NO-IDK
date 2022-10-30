# NLP-YES-NO-IDK
## Code for training and evaluation

- Installing requirments:
-       pip install transformers
        pip install tensorflow
- Downloading BERT LARGE CASED model:
https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip     
It should be unzipped to some directory $BERT_MODEL.      
The pytorch_dump_path is used to store bin file transformed from ckpt file.   
save_path is used to store model which has been finetuned by data we provide.  
- Command for training and evaluating on the dev set:      
        python bert_pytorch.py \
              --boolq_train_data_path DATA/BoolQ_3L/train_full.json\
              --boolq_dev_data_path DATA/BoolQ_3L/dev_full.json\
              --boolq_test_data_path DATA/ACE-YNQA/ace_ynqa_full_complete.json\
              --checkpoint_path $BERT_MODEL/bert_model.ckpt\
              --config_file $BERT_MODEL/bert_config.json\
              --pytorch_dump_path $BERT_MODEL/pytorch_model.bin\
              --save_path $BERT_MODEL/Test_train.pkl\
              --tokenizer_path $BERT_MODEL/
              
              
              
