# NLP-YES-NO-IDK
## Code for training and evaluation

- Installing requirments:
-       pip install transformers
        pip install tensorflow
- Downloading BERT LARGE CASED model:
https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip     
It should be unzipped to some directory $BERT_MODEL.      
The **pytorch_dump_path** is used to store bin file transformed from ckpt file.   
**save_path** is used to store model which has been finetuned by data we provide.  
- Command for training and evaluating on the dev set:      
-        python bert_pytorch.py \
              --boolq_train_data_path DATA/BoolQ_3L/train_full.json\
              --boolq_dev_data_path DATA/BoolQ_3L/dev_full.json\
              --boolq_test_data_path DATA/ACE-YNQA/ace_ynqa_full_complete.json\
              --checkpoint_path $BERT_MODEL/bert_model.ckpt\
              --config_file $BERT_MODEL/bert_config.json\
              --pytorch_dump_path $BERT_MODEL/pytorch_model.bin\
              --save_path $BERT_MODEL/Test_train.pkl\
              --tokenizer_path $BERT_MODEL/
              
## Results shown by the run
len(train_dataset):  14141       
      
Training epoch0: 100%|██████████| 590/590 [17:11<00:00,  1.75s/it]
0.5154515239374867     
Epoch: 000; loss = 1.0822 cost time  1031.2603      
0.5154515239374867       
                       precision    recall  f1-score   support
  
                 Yes     0.5043    0.8825    0.6418      5874
                  No     0.0062    0.0011    0.0019      3553
           no-answer     0.6529    0.4457    0.5298      4714

            accuracy                         0.5155     14141
           macro avg     0.3878    0.4431    0.3912     14141
        weighted avg     0.4287    0.5155    0.4437     14141      
 
        [[5184    4  686]
        [3118    4  431]
        [1978  635 2101]]            
Training epoch1: 100%|██████████| 590/590 [17:22<00:00,  1.77s/it]
0.6508733470051623
Epoch: 001; loss = 0.7667 cost time  1042.6599
0.6508733470051623
              precision    recall  f1-score   support

         Yes     0.5714    0.8951    0.6975      5874
          No     0.3247    0.0141    0.0270      3553
   no-answer     0.8142    0.8265    0.8203      4714

    accuracy                         0.6509     14141
   macro avg     0.5701    0.5786    0.5149     14141
weighted avg     0.5904    0.6509    0.5700     14141
 
 [[5258   83  533]
 [3147   50  356]
 [ 797   21 3896]]                
Training epoch2: 100%|██████████| 590/590 [17:21<00:00,  1.77s/it]
0.6669259599745421
Epoch: 002; loss = 0.7071 cost time  1041.9952
0.6669259599745421
              precision    recall  f1-score   support

         Yes     0.5793    0.9198    0.7109      5874
          No     0.2718    0.0079    0.0153      3553
   no-answer     0.8489    0.8485    0.8487      4714

    accuracy                         0.6669     14141
   macro avg     0.5667    0.5921    0.5250     14141
weighted avg     0.5919    0.6669    0.5821     14141
 
 [[5403   54  417]
 [3230   28  295]
 [ 693   21 4000]]                  
Training epoch3: 100%|██████████| 590/590 [17:22<00:00,  1.77s/it]
0.6693303161021145
Epoch: 003; loss = 0.6979 cost time  1042.4036
0.6693303161021145
                      precision    recall  f1-score   support

                 Yes     0.5786    0.9229    0.7113      5874
                  No     0.3913    0.0127    0.0245      3553
           no-answer     0.8587    0.8483    0.8535      4714

            accuracy                         0.6693     14141
           macro avg     0.6095    0.5946    0.5298     14141
        weighted avg     0.6249    0.6693    0.5861     14141
 
         [[5421   61  392]
         [3242   45  266]
         [ 706    9 3999]]                   
EVAL
Accuracy: 0.6779 Loss in test 1.1083
                      precision    recall  f1-score   support

                 Yes     0.5818    0.9449    0.7201      2033
                  No     0.0000    0.0000    0.0000      1237
           no-answer     0.8759    0.8588    0.8673      1636

            accuracy                         0.6779      4906
           macro avg     0.4859    0.6012    0.5291      4906
        weighted avg     0.5332    0.6779    0.5876      4906
 
         [[1921    0  112]
         [1150    0   87]
         [ 231    0 1405]]

              
