# Introduction

This project is experimental code for SHAP-based iterative adjustment inputs of the ABSA prediction model TD-BERT. The project was developed based on modifications to the TD-BERT project code and uses the explainer of SHAP project to provide model explanations.

# How to run it?

## Train TD-BERT model

### 1. Step 1

Download the pretrained TensorFlow model: [uncased_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)

### 2. Step 2

Change the TensorFlow Pretrained Model into Pytorch

```shell
cd  convert_tf_to_pytorch
```

```
export BERT_BASE_DIR=./uncased_L-12_H-768_A-12

python3 convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin
```

### 3. Step 3: Training

Change the configeration and use the command:

```
python run_classifier_word.py \
--task_name=restaurant \
--data_dir=/root/Thesis/TD-BERT/datasets/semeval14/restaurants/3way \
--vocab_file=convert_tf_to_pytorch/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=convert_tf_to_pytorch/uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=convert_tf_to_pytorch/uncased_L-12_H-768_A-12/pytorch_model.bin \
--max_seq_length=128 \
--train_batch_size 32 \
--eval_batch_size 32 \
--learning_rate 5e-5 \
--num_train_epochs 6.0 \
--model_name=td_bert \
--output_dir=detailed_test_results \
--model_save_path=./save_model/restaurant \
--do_predict=false \
--do_train=true \
--do_eval=true \
--train_proportion 0.6

```

A TD-BERT model and remaining dataset (in fold `\datasets`) are saved.

## Prepare data for compare model

Change the configeration and use the command:

* The data_dir use the remaining dataset generated in last step.
* Model_save_path is the path of trained model in last step.
* "--do_iterative_predict" is for collect data for XGBoost.

```
python test_model.py \
--task_name=restaurant \
--data_dir=/root/Thesis/TD-BERT/datasets/semeval14/restaurants/3_way_remaining_0.4 \
--vocab_file=convert_tf_to_pytorch/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=convert_tf_to_pytorch/uncased_L-12_H-768_A-12/bert_config.json \
--max_seq_length=128 \
--eval_batch_size=32 \
--model_name=td_bert \
--output_dir=detailed_test_results \
--model_save_path=./save_model/restaurant \
--do_predict=true \
--do_train=false \
--do_eval=false \
--test_target=train \
--do_iterative_predict

```

In fold `\detailed_test_results` will save the detailed result of each prediction.

## Train compare model

Change the file path in `XGBoost.py`, and run the code to get a final model optimized by Optuna.

## Use iterative SHAP-Based Prediction Adjustment Approach to make prediction

Change the configeration and use the command:

* The xgb_model_path is the path of XGBoost model.
* Decision_metric is the metric of uncertainty score (margin,entropy,msp,all)
* "--do_hybrid_predict" is for make prediction.

```
python test_model.py \
--task_name=restaurant \
--data_dir=datasets/semeval14/restaurants/3way \
--vocab_file=convert_tf_to_pytorch/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=convert_tf_to_pytorch/uncased_L-12_H-768_A-12/bert_config.json \
--max_seq_length=128 \
--eval_batch_size=32 \
--model_name=td_bert \
--output_dir=detailed_test_results \
--model_save_path=./save_model/td_bert_60_84_6 \
--do_predict=true \
--do_train=false \
--do_eval=false \
--test_target=eval \
--do_hybrid_predict \
--xgb_model_path=./xgb_aggregated_feats_best.json \
--decision_metric margin
```

## Use LIME instead of SHAP

You can change the SHAP explainer into LIME explainer by this way:

```
explainer = LimeTextExplainer(class_names=label_list)
predictor = create_predictor_for_example(temp_example)
lime_explanation = explainer.explain_instance(
     current_sentence, 
     predictor, 
     num_features=len(current_sentence.split()),
     num_samples=500 
     )
```

# Explain the function of each file used and changed

## run_classifier_word.py

The training and validation program for the original TD-BERT project. It remains almost unchanged, with some modifications to the model loading method to accommodate PyTorch 2.6+ on the experimental equipment. The Chinese comments in the original project have been translated into English.

## test_model.py

This program is based on run_classifier_word.py with modifications. The code for training has been removed. Only the functions related to the prediction have been retained, and two additional functions have been added:

* do_predict and do_predict_with_model: Based on the original do_predict and do_eval functions. These functions are used to predict results using the chosen set and save detailed experimental results in a JSON file. Also, the function of predicting the training set and collecting the results is added to provide data for training the XGBoost model.
* iterative_predict_with_shap: An function for generating data to train the XGBoost model. It adds iterative functionality to do_predict and calculates SHAP when iteration is needed, masking the word with the highest SHAP value. This program records detailed data for each iteration without making a final prediction.
* hybrid_predict_with_xgboost: The complete prediction function of our approach. It applies the trained XGBoost model to each sentence after the iteration is completed, based on iterative_predict_with_shap, and makes the final prediction result. It also records all the results.

## XGBoost.py

This program trains the compare model based on XGBoost. It trains the XGBoost model using iterative prediction results based on SHAP generated by the previous steps. The program is mainly divided into three stages: data preparation, automatic parameter optimization using Optuna, and training the final XGBoost model with the best parameters.

* The data preparation stage is to extract all iteration results from each sample and organize them into a feature vector.
* The automatic parameter optimization stage is to use Optuna to optimize the hyperparameters of the XGBoost model.
* The training stage is to train the XGBoost model with the best parameters.

## data_util.py

This file is used to read data and preprocess it. A function of using only part of the training data has been added, other parts are the same as the original code.

## configs.py

Some more options are added, which are used in test_model.py. Other parts are the same as the original code.

# References

We gratefully acknowledge the work of [Gao Zhengjie et al. (2019)](https://github.com/gaozhengjie/TD-BERT), whose research and open-source TD-BERT code served as the foundation for our experiments. The original TD-BERT implementation can be accessed at: [https://github.com/gaozhengjie/TD-BERT](https://github.com/gaozhengjie/TD-BERT).

We also thank [Lundberg &amp; Lee (2017)](https://github.com/shap/shap) for their research on SHAP and the public release of the SHAP library, which played a critical role in our model interpretation process. The SHAP project is available at: [https://github.com/shap/shap](https://github.com/shap/shap).

Note: During the development of this project, Gemini family models were used as tools to assist in code debugging and problem-solving. This includes tasks such as applying SHAP to the non-standard TD-BERT architecture, converting data formats, and resolving dependency compatibility issues (e.g., PyTorch version conflicts).
