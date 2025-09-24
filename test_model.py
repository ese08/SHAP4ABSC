# coding=utf-8
"""
This program is based on run_classifier_word.py with modifications. The code for training has been removed. Only the functions related to the prediction have been retained, and two additional functions have been added:

    do_predict and do_predict_with_model: 
        Based on the original do_predict and do_eval functions. These functions are used to predict results using the chosen set and save detailed experimental results in a JSON file. Also, the function of predicting the training set and collecting the results is added to provide data for training the XGBoost model.

    iterative_predict_with_shap: 
        An function for generating data to train the XGBoost model. It adds iterative functionality to do_predict and calculates SHAP when iteration is needed, masking the word with the highest SHAP value. This program records detailed data for each iteration without making a final prediction.

    hybrid_predict_with_xgboost: 
        The complete prediction function of our approach. It applies the trained XGBoost model to each sentence after the iteration is completed, based on iterative_predict_with_shap, and makes the final prediction result. It also records all the results.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from tqdm import tqdm
import numpy as np
import torch
import json
import csv
from datetime import datetime
import copy
import re
import shap
from modeling import BertConfig, BertForSequenceClassification
from configs import get_config
from models import CNN, CLSTM, PF_CNN, TCN, Bert_PF, BBFC, TC_CNN, RAM, IAN, ATAE_LSTM, AOA, MemNet, Cabasc, TNet_LF, \
    MGAN, BERT_IAN, TC_SWEM, MLP, AEN_BERT, TD_BERT, TD_BERT_QA, DTD_BERT
from utils.data_util import ReadData, RestaurantProcessor, LaptopProcessor, TweetProcessor

# Adapt for PyTorch 2.6+
try:
    from models.td_bert import TD_BERT as TD_BERT_Class
    torch.serialization.add_safe_globals([TD_BERT_Class])
except (ImportError, AttributeError):
    pass

from xgboost import XGBClassifier
import torch.nn.functional as F



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


class Tester:
    """
    To avoid making the program too complex, we removed the model training section from the original run_classifier_word.py 
    and built this new program based solely on the prediction function to expand our architecture.
    """
    def __init__(self, args):
        self.opt = args
        bert_config = BertConfig.from_json_file(args.bert_config_file)

        if args.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                    args.max_seq_length, bert_config.max_position_embeddings))

        self.dataset = ReadData(self.opt)  
        self.opt.label_size = len(self.dataset.label_list)
        args.output_dim = len(self.dataset.label_list)
        print("label size: {}".format(args.output_dim))

        print("initialize model ...")
        if args.model_class == BertForSequenceClassification:
            self.model_structure = model_classes['fc'](bert_config, len(self.dataset.label_list))
        else:
            self.model_structure = model_classes[args.model_name.lower()](bert_config, args)

        self.model_structure.to(args.device)

    def do_predict(self):
        """Load model for prediction"""
        if not self.opt.model_save_path or not os.path.exists(self.opt.model_save_path):
            raise ValueError(f"Model path not provided or path incorrect'")
            
        try:
            # adapt for pytorch 2.6+
            saved_model = torch.load(self.opt.model_save_path, weights_only=False)
        except Exception as e:
            try:
                saved_model = torch.load(self.opt.model_save_path, map_location=self.opt.device, weights_only=False)
            except Exception as e2:
                raise RuntimeError(f"failed to load model from {self.opt.model_save_path}")

        saved_model.to(self.opt.device)
        saved_model.eval()
        return self.do_predict_with_model(saved_model)

    def do_predict_with_model(self, saved_model):
        """Make predictions using the loaded model."""

        # select the training set or validation set for prediction
        # use training set to prepare data for the XGBoost model
        if self.opt.test_target.lower() == 'train':
            dataloader = self.dataset.train_dataloader
            examples = self.dataset.train_examples
            config_batch_size = self.opt.train_batch_size
            data_desc = "Training Set"
        elif self.opt.test_target.lower() == 'eval':
            dataloader = self.dataset.eval_dataloader
            examples = self.dataset.eval_examples
            config_batch_size = self.opt.eval_batch_size
            data_desc = "Validation Set"
        else:
            raise ValueError(f"Invalid predict target")

        saved_model.to(self.opt.device)
        saved_model.eval()
        nb_test_examples = 0
        test_accuracy = 0
        
        detailed_results = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{data_desc}")):
            # change: use the same way as used in aforemention functions do_eval
            input_ids, input_mask, segment_ids, label_ids, \
            input_t_ids, input_t_mask, segment_t_ids, \
            input_without_t_ids, input_without_t_mask, segment_without_t_ids, \
            input_left_t_ids, input_left_t_mask, segment_left_t_ids, \
            input_right_t_ids, input_right_t_mask, segment_right_t_ids, \
            input_left_ids, input_left_mask, segment_left_ids = batch

            input_ids = input_ids.to(self.opt.device)
            segment_ids = segment_ids.to(self.opt.device)
            input_mask = input_mask.to(self.opt.device)
            label_ids = label_ids.to(self.opt.device)

            with torch.no_grad():  # Do not calculate gradient
                if self.opt.model_class in [BertForSequenceClassification, CNN]:
                    loss, logits = saved_model(input_ids, segment_ids, input_mask, label_ids)
                else:
                    input_t_ids = input_t_ids.to(self.opt.device)
                    input_t_mask = input_t_mask.to(self.opt.device)
                    segment_t_ids = segment_t_ids.to(self.opt.device)
                    if self.opt.model_class == MemNet:
                        input_without_t_ids = input_without_t_ids.to(self.opt.device)
                        input_without_t_mask = input_without_t_mask.to(self.opt.device)
                        segment_without_t_ids = segment_without_t_ids.to(self.opt.device)
                        loss, logits = saved_model(input_without_t_ids, segment_without_t_ids, input_without_t_mask,
                                              label_ids, input_t_ids, input_t_mask, segment_t_ids)
                    elif self.opt.model_class in [Cabasc]:
                        input_left_t_ids = input_left_t_ids.to(self.opt.device)
                        input_left_mask = input_left_t_mask.to(self.opt.device)
                        segment_left_t_ids = segment_left_t_ids.to(self.opt.device)
                        input_right_t_ids = input_right_t_ids.to(self.opt.device)
                        input_right_t_mask = input_right_t_mask.to(self.opt.device)
                        segment_right_t_ids = segment_right_t_ids.to(self.opt.device)
                        loss, logits = saved_model(input_ids, segment_ids, input_mask, label_ids,
                                              input_t_ids, input_t_mask, segment_t_ids,
                                              input_left_t_ids, input_left_t_mask, segment_left_t_ids,
                                              input_right_t_ids, input_right_t_mask, segment_right_t_ids)
                    elif self.opt.model_class in [RAM, TNet_LF, MGAN, MLP, TD_BERT, TD_BERT_QA, DTD_BERT]:
                        input_left_ids = input_left_ids.to(self.opt.device)
                        input_left_mask = input_left_mask.to(self.opt.device)
                        segment_left_ids = segment_left_ids.to(self.opt.device)
                        loss, logits = saved_model(input_ids, segment_ids, input_mask, label_ids,
                                              input_t_ids, input_t_mask, segment_t_ids,
                                              input_left_ids, input_left_mask, segment_left_ids)
                    else:
                        loss, logits = saved_model(input_ids, segment_ids, input_mask, label_ids, input_t_ids,
                                              input_t_mask, segment_t_ids)
            
            # convert logits to numpy
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            logits_tensor = torch.tensor(logits)
            probabilities = F.softmax(logits_tensor, dim=1).numpy()
            
            # calculate softmax probability
            predictions = np.argmax(logits, axis=1)
            max_softmax_probabilities = np.max(probabilities, axis=1)
            
            current_batch_size = input_ids.size(0)
            start_idx = batch_idx * config_batch_size
            
            for i in range(current_batch_size):
                example_idx = start_idx + i
                if example_idx < len(examples):
                    example = examples[example_idx]
                    
                    label_map = {0: 'positive', 1: 'neutral', 2: 'negative'} 
                    true_label = label_map.get(label_ids[i], str(label_ids[i]))
                    pred_label = label_map.get(predictions[i], str(predictions[i]))
                    is_correct = bool(label_ids[i] == predictions[i])
                    
                    class_probabilities = {label_map.get(idx, str(idx)): float(prob) for idx, prob in enumerate(probabilities[i])}
                    
                    result_record = {
                        'id': example_idx + 1,
                        'guid': example.guid,
                        'sentence': example.text_a,
                        'target_word': example.aspect,
                        'true_label': true_label,
                        'predicted_label': pred_label,
                        'is_correct': is_correct,
                        'prediction_confidence': float(max_softmax_probabilities[i]),
                        'all_class_probabilities': class_probabilities,
                        'raw_logits': [float(x) for x in logits[i]],
                        'text_left': getattr(example, 'text_left', ''),
                        'text_without_target': getattr(example, 'text_without_target', ''),
                        'entropy': float(-np.sum(probabilities[i] * np.log(probabilities[i] + 1e-9))),
                        'second_highest_prob': float(np.sort(probabilities[i])[-2]) if len(probabilities[i]) > 1 else 0.0
                    }
                    detailed_results.append(result_record)
            
            tmp_test_accuracy = accuracy(logits, label_ids)
            test_accuracy += tmp_test_accuracy
            nb_test_examples += input_ids.size(0)
        
        overall_accuracy = test_accuracy / nb_test_examples if nb_test_examples > 0 else 0
        
        os.makedirs(self.opt.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.opt.output_dir, f"prediction_results_{self.opt.task_name}_{self.opt.model_name}_{self.opt.test_target}_{timestamp}.json")
        
        save_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'task_name': self.opt.task_name,
                'model_name': self.opt.model_name,
                'model_path': self.opt.model_save_path,
                'data_dir': self.opt.data_dir,
                'tested_on': self.opt.test_target,
                'total_samples': len(detailed_results),
                'overall_accuracy': float(overall_accuracy),
                'total_correct': int(test_accuracy),
                'total_examples': int(nb_test_examples)
            },
            'results': detailed_results
        }
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as e:
            print(f"JSON save failed: {e}")
        
        return detailed_results

    def iterative_predict_with_shap(self, saved_model):
        """
        Adjust the input type to meet the requirements of the SHAP package, 
        and use SHAP Explainer to calculate the SHAP values. 
        Then, filter the words that need to be masked based on the SHAP values, and complete the iteration.
        Also, record detailed results during the iteration.
        """
        
        # set iteration parameters
        MAX_ITERATIONS = self.opt.max_iterations
        PREDICT_MARGIN_THRESHOLD = self.opt.margin_threshold
        ENTROPY_THRESHOLD = self.opt.entropy_threshold
        MSP_THRESHOLD = self.opt.msp_threshold
        
        os.makedirs(self.opt.output_dir, exist_ok=True)
        tokenizer = self.dataset.tokenizer
        label_list = self.dataset.label_list
        label_map = {i: label for i, label in enumerate(label_list)}
        
        # select the training set or validation set for prediction
        # use training set to prepare data for the XGBoost model
        if self.opt.test_target.lower() == 'train':
            dataloader = self.dataset.train_dataloader
            examples = self.dataset.train_examples
            config_batch_size = self.opt.train_batch_size
            data_desc = "Training Set"
        elif self.opt.test_target.lower() == 'eval':
            dataloader = self.dataset.eval_dataloader
            examples = self.dataset.eval_examples
            config_batch_size = self.opt.eval_batch_size
            data_desc = "Validation Set"
        else:
            raise ValueError(f"Invalid predict target")

        # create SHAP explainer
        # let SHAP explainer correctly handle [MASK] token
        masker = shap.maskers.Text(r"\[MASK\]|[\w']+|[.,!?;:]")
        
        def create_predictor_for_example(original_example):
            """
            Adjust the input value format and build a prediction function which complies with SHAP.
            """
            def predictor(texts):
                input_features = []
                for text in texts:
                    aspect = original_example.aspect
                    
                    # reconstruct 
                    if aspect in text:
                        text_parts = text.split(aspect)
                        text_left = text_parts[0] if len(text_parts) > 0 else ""
                        text_without_target = text.replace(aspect, "")
                        text_left_with_target = text_left + aspect
                        text_right_with_target = aspect + (text_parts[1] if len(text_parts) > 1 else "")
                    else:
                        text_left = getattr(original_example, 'text_left', '')
                        text_without_target = text
                        text_left_with_target = text_left + aspect
                        text_right_with_target = aspect

                    new_example = copy.copy(original_example)
                    new_example.text_a = text
                    new_example.text_left = text_left.strip()
                    new_example.text_without_target = text_without_target.strip()
                    new_example.text_left_with_target = text_left_with_target.strip()
                    new_example.text_right_with_target = text_right_with_target.strip()

                    converted_features = self.dataset.convert_examples_to_features(
                        [new_example], label_list, self.opt.max_seq_length, tokenizer)[0]
                    input_features.append(converted_features)

                input_ids = torch.tensor([f.input_ids for f in input_features], dtype=torch.long).to(self.opt.device)
                input_mask = torch.tensor([f.input_mask for f in input_features], dtype=torch.long).to(self.opt.device)
                segment_ids = torch.tensor([f.segment_ids for f in input_features], dtype=torch.long).to(self.opt.device)
                label_ids = torch.tensor([f.label_id for f in input_features], dtype=torch.long).to(self.opt.device)
                input_t_ids = torch.tensor([f.input_t_ids for f in input_features], dtype=torch.long).to(self.opt.device)
                input_t_mask = torch.tensor([f.input_t_mask for f in input_features], dtype=torch.long).to(self.opt.device)
                segment_t_ids = torch.tensor([f.segment_t_ids for f in input_features], dtype=torch.long).to(self.opt.device)
                input_left_ids = torch.tensor([f.input_left_ids for f in input_features], dtype=torch.long).to(self.opt.device)
                input_left_mask = torch.tensor([f.input_left_mask for f in input_features], dtype=torch.long).to(self.opt.device)
                segment_left_ids = torch.tensor([f.segment_left_ids for f in input_features], dtype=torch.long).to(self.opt.device)
                
                #predict
                with torch.no_grad():
                    if self.opt.model_class in [RAM, TNet_LF, MGAN, MLP, TD_BERT, TD_BERT_QA, DTD_BERT]:
                        _, logits = saved_model(input_ids, segment_ids, input_mask, label_ids,
                                              input_t_ids, input_t_mask, segment_t_ids,
                                              input_left_ids, input_left_mask, segment_left_ids)
                    else:
                        _, logits = saved_model(input_ids, segment_ids, input_mask, label_ids, input_t_ids,
                                              input_t_mask, segment_t_ids)
                
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                return probs
            return predictor

        all_iterative_results = []
        
        # iterate over all samples
        total_samples = len(examples)
        for sample_idx, example in enumerate(tqdm(examples, desc=f"{data_desc}")):
            
            sample_iterations = {
                'sample_id': sample_idx + 1,
                'guid': example.guid,
                'original_sentence': example.text_a,
                'target_word': example.aspect,
                'true_label': label_map.get(example.label, str(example.label)),
                'iterations': []
            }
            
            current_sentence = example.text_a
            masked_words = []  

            for iteration in range(MAX_ITERATIONS):
                
                temp_example = copy.copy(example)

                # replace the original sentence with the sentence that has been masked
                temp_example.text_a = current_sentence
                
                aspect = example.aspect
                if aspect in current_sentence:
                    text_parts = current_sentence.split(aspect)
                    text_left = text_parts[0] if len(text_parts) > 0 else ""
                    text_without_target = current_sentence.replace(aspect, "")
                    text_left_with_target = text_left + aspect
                    text_right_with_target = aspect + (text_parts[1] if len(text_parts) > 1 else "")
                else:
                    text_left = getattr(example, 'text_left', '')
                    text_without_target = current_sentence
                    text_left_with_target = text_left + aspect
                    text_right_with_target = aspect
                
                temp_example.text_left = text_left.strip()
                temp_example.text_without_target = text_without_target.strip()
                temp_example.text_left_with_target = text_left_with_target.strip()
                temp_example.text_right_with_target = text_right_with_target.strip()
                
                converted_features = self.dataset.convert_examples_to_features(
                    [temp_example], label_list, self.opt.max_seq_length, tokenizer)[0]
                
                input_ids = torch.tensor([converted_features.input_ids], dtype=torch.long).to(self.opt.device)
                input_mask = torch.tensor([converted_features.input_mask], dtype=torch.long).to(self.opt.device)
                segment_ids = torch.tensor([converted_features.segment_ids], dtype=torch.long).to(self.opt.device)
                label_ids = torch.tensor([converted_features.label_id], dtype=torch.long).to(self.opt.device)
                input_t_ids = torch.tensor([converted_features.input_t_ids], dtype=torch.long).to(self.opt.device)
                input_t_mask = torch.tensor([converted_features.input_t_mask], dtype=torch.long).to(self.opt.device)
                segment_t_ids = torch.tensor([converted_features.segment_t_ids], dtype=torch.long).to(self.opt.device)
                input_left_ids = torch.tensor([converted_features.input_left_ids], dtype=torch.long).to(self.opt.device)
                input_left_mask = torch.tensor([converted_features.input_left_mask], dtype=torch.long).to(self.opt.device)
                segment_left_ids = torch.tensor([converted_features.segment_left_ids], dtype=torch.long).to(self.opt.device)
                
                # predict
                with torch.no_grad():
                    if self.opt.model_class in [RAM, TNet_LF, MGAN, MLP, TD_BERT, TD_BERT_QA, DTD_BERT]:
                        _, logits = saved_model(input_ids, segment_ids, input_mask, label_ids,
                                              input_t_ids, input_t_mask, segment_t_ids,
                                              input_left_ids, input_left_mask, segment_left_ids)
                    else:
                        _, logits = saved_model(input_ids, segment_ids, input_mask, label_ids, input_t_ids,
                                              input_t_mask, segment_t_ids)
                
                logits = logits.detach().cpu().numpy()[0]
                probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()
                
                # calculate metrics (MSP, Margin, Entropy)
                prediction = np.argmax(logits)
                max_prob = np.max(probabilities)  
                second_max_prob = np.sort(probabilities)[-2] if len(probabilities) > 1 else 0.0
                predict_margin = max_prob - second_max_prob
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
                pred_label = label_map.get(prediction, str(prediction))
                is_correct = bool(prediction == label_ids.item())
                
                # calculate SHAP values
                predictor = create_predictor_for_example(temp_example)
                explainer = shap.Explainer(predictor, masker, output_names=label_list)
                shap_values = explainer([current_sentence])
                
                tokens = shap_values.data[0]
                raw_values = shap_values.values[0]  # (num_tokens, num_classes)
                
                # only record the SHAP values for the predicted class
                pred_class_shap_values = raw_values[:, prediction]
                
                token_shap_info = []
                for token_idx, token in enumerate(tokens):
                    token_info = {
                        'token': token,
                        'shap_value_for_prediction': float(pred_class_shap_values[token_idx]),
                        'abs_shap_value': abs(float(pred_class_shap_values[token_idx]))
                    }
                    token_shap_info.append(token_info)
                
                # save iteration result for XGBoost
                iteration_result = {
                    'iteration': iteration + 1,
                    'sentence': current_sentence,
                    'masked_words': masked_words.copy(),
                    'prediction': pred_label,
                    'prediction_id': int(prediction),
                    'is_correct': is_correct,
                    'probabilities': [float(p) for p in probabilities],
                    'raw_logits': [float(l) for l in logits],
                    'msp': float(max_prob),
                    'predict_margin': float(predict_margin),
                    'entropy': float(entropy),
                    'second_highest_prob': float(second_max_prob),
                    'tokens_with_shap': token_shap_info
                }
                
                # stop condition
                if self.opt.decision_metric == 'margin':
                    meets_criteria = predict_margin >= PREDICT_MARGIN_THRESHOLD
                elif self.opt.decision_metric == 'entropy':
                    meets_criteria = entropy <= ENTROPY_THRESHOLD
                elif self.opt.decision_metric == 'msp':
                    meets_criteria = max_prob >= MSP_THRESHOLD
                else:  # all (used in data for XGBoost)
                    meets_criteria = (predict_margin >= PREDICT_MARGIN_THRESHOLD and 
                                     entropy <= ENTROPY_THRESHOLD and 
                                     max_prob >= MSP_THRESHOLD)
                
                iteration_result['meets_criteria'] = meets_criteria
                
                sample_iterations['iterations'].append(iteration_result)
                
                if iteration_result['meets_criteria']:
                    break
                
                # mask word with highest SHAP for next iteration
                if iteration < MAX_ITERATIONS - 1:
                    # no mask on aspect target
                    target_tokens = set(re.findall(r'\w+', example.aspect.lower()))
                    
                    candidate_tokens = []
                    for token_info in token_shap_info:
                        token_str = token_info['token']
                        
                        # no mask on [MASK] 
                        if token_str == '[MASK]':
                            continue
                        
                        # no mask on masked words
                        if token_str in masked_words:
                            continue

                        token_stripped = token_str.strip()
                        if token_stripped.lower() in target_tokens:
                            continue
                        
                        # no mask on non-word 
                        if not re.match(r'\w+', token_stripped):
                            continue
                        
                        candidate_tokens.append(token_info)
                    
                    if candidate_tokens:
                        candidate_tokens.sort(key=lambda x: x['abs_shap_value'], reverse=True)
                        word_to_mask = candidate_tokens[0]['token']
                        masked_words.append(word_to_mask)
                        
                        # mask word with highest SHAP
                        current_sentence = current_sentence.replace(word_to_mask, '[MASK]', 1)

                    else:
                        break

            all_iterative_results.append(sample_iterations)
        
        # save all iteration results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.opt.output_dir, f"iterative_prediction_results_{self.opt.task_name}_{self.opt.model_name}_{self.opt.test_target}_{timestamp}.json")
        
        save_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'task_name': self.opt.task_name,
                'model_name': self.opt.model_name,
                'model_path': self.opt.model_save_path,
                'data_dir': self.opt.data_dir,
                'tested_on': self.opt.test_target,
                'max_iterations': MAX_ITERATIONS,
                'thresholds': {
                    'predict_margin': PREDICT_MARGIN_THRESHOLD,
                    'entropy': ENTROPY_THRESHOLD,
                    'msp': MSP_THRESHOLD
                },
                'total_samples': len(all_iterative_results)
            },
            'results': all_iterative_results
        }
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print(f"Iterative prediction results saved to: {results_file}")
        except (TypeError, ValueError) as e:
            print(f"JSON save failed: {e}")
        
        return all_iterative_results

    def hybrid_predict_with_xgboost(self, saved_model):
        """
        Based on previous iteration function structure, 
        introduce trained XGBoost (compare model) to make the final prediction
        """
        
        if not hasattr(self.opt, 'xgb_model_path') or not self.opt.xgb_model_path:
            raise ValueError("XGBoost model file path invalid")
        
        if not os.path.exists(self.opt.xgb_model_path):
            raise ValueError(f"XGBoost model file path invalid")
        
        xgb_model = XGBClassifier()
        xgb_model.load_model(self.opt.xgb_model_path)
        
        # set iteration parameters
        MAX_ITERATIONS = self.opt.max_iterations
        PREDICT_MARGIN_THRESHOLD = self.opt.margin_threshold
        ENTROPY_THRESHOLD = self.opt.entropy_threshold
        MSP_THRESHOLD = self.opt.msp_threshold
        
        os.makedirs(self.opt.output_dir, exist_ok=True)
        tokenizer = self.dataset.tokenizer
        label_list = self.dataset.label_list
        label_map = {i: label for i, label in enumerate(label_list)}
        

        # select the training set or validation set for prediction
        # use training set to prepare data for the XGBoost model
        if self.opt.test_target.lower() == 'train':
            dataloader = self.dataset.train_dataloader
            examples = self.dataset.train_examples
            config_batch_size = self.opt.train_batch_size
            data_desc = "Training Set"
        elif self.opt.test_target.lower() == 'eval':
            dataloader = self.dataset.eval_dataloader
            examples = self.dataset.eval_examples
            config_batch_size = self.opt.eval_batch_size
            data_desc = "Validation Set"
        else:
            raise ValueError(f"Invalid predict target")

        # create SHAP explainer
        # let SHAP explainer correctly handle [MASK] token
        masker = shap.maskers.Text(r"\[MASK\]|[\w']+|[.,!?;:]")
        
        def create_predictor_for_example(original_example):
            """
            Adjust the input value format and build a prediction function which complies with SHAP.
            """
            def predictor(texts):
                input_features = []
                for text in texts:
                    aspect = original_example.aspect
                    
                    # reconstruct 
                    if aspect in text:
                        text_parts = text.split(aspect)
                        text_left = text_parts[0] if len(text_parts) > 0 else ""
                        text_without_target = text.replace(aspect, "")
                        text_left_with_target = text_left + aspect
                        text_right_with_target = aspect + (text_parts[1] if len(text_parts) > 1 else "")
                    else:
                        text_left = getattr(original_example, 'text_left', '')
                        text_without_target = text
                        text_left_with_target = text_left + aspect
                        text_right_with_target = aspect

                    new_example = copy.copy(original_example)
                    new_example.text_a = text
                    new_example.text_left = text_left.strip()
                    new_example.text_without_target = text_without_target.strip()
                    new_example.text_left_with_target = text_left_with_target.strip()
                    new_example.text_right_with_target = text_right_with_target.strip()

                    converted_features = self.dataset.convert_examples_to_features(
                        [new_example], label_list, self.opt.max_seq_length, tokenizer)[0]
                    input_features.append(converted_features)

                input_ids = torch.tensor([f.input_ids for f in input_features], dtype=torch.long).to(self.opt.device)
                input_mask = torch.tensor([f.input_mask for f in input_features], dtype=torch.long).to(self.opt.device)
                segment_ids = torch.tensor([f.segment_ids for f in input_features], dtype=torch.long).to(self.opt.device)
                label_ids = torch.tensor([f.label_id for f in input_features], dtype=torch.long).to(self.opt.device)
                input_t_ids = torch.tensor([f.input_t_ids for f in input_features], dtype=torch.long).to(self.opt.device)
                input_t_mask = torch.tensor([f.input_t_mask for f in input_features], dtype=torch.long).to(self.opt.device)
                segment_t_ids = torch.tensor([f.segment_t_ids for f in input_features], dtype=torch.long).to(self.opt.device)
                input_left_ids = torch.tensor([f.input_left_ids for f in input_features], dtype=torch.long).to(self.opt.device)
                input_left_mask = torch.tensor([f.input_left_mask for f in input_features], dtype=torch.long).to(self.opt.device)
                segment_left_ids = torch.tensor([f.segment_left_ids for f in input_features], dtype=torch.long).to(self.opt.device)
                
                #predict
                with torch.no_grad():
                    if self.opt.model_class in [RAM, TNet_LF, MGAN, MLP, TD_BERT, TD_BERT_QA, DTD_BERT]:
                        _, logits = saved_model(input_ids, segment_ids, input_mask, label_ids,
                                              input_t_ids, input_t_mask, segment_t_ids,
                                              input_left_ids, input_left_mask, segment_left_ids)
                    else:
                        _, logits = saved_model(input_ids, segment_ids, input_mask, label_ids, input_t_ids,
                                              input_t_mask, segment_t_ids)
                
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                return probs
            return predictor

        all_hybrid_results = []
        
        direct_decisions = 0    
        
        # iterate over all samples
        total_samples = len(examples)
        for sample_idx, example in enumerate(tqdm(examples, desc=f"{data_desc}")):

            sample_result = {
                'sample_id': sample_idx + 1,
                'guid': example.guid,
                'original_sentence': example.text_a,
                'target_word': example.aspect,
                'true_label': label_map.get(example.label, str(example.label)),
                'true_label_id': example.label,
                'iterations': [],
                'decision_source': '',  # 'direct' or 'xgboost'
                'final_prediction': '',
                'final_prediction_id': -1,
                'is_correct': False,
                'xgb_features': None,
                'xgb_probabilities': None
            }
            
            current_sentence = example.text_a
            masked_words = []
            
            for iteration in range(MAX_ITERATIONS):
                
                temp_example = copy.copy(example)

                # replace the original sentence with the sentence that has been masked
                temp_example.text_a = current_sentence
                
                aspect = example.aspect
                if aspect in current_sentence:
                    text_parts = current_sentence.split(aspect)
                    text_left = text_parts[0] if len(text_parts) > 0 else ""
                    text_without_target = current_sentence.replace(aspect, "")
                    text_left_with_target = text_left + aspect
                    text_right_with_target = aspect + (text_parts[1] if len(text_parts) > 1 else "")
                else:
                    text_left = getattr(example, 'text_left', '')
                    text_without_target = current_sentence
                    text_left_with_target = text_left + aspect
                    text_right_with_target = aspect
                
                temp_example.text_left = text_left.strip()
                temp_example.text_without_target = text_without_target.strip()
                temp_example.text_left_with_target = text_left_with_target.strip()
                temp_example.text_right_with_target = text_right_with_target.strip()
                
                converted_features = self.dataset.convert_examples_to_features(
                    [temp_example], label_list, self.opt.max_seq_length, tokenizer)[0]
                
                input_ids = torch.tensor([converted_features.input_ids], dtype=torch.long).to(self.opt.device)
                input_mask = torch.tensor([converted_features.input_mask], dtype=torch.long).to(self.opt.device)
                segment_ids = torch.tensor([converted_features.segment_ids], dtype=torch.long).to(self.opt.device)
                label_ids = torch.tensor([converted_features.label_id], dtype=torch.long).to(self.opt.device)
                input_t_ids = torch.tensor([converted_features.input_t_ids], dtype=torch.long).to(self.opt.device)
                input_t_mask = torch.tensor([converted_features.input_t_mask], dtype=torch.long).to(self.opt.device)
                segment_t_ids = torch.tensor([converted_features.segment_t_ids], dtype=torch.long).to(self.opt.device)
                input_left_ids = torch.tensor([converted_features.input_left_ids], dtype=torch.long).to(self.opt.device)
                input_left_mask = torch.tensor([converted_features.input_left_mask], dtype=torch.long).to(self.opt.device)
                segment_left_ids = torch.tensor([converted_features.segment_left_ids], dtype=torch.long).to(self.opt.device)
                
                # predict
                with torch.no_grad():
                    if self.opt.model_class in [RAM, TNet_LF, MGAN, MLP, TD_BERT, TD_BERT_QA, DTD_BERT]:
                        _, logits = saved_model(input_ids, segment_ids, input_mask, label_ids,
                                              input_t_ids, input_t_mask, segment_t_ids,
                                              input_left_ids, input_left_mask, segment_left_ids)
                    else:
                        _, logits = saved_model(input_ids, segment_ids, input_mask, label_ids, input_t_ids,
                                              input_t_mask, segment_t_ids)
                
                logits = logits.detach().cpu().numpy()[0]
                probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()
                
                # calculate prediction (MSP,Margin,Entropy)
                prediction = np.argmax(logits)
                max_prob = np.max(probabilities)
                second_max_prob = np.sort(probabilities)[-2] if len(probabilities) > 1 else 0.0
                predict_margin = max_prob - second_max_prob
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
                pred_label = label_map.get(prediction, str(prediction))
                is_correct = bool(prediction == label_ids.item())
                
                # calculate SHAP values
                predictor = create_predictor_for_example(temp_example)
                explainer = shap.Explainer(predictor, masker, output_names=label_list)
                shap_values = explainer([current_sentence])
                
                tokens = shap_values.data[0]
                raw_values = shap_values.values[0] # (num_tokens, num_classes)

                # only record the SHAP values for the predicted class
                pred_class_shap_values = raw_values[:, prediction]
                
                token_shap_info = []
                for token_idx, token in enumerate(tokens):
                    token_info = {
                        'token': token,
                        'shap_value_for_prediction': float(pred_class_shap_values[token_idx]),
                        'abs_shap_value': abs(float(pred_class_shap_values[token_idx]))
                    }
                    token_shap_info.append(token_info)
                
                # stop condition
                if self.opt.decision_metric == 'margin':
                    meets_criteria = predict_margin >= PREDICT_MARGIN_THRESHOLD
                elif self.opt.decision_metric == 'entropy':
                    meets_criteria = entropy <= ENTROPY_THRESHOLD
                elif self.opt.decision_metric == 'msp':
                    meets_criteria = max_prob >= MSP_THRESHOLD
                else: # all (used in data for XGBoost)
                    meets_criteria = (predict_margin >= PREDICT_MARGIN_THRESHOLD and 
                                     entropy <= ENTROPY_THRESHOLD and 
                                     max_prob >= MSP_THRESHOLD)
                
                iteration_result = {
                    'iteration': iteration + 1,
                    'sentence': current_sentence,
                    'masked_words': masked_words.copy(),
                    'prediction': pred_label,
                    'prediction_id': int(prediction),
                    'is_correct': is_correct,
                    'msp': float(max_prob),
                    'predict_margin': float(predict_margin),
                    'entropy': float(entropy),
                    'second_highest_prob': float(second_max_prob),
                    'tokens_with_shap': token_shap_info,
                    'meets_criteria': meets_criteria
                }
                
                sample_result['iterations'].append(iteration_result)
                
                # if first iteration meets confidence criteria, use it as result
                if iteration == 0 and meets_criteria:
                    sample_result['decision_source'] = 'direct'
                    sample_result['final_prediction'] = pred_label
                    sample_result['final_prediction_id'] = int(prediction)
                    sample_result['is_correct'] = is_correct
                    direct_decisions += 1
                    break
                
                # if other iteration meets confidence criteria, stop and use XGBoost
                if iteration > 0 and meets_criteria:
                    break
                
                # mask word with highest SHAP for next iteration
                if iteration < MAX_ITERATIONS - 1:
                    # no mask on aspect target
                    target_tokens = set(re.findall(r'\w+', example.aspect.lower()))
                    
                    candidate_tokens = []
                    for token_info in token_shap_info:
                        token_str = token_info['token']
                        
                        # no mask on [MASK] 
                        if token_str == '[MASK]':
                            continue
                        
                        # no mask on masked words
                        if token_str in masked_words:
                            continue

                        token_stripped = token_str.strip()
                        if token_stripped.lower() in target_tokens:
                            continue
                        
                        # no mask on non-word 
                        if not re.match(r'\w+', token_stripped):
                            continue
                        
                        candidate_tokens.append(token_info)
                    
                    if candidate_tokens:
                        candidate_tokens.sort(key=lambda x: x['abs_shap_value'], reverse=True)
                        word_to_mask = candidate_tokens[0]['token']
                        masked_words.append(word_to_mask)

                        # mask word with highest SHAP
                        current_sentence = current_sentence.replace(word_to_mask, '[MASK]', 1)
                    else:
                        break
            
            # if no early success, use XGBoost
            if sample_result.get('decision_source') != 'direct':
                
                # extract features of all iterations for XGBoost
                predictions = []
                msps = []
                margins = []
                entropies = []
                
                for it in sample_result['iterations']:
                    predictions.append(it['prediction_id'])
                    msps.append(it['msp'])
                    margins.append(it['predict_margin'])
                    entropies.append(it['entropy'])
                
                num_iters = len(sample_result['iterations'])
                unique_sentiments_count = len(set(predictions))
                TRAINING_MAX_ITERS = MAX_ITERATIONS  
                
                xgb_features = [num_iters, unique_sentiments_count]
                
                iter_features_map = {
                    "pred": predictions,
                    "msp": msps,
                    "margin": margins,
                    "entropy": entropies
                }
                
                for feat_name in ["pred", "msp", "margin", "entropy"]:
                    values = iter_features_map[feat_name]
                    for i in range(TRAINING_MAX_ITERS):
                        value = values[i] if i < len(values) else 0.0
                        xgb_features.append(value)
                
                # predict use XGBoost
                xgb_input = np.array(xgb_features).reshape(1, -1)
                xgb_probs = xgb_model.predict_proba(xgb_input)[0]
                xgb_prediction = np.argmax(xgb_probs)
                
                xgb_pred_label = label_map.get(xgb_prediction, str(xgb_prediction))
                xgb_is_correct = bool(xgb_prediction == example.label)
                                
                sample_result['decision_source'] = 'xgboost'
                sample_result['final_prediction'] = xgb_pred_label
                sample_result['final_prediction_id'] = int(xgb_prediction)
                sample_result['is_correct'] = xgb_is_correct
                sample_result['xgb_probabilities'] = [float(p) for p in xgb_probs]
            
            all_hybrid_results.append(sample_result)
        
        # calculate overall accuracy
        total_correct = sum(1 for r in all_hybrid_results if r['is_correct'])
        overall_accuracy = total_correct / len(all_hybrid_results) if all_hybrid_results else 0
                
        # save results (integrated from _save_hybrid_results)
        os.makedirs(self.opt.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"hybrid_prediction_results_{self.opt.task_name}_{self.opt.model_name}_{self.opt.test_target}_{timestamp}"
        
        json_file = os.path.join(self.opt.output_dir, f"{base_filename}.json")
        summary_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'task_name': self.opt.task_name,
                'model_name': self.opt.model_name,
                'model_path': self.opt.model_save_path,
                'xgb_model_path': self.opt.xgb_model_path,
                'data_dir': self.opt.data_dir,
                'tested_on': self.opt.test_target,
                'total_samples': len(all_hybrid_results),
                'overall_accuracy': float(overall_accuracy),
                'direct_decisions': direct_decisions,
                'decision_metric': self.opt.decision_metric,
                'thresholds': {
                    'predict_margin': self.opt.margin_threshold,
                    'entropy': self.opt.entropy_threshold,
                    'msp': self.opt.msp_threshold
                }
            },
            'results': all_hybrid_results
        }
        
        final_results = []
        for result in all_hybrid_results:
            final_result = {
                'sample_id': result['sample_id'],
                'guid': result['guid'],
                'original_sentence': result['original_sentence'],
                'target_word': result['target_word'],
                'true_label': result['true_label'],
                'decision_source': result['decision_source'],
                'final_prediction': result['final_prediction'],
                'is_correct': result['is_correct'],
                'iteration_summary': [
                    {
                        'iteration': it['iteration'],
                        'prediction': it['prediction'],
                        'is_correct': it['is_correct'],
                        'msp': it['msp'],
                        'predict_margin': it['predict_margin'],
                        'entropy': it['entropy'],
                        'meets_criteria': it.get('meets_criteria', False)
                    }
                    for it in result['iterations']
                ],
                'xgb_probabilities': result.get('xgb_probabilities', None)
            }
            final_results.append(final_result)
        
        summary_data['final_results'] = final_results
        
        # debug: avoid incompatible JSON format errors
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (str, int, float, type(None))):
                return obj
            else:
                return str(obj)  

        try:
            safe_summary_data = convert_for_json(summary_data)
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(safe_summary_data, f, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as e:
            print(f"JSON save failed: {e}")
        
        return all_hybrid_results

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    args = get_config()

    if not args.model_save_path or not os.path.exists(args.model_save_path):
        raise ValueError(f"Model path not provided or path incorrect'")

    processors = {
        "restaurant": RestaurantProcessor,
        "laptop": LaptopProcessor,
        "tweet": TweetProcessor,
    }
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    args.processor = processors[task_name]()

    model_classes = {
        'cnn': CNN,
        'fc': BertForSequenceClassification,
        'clstm': CLSTM,
        'pf_cnn': PF_CNN,
        'tcn': TCN,
        'bert_pf': Bert_PF,
        'bbfc': BBFC,
        'tc_cnn': TC_CNN,
        'ram': RAM,
        'ian': IAN,
        'atae_lstm': ATAE_LSTM,
        'aoa': AOA,
        'memnet': MemNet,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'mgan': MGAN,
        'bert_ian': BERT_IAN,
        'tc_swem': TC_SWEM,
        'mlp': MLP,
        'aen': AEN_BERT,
        'td_bert': TD_BERT,
        'td_bert_qa': TD_BERT_QA,
        'dtd_bert': DTD_BERT,
    }
    args.model_class = model_classes[args.model_name.lower()]

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}".format(args.device, args.n_gpu))

    tester = Tester(args)

    try:
            # adapt for pytorch 2.6+
            saved_model = torch.load(args.model_save_path, weights_only=False)
    except Exception as e:
            try:
                saved_model = torch.load(args.model_save_path, map_location=args.device, weights_only=False)
            except Exception as e2:
                raise RuntimeError(f"failed to load model from {args.model_save_path}")

    saved_model.to(args.device)
    saved_model.eval()
    
    if args.do_hybrid_predict:
        tester.hybrid_predict_with_xgboost(saved_model)
    elif args.do_iterative_predict:
        tester.iterative_predict_with_shap(saved_model)
    else:
        if args.do_predict:
             tester.do_predict_with_model(saved_model)
         
 