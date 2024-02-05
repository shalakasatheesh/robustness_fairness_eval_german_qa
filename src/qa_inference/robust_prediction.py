'''
Code heavily based on: 
1. Evaluating QA: Metrics, Predictions, and the Null Response 
https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Using-the-null-threshold
'''

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, default_data_collator
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from evaluate import load
import tqdm
import collections

class RobustPredictor():
    def __init__(self, model_checkpoint: str, sep_token_id: int, dataset_path: str, split: str, from_file: str):
        self.model_checkpoint = model_checkpoint
        self.sep_token_id = sep_token_id # 102 or #3 or #103
        self.dataset_path = dataset_path
        self.split = split
        self.from_file = from_file
        self.dataset = self.get_data()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.max_lenth = 384
        self.stride = 128
        self.batch_size = 1
        self.nbest = 2

    def get_data(self):
        print("Predicting on data from", self.dataset_path+". Using", self.split, "split.")
        if self.from_file:
            self.dataset = load_dataset("json", data_files=self.dataset_path, split=self.split) # split is not train. this is a huggingface bug
        else:
            self.dataset = load_dataset(path=self.dataset_path, split=self.split)
        return self.dataset
    
    def preprocess_validation_examples(self, dataset):
        questions = [q.strip() for q in dataset["question"]]
        inputs = self.tokenizer(questions,
                                dataset["context"],
                                truncation="only_second")
        return inputs

    def get_outputs(self):
        test_tokens = self.dataset.map(self.preprocess_validation_examples,
                                       batched=True,
                                       remove_columns=self.dataset.column_names)
        test_tokens.set_format("torch")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Inferencing on the model:", self.model_checkpoint, ", using", device)
        trained_model = AutoModelForQuestionAnswering.from_pretrained(self.model_checkpoint).to(device)
        test_dataloader = DataLoader(test_tokens,
                                     shuffle=False,
                                     collate_fn=default_data_collator,
                                     batch_size=self.batch_size,)
        outputs = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm.tqdm(test_dataloader)):
                outputs.append(self.model(**batch))
        return test_tokens, outputs
    
    def to_list(self, tensor):
        return tensor.detach().cpu().tolist()

    def preliminary_predictions(self, start_logits, end_logits, input_ids):
        # convert tensors to lists
        start_logits = self.to_list(start_logits)[0]
        end_logits = self.to_list(end_logits)[0]
        tokens = self.to_list(input_ids)

        # sort our start and end logits from largest to smallest, keeping track of the index
        start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
        end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)
        
        start_indexes = [idx for idx, logit in start_idx_and_logit[:self.nbest]]
        end_indexes = [idx for idx, logit in end_idx_and_logit[:self.nbest]]
        # question tokens are between the CLS token (101, at position 0) and first SEP (102) token 
        question_indexes = [i+1 for i, token in enumerate(tokens[1:tokens.index(self.sep_token_id)])]

        # keep track of all preliminary predictions
        PrelimPrediction = collections.namedtuple(
            "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
            )
        prelim_preds = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                # throw out invalid predictions
                if start_index in question_indexes:
                    continue
                if end_index in question_indexes:
                    continue
                if end_index < start_index:
                    continue
                prelim_preds.append(
                    PrelimPrediction(
                        start_index = start_index,
                        end_index = end_index,
                        start_logit = start_logits[start_index],
                        end_logit = end_logits[end_index]
                    )
                )
        # sort prelim_preds in descending score order
        prelim_preds = sorted(prelim_preds, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
        return prelim_preds
    
    def get_clean_text(self, tokens, tokenizer):
        text = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(tokens)
            )
        # Clean whitespace
        text = text.strip()
        text = " ".join(text.split())
        return text
    
    def best_predictions(self, prelim_preds, tokens, start_logits, end_logits, tokenizer):
        start_logits = self.to_list(start_logits)[0]
        end_logits = self.to_list(end_logits)[0]
        # keep track of all best predictions

        # This will be the pool from which answer probabilities are computed 
        BestPrediction = collections.namedtuple(
            "BestPrediction", ["text", "start_logit", "end_logit"]
        )
        nbest_predictions = []
        seen_predictions = []
        for pred in prelim_preds:
            if len(nbest_predictions) >= self.nbest: 
                break
            if pred.start_index > 0: # non-null answers have start_index > 0

                toks = tokens[pred.start_index : pred.end_index+1]
                text = self.get_clean_text(toks, tokenizer)

                # if this text has been seen already - skip it
                if text in seen_predictions:
                    continue

                # flag text as being seen
                seen_predictions.append(text) 

                # add this text to a pruned list of the top nbest predictions
                nbest_predictions.append(
                    BestPrediction(
                        text=text, 
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit
                        )
                    )
            
        # Add the null prediction
        nbest_predictions.append(
            BestPrediction(
                text="", 
                start_logit=start_logits[0], 
                end_logit=end_logits[0]
                )
            )
        return nbest_predictions
    
    def prediction_probabilities(self, predictions):
        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        all_scores = [pred.start_logit+pred.end_logit for pred in predictions] 
        return softmax(np.array(all_scores))
    
    def compute_score_difference(self, predictions):
        """ Assumes that the null answer is always the last prediction """
        score_null = predictions[-1].start_logit + predictions[-1].end_logit
        score_non_null = predictions[0].start_logit + predictions[0].end_logit
        return score_null - score_non_null
    
    def get_robust_prediction(self, outputs, test_tokens, null_threshold=1.0):
        start_logits = outputs['start_logits']
        end_logits = outputs['end_logits']
        # get sensible preliminary predictions, sorted by score
        prelim_preds = self.preliminary_predictions(start_logits, 
                                                    end_logits, 
                                                    test_tokens['input_ids'])
        
        # narrow that down to the top nbest predictions
        nbest_preds = self.best_predictions(prelim_preds, 
                                            test_tokens['input_ids'], 
                                            start_logits, 
                                            end_logits, 
                                            self.tokenizer)
        
        # compute the probability of each prediction - nice but not necessary
        probabilities = self.prediction_probabilities(nbest_preds)
            
        # compute score difference
        score_difference = self.compute_score_difference(nbest_preds)

        # if score difference > threshold, return the null answer
        if score_difference > null_threshold:
            return "", probabilities[-1]
        else:
            if (nbest_preds[0].text == "[SEP]"):
                return "", probabilities[0]
            elif (nbest_preds[0].text == "[CLS]"): 
                return "", probabilities[0]
            elif (nbest_preds[0].text == "."): 
                return "", probabilities[0]
            else:
                return nbest_preds[0].text, probabilities[0]
            
