from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer, BertForQuestionAnswering
import torch
import numpy as np
import pandas  as pd
from numpy import dot
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from typing import List, Dict, DefaultDict, Tuple
import argparse
import tqdm

from embedding import GetEmbeddings

def main(model_name, dataset_1, dataset_2, folder_to_save_results):

    german_quad_results = pd.read_json(dataset_1)
    perturbed_results = pd.read_json(dataset_2)
    compare = german_quad_results.merge(perturbed_results, on=['id',], 
                                        suffixes=['_german_quad','_perturbed'])
    
    questions_1 = list(compare['question_german_quad'])
    questions_2 = list(compare['question_perturbed'])
    ids = list(perturbed_results['id'])

    dataframe_dict = []
    get_embeddings_1 = GetEmbeddings(text=str(questions_1), model_checkpoint=model_name)
    sentence_tokens_1 = get_embeddings_1.tokenize()

    get_embeddings_2 = GetEmbeddings(text=str(questions_2), model_checkpoint=model_name)
    sentence_tokens_2 = get_embeddings_2.tokenize()

    dataframe_dict.append({'id': ids, 
                           'questions': questions_1,
                           'questions_perturbed': questions_2,
                           'sentence_tokens_1': sentence_tokens_1,
                           'no_of_tokens_sent_1': len(sentence_tokens_1),
                           'sentence_tokens_2': sentence_tokens_2, 
                           'no_of_tokens_sent_2': len(sentence_tokens_2),})
    
    dataframe = pd.DataFrame(dataframe_dict)

    if folder_to_save_results:
        dataframe.to_csv(folder_to_save_results+"/fertility_"+compare['data_type_perturbed'][0]+".csv")
    
    return dataframe_dict


if __name__ == "__main__":
    tokenizer_de = AutoTokenizer.from_pretrained('deepset/gelectra-large-germanquad')
    tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='enter model name', default="deepset/gelectra-large-germanquad", required=False)
    parser.add_argument('--dataset_1', help='enter the path to first dataset', required=False)
    parser.add_argument('--dataset_2', help='enter the path to second dataset', required=False)
    parser.add_argument('--folder_to_save_results', help='enter the path to save the results', required=False)
    args = parser.parse_args()
    main(model_name=args.model_name, 
               dataset_1=args.dataset_1, 
               dataset_2=args.dataset_2, 
               folder_to_save_results=args.folder_to_save_results)