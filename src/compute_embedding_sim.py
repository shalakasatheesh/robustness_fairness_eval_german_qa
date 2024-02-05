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

def get_similarity_score(vec_1, vec_2):
    '''
    Get similarity score between two vectors.
    '''
    # similarity = 1 - cosine(vec_1, vec_2)
    return dot(vec_1, vec_2)/(norm(vec_1)*norm(vec_2))

def main(model_name, dataset_1, dataset_2, folder_to_save_results):

    german_quad_results = pd.read_json(dataset_1)
    perturbed_results = pd.read_json(dataset_2)
    compare = german_quad_results.merge(perturbed_results, on=['id',],
                                    suffixes=['_german_quad','_perturbed'])
    
    questions_1 = list(compare['question_german_quad'])
    questions_2 = list(compare['question_perturbed'])
    ids = list(perturbed_results['id'])

    dataframe_dict = []

    for id, question_1, question_2 in zip(tqdm.tqdm(ids), questions_1, questions_2):
        get_embeddings_1 = GetEmbeddings(text=question_1, model_checkpoint=model_name)
        sentence_vecs_1 = get_embeddings_1.get_token_vec_mean()

        get_embeddings_2 = GetEmbeddings(text=question_2, model_checkpoint=model_name)
        sentence_vecs_2 = get_embeddings_2.get_token_vec_mean()

        dataframe_dict.append({'id': id, 
                            'questions': question_1,
                            'questions_perturbed': question_2,
                            'similarity_scores': get_similarity_score(sentence_vecs_1, sentence_vecs_2)})
    
    dataframe = pd.DataFrame(dataframe_dict)

    if folder_to_save_results:
        dataframe.to_csv(folder_to_save_results+"/sim_scores_"+compare['data_type_perturbed'][0]+".csv")
    
    return dataframe

if __name__ == "__main__":
    tokenizer_de = AutoTokenizer.from_pretrained('deepset/gelectra-large-germanquad')
    tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='enter model name', default="deepset/gelectra-large-germanquad", required=False)
    parser.add_argument('--dataset_1', help='enter the path to first dataset', required=False)
    parser.add_argument('--dataset_2', help='enter the path to second dataset', required=False)
    parser.add_argument('--folder_to_save_results', help='enter the path to save the results', required=False)
    args = parser.parse_args()
    print(main(model_name=args.model_name, 
               dataset_1=args.dataset_1, 
               dataset_2=args.dataset_2, 
               folder_to_save_results=args.folder_to_save_results))