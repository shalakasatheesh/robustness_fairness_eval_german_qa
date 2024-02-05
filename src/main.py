from datasets import load_dataset
from torch.utils.data import Dataset
import pandas as pd
from  tqdm import tqdm
import random
import warnings
import argparse

from qa_inference import utils
from qa_inference import predict
from qa_inference import calculate

def get_data(dataset_path: str = "deepset/germanquad", split: str = "test", from_file: bool=False) -> Dataset:
    '''
    Converts dataset into the format required for prediction
    '''
    print("Predicting on data from", dataset_path+". Using", split, "split.")
    if from_file:
        dataset = load_dataset("json", data_files=dataset_path, split='train') # split is not train. this is a huggingface bug
    else:
        dataset = load_dataset(path=dataset_path, split=split)
    data = utils.QuestionContextDataset(dataset, question='question', context='context')
    return data

def get_output(data: Dataset, model_checkpoint: str, BLOOM: bool = False, device: str = "cuda:0"):
    '''
    Prints the predicted answer from the model
    '''
    for output in predict(data, model_checkpoint=model_checkpoint, BLOOM=BLOOM, device=device):
        print(output)

def get_scores(data: Dataset, model_checkpoint: str, data_type: str, device: str = "cuda:0", BLOOM: bool = False):
    '''
    First gets predictions and then computes scores
    '''
    outputs = []
    predictions = []
    references = []
    dataframe_dict = []

    print('Predicting..')
    for output in tqdm(predict(data, model_checkpoint=model_checkpoint, BLOOM=BLOOM, device=device)):
        warnings.filterwarnings("ignore") # see issue here: https://github.com/huggingface/transformers/issues/23003
        outputs.append(output)
    
    print('Computing scores..')
    # Iterate through predictions and compute the F1 and EM scores
    for i in tqdm(range(len(data))):
        prediction = {'prediction_text': outputs[i]['answer'],'id': data.get_id(i)}
        reference = {'answers': data.get_gold_answers(i),'id': data.get_id(i)}
        predictions.append(prediction)
        references.append(reference)
        scores = calculate([prediction], [reference])
        dataframe_dict.append({'id': data.get_id(i), 
                               'question': data[i]['question'],
                               'answers': data.get_gold_answers(i), 
                               'prediction_text': outputs[i]['answer'], 
                               'exact_match': scores['exact_match'], 
                               'f1': scores['f1'],
                               'confidence_score':  outputs[i]['score'],
                               'data_type': data_type})
    
    dataframe = pd.DataFrame(dataframe_dict)
    final_score = calculate(predictions, references)
    return dataframe, final_score

def main(model_name: str, path_to_dataset: str, folder_to_save_results: str, data_type: str):
    random.seed(42)
    model_checkpoint=model_name 
    BLOOM = False

    print("Using model:", model_checkpoint)

    if path_to_dataset:
        data = get_data(dataset_path=path_to_dataset, from_file=True) # test on perturbed data
        dataframe, final_score = get_scores(data=data, model_checkpoint=model_checkpoint, data_type=data_type, device="cuda:0", BLOOM=BLOOM)
    else:
        data = get_data() # test on regular data
        data_type = 'germanquad_test'
        dataframe, final_score = get_scores(data=data, model_checkpoint=model_checkpoint, data_type=data_type, device="cuda:0", BLOOM=BLOOM)

    if folder_to_save_results:
        dataframe.to_json(folder_to_save_results+"/"+model_checkpoint.split('/')[-1]+"_"+data_type+".json", force_ascii=False)
    else:
        dataframe.to_json("results/"+model_checkpoint.split('/')[-1]+"_"+data_type+".json", force_ascii=False)
    print(final_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='enter model name', default="deepset/xlm-roberta-base-squad2", required=False)
    parser.add_argument('--path_to_dataset', help='enter the path to dataset to be evaluated', required=False)
    parser.add_argument('--folder_to_save_results', help='enter path to the folder to save results', required=False)
    parser.add_argument('--data_type', help='enter type of dataset being evaluated. eg: chara_perturb', default="peturbed", required=False)
    args = parser.parse_args()
    main(model_name=args.model_name, 
         path_to_dataset=args.path_to_dataset, 
         folder_to_save_results=args.folder_to_save_results, 
         data_type=args.data_type
         )
    
'''

model_checkpoint="deepset/gelectra-base-germanquad" # monolingual
model_checkpoint = 'deutsche-telekom/bert-multi-english-german-squad2' # multilingual
model_checkpoint="deepset/xlm-roberta-base-squad2" # multilingual 
model_checkpoint="deutsche-telekom/electra-base-de-squad2" # monolingual

'/home/IAIS/ssatheesh/home/projects/thesis_code/data/perturbations/first_perturb_encoding.jsonl'
'''