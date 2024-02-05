import pandas as pd
import argparse
import os
import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline, set_seed
from datasets import load_dataset

set_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--file_path', type=str, required=False)
parser.add_argument('--model', type=str, required=False)
parser.add_argument('--path_to_save', type=str, required=False)
args = parser.parse_args()

def read_json():
    '''
    Function to read the json file to be translated
    
    Inputs:
    -------
    file_path: str

    Returns:
    --------
    pd.DataFrame
    '''
    if args.file_path:
        file_path = args.file_path
    else:
        file_path = '/home/IAIS/ssatheesh/home/projects/BBQ_de/data/Gender_identity.jsonl'

    return pd.read_json(path_or_buf=file_path, lines=True)

def translate(dataset):
    '''
    Translate the English language data to German
    Inputs:
    -------
    dataset: List

    Returns:
    pd.DataFrame
    '''
    if args.model:
        model = args.model
    else:
        # model_meta = 'facebook/nllb-200-3.3B'; meta == True
        # model_tanhim  = 'Tanhim/translation-En2De'
        model = 'Helsinki-NLP/opus-mt-en-de'

    text_En2De = pipeline('translation', model=model, tokenizer=model, src_lang="eng_Latn", tgt_lang='deu_Latn')

    # return [text_En2De(data)[0]['translation_text'] for data in tqdm.tqdm(list(dataset))]
    return [text_En2De(data)[0]['translation_text'] for data in tqdm.tqdm(list(dataset))]
    
def main():
   
    print('Reading file..')
    dataset = read_json()

    dataset_coloumns = list(dataset.columns)

    print('Translating contexts..')
    context_de = translate(list(dataset['context']))
    print('Translating questions..')
    question_de = translate(list(dataset['question']))
    print('Translating answers..')
    ans0_de = translate(list(dataset['ans0']))
    ans1_de = translate(list(dataset['ans1']))   
    ans2_de = translate(list(dataset['ans2']))   
    
    print('Compiling translated dataset..')
    
    dataset_de_dict = {'example_id': dataset['example_id'],
                    'question_index': dataset['question_index'],
                    'question_polarity': dataset['question_polarity'],
                    'context_condition': dataset['context_condition'],
                    'category': dataset['category'],
                    'answer_info': dataset['answer_info'],
                    'additional_metadata': dataset['additional_metadata'],
                    'context': context_de,
                    'question': question_de,
                    'ans0': ans0_de,
                    'ans1': ans1_de,
                    'ans2': ans2_de,
                    'label': dataset['label']}

    dataset_de = pd.DataFrame.from_dict(dataset_de_dict)

    if args.path_to_save:
        file_name = '/gender_identity_helsinki_de.csv'
        print('Saving translated dataset at '+args.path_to_save+file_name)
        try:
            print('Creating folder..')
            os.mkdir(args.path_to_save)
            dataset_de.to_csv(args.path_to_save+file_name)
        except:
            print('Folder exists..')
            dataset_de.to_csv(args.path_to_save+file_name)

        print('..successfully saved!')
    else:
        print('Translated dataset not saved..')
        print(dataset_de)

if __name__ == "__main__":
   main()

# Files to translate:
# ---------------------
# 1. gender: '/home/shalaka/repos/BBQ/data/Gender_identity.jsonl'
# 2. age: '/home/shalaka/repos/BBQ/data/Age.jsonl'
# 3. disability: '/home/shalaka/repos/BBQ/data/Disability_status.jsonl'
# 4. nationality: '/home/shalaka/repos/BBQ/data/Nationality.jsonl'
# 5. physical_appearance: '/home/shalaka/repos/BBQ/data/Physical_appearance.jsonl'
# 6. race_ethnicity: '/home/shalaka/repos/BBQ/data/Race_ethnicity.jsonl'
# 7. race_x_gender: '/home/shalaka/repos/BBQ/data/Race_x_gender.jsonl'
# 8. race_x_ses: '/home/shalaka/repos/BBQ/data/Race_x_SES.jsonl'
# 9. religion: '/home/shalaka/repos/BBQ/data/Religion.jsonl'
# 10. ses: '/home/shalaka/repos/BBQ/data/SES.jsonl'
# 11. sexual_orientation: '/home/shalaka/repos/BBQ/data/Sexual_orientation.jsonl'