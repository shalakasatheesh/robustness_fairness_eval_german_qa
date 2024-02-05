import random
from datasets import load_dataset
import json
import pandas as pd
import os 

from qa_perturb.chara_perturb import InsertChara, DeleteChara, RepeatChara, ReplaceChara, \
    SwapChara, DeletePunctuation, ChangeCase, ReplaceUmlaute, KeyboardTypo, InsertPunctuation
from qa_perturb.word_perturb import RepeatWord, DeleteWord, SplitWord, Synonym, SwapWords
from qa_perturb.sentence_perturb import BackTranslate, RepeatSentence

def main():
    random.seed(42)
    
    # load data
    data = load_dataset("deepset/germanquad", split="test")

    # perturb data
    max_perturbs = 50
    data_field = 'context'
    Perturbation = InsertPunctuation(data, data_field=data_field, \
                                max_perturbs=max_perturbs, length_of_word_to_perturb=2)
    output = Perturbation.insert_punct()
    
    # path to save output file with perturbations
    PARENT_DIR = '/home/IAIS/ssatheesh/home/projects/thesis_code/data/perturbations/'
    DIRECTORY = 'punctuation'
    FILE_NAME = Perturbation.name+'_'+Perturbation.data_field+'_'+str(max_perturbs)+'_charas.json' 
    DIR_PATH = os.path.join(PARENT_DIR, DIRECTORY)

    try:
        os.mkdir(DIR_PATH)
    except OSError:
        pass 
    PATH = os.path.join(DIR_PATH, FILE_NAME)
    with open(PATH, "w", encoding='utf8') as f:
        f.write(json.dumps(output[0], ensure_ascii=False))

    print("Saved perturbed data at "+PATH)

    # unaltered questions/contexts
    print(output[1])

if __name__ == "__main__":
   main()