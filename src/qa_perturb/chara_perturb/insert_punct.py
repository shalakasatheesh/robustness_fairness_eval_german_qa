from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from datasets import Dataset
import string
import random
import re

'''
references = https://github.com/ranvijaykumar/typo
'''

class InsertPunctuation():
    def __init__(self, data: Dataset, data_field: str='question', max_perturbs: int=1, length_of_word_to_perturb: int=2):
        self.data: Dataset = data
        self.data_field = data_field.lower() # specify if question or context
        self.name: str = None
        self.max_perturbs = max_perturbs
        self.length_of_word_to_perturb = length_of_word_to_perturb
        self.total_perturbations: int = 0 # total characters that have been perturbed per text
        self.perturbed_data: List[Dict[str, Union[str, Dict[str, Union[int, str, str, str]], List[Dict[str, Union[str, str, int, str]]]]]] = [] 
    
    def execute_insertion(self, word: str):
        '''
        Function to insert a random punctuation at a random position.

        Inputs:
        -------
        word: The word to be perturbed

        Returns:
        --------
        new_word: The resultant perturbed word
        '''
        new_word: str = word
        for _ in range(self.max_perturbs): 
            self.chosen_index = None
            chosen_punct = random.choice(string.punctuation)
            indices = [m.start() for m in re.finditer(r'\w', new_word)]
            if len(indices) > self.length_of_word_to_perturb: 
                self.chosen_index = random.randint(1,len(indices)-2)
                # print("Inserting", chosen_punct, "at index", self.chosen_index, "from", new_word)
                # print("Result:", new_word[:self.chosen_index]+chosen_punct+new_word[self.chosen_index:])
                new_word = new_word[:self.chosen_index]+chosen_punct+new_word[self.chosen_index:]   
                self.total_perturbations += 1
        return new_word

    def insert_punct(self):
        '''
        Function that perturbs the given dataset
        '''
        unaltered_data: List = []

        # enumerate through the dataset
        for index, sample in enumerate(tqdm(self.data)):
            self.name = 'insert_punctuation'
            self.total_perturbations = 0
            self.perturbed_data.append({'info': self.name, 
                                        'Input': {'id': sample['id'], 'context': sample['context'], 
                                        'question': sample['question'], 'answers': sample['answers']},
                                        'Perturbation':[],
                                        'Output': []})

            text = sample[self.data_field]
            new_text = self.execute_insertion(sample[self.data_field])
            self.perturbed_data[index]['Perturbation'].append({'data_field': self.data_field,
                                                               'original_'+self.data_field: text,})

            if self.total_perturbations == 0:
                unaltered_data.append(sample['id'])
            else:
                self.perturbed_data[index]['Output'].append({'data_field': self.data_field,
                                                            'total_perturbations': self.total_perturbations,
                                                            'perturbed_'+self.data_field: new_text})
        
        return self.perturbed_data, unaltered_data 
