from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from datasets import Dataset
import string

'''
References:
1. Code for translation_function() was taken from: 
    https://stackoverflow.com/questions/2054746/how-to-search-and-replace-utf-8-special-characters-in-python
'''

class ReplaceUmlaute():
    def __init__(self, data: Dataset, data_field: str='question'):
        self.data: Dataset = data
        self.data_field = data_field.lower() # specify if question or context
        self.name: str = None
        self.perturbed_data: List[Dict[str, Union[str, Dict[str, Union[int, str, str, str]], List[Dict[str, Union[str, str, int, str]]]]]] = [] 
    
    
    def translation_function(self, text: str):
        '''
        Function to replace umlaute with equivalent 
        characters from the English alphabet.

        Inputs:
        -------
        text: The text to be perturbed

        Returns:
        --------
        The resultant perturbed text
        '''
        translation_table = {ord('ä'):'ae', ord('Ä'):'AE',
                             ord('ü'):'ue', ord('Ü'):'UE', 
                             ord('ö'):'oe', ord('Ö'):'OE', 
                             ord('ß'):'ss'}
        return text.translate(translation_table)

    def replace_umlaute(self):
        '''
        Function that perturbs the given dataset
        '''
        unaltered_data: List = []

        # enumerate through the dataset
        for index, sample in enumerate(tqdm(self.data)):
            self.name = 'replace_umlaute'
            self.perturbed_data.append({'info': self.name, 
                                        'Input': {'id': sample['id'], 'context': sample['context'], 
                                        'question': sample['question'], 'answers': sample['answers']},
                                        'Output': []})

            text = sample[self.data_field]
            new_text = self.translation_function(sample[self.data_field])
            self.perturbed_data[index]['Output'].append({'data_field': self.data_field,
                                                         'original_'+self.data_field: text,
                                                         'perturbed_'+self.data_field: new_text})
        
        return self.perturbed_data, unaltered_data 
