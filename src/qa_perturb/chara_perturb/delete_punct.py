from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from datasets import Dataset
import string

'''
references = https://github.com/ranvijaykumar/typo
'''

class DeletePunctuation():
    def __init__(self, data: Dataset, data_field: str='question'):
        self.data: Dataset = data
        self.data_field = data_field.lower() # specify if question or context
        self.name: str = None
        self.total_perturbations: int = 0 # total characters that have been perturbed per text
        self.perturbed_data: List[Dict[str, Union[str, Dict[str, Union[int, str, str, str]], List[Dict[str, Union[str, str, int, str]]]]]] = [] 
    
    
    def execute_deletion(self, text: str):
        '''
        Function to delete all punctuations from the sentence

        Inputs:
        -------
        text: The text to be perturbed

        Returns:
        --------
        The resultant perturbed text
        '''
        new_text = []
        puncts = set(string.punctuation)   
        for ch in text:
            if ch not in puncts:
                new_text.append(ch)
            else:
                self.total_perturbations += 1
        return "".join(new_text)

    def delete_punct(self):
        '''
        Function that perturbs the given dataset
        '''
        unaltered_data: List = []

        # enumerate through the dataset
        for index, sample in enumerate(tqdm(self.data)):
            self.name = 'delete_punctuation'
            self.total_perturbations = 0
            self.perturbed_data.append({'info': self.name, 
                                        'Input': {'id': sample['id'], 'context': sample['context'], 
                                        'question': sample['question'], 'answers': sample['answers']},
                                        'Perturbation':[],
                                        'Output': []})

            text = sample[self.data_field]
            new_text = self.execute_deletion(sample[self.data_field])
            self.perturbed_data[index]['Perturbation'].append({'data_field': self.data_field,
                                                               'original_'+self.data_field: text,})

            if self.total_perturbations == 0:
                unaltered_data.append(sample['id'])
            else:
                self.perturbed_data[index]['Output'].append({'data_field': self.data_field,
                                                            'total_perturbations': self.total_perturbations,
                                                            'perturbed_'+self.data_field: new_text})
        
        return self.perturbed_data, unaltered_data 
