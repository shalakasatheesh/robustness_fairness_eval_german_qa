from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from datasets import Dataset
import string

'''
references = https://github.com/ranvijaykumar/typo
'''

class ChangeCase():
    def __init__(self, data: Dataset, data_field: str='question', case='lower'):
        self.data: Dataset = data
        self.data_field = data_field.lower() # specify if question or context
        self.name: str = None
        self.case = case # lower or upper case
        self.perturbed_data: List[Dict[str, Union[str, Dict[str, Union[int, str, str, str]], List[Dict[str, Union[str, str, int, str]]]]]] = [] 
    
    
    def convert_to_lower(self, text: str):
        '''
        Function to convert text to lower

        Inputs:
        -------
        text: The text to be perturbed

        Returns:
        --------
        The resultant perturbed text
        '''
        return text.lower()

    def convert_to_upper(self, text: str):
        '''
        Function to convert text to upper

        Inputs:
        -------
        text: The text to be perturbed

        Returns:
        --------
        The resultant perturbed text
        '''
        return text.upper()

    def invert_case(self, text: str):
        '''
        Function to swap the case of text 

        Inputs:
        -------
        text: The text to be perturbed

        Returns:
        --------
        The resultant perturbed text
        '''
        return text.swapcase()

    def title_case(self, text: str):
        '''
        Function to convert text to title case

        Inputs:
        -------
        text: The text to be perturbed

        Returns:
        --------
        The resultant perturbed text
        '''
        return text.title()

    def change_case(self):
        '''
        Function that perturbs the given dataset
        '''
        unaltered_data: List = []
        
        # enumerate through the dataset
        for index, sample in enumerate(tqdm(self.data)):
            self.name = 'change_case_to_'+self.case.lower()
            self.perturbed_data.append({'info': self.name, 
                                        'Input': {'id': sample['id'], 'context': sample['context'], 
                                        'question': sample['question'], 'answers': sample['answers']},
                                        'Output': []})

            text = sample[self.data_field]
            if self.case.lower() == 'lower':
                new_text = self.convert_to_lower(text)
            elif self.case.lower() == 'upper':
                new_text = self.convert_to_upper(text)
            elif self.case.lower() == 'invert':
                new_text = self.invert_case(text)
            elif self.case.lower() == 'title':
                new_text = self.title_case(text)
            self.perturbed_data[index]['Output'].append({'data_field': self.data_field,
                                                         'original_'+self.data_field: text,
                                                         'perturbed_'+self.data_field: new_text})
        
        return self.perturbed_data, unaltered_data 
