import random
import re
import copy
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from datasets import Dataset
import itertools

from utils import get_word, get_verb, initialise_models

'''
references = https://github.com/ranvijaykumar/typo
'''

class SplitWord():
    def __init__(self, data: Dataset, data_field: str='question', max_words: int=1, length_of_word_to_perturb: int=2, pos_tag: str=None):
        self.max_words: int = max_words # Maximum number of words to be perturbed.
        self.length_of_word_to_perturb = length_of_word_to_perturb # perturbation done only if the length of the word is greater than this number
        self.pos_tag = pos_tag # default_value = None. Can also take value = 'verb'
        self.data: Dataset = data
        self.data_field = data_field.lower() # specify if question or context
        self.name: str = None
        self.total_perturbations: int = 0 # total words that have been perturbed
        self.token = None
        self.chosen_index = None
        self.perturbed_data: List[Dict[str, Union[str, Dict[str, Union[int, str, str, str]], List[Dict[str, Union[str, str, int, str]]]]]] = [] 
    
    def split_word(self):
        '''
        Function that perturbs the given dataset
        '''
        unaltered_data: List = []
        nlp_stanza = initialise_models.initialise_stanza(language='de')
        # enumerate through the dataset
        for index, sample in enumerate(tqdm(self.data)):
            self.total_perturbations = 0
            doc = nlp_stanza(sample[self.data_field]) # process the raw text data
            words = list(itertools.chain.from_iterable([sent.words for sent in (doc.sentences)])) # get all the words from the text
            new_doc = copy.deepcopy(doc.text)

            if self.pos_tag is None: # get random words to perturb
                random_words = get_word(words, self.max_words, self.length_of_word_to_perturb) 
                self.name = 'split_random_word'
            elif self.pos_tag.lower() == 'verb': # get random verbs to perturb
                random_words = get_verb(words, self.max_words, self.length_of_word_to_perturb)
                self.name = 'split_random_verb'

            self.perturbed_data.append({'info': self.name, 
                                        'Input': {'id': sample['id'], 'context': sample['context'], 
                                        'question': sample['question'], 'answers': sample['answers']},
                                        'Perturbation':[],
                                        'Output': []})
            
            # execute perturbation on the text
            for token in random_words:
                word = token.text
                new_word = word
                indices = [m.start() for m in re.finditer(r'\w', new_word)]
                if len(indices) > self.length_of_word_to_perturb: 
                    self.chosen_index = random.randint(1,len(indices)-2)
                    new_word = new_word[:self.chosen_index]+" "+new_word[self.chosen_index:] 
                try:
                    result = re.subn(r"\b{}\b".format(word), new_word, new_doc)
                    new_doc = result[0]
                    self.total_perturbations = self.total_perturbations+result[1]
                    # new_doc = new_doc.replace(" "+word, " "+new_word, 1)
                    self.perturbed_data[index]['Perturbation'].append({'data_field': self.data_field,
                                                                    'original_word': word,
                                                                    'perturbed_word': new_word,})
                except:
                    print('skipping ', sample['id'])
                    pass
            
            if self.total_perturbations == 0:
                unaltered_data.append(sample['id'])
            else:
                self.perturbed_data[index]['Output'].append({'data_field': self.data_field,
                                                             'total_perturbations': self.total_perturbations,
                                                             'perturbed_'+self.data_field: new_doc})
        
        return self.perturbed_data, unaltered_data