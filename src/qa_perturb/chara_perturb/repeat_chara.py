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

class RepeatChara():
    def __init__(self, data: Dataset, data_field: str='question', max_perturbs: int=1, max_words: int=1, length_of_word_to_perturb: int=2, pos_tag: str=None):
        self.max_perturbs: int = max_perturbs # Maximum number of characters to be perturbed per word.
        self.max_words: int = max_words # Maximum number of words to be perturbed.
        self.length_of_word_to_perturb = length_of_word_to_perturb # perturbation done only if the length of the word is greater than this number
        self.pos_tag = pos_tag # default_value = None. Can also take value = 'verb'
        self.data: Dataset = data
        self.data_field = data_field.lower() # specify if question or context
        self.name: str = None
        self.total_perturbations: int = 0 # total perturbations that have been made
        self.total_word_perturbations: int = 0 # total words that have been perturbed
        self.total_chara_perturbations: int = 0 # total characters that have been perturbed per word
        self.total_unique_words: int = 0
        self.token = None
        self.chosen_index = None
        self.perturbed_data: List[Dict[str, Union[str, Dict[str, Union[int, str, str, str]], List[Dict[str, Union[str, str, int, str]]]]]] = [] 
    
    def execute_repetition(self, word: str):
            
        '''
        Function to alter the word by repeating a character at a 
        randomly chosen index

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
            indices = [m.start() for m in re.finditer(r'\w', new_word)]
            if len(indices) > self.length_of_word_to_perturb: 
                self.chosen_index = random.randint(1,len(indices)-2)
                # print("Repeating", new_word[self.chosen_index], "at index", self.chosen_index, "from", new_word)
                # print("Result:", new_word[:self.chosen_index]+new_word[self.chosen_index]+new_word[self.chosen_index:])
                new_word = new_word[:self.chosen_index]+new_word[self.chosen_index]+new_word[self.chosen_index:]   
                self.total_perturbations += 1
                self.total_chara_perturbations += 1
            else:
                pass
        self.total_unique_words += 1
        return new_word
    
    def repeat_chara(self):
        '''
        Function that perturbs the given dataset
        '''
        unaltered_data: List = []
        nlp_stanza = initialise_models.initialise_stanza(language='de')
        # enumerate through the dataset
        for index, sample in enumerate(tqdm(self.data)):
            self.total_perturbations = 0
            self.total_unique_words = 0
            doc = nlp_stanza(sample[self.data_field]) # process the raw text data
            words = list(itertools.chain.from_iterable([sent.words for sent in (doc.sentences)])) # get all the words from the text
            new_doc = copy.deepcopy(doc.text)

            if self.pos_tag is None: # get random words to perturb
                random_words = get_word(words, self.max_words, self.length_of_word_to_perturb) 
                self.name = 'repeat_chara_random_word'
            elif self.pos_tag.lower() == 'verb': # get random verbs to perturb
                random_words = get_verb(words, self.max_words, self.length_of_word_to_perturb)
                self.name = 'repeat_chara_random_verb'
            
            self.perturbed_data.append({'info': self.name, 
                                        'Input': {'id': sample['id'], 'context': sample['context'], 
                                        'question': sample['question'], 'answers': sample['answers']},
                                        'Perturbation':[],
                                        'Output': []})
            self.total_word_perturbations = 0
            # execute perturbation on the text
            for token in random_words:
                word = token.text
                self.total_chara_perturbations = 0
                try:
                    new_word = self.execute_repetition(word)
                    result = re.subn(r"\b{}\b".format(word), new_word, new_doc)
                    new_doc = result[0]
                    self.total_word_perturbations = self.total_word_perturbations+result[1]
                    #new_doc = new_doc.replace(" "+word, " "+new_word, 1)
                    self.perturbed_data[index]['Perturbation'].append({'data_field': self.data_field,
                                                                    'original_word': word,
                                                                    'perturbed_word': new_word, })
                except:
                    pass

            if self.total_perturbations == 0:
                unaltered_data.append(sample['id'])
            else:
                self.perturbed_data[index]['Output'].append({'data_field': self.data_field,
                                                            'total_chara_perturbations_per_word': self.total_chara_perturbations,
                                                            'total_unique_words': self.total_unique_words, 
                                                            'total_words_perturbed': self.total_word_perturbations,
                                                            'total_perturbations': self.total_perturbations,
                                                            'perturbed_'+self.data_field: new_doc})
        
        return self.perturbed_data, unaltered_data