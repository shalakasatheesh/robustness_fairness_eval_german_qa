from datasets import Dataset
from tqdm import tqdm
from typing import List
import copy
import random
import re
import itertools
from utils import initialise_models, get_word

class SwapWords():
    def __init__(self, data: Dataset, data_field: str='question'):
        self.data: Dataset = data
        self.data_field = data_field
        self.name: str = 'new_pertubation_name' # be as descriptive as possible
        self.perturbed_data = []
        self.total_perturbations = 0

    def swap_words(self):
        '''
        Add details about the peturbation here.
        '''
        self.name = 'swap_words'
        unaltered_data: List = []

        nlp_stanza = initialise_models.initialise_stanza(language='de')


        for index, sample in enumerate(tqdm(self.data)):
            self.total_perturbations = 0
            self.perturbed_data.append({'info': self.name, 
                                        'Input': {'id': sample['id'], 'context': sample['context'], 
                                        'question': sample['question'], 'answers': sample['answers']},
                                        'Perturbation':[],
                                        'Output': []})
            doc = nlp_stanza(sample[self.data_field]) # process the raw text data

            words = list(itertools.chain.from_iterable([sent.words for sent in (doc.sentences)]))
            words_to_swap = get_word(words=words,
                                    max_words=2, 
                                    length_of_word_to_perturb=2)
            
            word1 = words_to_swap[0].text
            word2 = words_to_swap[1].text
            
            # swap
            new_doc = copy.deepcopy(doc.text)
            try:
                temp_1 = re.sub(r"\b{}\b".format(word1), "TEMP", new_doc)
                temp_2 = re.sub(r"\b{}\b".format(word2), word1, temp_1)
                result = re.subn(r"\bTEMP\b", word2, temp_2)
                new_doc = result[0]
                self.total_perturbations = result[1]
                # new_doc = new_doc.replace(" "+word1, "TEMP").replace(" "+word2, " "+word1).replace("TEMP", " "+word2)
                self.perturbed_data[index]['Perturbation'].append({'data_field': self.data_field,
                                                                'original_'+self.data_field: doc.text,
                                                                'perturbed_words': [word1, word2],})
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
    

    