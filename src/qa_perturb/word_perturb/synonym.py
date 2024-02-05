from datasets import Dataset, load_dataset
from transformers import pipeline
from tqdm import tqdm
from typing import List
import itertools
import copy
from utils import initialise_models, get_verb

class Synonym():
    def __init__(self, data: Dataset, data_field: str='question', model: str="xlm-roberta-base"):
        self.data: Dataset = data
        self.data_field = data_field
        self.name: str = 'new_pertubation_name' # be as descriptive as possible
        self.model = model
        self.perturbed_data = []
        self.total_perturbations = 0
    
    def replace_with_contextual_synonym(self):
        '''
        Replace a word with contextual synonym. 
        Current implementation only supports retrieving synonyms for verbs.
        '''
        self.name = 'replace_with_contextual_synonym'
        unaltered_data: List = []

        nlp_stanza = initialise_models.initialise_stanza(language='de')
        fill_mask_model = pipeline("fill-mask", model=self.model)

        # enumerate through the dataset
        for index, sample in enumerate(tqdm(self.data)):
            self.total_perturbations = 0
            self.perturbed_data.append({'info': self.name, 
                                        'Input': {'id': sample['id'], 'context': sample['context'], 
                                        'question': sample['question'], 'answers': sample['answers']},
                                        'Perturbation':[],
                                        'Output': []})
            doc = nlp_stanza(sample[self.data_field]) # process the raw text data
            words = list(itertools.chain.from_iterable([sent.words for sent in (doc.sentences)])) # get all the words from the text
            new_doc = copy.deepcopy(doc.text)
            
            words_to_replace = get_verb(words=words, max_words=1, length_of_word_to_perturb=2)

            for word in words_to_replace:
                masked_sentence = new_doc.replace(" "+word.text, " <mask>", 1) # create masked sentence by replacing  'VERB'
                try:
                    replacement_candidates = fill_mask_model(masked_sentence)# replace masked word with contextual synonym
                    if len(replacement_candidates) == 1:
                        replacement_candidates = (list(itertools.chain(*replacement_candidates)))
                    if word.text != replacement_candidates[0]['token_str']:
                        replacement = replacement_candidates[0]
                    else:
                        replacement = replacement_candidates[1]
                    self.total_perturbations += 1
                    self.perturbed_data[index]['Perturbation'].append({'data_field': self.data_field,
                                                                        'original_word': word.text,
                                                                        'perturbed_word': replacement['token_str'], 
                                                                        'masked_sentence': masked_sentence,})
                except:
                    pass
                    # print(len(masked_sentence))

                if self.total_perturbations == 0:
                    unaltered_data.append(sample['id'])
                else:
                    self.perturbed_data[index]['Output'].append({'data_field': self.data_field,
                                                                'total_perturbations': self.total_perturbations,
                                                                'perturbed_'+self.data_field: replacement['sequence']})
        return self.perturbed_data, unaltered_data