from datasets import Dataset
from transformers import pipeline, set_seed
from tqdm import tqdm

set_seed(42)

class BackTranslate():
    def __init__(self, data: Dataset, data_field: str='question', translation_model='facebook/nllb-200-3.3B'):
        self.data: Dataset = data
        self.data_field = data_field
        self.name: str = 'back_translate' # be as descriptive as possible
        self.translation_model = translation_model
        self.perturbed_data = []

    def back_translate(self):
        '''
        Translate given instance of the dataset from German to English 
        and then back to German using a translation model.
        '''
        text_En2De = pipeline('translation', model=self.translation_model, tokenizer=self.translation_model, src_lang="eng_Latn", tgt_lang='deu_Latn')
        text_De2En = pipeline('translation', model=self.translation_model, tokenizer=self.translation_model, src_lang="deu_Latn", tgt_lang='eng_Latn')
        for index, sample in enumerate(tqdm(self.data)):
            self.perturbed_data.append({'info': self.name, 
                                        'Input': {'id': sample['id'], 'context': sample['context'], 
                                        'question': sample['question'], 'answers': sample['answers']},
                                        'Output': []})
            text = sample[self.data_field]
            new_text_en = text_De2En(text)[0]['translation_text']
            new_text_de = text_En2De(new_text_en)[0]['translation_text']
            self.perturbed_data[index]['Output'].append({'data_field': self.data_field,
                                                         'original_'+self.data_field: text,
                                                         'perturbed_'+self.data_field: new_text_de})
        return self.perturbed_data

    