from datasets import Dataset
from tqdm import tqdm

class RepeatSentence():
    def __init__(self, data: Dataset, data_field: str='question'):
        self.data: Dataset = data
        self.data_field = data_field
        self.name: str = None # be as descriptive as possible
        self.perturbed_data = []

    def repeat_data_field(self):
        '''
        Repeat the question or context.
        '''
        self.name = 'repeat_'+self.data_field
        for index, sample in enumerate(tqdm(self.data)):
            self.perturbed_data.append({'info': self.name, 
                                        'Input': {'id': sample['id'], 'context': sample['context'], 
                                        'question': sample['question'], 'answers': sample['answers']},
                                        'Output': []})
            text = sample[self.data_field]
            new_text = text+' '+text
            self.perturbed_data[index]['Output'].append({'data_field': self.data_field,
                                                         'original_'+self.data_field: text,
                                                         'perturbed_'+self.data_field: new_text})
        print(len(self.perturbed_data))
        return self.perturbed_data

    