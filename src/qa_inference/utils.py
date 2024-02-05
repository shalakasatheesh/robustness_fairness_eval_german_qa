from torch.utils.data import Dataset

'''
References: 
1. https://stackoverflow.com/questions/75932605/getting-the-input-text-from-transformers-pipeline
2. https://huggingface.co/docs/transformers/main_classes/pipelines 
'''

class QuestionContextDataset(Dataset):
    def __init__(self, dataset: Dataset, question: str, context: str):
        self.dataset = dataset
        self.question = question
        self.context = context
        self.id = dataset['id']
        self.answers = dataset['answers']

    def __len__(self):
        ''' 
        Returns total number of sample pairs in the dataset.
        '''
        return len(self.dataset)

    def __getitem__(self, index: int):
        ''' 
        Returns a particular question and it's context.
        '''
        return {"question": self.dataset[index][self.question], "context": self.dataset[index][self.context]}
    
    def get_id(self, index):
        return str(self.id[index])
    
    def get_gold_answers(self, index):
        return self.answers[index]



