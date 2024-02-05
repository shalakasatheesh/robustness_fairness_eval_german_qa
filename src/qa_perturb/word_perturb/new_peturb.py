from datasets import Dataset

class NewPerturb():
    def __init__(self, data: Dataset, data_field: str='question', max_perturbs=1):
        self.max_perturbs: int = max_perturbs
        self.data: Dataset = data
        self.data_field = data_field
        self.name: str = 'new_pertubation_name' # be as descriptive as possible

    def execute_perturbation(self):
        '''
        Add details about the peturbation here.
        '''
        return NotImplementedError
    

    