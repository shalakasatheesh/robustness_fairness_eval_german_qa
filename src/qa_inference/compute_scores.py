from evaluate import load

def calculate(predictions, references, metric: str = '/home/IAIS/ssatheesh/home/projects/thesis_code/src/squad/squad.py'):
    '''
    metric: File that will be used for computing score. 
            A module from evaluate_modules.metrics can also be directly used.
    predictions:  List[Dict['score': float, 'start': int32, 'end': int32, 'answer': str]]
    references: List[Dict['answers': ,'id': int]]

    Returns:
    --------
    Dict['exact_match': float, 'f1': float]
    '''
    metric = load(metric)
    results = metric.compute(predictions=predictions, references=references)
    return results