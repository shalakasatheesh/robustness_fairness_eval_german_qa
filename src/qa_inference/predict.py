from transformers import pipeline
from transformers import AutoTokenizer, BloomForQuestionAnswering
from torch.utils.data import Dataset

def predict(data: Dataset, model_checkpoint: str, device: str, BLOOM: bool = False):
    '''
    Returns:
    A list of answers of the format List[Dict['score': float, 'start': int32, 'end': int32, 'answer': str]]
    ''' 
    if not BLOOM:
        # create a pipeline for the task
        question_answerer = pipeline("question-answering", model=model_checkpoint, tokenizer=model_checkpoint, 
                                     handle_impossible_answer=False, do_lower_case=False, device=device)
    elif BLOOM:
        # create a pipeline for the task
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = BloomForQuestionAnswering.from_pretrained(model_checkpoint)
        question_answerer = pipeline('question-answering', model=tokenizer, tokenizer=model, 
                                     handle_impossible_answer=False, do_lower_case=False)
    return question_answerer(data)