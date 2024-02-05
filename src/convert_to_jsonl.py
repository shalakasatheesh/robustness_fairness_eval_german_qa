import json
from datasets import load_dataset
import argparse

'''
References:
https://discuss.huggingface.co/t/question-answering-bot-fine-tuning-with-custom-dataset/4412/2
'''

# input_filename = '/home/IAIS/ssatheesh/home/projects/thesis_code/data/perturbations/insert_chara_verb.json'

def main(input_filename:str):
    # output_filename = input_filename.split(".")[0]+'.jsonl'
    output_filename = '/home/IAIS/ssatheesh/home/projects/thesis_code/data/perturbations/germanquad_train_make_typo_qwertz_question.jsonl'

    with open(input_filename) as f:
        dataset = json.load(f)

    with open(output_filename, "w", encoding='utf8') as f:
        for data in dataset:
            try:
                if data['Output'][0]['data_field'] == 'question':
                    question = data['Output'][0]['perturbed_question']
                    context = data['Input']['context']
                elif data['Output'][0]['data_field'] == 'context':
                    context = data['Output'][0]['perturbed_context']
                    question = data['Input']['question']

                idx = data['Input']['id']
                answers = data['Input']['answers']
                f.write(
                    json.dumps(
                        {
                            "id": idx,
                            "context": context,
                            "question": question,
                            "answers": answers,
                        }, ensure_ascii=False)
                )
                f.write("\n")
            except:
                print("ID "+str(data['Input']['id'])+" has not been perturbed")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename', help='enter the path to .json file', required=True)
    args = parser.parse_args()
    main(input_filename=args.input_filename)