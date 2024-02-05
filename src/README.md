# Run robustness evaluation

## On GermanQuAD test data

    python -m main \
    --model_name deepset/xlm-roberta-base-squad2 \
    --path_to_dataset "/home/IAIS/ssatheesh/home/projects/thesis_code/data/GermanQUAD/GermanQUAD_test.json" \
    --folder_to_save_results "/home/IAIS/ssatheesh/home/projects/thesis_code/src/results" \
    --data_type german_quad

## On perturbed data

    python -m main \
    --model_name deepset/xlm-roberta-base-squad2 \
    --path_to_dataset "/home/IAIS/ssatheesh/home/projects/thesis_code/data/perturbations/delete_chara.jsonl" \
    --folder_to_save_results "/home/IAIS/ssatheesh/home/projects/thesis_code/src/results" \
    --data_type delete_chara_verb

# To perturb data

1. [Create](./qa_perturb/README.md) desired pertubation by editing `perturb.py`

2. Run the edited file to apply desired perturbation on data

        python -m perturb

3. Convert file to SQuAD format. File type will be `.jsonl`.

        python convert_to_jsonl.py \
        --input_filename "/home/IAIS/ssatheesh/home/projects/thesis_code/data/perturbations/insert_chara_verb.json"

# To calculate similarity scores between embeddings

        python compute_embedding_sim.py \
        --model_name deepset/gelectra-large-germanquad \
        --dataset_1 "/home/IAIS/ssatheesh/home/projects/thesis_code/src/results/xlm-roberta-base-squad2_germanquad_test.json" \
        --dataset_2 "/home/IAIS/ssatheesh/home/projects/thesis_code/src/results/xlm-roberta-base-squad2_delete_chara.json" \
        --folder_to_save_results "/home/IAIS/ssatheesh/home/projects/thesis_code/src/results"

# To get tokens before and after perturbation
        python compare_tokens.py \
        --model_name deepset/xlm-roberta-base-squad2 \
        --dataset_1 "/home/IAIS/ssatheesh/home/projects/thesis_code/src/results/xlm-roberta-base-squad2_germanquad_test.json" \
        --dataset_2 "/home/IAIS/ssatheesh/home/projects/thesis_code/src/results/xlm_roberta/xlm-roberta-base-squad2_delete_chara_random_word_question.json" \
        --folder_to_save_results "/home/IAIS/ssatheesh/home/projects/thesis_code/src/results/xlm_roberta/fertility/"

# Run fairness evaluation 

## To translate the templates

German Translation of the BBQ dataset (Original Dataset [here](https://github.com/nyu-mll/BBQ))
Model(s) used for translation: 
1. https://huggingface.co/facebook/nllb-200-3.3B
2. https://huggingface.co/Tanhim/translation-En2De
3. https://huggingface.co/Helsinki-NLP/opus-mt-en-de

## Generate dataset from the template

Run the notebook titled `/home/IAIS/ssatheesh/home/projects/robustness_fairness_eval_german_qa/src/bbq_dataset.ipynb`

## Run evalaution

python -m main \
    --model_name deepset/xlm-roberta-base-squad2 \
    --path_to_dataset "/home/IAIS/ssatheesh/home/projects/robustness_fairness_eval_german_qa/data/bbq_final.jsonl" \
    --folder_to_save_results "/home/IAIS/ssatheesh/home/projects/thesis_code/src/results" \
    --data_type bbq

## Estimate bias scores

Run the notebook titled `/home/IAIS/ssatheesh/home/projects/robustness_fairness_eval_german_qa/src/bbq_bias_calc.ipynb`

# References:
1. [GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval](https://aclanthology.org/2021.mrqa-1.4/)(Möller et al., MRQA 2021)
2. [NL-Augmenter 🦎 → 🐍](https://github.com/GEM-benchmark/NL-Augmenter/tree/main)(Kaustubh D. Dhole et al.)
3. [BBQ: A hand-built bias benchmark for question answering](https://aclanthology.org/2022.findings-acl.165) (Parrish et al., Findings 2022)