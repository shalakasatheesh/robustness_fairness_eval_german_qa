# Usage

### 1. `BackTranslate`

First initialise the perturbation by choosing one of the packages. Example:

    from qa_perturb.sentence_perturb import BackTranslate 
    Perturb = BackTranslate(data, data_field='question', translation_model='facebook/nllb-200-3.3B') # initialise perturbation
    
- **data_field: *str*, *default='question'***: 
    
    Can take values `'question'` or `'context'` depending on which data instance the perturbation has to be applied to.

- **translation_model: *str, default='facebook/nllb-200-3.3B'***: 

    Model from huggingface repo to be used for translation
---

Example:

    output = Perturb.back_translate() # execute perturbation on the desired data

### 2. `RepeatSentence`

First initialise the perturbation by choosing one of the packages. Example:

    from qa_perturb.sentence_perturb import RepeatSentence 
    Perturb = RepeatSentence(data, data_field='question') # initialise perturbation
    
- **data_field: *str*, *default='question'***: 
    
    Can take values `'question'` or `'context'` depending on which data instance the perturbation has to be applied to.
---

Example:

    output = Perturb.repeat_data_field() # execute perturbation on the desired data