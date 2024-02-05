# Usage

### 1. `DeleteWord` / `RepeatWord` / `SplitWord`

First initialise the perturbation by choosing one of the packages. Example:

    from qa_perturb.word_perturb import DeleteWord
    Perturbation = DeleteWord(data, data_field='question', max_words=1, \
                                   length_of_word_to_perturb=2, pos_tag=None) # initialise perturbation
    
- **data_field: *str*, *default='question'***: 
    
    Can take values `'question'` or `'context'` depending on which data instance the perturbation has to be applied to.

- **max_words: *int, default=1***: 

    Maximum number of words to be perturbed.

- **length_of_word_to_perturb: *int, default=2***

    Minimum length required by the word to be perturbed

- **pos_tag: *str, default=None***, 

    Unless specified, random words are chosen for perturbation. If specifed, with `pos_tag='verb'` then words with either `AUX` or `VERB` tags are perturbed.
---
Now execute pertubation by calling one of the following functions:

- `delete_word()`: For deletion perturbation
- `repeat_word()`: For repetition perturbation
- `split_word()`: For inserting a space in random words

Example:

    output = Perturbation.delete_word() # execute perturbation on the desired data

### 2. `Synonym`

First initialise the perturbation:

    from qa_perturb.word_perturb import Synonym
    Perturbation = Synonym(data, data_field='question', model='xlm-roberta-base') # initialise perturbation
    
- **data_field: *str*, *default='question'***: 
    
    Can take value `'question'` or `'context'` depending on which data instance the perturbation has to be applied to.

- **model: *str, default='xlm-roberta-base'***: 

    Model to be used for fill-mask task. Currently accepts only models available from HuggingFace hub.

---
Now execute pertubation:

    output = Perturbation.replace_with_contextual_synonym() # execute perturbation on the desired data

### 3. `SwapWords`

First initialise the perturbation:

    from qa_perturb.word_perturb import SwapWords
    Perturbation = SwapWords(data, data_field='question') # initialise perturbation
    
- **data_field: *str*, *default='question'***: 
    
    Can take values `'question'` or `'context'` depending on which data instance the perturbation has to be applied to.

---
Now execute pertubation:

    output = Perturbation.swap_words() # execute perturbation on the desired data