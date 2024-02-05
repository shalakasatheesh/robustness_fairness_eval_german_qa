# Usage

### 1. `DeleteChara` / `InsertChara` / `RepeatChara` / `ReplaceChara` / `SwapChara`

First initialise the perturbation by choosing one of the packages. Example:

    from qa_perturb.chara_perturb import InsertChara
    Perturbation = InsertChara(data, data_field='question', \
                               max_perturbs=1, max_words=1, \
                               length_of_word_to_perturb=2, \ 
                               pos_tag=None) # initialise perturbation
    
- **data_field: *str*, *default='question'***: 
    
    Can take values `'question'` or `'context'` depending on which data instance the perturbation has to be applied to.

- **max_perturbs: *int, default=1***: 

    Maximum number of characters to be perturbed per word.

- **max_words: *int, default=1***: 

    Maximum number of words to be perturbed.

- **length_of_word_to_perturb: *int, default=2***:

    Minimum length required by the word to be perturbed

- **pos_tag: *str, default=None***, 

    Unless specified, random words are chosen for perturbation. If specifed, with `pos_tag='verb'` then words with either `AUX` or `VERB` tags are perturbed.
---
Now execute pertubation by calling one of the following functions:

- `insert_chara()`: For insertion perturbation
- `delete_chara()`: For deletion perturbation
- `repeat_chara()`: For repetition perturbation
- `replace_chara()`: For repetition perturbation
- `swap_chara()`: For repetition perturbation

Example:

    output = Perturbation.insert_chara() # execute perturbation on the desired data

### 2. `DeletePunctuation`

First initialise the perturbation:

    from qa_perturb.chara_perturb import DeletePunctuation
    Perturbation = DeletePunctuation(data, data_field='question') # initialise perturbation
    
- **data_field: *str*, *default='question'***: 
    
    Can take values `'question'` or `'context'` depending on which data instance the perturbation has to be applied to.

Now execute pertubation by calling the following function:

    output = Perturbation.delete_punct() # execute perturbation on the desired data

### 3. `ChangeCase`

First initialise the perturbation:

    from qa_perturb.chara_perturb import ChangeCase
    Perturbation = ChangeCase(data, data_field='question', case='lower') # initialise perturbation
    
- **data_field: *str*, *default='question'***: 
    
    Can take values `'question'` or `'context'` depending on which data instance the perturbation has to be applied to.

- **case: *str*, *default='lower'***: 
    
    Can take values from `['lower', 'upper', 'title', 'invert']`

Now execute pertubation by calling the following function:

    output = Perturbation.change_case() # execute perturbation on the desired data

### 4. `ReplaceUmlaute`

First initialise the perturbation:

    from qa_perturb.chara_perturb import ReplaceUmlaute
    Perturbation = ReplaceUmlaute(data, data_field='question') # initialise perturbation
    
- **data_field: *str*, *default='question'***: 
    
    Can take values `'question'` or `'context'` depending on which data instance the perturbation has to be applied to.

Now execute pertubation by calling the following function:

    output = Perturbation.replace_umlaute() # execute perturbation on the desired data

### 5. `KeyboardTypo`

First initialise the perturbation:

    from qa_perturb.chara_perturb import KeyboardTypo
    Perturbation = KeyboardTypo(data, data_field='question', probability_of_typo=0.1, keyboard='qwertz') # initialise perturbation
    
- **data_field: *str*, *default='question'***: 
    
    Can take values `'question'` or `'context'` depending on which data instance the perturbation has to be applied to.

- **probability_of_typo: *str*, *float=0.1***: 
    
    Specify the probability of producing a typo, between 0 and 1.

- **keyboard: *str*, *default='qwertz'***: 
    
    Can take values `'qwertz'` or `'qwerty'` depending on desired keyboard layout.

Now execute pertubation by calling the following function:

    output = Perturbation.make_typo() # execute perturbation on the desired data

##### - Reference: https://github.com/alexyorke/butter-fingers/blob/master/butterfingers/butterfingers.py

### 6. `InsertPunctuation`

First initialise the perturbation:

    from qa_perturb.chara_perturb import InsertPunctuation
    Perturbation = InsertPunctuation(data, data_field='question', \
                                    max_perturbs=1, length_of_word_to_perturb=2) # initialise perturbation
    
- **data_field: *str*, *default='question'***: 
    
    Can take values `'question'` or `'context'` depending on which data instance the perturbation has to be applied to.

- **max_perturbs: *int, default=1***: 

    Maximum number of characters to be perturbed per word.

- **length_of_word_to_perturb: *int, default=2***:

    Minimum length required by the word to be perturbed

---
Now execute pertubation:

    output = Perturbation.insert_punct() # execute perturbation on the desired data

