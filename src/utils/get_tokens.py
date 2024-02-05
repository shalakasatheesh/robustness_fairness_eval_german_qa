import random
import stanza
from typing import List, Tuple, Dict, Union
    
def get_word(words: List[stanza.models.common.doc.Word], max_words: int, length_of_word_to_perturb: int):
    '''
    Function to get random words to perturb from the given data sample.

    Inputs:
    -------
    words: A list of tokenized words from the data samples.

    Returns:
    --------
    words_to_perturb: A list of randomly chosen words for perturbation
    '''
    random_words: List = []
    words_to_perturb: List = []
    total_words: int = 0

    for word in words:
        if word.upos != 'PUNT' and len(word.text) > length_of_word_to_perturb:
            random_words.append(word)

    while True:
        try:
            word_chosen = random.choice(random_words)
        except:
            break
        random_words.remove(word_chosen)
        words_to_perturb.append(word_chosen)
        total_words += 1
        if total_words == max_words:
            break

    return words_to_perturb

def get_verb(words: List[stanza.models.common.doc.Word], max_words: int, length_of_word_to_perturb: int):
    '''
    Function to get random verbs to perturb from the given data sample.

    Inputs:
    -------
    words: List: A list of tokenized words from the data samples

    Returns:
    --------
    words_to_perturb: List: A list of words with a POS tag of either 'VERB' or 'AUX'
    '''
    verbs: List = []
    words_to_perturb: List = []
    total_words: int = 0

    for word in words:
        if word.upos == 'VERB' or word.upos == 'AUX' and len(word.text) > length_of_word_to_perturb:
            verbs.append(word)

    while True:
        try:
            word_chosen = random.choice(verbs)
        except:
            break
        verbs.remove(word_chosen)
        words_to_perturb.append(word_chosen)
        total_words += 1
        if total_words == max_words:
            break

    return words_to_perturb