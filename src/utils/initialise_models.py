import spacy
import stanza

def initialise_spacy():
    return NotImplementedError

def initialise_stanza(language='de'):
    stanza.download(language)
    nlp_stanza = stanza.Pipeline(lang=language, processors='tokenize,mwt,pos')
    return nlp_stanza