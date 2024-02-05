from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer, BertForQuestionAnswering
import torch
import numpy as np
import pandas  as pd
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from typing import List, Dict, DefaultDict, Tuple
import argparse
import tqdm

'''
Code is heavily referenced from: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
'''

class GetEmbeddings():
    def __init__(self, text: str, model_checkpoint: str, num_segments: int=1):
        self.text = text
        self.model_checkpoint = model_checkpoint
        self.num_segments = num_segments
        self.tokens = self.tokenize()
        self.segments_ids, self.indexed_tokens = None, None
        self.tokens_with_id = {} 
        self.hidden_states: Tuple = None
        self.token_embeddings: torch.Tensor = None
        self.token_vecs_concat: List[torch.Tensor] = []
        self.token_vecs_sum: List[torch.Tensor]  = []
        self.sentence_embedding: torch.Tensor = None
        
    def tokenize(self):
        '''
        Tokenize input text with model_checkpoint.

        Returns:
        --------
        tokens: List[str] - tokens after tokenization
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        marked_text = "[CLS] " + self.text + " [SEP]"
        tokens = tokenizer.tokenize(marked_text)
        
        self.tokens = tokens
        return self.tokens

    def just_tokenize(self):
        '''
        Tokenize input text with model_checkpoint.

        Returns:
        --------
        tokens: List[str] - tokens after tokenization
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        tokens = tokenizer.tokenize(self.text)
        self.tokens = tokens
        return self.tokens

    def convert_tokens_to_ids(self):
        '''
        Get the vocab IDs of tokens from the input text.

        Returns:
        --------
        indexed_tokens: List[int] - The IDs of tokens
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        indexed_tokens = tokenizer.convert_tokens_to_ids(self.tokens)
        return indexed_tokens
    
    def print_tokens_with_ids(self):
        ''' 
        Print tokens in the input sequence with their respective vocab ID's.
        '''
        
        self.indexed_tokens = self.convert_tokens_to_ids()
        for text, id in zip(self.tokens, self.indexed_tokens):
            self.tokens_with_id[text] = id
        print(self.tokens_with_id)
    
    def print_hidden_states_details(self, layer_i=0, batch_i=0, token_i=0):
        '''
        Print details about layers, batches, tokens and hidden_units. 
        Default value for layer, batch and token is 0.
        '''

        self.get_hidden_states()

        print ("No. of layers:", len(self.hidden_states), )
        print ("No. of batches:", len(self.hidden_states[layer_i]))
        print ("No. of tokens:", len(self.hidden_states[layer_i][batch_i]))
        print ("No. of hidden units:", len(self.hidden_states[layer_i][batch_i][token_i]))

    def plot_embedding_vector(self, layer_i=0, batch_i=0, token_i=0):
        ''' 
        Plot the values of the selected embedding vector as a histogram
        '''
        vec = self.hidden_states[layer_i][batch_i][token_i]

        # Plot the values as a histogram to show their distribution.
        plt.figure(figsize=(10,5))
        plt.hist(vec, bins=200, alpha=0.7)
        plt.title("Values of the embedding of the token "+str(token_i)+" from layer "+str(layer_i)+" and batch "+str(batch_i))
        plt.grid()
        plt.show()

    def get_tokens_with_segment_ids(self):
        ''' 
        Returns tokens with their segment ID's
        '''
        self.indexed_tokens = self.convert_tokens_to_ids()
        segments_ids = list(range(1, self.num_segments+1)) * len(self.tokens) # TO DO :  modify to take as input the number of segments!!
        self.segments_ids = segments_ids

    def get_hidden_states(self):
        ''' 
        Gets the hidden_states from the output of the model. 
        Then converts the hidden_states to token_embeddings by getting 
        rid of the batch_size dimension and then permuting it to
        the shape [#tokens, #layers, #hidden_units]
        '''
        # convert tokens and segment IDs to tensors
        self.get_tokens_with_segment_ids()
        tokens_tensor = torch.tensor([self.indexed_tokens])
        segments_tensors = torch.tensor([self.segments_ids])
        
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_checkpoint, 
                                                              output_hidden_states=True)
        model.eval()
        with torch.no_grad(): # TO DO: for inference pipeline also!! 
            outputs = model(tokens_tensor, segments_tensors) 
            self.hidden_states = outputs[2]
        
        # print(self.hidden_states[0].size()) # print shape of embedding in one layer

        # Stack tensors for each layer by using `torch.stack`. Now `hidden_states` is no longer a tuple
        token_embeddings = torch.stack(self.hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        self.token_embeddings = token_embeddings.permute(1,0,2)
        # print(self.token_embeddings.size()) # print re-shaped shape of embedding in one layer [#tokens, #layers, #hidden_units]

    def get_token_vec_concat(self):
        ''' 
        This is a word embedding. Concatenates the vectors from 
        last n layers for each token in the sentence.
        Output shape: [ # tokens, 4 * # hidden_units ]
        '''
        self.get_hidden_states()
        
        # For each token in the sequence
        # concatenate the vectors from last 'n' layers??? TO DO
        for token in self.token_embeddings:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            self.token_vecs_concat.append(cat_vec)

        # print("shape of token embedding (concat): ", (len(self.token_vecs_concat), len(self.token_vecs_concat[0])))
        return self.token_vecs_concat
    
    def get_token_vec_sum(self, n: int=4):
        '''
        This is a word embedding. Takes the sum of the 
        vectors from last n layers for each token in the sentence.
        Output shape: [ # tokens, # hidden_units ]
        '''
        self.get_hidden_states()
        
        # For each token in the sequence
        # Sum the vectors from the last 'n' layers. TO DO
        for token in self.token_embeddings:
            sum_vec = torch.sum(token[-n:], dim=0)
            self.token_vecs_sum.append(sum_vec)

        # print("shape of token embedding (sum): ", (len(self.token_vecs_sum), len(self.token_vecs_sum[0])))
        return self.token_vecs_sum
    
    def get_token_vec_mean(self, n: int=2):
        '''
        This is a sentence embedding. Mean of all the tokens 
        from a given sequence from the second last layer.
        Output shape:  [ # hidden_units ]
        '''
        self.get_hidden_states()

        token_vecs = self.hidden_states[-n][0]

        # Calculate the average of all tokens from the given sequence
        self.sentence_embedding = torch.mean(token_vecs, dim=0)

        # print("size of sentence embedding: ", self.sentence_embedding.size())
        return self.sentence_embedding