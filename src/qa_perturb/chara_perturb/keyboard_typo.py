from datasets import Dataset
import random
from tqdm import tqdm
from typing import List

'''
References: 
1. Code taken and adapted from: https://github.com/alexyorke/butter-fingers/blob/master/butterfingers/butterfingers.py
'''

class KeyboardTypo():
    def __init__(self, data: Dataset, data_field: str='question', probability_of_typo=0.1, keyboard:str='qwertz'):
        self.probability_of_typo: float = probability_of_typo
        self.data: Dataset = data
        self.data_field = data_field.lower()
        self.name: str = None # be as descriptive as possible
        self.keyboard = keyboard
        self.total_perturbations = 0
        self.perturbed_data = []

    def get_keyboard(self):
        keyApprox = {}
        if self.keyboard == "qwerty":
            keyApprox['q'] = "qwasedzx"
            keyApprox['w'] = "wqesadrfcx"
            keyApprox['e'] = "ewrsfdqazxcvgt"
            keyApprox['r'] = "retdgfwsxcvgt"
            keyApprox['t'] = "tryfhgedcvbnju"
            keyApprox['y'] = "ytugjhrfvbnji"
            keyApprox['u'] = "uyihkjtgbnmlo"
            keyApprox['i'] = "iuojlkyhnmlp"
            keyApprox['o'] = "oipklujm"
            keyApprox['p'] = "plo['ik"

            keyApprox['a'] = "aqszwxwdce"
            keyApprox['s'] = "swxadrfv"
            keyApprox['d'] = "decsfaqgbv"
            keyApprox['f'] = "fdgrvwsxyhn"
            keyApprox['g'] = "gtbfhedcyjn"
            keyApprox['h'] = "hyngjfrvkim"
            keyApprox['j'] = "jhknugtblom"
            keyApprox['k'] = "kjlinyhn"
            keyApprox['l'] = "lokmpujn"

            keyApprox['z'] = "zaxsvde"
            keyApprox['x'] = "xzcsdbvfrewq"
            keyApprox['c'] = "cxvdfzswergb"
            keyApprox['v'] = "vcfbgxdertyn"
            keyApprox['b'] = "bvnghcftyun"
            keyApprox['n'] = "nbmhjvgtuik"
            keyApprox['m'] = "mnkjloik"
            keyApprox[' '] = " "

        elif self.keyboard == "qwertz":
            keyApprox['q'] = "qasew321^"
            keyApprox['w'] = "qwerdsa^1234"
            keyApprox['e'] = "qwertfds2345"
            keyApprox['r'] = "wertzgfd3456"
            keyApprox['t'] = "ertzuhgfd4567"
            keyApprox['z'] = "rtzuijhgf5678"
            keyApprox['u'] = "tzuiokjhg6789"
            keyApprox['i'] = "zuioplkj7890"
            keyApprox['o'] = "uiopüölk890ß"
            keyApprox['p'] = "iopü+äöl90ß´"
            keyApprox['ü'] = "opü+´ß0löä#"

            keyApprox['a'] = "qwesdxy<"
            keyApprox['s'] = "asdfcxyqwer<"
            keyApprox['d'] = "asdfgwertyxcv"
            keyApprox['f'] = "sdfghbvcxertz"
            keyApprox['g'] = "dfghjcvbnrtzu"
            keyApprox['h'] = "fghjktzuivbnm"
            keyApprox['j'] = "ghjklö,mnbzuio"
            keyApprox['k'] = "hjklö.,mnbuiop"
            keyApprox['l'] = "jklöä-.,miuiopü"
            keyApprox['ö'] = "klöä#,.-opü+"
            keyApprox['ä'] = "llöä#+üpo.-"

            keyApprox['y'] = "yxcdsa<"
            keyApprox['x'] = "<yxcvdsa"
            keyApprox['c'] = "yxcvbgfds"
            keyApprox['v'] = "xcvbnhgfd"
            keyApprox['b'] = "cvbnmjhgf"
            keyApprox['n'] = "vbnm,kjhg"
            keyApprox['m'] = "bnm,.lkjh"
            keyApprox[' '] = " "
            
        else:
            print("Keyboard not supported.")
        return keyApprox


    def butterfinger(self, text):
        self.total_perturbations = 0
        keyApprox = self.get_keyboard()
        self.name = 'make_typo_'+self.keyboard
        probOfTypo = int(self.probability_of_typo * 100)
        buttertext = ""
        for letter in text:
            lcletter = letter.lower()
            if not lcletter in keyApprox.keys():
                newletter = lcletter
            else:
                if random.choice(range(0, 100)) <= probOfTypo:
                    newletter = random.choice(keyApprox[lcletter])
                    self.total_perturbations += 1
                else:
                    newletter = lcletter
            # go back to original case
            if not lcletter == letter:
                newletter = newletter.upper()
            buttertext += newletter
        return buttertext
    
    def make_typo(self):
        '''
        Function that perturbs the given dataset
        '''
        unaltered_data: List = []
        
        # enumerate through the dataset
        for index, sample in enumerate(tqdm(self.data)):
            self.name = 'keyboardtypo'
            self.perturbed_data.append({'info': self.name, 
                                        'Input': {'id': sample['id'], 'context': sample['context'], 
                                        'question': sample['question'], 'answers': sample['answers']},
                                        'Output': []})

            text = sample[self.data_field]
            new_text = self.butterfinger(sample[self.data_field])
            self.perturbed_data[index]['Output'].append({'data_field': self.data_field,
                                                         'total_per': self.total_perturbations,
                                                         'original_'+self.data_field: text,
                                                         'perturbed_'+self.data_field: new_text})
        
        return self.perturbed_data, unaltered_data 