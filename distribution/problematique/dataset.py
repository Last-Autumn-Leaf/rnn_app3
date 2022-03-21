# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename,normalisation=False,asVector=False):
        # Lecture du text
        self.pad_symbol     = pad_symbol ='@' #'<pad>'
        self.start_symbol   = start_symbol ='#' #'<sos>'
        self.stop_symbol    = stop_symbol ='$'  #'<eos>'

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)


        # Extraction des symboles
        self.symb2int={start_symbol:0, stop_symbol:1, pad_symbol:2}
        alphabet="abcdefghijklmnopqrstuvwxyz"

        for letter in alphabet :
            self.symb2int[letter] = len(self.symb2int)
        self.dict_size=len(self.symb2int)
        self.int2symb = dict()
        self.int2symb = {v: k for k, v in self.symb2int.items()}
        self.asVector=asVector

        
        # Ajout du padding aux séquences
        max_length = 5
        max_size_coords=0
        for words in self.data:
            words[0] += self.stop_symbol
            max_size_coords=len(words[1][0]) if max_size_coords<len(words[1][0]) else max_size_coords
            while (len(words[0]) < max_length + 1):
                words[0] += self.pad_symbol

        for words in self.data:
            array=words[1]
            if normalisation :
                #Translation
                min_xy=np.array([[np.min(array[0])],[np.min(array[1])]])
                array=array-min_xy
                #Scaling
                max_xy = np.array([[np.max(array[0])], [np.max(array[1])]])
                array/=max_xy

            if len(array[0]) != max_size_coords:
                n=max_size_coords-len(array[0])
                pad_points=np.ones((2,n))
                xy=array[:,-1]
                pad_points*=xy[:,None]
                #words[1]=np.concatenate((array,pad_points), axis=1)
                array=np.concatenate((array,pad_points), axis=1)

            if asVector:
                array = array[:, 1:] - array[:, 0:-1]
            words[1] =array


        #vérifcation padding
        if asVector:
            max_size_coords-=1
        for words in self.data :
            if len(words[0]) !=max_length+1:
                print("Error padding word len=",len(words[0]))
            if len(words[1][0]) !=max_size_coords:
                print("error pad seq, length=",len(words[1][0]))

        self.max_len={
            'points':max_size_coords,
            'en':max_length+1
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word,seq=self.data[idx]
        word=[self.symb2int[letter] for letter in word]
        seq=seq.T
        #word=torch.stack(word)
        return word, seq

    def visualisation(self, idx):
        word, seq = self[idx]
        word=''.join([self.int2symb[entier] for entier in word])
        if self.asVector:
            for i in range(len(seq)):
                seq[i]+= seq[i-1] if i!=0 else np.zeros((2))
        plt.plot(seq[:,0],seq[:,1],'-o',label=str(idx))
        plt.title(word)
        plt.show()

        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p',normalisation=True,asVector=True)
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))
