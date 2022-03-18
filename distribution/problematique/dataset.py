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

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)


        # Extraction des symboles
        self.symb2int={start_symbol:0, stop_symbol:1, pad_symbol:2}
        alphabet="abcdefghijklmnopqrstuvwxyz"

        for letter in alphabet :
            self.symb2int[letter] = len(self.symb2int)

        """cpt=0
        for symb in range (len(self.data)):

            if symb not in self.symb2int :
                self.symb2int[symb]=cpt
                cpt+=1"""

        self.int2symb = dict()
        self.int2symb = {v: k for k, v in self.symb2int.items()}

        
        # Ajout du padding aux séquences
        max_length = 5
        for words in self.data:
            words[0] += self.stop_symbol
            while (len(words[0]) < max_length + 1):
                words[0] += self.pad_symbol
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word,seq=self.data[idx]
        word=[self.symb2int[letter] for letter in word]
        return word, seq

    def visualisation(self, idx):
        # Visualisation des échantillons
        # À compléter (optionel)
        pass
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))
