# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class Trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen, attn=False):
        super(Trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen
        self.attn = attn

        # Definition des couches
        # Couches pour rnn
        self.decoder_embedding = nn.Embedding(self.dict_size, hidden_dim)
        self.encoder_layer = nn.RNN(457, hidden_dim, n_layers, batch_first=True)
        self.decoder_layer = nn.RNN(hidden_dim, hidden_dim, n_layers, batch_first=True)

        # Couches pour attention
        self.att_combine = nn.Linear(2*hidden_dim, hidden_dim)
        self.hidden2query = nn.Linear(hidden_dim, hidden_dim)
        self.soft_max = nn.Softmax()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # Couche dense pour la sortie
        self.fc = nn.Linear(hidden_dim, self.dict_size)
        self.to(device)

    def encoder(self, x):
        # Encodeur
        out, hidden = self.encoder_layer(x)
        return out, hidden

    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)
        # Attention
        simil = self.cos(values, query)
        attention_weights = self.soft_max(simil)
        attention_output = values * attention_weights[:, :, None]
        attention_output = torch.sum(attention_output, dim=1)
        return attention_output, attention_weights

    def decoder(self, encoder_outs, hidden):
        # Decodeur
        # Initialisation des variables
        max_len = self.maxlen['en']
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, self.dict_size, max_len)).to(self.device)  # Vecteur de sortie du décodage
        attn_w = None
        if self.attn:
            attn_w = torch.zeros((batch_size, self.max_len['fr'], self.max_len['en'])).to(self.device)  # Poids d'attention

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):

            vec_in = self.decoder_embedding(vec_in)
            out, hidden = self.decoder_layer(vec_in, hidden)

            if self.attn:
                attn_out, attn_w = self.attentionModule(out, encoder_outs)
                out = torch.cat((out[:, 0, :], attn_out), 1)
                out = self.att_combine(out)

            out = self.fc(out)
            vec_in = torch.argmax(out, dim=2)
            vec_out[:, :, i] = out[:, 0, :]

        return vec_out, hidden, attn_w

    def forward(self, x):
        # Passant avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out, h)
        return out, hidden, attn
    

