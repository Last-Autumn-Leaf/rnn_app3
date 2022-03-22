# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

mode_dict={
    'RNN':nn.RNN,
    'GRU':nn.GRU,
    'LSTM':nn.LSTM
}

class Trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen,
            attn=False,mode='RNN',bidirectional=False,batchNorm=False):
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
        self.mode=mode
        self.batchNorm=batchNorm
        # Definition des couches
        # Couches pour rnn
        self.decoder_embedding = nn.Embedding(self.dict_size, hidden_dim)

        self.encoder_layer =mode_dict[mode](input_size=2,hidden_size= hidden_dim,
                            num_layers=n_layers, batch_first=True,bidirectional=bidirectional)
        self.decoder_layer = mode_dict[mode](input_size=hidden_dim,hidden_size= hidden_dim,
                            num_layers=2*n_layers if bidirectional else n_layers, batch_first=True)

        # Couches pour attention
        self.att_combine = nn.Linear(3*hidden_dim if bidirectional else  2*hidden_dim, hidden_dim)
        self.hidden2query = nn.Linear(hidden_dim, 2*hidden_dim if bidirectional else hidden_dim)
        self.soft_max = nn.Softmax(dim=1)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)


        # BN
        if self.batchNorm :
            self.BN=nn.BatchNorm1d(1)
        # Couche dense pour la sortie
        self.fc = nn.Linear(hidden_dim , self.dict_size)
        self.to(device)

    def encoder(self, x):
        # Encodeur
        if self.mode=='LSTM':
            out, (hidden,c) = self.encoder_layer(x)
            return out, hidden,c
        else :
            out, hidden = self.encoder_layer(x)
            return out, hidden

    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)
        # Attention
        simil = self.cos(values, query)
        attention_weights = self.soft_max(simil)
        attention_output = values * attention_weights[:, :,None]
        attention_output = torch.sum(attention_output, dim=1)
        return attention_output, attention_weights

    def decoder(self, encoder_outs, hidden,c=None):
        # Decodeur
        # Initialisation des variables
        max_len = self.maxlen['en']
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, self.dict_size, max_len)).to(self.device)  # Vecteur de sortie du décodage
        attn_w = None
        if self.attn:
            attn_w = torch.zeros((batch_size,  self.maxlen['points'])).to(self.device)  # Poids d'attention
            attn_w_out= torch.zeros((batch_size,self.maxlen['en'],self.maxlen['points']))

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):

            vec_in = self.decoder_embedding(vec_in)

            if self.mode == 'LSTM':
                out, (hidden ,c)= self.decoder_layer(vec_in, (hidden,c))

            else:
                out, hidden = self.decoder_layer(vec_in, hidden)

            if self.attn:
                attn_out, attn_w = self.attentionModule(out, encoder_outs)
                out = torch.cat((out[:, 0, :], attn_out), 1)
                out = self.att_combine(out)[:,None,:]
                attn_w_out[:,i,:]=attn_w

            # BN
            if self.batchNorm:
                out=self.BN(out)
            out = self.fc(out)
            vec_in = torch.argmax(out, dim=2)
            vec_out[:, :, i] = out[:, 0, :]

        return vec_out, hidden, attn_w_out

    def forward(self, x):
        # Passant avant
        if self.mode=='LSTM':
            out, h,c = self.encoder(x)
            out, hidden, attn = self.decoder(out, h,c)
        else :
            out, h = self.encoder(x)
            out, hidden, attn = self.decoder(out, h)


        return out, hidden, attn
    

