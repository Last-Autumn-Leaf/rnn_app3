# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    batch_size=500
    train_val_split = .7
    hidden_dim=15
    n_layers=1
    lr=0.05
    with_attention=True
    bidir=False
    #'RNN', 'GRU' or 'LTSM'
    RNN_MODE='GRU'
    # À compléter
    n_epochs = 200
    TreatDataAsVectors=True

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset=HandwrittenWords('data_trainval.p',normalisation=True,asVector=TreatDataAsVectors)


    
    # Séparation de l'ensemble de données (entraînement et validation)
    n_train_samp = int(len(dataset) * train_val_split)
    n_val_samp = len(dataset) - n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    print('Number of epochs : ', n_epochs)
    print('Training data : ', len(dataset_train))
    print('Validation data : ', len(dataset_val))
    print('\n')

    # Instanciation du model
    #mode = 'RNN', 'GRU' or 'LTSM'
    model = Trajectory2seq(hidden_dim=hidden_dim, \
        n_layers=n_layers, device=device, symb2int=dataset.symb2int, \
        int2symb=dataset.int2symb, dict_size=dataset.dict_size,
        maxlen=dataset.max_len,mode=RNN_MODE,attn=with_attention,bidirectional=bidir)
    model = model.to(device)

    print("Number of parameters : ",sum(p.numel() for p in model.parameters()))


    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if trainning:

        # Initialisation affichage
        if learning_curves:
            train_dist = []  # Historique des distances
            train_loss = []  # Historique des coûts
            val_dist = []  # Historique des distances
            val_loss = []  # Historique des coûts
            fig, ax = plt.subplots(2)  # Initialisation figure

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2)  # ignorer les symboles <pad>
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            dist = 0
            for batch_idx, data in enumerate(dataload_train):
                word, seq = data
                seq = seq.to(device).float()
                word=torch.stack(word).T.long()

                optimizer.zero_grad()  # Mise a zero du gradient
                output, hidden, attn = model(seq)  # Passage avant
                # calcul de la distance d'édition
                output_list = torch.argmax(output, dim=1).detach().cpu().tolist()
                target_seq_list = word.cpu().tolist()
                M = len(output_list)
                for i in range(batch_size):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)  # longueur mot a (sans remplissage et eos)
                    Mb = b.index(1) if 1 in b else len(b)  # longueur mot b (sans remplissage et eos)
                    dist += edit_distance(a[:Ma], b[:Mb]) / batch_size

                loss = criterion(output, word)
                loss.backward()  # calcul du gradient
                optimizer.step()  # Mise a jour des poids
                running_loss_train += loss.item()

                    # Affichage pendant l'entraînement
                print(
                    'Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                        epoch, n_epochs, batch_idx * batch_size, len(dataload_train.dataset),
                                            100. * batch_idx * batch_size / len(dataload_train.dataset),
                                            running_loss_train / (batch_idx + 1),
                                            dist / ((batch_idx+1)* len(dataload_train))), end='\r')
                print(
                    'Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                        epoch, n_epochs, (batch_idx + 1) * batch_size, len(dataload_train.dataset),
                                         100. * (batch_idx + 1) * batch_size / len(dataload_train.dataset),
                                         running_loss_train / (batch_idx + 1),
                                         dist / len(dataload_train)), end='\r')
                print('\n')


                # Terminer l'affichage d'entraînement

            # Ajouter les loss aux listes
            train_loss.append(running_loss_train / len(dataload_train))
            train_dist.append(dist / len(dataload_train))

            # Validation
            running_loss_val = 0
            dist = 0
            for batch_idx, data in enumerate(dataload_val):

                word, seq = data
                seq = seq.to(device).float()
                word = torch.stack(word).T
                output, hidden, attn = model(seq)  # Passage avant

                M = len(output_list)
                for i in range(batch_size):
                    a = target_seq_list[i]
                    b = output_list[i]
                    M = a.index(1)
                    dist += edit_distance(a[:M], b[:M]) / batch_size

                loss =  criterion(output, word)
                running_loss_val += loss.item()
                # calcul de la distance d'édition
                output_list = torch.argmax(output, dim=1).detach().cpu().tolist()
                target_seq_list = word.cpu().tolist()

            # Ajouter les loss aux listes
            val_loss.append(running_loss_val / len(dataload_val))
            val_dist.append(dist / len(dataload_val))

            # Enregistrer les poids
            torch.save(model, 'model.pt')


            # Affichage
            if learning_curves:
                ax[0].cla()
                ax[1].cla()
                ax[0].plot(train_loss, label='training loss')
                ax[0].plot(val_loss, label='validation loss')
                ax[1].plot(train_dist, label='training distance')
                ax[1].plot(val_dist, label='validation distance')
                ax[0].legend()
                ax[1].legend()
                plt.draw()
                plt.pause(0.01)

        if learning_curves:
                plt.show()
                plt.close('all')

    if test:
        # Évaluation
        # À compléter

        # Charger les données de tests
        model = torch.load('model.pt')
        dataset.symb2int = model.symb2int
        dataset.int2symb = model.int2symb

        # Affichage de l'attention
        # À compléter (si nécessaire)

        D = np.zeros((29, 29))
        randintList=[]
        numberOfTest=10
        # Affichage des résultats de test
        for i in range(numberOfTest):
            # Extraction d'une séquence du dataset de validation
            randintList.append(np.random.randint(0, len(dataset)))
            word, seq = dataset[randintList[i]]


            seq=torch.from_numpy(seq).float().to(device)[None,:,:]
            word = torch.IntTensor(word)
            output, hidden, attn = model(seq)  # Passage avant
            output = torch.argmax(output, dim=1).detach().cpu()[0,:].tolist()

            D += confusion_matrix(word.detach().cpu().tolist(), output)

            #affichage
            out_seq = [model.int2symb[i] for i in output]
            target=[model.int2symb[i] for i in word.detach().cpu().tolist()]
            print('target : ',''.join(target))
            print('guessed : ',''.join(out_seq))
            print("  --------  ")

        # Affichage de la matrice de confusion
        norm =np.sum(D,axis=1)
        norm[norm ==0] = 1
        D=D/norm[None,:]
        plot_confusion_matrix(D)
        plt.figure()
        for i in range(len(randintList)):
            dataset.visualisation(randintList[i])
        plt.show()


