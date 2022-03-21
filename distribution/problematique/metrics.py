# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import numpy as np
import matplotlib.pyplot as plt


def edit_distance(a, b,ignoredSymbole=None):
    # Calcul de la distance d'édition
    if ignoredSymbole is not None :
        if type(a)==str and type(b)==str :
            a=a.replace(ignoredSymbole,'')
            b=b.replace(ignoredSymbole,'')
        elif type(a)==list and type(b)==list :
            while (ignoredSymbole in a ):
                a.remove(ignoredSymbole)
            while (ignoredSymbole in b):
                b.remove(ignoredSymbole)
    # ---------------------- Laboratoire 2 - Question 1 - Début de la section à compléter ------------------
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if a[0] == b[0]:
        return edit_distance(a[1:], b[1:])

    else:
        return 1 + min(edit_distance(a[1:], b), edit_distance(a, b[1:]), edit_distance(a[1:], b[1:]))


def confusion_matrix(true, pred, ignore=[0, 2]):
    # Calcul de la matrice de confusion
    confus_mat = np.zeros((29, 29))       # ATTENTION : AUGMENTER LA TAILLE DE LA MATRICE A 29 * 29

    for i in range(len(true)):
        if true[i] not in ignore and pred[i] not in ignore:    # ATTENTION : A GENERALISER AU BATCH
            confus_mat[true[i], pred[i]] += 1

    return confus_mat

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):

    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title)
    plt.colorbar()

def Attention_weight():
    ...


if __name__ == "__main__":
    true = 'aacd'
    pred = 'abcd'

    conf = edit_distance(true, pred,'a')

    print(conf)
