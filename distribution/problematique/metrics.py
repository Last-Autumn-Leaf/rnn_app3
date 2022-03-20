# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import numpy as np
import matplotlib.pyplot as plt
def edit_distance(token1,token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1 - 1] == token2[t2 - 1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


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

    plt.show()

if __name__ == "__main__":
    true = [1, 3, 6, 1, 4, 2, 0, 2]
    pred = [1, 4, 5, 1, 4, 3, 2, 2]

    conf = confusion_matrix(true, pred)

    print(conf)
