---
author:
- |
  Carl-André Gassette - gasc3203\
  Youcef Amine Aissaoui - aisy2303
date: 2022-03-21
title: RAPPORT D'APP3
---

::: titlepage
![image](logoUDES.png){width="60mm"}

UNIVERSITÉ DE SHERBROOKE\
Faculté de génie\
Département de génie électrique et génie informatique\

[]{.smallcaps}\

Réseaux de neurones récurrents\
GRO722\

Présenté à :\
Sean Wood\

Par :\
\
Sherbrooke -
:::

# Introduction

## Contexte de la problématique :

Dans le cadre du projet de l'APP3 relatif au module réseaux de neurones
nous allons nous intéresser aux réseaux de neurones récurrents. Nous
allons prendre le rôle d'un stagiaire dans une entreprise oeuvrant dans
le domaine de la réalité augmentée. Nous devrons utiliser des données
d'accélération préalablement traitées provenant d'une centrale
inertielle d'une montre intelligente et de les convertir en séquences de
lettres. le but étant de faire une preuve de concept afin de déterminer
si une technologie basée sur les réseaux de neurones récurrents est
réalisable dans un contexte de ressources limitées.

## Description de la tâche à accomplir (traduction):

### forme du Dataset :

Notre sous-ensemble d'entrainement est organisé sous forme d'un
dictionnaire.\
Chaque élément de ce dictionnaire est à son tour sous forme de liste
contenant:

-   Une séquence cible (séquence de lettre : str = 'MOT')

-   Une séquence d'entrée qui correspond aux coordonnées x et y de la
    centrale inertielle (un tableau de taille $\textsc{R}^{2*T}$).

Comme illustrer ci-dessous.
$$Dataset = \Bigg\{\bigg[ \textsc{'mot'}, array \Big[(x_0,...,x_T), (y_0,...,y_T) \Big]\bigg],...\Bigg\}$$
Lorsque l'on représente les données y en fonction de x nous avons alors
la figure ([1](#fig:Figure 1  ){reference-type="ref"
reference="fig:Figure 1  "}) ci-dessous :

![Représentation des données de l'accéléromètre pour le mot :
'ron'](sections/inputSequence.PNG){#fig:Figure 1   width="60mm"}

### Tâche de traduction :

Maintenant que nous connaissons la forme notre sous-ensemble nous devons
trouver une façon d'associer nos séquences d'entrées aux séquences de
sorties.\
A priori, nous ne cherchons pas à génerer des nouvelles séquences de
sorties à partir de notre séquence d'entrée (**Génération**), ni à
associer la séquence d'entrée à une seul valeur cible (**Prédiction**),
ni associer chaque élément de notre séquence d'entrée à une cible
(**Annotation**).\
En réalité, ce que nous cherchons à faire dans notre cas c'est
d'associer une séquence d'entrée à sa séquence de sortie qui est de
taille différente. Ceci correspond à la tâche de **Traduction**. Nous
devons donc traduire une séquence d'écriture cursive en texte.

# Architecture du réseau

## Pré-traitement des données:

Nous effectuons plusieurs pré-traitements des données de séquence et mot
cible.

### Vectorisation

Le premier pré-traitement consiste à ne garder que la dynamique de
mouvement plutôt que de coordonnées des séquences. C'est pourquoi nous
avons choisis une représentation vectorielle plutôt que sous forme de
coordonnées tel que :
$v(i)=\left\{\begin{array}{l}x(i+1)-x(i) \\ y(i+1)-y(i)\end{array}\right.$

### Remplissage

Afin de traiter les données en lots il est nécessaire qu'elles aient la
même taille. C'est pourquoi nous utilisons des techniques de
remplissages afin que les séquences d'entrées et sorties possèdent
toutes la taille maximale qui existe dans l'ensemble de donné. Le
remplissage de la séquence d'entrée se fait en remplissant de (0,0) les
vecteurs. Cela signifie qu'il n'y a plus de déplacement du stylo et
reviendrait à répéter plusieurs fois les coordonnées du dernier points.
Le pré-traitement des mots cibles se fait en ajoutant un symbole d'arrêt
ainsi qu'un symbole de remplissage jusqu'à obtenir un mot de 6 lettres(5
lettres + symbole d'arrêt), en sachant que nous avons au maximum des mot
de 5 lettres.

## Architecture choisie:

Pour répondre à cette problématique nous avons commencé par utiliser
l'architecture de base de la traduction vu en cours :

![Architecture de base de la
traduction](sections/images/architecture/traduction.png){#fig:Figure 2  
width="80mm"}

### unité d'Elman

Nous avons ensuite essayé les différentes unités de réseau de neurone
récurrent tel que Elman, GRU et LSTM. Dans un premier temps nous
remarquons un apprentissage difficile lors des premières itérations. Il
semble que nous arrivons à un minimum local, cela est visible par
l'apparition d'un plateau sur la courbe
([3](#fig:Figure 3  ){reference-type="ref" reference="fig:Figure 3  "})
de loss et de distance entre 25 et 50 itérations. Néanmoins,
l'apprentissage se poursuit correctement après la 75ème itérations.

![Évolution de la fonction de coût et de la distance au cours des
itérations pour
Elman](sections/images/architecture/RNN.png){#fig:Figure 3  
width="80mm"}

### unité LSTM

Nous avons pu observer de meilleures performances en utilisant GRU et
LSTM qui effectuaient un apprentissage plus rapide qu'en utilisant des
unités d'Elman. Le LSTM quant à lui semblait stagner et tomber plusieurs
fois dans des minimums locaux. (Figure
[4](#fig:Figure 4  ){reference-type="ref" reference="fig:Figure 4  "})

![Évolution de la fonction de coût et de la distance au cours des
itérations pour la
LSTM](sections/images/architecture/LSTM.png){#fig:Figure 4  
width="80mm"}

### Unité GRU

Enfin entre le GRU quant à lui nous donnait les meilleurs résultats pour
un faible nombre d'époque (100 époques). Cette différence peut être du
au fait que le GRU utilise moins de paramètres que le LSTM et donc
utilise moins de mémoire et s'exécute donc plus rapidement que le LSTM.
La mémoire à long terme du LSTM lui permettrait d'être plus performant
sur des séquences plus longues.

![Évolution de la fonction de coût et de la distance au cours des
itérations pour la
GRU](sections/images/architecture/GRU.png){#fig:Figure 5   width="80mm"}

### Module d'attention

Le module d'attention permet au réseau de se concentrer seulement sur
certaines parties de la séquence d'entrée lors de la prédiction d'une
lettre. Le but est d'utiliser de contexte a afin d'aider la
classification de la lettre.

![Module
d'attention](sections/images/architecture/attention_base.png){#fig:Figure 6  
width="80mm"}

Pour se faire nous comparons dans un premier temps la sortie de
l'encodeur à celle du décodeur en faisant passer préalablement la sortie
du décodeur dans une couche entièrement connectée de dimension H[^1].
Cela nous donne au vecteur q\[t\] de dimension(B[^2],1,H). La sortie du
décodeur V est directement utilisée pour le calcul de la similarité et
est de dimension (B,N[^3],H). La fonction de similarité choisie est
celle du cosinus, nous avons fait se choix afin de réduire le nombre de
paramètre à apprendre. Les valeurs en sortie du cosinus sont ensuite
envoyées dans une fonction softmax() afin de déterminer les poids
d'attentions w de la séquence tel que :

$$\begin{aligned}
            \tilde{w_{i}}=\mathrm{cos}\left(\mathbf{v_{i}},q[t]\right), i \in N \\
            w_{i}=\frac{\exp \left(\tilde{w_{i}}\right)}{\Sigma_{k=0}^{N-1} \exp \left(\tilde{w_{k}}\right)} \\
            dim(\tilde{w})=dim(w)=(B,N,1)
        \end{aligned}$$

La fonction weithing() permet de faire la somme pondérée des poids tel
que :

$$\begin{aligned}
            \tilde{a}=\mathbf{v}*w \\
            dim(\tilde{\tilde{a}})=(B,N,H) \\
            a=\Sigma_{k=0}^{N-1} \tilde{a_{k}}\\
            dim(a)=(B,1,H)
        \end{aligned}$$

La somme (6) est faite sur la dimension 1. Nous obtenons donc en sortie
un tenseur de dimension (B,1,H) (7) de même dimension que q\[t\]. Nous
faisons ensuite la concaténation de q et a et nous obtenons un nouveau
tenseur de dimension (B,2H) qui sera ensuite envoyé le FF (Feed forward
([7](#fig:Figure 7  ){reference-type="ref" reference="fig:Figure 7  "}))
composé de deux couches pleinement connectées. La première couche Att
est de dimension (2H,H) et la deuxième couche est de dimension (H,D[^4])

![Sortie du module
d'attention](sections/images/architecture/sortie_att.png){#fig:Figure 7  
width="80mm"}

### Réseaux récurrents bidirectionnels

Nous avons accès à la séquence d'entrée complète, il est donc possible
d'utiliser des unités bidirectionnelles. Son usage s'avère efficace car
elle nous permet d'accélérer grandement l'apprentissage et améliore nos
résultats jusqu'à avoir du sur-apprentissage. Nous utiliserons des
unités bidirectionnelles uniquement en entré du réseau car nous n'avons
pas accès au éléments futur de la séquence de sortie puisque nous la
générons un élément à la fois.

![réseau de neurone
bidirectionnel](sections/images/architecture/bidirection.png){#fig:Figure 8  
width="70mm"}

Lors de l'utilisation de la bidirectionnalité les dimensions du tenseur
de sortie du décodeur V change et deviennent (B,N,2H). Nous adaptons le
reste du réseau tel que :

-   La couche précédent q passe de dimension (H,H) à (H,2H) afin d'avoir
    q\[t\] de dimension (B,2H).

-   La couche Att passe de dimension (2H,H) à (3H,H).

-   Doubler le nombre de couche n du décodeur (2\*n couches).

## Schéma-bloc des couches du réseau :

Avec l'architecture suivante nous cumulons 32747 paramètres à entraîner.

![Schéma bloc final du
réseau](sections/images/architecture/schameBloc.png){#fig:Figure 9  
width="150mm"}

## Hyperparamètres :

-   La fonction de coût utilisée est l'entropie croisée (le symbole
    correspondant au remplissage est ignoré dans le calcul du coût)

-   Le taux d'apprentissage pendant tous les tests étaient fixé à 0.01

-   La taille des lots est de 100 données

-   L'entraînement/validation est séparé selon (70%/30%)

-   100 itérations

-   Optimisateur Adam

-   nombre de couche (verticale) : 3

-   taille des dimensions cachées : 19

# Interprétation 

## Évolution de la fonction coût et distance d'édition:

La figure suivante représente l'évolution de la fonction de coût et de
la distance de Levenstein de notre réseau au cours de l'entraînement de
la validation pour 100 époques.

![Évolution de la fonction de coût et de la
distance](sections/images/interpretation/bidirLoss.png){#fig:Figure 10  
width="120mm"}

### Interprétation de la fonction de coût

Au cours de premières itérations nous pouvons observer une décroissance
rapide de la fonction de coût pour la validation et l'entraînement. La
fonction de coût se stabilise entre 30 et 70 itérations avec
l'apparition de quelques sauts. Enfin, vers la 70ème itération nous
remarquons un sur-apprentissage du réseau car nous arrivons à annuler
complètement la fonction de coût pour l'entraînement alors qu'elle
commence à remonter pour la validation. Le réseau n'étant plus en mesure
de généraliser correctement nous gardons le modèle ayant la meilleure
distance en validation.

### Interprétation de la fonction de la distance de Levenstein

La distance moyenne de Levenstein suit la même évolution que la fonction
de coût et nous pouvons observer les mêmes phénomènes de
sur-apprentissages qu'avec la fonction de coût. Notre réseau arrive à
obtenir une distance moyenne nulle en entraînement et arrive à descendre
à environ 0.35 de distance en validation. Ces résultats sont corrects
car ils signifient qu'en moyenne notre réseau se trompe sur moins d'une
lettre.

## Matrice de confusion :

Une fois le réseau entraîné nous obtenons pour 100 mots pris au hasard
la matrice de confusion normalisée suivante :

![Matrice de confusion pour 100
mots](sections/images/interpretation/bidir matrix.png){#fig:Figure 11  
width="120mm"}

La matrice de confusion suivante possède en ordonné les indexes des
symboles cibles et en abscisse les indexes des symboles prédits. Pour
chaque mot nous cumulons ces valeurs dans la matrice puis nous
normalisons par rapport à la somme des valeurs cibles de chaque symbole
sur les lignes de la matrice afin d'avoir une représentation sous forme
de fréquence d'apparition (entre 0 et 1). De plus, nous ignorons
l'apparition du symbole de 'départ de séquence' car il n'apparaît pas,
ainsi que du 'remplissage' car il est traité en amont et ignoré pendant
l'apprentissage. Enfin nous pouvons commenter que les résultats
présentés par la matrice sont très bons car nous observons une haute
fréquence d'apparition des symboles cibles et prédits. C'est à dire par
la présence d'une diagonale avec des valeurs proches de 1.

## Représentation de l'attention :

Pour chaque lettre prédite nous pouvons récupérer les poids d'attentions
afin d'observer sur quelle partie de la séquence le réseau met
l'emphase. Sans bidirectionnalité nous obtenions les résultats suivants
:

![Affichage de l'attention sans
bidirectionnalité](sections/images/interpretation/lastGRUatt2.png){#fig:Figure 12  
width="120mm"}

Avec bidirectionnalité nous obtenons les résultats suivants :

![Affichage de l'attention avec
bidirectionnalité](sections/images/interpretation/bidirtest2.png){#fig:Figure 13  
width="120mm"}

Comme voulu, l'attention permet de porter l'emphase sur la partie de la
séquence à traduire. Néanmoins nous pouvons observer une différence avec
et sans bidirectionnalité. Lorsque cette dernière n'est pas présente le
réseau semble porter son attention sur la fin de la séquence
représentant la lettre. Par exemple sur la figure
([12](#fig:Figure 12  ){reference-type="ref"
reference="fig:Figure 12  "}) nous pouvons voir que nous nous
concentrons sur la fin de la lettre n sans bidirectionnalité alors que
nous prenons en un contexte content la lettre en entier lorsque la
bidirectionnalité est activée sur la figure
([13](#fig:Figure 13  ){reference-type="ref"
reference="fig:Figure 13  "}). Nous pouvons conclure que la
bidirectionnalité est donc bénéfique pour l'attention avec elle permet
d'avoir un contexte plus précis du mot en explorant en avant et en
arrière la séquence.

## Traduction avec le sous-ensemble test :

En testant plusieurs cas du sous-ensemble de test nous obtenions des
résultats corrects. Le réseau arrivait à traduire correctement la
séquence avec en moyenne moins d'une lettre d'erreur comme pour la
validation.

![Affichage de l'attention avec bidirectionnalité avec le sous ensemble
de
test](sections/images/interpretation/TEST_attention.png){#fig:Figure 14  
width="120mm"}

# Conclusion

Pour conclure, l'architecture qui offrait le meilleur compris en terme
de nombre de paramètres et vitesse d'apprentissage est l'utilisation de
GRU bidirectionnel avec attention. Nous arrivons au plus bas à une
distance moyenne de 0.35 sur 3500 mots. Ce résultat est correct car il
signifie que pour un mot de 5 lettres en moyenne nous nous trompons sur
moins d'une lettre.

[^1]: H : Taille de la couche cachée 23 []{#H label="H"}

[^2]: B : Taille du batch à 100 []{#B label="B"}

[^3]: N : Taille maximum de la séquence []{#N label="N"}

[^4]: D : Taille du dictionnaire de sortie (29 symboles) []{#D
    label="D"}
