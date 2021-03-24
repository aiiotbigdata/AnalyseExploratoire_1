# On commence par installer les librairies, dans le terminal pycharm : pip install pandas, pip install seaborn

# Identifier le répertoire courant du projet
import os
print(os.getcwd())

# Dans un nouveau projet Pycharm et un fichier Python, on commence par installer les librairies
# Importer les librairies

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Vérification des données : Avant d’aller plus loin, il est toujours bon de consulter les différentes colonnes et de
# déterminer le type d’attributs présents dans les données afin d’avoir une idée approximative de la façon de commencer
# notre AED.
# Entetes du fichier adultoldv3title :
# age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,
# hours-per-week,native-country,income

path = "test/adultoldv3.csv"
columns = ["age", "work-class", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship","race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
data = pd.read_csv(path, names=columns, sep=',', na_values='?', skipinitialspace=True)
print("\n=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=AFFICHAGE DES ENTËTES*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*")
print(data.head())

# Affiche les dix premiers
print("\n=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=AFFICHAGE DES DIX PREMIERS ÉLÉMENTS=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=")
print(data.head(10))
print("\n---------------------------------------------------------------------------------------------------------------")

# Maintenant, sur la base du résultat ci-dessus, nous pouvons classer les différentes colonnes de données en attributs
# numériques / catégoriques :
# Attributs numériques : Age, poids final, nombre d’année d’éducation, Gain en capital, Perte en capital, Heures par semaine.
# Attributs catégoriques : Classe de travail, Education, Situation familiale, Profession, Relation, Race, Sexe, Pays d'origine, Revenu.

# Vérification de la qualité des données : Nous devons toujours vérifier certaines choses concernant la qualité des données.
print("\n=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= QUALITÉ DES DONNÉES *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=**=*=")
print(data.info())
print("\n---------------------------------------------------------------------------------------------------------------")

print("\n=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= DONNÉES MANQUANTES OU NON *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=**=*=")

print("nombre de valeurs manquantes - occupation: ", data['occupation'].isnull().sum())
print("nombre de valeurs manquantes - work-class: ", data['work-class'].isnull().sum())
print("nombre de valeurs manquantes - native-country: ", data['native-country'].isnull().sum())

print("\n---------------------------------------------------------------------------------------------------------------")

# Statistiques de données :
# --------------------------
# En ce qui concerne les attributs numériques, nous commençons par observer diverses statistiques
# telles que le nombre, la moyenne, l’écart type, etc. Voyons quelques-unes des statistiques de nos catégories numériques.
# Les statistiques nous donnent une idée sur les attributs contenant des valeurs manquantes, la valeur moyenne et
# le maximum et le minimum pour un attribut particulier, etc. Ces informations seront très utiles lorsque nous déciderons
# de prétraiter nos données avant de les intégrer à notre modèle.

print("\n=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= STATISTIQUES SUR LES DONNÉES =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=**=*=")

print("Statistiques descriptive pour la variable : age")
print(data['age'].describe())
print("Age Median: ", data['age'].median())

#On fait la meme chose pour les autres attributs : workclass,fnlwgt,education,education-num,marital-status,occupation,
# relationship,race,sex,capital-gain,capital-loss, hours-per-week,native-country,income

#On fait la meme chose : education-num
print("Statistiques descriptive pour la variable :")
print(data['education-num'].describe())
print("Age Median: ", data['education-num'].median())

print("\n---------------------------------------------------------------------------------------------------------------")

# Visualisation de données La visualisation des attributs est peut-être la partie la plus importante et la plus intéressante
# de l’AED. Une information est mieux comprise si nous utilisons une image ou une visualisation. Cependant, nous devons
# toujours garder à l'esprit si la visualisation est appropriée pour le type de données en question. Nous ne devrions pas
# simplement essayer toutes les visualisations que nous connaissons et dire que nous en avons terminé avec L’AED.
# Nous devrions plutôt réaliser des visualisations spécifiques et veiller à bien comprendre ce que la visualisation nous dit.
# Commençons par tracer une visualisation pour chacun des attributs numériques.


print("\n=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= LISTE DES ATTRIBUTS NUMÉRIQUES =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=**=*=\n")


# On va définir la liste des attributs numériques (Mesures)
ListesAttributsNumeriques = data.select_dtypes(include=['int64'])
print(ListesAttributsNumeriques)



# Utilise la liste de mesures pour tracer des histogrammes, avec la taille de la figure avec la fonction (figsize)
ListesAttributsNumeriques.hist(figsize=(12,12))
plt.show()

# Maintenant, pour les attributs catégoriques (Dimension), il serait préférable de voir leurs distributions de fréquence. Nous allons
# utiliser le Count pour y parvenir.

ListesAttributsCategoriques = data.select_dtypes(include=['object'])
print(ListesAttributsCategoriques)
plt.figure(figsize=(12,6))

# On va afficher la catégorie, on utilise les countplot pour tracer un diagramme en barres
sns.countplot(data = ListesAttributsCategoriques, x = "work-class")


# On fait la meme chose pour les autres attributs
# Il faut enlever cette ligne de la section précédente pour avoir tous les graphiques
plt.show()

# Relations de données : Pour explorer les relations entre divers attributs, nous pouvons utiliser des diagrammes de
# dispersion et des cartes thermiques de corrélation entre les différents attributs. Nous examinerons la carte thermique
# de corrélation de divers attributs, à l'exclusion de l'attribut de classe, à savoir le revenu. Encore une fois, nous ne
# pouvons le tracer que pour les attributs numériques.

# On peut ici générer un graphique de la corrélation graphique (du côté diagramme) entre les différents paramètres
sns.pairplot(data)

# Explorez les relations entre les attributs, en excluant l'attribut de categorie (discriminant de couleur)
# selon la version du module le paramètre height n’est pas nécessaire

sns.pairplot(data, height=3, diag_kind = 'kde', hue='income')

# Calculer la matrice de corrélation, et utiliser la force de ces modules
corr = data.corr()
sns.heatmap(corr, annot=True)
plt.title("Corrélation de Pearson")
plt.show()

# Mettre en place la figure matplotlib
f, ax = plt.subplots(figsize=(16, 12))

# Générer une palette de couleurs personnalisée
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Dessine la carte de chaleur
_ = sns.heatmap(corr, cmap="YlGn", square=True, ax=ax, annot=True, linewidth = 0.1)
plt.title('Corrélation de Pearson', y=1.05, size=15)

# Il faut enlever cette ligne de la section précédente pour avoir tous les graphiques
plt.show()