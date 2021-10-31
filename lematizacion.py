import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


df = pd.read_csv("analisis_pos.csv")

# lematizacion
nltk.download('wordnet')
# convierto el POS a una sola letra para determinar el tipo de palabra
pos_simplificado = []
for i in df:
    for j in i:
        pos_simplificado.append((j[0], j[1][0]))

# Instancio el lematizador
wordnet_lemmatizer = WordNetLemmatizer()

lemmatized = []
for word, simbol in pos_simplificado:
    lemmatized.append(wordnet_lemmatizer.lemmatize(word, simbol.lower()))
    if len(lemmatized) > 100:
        break

text = " ".join(lemmatized)

# salta un error cuando j[1] es no era N A V o R
# hacer nube de palabras para sustantivo, adjetivo y verbo y ver las palabras a limpiar
