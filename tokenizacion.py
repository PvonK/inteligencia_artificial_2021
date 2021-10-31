import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# leo los requests generados como un dataframe de pandas
df = pd.read_csv("tweet_requests.csv")

# filtro la columna de texto
df = df[["text"]]

# instancio el tokenizador
tt = TweetTokenizer()

# aplico al texto de los tweets el tokenizador
tokenizedtext = df["text"].apply(tt.tokenize)

# guardo el texto tokenizado en el dataframe
df["tokenized_text"] = tokenizedtext

# googlear q hace esto
tokenized_list = df.explode('tokenized_text')

# Obtener frecuencia de cada término
fdist = FreqDist(tokenized_list['tokenized_text'])

# Convertir a dataframe
df_fdist = pd.DataFrame.from_dict(fdist, orient='index')
df_fdist.columns = ['Frequency']
df_fdist.index.name = 'Term'
df_fdist.sort_values(by=['Frequency'], inplace=True)

# guardo las frecuencias de las palabras en un csv 
df_fdist.to_csv('frecuencia_palabras.csv') 

# guardo los stopwords en ingles 
# sacar otras palabras irrelevantes
stop_words = set(stopwords.words('english'))

# elimino las stopwords del df con los tweets
def remove_stopwords(tokenized_text):
    return [x for x in tokenized_text if not x.lower() in stop_words]

no_stopwords = df["tokenized_text"].apply(remove_stopwords)

# guardo no_stopwords en un csv
no_stopwords.to_csv('tweets_no_stopwords.csv')

# hago el analisis de POS para el df
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')
pos = no_stopwords.apply(nltk.pos_tag)

pos.to_csv('analisis_pos')

# lematizacion
nltk.download('wordnet')
# convierto el POS a una sola letra para determinar el tipo de palabra
pos_simplificado = []
for i in pos:
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
