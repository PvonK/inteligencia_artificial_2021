import pandas as pd
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords


df = pd.read_csv("tokenized_text.csv")

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
