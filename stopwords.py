import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


df = pd.read_csv("tokenized_text.csv")

# guardo los stopwords en ingles 
# sacar otras palabras irrelevantes
stop_words = set(stopwords.words('english'))

# elimino las stopwords del df con los tweets
def remove_stopwords(tokenized_text):
    return [x for x in tokenized_text if not x.lower() in stop_words]

no_stopwords = df["tokenized_text"].apply(remove_stopwords)

# guardo no_stopwords en un csv
no_stopwords.to_csv('tweets_no_stopwords.csv')
