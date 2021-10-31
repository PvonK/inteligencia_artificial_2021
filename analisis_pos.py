import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


df = pd.read_csv("tweets_no_stopwords.csv")

# hago el analisis de POS para el df
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')
pos = df.apply(nltk.pos_tag)

pos.to_csv('analisis_pos.csv')
