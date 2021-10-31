from re import A
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


###                   ###
#   limpieza de listas  #
###                   ###

def delete_empty(palabras):
    palabras = list(filter(None, palabras))
    return palabras


def delete_spam(palabras):
    indices = []
    for index ,fila in palabras.iterrows():
        #if 'PACK' in fila['tokenized_text']:
        spam = fila['text'].find('PACK')
        if spam > 0:
            indices.append(index)
    palabras = palabras.drop(indices)
    return palabras


def replaceGOAT(palabras):
    
    palabras = palabras.replace(u"\U0001F410", " greatest player of all time ")
    palabras = palabras.replace("goat", " greatest player of all time ")
    palabras = palabras.replace("GOAT", " greatest player of all time ")

    return palabras



def delete_emojis(palabras):

    emoji_pattern = re.compile("["

            u"\u02B0-\u02FF"          # Spacing Modifier Letters
            u"\u2000-\u206F"          # General Punctuation (—)
            u"\u0000-\u002F"          # !"#$%&'()*,-./
            u"\u003A-\u0040"          # :;<=>?@
            u"\u005B-\u0060"          # [\]^_`
            u"\u007B-\u007F"          # {|}~
            u"\u25A0-\u25FF"          # Geometric Shapes (►)
            u"\u2600-\u26FF"          # Miscellaneous Symbols (♥)
            u"\u2580-\u259F"          # Block Elements (▓)
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs (🤩)
            u"\u2700-\u27BF"          # Dingbats (✅)
            u"\U0001F600-\U0001F64F"  # emoticons (😀)
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs (🌀)
            u"\U0001F680-\U0001F6FF"  # transport & map symbols (🚀)
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS) & letters (🇦)
                            "]+", flags=re.UNICODE)

    for i in range(len(palabras)):
        palabras[i] = emoji_pattern.sub('', palabras[i])


def delete_url(palabras):
    for i in range(len(palabras)):
        if palabras[i].startswith("https"):
            palabras[i] = ""


def delete_hashtag(palabras):
    for i in range(len(palabras)):
        if palabras[i].startswith("#"):
            palabras[i] = palabras[i][1:]


###               ###
#   Tokenizacion    #
###               ###


def tokenize(df):

    # filtro la columna de texto
    df = df[["text"]]

    # instancio el tokenizador
    tt = TweetTokenizer()

    # aplico al texto de los tweets el tokenizador
    tokenizedtext = df["text"].apply(tt.tokenize)

    # guardo el texto tokenizado en el dataframe
    df["tokenized_text"] = tokenizedtext

    return df


###                           ###
#   Frecuencia de distribucion  #
###                           ###


def freq_dist(df):
    tokenized_list = df.explode('tokenized_text')

    # Obtener frecuencia de cada término
    fdist = FreqDist(tokenized_list['tokenized_text'])

    # Convertir a dataframe
    df_fdist = pd.DataFrame.from_dict(fdist, orient='index')
    df_fdist.columns = ['Frequency']
    df_fdist.index.name = 'Term'
    df_fdist.sort_values(by=['Frequency'], inplace=True)

    # guardo las frecuencias de las palabras en un csv 
    return df_fdist


def remove_stopwords(df, stop_words):

    # elimino las stopwords del df con los tweets
    def remove_stopwords(tokenized_text):
        return [x for x in tokenized_text if not x.lower() in stop_words]

    df["tokenized_text"].apply(remove_stopwords)

    return df


###             ###
#   Analisis POS  #
###             ###


def POS(df):
    # hago el analisis de POS para el df
    # nltk.download('tagsets')
    # nltk.download('averaged_perceptron_tagger')
    
    a = df["tokenized_text"].apply(nltk.pos_tag)
    df["POS_text"] = a
    
    return df


###                 ###
#   Nube de Palabras  #
###                 ###


def word_cloud_adjetives(df):
    adjetives = []
    for i in df['POS_text']:
        for j in i:
            if j[1].startswith('J'):
                adjetives.append(j[0])
    adjetives = delete_empty(adjetives)

    wordcloud = WordCloud(max_words=100, background_color="white").generate((' '.join(adjetives)))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.rcParams['figure.figsize'] = [150, 150]
    plt.show()

def word_cloud_names(df):
    names = []
    for i in df['POS_text']:
        for j in i:
            if j[1] == 'NNP':
                names.append(j[0])
    names = delete_empty(names)
    wordcloud = WordCloud(max_words=100, background_color="white").generate((' '.join(names)))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.rcParams['figure.figsize'] = [150, 150]
    plt.show()


###             ###
#   Lematizacion  #
###             ###


def get_wordnet_pos(tag):

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def lematizacion(df):

    def simplificar_pos(lista):

        pos_simplificado = []
        for i in lista:
            etiqueta = get_wordnet_pos(i[1])
            if etiqueta == "":
                continue
            pos_simplificado.append((i[0], etiqueta))

        return pos_simplificado

    def lemmatizar(pos_simplificado):
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized = []
        for word, simbol in pos_simplificado:
            lemmatized.append(wordnet_lemmatizer.lemmatize(word, simbol.lower()))
        return lemmatized

    df["POS_text"] = df["POS_text"].apply(simplificar_pos)

    lemmatized = df["POS_text"].apply(lemmatizar)


    df["lemmatized_text"] = lemmatized

    return df


###          ###
#   Polaridad  #
###          ###


def polaridad(df):

    # nltk.download('vader_lexicon')

    # conda install -c conda-forge vadersentiment
    # acordarse de correr el script en el conda env para q no salte erroe con vaderSentiment
    # acordarse de no correrlo con el /bin/python3

    # Instanciar Analizador
    sentiment_analyzer = SentimentIntensityAnalyzer()

    df["negative"] = ""
    df["neutral"] = ""
    df["positive"] = ""
    df["result"] = ""
    for index, row in df.iterrows():
        #Analizar cada review
        analisis = sentiment_analyzer.polarity_scores(" ".join(row['tokenized_text']))
        row["negative"] = analisis["neg"]
        row["neutral"] = analisis["neu"]
        row["positive"] = analisis["pos"]
        # Evaluar que valores se considerarán positivo o negativo
        if analisis['compound'] > 0.5 :
            row["result"] = "Positive"
        elif analisis['compound'] <  0:
            row["result"] = "Negative"
        else :
            row["result"] = "Neutral"

    return df


###    ###
#  Main  #
###    ###


def main():

    # se leen los requests
    df = pd.read_csv("tweet_requests.csv")

    # reemplazo los emojis de cabra y las abreviaciones de GOAT por su significado literal
    df["text"] = df["text"].apply(replaceGOAT)

    # tokenizacion
    df = tokenize(df)

    # se guarda la tokenizacion en un csv
    df.to_csv('tokenized_text.csv')

    # se saca la frecuencia de distribucion
    df_dist = freq_dist(df)

    # se guarda la frecuencia de distribucion en un csv
    df_dist.to_csv('frecuencia_palabras.csv')

    # se buscan las stopwords en ingles
    stop_words = set(stopwords.words('english'))

    # se hace la limpieza de emojis, urls y simbolos
    df["tokenized_text"].apply(delete_emojis)
    df["tokenized_text"].apply(delete_url)
    df["tokenized_text"].apply(delete_hashtag)

    # se hace la limpieza de las stopwords
    df = remove_stopwords(df, stop_words)

    # se guardan los tweets limpiados a un csv
    df.to_csv('tweets_no_stopwords.csv')

    # se hace el analisis pos
    df = POS(df)

    # se guarda el analisis POS a un csv
    df.to_csv('pos.csv')

    # se hace la limpieza de spam
    df = delete_spam(df) 

    # wordcloud de adjetivos
    word_cloud_adjetives(df)

    #wordcloud de nombres propios
    word_cloud_names(df)




    # se hace la lematizacion
    df = lematizacion(df)

    # se limpian los strings vacios del df
    df["lemmatized_text"] = df["lemmatized_text"].apply(delete_empty)

    # se guarda la lematizacion en un csv
    df.to_csv("lemmatized.csv")

    ### POLARIZACION ###

    df = polaridad(df)
    df.to_csv("polarized.csv")

if __name__ == "__main__":
    main()


# palabras como "overrated" y "washed" no son vistas como negativas por el analisis y la palabra "GOAT" no es vista como positiva
# reemplaze goat y 🐐 por "greatest player of all time"