import os
import requests
import pandas as pd
from dotenv import load_dotenv

"""

En este proyecto se analizaran los sentimientos asociados a la tendencia #messi

"""
paginas = 10
# Cargar valores del archivo .env en las variables de entorno
load_dotenv()

# Cargar valor del token a variable
bearer_token = os.environ.get("BEARER_TOKEN")

# URL al que se le van a hacer los requests
url = "https://api.twitter.com/2/tweets/search/recent"

# Parametros de los requests a los tweets
params = {
    'query': 'messi -is:retweet lang:en',
    'tweet.fields':'created_at,author_id',
    'max_results':100
}

# headers para el request a la api
headers = {
        "Authorization": f"Bearer {bearer_token}",
        "User-Agent":"v2FullArchiveSearchPython"
        }

def do_request(header, param):

    # realizamos la request a la api y guardamos la respuesta
    response = requests.get(url, headers=header, params=param)

    # Generar excepción si la respuesta no es exitosa
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)

    return response


resp = do_request(headers, params)

result = [pd.json_normalize((resp.json()["data"]))]
for i in range(paginas):
    if "next_token" not in resp.json()["meta"]:
        break
    params["next_token"] = resp.json()["meta"]["next_token"]
    resp = do_request(headers, params)
    result.append(pd.json_normalize((resp.json()["data"])))



# se guarda la respuesta como un dataframe de pandas
#df = pd.json_normalize(result)
df = pd.concat(result)


# print(response.json()["meta"]["next_token"])

# Se guarda el dataframe en un archivo csv
df.to_csv('tweet_requests.csv') 

