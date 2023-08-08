from fastapi import FastAPI
app = FastAPI()
app.title = "Proyecto Individual 1: Steam - Emily"

import pandas as pd
dframe = pd.read_csv("steam_csv")

# Función Género
def genero(year):
    # Filtrar el DataFrame para obtener solo los datos del año proporcionado:
    df_anio = dframe[dframe['release_year'] == year]

    # Verificar si hay datos para el año ingresado en el DataFrame
    if df_anio.empty:
        return {"mensaje": "No hay información acerca de ese año"}

    top_5_generos = df_anio['genres'].explode().value_counts().head(5).index.tolist()

    generos_mas_vendidos_dict = {
        'Año': year,
        'Top 5 Géneros': top_5_generos
    }
    
    return generos_mas_vendidos_dict

# Define la ruta para llamar a la función genero(year)
@app.get("/genero/{year}", tags=["Top 5 Géneros"])
async def top_5_géneros(year: int):
    generos_mas_vendidos = genero(year)
    return generos_mas_vendidos


# Defino la función:
def juegos(year):
    # Filtro el DF para obtener solo los juegos lanzados en el año proporcionado:
    df_year = dframe[dframe['release_year'] == year]

    # Verifico si hay datos para el año ingresado en el DF
    if df_year.empty:
        return {"mensaje": "No hay información acerca de ese año"}

    # Creo un diccionario con la información del año y los juegos lanzados:
    juegos_lanzados = {
        'Año': year,
        'Juegos': df_year['app_name'].tolist()
    }
    
    return juegos_lanzados

# Defino la ruta para llamar a la función juegos(year)
@app.get("/juegos/{year}", tags=["Juegos Lanzados"])
async def get_juegos(year: int):
    juegos_lanzados = juegos(year)
    return juegos_lanzados

# Función Specs:
def specs(year):
    # Filtro el DF para obtener solo los datos del año proporcionado:
    df_anio = dframe[dframe['release_year'] == year]

    # Verifico si hay datos para el año ingresado en el DF:
    if df_anio.empty:
        return {"mensaje": "No hay información acerca de ese año"}

    # Obtener los 5 specs más comunes para el año proporcionado:
    top_5_specs = df_anio['specs'].explode().value_counts().head(5).index.tolist()

    # Crear un diccionario con la información del año y los 5 specs más comunes:
    top5_specs_dict = {
        'Año': year,
        'Top 5 Specs': top_5_specs
    }

    return top5_specs_dict

# Definir la ruta para llamar a la función specs(year):
@app.get("/specs/{year}", tags=["Top 5 Specs"])
async def get_specs(year: int):
    specs_info = specs(year)
    return specs_info

# Función Early_Access:
def earlyacces(year):
    df_year = dframe[dframe['release_year'] == year]
    
    if df_year.empty:
        return {"mensaje": "No hay información acerca de ese año"}
    else:
        cantidad_juegos_early = df_year['early_access'].sum()
    
    early_access_dict = {
        'Año': year,
        'Cantidad de Juegos': cantidad_juegos_early
    }
    
    return early_access_dict


# Ruta para llamar a la función earlyacces(year):
@app.get("/earlyacces/{year}", tags=["Juegos con Early Access"])
async def get_earlyacces(year: int):
    earlyacces_info = earlyacces(year)
    return earlyacces_info

# Función Sentiment:
def sentiment(year):
    # Filtro el DF para obtener solo los datos del año proporcionado:
    df_anio = dframe[dframe['release_year'] == year]

    # Verificar si hay datos para el año ingresado en el DataFrame
    if df_anio.empty:
        return {"mensaje": "No hay información acerca de ese año"}

    # Calcular la cantidad de registros por cada categoría de sentimiento y obtener el diccionario resultante
    sentimiento_contador = df_anio['sentiment'].value_counts().to_dict()

    # Crear un diccionario con la información del año y la cantidad de registros por cada categoría de sentimiento
    sentimiento_dict = {
        'Año': year,
        'Categorías': sentimiento_contador
    }

    return sentimiento_dict

# Ruta para llamar a la función sentiment(year):
@app.get("/sentiment/{year}", tags=["Calificaciones para el Año"])
async def get_sentiment(year: int):
    sentiment_info = sentiment(year)
    return sentiment_info

# Función Metascore:
def metascore(year):
    # Filtro el DF para obtener solo los datos del año proporcionado:
    df_anio = dframe[dframe['release_year'] == year]

    # Verificar si hay datos para el año ingresado en el DataFrame
    if df_anio.empty:
        return {"mensaje": "No hay información acerca de ese año"}

    # Ordenar los datos por metascore de forma descendente (mayor a menor):
    df_anio_ordenado = df_anio.sort_values(by='metascore', ascending=False)

    # Seleccionar los primeros 5 registros (los 5 juegos con mayor metascore) y obtener los nombres de esos juegos:
    top_5_juegos = df_anio_ordenado.head(5)['app_name'].tolist()

    # Crear un diccionario con la información del año y los top 5 juegos según el metascore:
    metascore_dict = {
        'Año': year,
        'Top 5 Juegos Metascore': top_5_juegos
    }

    return metascore_dict

# Define la ruta para llamar a la función metascore(year)
@app.get("/metascore/{year}", tags=["Top 5 Juegos por Metascore"])
async def get_metascore(year: int):
    metascore_info = metascore(year)
    return metascore_info

# Modelo de ML:

nuevo_df = dframe[['genres', 'precio', 'sentiment', "release_year"]].copy()
df1 = nuevo_df.dropna(subset=['genres'])

df1 = df1[df1['release_year']>= 2015]

df1 = df1.dropna(subset=['genres'])

df1 = df1.dropna(subset=['precio'])

valores_a_borrar = ['1 user reviews', '3 user reviews', '6 user reviews', '5 user reviews', '2 user reviews', '4 user reviews', '8 user reviews', '7 user reviews', '9 user reviews']

df1 = df1[~df1['sentiment'].isin(valores_a_borrar)]

from fastapi import FastAPI
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define la función prediccion(genero, sentiment) con tu lógica
def prediccion(genero, sentiment):
    # Comprobar si el género o el sentimiento no están presentes en el dataset
    if (df1['genres'].apply(lambda x: genero in x)).sum() == 0 or (df1['sentiment'] == sentiment).sum() == 0:
        return "No hay información disponible acerca de esos parámetros", None
    
    # Filtro el DF por género y sentimiento
    datos_filtrados = df1[df1['genres'].apply(lambda x: genero in x)]
    
    # Codificamos las categorías de sentimiento utilizando Label Encoding
    le = LabelEncoder()
    datos_filtrados['sentiment_encoded'] = le.fit_transform(datos_filtrados['sentiment'])
    
    # Divido los datos en conjuntos de entrenamiento y prueba
    X = datos_filtrados[['precio', 'release_year']]
    y = datos_filtrados['sentiment_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Verificar si hay suficientes datos para entrenar el modelo
    if len(X_train) < 2:
        return None, None
    
    # Creo y entreno el modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Realizamos predicciones en el conjunto de prueba
    y_pred = modelo.predict(X_test)
    
    # Calculo el RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    # Retorna el precio y el RMSE
    return datos_filtrados['precio'].values[0], rmse

# Define la ruta para llamar a la función prediccion(genero, sentiment)
@app.get("/prediccion/", tags=["Modelo Predicción"])
async def get_prediccion(genero: str, sentiment: str):
    precio, rmse = prediccion(genero, sentiment)
    if precio is None:
        return {"mensaje": "No hay suficientes datos para entrenar el modelo"}
    prediccion_info = prediccion(genero, sentiment)
    if "mensaje" in prediccion_info:
        return prediccion_info
    return {"precio": prediccion_info[0], "RMSE": prediccion_info[1]}