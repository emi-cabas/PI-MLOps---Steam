# Proyecto Final 1 - Machine Learning Operations

![PLATAFORMA STEAM](steam-1.jpg_759710130.webp)

El desarrollo de este proyecto se centra en hacer todas las transformaciones y exploraciones necesarias para poder arribar a un sistema capaz de predecir el precio de un juego basado en los géneros de los juegos y calificaciones de los mismos.
Extraje la información sobre juegos, reseñas, calificaciones y precios desde el extenso dataset de Steam.

Dividí el proyecto en dos notebooks: en [Funciones y EDA](Funciones y EDA.ipynb) se encuentra la construcción del dataset que usé para las funciones.

## Transformaciones de los Datos

Los datos fueron extraídos de un dataset en formato JSON:

- El primer paso fue observar cuál era la estructura del DF y posteriormente revisar los nulos que contenía cada columna. Observé que la columna "title" contenía muchos nulos a comparación de "app_name" entonces opté por trabajar con "app_name".
- Eliminé columnas como "url", "reviews_url", "title" porque me pareció que eran irrelevantes para las funciones que necesitaba realizar y para el modelo de ML.
- El formato de la columna 'release_date' lo modifiqué a DATETIME, y luego extraje el año para poder crear la columna "release_year".

### Precio

Creé una columna nueva en el DF a partir de "price" llamada "precio", pero con una serie de transformaciones y limpieza del conjunto de datos.
Había valores en formato STRING y para poder hacer un análisis preciso pasé todos a un formato numérico. Para lograr esto, reemplacé los valores string en la columna "precio" con valores numéricos correspondientes o con nulos, según fuera necesario. Los valores "Free To Play", "Free", "Play for Free!", y otros valores similares se reemplazaron por 0, lo que refleja la ausencia de costo. Los valores como "Free Movie" e "Install Now" se convirtieron en valores nulos ("NaN") para indicar que no se disponía de información de precio.

[DATASET MODIFICADO](steam_csv)

#### Modelo de Predicción

[MODELO](MLearning.ipynb)

Para construir el modelo filtré algunas columnas como 'genres', 'precio', 'sentiment' y "release_year".

La función creada permite realizar predicciones de precio y calcular el Root Mean Squared Error (RMSE) para juegos basados en género y sentimiento. Utiliza un modelo de regresión lineal para predecir el sentimiento codificado de los juegos en función de su precio y año de lanzamiento.

##### Razones por las cuales elegí "genres" y "sentiment" como parámetros

Usé un modelo de regresión lineal y elegí como parámetros "genre" y "sentiment"

- Porque son dos factores que a menudo influyen en la popularidad y demanda de un juego en Steam. Los consumidores pueden tener preferencias específicas por ciertos géneros y sentimientos, lo que puede afectar sus decisiones de compra. Por ejemplo, si un consumidor ve que el juego está calificado como malo (Overwhelmingly Negative, Negative) su decisión de compra se puede ver afectada y mientras menos demanda haya el precio puede tender a disminuir.

- Los géneros populares pueden atraer a una base de jugadores más grande, mientras que un sentimiento positivo podría indicar una recepción favorable por parte de los jugadores y, por lo tanto, un mayor interés.

- Son variables que son relativamente fáciles de entender e interpretar.

## Otras Funciones de la API

- def genero: Se ingresa un año y devuelve una lista con los 5 géneros más vendidos en el orden correspondiente.
- def juegos: Se ingresa un año y devuelve una lista con los juegos lanzados en el año.
- def specs: Se ingresa un año y devuelve una lista con los 5 specs que más se repiten en el mismo en el orden correspondiente.
- def earlyacces: Se ingresa un año y retorna la cantidad de juegos lanzados con early access.
- def sentiment: Según el año de lanzamiento, se devuelve una lista con la cantidad de registros que se encuentran categorizados con un análisis de sentimiento.
- def metascore: Top 5 juegos según año con mayor metascore.