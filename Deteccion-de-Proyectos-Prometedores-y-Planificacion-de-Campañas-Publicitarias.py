#!/usr/bin/env python
# coding: utf-8

# ## Descripción del proyecto
# Trabajas para la tienda online Ice que vende videojuegos por todo el mundo. Las reseñas de usuarios y expertos, los géneros, las plataformas (por ejemplo, Xbox o PlayStation) y los datos históricos sobre las ventas de juegos están disponibles en fuentes abiertas. Tienes que identificar patrones que determinen si un juego tiene éxito o no. Esto te permitirá detectar proyectos prometedores y planificar campañas publicitarias.  
# 
# Delante de ti hay datos que se remontan a 2016. Imaginemos que es diciembre de 2016 y estás planeando una campaña para 2017.  
# 
# Lo importante es adquirir experiencia de trabajo con datos. Realmente no importa si estás pronosticando las ventas de 2017 en función de los datos de 2016 o las ventas de 2027 en función de los datos de 2026.  
# 
# El dataset contiene una columna "rating" que almacena la clasificación ESRB de cada juego. El Entertainment Software Rating Board (la Junta de clasificación de software de entretenimiento) evalúa el contenido de un juego y asigna una clasificación de edad como Adolescente o Adulto.  

# ## Inicialización

# In[1]:


# Cargar todas las librerías
from scipy import stats as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import math


# ## Cargar los datos

# In[2]:


games_df = pd.read_csv("/datasets/games.csv")


# ## Preparar los datos

# In[3]:


#Imprimimos una muestra de los datos
games_df.head() 


# In[4]:


games_df.info()


# In[5]:


#Buscamos valores duplicados
games_df.duplicated().sum() 


# ### Comentarios y observaciones iniciales

# 1.- Los nombres de las columnas contienen mayusculas y requieren ser cambiadas a minusculas.  
# 2.- Hay valores NaN que requieren de atención, la mayoria son de los puntuajes y ratings, y relativamente pocos en nombre, año de lanzamiento y genero.  
# 3.- No parece haber valores duplicados en el dataframe.  
# 4.- La columna "Year_of_Release" cuenta con valores flotantes, se requieren cambiar a valores enteros.  
# 5.- La columna "User_Score" cuenta con valores tipo objeto, ocupan ser de tipo flotante.  
# 6.- No hay valores duplicados.

# ## Corregir los datos

# In[6]:


#Cambiamos los nombres de las columnas a minusculas
games_df.columns = games_df.columns.str.lower() 
print(games_df.columns)


# ### Columna year_of release

# In[7]:


#Obtenemos el numero de valores NaN en la columna
valores_NaN_year = games_df["year_of_release"].isna().sum()
print("La columna year_of_release tiene",valores_NaN_year, "valores NaN.")


# In[8]:


#Rellenamos los valores ausentes con 1979. Este valor querra decir que no contamos con esta información.
games_df["year_of_release"] = games_df["year_of_release"].fillna(1979) 

#Cambiamos los valores de flotantes a enteros.
games_df["year_of_release"] = games_df["year_of_release"].astype(int) 



# ### Columna critic_score

# In[9]:


# Analizamos los datos de la columna "critic_score" para saber como proceder con los datos ausentes.
print("Media: ", games_df["critic_score"].mean())
print("Mediana: ", games_df["critic_score"].median())

#Imprimimos para ver si podemos dectectar la causa de los datos faltantes.
print(games_df[games_df["critic_score"].isna()].head())

# Visualización
plt.figure(figsize=(12, 6))

# Histograma
plt.subplot(1, 2, 1)
sns.histplot(games_df["critic_score"], kde=True)
plt.title("Histograma de puntuaje de criticos")

# Diagrama de Caja
plt.subplot(1, 2, 2)
sns.boxplot(x= games_df["critic_score"])
plt.title("Diagrama de Caja de puntuaje de criticos")

plt.show()


# #### Comentarios
# Viendo el analisis anterior, podemos utilizar la media para complementar los datos ausentes de la columna "critic_score", ya que no hay señal de valores atipicos que puedan afectar los calculos posteriores.
# No parece haber una causa constante de la falta de los datos.

# In[10]:


#Guardamos la media en una variable y lo redondeamos a dos decimales
critic_score_mean = games_df["critic_score"].mean().round(2) 

#Llenamos los valores ausentes con el valor de la media
games_df["critic_score"] = games_df["critic_score"].fillna(critic_score_mean) 


# ### Columna user_score

# Como en esta columna existe el valor "tbd" (to be dertermined), cambiamos este valor a NaN, para poder convertir todos los valores a tipo flotante

# In[11]:


# Convertir la columna a valores numéricos, coerce convierte errores a NaN
games_df["user_score"] = pd.to_numeric(games_df["user_score"], errors="coerce")

#Imprimimos para ver si podemos dectectar la causa de los datos faltantes.
print(games_df[games_df["user_score"].isna()].head())

games_df["user_score"]= games_df["user_score"].astype(float)
# Analizamos los datos de la columna "user_score" para saber como proceder con los datos ausentes.
print("Media: ", games_df["user_score"].mean())
print("Mediana: ", games_df["user_score"].median())

# Visualización
plt.figure(figsize=(12, 6))

# Histograma
plt.subplot(1, 2, 1)
sns.histplot(games_df["user_score"], kde=True)
plt.title("Histograma de calificación de usuarios")

# Diagrama de Caja
plt.subplot(1, 2, 2)
sns.boxplot(x= games_df["user_score"])
plt.title("Diagrama de Caja de calificación de usuarios")

plt.show()


# #### Comentarios
# Viendo el analisis anterior, podemos utilizar la media para complementar los datos ausentes de la columna "user_score", ya que no hay señal de valores atipicos que puedan afectar los calculos posteriores.
# No parece haber una causa constante de la falta de los datos.

# In[12]:


#Guardamos la media en una variable y lo redondeamos a dos decimales
user_score_mean = games_df["user_score"].mean().round(2)

#Llenamos los valores ausentes con el valor de la media
games_df["user_score"] = games_df["user_score"].fillna(user_score_mean) 


# ### Columna Rating

# Como en esta columna no se puede usar la media o mediana para llenar los valores ausentes. Lo llenaremos con la moda de rating de todos los juegos.

# In[13]:


mode_rating = games_df["rating"].mode()

games_df["rating"] = games_df["rating"].fillna(mode_rating)


# ### Columna Name y Genre

# Analizaremos los valores ausentes de estas columnas

# In[14]:


#Veremos cuales son los valores NaN en la columna name
games_df[games_df["name"].isna()] 


# In[15]:


#Veremos cuales son los valores NaN en la columna genre
games_df[games_df["genre"].isna()]


# Como vemos que son los mismos, juntaremos estos dos ya que podria ser el mismo juego del que se habla, ya que son de la misma plataforma, mismo año y se complementan con sus datos faltantes de ventas.

# In[16]:


# Seleccionar el subconjunto específico
subset = games_df[games_df["name"].isna()]

# Calcular la suma de jp_sales en el subconjunto
total_jp_sales = subset['jp_sales'].sum()

# Actualizar la fila específica con la suma calculada
index_to_update = games_df[games_df['name'].isna()].index

# Actualizar el valor de jp_sales para esa fila específica
games_df.loc[index_to_update, 'jp_sales'] = total_jp_sales

#Elimino el renglon
games_df = games_df.drop(14244)


# In[17]:


# Verificamos que los valores se hayan sumado y se haya eliminado el renglon.
games_df[games_df["name"].isna()] 


# In[18]:


# LLenamos los valores NaN con unknown
games_df["name"] = games_df["name"].fillna("Unknown") 
games_df["genre"] = games_df["genre"].fillna("Unknown")


# ### Agregar columna total_sales

# In[19]:


#Sumamos las ventas y lo agregamos a la columna de ventas totales
games_df["total_sales"] = games_df["na_sales"]+games_df["eu_sales"]+games_df["jp_sales"]+games_df["other_sales"]


# In[20]:


#Vemos estado del dataframe
games_df.info() 


# Ya no contamos con valores ausentes y cada columna tiene su tipo de variable correspondiente

# ## Analisis de datos

# ### Mira cuántos juegos fueron lanzados en diferentes años. ¿Son significativos los datos de cada período?

# In[21]:


#Agrupamos por año de lanzamiento
games_per_year = games_df.groupby("year_of_release").size()

# Crear el histograma
plt.figure(figsize=(10, 6))
plt.bar(games_per_year.index, games_per_year.values)

# Etiquetas y título
plt.xlabel("Year of Release")
plt.ylabel("Number of Games")
plt.title("Number of Games Released by Year")
plt.xticks(games_per_year.index, rotation=90)  # Rotar las etiquetas del eje x

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# #### Comentarios
# Del 2002 en adelante es cuando empieza a tener un impacto la cantidad de juegos que van saliendo por año, a comparación de años anteriores que fueron relativamente pocos.


# ### Observa cómo varían las ventas de una plataforma a otra. Elige las plataformas con las mayores ventas totales y construye una distribución basada en los datos de cada año. Busca las plataformas que solían ser populares pero que ahora no tienen ventas. ¿Cuánto tardan generalmente las nuevas plataformas en aparecer y las antiguas en desaparecer?

# In[22]:


#Agrupamos por plataforma y obtenemos sus ventas totales
platform_sales = games_df.groupby("platform")["total_sales"].sum().sort_values(ascending=False)

# Seleccionar las principales diez plataformas
top_platforms = platform_sales.head(10).index

# Filtrar el DataFrame para las principales plataformas
filtered_df = games_df[games_df["platform"].isin(top_platforms)]

# Agrupar y sumar ventas por año y plataforma
annual_sales = filtered_df.groupby(["year_of_release", "platform"])[["total_sales"]].sum().unstack()
annual_sales.columns = annual_sales.columns.droplevel()

# Crear el gráfico
plt.figure(figsize=(14, 8))
for platform in top_platforms:
    if platform in annual_sales.columns:
        plt.plot(annual_sales.index, annual_sales[platform], marker='o', label=platform)

plt.xlabel("Year of Release")
plt.ylabel("Total Sales")
plt.title("Sales Distribution by Platform")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #### Comentarios
# A lo mucho la vida de una plataforma es de 10 años, algunos duran hasta 5 años. Parece que las plataformas salen al mismo tiempo, para hacerle competencia a otras plataformas y no quedarse atras, ronda como cada 7 años sacan otras consolas.

# ### Determina para qué período debes tomar datos. Para hacerlo mira tus respuestas a las preguntas anteriores. Los datos deberían permitirte construir un modelo para 2017. Trabaja solo con los datos que consideras relevantes. Ignora los datos de años anteriores.

# #### Comentarios
# Yo creo que solo debemos trabajar con datos del 2010 en adelante. Ya que las consolas a lo mucho duran 10 años y la mayoria de los que han tenido mayor ventas van de manera descendiente por su antiguedad. 

# In[23]:


#Obtenemos un dataframe con solo los datos que hayan salido durante o despues del 2010.
games_filtered_df = games_df [games_df["year_of_release"] >= 2010] 


# ### ¿Qué plataformas son líderes en ventas? ¿Cuáles crecen y cuáles se reducen? Elige varias plataformas potencialmente rentables.

# In[24]:


#Obtenemos las ventas totales por plataforma y los acomodamos de manera descendente
platform_sales = games_filtered_df.groupby("platform")["total_sales"].sum().sort_values(ascending=False)

# Seleccionar las principales diez plataformas
top_platforms = platform_sales.head(10).index

# Filtrar el DataFrame para las principales plataformas
filtered_df = games_filtered_df[games_filtered_df["platform"].isin(top_platforms)]

# Agrupar y sumar ventas por año y plataforma
annual_sales = filtered_df.groupby(["year_of_release", "platform"])[["total_sales"]].sum().unstack()
annual_sales.columns = annual_sales.columns.droplevel()

# Crear el gráfico
plt.figure(figsize=(14, 8))
for platform in top_platforms:
    if platform in annual_sales.columns:
        plt.plot(annual_sales.index, annual_sales[platform], marker='o', label=platform)

plt.xlabel("Year of Release")
plt.ylabel("Total Sales")
plt.title("Sales Distribution by Platform")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #### Comentarios
# En 2016 parece que hay una baja de ventas o hace falta datos. Sin considerar esto ya que en 2016 todas las plataformas van descendiendo, PS4, XOne van ascendiendo, mientras las demas plataformas van descendiendo.

# ### Crea un diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma. ¿Son significativas las diferencias en las ventas? ¿Qué sucede con las ventas promedio en varias plataformas? Describe tus hallazgos.

# In[25]:


# Diagrama de Caja
plt.figure(figsize=(14, 8))
sns.boxplot(x="platform", y="total_sales", data=games_filtered_df.sort_values(by="total_sales", ascending= True))

# Etiquetas y título
plt.xlabel("Platform")
plt.ylabel("Total Sales")
plt.title("Boxplot of Global Sales by Platform")
plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para mejor legibilidad
plt.grid(True)

plt.show()

# #### Comentarios
# 

# ### Mira cómo las reseñas de usuarios y profesionales afectan las ventas de una plataforma popular (tu elección). Crea un gráfico de dispersión y calcula la correlación entre las reseñas y las ventas. Saca conclusiones.

# In[26]:


#Filtramos el dataframe para que nomas sean juegos de la plataforma X360
X360_games_df = games_filtered_df[games_filtered_df["platform"]== "X360"] 

# Crear gráficos de dispersión para las reseñas de críticos y usuarios contra ventas
plt.figure(figsize=(14, 6))

# Gráfico de dispersión: Reseñas de Críticos vs Ventas
plt.subplot(1, 2, 1)
sns.scatterplot(x="critic_score", y= "total_sales", data=X360_games_df, color='blue', alpha=0.6)
plt.xlabel("Critic Score")
plt.ylabel("Total Sales")
plt.title("Critic Score vs Total Sales")

# Gráfico de dispersión: Reseñas de Usuarios vs Ventas
plt.subplot(1, 2, 2)
sns.scatterplot(x="user_score", y="total_sales", data=X360_games_df, color="green", alpha=0.6)
plt.xlabel("User Score")
plt.ylabel("Total Sales")
plt.title("User Score vs Total Sales")

plt.tight_layout()
plt.show()


# In[27]:


# Calcular la correlación entre las reseñas de críticos y las ventas de X360
critic_corr = X360_games_df[["critic_score", "total_sales"]].corr().iloc[0, 1]
print("Correlation between Critic Score and Total Sales: ", critic_corr)

# Calcular la correlación entre las reseñas de usuarios y las ventas de X360
user_corr = X360_games_df[["user_score", "total_sales"]].corr().iloc[0, 1]
print("Correlation between User Score and Total Sales: ", user_corr)


# #### Comentarios
# La corelación entre el puntuaje de criticos y las ventas totales es ligeramente positiva, mientras que la corelación entre el puntuaje de usuarios y las ventas es practicamente nula. 
# Las ventas iniciales pueden ser ligeramente afectadas por los criticios porque puede haber clientes que lo usen de referencia para ver si compran el juego o no. 
# Mientras que el score de usuario como este surge despues, ya en el transcurso de las compras, probablemente el "hype" del juego ya cayo y de todos modos ya no hay mucha gente interesada en el juego. 

# ### Teniendo en cuenta tus conclusiones compara las ventas de los mismos juegos en otras plataformas.

# In[28]:


#Obtenemos los nombres de los juegos en X360
games_in_X360 = X360_games_df["name"] 

#Obtenemos los juegos que estan en el X360 y se encuentran en otras plataformas
sales_other_platforms =  games_filtered_df[(games_filtered_df["name"].isin(games_in_X360)) & 
                                           (games_filtered_df["platform"] != "X360")]

#Obtenemos los nombres de los juegos que estan en X360 y en otras plataformas
names_other_platforms = sales_other_platforms["name"]

#Filtramos para excluir los juegos exclusivos
same_games_in_X360 = X360_games_df[X360_games_df["name"].isin(names_other_platforms)]

#Agrupamos por genre
same_games_in_X360_genre = same_games_in_X360.groupby("genre")["total_sales"].sum().reset_index().sort_values(by="total_sales")
sales_other_platforms_genre = sales_other_platforms.groupby("genre")["total_sales"].sum().reset_index().sort_values(by="total_sales")


# In[29]:


# Crear gráficos de barra para comparar ventas en la plataforma X360 y en las otras plataformas
plt.figure(figsize=(14, 6))

# Gráfico de barra: Ventas por genero en la plataforma X360
plt.subplot(1, 2, 1)
plt.bar(same_games_in_X360_genre["genre"], same_games_in_X360_genre["total_sales"] , color='blue', alpha=0.6)
plt.xlabel("Genre")
plt.ylabel("Total Sales")
plt.title("Genre vs Total Sales in X360")
plt.xticks(rotation=45)

# Gráfico de barra: Ventas por genero en otras plataformas
plt.subplot(1, 2, 2)
plt.bar(sales_other_platforms_genre["genre"], sales_other_platforms_genre["total_sales"], color="green", alpha=0.6)
plt.xlabel("Genre")
plt.ylabel("Total Sales")
plt.title("Genre vs Total Sales in other platforms")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# #### Comentarios
# Debido a que era una cantidad considerable de juegos, decidi clasificarlo por genero y asi compararlo. Vemos que la relación de top 3 se mantiene (Action, Shooter y Sports). Aunque la cantidad total de ventas en X360 fue menor que en las otras plataformas, hay que recordar que estamos comparando una plataforma contra varias otras, asi que una relacion que X360 sea aproximadamente la mitad que las otras plataformas, muestra que las ventas en X360 son buenas. 

# ### Echa un vistazo a la distribución general de los juegos por género. ¿Qué se puede decir de los géneros más rentables? ¿Puedes generalizar acerca de los géneros con ventas altas y bajas?

# In[30]:


#Agrupamos los juegos por genre y obtenemos los total sales por genre
sales_per_genre = games_filtered_df.groupby("genre")["total_sales"].sum().reset_index().sort_values(by="total_sales")

# Gráfico de barra: Ventas por genero 
plt.bar(sales_per_genre["genre"], sales_per_genre["total_sales"], color="blue", alpha=0.6)
plt.xlabel("Genre")
plt.ylabel("Total Sales")
plt.title("Genre vs Total Sales")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# #### Comentarios
# Los top 3 son los mismos que en la plataforma de X360, nomas se cambio de lugar Sports y Shooter. Puede que en ese entonces los juegos mas sencillos para jugadores causales fueran esos tres, ya que en si una partida puede ser relativamente corta y el jugador puede dejar de jugar en ese momento si asi lo desea sin realmente afectar en la partida.
# 
# Mientras juegos como estrategia, aventura, rompecabezas, requieren de mayor tiempo de inversión para sentir algún progreso o para llegar algún punto donde se puede guardar la partida. No dando espacio para jugadores casuales.

# ## Crear un perfil de usuario para cada región

# ### Región NA

# #### Las cinco plataformas principales. 

# In[31]:


#Obtenemos los top 5 plataformas en NA
top_5_platforms_in_NA = games_filtered_df.groupby("platform")["na_sales"].sum().reset_index().sort_values(by="na_sales", ascending= False).head(5)

# Gráfico de barra: Ventas por genero 
plt.bar(top_5_platforms_in_NA["platform"], top_5_platforms_in_NA["na_sales"], color="blue", alpha=0.6)
plt.xlabel("Platform")
plt.ylabel("Total Sales")
plt.title("Top 5 platform sales in NA")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# #### Los cinco géneros principales. 

# In[32]:


#Obtenemos los top 5 generos en NA
top_5_genres_in_NA = games_filtered_df.groupby("genre")["na_sales"].sum().reset_index().sort_values(by="na_sales", ascending= False).head(5)

# Gráfico de barra: Ventas por genero 
plt.bar(top_5_genres_in_NA["genre"], top_5_genres_in_NA["na_sales"], color="blue", alpha=0.6)
plt.xlabel("Genre")
plt.ylabel("Total Sales")
plt.title("Top 5 genre sales in NA")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# #### Si las clasificaciones de ESRB afectan a las ventas en regiones individuales.

# In[33]:


#Obtenemos los top 5 rating en NA
top_5_rating_in_NA = games_filtered_df.groupby("rating")["na_sales"].sum().reset_index().sort_values(by="na_sales", ascending= False).head(5)

# Gráfico de barra: Ventas por genero 
plt.bar(top_5_rating_in_NA["rating"], top_5_rating_in_NA["na_sales"], color="blue", alpha=0.6)
plt.xlabel("Rating")
plt.ylabel("Total Sales")
plt.title("Top 5 rating sales in NA")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# ### Región EU

# #### Las cinco plataformas principales. Describe las variaciones en sus cuotas de mercado de una región a otra.

# In[34]:


#Obtenemos los top 5 plataformas en EU
top_5_platforms_in_EU = games_filtered_df.groupby("platform")["eu_sales"].sum().reset_index().sort_values(by="eu_sales", ascending= False).head(5)

# Gráfico de barra: Ventas por plataforma
plt.bar(top_5_platforms_in_EU["platform"], top_5_platforms_in_EU["eu_sales"], color="blue", alpha=0.6)
plt.xlabel("Platform")
plt.ylabel("Total Sales")
plt.title("Top 5 platform sales in EU")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# #### Los cinco géneros principales. Explica la diferencia.

# In[35]:


#Obtenemos los top 5 generos en EU
top_5_genres_in_EU = games_filtered_df.groupby("genre")["eu_sales"].sum().reset_index().sort_values(by="eu_sales", ascending= False).head(5)

# Gráfico de barra: Ventas por genero 
plt.bar(top_5_genres_in_EU["genre"], top_5_genres_in_EU["eu_sales"], color="blue", alpha=0.6)
plt.xlabel("Genre")
plt.ylabel("Total Sales")
plt.title("Top 5 genre sales in EU")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# #### Si las clasificaciones de ESRB afectan a las ventas en regiones individuales.

# In[36]:


#Obtenemos los top 5 rating en EU
top_5_rating_in_EU = games_filtered_df.groupby("rating")["eu_sales"].sum().reset_index().sort_values(by="eu_sales", ascending= False).head(5)

# Gráfico de barra: Ventas por rating 
plt.bar(top_5_rating_in_EU["rating"], top_5_rating_in_EU["eu_sales"], color="blue", alpha=0.6)
plt.xlabel("Rating")
plt.ylabel("Total Sales")
plt.title("Top 5 rating sales in EU")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# ### Región JP

# #### Las cinco plataformas principales. Describe las variaciones en sus cuotas de mercado de una región a otra.

# In[37]:


#Obtenemos los top 5 plataformas en JP
top_5_platforms_in_JP = games_filtered_df.groupby("platform")["jp_sales"].sum().reset_index().sort_values(by="jp_sales", ascending= False).head(5)

# Gráfico de barra: Ventas por plataforma
plt.bar(top_5_platforms_in_JP["platform"], top_5_platforms_in_JP["jp_sales"], color="blue", alpha=0.6)
plt.xlabel("Platform")
plt.ylabel("Total Sales")
plt.title("Top 5 platform sales in JP")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# #### Los cinco géneros principales. Explica la diferencia.

# In[38]:


#Obtenemos los top 5 generos en JP
top_5_genres_in_JP = games_filtered_df.groupby("genre")["jp_sales"].sum().reset_index().sort_values(by="jp_sales", ascending= False).head(5)

# Gráfico de barra: Ventas por genero 
plt.bar(top_5_genres_in_JP["genre"], top_5_genres_in_JP["jp_sales"], color="blue", alpha=0.6)
plt.xlabel("Genre")
plt.ylabel("Total Sales")
plt.title("Top 5 genre sales in JP")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# #### Si las clasificaciones de ESRB afectan a las ventas en regiones individuales.

# In[39]:


#Obtenemos los top 5 rating en JP
top_5_rating_in_JP = games_filtered_df.groupby("rating")["jp_sales"].sum().reset_index().sort_values(by="jp_sales", ascending= False).head(5)

# Gráfico de barra: Ventas por rating 
plt.bar(top_5_rating_in_JP["rating"], top_5_rating_in_JP["jp_sales"], color="blue", alpha=0.6)
plt.xlabel("Rating")
plt.ylabel("Total Sales")
plt.title("Top 5 rating sales in JP")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# ### Explica la diferencias de una a otra

# #### En plataformas
# En NA fue X360, PS3, Wii, PS4 y XOne.  
# En EU fue PS3, X360, PS4, PC y Wii.  
# En JP fue 3DS, PS3, PSP, DS y PSV.  
# 
# En NA y EU optaron por las mismas consolas, solo que EU prefirio el X360, mientras que EU prefiere el PS3. Puede ser por los diferentes juegos que hay en cada plataforma.  
# Mientras JP prefiere las consolas portatiles en su mayoria.

# #### En generos
# En NA fue action, shooter, sports, misc y role-playing.  
# En EU fue action, shooter, sports, role-playing y misc.  
# En JP fue role-playing, action, misc, plaftorm y adventure.  
# 
# Las regiones parece que le gustan los mismos generos, JP prefiere role-playing en lugar de los shooter. Esta diferencia debe ser por la preferencia en las plataformas ya que puede que no haya muchos juegos shooter en consolas portatiles.

# #### En ratings
# En NA fue M, E, T, E10+ y EC.  
# En EU fue M, E, T, E10+ y RP.  
# En JP fue E, T, M, E10+ y EC.    
# 
# Las 3 regiones comparten los mismos 4top rating, y el 5to es minoria asi que no tiene impacto. JP fue el diferente que prefiere juegos tipo E mientras EU y NA prefieren juegos tipo M.


# ## Prueba las siguientes hipótesis

# ### Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas.
# 
# 

# In[40]:


#Filtramos los datos para las plataformas en cuestión
XOne_user_score = games_filtered_df[games_filtered_df["platform"]=="XOne"]["user_score"]
PC_user_score = games_filtered_df[games_filtered_df["platform"]=="PC"]["user_score"]

#Calculamos las varianzas
XOne_user_score_var = XOne_user_score.var()
PC_user_score_var = PC_user_score.var()

print("La varianza en XOne es: ", XOne_user_score_var)
print("La varianza en PC es: ", PC_user_score_var)

# Realizar la prueba t para muestras independientes con varianza diferente
t_statistic, p_value = st.ttest_ind(XOne_user_score, PC_user_score, equal_var=False)

# Mostrar el resultado
print("Estadístico t: ", t_statistic)
print("Valor p: ", p_value)

alpha = 0.05  # Nivel de significancia comúnmente usado

# Comparar el valor p con el nivel de significancia
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Hay una diferencia significativa.")
else:
    print("No rechazamos la hipótesis nula: No hay una diferencia significativa.")



# En base los resultados podemos concluir que las calificación de usuario son parecidas, y tiene algo de sentido ya que XOne y las PC ("la mayoria), son de microsoft asi que ambas deben de rondar con las mismas capacidades y compatibilidad. 

# ### Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.

# In[41]:


#Filtramos los datos para las plataformas en cuestión
Action_user_score = games_filtered_df[games_filtered_df["genre"]=="Action"]["user_score"]
Sports_user_score = games_filtered_df[games_filtered_df["genre"]=="Sports"]["user_score"]

#Calculamos las varianzas
Action_user_score_var = Action_user_score.var()
Sports_user_score_var = Sports_user_score.var()

print("La varianza en Action es: ", Action_user_score_var)
print("La varianza en Sports es: ", Sports_user_score_var)

# Realizar la prueba t para muestras independientes con varianza diferente
t_statistic, p_value = st.ttest_ind(Action_user_score, Sports_user_score, equal_var=False)

# Mostrar el resultado
print("Estadístico t: ", t_statistic)
print("Valor p: ", p_value)

alpha = 0.05  # Nivel de significancia comúnmente usado

# Comparar el valor p con el nivel de significancia
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Hay una diferencia significativa.")
else:
    print("No rechazamos la hipótesis nula: No hay una diferencia significativa.")



# Si hay una diferencia entre ambos, dado que sports tiene una gran varianza de un juego a otro, este tiene mas fama de que la mayoria juegos de deportes son de mala calidad, mientras lo de accion si tiene buenos juegos.

# ## Conclusión general

# Diria enforcarse en el mercado de NA y EU porque tienen gustos similares y que sea compatible con las plataformas de Xbox y PS, el que sea de ultima generacion.
# Ya que las consolas como X360 y PS3, apesar de tener buenas ventas, ya estan llegando a sus ultimas.  
# Un juego rating M y que sea action o shooter tiene buenas ventas en estas regiones, asi que si el juego puede ser de esos ratings y generos. Tendrá ya una base solida de prospectos que le gustan ese tipo de juegos.


# In[ ]:




