# Trabajo Practico 2, Laboratorio de Datos 1C 2024, Grupo 3

# INTEGRANTES:
#   Pauletti, Guido
#   Paredes, Gerson
#   Catania, Juan Ignacio

# Documentacion del proceso: https://github.com/GuidoPauletti/EMNIST-Classifier

#%%
# Importamos las carpetas necesarias 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, tree
import seaborn as sns


#%%
#Abrimos el csv de trabajo

df = pd.read_csv("emnist_letters_tp.csv", header= None)

#%%

#########################
# ANALISIS EXPLORATORIO #
#########################

print(df.info())

print(df[0].nunique())

print(df[0].value_counts())

"""
#Cantidad de datos: Hay 62400 filas, cada una con 785 columnas.

#Atributos: Hay 785 atributos, 784 numericos y el primero es un caracter, 
    el indicador de la letra (variable de interes).
    
#Clase de interes: Hay 26 valores distintos para la clase de interes, 
    todos presentes en igual proporcion.
    
"""

#%%

# rotador de imagen:
    
def flip_rotate(image):
    """
    Función que recibe un array de numpy representando una
    imagen de 28x28. Espeja el array y lo rota en 90°.
    """
    W = 28
    H = 28
    image = image.reshape(W, H)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

#%%
# Veamos algunas letras:

Es = df[df[0] == "E"]
Ls = df[df[0] == "L"]
Ms = df[df[0] == "M"]

# Definimos los dataframes en un diccionario para facilitar el bucle
letras = {'E': Es, 'L': Ls, 'M': Ms}

fig, axs = plt.subplots(3, 5, figsize=(4, 4))  # 3 filas para las letras, 5 columnas para ejemplos de cada letra

for i, (letras, data) in enumerate(letras.items()):
    # Obtener las primeras 5 imágenes del dataframe actual
    for j in range(5):
        row = data.iloc[j].drop(0)
        image_array = np.array(row).astype(np.float32)
        image = flip_rotate(image_array)
        axs[i, j].imshow(image, cmap='gray')
        axs[i, j].axis('off')
        if j == 0:
            axs[i, j].set_title(letras)

plt.tight_layout()
plt.show()
#%%

# Veamos las C's

Cs = df[df[0] == "C"]

plt.clf()
fig, axs = plt.subplots(4,4, figsize=(4,4))

for i in range(4):
    for j in range(4):
        random = np.random.randint(0,2399) #elejimos un C al azar
        row = Cs.iloc[random].drop(0)
        image_array = np.array(row).astype(np.float32)
        image = flip_rotate(image_array)
        axs[i, j].imshow(image, cmap='gray')
        axs[i, j].axis('off')

plt.tight_layout()
plt.show()

#%%

###############
# EJERCICIO 1 #
###############

"""
(a)
Los atributos de los pixeles son la unica informacion que tenemos para predecir la letra correspondiente, 
por lo que son estos atributos (o columnas) fundamentales para lograr nuestra tarea. 
Sin embargo se puede notar que los valores correspondientes a las esquinas y bordes de la imagen son en su mayoria igual a cero. 
Quizas podriamos reducir la cantidad de piexeles por imagen y prescindir de esos datos.

(b, c)
Las letras no son parecidas, incluso varian mucho las imagenes de una misma letra. 
Sin embargo cabe notar que si se comparan de a tres (como con la E, la M y la L), 
se podría decir que hay un par con mas similitudes entre sí. Para el ejemplo arriba visto podemos observar mayores similitudes entre la M y la E, 
pues ambas presentan un patron de tres rayas, la E hacia la derecha y la M los presenta hacia abajo.

(d)
En este caso la exploración de datos se debe tomar por otro enfoque, 
no se tratará de graficos de caja o de histogramas, ya que no servirian ningún proposito, 
lo importante en una base de datos de imagenes es poder verlas, ver patrones en ellas, 
identificar la cantidad de colores tal vez, el tamaño de cada imagen; y por sobre todas las cosas,
 ver que valores puede tomar cada imagen, para responder la cantidad de posibles valores distintos, 
 y que porcentaje representa cada uno de estos en el conjunto de datos.

"""


#%%
###############
# EJERCICIO 2 #
###############

# (item a)
# Tomamos el dataframe compuesto por las letras A's y L's
LAs = df[df[0].isin(["A","L"])]

# (item b) por analisis anteriores sabemos que estos datos estan
# balanceados, ambas letras estan en 2400 filas

#%%
#(item c)
# separamos las labels

X = LAs.drop(0, axis=1)
y = LAs[0]

#%%
# separamos los datos en conjuntos train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

#%%
#(item d)

resultados3=[]

for i in range(15):   # probamos con 15 combinaciones distintas
    combinacion = np.random.randint(1, 785, size=3)    # tomamos de forma random 3 de las columnas 
    X_train_subset = X_train[combinacion]     # asignamos un subset donde tomo del X_train las columnas random
    X_test_subset = X_test[combinacion]
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_subset,y_train) 
    y_pred = knn.predict(X_test_subset)
    resultados3.append((combinacion,metrics.accuracy_score(y_test, y_pred)))

resultados3 = sorted(resultados3, key=lambda x: x[1], reverse=True)

print(resultados3)

#Notamos que las columnas del medio nos dan un buen resultado, como es de esperarse.

#%%
#Probemos eligiendolas manualmente

combinacion = [391,392,393]    # tomamos las 3 columnas del medio
X_train_subset = X_train[combinacion]     # asignamos un subset donde tomo del X_train las columnas seleccionadas
X_test_subset = X_test[combinacion]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_subset,y_train) 
y_pred = knn.predict(X_test_subset)

print(metrics.accuracy_score(y_test, y_pred))

"""
Dio bastante mal, podemos interpretarlo como que las 3 colunas son muy redundantes in informacion
por lo reduzco la cantidad de datos significativos con los que el modelo puede predecir.

"""

#%%
#Veamos en un gráfico la idea.
plt.clf()

np.random.seed(42)
random = np.random.randint(0,2399, size=16)
fig, axs = plt.subplots(4,4,figsize=(4,4))

for j in range(4):
    for i, letra in enumerate(["A","L","A","L"]):
        row = df[df[0] == letra].iloc[random[i + j + 2]].drop(0)
        image_array = np.array(row).astype(np.float32)
        image = flip_rotate(image_array)
        axs[j,i].imshow(image, cmap='gray')
        axs[j,i].scatter([18,19,20], [19,19,19],color='red') # señalamos 3 atributos cercanos
        axs[j,i].axis('off')

"""
Se puede ver que 3 puntos cercanos no nos dan tanta informacion, 
ambas letras presentan ejemplos de puntos similares, donde hay casos que son todos blancos, 
o todos negros, o donde hay ambos colores; pero la proporcion de estos casos no varía mucho entre estos caracteres, 
por lo que dificulta la obtencion de un buen resultado usando el algoritmo de KNN.

"""

#%%

# Veamos que pasa si ademas cambiamos la cantidad de atributos

# probamos con 9 atributos

resultados9=[]

for i in range(15):   # probamos con 15 combinaciones distintas
    combinacion = np.random.randint(1, 785, size=9)    # tomamos de forma random 9 de las columnas 
    X_train_subset = X_train[combinacion]     # asignamos un subset donde tomo del X_train las columnas random
    X_test_subset = X_test[combinacion]
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_subset,y_train) 
    y_pred = knn.predict(X_test_subset)
    resultados9.append((combinacion,metrics.accuracy_score(y_test, y_pred)))

resultados9 = sorted(resultados9, key=lambda x: x[1], reverse=True)

print(resultados9)

#Hubo una mejora, de un 1%. Veamos que pasa si aumentamos la cantidad de atributos.

#%%
#probemos con 21

resultados21=[]

for i in range(15):   # probamos con 15 combinaciones distintas
    combinacion = np.random.randint(1, 785, size=21)    # tomamos de forma random 21 de las columnas 
    X_train_subset = X_train[combinacion]     # asignamos un subset donde tomo del X_train las columnas random
    X_test_subset = X_test[combinacion]
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_subset,y_train) 
    y_pred = knn.predict(X_test_subset)
    resultados21.append((combinacion,metrics.accuracy_score(y_test, y_pred)))

resultados21 = sorted(resultados21, key=lambda x: x[1], reverse=True)

print(resultados21[:5]) # imprimimos solo los primeros 5, sino es muy largo

#%%
#veamos un poco de informacion sobre los resultados obtenidos

print(f"""
    Con 3 atributos\n
         Exactitud máxima: {resultados3[0][1]:.3}.\n
         Exactitud media: {np.mean([exactitud[1] for exactitud in resultados3]):.3}.\n\n
    Con 9 atributos\n
         Exactitud máxima: {resultados9[0][1]:.3}.\n
         Exactitud media: {np.mean([exactitud[1] for exactitud in resultados9]):.3}.\n\n
    Con 21 atributos\n
         Exactitud máxima: {resultados21[0][1]:.3}.\n
         Exactitud media: {np.mean([exactitud[1] for exactitud in resultados21]):.3}.
""")

"""Observamos que la mejor marca no aumenta significativamente, sin embargo, 
en promedio los resultados para 15 experimentos son mucho mejores."""

#%%
#(item e)

# La idea ahora es comparar los modelos de KNN, perov variando tambien los numeros de vecinos

"""
Nos va a facilitar la tarea modularizar el codigo en funciones que tomen lo que ya hemos hecho hasta ahora. 
Asi que primero hacemos eso para que luego sea mas sencillo entrenar modelos con diferentes parametros.

Elegimos usar 9 atributos, ya que la mayor exactitud no muestra diferencias en comparacion con elegir 21. 
De esta manera ahorramos tiempo, espacio y energia sin comprometer los resultados.

"""

def kneigh(k):
    """
    Dado un k, entrena 15 modelos de kNN
    usando en cada iteración 9 atributos distintos.
    """
    resultados=[]

    for i in range(15):   # pruebo con 15 combinaciones distintas
        combinacion = np.random.randint(1, 785, size=9)    # tomo de forma random 9 de las columnas 
        X_train_subset = X_train[combinacion]     # asigno un subset donde tomo del X_train las columnas random
        X_test_subset = X_test[combinacion]
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_subset,y_train) 
        y_pred = knn.predict(X_test_subset)
        resultados.append((combinacion,metrics.accuracy_score(y_test, y_pred)))

    # ordenamos los resultados obtenidos segun la exactitud
    # para así poder acceder a los atributos con mejor resultado
    # con tan solo elegir el primer elemento de la lista.

    resultados = sorted(resultados, key=lambda x: x[1], reverse=True)
    return resultados[0] 

#%%
#Probamos distintos valores de k.

ks = [3,6,9,12,15,20,25,32]

resultados_ks = {} # guardamos los datos en un dicc.

for k in ks:
    resultado = kneigh(k)
    resultados_ks[k] = resultado

print(resultados_ks)

"""
Observamos que el mejor resultado se obtiene con 3, 
y que ademas no varía mucho la exactitud al cambiar el k, 
las exactitudes tienen un rango entre 94 y 96.5

"""

#%%

"""
Como la mejor exactitud lo dio el menor k de todos, 
probamos con k = 1 y k = 2, tal vez veamos alguna mejora.
"""
print(kneigh(1)) # k = 1
print(kneigh(2)) # k = 2

"""
Ambos dieron mas bajo que k = 3. 
Por lo que nos terminamos por decidir que k = 3 es la mejor opción.

"""
 
#%%
# Concluimos entonces :

print(f"""
    El mejor modelo de kNN (en exactitud):\n
         - Usa k = 3
         - Tiene en cuenta 9 atributos: {resultados_ks[3][0]}
         - Tiene exactitud (en el conjunto de test): {resultados_ks[3][1]:.3} 
""")

#%%
###############
# EJERCICIO 3 #
###############

#(item a)
# separamos las vocales
df_vocales = df[df[0].isin(["A","E","I","O","U"])]

# separamos atributos de etiquetas
X_vocales = df_vocales.drop(0,axis="columns")
y_vocales = df_vocales[0]

# separamos los datos en train y en test
X_voc_train, X_voc_test, y_voc_train, y_voc_test = train_test_split(X_vocales, y_vocales, random_state=42, test_size=0.3)

#%%
#(item b)

# definimos una funcion para entrenar un modelo de arbol de decision
def accuracy_arbol(X, Y, testX, testY, hmax, criterio="gini"):
    arbol = tree.DecisionTreeClassifier(max_depth=hmax, criterion=criterio)
    arbol.fit(X, Y)

    pred = arbol.predict(testX)
    accuracy = np.mean([pred == testY])
    return accuracy

#%%
# probamos distintas profundidades

for h in range(1,12,2):
    acc_arbol = accuracy_arbol(X=X_voc_train,Y=y_voc_train,testX=X_voc_train,testY=y_voc_train,hmax=h)
    print(f'Para {h} de profundid, la exactitud fue de {acc_arbol:.2}')

# Vemos que a partir de 11 la exactitud deja de mejorar, no se puede mejorar mas, estamos arriesgando a un problema de overfitting.

#%%
#(item c)

"""
Ahora hacemos validacion cruzada con k-folding
De esta manera tratamos de encontrar el mejor modelo.

Encontramos conveniente usar un DataFrame para registar los resultados de los experimentos.

"""

# Creamos un DataFrame para registrar altura, criterio,y promedio de exactitud de cada arbol.
data = {"hmax":[],"criterio":[],"promedio_exactitud":[]}
registro = pd.DataFrame(data=data)

#%%
# Ahora creamos una funcion para evaluar los arboles según su altura.

def matriz_kfold(criterio):
    """
    Devuelve el promedio de la exactitud  a travez de k-folds
    para cada altura de arbol de clasificacion.
    """
    alturas = [1,2,3,5,10,15]
    nsplits = 5
    kf = KFold(n_splits=nsplits)

    resultados = np.zeros((nsplits, len(alturas)))

    for i, (train_index, test_index) in enumerate(kf.split(X_voc_train)):

        kf_X_train, kf_X_test = X_voc_train.iloc[train_index], X_voc_train.iloc[test_index]
        kf_y_train, kf_y_test = y_voc_train.iloc[train_index], y_voc_train.iloc[test_index]
        
        for j, hmax in enumerate(alturas):
            score = accuracy_arbol(hmax=hmax,X=kf_X_train,Y=kf_y_train,testX=kf_X_test,testY=kf_y_test,criterio=criterio)
            resultados[i, j] = score

    # para saber a que altura corresponde el puntaje lo devolvemos en tuplas
    scores_promedio = zip(alturas,resultados.mean(axis = 0))
    
    return scores_promedio

#%%
# Nos queda correr el experimento una vez por cada criterio, registrando todo en cada iteracion.

criterios = ["gini", "entropy", "log_loss"] # lista para iterar y realizar los experimentos.

for criterio in criterios:
    scores = matriz_kfold(criterio=criterio)

    # agregamos los resultados al dataframe
    for h, score in scores:
        # agregamos los resultados en una nueva fila
        registro.loc[len(registro.index)] = [h, criterio, score]

# lo ordenamos por exactitud
registro.sort_values(by="promedio_exactitud", ascending=False, inplace=True, ignore_index = True)

#%%
print(registro.head())

"""
Vemos entonces que en el experimento de k-fold el mejor resultado lo obtenemos con arboles:

# De profundidad maxima 10
# Con criterio de 'entropy'
# Obtiene un 92% de exactitud.

"""
#%%
#(item d)

# entrenamos el modelo elegido en el item (c), esta vez con todo el df de entrenamiento
resultado_final = accuracy_arbol(X=X_voc_train
                                ,Y=y_voc_train
                                ,testX=X_voc_test
                                ,testY=y_voc_test
                                ,hmax=10
                                ,criterio="entropy")

print(f"La exactitud (sobre el conjunto de test) del modelo seleccionado es: {resultado_final:.2}")


#Obtenemos una exactitud del 92% como se esperaba por los resultados del k-fold.

#%%
# Ahora analicemos los errores que comete el modelo mediante una matriz de confusion.

def matriz_confusion(pred, actual):
    señalador = {"A":0,"E":1,"I":2,"O":3,"U":4}
    matriz = np.zeros((5,5))

    for i in range(len(pred)):
        matriz[señalador[pred[i]], señalador[actual[i]]] += 1

    return matriz

arbol_elegido = tree.DecisionTreeClassifier(max_depth=10,criterion="entropy")
arbol_elegido.fit(X_voc_train, y_voc_train)
pred = arbol_elegido.predict(X_voc_test)
matrix = matriz_confusion(pred, list(y_voc_test)) # lo pasamos como lista para sacar los indices del split

#%%
# Graficamos

sns.heatmap(matrix, annot=True, fmt=".1f", xticklabels=["A","E","I","O","U"],yticklabels=["A","E","I","O","U"])
plt.xlabel("Valores verdaderos")
plt.ylabel("Valores predichos")

# En el grafico podemos ver como la letra I es la mas distinguibles de todas

