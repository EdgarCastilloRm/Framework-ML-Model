#Edgar Castillo Ramírez A00827826
#11 de septiembre del 2022
#Modelo de Regresión por Árbol de Decisión utilizando librerías/framework de Machine Learning (Versión 1.0).

#Librerías Generales
import pandas as pd #Para manejar dataset
import matplotlib.pyplot as plt #Para graficar
import numpy as np

#Se importa el dataset
df = pd.read_csv("Estatura-peso_HyM.csv")

#Se imprimen los primeros 5 valores para ver los datos
print("---------------VISUALIZACIÓN DE DATOS---------------\n")
print(df.head())

#Se llama esta función para revisar que las columnas a utilizar tengan todos sus valores.
print("\n---------------VALORES FALTANTES---------------")
print(df.info())

#Revisar datos duplicados
print("\n---------------VALORES DUPLICADOS---------------")
print("Suma de duplicados: " + str(df.duplicated().sum()))

#Eliminar filas
df=df.drop(["M_estat","M_peso"],axis=1).copy()

#Librerías de ML para el modelo
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#Modelo

#Definición de X y Y
Y=df.H_peso
X=df.drop(["H_peso"],axis=1).copy()

x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.2,random_state=42) #Un 20% de los datos serán de prueba

print("\n---------------DIMENSIONES DE DATOS DE ENTRENO Y PRUEBA---------------")
print("De 220 datos dentro de la tabla: ")
print("x_train: " + str(x_train.shape))
print("x_test: " + str(x_test.shape))
print("y_train: " + str(y_train.shape))
print("y_test: " + str(y_test.shape))

#Creación del modelo
model = DecisionTreeRegressor()

#Se entrena el modelo
model.fit(x_train, y_train)

#Métricas para evaluar nuestro modelo (valores de entreno)
y_hat_train=model.predict(x_train)
mse_train=mean_squared_error(y_hat_train,y_train)
mae_train=mean_absolute_error(y_hat_train,y_train)
r2_train=r2_score(y_hat_train,y_train)

print("\n---------------MÉTRICAS DE EVALUACIÓN (TRAIN)---------------")
print("RMSE: ", np.sqrt(mse_train))
print("MAE: ", mae_train)
print("R^2: " ,r2_train)

#Métricas para evaluar nuestro modelo (valores de prueba)
y_hat_test=model.predict(x_test)
mse_test=mean_squared_error(y_hat_test,y_test)
mae_test=mean_absolute_error(y_hat_test,y_test)
r2_test=r2_score(y_hat_test,y_test)

print("\n---------------MÉTRICAS DE EVALUACIÓN (TEST)---------------")
print("RMSE: ", np.sqrt(mse_test))
print("MAE: ", mae_test)
print("R^2: " ,r2_test)


#Se imprime la comparación de los datos
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))
axes[0].scatter(x_test, y_test, edgecolors='black', alpha=0.5)
axes[0].set_title("Original Test Data")

axes[1].scatter(x_test, y_hat_test, color="red", edgecolors='black', alpha=0.5)
axes[1].set_title("Predicted Test Data")

axes[2].scatter(x_test, y_test, edgecolors='black', alpha=0.5)
axes[2].scatter(x_test, y_hat_test, color="red", edgecolors='black', alpha=0.5)
axes[2].set_title("Predicted vs Original")
fig.tight_layout()
plt.show()