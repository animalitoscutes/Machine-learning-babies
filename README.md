# Machine-learning-babies
en este codigo se puede usar para determinar cual va a ser la gestación de la mujer en base a los datos obtenidos 

Lo primero que se hace es agregar las librerias que se usaran para el pequeño proyecto
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

Luego se carga el documento csv para utilizarlo y poder leerlo en python, esto se hizo de la siguiente manera

datos=pd.read_csv("babies.csv")   (el archivo tiene que estar en la misma carpeta)

luego se agrega otra linea de codigo para que se eliminen las filas vacias en nuestro codigo, ya que los argoritmos LinearRegression no aceptan datos dispares.
datos=datos.dropna()

luego se cargan los datos de la variable x y la variable y que se tomaran de parametro 
x=datos[["age","parity","height","weight","smoke"]]
y=datos[["gestation"]]

luego empezamos con la pruebas y entrenamiento del modelo
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

una vez todo lista hacemos nuesto modelo y lo cargamos con los datos del entrenamiento, en este caso x_train,y_train
modelo=LinearRegression()
modelo.fit(x_train,y_train)

una vez en este punto ya el modelo esta listo para ser usado, solo debemos usarlo, y lo haremos de la siguiente manera.
caso=pd.DataFrame({"age":[27],"parity":[0],"height":[60],"weight":[200],"smoke":[1]})

print(f"los resultados son: {modelo.predict(caso)}")

en este ejercicio se llama a la variable caso dentro del modelo.predict para predecir los datos en la variable. 
los resultados son: [[273.37391868]] -----este es el resultado del modelo. 
