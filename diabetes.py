import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Leer el archivo CSV
datos=pd.read_csv("diabetes.csv")

# Eliminar filas con valores nulos
datos=datos.dropna()

# Seleccionar las columnas que serán utilizadas como características
x=datos[["Pregnancies","Glucose", "BloodPressure","SkinThickness", "Insulin", "BMI","Age"]]

# Seleccionar la columna que será utilizada como variable de destino
y=datos[["Outcome"]]

# Separar los datos en conjuntos de entrenamiento y prueba
x_train,x_test,y_train,y_test=train_test_split(x,y.values.ravel(),test_size=0.4,random_state=42)

# Crear un modelo de regresión logística
modelo=LogisticRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(x_train,y_train)

# Crear un caso de prueba
caso=pd.DataFrame({"Pregnancies":[5],"Glucose":[170],"BloodPressure":[80],"SkinThickness":[30],"Insulin":[32],"BMI":[34],"Age":21})

# Realizar una predicción con el modelo entrenado
print(modelo.predict(caso))

# Calcular la precisión del modelo con los datos de prueba
precision=modelo.score(x_test,y_test)
print("precision del modelo: {:2f}%".format(precision*100))