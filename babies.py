import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

datos=pd.read_csv("babies.csv")
datos=datos.dropna()

x=datos[["age","parity","height","weight","smoke"]]
y=datos[["gestation"]]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
modelo=LinearRegression()
modelo.fit(x_train,y_train)
caso=pd.DataFrame({"age":[27],"parity":[1],"height":[60],"weight":[200],"smoke":[1]})

print(f"los resultados son: {modelo.predict(caso)}")


