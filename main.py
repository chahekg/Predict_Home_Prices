import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data=pd.read_csv("C://Users//Chahek Gupta//Downloads//home_dataset.csv")
house_sizes=data["HouseSize"].values
house_prices=data["HousePrice"].values

x_train,x_test,y_train,y_test=train_test_split(house_sizes,house_prices,test_size=0.2,random_state=42)
x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
model_lr=LinearRegression()
model_lr.fit(x_train,y_train)
prediction=model_lr.predict(x_test)


plt.scatter(house_sizes,house_prices,marker='o',color='red',label="Actual Prices")
plt.plot(x_test,prediction,color="blue",linewidth=2,label="Prediction")
plt.title("Predicting Home Prices using Linear Regression")
plt.xlabel("House Sizes")
plt.ylabel("House Prices")
plt.legend()
plt.show()

