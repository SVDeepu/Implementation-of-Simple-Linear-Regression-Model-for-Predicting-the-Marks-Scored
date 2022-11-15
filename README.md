# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: B S SAI HARSHITHA
RegisterNumber:  212222004139
*/
```
```
# implement a simple regression model for predicting the marks scored by the students

import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset)

# assigning hours to X & Scores to Y
X=dataset.iloc[:,:1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:

![image](https://github.com/saiharshithabs/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/c7d928376eb61a4f47b0d21ac96c5b16944b3142/WhatsApp%20Image%202022-10-13%20at%2011.01.57%20AM.jpeg)

![image](https://github.com/saiharshithabs/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/709c7ee4312ada713e0c5145a04914549df922f4/WhatsApp%20Image%202022-10-13%20at%2011.14.23%20AM.jpeg)

![image](https://github.com/saiharshithabs/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/050156e0335af8b746ee7cabdbf5d0181c92f57f/WhatsApp%20Image%202022-10-13%20at%2011.18.35%20AM.jpeg)

![image](https://github.com/saiharshithabs/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/f8b935fd6d9469f2022a94604ece55b8ef9e6929/WhatsApp%20Image%202022-10-13%20at%2011.21.30%20AM.jpeg)

![image](https://github.com/saiharshithabs/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/c2d4e224135372d7c39a6e0ed40d8b6f71bdf551/WhatsApp%20Image%202022-10-13%20at%2011.24.20%20AM.jpeg)

![image](https://github.com/saiharshithabs/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/bebc924ea333ee53dd33a17c5ea474262e67e906/WhatsApp%20Image%202022-10-13%20at%2011.27.23%20AM.jpeg)

![image](https://github.com/saiharshithabs/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/fb229bdead460d6ef57530c248b60746f6ff3bd4/WhatsApp%20Image%202022-10-13%20at%2011.30.16%20AM.jpeg)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
