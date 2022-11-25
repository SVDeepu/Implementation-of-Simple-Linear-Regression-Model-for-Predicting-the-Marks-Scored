# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start the program
2.Import the numpy,pandas,matplotlib
3.Read the dataset of student scores
4.Assign the columns hours to x and columns scores to y
5.From sklearn library select the model to train and test the dataset
6.Plot the training set and testing set in the graph using matplotlib library
7.Stop the program
```   

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: prathima
RegisterNumber: 212220040156
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/content/student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values #assigning colum hours to X
X  
Y=dataset.iloc[:,1].values 
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="yellow")
plt.title("Hours vs Scores(Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:
![image](https://github.com/prathima2002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/f2dd2f92bb2b3fedd59052d2ba729eb8b5873687/WhatsApp%20Image%202022-10-31%20at%2020.25.37.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/35c06dd5ea5af3f210aa6e8fcdfecfa731e79d32/WhatsApp%20Image%202022-10-31%20at%2020.25.46.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/afdccccf2c5f48bef94fa5134fb7430b4cb0c1b1/WhatsApp%20Image%202022-10-31%20at%2020.26.12.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/4e2ff2beda5c4f05995ea4ff1bdc63d253d4e536/WhatsApp%20Image%202022-10-31%20at%2020.26.24.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/3572911b7c5b99f28528e1db78a46b423e839a45/WhatsApp%20Image%202022-11-18%20at%2015.49.26.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/e76a48e00d140429a69da99281c56ae735f57779/WhatsApp%20Image%202022-11-24%20at%2013.43.00.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/b934ac2ae5c21471b3f8fb9624009986756cd5c1/WhatsApp%20Image%202022-10-31%20at%2020.26.56.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/04ff39ced2240f19a9757b47c49aad411673ddbe/WhatsApp%20Image%202022-11-18%20at%2015.50.03.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/f8caed71acae6331d1f711f16cc80b10c42135ac/WhatsApp%20Image%202022-11-18%20at%2015.54.30.jpeg)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
