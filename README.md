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
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Yathin Redddy T
RegisterNumber: 212223100062


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
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
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
  
*/
```

## Output:
![Screenshot 2024-09-03 112333](https://github.com/user-attachments/assets/6b7f5280-e181-4b93-9aa9-b441980aac07)
![Screenshot 2024-09-03 112351](https://github.com/user-attachments/assets/ea7576e7-a9ba-4ea8-a646-5747a569a5f5)
![Screenshot 2024-09-03 112357](https://github.com/user-attachments/assets/678ded29-d3cc-4d88-bde9-44013d0d7953)
![Screenshot 2024-09-03 112404](https://github.com/user-attachments/assets/c34259d0-f47b-41d9-8a00-6c8382dee1a8)
![Screenshot 2024-09-03 112410](https://github.com/user-attachments/assets/f4514d6f-fc40-4642-aa85-59ac6da5c0fc)
![Screenshot 2024-09-03 112415](https://github.com/user-attachments/assets/103b9be8-6cac-487d-b7bb-eaf0f11ae229)
![Screenshot 2024-09-03 112424](https://github.com/user-attachments/assets/2e30d1e2-fe2a-4534-991c-a7b19b2c5352)
![Screenshot 2024-09-03 112433](https://github.com/user-attachments/assets/aacdcd54-53c4-4d71-8a5e-627ee4f2da5f)
![Screenshot 2024-09-03 112440](https://github.com/user-attachments/assets/91480ad6-9f46-4374-aea7-2de492d02c4e)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
