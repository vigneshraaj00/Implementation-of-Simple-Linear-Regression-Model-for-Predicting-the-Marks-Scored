# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary libraries for data handling, visualization, and model building.

2.Load the dataset and inspect the first and last few records to understand the data structure.

3.Prepare the data by separating the independent variable (hours studied) and the dependent variable (marks scored).

4.Split the dataset into training and testing sets to evaluate the model's performance.

5.Initialize and train a linear regression model using the training data.

6.Predict the marks for the test set using the trained model.

7.Evaluate the model by comparing the predicted marks with the actual marks from the test set.

8.Visualize the results for both the training and test sets by plotting the actual data points and the regression line

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Name : Vignesh raaj S
RegisterNumber:  212223230239
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
## Output

<img width="281" height="361" alt="Screenshot 2025-08-18 092822" src="https://github.com/user-attachments/assets/1adf4404-ccf9-4e0a-9ee5-905e22e37463" />

```
dataset.info()
```
<img width="484" height="252" alt="Screenshot 2025-08-18 093556" src="https://github.com/user-attachments/assets/19339917-cecf-4da1-90c1-ded2e73207ee" />

```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
<img width="884" height="727" alt="Screenshot 2025-08-18 093649" src="https://github.com/user-attachments/assets/bceaa5d1-8f77-4128-b19e-f56b5d7a4156" />


```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train.shape)
print(X_test.shape)
```
<img width="107" height="67" alt="Screenshot 2025-08-18 093720" src="https://github.com/user-attachments/assets/e90ddc94-cdac-419d-8f9a-6206984023d3" />
```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
```
<img width="338" height="85" alt="Screenshot 2025-08-18 093752" src="https://github.com/user-attachments/assets/5a48ffce-dc55-4b10-9082-447c3ea12aff" />

```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
<img width="873" height="93" alt="Screenshot 2025-08-18 093845" src="https://github.com/user-attachments/assets/cb085c32-461c-4d99-956c-2db5aed8b2b0" />

```
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,reg.predict(X_train),color="green")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
<img width="867" height="688" alt="Screenshot 2025-08-18 093937" src="https://github.com/user-attachments/assets/66ad5eb8-3eb0-4375-9136-05448d65aaa0" />


```
plt.scatter(X_test, Y_test,color="blue")
plt.plot(X_test, reg.predict(X_test), color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

<img width="868" height="695" alt="Screenshot 2025-08-18 094013" src="https://github.com/user-attachments/assets/c11187d9-8be2-4971-bad8-01121c29e334" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
