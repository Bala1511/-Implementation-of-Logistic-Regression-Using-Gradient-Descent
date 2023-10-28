# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: BALA MURUGAN P
RegisterNumber:  212222230017
*/
```
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SANJAY S
RegisterNumber:  212222230132
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data2.txt",delimiter = ',')
x= data[:,[0,1]]
y= data[:,2]
print('Array Value of x:')
x[:5]

print('Array Value of y:')
y[:5]

print('Exam 1-Score graph')
plt.figure()
plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label=' Not Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()


def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print('Sigmoid function graph: ')
plt.plot()
x_plot = np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()


def costFunction(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad = np.dot(x.T,h-y)/x.shape[0]
  return j,grad


print('X_train_grad_value: ')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
j,grad = costFunction(theta,x_train,y)
print(j)
print(grad)


print('y_train_grad_value: ')
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j


def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j

print('res.x:')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)


def plotDecisionBoundary(theta,x,y):
  x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot = np.c_[xx.ravel(),yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
  plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label='Not Admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel('Exam  1 score')
  plt.ylabel('Exam 2 score')
  plt.legend()
  plt.show()

print('DecisionBoundary-graph for exam score: ')
plotDecisionBoundary(res.x,x,y)

print('Proability value: ')
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)


def predict(theta,x):
  x_train = np.hstack((np.ones((x.shape[0],1)),x))
  prob = sigmoid(np.dot(x_train,theta))
  return (prob >=0.5).astype(int)


print('Prediction value of mean:')
np.mean(predict(res.x,x)==y)
*/
```

## Output:
![logistic regression using gradient descent](sam.png)

![image](https://github.com/Bala1511/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680410/820e8e42-da6b-438f-bea0-df3957cc362e)
![image](https://github.com/Bala1511/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680410/018d7465-a179-4a25-bb8a-9e7472626cab)
![image](https://github.com/Bala1511/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680410/ee34d2fa-ee7a-4675-ba9a-33227c8aee5f)
![image](https://github.com/Bala1511/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680410/25e18fbd-2c71-4091-9eac-c75f9073a9ee)
![233582100-cc19d7f9-5098-482f-b21d-2d93e55182c2](https://github.com/Bala1511/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680410/e7205d18-ac4a-4c34-afbc-0fa14143c8f6)
![233582146-b08d7d0c-d057-407c-8a8b-07d6ce44c6fa](https://github.com/Bala1511/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680410/c674163f-a2d0-4469-9e7b-4a46362563b8)
![233583099-a9d2c786-b309-4c5d-823f-68dfa4b08c44](https://github.com/Bala1511/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680410/549203e0-89e2-46f9-a630-f0e58a4e23f3)

![233583836-f0176c08-22f9-4dd0-879f-dace6af9ad6c](https://github.com/Bala1511/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680410/3820fa92-df97-4a21-8e8d-354281984091)

![233583637-3c3321e7-55ba-4c54-b1ee-68e567618e7e](https://github.com/Bala1511/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680410/3b8c033a-31b7-4976-9ad6-2d3076253ec7)

![233585133-d0416985-066b-4a55-9093-8a4a0e657bec](https://github.com/Bala1511/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680410/c25a2c4c-1d7f-4d00-b795-5c0cbb4e874d)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

