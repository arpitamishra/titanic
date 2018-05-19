import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score
#reading data
df = pd.read_csv("F:\\VIT material\\Datasets\\titanic_training data.csv")
#print(df)
#print(df.columns)
#deleting unnecessary columns
df.drop(['PassengerId','Name','SibSp','Parch','Fare','Cabin','Ticket'],axis=1,inplace=True)
#print(df)
df.isnull()
#print(df.isnull())
#replacing string value with number
df.Embarked[df.Embarked =='S']=1
df.Embarked[df.Embarked =='C']=2
df.Embarked[df.Embarked=='Q']=3
#print(df)
df.Sex[df.Sex=='male']=1
df.Sex[df.Sex=='female']=2
#print(df)
#print(df.isnull())
newdata = df.fillna(int(df.Age.mean()))
print(newdata)
print(newdata.isnull())
print(newdata.columns)
X=newdata['Sex']
Y=newdata['Survived']
X = X.reshape(len(X), 1)
Y = Y.reshape(len(Y), 1)
# Split the data into training/testing sets
X_train = X[:-10]
X_test = X[-10:]
# Split the targets into training/testing sets
Y_train = Y[:-10]
Y_test = Y[-10:]
#print(X_train)
#print(Y_train)
#print(X_test)
#print(Y_test)
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
print(predictions)
#accuracy_score = accuracy_score(X_train,X_test,normalize=True,sample_weight=None)
#print(accuracy_score)
prediction = pd.DataFrame(predictions, columns=['predictions']).to_csv('F:\\VIT material\\Datasets\\prediction.csv')



