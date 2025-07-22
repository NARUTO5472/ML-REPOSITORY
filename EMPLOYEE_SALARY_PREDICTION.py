import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Indransh\OneDrive\Desktop\INTERNSHIPS\PROJECTS\IBM SKILLSBUILD\DATA\adult 3.csv")

print(data)

# shape of data set
print(data.shape)

print(data.head())
print(data.tail())
print(data.head(7))
print(data.tail(7))

# null values
print(data.isna())
print(data.isna().sum())

print(data.occupation.value_counts())

print(data.gender.value_counts())

print(data.education.value_counts())

print(data["marital-status"].value_counts())

print(data["workclass"].value_counts())

print(data.age.value_counts())

# Data Manipulation

data.occupation.replace({'?':'Others'},inplace = True)
print(data.occupation.value_counts())

data.workclass.replace({'?':'NotListed'},inplace = True)
print(data["workclass"].value_counts())

data = data[data["workclass"] != "Without-pay"]
data = data[data["workclass"] != "Never-worked"]

print(data["workclass"].value_counts())

print(data.shape)

data = data[data["education"] != "5th-6th"]
data = data[data["education"] != "1st-4th"]
data = data[data["education"] != "Preschool"]

print(data.shape)

data.drop(columns = ['education'],inplace = True)

print(data)

plt.boxplot(data['age'])
plt.show()

data = data[(data['age']<= 75) & (data['age']> 17)]
plt.boxplot(data['age'])
plt.show()

# Label Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['workclass'] = encoder.fit_transform(data['workclass'])
data['marital-status'] = encoder.fit_transform(data['marital-status'])
data['occupation'] = encoder.fit_transform(data['occupation'])
data['relationship'] = encoder.fit_transform(data['relationship'])
data['race'] = encoder.fit_transform(data['race'])
data['gender'] = encoder.fit_transform(data['gender'])
data['native-country'] = encoder.fit_transform(data['native-country'])

# Split Data into Input and Output
x = data.drop(columns=['income'])
y = data['income']
print(x)

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
x = scalar.fit_transform(x)
print(x)

# stratify means data divided equaly
# random_state will ensure that "yogeshwaran(name of a student)" is assigned to same internship everytime
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.2, random_state = 23, stratify = y)
print(xtrain)

# MACHINE LEARNING ALGORITHM

# USING KNN CLASSIFIER ALOGORITHM
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(xtrain, ytrain) # input and output training data
predict = knn.predict(xtest)
print(predict)

from sklearn.metrics import accuracy_score
accuracy_score(ytest,predict)

# USING LOGISTIC REGRESSION ALGORITHM
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit (xtrain, ytrain)
predict1 = lr.predict(xtest)
print(predict1)

accuracy_score(ytest,predict)

# USING DEEP LEARNING ALGORITHM
from sklearn.neural_network import MLPClassifier 
clf = MLPClassifier(solver = 'adam', hidden_layer_sizes = (5,2), random_state = 2, max_iter = 2000)
clf.fit(xtrain, ytrain)
predict2 = clf.predict(xtest)
print(predict2)

accuracy_score(ytest, predict2)

print("KNN Accuracy:", accuracy_score(ytest, predict))
print("Logistic Regression Accuracy:", accuracy_score(ytest, predict1))
print("MLP Classifier Accuracy:", accuracy_score(ytest, predict2))

# Therefore after calcuation accuracy of all models the MLP classifier is the most accurate.