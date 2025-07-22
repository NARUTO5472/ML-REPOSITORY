from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv(r"C:\Users\Indransh\OneDrive\Desktop\INTERNSHIPS\PROJECTS\IBM SKILLSBUILD\DATA\adult 3.csv")

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

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

# Define models
models = {
    "LogisticRegression" : LogisticRegression(max_iter=1000),
    "RandomForest" : RandomForestClassifier(),
    "KNN" : KNeighborsClassifier(),
    "SVM" : SVC(),
    "GradientBoosting" : GradientBoostingClassifier()
}

results = {}

'''for name, model in models.items():
    pipe = Pipeline([
        ('scalar', StandardScaler()),
        ('model', model)
    ])

    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy : {acc:.4f}")
    print(classification_report(y_test,y_pred))
'''

for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Get best model
best_model_name = max(results, key = results.get)
best_model = models[best_model_name]
print(f"\n Best model : {best_model_name} with accuracy {results[best_model_name] :.4f}")

# Save the best model
joblib.dump(best_model, "best_model.pkl")
print("Saved best model as best_model.pkl")

plt.bar(results.keys(), results.values(), color = 'skyblue')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()