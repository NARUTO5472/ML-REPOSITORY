import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)
my_data

my_data.info()

label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex']) 
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol']) 
my_data

my_data.isnull().sum()

custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)
my_data

category_counts = my_data['Drug'].value_counts()

# Plot the count plot
plt.bar(category_counts.index, category_counts.values, color='blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)  # Rotate labels for better readability if needed
plt.show()

y = my_data['Drug']
X = my_data.drop(['Drug','Drug_num'], axis=1)

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

drugTree.fit(X_trainset,y_trainset)

tree_predictions = drugTree.predict(X_testset)

print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))

plot_tree(drugTree)
plt.show()

# Input mapping based on LabelEncoder usage
sex_map = {'F': 0, 'M': 1}
bp_map = {'LOW': 1, 'NORMAL': 2, 'HIGH': 0}  # based on how LabelEncoder encoded it
chol_map = {'NORMAL': 1, 'HIGH': 0}          # based on how LabelEncoder encoded it

print("Predict Drug Type by giving input : ")
try:
    age = float(input("Enter age: "))
    sex = input("Enter sex (M/F): ").upper()
    bp = input("Enter blood pressure (LOW/NORMAL/HIGH): ").upper()
    chol = input("Enter cholesterol level (NORMAL/HIGH): ").upper()
    na_to_k = float(input("Enter sodium-to-potassium ratio: "))

    # Validate and encode
    sex_encoded = sex_map.get(sex)
    bp_encoded = bp_map.get(bp)
    chol_encoded = chol_map.get(chol)

    if None in [sex_encoded, bp_encoded, chol_encoded]:
        raise ValueError("Invalid categorical input.")

    # Create input array
    new_sample = np.array([[age, sex_encoded, bp_encoded, chol_encoded, na_to_k]])

    # Predict
    prediction = drugTree.predict(new_sample)[0]
    print(f"Predicted Drug: {prediction}")

except Exception as e:
    print(f"Prediction failed: {e}")
