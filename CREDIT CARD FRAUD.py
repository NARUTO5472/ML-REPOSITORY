from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')

#Load the data

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

# read the input data
raw_data=pd.read_csv(url)
raw_data

# get the set of distinct classes
labels = raw_data.Class.unique()

# get the count of each class
sizes = raw_data.Class.value_counts().values

# plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.show()

correlation_values = raw_data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10, 6))

# standardize features by removing the mean and scaling to unit variance
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
data_matrix = raw_data.values

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]

# y: labels vector
y = data_matrix[:, 30]

# data normalization
X = normalize(X, norm="l1")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

w_train = compute_sample_weight('balanced', y_train)

# for reproducible output across multiple function calls, set random_state to a given integer value
dt = DecisionTreeClassifier(max_depth=4, random_state=35)

dt.fit(X_train, y_train, sample_weight=w_train)

# for reproducible output across multiple function calls, set random_state to a given integer value
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

svm.fit(X_train, y_train)

y_pred_dt = dt.predict_proba(X_test)[:,1]

roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

y_pred_svm = svm.decision_function(X_test)

roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

print("Predict Credit Card Fraud from user Input")
try:
    # List of input features (V1 to V28 + Amount), excluding Time and Class
    feature_names = raw_data.columns[1:30].tolist()
    
    # Gather input
    user_input = []
    for feature in feature_names:
        val = float(input(f"Enter value for '{feature}': "))
        user_input.append(val)
    
    # Convert to DataFrame for scaling
    input_df = pd.DataFrame([user_input], columns=feature_names)

    # Standardize using same process as training
    input_scaled = StandardScaler().fit(raw_data.iloc[:, 1:30]).transform(input_df)

    # Normalize using L1 norm (same as training)
    input_normalized = normalize(input_scaled, norm="l1")

    # Decision Tree Prediction
    prob_dt = dt.predict_proba(input_normalized)[0][1]
    class_dt = 'Fraud' if prob_dt > 0.5 else 'Not Fraud'

    # SVM Prediction
    score_svm = svm.decision_function(input_normalized)[0]
    class_svm = 'Fraud' if score_svm > 0 else 'Not Fraud'

    # Display results
    print(f"Decision Tree Prediction: {class_dt} (Probability of fraud: {prob_dt:.4f})")
    print(f"SVM Prediction: {class_svm} (Decision function score: {score_svm:.4f})")

except Exception as e:
    print(f"Prediction failed: {e}")
