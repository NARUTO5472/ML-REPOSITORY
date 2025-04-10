import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# load the data

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)

churn_df

#Data preprocessing

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

X_norm[0:5]

y = np.asarray(churn_df['churn'])

X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)

LR = LogisticRegression().fit(X_train,y_train)

yhat = LR.predict(X_test)
yhat[:10]

yhat_prob = LR.predict_proba(X_test)
yhat_prob[:10]

coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

loss = log_loss(y_test, yhat_prob)
print(f"Log-loss on test set: {loss:.4f}")

print("Churn Prediction for a New Customer:")
try:
    input_features = []
    feature_names = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
    for feature in feature_names:
        val = float(input(f"Enter {feature}: "))
        input_features.append(val)

    input_array = np.array([input_features])
    input_scaled = scaler.transform(input_array)

    prediction = LR.predict(input_scaled)[0]
    prediction_prob = LR.predict_proba(input_scaled)[0]

    print(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")
    print(f"Probability of Churn: {prediction_prob[1]:.2%}")
    print(f"Probability of Staying: {prediction_prob[0]:.2%}")

except Exception as e:
    print("Error during prediction:", e)