from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)
raw_data

correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh', figsize=(10, 6))

# extract the labels from the dataframe
y = raw_data[['tip_amount']].values.astype('float32')

# drop the target variable from the feature matrix
proc_data = raw_data.drop(['tip_amount'], axis=1)

# get the feature matrix used for training
X = proc_data.values

# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# import the Decision Tree Regression Model from scikit-learn
from sklearn.tree import DecisionTreeRegressor

# for reproducible output across multiple function calls, set random_state to a given integer value
dt_reg = DecisionTreeRegressor(criterion = 'squared_error',
                               max_depth=8, 
                               random_state=35)
dt_reg.fit(X_train, y_train)
# run inference using the sklearn model
y_pred = dt_reg.predict(X_test)

# evaluate mean squared error on the test dataset
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))

r2_score = dt_reg.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))

print("Predict Tip Amount from Trip Input : ")
try:
    # Get the list of feature names (excluding target)
    feature_names = proc_data.columns.tolist()
    
    # Gather input values from the user
    user_input = []
    for feature in feature_names:
        val = float(input(f"Enter value for '{feature}': "))
        user_input.append(val)
    
    # Convert to numpy array and normalize using L1 norm
    input_array = np.array([user_input])
    input_normalized = normalize(input_array, axis=1, norm='l1')

    # Predict using trained decision tree regressor
    predicted_tip = dt_reg.predict(input_normalized)[0]
    print(f"Predicted Tip Amount: ${predicted_tip:.2f}")

except Exception as e:
    print(f"Prediction failed: {e}")
