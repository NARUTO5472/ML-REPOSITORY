import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Load the data
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
data.head()

# Distribution of target variable
sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')
plt.show()

# Standardizing continuous numerical features
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])

# Converting to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

# Combining with the original dataset
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

# Identifying categorical columns
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column

# Applying one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

# Encoding the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
prepped_data.head()

# Preparing final dataset
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Training logistic regression model using One-vs-All (default)
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# Predictions
y_pred_ova = model_ova.predict(X_test)

# Evaluation metrics for OvA
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

# Decode numeric labels back to original categories
label_map = dict(enumerate(data['NObeyesdad'].astype('category').cat.categories))

print("Predict Obesity Level by input :")
try:
    user_inputs = {}

    # Input continuous features
    for col in continuous_columns:
        val = float(input(f"Enter value for {col}: "))
        user_inputs[col] = val

    # Input categorical features
    for col in categorical_columns:
        options = data[col].unique()
        val = input(f"Enter value for {col} (options: {list(options)}): ")
        if val not in options:
            raise ValueError(f"Invalid value for {col}. Expected one of {options}")
        user_inputs[col] = val

    # Prepare continuous input
    cont_input_scaled = scaler.transform([ [user_inputs[col] for col in continuous_columns] ])
    cont_df = pd.DataFrame(cont_input_scaled, columns=scaler.get_feature_names_out(continuous_columns))

    # Prepare categorical input
    cat_input = pd.DataFrame([[user_inputs[col] for col in categorical_columns]], columns=categorical_columns)
    cat_encoded = encoder.transform(cat_input)
    cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Combine inputs
    final_input = pd.concat([cont_df, cat_df], axis=1)

    # Align input to training columns
    final_input = final_input.reindex(columns=X.columns, fill_value=0)

    # Predict
    pred_class = model_ova.predict(final_input)[0]
    pred_label = label_map[pred_class]

    print(f"Predicted Obesity Level: {pred_label}")

except Exception as e:
    print(f"Prediction failed: {e}")
