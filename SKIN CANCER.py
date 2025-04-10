import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1) Load CSV
df = pd.read_csv(r"C:\Users\Indransh\OneDrive\Desktop\MACHINE LEARNING\DATA\SKIN_CANCER DATASET.csv")

# 2) Select columns
cols = [
    'smoke','drink','age','gender','skin_cancer_history','cancer_history',
    'has_piped_water','has_sewage_system','fitspatrick','region',
    'diameter_1','diameter_2','diagnostic','itch','grew','hurt',
    'changed','bleed','elevation','biopsed'
]
df = df[cols]
df['biopsed'] = df['biopsed'].astype(int)

# 3) Boolean‑like → 0/1
bool_cols = [
    'smoke','drink','skin_cancer_history','cancer_history',
    'has_piped_water','has_sewage_system',
    'itch','grew','hurt','changed','bleed'
]
for col in bool_cols:
    df[col] = df[col].astype(bool).astype(int)

# 4) Elevation mapping
df['elevation'] = df['elevation'].map({'False': 0, 'True': 1, 'UNK': np.nan})

# 5) One‑hot encode categoricals
df = pd.get_dummies(
    df,
    columns=['gender','fitspatrick','region','diagnostic'],
    drop_first=True
)

# 6) Handle remaining missing values
# Option A: Drop any rows with NaN
df = df.dropna()

# Option B: Impute with column mean instead of dropping
# imputer = SimpleImputer(strategy='mean')
# df[df.columns] = imputer.fit_transform(df)

# 7) Verify no NaNs remain
print("Any NaNs left? ", df.isna().any().any())

# 8) Prepare X and y
X = df.drop('biopsed', axis=1).values
y = df['biopsed'].values

# 9) Scale features
X_norm = StandardScaler().fit_transform(X)

# 10) Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=4
)

# 11) Train logistic regression
LR = LogisticRegression().fit(X_train, y_train)

# 12) Predict & get probabilities
yhat      = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

# 13) Plot feature coefficients
feature_names = df.drop('biopsed', axis=1).columns
coefficients = pd.Series(LR.coef_[0], index=feature_names)
coefficients.sort_values().plot(kind='barh', figsize=(8,6))
plt.title("Feature Coefficients in Logistic Regression Skin Cancer Model")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()

# 14) Compute and print log‑loss
print("Log‑loss:", log_loss(y_test, yhat_prob))

# After training:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
model = LogisticRegression().fit(X_train, y_train)

# --- Prediction Input ---
# Binary and numerical inputs
sample_dict = {
    'smoke': int(input("Do you smoke? (0/1): ")),
    'drink': int(input("Do you drink? (0/1): ")),
    'age': int(input("Age: ")),
    'skin_cancer_history': int(input("Skin cancer history? (0/1): ")),
    'cancer_history': int(input("Other cancer history? (0/1): ")),
    'has_piped_water': int(input("Do you have clean drinking water? (0/1): ")),
    'has_sewage_system': int(input("do you have proper sewage system? (0/1): ")),
    'diameter_1': float(input("Diameter 1: ")),
    'diameter_2': float(input("Diameter 2: ")),
    'itch': int(input("Does the patch Itch ? (0/1): ")),
    'grew': int(input("Has the patch Grown ? (0/1): ")),
    'hurt': int(input("Does the patch Hurt ? (0/1): ")),
    'changed': int(input("Has the patch Changed in any way ? (0/1): ")),
    'bleed': int(input("Does the patch Bleed ? (0/1): ")),
    'elevation': int(input("Is the patch Elivated ? (0/1): "))
}

# Categorical inputs
gender = input("Gender (MALE/FEMALE): ").upper()
fitspatrick = int(input("Fitspatrick type (color darkness index no.) (2 to 6): "))
region = input("Region (e.g. FACE, NECK, EAR, etc.): ").upper()
diagnostic = input("Diagnostic (e.g. BCC, MEL, etc.): ").upper()

# Create base DataFrame with 0s for all columns
input_df = pd.DataFrame(columns=df.drop('biopsed', axis=1).columns)
input_df.loc[0] = 0

# Fill values
for k, v in sample_dict.items():
    input_df.at[0, k] = v

# Handle one-hot encoded categorical features
if f'gender_{gender}' in input_df.columns:
    input_df.at[0, f'gender_{gender}'] = 1

if f'fitspatrick_{fitspatrick}.0' in input_df.columns:
    input_df.at[0, f'fitspatrick_{fitspatrick}.0'] = 1

if f'region_{region}' in input_df.columns:
    input_df.at[0, f'region_{region}'] = 1

if f'diagnostic_{diagnostic}' in input_df.columns:
    input_df.at[0, f'diagnostic_{diagnostic}'] = 1

# Scale using trained scaler
input_scaled = scaler.transform(input_df)

# Predict
pred = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0]

print("\nBiopsy Prediction:", "Yes" if pred == 1 else "No")
print("Probability (No, Yes):", proba)
