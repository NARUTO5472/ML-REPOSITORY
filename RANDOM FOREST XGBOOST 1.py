import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Load the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
n_estimators = 100
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)

# Fit Random Forest
start_time_rf = time.time()
rf.fit(X_train, y_train)
end_time_rf = time.time()
rf_train_time = end_time_rf - start_time_rf

# Predict using Random Forest
y_pred_rf = rf.predict(X_test)

# Fit XGBoost
start_time_xgb = time.time()
xgb.fit(X_train, y_train)
end_time_xgb = time.time()
xgb_train_time = end_time_xgb - start_time_xgb

# Predict using XGBoost
y_pred_xgb = xgb.predict(X_test)

# Standard deviation of target for ±1 std dev lines
std_y = np.std(y_test)

# Evaluation (optional but helpful to print)
print(f"Random Forest - R²: {r2_score(y_test, y_pred_rf):.3f}, MSE: {mean_squared_error(y_test, y_pred_rf):.3f}, Train time: {rf_train_time:.2f}s")
print(f"XGBoost       - R²: {r2_score(y_test, y_pred_xgb):.3f}, MSE: {mean_squared_error(y_test, y_pred_xgb):.3f}, Train time: {xgb_train_time:.2f}s")

# Plotting
plt.figure(figsize=(14, 6))

# Random Forest
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.5, color="blue", edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Perfect Prediction")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1)
plt.ylim(0, 6)
plt.title("Random Forest: Predictions vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()

# XGBoost
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color="orange", edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Perfect Prediction")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1)
plt.ylim(0, 6)
plt.title("XGBoost: Predictions vs Actual")
plt.xlabel("Actual Values")
plt.legend()

plt.tight_layout()
plt.show()

# Predict with user input
print("Predict California House Value Based on Input")

try:
    feature_names = data.feature_names
    user_values = []

    for feature in feature_names:
        val = float(input(f"Enter value for '{feature}': "))
        user_values.append(val)

    user_input = np.array([user_values])  # 2D array for prediction

    # Make predictions
    rf_pred = rf.predict(user_input)[0]
    xgb_pred = xgb.predict(user_input)[0]

    print(f"\n🔍 Predicted Median House Value (in $100,000s):")
    print(f" - Random Forest: ${rf_pred:.2f}00")
    print(f" - XGBoost      : ${xgb_pred:.2f}00")

except Exception as e:
    print(f"Prediction failed: {e}")
