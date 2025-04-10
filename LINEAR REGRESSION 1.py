import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = pd.read_csv(url)
df.sample(5)
df.describe()

cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
cdf.sample(9)
cdf.hist()
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("EMISSIONS")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("ENGINESIZE")
plt.ylabel("EMISSIONS")
plt.show()

x = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
regressor = linear_model.LinearRegression()
regressor.fit(x_train.reshape(-1,1),y_train)
print("Coefficient : ", regressor.coef_)
print("Intercept : ", regressor.intercept_)

plt.scatter(x_train,y_train,color = "blue")
plt.plot(x_train,regressor.coef_*x_train + regressor.intercept_, '-r')
plt.xlabel("Engine Size")
plt.ylabel("Emissions")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
y_test_ = regressor.predict(x_test.reshape(-1,1))
print("Mean absolute error : ", mean_absolute_error(y_test,y_test_))
print("Mean squared error : ", mean_squared_error(y_test,y_test_))
print("Root mean squared error : ", root_mean_squared_error(y_test,y_test_))
print("R2 score : ", r2_score(y_test,y_test_))