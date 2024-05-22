# TODO: IMPORT ALL NEEDED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# Ensure the 'images' directory exists
os.makedirs('images', exist_ok=True)

# Read data from Excel file
df = pd.read_excel("./data/home_four_dataOne.xlsx")

# Extract GE revenue data
GE_rev = df["GE_REVENUE"].tolist()

# Extract GE earning data
earnings = df["GE_EARNINGS"].tolist()

# Extract GE dividends data
GE_dividends = df["GE_DIVIDENDS"].tolist()

# Remove the last data point
GE_rev.pop(7)

# Convert to NumPy array
revenue = np.array(GE_rev)

# Reshape quartile data into 2D
quartiles = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(quartiles, revenue, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the dependent variable values using the testing data
y_pred = model.predict(X_test)

# Logarithmic curve fitting
log_model = np.polyfit(np.log(X_train.flatten()), y_train, 1)

# Predict GE revenue using logarithmic curve
y_log_pred = log_model[1] + log_model[0] * np.log(X_test.flatten())

# Exponential curve fitting
exp_model = np.polyfit(X_train.flatten(), np.log(y_train), 1)

# Predict GE revenue using exponential curve
y_exp_pred = np.exp(exp_model[1]) * np.exp(exp_model[0] * X_test.flatten())

# Power curve fitting
power_model = np.polyfit(np.log(X_train.flatten()), np.log(y_train), 1)

# Predict GE revenue using power curve
y_power_pred = np.exp(power_model[1]) * (X_test.flatten() ** power_model[0])

# Visualize power curve fitting results
plt.scatter(X_test, y_test, color='blue', label='Actual GE Revenue')
plt.plot(X_test, y_power_pred, color='red', label='Power Curve Fit')
plt.title('Power Curve Fitting: Actual vs Predicted GE Revenue')
plt.xlabel('Quartiles')
plt.ylabel('GE Revenue')
plt.legend()
plt.savefig('images/power_curve_fit.png')
plt.show()

# Linear regression model R^2 score
linear_r2 = model.score(X_test, y_test)

# Logarithmic curve fitting R^2 score
log_r2 = r2_score(y_test, y_log_pred)

# Exponential curve fitting R^2 score
exp_r2 = r2_score(y_test, y_exp_pred)

# Power curve fitting R^2 score
power_r2 = r2_score(y_test, y_power_pred)

print("Linear Regression R^2 score:", linear_r2)
print("Logarithmic Curve Fitting R^2 score:", log_r2)
print("Exponential Curve Fitting R^2 score:", exp_r2)
print("Power Curve Fitting R^2 score:", power_r2)

print("log: ", y_log_pred)
print("exponential: ", y_exp_pred)
print("power: ", y_power_pred)
print("linear: ", y_pred)
