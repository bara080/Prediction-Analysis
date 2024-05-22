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

# Extract MSFT revenue data
MSFT_rev = df["MSFT_REVENUE"].tolist()

# Extract MSFT earning data
earnings = df["MSFT_EARNINGS"].tolist()

# Extract MSFT dividends data
MSFT_dividends = df["MSFT_DIVIDENDS"].tolist()

# Remove the last data point
MSFT_rev.pop(7)

# Convert to NumPy array
revenue = np.array(MSFT_rev)

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

# Predict MSFT revenue using logarithmic curve
y_log_pred = log_model[1] + log_model[0] * np.log(X_test.flatten())

# Exponential curve fitting
exp_model = np.polyfit(X_train.flatten(), np.log(y_train), 1)

# Predict MSFT revenue using exponential curve
y_exp_pred = np.exp(exp_model[1]) * np.exp(exp_model[0] * X_test.flatten())

# Power curve fitting
power_model = np.polyfit(np.log(X_train.flatten()), np.log(y_train), 1)

# Predict MSFT revenue using power curve
y_power_pred = np.exp(power_model[1]) * (X_test.flatten() ** power_model[0])

# Visualize power curve fitting results
plt.scatter(X_test, y_test, color='blue', label='Actual MSFT Revenue')
plt.plot(X_test, y_power_pred, color='red', label='Power Curve Fit')
plt.title('Power Curve Fitting: Actual vs Predicted MSFT Revenue')
plt.xlabel('Quartiles')
plt.ylabel('MSFT Revenue')
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
