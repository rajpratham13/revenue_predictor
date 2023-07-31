import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data from the CSV file
data = pd.read_csv("revenue_data.csv")

# Split the data into features (X) and target variable (y)
X = data["Months"].values.reshape(-1, 1)
y = data["Revenue"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Predict revenue for a new month (e.g., month 13)
new_month = 13
new_month_revenue = model.predict([[new_month]])
print(f"Predicted revenue for month {new_month}: ${new_month_revenue[0]:.2f}")
