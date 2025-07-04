import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("Housing.csv")
print("First 5 rows of the dataset:")
print(df.head())

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Split the dataset into features and target
X = df.drop("price", axis=1)
y = df["price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Optional: Plotting (only works well for single feature like 'area')
if 'area' in X.columns:
    plt.scatter(X_test['area'], y_test, color='blue', label='Actual')
    plt.plot(X_test['area'], y_pred, color='red', label='Predicted')
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Linear Regression: Area vs Price")
    plt.legend()
    plt.show()

# Show coefficients
print("\n--- Coefficients ---")
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print(coeff_df)
