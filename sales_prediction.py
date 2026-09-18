# ============================================================
# TASK 4 - SALES PREDICTION USING PYTHON
# Machine Learning Project using Linear Regression
# ============================================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Step 2: Load the dataset
# ============================================================

# The CSV file must be in the same folder as this Python file
data = pd.read_csv("advertising.csv.csv")

print("\n================ DATASET LOADED SUCCESSFULLY ================\n")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(data.head())


# ============================================================
# Step 3: Understand the dataset
# ============================================================

print("\n================ DATASET INFORMATION ================\n")

print("Number of rows and columns:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nDataset information:")
print(data.info())

print("\nStatistical summary:")
print(data.describe())


# ============================================================
# Step 4: Check for missing values
# ============================================================

print("\n================ MISSING VALUES ================\n")

print(data.isnull().sum())


# ============================================================
# Step 5: Define input and output
# ============================================================

# Input features
X = data[["TV", "Radio", "Newspaper"]]

# Target variable
y = data["Sales"]

print("\n================ FEATURES AND TARGET ================\n")

print("Input features:")
print(X.head())

print("\nTarget variable:")
print(y.head())


# ============================================================
# Step 6: Split data into training and testing data
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n================ TRAIN TEST SPLIT ================\n")

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# ============================================================
# Step 7: Create Linear Regression model
# ============================================================

model = LinearRegression()


# ============================================================
# Step 8: Train the model
# ============================================================

model.fit(X_train, y_train)

print("\n================ MODEL TRAINING COMPLETE ================\n")

print("Linear Regression model trained successfully!")


# ============================================================
# Step 9: Make predictions
# ============================================================

y_pred = model.predict(X_test)

print("\n================ PREDICTIONS ================\n")

print("Actual Sales:")
print(y_test.values[:10])

print("\nPredicted Sales:")
print(y_pred[:10])


# ============================================================
# Step 10: Evaluate the model
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n================ MODEL EVALUATION ================\n")

print("Mean Absolute Error (MAE):", round(mae, 4))
print("Mean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))
print("R2 Score:", round(r2, 4))


# ============================================================
# Step 11: Display model coefficients
# ============================================================

print("\n================ MODEL COEFFICIENTS ================\n")

print("Intercept:", round(model.intercept_, 4))

print("TV coefficient:", round(model.coef_[0], 4))
print("Radio coefficient:", round(model.coef_[1], 4))
print("Newspaper coefficient:", round(model.coef_[2], 4))


# ============================================================
# Step 12: Compare actual and predicted values
# ============================================================

comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\n================ ACTUAL VS PREDICTED ================\n")

print(comparison.head(10))


# ============================================================
# Step 13: Visualize Actual vs Predicted Sales
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs Predicted Sales")

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.tight_layout()
plt.show()


# ============================================================
# Step 14: Visualize Sales Distribution
# ============================================================

plt.figure(figsize=(8, 6))

sns.histplot(data["Sales"], kde=True)

plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Sales Distribution")

plt.tight_layout()
plt.show()


# ============================================================
# Step 15: Correlation Heatmap
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# Step 16: Predict Sales for a new advertising budget
# ============================================================

# Example:
# TV = 150
# Radio = 30
# Newspaper = 20

new_advertising_data = pd.DataFrame({
    "TV": [150],
    "Radio": [30],
    "Newspaper": [20]
})

new_prediction = model.predict(new_advertising_data)

print("\n================ NEW SALES PREDICTION ================\n")

print("TV Advertising:", 150)
print("Radio Advertising:", 30)
print("Newspaper Advertising:", 20)

print(
    "Predicted Sales:",
    round(new_prediction[0], 2)
)

print("\n========================================================")
print("       SALES PREDICTION PROJECT COMPLETED")
print("========================================================")