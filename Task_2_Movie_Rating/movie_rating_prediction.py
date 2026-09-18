# ============================================================
# TASK 2 - MOVIE RATING PREDICTION WITH PYTHON
# IMDb Movies India Dataset
# ============================================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("MOVIE RATING PREDICTION")
print("=" * 70)

# Load the IMDb dataset
# Load the IMDb dataset
df = pd.read_csv(
    "IMDb_Data/IMDb Movies India.csv",
    encoding="latin1"
)

print("\nDataset loaded successfully!")


# ============================================================
# 2. VIEW DATASET
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 3. CHECK DATASET INFORMATION
# ============================================================

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. DATA CLEANING
# ============================================================

print("\nCleaning the dataset...")


# Convert Year to numeric
df["Year"] = df["Year"].str.extract(r"(\d{4})")

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)


# Convert Duration to numeric
df["Duration"] = df["Duration"].str.extract(r"(\d+)")

df["Duration"] = pd.to_numeric(
    df["Duration"],
    errors="coerce"
)


# Convert Votes to numeric
df["Votes"] = pd.to_numeric(
    df["Votes"],
    errors="coerce"
)


# Convert Rating to numeric
df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)


# ============================================================
# 5. SELECT IMPORTANT COLUMNS
# ============================================================

# We will use these features for prediction
features = [
    "Year",
    "Duration",
    "Votes",
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3"
]

target = "Rating"


# Keep only required columns
data = df[
    features + [target]
].copy()


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")
print(data.isnull().sum())


# Fill categorical missing values
categorical_columns = [
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3"
]

for column in categorical_columns:
    data[column] = data[column].fillna("Unknown")


# Fill numeric missing values
numeric_columns = [
    "Year",
    "Duration",
    "Votes"
]

for column in numeric_columns:
    data[column] = data[column].fillna(
        data[column].median()
    )


# Remove rows where Rating is missing
data = data.dropna(
    subset=["Rating"]
)


print("\nDataset after cleaning:")
print(data.shape)


# ============================================================
# 7. CREATE NUMERIC FEATURES FROM CATEGORICAL DATA
# ============================================================

# Convert Genre into a simple numeric feature
data["Genre_Count"] = data["Genre"].apply(
    lambda x: len(str(x).split(","))
)


# Convert Director into frequency
director_frequency = data["Director"].value_counts()

data["Director_Frequency"] = data["Director"].map(
    director_frequency
)


# Convert Actor 1 into frequency
actor1_frequency = data["Actor 1"].value_counts()

data["Actor1_Frequency"] = data["Actor 1"].map(
    actor1_frequency
)


# Convert Actor 2 into frequency
actor2_frequency = data["Actor 2"].value_counts()

data["Actor2_Frequency"] = data["Actor 2"].map(
    actor2_frequency
)


# Convert Actor 3 into frequency
actor3_frequency = data["Actor 3"].value_counts()

data["Actor3_Frequency"] = data["Actor 3"].map(
    actor3_frequency
)


# ============================================================
# 8. SELECT FINAL FEATURES
# ============================================================

X = data[
    [
        "Year",
        "Duration",
        "Votes",
        "Genre_Count",
        "Director_Frequency",
        "Actor1_Frequency",
        "Actor2_Frequency",
        "Actor3_Frequency"
    ]
]

y = data["Rating"]


print("\nFeatures used for prediction:")
print(X.head())

print("\nTarget variable:")
print(y.head())


# ============================================================
# 9. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 10. CREATE MACHINE LEARNING MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining the model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================================
# 12. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print(
    f"\nMean Absolute Error (MAE): {mae:.3f}"
)

print(
    f"Mean Squared Error (MSE): {mse:.3f}"
)

print(
    f"Root Mean Squared Error (RMSE): {rmse:.3f}"
)

print(
    f"R² Score: {r2:.3f}"
)


# ============================================================
# 14. ACTUAL VS PREDICTED RATINGS
# ============================================================

results = pd.DataFrame(
    {
        "Actual Rating": y_test.values,
        "Predicted Rating": y_pred
    }
)

print("\nActual vs Predicted ratings:")
print(results.head(10))


# ============================================================
# 15. VISUALIZATION - RATING DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    data["Rating"],
    bins=20,
    kde=True
)

plt.title(
    "Movie Rating Distribution"
)

plt.xlabel(
    "Rating"
)

plt.ylabel(
    "Number of Movies"
)

plt.tight_layout()

plt.show()


# ============================================================
# 16. VISUALIZATION - ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel(
    "Actual Rating"
)

plt.ylabel(
    "Predicted Rating"
)

plt.title(
    "Actual vs Predicted Movie Ratings"
)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()

plt.show()


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    feature_importance
)


# Plot feature importance
plt.figure(
    figsize=(9, 6)
)

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title(
    "Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.show()


# ============================================================
# 18. PREDICT RATING FOR A NEW MOVIE
# ============================================================

print("\n" + "=" * 70)
print("NEW MOVIE RATING PREDICTION")
print("=" * 70)


# Example new movie information
new_movie = pd.DataFrame(
    {
        "Year": [2025],
        "Duration": [120],
        "Votes": [1000],
        "Genre_Count": [2],
        "Director_Frequency": [5],
        "Actor1_Frequency": [10],
        "Actor2_Frequency": [8],
        "Actor3_Frequency": [6]
    }
)


predicted_rating = model.predict(
    new_movie
)


print(
    f"\nPredicted Movie Rating: {predicted_rating[0]:.2f}"
)


# ============================================================
# 19. PROJECT COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("MOVIE RATING PREDICTION PROJECT COMPLETED!")
print("=" * 70)
