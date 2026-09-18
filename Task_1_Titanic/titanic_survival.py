# ============================================================
# TITANIC SURVIVAL PREDICTION
# ============================================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

# Make sure Titanic-Dataset.csv is in the same folder as this file
df = pd.read_csv(r"C:\Users\msdmo\Downloads\Titanic-Dataset.csv")

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION")
print("=" * 60)

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Display dataset shape
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# ============================================================
# 2. BASIC DATA ANALYSIS
# ============================================================

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

# ============================================================
# 3. VISUALIZE SURVIVAL
# ============================================================

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Survived")
plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# Survival by gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()

# Survival by passenger class
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

# ============================================================
# 4. SELECT FEATURES
# ============================================================

# Target variable
y = df["Survived"]

# Features used for prediction
X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]
]

print("\nFeatures used for prediction:")
print(X.columns.tolist())

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================
# 6. PREPROCESSING
# ============================================================

# Numerical columns
numerical_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

# Categorical columns
categorical_features = [
    "Sex",
    "Embarked"
]

# Numerical preprocessing
numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# ============================================================
# 7. CREATE MACHINE LEARNING MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=8
)

# Create complete pipeline
pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)

# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nTraining the model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")

# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)

# ============================================================
# 10. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Did Not Survive", "Survived"]
)

disp.plot()
plt.title("Titanic Survival Prediction - Confusion Matrix")
plt.show()

# ============================================================
# 13. TEST MODEL WITH A NEW PASSENGER
# ============================================================

# Example passenger
new_passenger = pd.DataFrame(
    {
        "Pclass": [3],
        "Sex": ["male"],
        "Age": [25],
        "SibSp": [0],
        "Parch": [0],
        "Fare": [10.5],
        "Embarked": ["S"]
    }
)

# Predict
prediction = pipeline.predict(new_passenger)

# Get probability
probability = pipeline.predict_proba(new_passenger)

print("\n" + "=" * 60)
print("NEW PASSENGER PREDICTION")
print("=" * 60)

if prediction[0] == 1:
    print("\nPrediction: SURVIVED")
else:
    print("\nPrediction: DID NOT SURVIVE")

print(
    f"Probability of not surviving: "
    f"{probability[0][0] * 100:.2f}%"
)

print(
    f"Probability of surviving: "
    f"{probability[0][1] * 100:.2f}%"
)

# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

# Get trained Random Forest model
trained_model = pipeline.named_steps["model"]

# Get transformed feature names
feature_names = (
    pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

# Feature importance
importance = trained_model.feature_importances_

feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()

# ============================================================
# 15. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
