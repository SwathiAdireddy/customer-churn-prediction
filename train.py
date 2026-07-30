import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("Original shape:", df.shape)


# -----------------------------------
# Data Cleaning
# -----------------------------------

# Remove customerID
df.drop("customerID", axis=1, inplace=True)


# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# Remove missing values
df.dropna(inplace=True)


# Encode target variable
df["Churn"] = df["Churn"].map(
    {
        "No": 0,
        "Yes": 1
    }
)


# -----------------------------------
# Separate Features and Target
# -----------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]


# -----------------------------------
# Train Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------
# Define Feature Types
# -----------------------------------

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


categorical_features = [
    "SeniorCitizen",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# -----------------------------------
# Preprocessing
# -----------------------------------

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            StandardScaler(),
            numeric_features
        ),

        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# -----------------------------------
# Create Pipeline
# -----------------------------------

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            LogisticRegression(
                random_state=42
            )
        )

    ]
)


# -----------------------------------
# Train Model
# -----------------------------------

pipeline.fit(
    X_train,
    y_train
)


# -----------------------------------
# Evaluation
# -----------------------------------

y_pred = pipeline.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\nAccuracy:", accuracy)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# -----------------------------------
# Save Pipeline
# -----------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    pipeline,
    "models/churn_pipeline.pkl"
)


print("\nPipeline saved successfully!")
