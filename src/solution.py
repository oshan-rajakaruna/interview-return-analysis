from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# =========================
# Project paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ecommerce_orders.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Part 1: Data Wrangling
# =========================

print("\n========== PART 1: DATA WRANGLING ==========")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# Convert order_date to datetime format
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

print("\nDate range:")
print("Minimum order date:", df["order_date"].min())
print("Maximum order date:", df["order_date"].max())

# Create net_amount column
df["net_amount"] = df["amount"] * (1 - df["discount_pct"] / 100)

print("\nSample after creating net_amount:")
print(df[["order_id", "amount", "discount_pct", "net_amount"]].head())

# Filter orders placed in the last 6 months of the dataset
max_order_date = df["order_date"].max()
cutoff_date = max_order_date - pd.DateOffset(months=6)

df_last_6_months = df[df["order_date"] >= cutoff_date].copy()

print("\nLast 6 months filter:")
print("Maximum order date:", max_order_date)
print("Cutoff date:", cutoff_date)
print("Shape before filtering:", df.shape)
print("Shape after filtering:", df_last_6_months.shape)

# Save processed dataset
processed_path = PROCESSED_DIR / "orders_last_6_months.csv"
df_last_6_months.to_csv(processed_path, index=False)

print("\nProcessed dataset saved to:", processed_path)


# =========================
# Part 2: EDA
# =========================

print("\n========== PART 2: EDA ==========")

# 1. Product category with highest return rate
category_return_rate = (
    df_last_6_months
    .groupby("product_category")["is_returned"]
    .agg(
        total_orders="count",
        returned_orders="sum",
        return_rate="mean"
    )
    .sort_values("return_rate", ascending=False)
)

print("\nReturn rate by product category:")
print(category_return_rate)

highest_return_category = category_return_rate.index[0]
highest_return_rate = category_return_rate.iloc[0]["return_rate"]

print("\nProduct category with highest return rate:", highest_return_category)
print("Highest return rate:", round(highest_return_rate * 100, 2), "%")

# Plot return rate by category
plt.figure(figsize=(8, 5))
category_return_rate["return_rate"].plot(kind="bar")
plt.title("Return Rate by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Return Rate")
plt.xticks(rotation=45)
plt.tight_layout()

category_plot_path = PLOTS_DIR / "category_return_rate.png"
plt.savefig(category_plot_path, bbox_inches="tight")
plt.close()

print("Category return rate plot saved to:", category_plot_path)


# 2. Correlation between discount_pct and is_returned
discount_return_corr = df_last_6_months["discount_pct"].corr(
    df_last_6_months["is_returned"]
)

print("\nCorrelation between discount_pct and is_returned:")
print(round(discount_return_corr, 4))

# Plot discount percentage vs return status
np.random.seed(42)

plt.figure(figsize=(8, 5))

y_jitter = df_last_6_months["is_returned"] + np.random.normal(
    0,
    0.03,
    size=len(df_last_6_months)
)

plt.scatter(
    df_last_6_months["discount_pct"],
    y_jitter,
    alpha=0.5
)

plt.title("Discount Percentage vs Return Status")
plt.xlabel("Discount Percentage")
plt.ylabel("Is Returned")
plt.yticks([0, 1], ["Not Returned", "Returned"])
plt.tight_layout()

discount_plot_path = PLOTS_DIR / "discount_vs_return.png"
plt.savefig(discount_plot_path, bbox_inches="tight")
plt.close()

print("Discount vs return plot saved to:", discount_plot_path)


# 3. Top 5 customers by total net amount spent
top_5_customers = (
    df_last_6_months
    .groupby("customer_id")["net_amount"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

print("\nTop 5 customers by total net amount spent:")
print(top_5_customers)

# Plot top 5 customers
plt.figure(figsize=(8, 5))
top_5_customers.plot(kind="bar")
plt.title("Top 5 Customers by Total Net Amount")
plt.xlabel("Customer ID")
plt.ylabel("Total Net Amount")
plt.xticks(rotation=45)
plt.tight_layout()

customer_plot_path = PLOTS_DIR / "top_5_customers.png"
plt.savefig(customer_plot_path, bbox_inches="tight")
plt.close()

print("Top 5 customers plot saved to:", customer_plot_path)


# =========================
# Part 3: Modeling
# =========================

print("\n========== PART 3: MODELING ==========")

# Check target distribution
print("\nTarget distribution:")
print(df_last_6_months["is_returned"].value_counts())

print("\nTarget distribution percentage:")
print(df_last_6_months["is_returned"].value_counts(normalize=True) * 100)

# Create modeling copy
model_df = df_last_6_months.copy()

# Create a simple date-based feature
model_df["order_month"] = model_df["order_date"].dt.month

# Select features and target
features = [
    "amount",
    "discount_pct",
    "net_amount",
    "product_category",
    "order_month"
]

target = "is_returned"

X = model_df[features]
y = model_df[target]

numeric_features = [
    "amount",
    "discount_pct",
    "net_amount",
    "order_month"
]

categorical_features = [
    "product_category"
]

# Numeric preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\nModel: Logistic Regression")
print("Accuracy:", round(accuracy, 4))
print("F1 Score:", round(f1, 4))
print("AUC:", round(auc, 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nMetric explanation:")
print(
    "F1 Score was selected because return prediction can be imbalanced. "
    "Accuracy alone can be misleading when most orders are not returned. "
    "F1 Score balances precision and recall."
)