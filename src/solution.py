from pathlib import Path

import pandas as pd


# =========================
# File paths
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "ecommerce_orders.csv"


# =========================
# Part 1: Load data
# =========================

df = pd.read_csv(DATA_PATH)


# =========================
# Basic data inspection
# =========================

print("========== BASIC DATA REPORT ==========")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())