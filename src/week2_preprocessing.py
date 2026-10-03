"""
Week 2 - Data Collection, Cleaning, and Preprocessing
Logistics Data Analytics Internship

Author: Yash Patil
Role: Logistics Data Analyst Intern

Purpose:
    Demonstrate data quality checks, missing-value handling,
    duplicate detection, outlier detection, normalization,
    categorical encoding, and feature engineering.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATA
# ============================================================

# Example:
# df = pd.read_csv("logistics_data.csv")

# For demonstration, a hypothetical logistics dataset is created.
np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "region": np.random.choice(
        ["West", "North", "South", "East", "Central"],
        n
    ),
    "shipment_volume": np.random.randint(1, 50, n),
    "distance_km": np.random.uniform(10, 700, n),
    "transport_cost": np.random.uniform(500, 8000, n),
    "delivery_time_days": np.random.uniform(1, 8, n),
    "fuel_cost": np.random.uniform(100, 2500, n),
    "order_value": np.random.uniform(500, 30000, n)
})


# ============================================================
# 2. INTRODUCE MISSING VALUES FOR DEMONSTRATION
# ============================================================

df.loc[np.random.choice(df.index, 15, replace=False),
       "transport_cost"] = np.nan

df.loc[np.random.choice(df.index, 10, replace=False),
       "delivery_time_days"] = np.nan

df.loc[np.random.choice(df.index, 8, replace=False),
       "fuel_cost"] = np.nan


# ============================================================
# 3. BASIC DATA EXPLORATION
# ============================================================

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nFirst five records:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 4. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print("\nMissing values before treatment:")
print(missing_values)


# Fill numeric missing values using median
numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())


print("\nMissing values after treatment:")
print(df.isnull().sum())


# ============================================================
# 5. DUPLICATE DETECTION
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")


# ============================================================
# 6. OUTLIER DETECTION USING IQR
# ============================================================

print("\n" + "=" * 60)
print("OUTLIER DETECTION")
print("=" * 60)


def detect_outliers_iqr(data, column):
    """
    Detect outliers using the Interquartile Range (IQR).
    """

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    return outliers, lower_bound, upper_bound


for column in numeric_columns:

    outliers, lower, upper = detect_outliers_iqr(
        df,
        column
    )

    print(
        f"{column}: "
        f"{len(outliers)} outliers"
    )


# ============================================================
# 7. OUTLIER TREATMENT
# ============================================================

# Winsorization-style clipping using IQR boundaries

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower=lower_bound,
        upper=upper_bound
    )


print("\nOutlier treatment completed.")


# ============================================================
# 8. CATEGORICAL ENCODING
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL ENCODING")
print("=" * 60)

# One-hot encoding for region
df_encoded = pd.get_dummies(
    df,
    columns=["region"],
    drop_first=False
)

print("\nEncoded dataset columns:")
print(df_encoded.columns.tolist())


# ============================================================
# 9. MIN-MAX NORMALIZATION
# ============================================================

print("\n" + "=" * 60)
print("MIN-MAX NORMALIZATION")
print("=" * 60)

scaler = MinMaxScaler()

scale_columns = [
    "shipment_volume",
    "distance_km",
    "transport_cost",
    "delivery_time_days",
    "fuel_cost",
    "order_value"
]

df_encoded[scale_columns] = scaler.fit_transform(
    df_encoded[scale_columns]
)

print("\nNormalized data:")
print(df_encoded[scale_columns].head())


# ============================================================
# 10. STANDARDIZATION
# ============================================================

standard_scaler = StandardScaler()

df_standardized = df.copy()

df_standardized[scale_columns] = standard_scaler.fit_transform(
    df_standardized[scale_columns]
)

print("\nStandardized data:")
print(df_standardized[scale_columns].head())


# ============================================================
# 11. FEATURE ENGINEERING
# ============================================================

# Cost per shipment unit
df_encoded["cost_per_shipment"] = (
    df_encoded["transport_cost"] /
    df_encoded["shipment_volume"].replace(0, np.nan)
)

# Delivery efficiency
df_encoded["delivery_efficiency"] = (
    1 / df_encoded["delivery_time_days"].replace(0, np.nan)
)

# Distance category
df_encoded["distance_category"] = pd.cut(
    df["distance_km"],
    bins=[0, 100, 300, 500, np.inf],
    labels=[
        "Short",
        "Medium",
        "Long",
        "Very Long"
    ]
)


# ============================================================
# 12. FINAL DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATA QUALITY CHECK")
print("=" * 60)

print("\nFinal shape:")
print(df_encoded.shape)

print("\nRemaining missing values:")
print(df_encoded.isnull().sum())

print("\nFinal dataset preview:")
print(df_encoded.head())


# ============================================================
# 13. TRAIN-TEST SPLIT DEMONSTRATION
# ============================================================

# Example target for future predictive modeling
target = "delivery_time_days"

X = df_encoded.drop(
    columns=[target],
    errors="ignore"
)

y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print(f"Training records: {X_train.shape[0]}")
print(f"Testing records: {X_test.shape[0]}")


# ============================================================
# 14. SAVE PROCESSED DATA
# ============================================================

output_file = "processed_logistics_data.csv"

df_encoded.to_csv(
    output_file,
    index=False
)

print("\nProcessed dataset saved as:")
print(output_file)

print("\nWeek 2 preprocessing completed successfully.")
