"""
Week 3 - Advanced Data Analysis and Visualization
Logistics Data Analytics Internship

Author: Yash Patil
Role: Logistics Data Analyst Intern

Purpose:
    Perform Exploratory Data Analysis (EDA), descriptive statistics,
    correlation analysis, and logistics data visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CREATE HYPOTHETICAL LOGISTICS DATASET
# ============================================================

np.random.seed(42)

n = 1200

regions = np.random.choice(
    ["West", "North", "South", "East", "Central"],
    size=n
)

shipment_volume = np.random.poisson(
    lam=18,
    size=n
).clip(1, 55)

distance_km = np.random.gamma(
    shape=2.4,
    scale=95,
    size=n
).clip(10, 700)

transport_cost = (
    450
    + distance_km * 7.2
    + shipment_volume * 38
    + np.random.normal(0, 280, n)
).clip(350, 8000)

fuel_cost = (
    transport_cost
    * np.random.uniform(0.18, 0.34, n)
).clip(80, 2400)

order_value = np.random.lognormal(
    mean=8.6,
    sigma=0.55,
    size=n
).clip(800, 30000)

warehouse_load = np.random.uniform(
    0.25,
    0.98,
    n
)

vehicle_utilization = (
    0.48
    + shipment_volume / 80
    + np.random.normal(0, 0.08, n)
).clip(0.35, 1.0)

region_effect = pd.Series(regions).map({
    "West": 0.05,
    "North": 0.25,
    "South": 0.10,
    "East": 0.35,
    "Central": 0.18
}).to_numpy()

delivery_time_days = (
    1.0
    + distance_km / 190
    + shipment_volume * 0.018
    + transport_cost / 12500
    + warehouse_load * 0.9
    + (1 - vehicle_utilization) * 0.35
    + region_effect
    + np.random.normal(0, 0.32, n)
).clip(0.7, 8.5)

delay_days = np.maximum(
    0,
    delivery_time_days - 3.5
)

on_time = (
    delivery_time_days <= 3.5
).astype(int)

customer_rating = (
    5
    - delay_days * 0.45
    + np.random.normal(0, 0.35, n)
).clip(1, 5)


df = pd.DataFrame({
    "region": regions,
    "shipment_volume": shipment_volume,
    "distance_km": distance_km,
    "transport_cost": transport_cost,
    "delivery_time_days": delivery_time_days,
    "delay_days": delay_days,
    "on_time": on_time,
    "fuel_cost": fuel_cost,
    "order_value": order_value,
    "customer_rating": customer_rating
})


# ============================================================
# 2. INTRODUCE MISSING VALUES
# ============================================================

for column, fraction in [
    ("transport_cost", 0.012),
    ("delivery_time_days", 0.010),
    ("customer_rating", 0.010)
]:

    missing_count = int(n * fraction)

    indices = np.random.choice(
        df.index,
        size=missing_count,
        replace=False
    )

    df.loc[indices, column] = np.nan


# ============================================================
# 3. BASIC DATA EXPLORATION
# ============================================================

print("=" * 60)
print("LOGISTICS DATASET OVERVIEW")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nFirst five records:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. MISSING VALUE TREATMENT
# ============================================================

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


print("\nMissing values after treatment:")
print(df.isnull().sum())


# ============================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe())


# ============================================================
# 6. CENTRAL TENDENCY
# ============================================================

analysis_columns = [
    "shipment_volume",
    "distance_km",
    "transport_cost",
    "delivery_time_days",
    "delay_days",
    "fuel_cost",
    "order_value",
    "customer_rating"
]

central_tendency = pd.DataFrame({
    "Mean": df[analysis_columns].mean(),
    "Median": df[analysis_columns].median(),
    "Standard Deviation": df[analysis_columns].std()
})

print("\nCentral tendency:")
print(central_tendency)


# ============================================================
# 7. REGION-WISE ANALYSIS
# ============================================================

regional_summary = df.groupby("region").agg({
    "shipment_volume": "mean",
    "delivery_time_days": "mean",
    "transport_cost": "mean",
    "delay_days": "mean",
    "customer_rating": "mean"
}).round(2)

print("\nRegional performance:")
print(regional_summary)


# ============================================================
# 8. CORRELATION ANALYSIS
# ============================================================

correlation_columns = [
    "shipment_volume",
    "distance_km",
    "transport_cost",
    "delivery_time_days",
    "delay_days",
    "fuel_cost",
    "order_value",
    "customer_rating"
]

correlation_matrix = df[
    correlation_columns
].corr()

print("\nCorrelation matrix:")
print(correlation_matrix.round(2))


# ============================================================
# 9. VISUALIZATION 1
# SHIPMENT VOLUME BY REGION
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="region",
    y="shipment_volume",
    estimator=np.mean
)

plt.title("Average Shipment Volume by Region")
plt.xlabel("Region")
plt.ylabel("Average Shipment Volume")

plt.tight_layout()

plt.savefig(
    "shipment_volume_by_region.png",
    dpi=200
)

plt.show()


# ============================================================
# 10. VISUALIZATION 2
# DELIVERY TIME DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="delivery_time_days",
    bins=30,
    kde=True
)

plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Shipments")

plt.tight_layout()

plt.savefig(
    "delivery_time_distribution.png",
    dpi=200
)

plt.show()


# ============================================================
# 11. VISUALIZATION 3
# TRANSPORTATION COST DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="transport_cost",
    bins=30,
    kde=True
)

plt.title("Transportation Cost Distribution")
plt.xlabel("Transportation Cost")
plt.ylabel("Number of Shipments")

plt.tight_layout()

plt.savefig(
    "transportation_cost_distribution.png",
    dpi=200
)

plt.show()


# ============================================================
# 12. VISUALIZATION 4
# DISTANCE VS DELIVERY TIME
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="distance_km",
    y="delivery_time_days",
    alpha=0.5
)

plt.title(
    "Distance vs Delivery Time"
)

plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (Days)")

plt.tight_layout()

plt.savefig(
    "distance_vs_delivery_time.png",
    dpi=200
)

plt.show()


# ============================================================
# 13. VISUALIZATION 5
# SHIPMENT VOLUME VS TRANSPORT COST
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="shipment_volume",
    y="transport_cost",
    alpha=0.5
)

plt.title(
    "Shipment Volume vs Transportation Cost"
)

plt.xlabel("Shipment Volume")
plt.ylabel("Transportation Cost")

plt.tight_layout()

plt.savefig(
    "shipment_volume_vs_transport_cost.png",
    dpi=200
)

plt.show()


# ============================================================
# 14. VISUALIZATION 6
# DELIVERY TIME BY REGION
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="region",
    y="delivery_time_days"
)

plt.title(
    "Delivery Time Distribution by Region"
)

plt.xlabel("Region")
plt.ylabel("Delivery Time (Days)")

plt.tight_layout()

plt.savefig(
    "delivery_time_by_region.png",
    dpi=200
)

plt.show()


# ============================================================
# 15. VISUALIZATION 7
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Logistics Variable Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    "logistics_correlation_heatmap.png",
    dpi=200
)

plt.show()


# ============================================================
# 16. VISUALIZATION 8
# ON-TIME DELIVERY RATE BY REGION
# ============================================================

on_time_rate = (
    df.groupby("region")["on_time"]
    .mean()
    * 100
)

plt.figure(figsize=(8, 5))

on_time_rate.plot(
    kind="bar"
)

plt.title(
    "On-Time Delivery Rate by Region"
)

plt.xlabel("Region")
plt.ylabel("On-Time Delivery Rate (%)")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "on_time_delivery_by_region.png",
    dpi=200
)

plt.show()


# ============================================================
# 17. IDENTIFY KEY LOGISTICS INSIGHTS
# ============================================================

highest_cost_region = (
    regional_summary["transport_cost"]
    .idxmax()
)

highest_delivery_region = (
    regional_summary["delivery_time_days"]
    .idxmax()
)

lowest_rating_region = (
    regional_summary["customer_rating"]
    .idxmin()
)


print("\n" + "=" * 60)
print("KEY LOGISTICS INSIGHTS")
print("=" * 60)

print(
    f"Region with highest average transportation cost: "
    f"{highest_cost_region}"
)

print(
    f"Region with highest average delivery time: "
    f"{highest_delivery_region}"
)

print(
    f"Region with lowest average customer rating: "
    f"{lowest_rating_region}"
)


# ============================================================
# 18. SAVE ANALYSIS DATA
# ============================================================

df.to_csv(
    "week3_logistics_eda_dataset.csv",
    index=False
)

regional_summary.to_csv(
    "regional_logistics_summary.csv"
)

correlation_matrix.to_csv(
    "logistics_correlation_matrix.csv"
)


print("\nEDA analysis completed successfully.")
