"""
Week 4 - Predictive Modeling and Optimization in Logistics

Author: Yash Patil
Role: Logistics Data Analyst Intern

Purpose:
    Predict logistics delivery time and demonstrate how
    predictive insights can support operational optimization.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. CREATE HYPOTHETICAL LOGISTICS DATASET
# ============================================================

np.random.seed(42)

n = 1500

regions = np.random.choice(
    ["West", "North", "South", "East", "Central"],
    size=n,
    p=[0.24, 0.21, 0.20, 0.18, 0.17]
)

distance_km = np.random.gamma(
    shape=2.4,
    scale=95,
    size=n
).clip(10, 700)

shipment_volume = np.random.poisson(
    lam=18,
    size=n
).clip(1, 55)

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


# ============================================================
# 2. SIMULATE DELIVERY TIME
# ============================================================

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


df = pd.DataFrame({
    "region": regions,
    "shipment_volume": shipment_volume,
    "distance_km": distance_km,
    "transport_cost": transport_cost,
    "fuel_cost": fuel_cost,
    "order_value": order_value,
    "warehouse_load": warehouse_load,
    "vehicle_utilization": vehicle_utilization,
    "delivery_time_days": delivery_time_days
})


# ============================================================
# 3. INTRODUCE MISSING VALUES
# ============================================================

for column, fraction in [
    ("transport_cost", 0.012),
    ("fuel_cost", 0.008),
    ("vehicle_utilization", 0.010)
]:

    count = int(n * fraction)

    indices = np.random.choice(
        df.index,
        size=count,
        replace=False
    )

    df.loc[indices, column] = np.nan


# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["delivery_time_days"]
)

y = df["delivery_time_days"]


# ============================================================
# 5. DEFINE FEATURES
# ============================================================

numeric_features = [
    "shipment_volume",
    "distance_km",
    "transport_cost",
    "fuel_cost",
    "order_value",
    "warehouse_load",
    "vehicle_utilization"
]

categorical_features = [
    "region"
]


# ============================================================
# 6. PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            SimpleImputer(
                strategy="median"
            ),
            numeric_features
        ),

        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            categorical_features
        )
    ]
)


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=8,
            min_samples_leaf=8
        ),

    "Random Forest":
        RandomForestRegressor(
            random_state=42,
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=3,
            n_jobs=-1
        )
}


# ============================================================
# 9. MODEL TRAINING AND EVALUATION
# ============================================================

results = []

fitted_models = {}


for name, model in models.items():

    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({

        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    fitted_models[name] = pipeline


results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    results_df.round(3)
)


# ============================================================
# 10. RANDOM FOREST CROSS-VALIDATION
# ============================================================

random_forest_pipeline = fitted_models[
    "Random Forest"
]

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_r2 = cross_val_score(
    random_forest_pipeline,
    X,
    y,
    cv=cv,
    scoring="r2"
)


cv_rmse = -cross_val_score(
    random_forest_pipeline,
    X,
    y,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)


print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

print(
    "Mean R2:",
    round(cv_r2.mean(), 3)
)

print(
    "R2 Standard Deviation:",
    round(cv_r2.std(), 3)
)

print(
    "Mean RMSE:",
    round(cv_rmse.mean(), 3)
)


# ============================================================
# 11. HYPERPARAMETER TUNING
# ============================================================

parameter_grid = {

    "model__n_estimators": [
        120,
        180
    ],

    "model__max_depth": [
        8,
        12,
        None
    ],

    "model__min_samples_leaf": [
        2,
        4
    ]
}


tuning_pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )
    )
])


grid_search = GridSearchCV(

    tuning_pipeline,

    parameter_grid,

    cv=3,

    scoring="neg_root_mean_squared_error",

    n_jobs=-1
)


grid_search.fit(
    X_train,
    y_train
)


print("\n" + "=" * 60)
print("BEST HYPERPARAMETERS")
print("=" * 60)

print(
    grid_search.best_params_
)


# ============================================================
# 12. TUNED MODEL EVALUATION
# ============================================================

tuned_predictions = grid_search.predict(
    X_test
)


tuned_mae = mean_absolute_error(
    y_test,
    tuned_predictions
)

tuned_rmse = mean_squared_error(
    y_test,
    tuned_predictions
) ** 0.5

tuned_r2 = r2_score(
    y_test,
    tuned_predictions
)


print("\n" + "=" * 60)
print("TUNED RANDOM FOREST PERFORMANCE")
print("=" * 60)

print(
    "MAE:",
    round(tuned_mae, 3)
)

print(
    "RMSE:",
    round(tuned_rmse, 3)
)

print(
    "R2:",
    round(tuned_r2, 3)
)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

feature_names = (
    grid_search
    .best_estimator_
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


feature_importances = (
    grid_search
    .best_estimator_
    .named_steps["model"]
    .feature_importances_
)


feature_importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": feature_importances

})


feature_importance_df = (
    feature_importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)


print("\n" + "=" * 60)
print("TOP PREDICTIVE FEATURES")
print("=" * 60)

print(
    feature_importance_df.head(10)
)


# ============================================================
# 14. OPTIMIZATION SCENARIO
# ============================================================

baseline_distance = df[
    "distance_km"
].mean()

baseline_volume = df[
    "shipment_volume"
].mean()

baseline_warehouse_load = df[
    "warehouse_load"
].mean()

baseline_vehicle_utilization = df[
    "vehicle_utilization"
].mean()


# Optimization assumptions

optimized_distance = (
    baseline_distance * 0.90
)

optimized_volume = (
    baseline_volume * 1.05
)

optimized_warehouse_load = (
    baseline_warehouse_load * 0.85
)

optimized_vehicle_utilization = min(
    0.95,
    baseline_vehicle_utilization + 0.10
)


region_mode = df[
    "region"
].mode()[0]


baseline_row = pd.DataFrame([{

    "region": region_mode,

    "shipment_volume":
        baseline_volume,

    "distance_km":
        baseline_distance,

    "transport_cost":
        df["transport_cost"].median(),

    "fuel_cost":
        df["fuel_cost"].median(),

    "order_value":
        df["order_value"].median(),

    "warehouse_load":
        baseline_warehouse_load,

    "vehicle_utilization":
        baseline_vehicle_utilization

}])


optimized_row = baseline_row.copy()


optimized_row.loc[
    0,
    "distance_km"
] = optimized_distance


optimized_row.loc[
    0,
    "shipment_volume"
] = optimized_volume


optimized_row.loc[
    0,
    "warehouse_load"
] = optimized_warehouse_load


optimized_row.loc[
    0,
    "vehicle_utilization"
] = optimized_vehicle_utilization


baseline_prediction = (
    grid_search.predict(
        baseline_row
    )[0]
)


optimized_prediction = (
    grid_search.predict(
        optimized_row
    )[0]
)


improvement = (

    (
        baseline_prediction
        - optimized_prediction
    )
    / baseline_prediction
) * 100


print("\n" + "=" * 60)
print("OPTIMIZATION SCENARIO")
print("=" * 60)

print(
    "Baseline predicted delivery time:",
    round(baseline_prediction, 2),
    "days"
)

print(
    "Optimized predicted delivery time:",
    round(optimized_prediction, 2),
    "days"
)

print(
    "Illustrative improvement:",
    round(improvement, 2),
    "%"
)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "week4_model_results.csv",
    index=False
)


feature_importance_df.to_csv(
    "week4_feature_importance.csv",
    index=False
)


print("\nResults saved successfully.")


# ============================================================
# 16. OPERATIONAL RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 60)
print("OPTIMIZATION RECOMMENDATIONS")
print("=" * 60)

recommendations = [

    "Review shorter feasible routes for high predicted delivery times.",

    "Consolidate compatible shipments to improve vehicle utilization.",

    "Balance warehouse workloads across available facilities.",

    "Create alerts for shipments exceeding delivery-time targets.",

    "Use predicted delivery time together with transportation cost when selecting feasible plans.",

    "Retrain the model periodically as new logistics data becomes available."
]


for number, recommendation in enumerate(
    recommendations,
    start=1
):

    print(
        f"{number}. {recommendation}"
    )


print("\nWeek 4 predictive modeling completed successfully.")
