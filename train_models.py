import sys
from pathlib import Path

# Ensure local imports work
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # <- third, different algo

from config import (
    CLEAN_FOOTFALL, CLEAN_DELIVERY, CLEAN_CLV,
    FOOTFALL_MODEL, DELIVERY_MODEL, CLV_MODEL
)

print("\n=== PHASE 4: MODEL TRAINING (3 different algorithms) ===\n")

# ----------------------------------------------------
# 1️⃣ Footfall Model – Linear Regression
# ----------------------------------------------------
print(">> Training Footfall model (Linear Regression)...")

df_foot = pd.read_csv(CLEAN_FOOTFALL)

X_foot = df_foot[["day_of_week", "is_weekend", "is_holiday", "promo_active", "month"]]
y_foot = df_foot["footfall"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_foot, y_foot, test_size=0.2, random_state=42
)

foot_model = LinearRegression()
foot_model.fit(X_train_f, y_train_f)

y_pred_f = foot_model.predict(X_test_f)
mae_f = mean_absolute_error(y_test_f, y_pred_f)
r2_f = r2_score(y_test_f, y_pred_f)

print(f"   Footfall MAE: {mae_f:.2f}")
print(f"   Footfall R²:  {r2_f:.3f}")

joblib.dump(foot_model, FOOTFALL_MODEL)
print(f"   ✔ Saved Linear Regression model → {FOOTFALL_MODEL}\n")


# ----------------------------------------------------
# 2️⃣ Delivery Time Model – Random Forest
# ----------------------------------------------------
print(">> Training Delivery Time model (Random Forest Regressor)...")

df_del = pd.read_csv(CLEAN_DELIVERY)

X_del = df_del[
    [
        "distance_km",
        "num_items",
        "order_value",
        "time_of_day_bucket",
        "traffic_level",
        "rider_experience_months",
    ]
]
y_del = df_del["delivery_time_min"]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_del, y_del, test_size=0.2, random_state=42
)

del_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)
del_model.fit(X_train_d, y_train_d)

y_pred_d = del_model.predict(X_test_d)
mae_d = mean_absolute_error(y_test_d, y_pred_d)
r2_d = r2_score(y_test_d, y_pred_d)

print(f"   Delivery MAE: {mae_d:.2f} minutes")
print(f"   Delivery R²:  {r2_d:.3f}")

joblib.dump(del_model, DELIVERY_MODEL)
print(f"   ✔ Saved Random Forest model → {DELIVERY_MODEL}\n")


# ----------------------------------------------------
# 3️⃣ CLV Model – XGBoost Regressor
# ----------------------------------------------------
print(">> Training CLV model (XGBoost Regressor)...")

df_clv = pd.read_csv(CLEAN_CLV)

X_clv = df_clv[
    [
        "tenure_months",
        "orders_per_month",
        "avg_order_value",
        "recency_days",
        "discount_usage_rate",
        "return_rate",
        "loyalty_index",
        "monetary_value",
    ]
]
y_clv = df_clv["clv_next_12m"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clv, y_clv, test_size=0.2, random_state=42
)

clv_model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42,
)
clv_model.fit(X_train_c, y_train_c)

y_pred_c = clv_model.predict(X_test_c)
mae_c = mean_absolute_error(y_test_c, y_pred_c)
r2_c = r2_score(y_test_c, y_pred_c)

print(f"   CLV MAE: ₹{mae_c:,.2f}")
print(f"   CLV R²:  {r2_c:.3f}")

joblib.dump(clv_model, CLV_MODEL)
print(f"   ✔ Saved XGBoost model → {CLV_MODEL}\n")


print("✅ PHASE 4 DONE: All three models trained with different algorithms.\n")
