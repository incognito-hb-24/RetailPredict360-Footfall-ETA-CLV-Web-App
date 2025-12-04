# prepare_data.py
import pandas as pd
import numpy as np

from config import (
    FOOTFALL_CSV, DELIVERY_CSV, CLV_CSV,
    CLEAN_FOOTFALL, CLEAN_DELIVERY, CLEAN_CLV
)

print("\n=== PHASE 2: DATA CLEANING & FEATURE ENGINEERING ===\n")

# ----------------------------------------------------
# 1️⃣ Footfall Dataset
# ----------------------------------------------------
print("Processing Footfall Dataset...")

df_foot = pd.read_csv(FOOTFALL_CSV, parse_dates=["date"])

# Time-based features
df_foot["day_of_week"] = df_foot["date"].dt.weekday
df_foot["is_weekend"] = df_foot["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
df_foot["month"] = df_foot["date"].dt.month

# Handle missing values
df_foot["is_holiday"] = df_foot["is_holiday"].fillna(0).astype(int)
df_foot["promo_active"] = df_foot["promo_active"].fillna(0).astype(int)

# Remove unrealistic footfall values
df_foot = df_foot[df_foot["footfall"] >= 30]

# Sort by date
df_foot = df_foot.sort_values("date").reset_index(drop=True)

df_foot.to_csv(CLEAN_FOOTFALL, index=False)
print(f"✔ Saved cleaned footfall data → {CLEAN_FOOTFALL}")


# ----------------------------------------------------
# 2️⃣ Delivery Dataset
# ----------------------------------------------------
print("\nProcessing Delivery Dataset...")

df_del = pd.read_csv(DELIVERY_CSV)

# Basic cleaning
for col in ["distance_km", "order_value", "delivery_time_min"]:
    if df_del[col].isna().any():
        df_del[col] = df_del[col].fillna(df_del[col].median())

# Remove unrealistic values
df_del = df_del[df_del["delivery_time_min"] >= 5]
df_del = df_del[df_del["distance_km"] > 0]

# Ensure integers where needed
df_del["traffic_level"] = df_del["traffic_level"].astype(int)
df_del["time_of_day_bucket"] = df_del["time_of_day_bucket"].astype(int)
df_del["num_items"] = df_del["num_items"].astype(int)
df_del["rider_experience_months"] = df_del["rider_experience_months"].astype(int)

df_del.to_csv(CLEAN_DELIVERY, index=False)
print(f"✔ Saved cleaned delivery data → {CLEAN_DELIVERY}")


# ----------------------------------------------------
# 3️⃣ CLV Dataset
# ----------------------------------------------------
print("\nProcessing CLV Dataset...")

df_clv = pd.read_csv(CLV_CSV)

# Fill missing numeric columns with median
for col in ["discount_usage_rate", "return_rate"]:
    if df_clv[col].isna().any():
        df_clv[col] = df_clv[col].fillna(df_clv[col].median())

# Remove impossible values
df_clv = df_clv[df_clv["avg_order_value"] > 0]
df_clv = df_clv[df_clv["orders_per_month"] > 0]

# Derived features
df_clv["loyalty_index"] = (1 - df_clv["discount_usage_rate"]) * (1 - df_clv["return_rate"])
df_clv["monetary_value"] = df_clv["orders_per_month"] * df_clv["avg_order_value"]

df_clv.to_csv(CLEAN_CLV, index=False)
print(f"✔ Saved cleaned CLV data → {CLEAN_CLV}")

print("\n✅ PHASE 2 DONE: Cleaned datasets are ready.\n")
