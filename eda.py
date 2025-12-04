# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import CLEAN_FOOTFALL, CLEAN_DELIVERY, CLEAN_CLV, CHARTS_DIR

sns.set()

print("\n=== PHASE 3: EXPLORATORY DATA ANALYSIS (EDA) ===\n")

# ----------------------------------------------------
# 1️⃣ Footfall EDA
# ----------------------------------------------------
df_foot = pd.read_csv(CLEAN_FOOTFALL, parse_dates=["date"])

# Plot 1: Footfall over time
plt.figure(figsize=(10, 4))
plt.plot(df_foot["date"], df_foot["footfall"])
plt.title("Daily Store Footfall Over Time")
plt.xlabel("Date")
plt.ylabel("Footfall")
plt.tight_layout()
p1 = CHARTS_DIR / "footfall_time_series.png"
plt.savefig(p1)
plt.close()
print(f"✔ Saved chart → {p1}")

# Plot 2: Boxplot by day of week
plt.figure(figsize=(8, 4))
sns.boxplot(x="day_of_week", y="footfall", data=df_foot)
plt.title("Footfall by Day of Week (0=Mon, 6=Sun)")
plt.tight_layout()
p2 = CHARTS_DIR / "footfall_by_weekday.png"
plt.savefig(p2)
plt.close()
print(f"✔ Saved chart → {p2}")


# ----------------------------------------------------
# 2️⃣ Delivery EDA
# ----------------------------------------------------
df_del = pd.read_csv(CLEAN_DELIVERY)

# Scatter: distance vs delivery time
plt.figure(figsize=(8, 4))
plt.scatter(df_del["distance_km"], df_del["delivery_time_min"], alpha=0.5)
plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (min)")
plt.tight_layout()
p3 = CHARTS_DIR / "delivery_distance_vs_time.png"
plt.savefig(p3)
plt.close()
print(f"✔ Saved chart → {p3}")

# Boxplot: traffic level vs delivery time
plt.figure(figsize=(6, 4))
sns.boxplot(x="traffic_level", y="delivery_time_min", data=df_del)
plt.title("Delivery Time by Traffic Level")
plt.tight_layout()
p4 = CHARTS_DIR / "delivery_traffic_boxplot.png"
plt.savefig(p4)
plt.close()
print(f"✔ Saved chart → {p4}")


# ----------------------------------------------------
# 3️⃣ CLV EDA
# ----------------------------------------------------
df_clv = pd.read_csv(CLEAN_CLV)

# Histogram of CLV
plt.figure(figsize=(8, 4))
plt.hist(df_clv["clv_next_12m"], bins=30)
plt.title("Distribution of Customer Lifetime Value (Next 12m)")
plt.xlabel("CLV (₹)")
plt.ylabel("Count")
plt.tight_layout()
p5 = CHARTS_DIR / "clv_distribution.png"
plt.savefig(p5)
plt.close()
print(f"✔ Saved chart → {p5}")

# Correlation heatmap
plt.figure(figsize=(8, 6))
corr_cols = ["tenure_months", "orders_per_month", "avg_order_value",
             "recency_days", "discount_usage_rate", "return_rate",
             "monetary_value", "clv_next_12m"]
corr = df_clv[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation Heatmap – CLV Features")
plt.tight_layout()
p6 = CHARTS_DIR / "clv_correlation_heatmap.png"
plt.savefig(p6)
plt.close()
print(f"✔ Saved chart → {p6}")

print("\n✅ PHASE 3 DONE: Charts saved in outputs/charts.\n")
