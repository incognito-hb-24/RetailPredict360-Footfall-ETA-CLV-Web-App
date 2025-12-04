import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from flask import Flask, render_template, request
import numpy as np
import joblib

from config import FOOTFALL_MODEL, DELIVERY_MODEL, CLV_MODEL

app = Flask(__name__)

# Load models
footfall_model = joblib.load(FOOTFALL_MODEL)
delivery_model = joblib.load(DELIVERY_MODEL)
clv_model = joblib.load(CLV_MODEL)


# ---------------- HOME / MODEL SELECTOR ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- FOOTFALL PAGE ----------------
@app.route("/footfall", methods=["GET", "POST"])
def footfall_page():
    result = None

    if request.method == "POST":
        day_of_week = int(request.form.get("day_of_week"))
        month = int(request.form.get("month"))
        is_holiday = int(request.form.get("is_holiday"))
        promo_active = int(request.form.get("promo_active"))
        is_weekend = 1 if day_of_week >= 5 else 0

        X = np.array([[day_of_week, is_weekend, is_holiday, promo_active, month]])
        pred = footfall_model.predict(X)[0]

        if pred > 500:
            rec = "High traffic day: increase staff and run in-store promotions."
        elif pred > 300:
            rec = "Medium traffic day: normal staff, consider peak-hour offers."
        else:
            rec = "Low traffic day: optimise staffing and push digital campaigns."

        result = {
            "prediction": f"{pred:.0f}",
            "recommendation": rec,
        }

    return render_template("footfall.html", result=result)


# ---------------- DELIVERY PAGE ----------------
@app.route("/delivery", methods=["GET", "POST"])
def delivery_page():
    result = None

    if request.method == "POST":
        distance_km = float(request.form.get("distance_km"))
        num_items = int(request.form.get("num_items"))
        order_value = float(request.form.get("order_value"))
        time_of_day_bucket = int(request.form.get("time_of_day_bucket"))
        traffic_level = int(request.form.get("traffic_level"))
        rider_experience_months = int(request.form.get("rider_experience_months"))

        X = np.array([[distance_km, num_items, order_value,
                       time_of_day_bucket, traffic_level,
                       rider_experience_months]])
        pred = delivery_model.predict(X)[0]

        if pred > 45:
            rec = "High delay risk: inform customer early and avoid tight SLAs."
        elif pred > 30:
            rec = "Moderate delay risk: assign experienced rider and check route."
        else:
            rec = "Low delay risk: use this slot for express delivery promises."

        result = {
            "prediction": f"{pred:.1f}",
            "recommendation": rec,
        }

    return render_template("delivery.html", result=result)


# ---------------- CLV PAGE ----------------
@app.route("/clv", methods=["GET", "POST"])
def clv_page():
    result = None

    if request.method == "POST":
        tenure_months = int(request.form.get("tenure_months"))
        orders_per_month = float(request.form.get("orders_per_month"))
        avg_order_value = float(request.form.get("avg_order_value"))
        recency_days = int(request.form.get("recency_days"))
        discount_usage_rate = float(request.form.get("discount_usage_rate"))
        return_rate = float(request.form.get("return_rate"))

        loyalty_index = (1 - discount_usage_rate) * (1 - return_rate)
        monetary_value = orders_per_month * avg_order_value

        X = np.array([[tenure_months, orders_per_month, avg_order_value,
                       recency_days, discount_usage_rate, return_rate,
                       loyalty_index, monetary_value]])
        pred = clv_model.predict(X)[0]

        if pred > 50000:
            segment = "High Value"
            rec = "Pamper with VIP perks, loyalty rewards, and early access."
        elif pred > 20000:
            segment = "Medium Value"
            rec = "Use targeted offers and personalised nudges to grow value."
        else:
            segment = "Low Value"
            rec = "Use low-cost campaigns to increase frequency and engagement."

        result = {
            "prediction": f"{pred:,.2f}",
            "segment": segment,
            "recommendation": rec,
        }

    return render_template("clv.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
