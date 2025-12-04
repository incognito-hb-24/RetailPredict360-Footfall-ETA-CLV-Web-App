# app.py
import numpy as np
import joblib
import matplotlib.pyplot as plt

from config import (
    FOOTFALL_MODEL, DELIVERY_MODEL, CLV_MODEL,
    CHARTS_DIR
)

# Load models
footfall_model = joblib.load(FOOTFALL_MODEL)
delivery_model = joblib.load(DELIVERY_MODEL)
clv_model = joblib.load(CLV_MODEL)


# ---------------------------
# Prediction Functions
# ---------------------------
def predict_footfall():
    print("\n--- Store Footfall Prediction ---")
    day_of_week = int(input("Day of week (0=Mon ... 6=Sun): "))
    is_weekend = 1 if day_of_week >= 5 else 0
    is_holiday = int(input("Is it a holiday? (0=No, 1=Yes): "))
    promo_active = int(input("Is promotion active? (0=No, 1=Yes): "))
    month = int(input("Month (1-12): "))

    X = np.array([[day_of_week, is_weekend, is_holiday, promo_active, month]])
    pred = footfall_model.predict(X)[0]

    print(f"\n👉 Expected Footfall Today: {pred:.0f} people\n")

    # Simple AI recommendation
    if pred > 500:
        print("⚙ Recommendation: HIGH traffic day.")
        print("- Add more staff on the floor.")
        print("- Run in-store promotions and cross-sell.\n")
    elif pred > 300:
        print("⚙ Recommendation: MEDIUM traffic day.")
        print("- Keep normal staff levels.")
        print("- Consider time-bound offers during peak hours.\n")
    else:
        print("⚙ Recommendation: LOW traffic day.")
        print("- Reduce staff slightly to save cost.")
        print("- Push digital marketing to boost visits.\n")


def predict_delivery_time():
    print("\n--- Delivery Time Prediction ---")
    distance_km = float(input("Distance (km): "))
    num_items = int(input("Number of items: "))
    order_value = float(input("Order value (₹): "))
    time_of_day_bucket = int(input("Time of day (0=Morning, 1=Afternoon, 2=Night): "))
    traffic_level = int(input("Traffic level (1=Low, 2=Medium, 3=High): "))
    rider_experience_months = int(input("Rider experience (months): "))

    X = np.array([[distance_km, num_items, order_value,
                   time_of_day_bucket, traffic_level,
                   rider_experience_months]])
    pred = delivery_model.predict(X)[0]

    print(f"\n👉 Expected Delivery Time: {pred:.1f} minutes\n")

    if pred > 45:
        print("⚙ Recommendation: HIGH delay risk.")
        print("- Inform customer proactively about delay.")
        print("- Avoid committing aggressive SLAs in this time slot.\n")
    elif pred > 30:
        print("⚙ Recommendation: MODERATE delay risk.")
        print("- Assign experienced rider.")
        print("- Review route selection.\n")
    else:
        print("⚙ Recommendation: LOW delay risk.")
        print("- Use this slot for express delivery promises.\n")


def predict_clv():
    print("\n--- Customer Lifetime Value (CLV) Prediction ---")
    tenure_months = int(input("Customer tenure (months): "))
    orders_per_month = float(input("Orders per month: "))
    avg_order_value = float(input("Average order value (₹): "))
    recency_days = int(input("Days since last order: "))
    discount_usage_rate = float(input("Discount usage rate (0-1): "))
    return_rate = float(input("Return rate (0-1): "))

    loyalty_index = (1 - discount_usage_rate) * (1 - return_rate)
    monetary_value = orders_per_month * avg_order_value

    X = np.array([[tenure_months, orders_per_month, avg_order_value,
                   recency_days, discount_usage_rate, return_rate,
                   loyalty_index, monetary_value]])
    pred = clv_model.predict(X)[0]

    print(f"\n👉 Predicted 12-month CLV: ₹ {pred:,.2f}\n")

    if pred > 50000:
        segment = "HIGH VALUE"
    elif pred > 20000:
        segment = "MEDIUM VALUE"
    else:
        segment = "LOW VALUE"

    print(f"📊 Segment: {segment}")
    if segment == "HIGH VALUE":
        print("⚙ Recommendation: Offer VIP perks, loyalty rewards, and early access to sales.\n")
    elif segment == "MEDIUM VALUE":
        print("⚙ Recommendation: Nudge with targeted discounts and personalized recommendations.\n")
    else:
        print("⚙ Recommendation: Use low-cost campaigns to increase engagement and frequency.\n")


# ---------------------------
# AR-style Store Heatmap (Visual)
# ---------------------------
def generate_store_heatmap():
    """
    Simulated AR-style heatmap of store zones.
    Just a visual to make your app feel futuristic.
    """
    print("\n--- AR View: Store Heatmap (Simulation) ---")

    # 5x5 grid = 25 zones
    rows, cols = 5, 5
    # Random intensity for demo (in real life, link with footfall or sales)
    intensity = np.random.randint(10, 100, size=(rows, cols))

    plt.figure(figsize=(5, 4))
    plt.imshow(intensity, aspect="auto")
    plt.colorbar(label="Zone Intensity")
    plt.title("Simulated Store Heatmap (AR-style View)")
    plt.xlabel("Store X-axis")
    plt.ylabel("Store Y-axis")
    plt.tight_layout()

    path = CHARTS_DIR / "store_heatmap.png"
    plt.savefig(path)
    plt.show()
    plt.close()

    print(f"✔ Heatmap generated and saved to → {path}\n")
    print("Imagine this overlaid on a real store map in an AR app.\n")


# ---------------------------
# Simple Chatbot
# ---------------------------
def chatbot():
    print("\n=== RetailPredict 360 – Assistant Chatbot ===")
    print("Type 'exit' to go back to main menu.\n")

    while True:
        user = input("You: ").strip().lower()

        if user in ("exit", "quit", "back"):
            print("Bot: Going back to main menu.\n")
            break

        elif "footfall" in user:
            print("Bot: The footfall model predicts how many people will visit the store on a given day.")
            print("     It uses features like day of week, weekend, holiday, and promotions.\n")

        elif "delivery" in user or "time" in user:
            print("Bot: The delivery model predicts expected delivery time in minutes,")
            print("     using distance, items, order value, time of day, traffic, and rider experience.\n")

        elif "clv" in user or "lifetime" in user:
            print("Bot: CLV (Customer Lifetime Value) estimates how much revenue a customer will bring")
            print("     in the next 12 months. We use tenure, order frequency, order value, recency,")
            print("     discount usage, and return rate to estimate this.\n")

        elif "help" in user or "what can you do" in user:
            print("Bot: I can explain the three models (footfall, delivery, CLV),")
            print("     guide you on how to use them, and suggest which module to run.\n")

        elif "which model" in user or "use" in user:
            print("Bot: Use Footfall if you’re planning staff and promotions,")
            print("     Delivery Time if you're optimizing logistics,")
            print("     and CLV if you’re designing marketing and loyalty programs.\n")

        else:
            print("Bot: I’m not sure about that yet. Try asking about 'footfall', 'delivery', or 'CLV'.\n")


# ---------------------------
# Main Menu
# ---------------------------
def main_menu():
    while True:
        print("=" * 60)
        print("        RetailPredict 360 – AI Store Intelligence Suite")
        print("=" * 60)
        print("1. Forecast Daily Store Footfall")
        print("2. Predict Delivery Time for an Order")
        print("3. Predict Customer Lifetime Value (CLV)")
        print("4. AR-style Store Heatmap (Visualization)")
        print("5. Assistant Chatbot")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ").strip()

        if choice == "1":
            predict_footfall()
        elif choice == "2":
            predict_delivery_time()
        elif choice == "3":
            predict_clv()
        elif choice == "4":
            generate_store_heatmap()
        elif choice == "5":
            chatbot()
        elif choice == "6":
            print("\nThank you for using RetailPredict 360. Goodbye!\n")
            break
        else:
            print("\nInvalid choice. Please enter a number between 1 and 6.\n")


if __name__ == "__main__":
    main_menu()
