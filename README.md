# RetailPredict 360 – AI Store Intelligence

RetailPredict 360 is a simple machine-learning suite that combines 3 business models:

- Store Footfall Forecast  
- Delivery Time Prediction  
- Customer Lifetime Value (CLV) Prediction  

The project includes both:  
a console app, and  
a modern Flask web app with 3D cards and AI/AR-style UI.

---

## ⭐ Tech Stack

Python  
Pandas, NumPy  
scikit-learn, joblib  
Flask + Bootstrap  
Matplotlib/Seaborn (optional, for EDA)

---

## ⭐ Models

- Footfall: Predicts daily visitor count  
- Delivery: Predicts delivery time in minutes  
- CLV: Predicts 12-month customer value (High/Med/Low)

---

## ⭐ HOW TO RUN THE PROJECT

### 🔹 1. Install Dependencies
Run in CMD / Anaconda Prompt:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn flask joblib
🔹 2. Prepare (Clean) the Data
This generates cleaned CSV files used for model training.

```bash
Copy code
python prepare_data.py
🔹 3. Train All Three Models
This creates the .pkl model files.

bash
Copy code
python train_models.py
After running, you will have:

Copy code
footfall_model.pkl  
delivery_model.pkl  
clv_model.pkl
🔹 4. Run the Web App
This starts the animated Flask interface.

bash
Copy code
python web_app.py
Open in your browser:

arduino
Copy code
http://xxx.x.x.x:xxxx
This gives you a text-based menu to run all 3 models.

⭐ Business Applications
Footfall: Staff optimization, inventory planning

Delivery: ETA accuracy, route planning

CLV: Customer segmentation, loyalty strategy
