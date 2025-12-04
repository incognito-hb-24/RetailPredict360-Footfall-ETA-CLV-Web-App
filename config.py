# config.py
from pathlib import Path

# Base directory = folder where this file is saved
BASE_DIR = Path(__file__).resolve().parent

# Core folders
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Create folders if missing (portable across PCs)
for d in [DATA_DIR, MODELS_DIR, TEMPLATES_DIR, OUTPUTS_DIR, CHARTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Raw data files (from Mockaroo / my CSVs)
FOOTFALL_CSV = DATA_DIR / "footfall_data.csv"
DELIVERY_CSV = DATA_DIR / "delivery_data.csv"
CLV_CSV = DATA_DIR / "clv_data.csv"

# Cleaned data outputs
CLEAN_FOOTFALL = DATA_DIR / "clean_footfall.csv"
CLEAN_DELIVERY = DATA_DIR / "clean_delivery.csv"
CLEAN_CLV = DATA_DIR / "clean_clv.csv"

# Model files
FOOTFALL_MODEL = MODELS_DIR / "footfall_model.pkl"
DELIVERY_MODEL = MODELS_DIR / "delivery_model.pkl"
CLV_MODEL = MODELS_DIR / "clv_model.pkl"
