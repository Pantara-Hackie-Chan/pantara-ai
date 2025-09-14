
# Pantara Demand Forecasting API

![Pantara Banner](./Notification.png)


This project provides a **FastAPI-based microservice** for demand forecasting, purchase recommendation, and stock risk analysis.  
It is part of the **Pantara** system, designed to help optimize inventory and procurement planning using time-series forecasting models.

---

## 🚀 Features

- **Forecasting (`/forecast`)** – Predict demand for predefined items within a date range.
- **Prediction Summary (`/prediction-summary`)** – Summarize forecast results and estimated costs.
- **Purchase Recommendation (`/purchase-recommendation`)** – Recommend purchase qty based on forecast + inventory + latest prices.
- **Stock Risk Analysis (`/stock-risk`)** – Analyze stock sufficiency/risk levels.
- **Top 5 Predicted Items (`/top-5-ingredient`)** – Highest predicted demand items.
- **Prediction Dashboard (`/prediction-dashboard`)** – Dashboard-friendly JSON (top items, by-date forecast, 7-day summary, costs).
- **Spoilage Prediction (Azure ML Endpoint)** – Separate ML service for shelf-life/spoilage scoring (see section below).
- **CORS Enabled** – Open to frontend/mobile clients.

---

## 📦 Project Structure

```

.
├── server.py                         # FastAPI app
├── libs/
│   └── modul\_demand\_forecast.py      # Forecasting & recommendation utilities
└── best\_forecasting\_models.pkl       # Pre-trained forecasting models

````

---

## 🛠️ Installation & Setup

### 1) Create & activate venv
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
````

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the API

```bash
uvicorn server:app --host 0.0.0.0 --port 8010 --reload
```

Base URL:

```
http://localhost:8010/Pantara
```

---

## 📑 API Endpoints (Quick Reference)

* `GET /` – Health check
  **Response:** `{ "message": "Welcome to Pantara API" }`

* `POST /forecast` – Generate forecasts for a date range
  **Body:** `{ "forecast_start_date": "YYYY-MM-DD", "forecast_end_date": "YYYY-MM-DD" }`

* `POST /prediction-summary` – Summarize forecast demand and costs

* `POST /purchase-recommendation` – Recommendations based on forecast + inventory

* `POST /stock-risk` – Risk analysis from recommendation rows

* `POST /top-5-ingredient` – Top 5 predicted items

* `GET /prediction-dashboard?forecast_start_date=YYYY-MM-DD&forecast_end_date=YYYY-MM-DD` – Dashboard JSON

> **Note:** Dates must be `YYYY-MM-DD`. The service assumes `best_forecasting_models.pkl` and functions in `libs/modul_demand_forecast.py` are available.

---

## 🧾 Data Models (Pydantic)

**SalesRecord**

* `Tanggal` (str), `Nama_Barang` (str), `Kategori` (str),
* `Harga_Satuan` (int), `Jumlah_Terjual` (int),
* `Ada_Promosi` (int), `Hari_Libur` (int)

**InventoryItem**

* `Nama_Barang` (str), `Stok_Saat_Ini_kg` (int)

**RecommendationRecord**

* `Nama_Barang`, `Kategori`, `Stok_Saat_Ini_kg`,
  `Prediksi_Kebutuhan_Total_kg`, `Rekomendasi_Pembelian_kg`,
  `Estimasi_Harga_Pembelian_Rp`, `Status`

---

## 🧪 Spoilage Prediction (Azure ML Endpoint)

Pantara’s spoilage prediction is served by **Azure Machine Learning Managed Online Endpoint**:

**Endpoint URL**

```
https://Pantara-spoilage.southeastasia.inference.ml.azure.com/score
```

> **Auth:** Provide the endpoint’s primary/secondary key, AMLToken, or Microsoft Entra ID token as a Bearer token.
> **Payload:** The request body format depends on the model’s deployed entry script. Adjust `data = {}` accordingly.

**Python example (as provided):**

```python
import urllib.request
import json

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data = {}

body = str.encode(json.dumps(data))

url = 'https://Pantara-spoilage.southeastasia.inference.ml.azure.com/score'
# Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
api_key = ''
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

headers = {
    'Content-Type':'application/json',
    'Accept': 'application/json',
    'Authorization':('Bearer '+ api_key)
}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)
    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))
    # Useful for debugging failures
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
```

> **Tip:** keep the spoilage service logically separate from this FastAPI app or add a thin proxy route if you want to call it from the same backend. Make sure to store the API key securely (e.g., environment variable) and **never** commit it to source control.

---

## 📊 Dependencies

* fastapi, uvicorn
* pandas, numpy
* pydantic
* pickle (Python stdlib)

---
