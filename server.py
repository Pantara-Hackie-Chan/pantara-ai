from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List
import pandas as pd
from datetime import datetime
import pickle
from fastapi import Query
from fastapi.responses import JSONResponse
import numpy as np
from libs.modul_demand_forecast import (
    generate_final_forecast,
    summarize_forecast_costs,
    generate_purchase_recommendations_simple,
    analisis_risiko_stok,
    get_top_predicted_items,
    get_top_predicted_items_json,
    generate_forecast_json_from_df,
    sum_forecast_next_7_days_per_barang,
    summarize_forecast_costs_json,
    generate_final_forecast
)

# Load model terbaik
with open('best_forecasting_models.pkl', 'rb') as f:
    best_models = pickle.load(f)

# FastAPI app
app = FastAPI(
    title="Pantara Demand Forecasting API",
    root_path="/Pantara"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Welcome to Pantara API"}


# === Pydantic Schemas ===

class SalesRecord(BaseModel):
    Tanggal: str
    Nama_Barang: str
    Kategori: str
    Harga_Satuan: int
    Jumlah_Terjual: int
    Ada_Promosi: int
    Hari_Libur: int

class InventoryItem(BaseModel):
    Nama_Barang: str
    Stok_Saat_Ini_kg: int

class ForecastRequest(BaseModel):
    forecast_start_date: str
    forecast_end_date: str

class PurchaseRecommendationRequest(ForecastRequest):
    data_inventory_sekarang: List[InventoryItem]

class RecommendationRecord(BaseModel):
    Nama_Barang: str
    Kategori: str
    Stok_Saat_Ini_kg: float
    Prediksi_Kebutuhan_Total_kg: int
    Rekomendasi_Pembelian_kg: int
    Estimasi_Harga_Pembelian_Rp: int
    Status: str

class ForecastRecord(BaseModel):
    Tanggal: str
    Nama_Barang: str
    Kategori: str
    Prediksi_Jumlah_Terjual: int
    
class PurchaseRecommendationSimpleRequest(BaseModel):
    data_raw: List[SalesRecord]
    data_inventory_sekarang: List[InventoryItem]


# === Utility ===
def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['dayofweek'] = df['Tanggal'].dt.dayofweek
    df['dayofyear'] = df['Tanggal'].dt.dayofyear
    df['month'] = df['Tanggal'].dt.month
    df['year'] = df['Tanggal'].dt.year
    df['weekofyear'] = df['Tanggal'].dt.isocalendar().week.astype(int)
    return df

def get_forecast_df(request: ForecastRequest) -> pd.DataFrame:
    df_raw = pd.DataFrame([r.dict() for r in request.data_raw])
    df_raw['Tanggal'] = pd.to_datetime(df_raw['Tanggal'])

    if df_raw.empty:
        raise HTTPException(status_code=400, detail="Data kosong untuk forecasting.")

    df_sorted = df_raw.sort_values(by='Tanggal').reset_index(drop=True)
    df_featured = create_time_series_features(df_sorted)

    last_date = pd.to_datetime(request.last_date_for_training)
    start_date = pd.to_datetime(request.forecast_start_date)
    end_date = pd.to_datetime(request.forecast_end_date)

    df_forecast = generate_final_forecast(
        df_featured=df_featured,
        best_models=best_models,
        last_date_for_training=last_date,
        forecast_start_date=start_date,
        forecast_end_date=end_date
    )

    if df_forecast.empty:
        raise HTTPException(status_code=400, detail="Gagal membuat prediksi.")

    return df_forecast

# === Endpoint: /forecast ===
class ForecastRequest(BaseModel):
    forecast_start_date: str
    forecast_end_date: str

# List item yang akan diprediksi
items_to_forecast = ['Ayam', 'Bayam', 'Beras', 'Jeruk', 'Telur', 'Tomat', 'Wortel']

@app.post("/forecast")
def forecast_endpoint(request: ForecastRequest):
    try:
        start = pd.to_datetime(request.forecast_start_date).strftime('%Y-%m-%d')
        end = pd.to_datetime(request.forecast_end_date).strftime('%Y-%m-%d')

        if start > end:
            raise HTTPException(status_code=400, detail="Tanggal mulai harus lebih awal dari tanggal akhir.")

        # ⬇️ Ganti semua loop dengan fungsi langsung
        result_df = generate_final_forecast(start_date=start, end_date=end, items_to_forecast=items_to_forecast)

        if result_df.empty:
            raise HTTPException(status_code=404, detail="Tidak ada prediksi yang berhasil.")

        # Format ke JSON
        result_json = result_df.rename(columns={
            'Date': 'Tanggal',
            'Item_Name': 'Nama_Barang'
        }).to_dict(orient='records')

        return {
            "forecast_start_date": start,
            "forecast_end_date": end,
            "total_items": len(items_to_forecast),
            "predictions": result_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi: {e}")


# === Endpoint: /prediction-summary ===
@app.post("/prediction-summary")
def prediction_summary(request: ForecastRequest):
    df_forecast = get_forecast_df(request)
    df_raw = pd.DataFrame([r.dict() for r in request.data_raw])
    df_raw['Tanggal'] = pd.to_datetime(df_raw['Tanggal'])

    summary = summarize_forecast_costs(df_raw, df_forecast)

    return {
        "total_predicted_kg": round(float(summary['total_predicted_kg']), 2),
        "total_estimated_cost": round(float(summary['total_estimated_cost']), 2),
        "category_summary": summary['category_summary'].to_dict(orient="records"),
        "latest_item_prices": summary['latest_item_prices_df'].to_dict(orient="records"),
        "total_predicted_demand": summary['total_predicted_demand_df'].to_dict(orient="records")
    }

# === Endpoint: /purchase-recommendation ===
@app.post("/purchase-recommendation")
def purchase_recommendation(request: PurchaseRecommendationRequest):
    df_forecast = get_forecast_df(request)

    inventory_df = pd.DataFrame([i.dict() for i in request.data_inventory_sekarang])
    price_df = pd.DataFrame([r.dict() for r in request.data_raw])
    price_df['Tanggal'] = pd.to_datetime(price_df['Tanggal'])

    price_latest = price_df.loc[
        price_df.groupby('Nama_Barang')['Tanggal'].idxmax()
    ][['Nama_Barang', 'Harga_Satuan', 'Kategori']].reset_index(drop=True)

    rec_df = generate_purchase_recommendations_simple(
        forecast_df=df_forecast,
        inventory_df=inventory_df,
        price_df=price_latest
    )

    return {"recommendations": rec_df.to_dict(orient="records")}

# === Endpoint: /stock-risk ===
@app.post("/stock-risk")
def stock_risk(recommendations: List[RecommendationRecord] = Body(...)):
    df_rec = pd.DataFrame([r.dict() for r in recommendations])
    risk_df = analisis_risiko_stok(df_rec)
    return {"stock_risk_analysis": risk_df.to_dict(orient="records")}

# === Endpoint: /top-5-ingredient ===
@app.post("/top-5-ingredient")
def top_5_ingredient(request: ForecastRequest):
    df_forecast = get_forecast_df(request)
    top5 = get_top_predicted_items(df_forecast, top_n=5)
    return {"top_5_predicted_items": top5.to_dict(orient="records")}

# === Endpoint: /prediction-dashboard ===
@app.get("/prediction-dashboard")
def prediction_dashboard(
    forecast_start_date: str = Query(...),
    forecast_end_date: str = Query(...)
):
    try:
        start_date = pd.to_datetime(forecast_start_date).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(forecast_end_date).strftime('%Y-%m-%d')

        if start_date > end_date:
            raise HTTPException(status_code=400, detail="Tanggal mulai harus lebih awal dari tanggal akhir.")

        # Dapatkan dataframe langsung dari fungsi generate_final_forecast
        forecast_df = generate_final_forecast(start_date=start_date, end_date=end_date, items_to_forecast=items_to_forecast)

        if forecast_df.empty:
            raise HTTPException(status_code=404, detail="Tidak ada data prediksi untuk rentang tanggal tersebut.")

        # Rename kolom supaya konsisten
        forecast_df = forecast_df.rename(columns={'Date': 'Tanggal', 'Item_Name': 'Nama_Barang'})

        # Pastikan Tanggal bertipe datetime
        forecast_df['Tanggal'] = pd.to_datetime(forecast_df['Tanggal'])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memuat data prediksi: {e}")

    try:
        top_5_items = get_top_predicted_items_json(forecast_df, top_n=5)
        pred_by_tanggal = generate_forecast_json_from_df(forecast_df)
        pred_7_day = sum_forecast_next_7_days_per_barang(forecast_df)
        summary = summarize_forecast_costs_json(forecast_df)

        return {
            "top_5_predicted_items": top_5_items,
            "forecast_by_date": pred_by_tanggal,
            "forecast_next_7_days_summary": pred_7_day,
            "prediction_summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghasilkan dashboard: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8010, reload=True)
