import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb

def prepare_prophet_input(df_item):
    """Prepares item-specific data for Prophet."""
    prophet_df = df_item[['Tanggal', 'Jumlah_Terjual']].copy()
    prophet_df = prophet_df.rename(columns={'Tanggal': 'ds', 'Jumlah_Terjual': 'y'})

    if 'Ada_Promosi' in df_item.columns:
        prophet_df['promo'] = df_item['Ada_Promosi'].values

    return prophet_df

def create_time_series_features(df):
    """Creates time-based features from a datetime column."""
    df = df.copy()
    df['dayofweek'] = df['Tanggal'].dt.dayofweek
    df['dayofyear'] = df['Tanggal'].dt.dayofyear
    df['month'] = df['Tanggal'].dt.month
    df['year'] = df['Tanggal'].dt.year
    df['weekofyear'] = df['Tanggal'].dt.isocalendar().week.astype(int)
    return df

def generate_final_forecast(df_featured, best_models,
                             last_date_for_training, 
                             forecast_start_date, 
                             forecast_end_date):
    NUM_OF_FUTURE_PERIODS_TO_FORECAST = (forecast_end_date - last_date_for_training).days
    final_forecasts_list = []

    # Buat holiday dataframe jika ada
    prophet_holidays_df = pd.DataFrame()
    if not df_featured.empty and 'Hari_Libur' in df_featured.columns:
        holiday_dates = df_featured[df_featured['Hari_Libur'] == 1]['Tanggal'].unique()
        if len(holiday_dates) > 0:
            prophet_holidays_df = pd.DataFrame({
                'holiday': 'hari_libur_umum',
                'ds': pd.to_datetime(holiday_dates),
                'lower_window': 0,
                'upper_window': 0
            })

    if best_models:
        print(f"\n--- Membuat Peramalan Akhir Menggunakan Model Terbaik ---")
        for item, model_info in best_models.items():
            model_name = model_info['model_name']
            item_full_history = df_featured[df_featured['Nama_Barang'] == item].copy()
            train_full = item_full_history[item_full_history['Tanggal'] <= last_date_for_training]

            print(f"Membuat peramalan untuk '{item}' menggunakan model '{model_name}'...")
            preds = []

            if model_name == 'Prophet':
                try:
                    prophet_train_df_full = prepare_prophet_input(train_full)
                    m_prophet_final = Prophet(holidays=prophet_holidays_df)
                    if 'Ada_Promosi' in train_full.columns:
                        m_prophet_final.add_regressor('promo')
                    m_prophet_final.fit(prophet_train_df_full)

                    future_df_final = m_prophet_final.make_future_dataframe(periods=NUM_OF_FUTURE_PERIODS_TO_FORECAST, freq='D')

                    if 'Ada_Promosi' in train_full.columns:
                        promo_df = item_full_history[['Tanggal', 'Ada_Promosi']].rename(columns={'Tanggal': 'ds', 'Ada_Promosi': 'promo'})
                        future_df_final = pd.merge(future_df_final, promo_df, on='ds', how='left').fillna(0)

                    forecast_final = m_prophet_final.predict(future_df_final)
                    preds = forecast_final['yhat'].iloc[-NUM_OF_FUTURE_PERIODS_TO_FORECAST:]
                except Exception as e:
                    print(f"Gagal saat peramalan Prophet untuk {item}: {e}")

            elif model_name == 'SARIMA':
                try:
                    m_sarima_final = SARIMAX(item_full_history.set_index('Tanggal')['Jumlah_Terjual'],
                                              order=(1, 1, 1),
                                              seasonal_order=(1, 1, 0, 7)).fit(disp=False)
                    preds = m_sarima_final.predict(start=forecast_start_date, end=forecast_end_date)
                except Exception as e:
                    print(f"Gagal saat peramalan SARIMA untuk {item}: {e}")

            elif model_name == 'XGBoost':
                try:
                    features = ['dayofweek', 'dayofyear', 'month', 'year', 'weekofyear']
                    if 'Ada_Promosi' in train_full.columns:
                        features.append('Ada_Promosi')

                    X_train_full = train_full[features]
                    y_train_full = train_full['Jumlah_Terjual']
                    best_iteration = model_info['model_obj'].best_iteration

                    m_xgb_final = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=best_iteration)
                    m_xgb_final.fit(X_train_full, y_train_full, verbose=False)

                    future_dates_xgb = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')
                    future_df_xgb = pd.DataFrame({'Tanggal': future_dates_xgb})
                    future_df_xgb = create_time_series_features(future_df_xgb)

                    if 'Ada_Promosi' in train_full.columns:
                        future_df_xgb['Ada_Promosi'] = 0

                    preds = m_xgb_final.predict(future_df_xgb[features])
                except Exception as e:
                    print(f"Gagal saat peramalan XGBoost untuk {item}: {e}")

            # Gabung hasil prediksi jika ada
            if len(preds) > 0:
                forecast_df = pd.DataFrame({
                    'Tanggal': pd.date_range(start=forecast_start_date, periods=len(preds)),
                    'Nama_Barang': item,
                    'Kategori': item_full_history['Kategori'].iloc[0],
                    'Prediksi_Jumlah_Terjual': preds
                })
                final_forecasts_list.append(forecast_df)

        if final_forecasts_list:
            master_future_forecast_df = pd.concat(final_forecasts_list, ignore_index=True)
            master_future_forecast_df['Prediksi_Jumlah_Terjual'] = np.maximum(0, master_future_forecast_df['Prediksi_Jumlah_Terjual'].round().astype(int))
            print("\n--- Peramalan Akhir Berhasil Dibuat ---")
        else:
            print("\nTidak dapat membuat peramalan akhir.")
            master_future_forecast_df = pd.DataFrame()
    else:
        print("\nTidak ada model terbaik yang terseleksi, peramalan akhir dilewati.")
        master_future_forecast_df = pd.DataFrame()

    return master_future_forecast_df

import pandas as pd

def summarize_forecast_costs(raw_sales_data_df: pd.DataFrame, forecast_df: pd.DataFrame):
    """
    Membuat fitur tanggal, menentukan harga terbaru per barang, 
    menghitung total prediksi kebutuhan dan estimasi biaya,
    serta ringkasan kebutuhan per kategori.

    Input:
    - raw_sales_data_df: DataFrame berisi data penjualan asli, minimal kolom:
        'Tanggal' (datetime/string), 'Nama_Barang', 'Harga_Satuan', 'Kategori' (opsional)
    - forecast_df: DataFrame berisi prediksi dengan minimal kolom:
        'Nama_Barang', 'Prediksi_Jumlah_Terjual' (jumlah kebutuhan per periode)

    Output: dictionary berisi
    - 'total_predicted_kg' : total kebutuhan prediksi (float)
    - 'total_estimated_cost' : total estimasi biaya pembelian (float)
    - 'category_summary' : DataFrame ringkasan kebutuhan per kategori (Kategori, Prediksi_Kebutuhan_Total_kg)
    - 'latest_item_prices_df' : DataFrame harga satuan terbaru per barang
    - 'total_predicted_demand_df' : DataFrame kebutuhan total dan rata2 per barang
    """
    raw_sales_data_df['Tanggal'] = pd.to_datetime(raw_sales_data_df['Tanggal'])

    def create_time_series_features(df):
        df = df.copy()
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df['dayofweek'] = df['Tanggal'].dt.dayofweek
        df['dayofyear'] = df['Tanggal'].dt.dayofyear
        df['month'] = df['Tanggal'].dt.month
        df['year'] = df['Tanggal'].dt.year
        df['weekofyear'] = df['Tanggal'].dt.isocalendar().week.astype(int)
        return df

    if raw_sales_data_df.empty or forecast_df.empty:
        print("Data input kosong.")
        return None

    # Proses data penjualan asli
    processed_data_df = raw_sales_data_df.sort_values(by='Tanggal').reset_index(drop=True)
    processed_data_df = create_time_series_features(processed_data_df)

    # Ambil harga terbaru per barang berdasarkan tanggal terakhir
    if 'Harga_Satuan' in processed_data_df.columns:
        latest_item_prices_df = processed_data_df.loc[
            processed_data_df.groupby('Nama_Barang')['Tanggal'].idxmax()
        ][['Nama_Barang', 'Harga_Satuan', 'Kategori']].reset_index(drop=True)
    else:
        latest_item_prices_df = pd.DataFrame(columns=['Nama_Barang', 'Harga_Satuan', 'Kategori'])

    # Agregasi total kebutuhan dan rata-rata harian dari prediksi
    total_predicted_demand_df = forecast_df.groupby('Nama_Barang').agg(
        Prediksi_Kebutuhan_Total_kg=('Prediksi_Jumlah_Terjual', 'sum'),
        Prediksi_Harian_Rata2_kg=('Prediksi_Jumlah_Terjual', 'mean')
    ).reset_index()

    # Gabungkan harga dengan prediksi kebutuhan
    demand_with_price_df = pd.merge(
        total_predicted_demand_df,
        latest_item_prices_df[['Nama_Barang', 'Harga_Satuan']],
        on='Nama_Barang',
        how='left'
    )

    demand_with_price_df['Estimasi_Biaya'] = (
        demand_with_price_df['Prediksi_Kebutuhan_Total_kg'] * demand_with_price_df['Harga_Satuan']
    )
    demand_with_price_df['Harga_Satuan'] = demand_with_price_df['Harga_Satuan'].fillna(0)


    total_predicted_kg = demand_with_price_df['Prediksi_Kebutuhan_Total_kg'].sum()
    total_estimated_cost = demand_with_price_df['Estimasi_Biaya'].sum()

    # Ringkasan kebutuhan per kategori
    # Gunakan kategori dari latest_item_prices_df, gabungkan dulu
    recommendation_df = pd.merge(
        total_predicted_demand_df,
        latest_item_prices_df[['Nama_Barang', 'Kategori']],
        on='Nama_Barang',
        how='left'
    )

    category_summary = recommendation_df.groupby('Kategori')['Prediksi_Kebutuhan_Total_kg'].sum().reset_index()

    # Return semua hasil untuk bisa dipakai/diproses lagi
    return {
        'total_predicted_kg': total_predicted_kg,
        'total_estimated_cost': total_estimated_cost,
        'category_summary': category_summary,
        'latest_item_prices_df': latest_item_prices_df,
        'total_predicted_demand_df': total_predicted_demand_df
    }

import pandas as pd

def generate_purchase_recommendations_simple(forecast_df, inventory_df, price_df):
    """
    Menghasilkan rekomendasi pembelian berbasis prediksi penjualan tanpa mempertimbangkan lead time atau safety stock.

    Input:
    - forecast_df: DataFrame dengan kolom ['Nama_Barang', 'Prediksi_Jumlah_Terjual']
    - inventory_df: DataFrame dengan kolom ['Nama_Barang', 'Stok_Saat_Ini_kg']
    - price_df: DataFrame dengan kolom ['Nama_Barang', 'Harga_Satuan', 'Kategori']

    Output:
    - DataFrame dengan kolom ['Bahan', 'Kategori', 'Stok Saat Ini', 'Prediksi Kebutuhan',
                               'Rekomendasi Pembelian', 'Estimasi Harga', 'Status']
    """
    # Total & rata-rata prediksi penjualan per barang

    total_predicted = forecast_df.groupby('Nama_Barang').agg(
        Prediksi_Kebutuhan_Total_kg=('Prediksi_Jumlah_Terjual', 'sum')
    ).reset_index()

    # Gabung semua data
    df = pd.merge(total_predicted, inventory_df, on='Nama_Barang', how='left')
    df = pd.merge(df, price_df, on='Nama_Barang', how='left')

    # Hitung rekomendasi & estimasi harga
    df['Rekomendasi_Pembelian_kg'] = (
    df['Prediksi_Kebutuhan_Total_kg'] - df['Stok_Saat_Ini_kg']
).fillna(0).clip(lower=0).round().astype(int)

    df['Estimasi_Harga_Pembelian_Rp'] = (
    df['Rekomendasi_Pembelian_kg'] * df['Harga_Satuan'].fillna(0)
)


    # Tentukan status stok
    def determine_status(row):
        stok = row['Stok_Saat_Ini_kg']
        kebutuhan = row['Prediksi_Kebutuhan_Total_kg']
        if stok < 0.5 * kebutuhan:
            return "Kritis"
        elif stok < kebutuhan:
            return "Perlu"
        return "Cukup"

    df['Status'] = df.apply(determine_status, axis=1)

    # Final tampilkan
    display_df = df[[
        'Nama_Barang', 'Kategori', 'Stok_Saat_Ini_kg', 'Prediksi_Kebutuhan_Total_kg',
        'Rekomendasi_Pembelian_kg', 'Estimasi_Harga_Pembelian_Rp', 'Status'
    ]].copy()

    return display_df

import pandas as pd

def summarize_forecast_costs(raw_sales_data_df: pd.DataFrame, forecast_df: pd.DataFrame):
    """
    Membuat fitur tanggal, menentukan harga terbaru per barang, 
    menghitung total prediksi kebutuhan dan estimasi biaya,
    serta ringkasan kebutuhan per kategori.

    Input:
    - raw_sales_data_df: DataFrame berisi data penjualan asli, minimal kolom:
        'Tanggal' (datetime/string), 'Nama_Barang', 'Harga_Satuan', 'Kategori' (opsional)
    - forecast_df: DataFrame berisi prediksi dengan minimal kolom:
        'Nama_Barang', 'Prediksi_Jumlah_Terjual' (jumlah kebutuhan per periode)

    Output: dictionary berisi
    - 'total_predicted_kg' : total kebutuhan prediksi (float)
    - 'total_estimated_cost' : total estimasi biaya pembelian (float)
    - 'category_summary' : DataFrame ringkasan kebutuhan per kategori (Kategori, Prediksi_Kebutuhan_Total_kg)
    - 'latest_item_prices_df' : DataFrame harga satuan terbaru per barang
    - 'total_predicted_demand_df' : DataFrame kebutuhan total dan rata2 per barang
    """
    raw_sales_data_df['Tanggal'] = pd.to_datetime(raw_sales_data_df['Tanggal'])

    def create_time_series_features(df):
        df = df.copy()
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df['dayofweek'] = df['Tanggal'].dt.dayofweek
        df['dayofyear'] = df['Tanggal'].dt.dayofyear
        df['month'] = df['Tanggal'].dt.month
        df['year'] = df['Tanggal'].dt.year
        df['weekofyear'] = df['Tanggal'].dt.isocalendar().week.astype(int)
        return df

    if raw_sales_data_df.empty or forecast_df.empty:
        print("Data input kosong.")
        return None

    # Proses data penjualan asli
    processed_data_df = raw_sales_data_df.sort_values(by='Tanggal').reset_index(drop=True)
    processed_data_df = create_time_series_features(processed_data_df)

    # Ambil harga terbaru per barang berdasarkan tanggal terakhir
    if 'Harga_Satuan' in processed_data_df.columns:
        latest_item_prices_df = processed_data_df.loc[
            processed_data_df.groupby('Nama_Barang')['Tanggal'].idxmax()
        ][['Nama_Barang', 'Harga_Satuan', 'Kategori']].reset_index(drop=True)
    else:
        latest_item_prices_df = pd.DataFrame(columns=['Nama_Barang', 'Harga_Satuan', 'Kategori'])

    # Agregasi total kebutuhan dan rata-rata harian dari prediksi
    total_predicted_demand_df = forecast_df.groupby('Nama_Barang').agg(
        Prediksi_Kebutuhan_Total_kg=('Prediksi_Jumlah_Terjual', 'sum'),
        Prediksi_Harian_Rata2_kg=('Prediksi_Jumlah_Terjual', 'mean')
    ).reset_index()

    # Gabungkan harga dengan prediksi kebutuhan
    demand_with_price_df = pd.merge(
        total_predicted_demand_df,
        latest_item_prices_df[['Nama_Barang', 'Harga_Satuan']],
        on='Nama_Barang',
        how='left'
    )

    demand_with_price_df['Estimasi_Biaya'] = (
        demand_with_price_df['Prediksi_Kebutuhan_Total_kg'] * demand_with_price_df['Harga_Satuan']
    )

    total_predicted_kg = demand_with_price_df['Prediksi_Kebutuhan_Total_kg'].sum()
    total_estimated_cost = demand_with_price_df['Estimasi_Biaya'].sum()

    # Ringkasan kebutuhan per kategori
    # Gunakan kategori dari latest_item_prices_df, gabungkan dulu
    recommendation_df = pd.merge(
        total_predicted_demand_df,
        latest_item_prices_df[['Nama_Barang', 'Kategori']],
        on='Nama_Barang',
        how='left'
    )

    category_summary = recommendation_df.groupby('Kategori')['Prediksi_Kebutuhan_Total_kg'].sum().reset_index()

    # Return semua hasil untuk bisa dipakai/diproses lagi
    return {
        'total_predicted_kg': total_predicted_kg,
        'total_estimated_cost': total_estimated_cost,
        'category_summary': category_summary,
        'latest_item_prices_df': latest_item_prices_df,
        'total_predicted_demand_df': total_predicted_demand_df
    }

import pandas as pd

def analisis_risiko_stok(recommendation_df, threshold_overstock=1.2):
    """
    Menganalisis risiko stok berdasarkan stok saat ini dan prediksi kebutuhan.
    
    Parameter:
        recommendation_df (DataFrame): DataFrame dengan kolom 'Nama_Barang', 'Stok_Saat_Ini_kg', dan 'Prediksi_Kebutuhan_Total_kg'
        threshold_overstock (float): Ambang batas overstock (default = 1.2)
    
    Return:
        DataFrame hasil analisis risiko stok
    """
    risk_analysis_list = []

    for _, row in recommendation_df.iterrows():
        stok = row['Stok_Saat_Ini_kg']
        prediksi = row['Prediksi_Kebutuhan_Total_kg']
        balance = stok - prediksi

        if stok < prediksi:
            kekurangan = abs(balance)
            persentase = kekurangan / prediksi

            if persentase > 0.5:
                level = 'Tinggi'
            elif persentase > 0.2:
                level = 'Medium'
            else:
                level = 'Rendah'

            risk_analysis_list.append({
                'Bahan': row['Nama_Barang'],
                'Stok Saat Ini': f"{stok:.0f} kg",
                'Prediksi Kebutuhan': f"{prediksi:.0f} kg",
                'Keterangan': f"Kekurangan {kekurangan:.0f} kg",
                'Level Risiko': 'Stockout',
                'Resiko': level
            })

        elif stok > (prediksi * threshold_overstock):
            kelebihan = balance
            persentase = kelebihan / prediksi

            if persentase > 0.5:
                level = 'Tinggi'
            elif persentase > 0.2:
                level = 'Medium'
            else:
                level = 'Rendah'

            risk_analysis_list.append({
                'Bahan': row['Nama_Barang'],
                'Stok Saat Ini': f"{stok:.0f} kg",
                'Prediksi Kebutuhan': f"{prediksi:.0f} kg",
                'Keterangan': f"Kelebihan {kelebihan:.0f} kg",
                'Level Risiko': 'Overstock',
                'Resiko': level
            })

    return pd.DataFrame(risk_analysis_list)

import pandas as pd

def get_top_predicted_items(forecast_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Mengambil top N bahan berdasarkan prediksi kebutuhan total tertinggi.

    Input:
    - forecast_df: DataFrame berisi kolom 'Nama_Barang' dan 'Prediksi_Jumlah_Terjual'.
    - top_n: Jumlah item teratas yang ingin ditampilkan (default = 5).

    Output:
    - top_items_df: DataFrame berisi top N bahan dengan kolom:
        ['Nama_Barang', 'Prediksi_Kebutuhan_Total_kg']
    """

    if forecast_df.empty or 'Nama_Barang' not in forecast_df.columns or 'Prediksi_Jumlah_Terjual' not in forecast_df.columns:
        raise ValueError("DataFrame tidak valid. Harus berisi kolom 'Nama_Barang' dan 'Prediksi_Jumlah_Terjual'.")

    # Hitung total dan rata-rata prediksi per bahan
    total_predicted_demand_df = forecast_df.groupby('Nama_Barang').agg(
        Prediksi_Kebutuhan_Total_kg=('Prediksi_Jumlah_Terjual', 'sum'),
        Prediksi_Harian_Rata2_kg=('Prediksi_Jumlah_Terjual', 'mean')
    ).reset_index()

    # Ambil top N berdasarkan total kebutuhan
    top_items_df = total_predicted_demand_df.sort_values(
        by='Prediksi_Kebutuhan_Total_kg', ascending=False
    ).head(top_n)[['Nama_Barang', 'Prediksi_Kebutuhan_Total_kg']]

    return top_items_df

def summarize_forecast_costs_json(forecast_df: pd.DataFrame):
    import json

    raw_sales_data_df = pd.read_csv('dataset/pantara_demand_data.csv')
    raw_sales_data_df['Tanggal'] = pd.to_datetime(raw_sales_data_df['Tanggal'])

    def create_time_series_features(df):
        df = df.copy()
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df['dayofweek'] = df['Tanggal'].dt.dayofweek
        df['dayofyear'] = df['Tanggal'].dt.dayofyear
        df['month'] = df['Tanggal'].dt.month
        df['year'] = df['Tanggal'].dt.year
        df['weekofyear'] = df['Tanggal'].dt.isocalendar().week.astype(int)
        return df

    if raw_sales_data_df.empty or forecast_df.empty:
        print("Data input kosong.")
        return None

    # Proses data penjualan asli
    processed_data_df = raw_sales_data_df.sort_values(by='Tanggal').reset_index(drop=True)
    processed_data_df = create_time_series_features(processed_data_df)

    # Ambil harga terbaru per barang berdasarkan tanggal terakhir
    if 'Harga_Satuan' in processed_data_df.columns:
        latest_item_prices_df = processed_data_df.loc[
            processed_data_df.groupby('Nama_Barang')['Tanggal'].idxmax()
        ][['Nama_Barang', 'Harga_Satuan', 'Kategori']].reset_index(drop=True)
    else:
        latest_item_prices_df = pd.DataFrame(columns=['Nama_Barang', 'Harga_Satuan', 'Kategori'])

    # Agregasi total kebutuhan dan rata-rata harian dari prediksi
    total_predicted_demand_df = forecast_df.groupby('Nama_Barang').agg(
        Prediksi_Kebutuhan_Total_kg=('Prediksi_Jumlah_Terjual', 'sum'),
        Prediksi_Harian_Rata2_kg=('Prediksi_Jumlah_Terjual', 'mean')
    ).reset_index()

    # Gabungkan harga dengan prediksi kebutuhan
    demand_with_price_df = pd.merge(
        total_predicted_demand_df,
        latest_item_prices_df[['Nama_Barang', 'Harga_Satuan']],
        on='Nama_Barang',
        how='left'
    )

    demand_with_price_df['Estimasi_Biaya'] = (
        demand_with_price_df['Prediksi_Kebutuhan_Total_kg'] * demand_with_price_df['Harga_Satuan']
    )

    total_predicted_kg = demand_with_price_df['Prediksi_Kebutuhan_Total_kg'].sum()
    total_estimated_cost = demand_with_price_df['Estimasi_Biaya'].sum()

    # Ringkasan kebutuhan per kategori
    recommendation_df = pd.merge(
        total_predicted_demand_df,
        latest_item_prices_df[['Nama_Barang', 'Kategori']],
        on='Nama_Barang',
        how='left'
    )

    category_summary = recommendation_df.groupby('Kategori')['Prediksi_Kebutuhan_Total_kg'].sum().reset_index()

    # Convert DataFrames ke list of dict supaya json serializable
    result_json = {
        'total_predicted_kg': float(total_predicted_kg),
        'total_estimated_cost': float(total_estimated_cost),
        'category_summary': category_summary.to_dict(orient='records'),
        'latest_item_prices': latest_item_prices_df.to_dict(orient='records'),
        'total_predicted_demand': total_predicted_demand_df.to_dict(orient='records')
    }

    return result_json

def sum_forecast_next_7_days_per_barang(forecast_df: pd.DataFrame) -> list[dict]:
    """
    Menjumlahkan total prediksi penjualan untuk 7 hari ke depan per Nama_Barang.

    Args:
        forecast_df (pd.DataFrame): DataFrame dengan kolom:
            - Tanggal
            - Nama_Barang
            - Prediksi_Jumlah_Terjual

    Returns:
        List[Dict]: Total prediksi penjualan 7 hari ke depan per barang.
    """
    # Pastikan Tanggal datetime dan urutkan
    forecast_df['Tanggal'] = pd.to_datetime(forecast_df['Tanggal'])
    forecast_df = forecast_df.sort_values('Tanggal')

    # Ambil tanggal awal dan batas 7 hari
    start_date = forecast_df['Tanggal'].min()
    end_date = start_date + pd.Timedelta(days=6)

    # Filter hanya 7 hari ke depan
    filtered_df = forecast_df[
        (forecast_df['Tanggal'] >= start_date) & 
        (forecast_df['Tanggal'] <= end_date)
    ]

    # Group by Nama_Barang dan sum prediksi
    grouped = (
        filtered_df.groupby('Nama_Barang')['Prediksi_Jumlah_Terjual']
        .sum()
        .reset_index()
        .rename(columns={'Prediksi_Jumlah_Terjual': 'Total_Prediksi_7_Hari'})
    )

    # Convert to list of dict
    return grouped.to_dict(orient='records')

import pandas as pd

import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

DATA_FILE_PATH = 'dataset/pantara_demand_data.csv'
MODELS_FILE_PATH = 'best_forecasting_models.pkl'

item_price_map = {}
try:
    price_df = pd.read_csv(DATA_FILE_PATH)
    item_price_map = price_df.groupby('Nama_Barang')['Harga_Satuan'].mean().to_dict()
except FileNotFoundError:
    item_price_map = {}
except Exception as e:
    item_price_map = {}

def create_time_series_features(df):
    df = df.copy()
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['dayofweek'] = df['Tanggal'].dt.dayofweek
    df['dayofyear'] = df['Tanggal'].dt.dayofyear
    df['month'] = df['Tanggal'].dt.month
    df['year'] = df['Tanggal'].dt.year
    df['weekofyear'] = df['Tanggal'].dt.isocalendar().week.astype(int)
    return df
def predict_from_saved_models(item_to_predict, start_date, end_date, models_file_path=MODELS_FILE_PATH):
    item_category_map = {
        'Bayam': 'Sayuran',
        'Tomat': 'Sayuran',
        'Ayam': 'Protein',
        'Telur': 'Protein',
        'Wortel': 'Sayuran',
        'Jeruk': 'Buah',
        'Beras': 'Bahan Pokok'
    }

    try:
        with open(models_file_path, 'rb') as f:
            loaded_best_models = pickle.load(f)

        if item_to_predict not in loaded_best_models:
            return pd.DataFrame()

        model_info = loaded_best_models[item_to_predict]
        model_name = model_info['model_name']
        model_obj = model_info['model_obj']

        predictions = None
        if model_name == 'Prophet':
            future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            future_df = pd.DataFrame({'ds': future_dates})
            if model_obj.extra_regressors:
                for regressor in model_obj.extra_regressors:
                    future_df[regressor] = 0
            forecast = model_obj.predict(future_df)
            predictions = forecast['yhat']

        elif model_name == 'SARIMA':
            predictions = model_obj.predict(start=start_date, end=end_date)

        elif model_name == 'XGBoost':
            future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            future_df = pd.DataFrame({'Tanggal': future_dates})
            future_df = create_time_series_features(future_df)
            features = ['dayofweek', 'dayofyear', 'month', 'year', 'weekofyear']
            if 'Ada_Promosi' in model_obj.get_booster().feature_names:
                future_df['Ada_Promosi'] = 0
                features.append('Ada_Promosi')
            predictions = model_obj.predict(future_df[features])

        if predictions is not None:
            prediction_df = pd.DataFrame({
                'Date': pd.date_range(start=start_date, periods=len(predictions)),
                'Item_Name': item_to_predict,
                'Prediksi_Jumlah_Terjual': np.maximum(0, np.round(predictions)).astype(int)
            })

            prediction_df['Kategori'] = prediction_df['Item_Name'].map(item_category_map).fillna('Unknown')
            avg_price = item_price_map.get(item_to_predict, 0)
            prediction_df['Estimasi_Harga'] = (prediction_df['Prediksi_Jumlah_Terjual'] * avg_price).round().astype(int)

            return prediction_df[['Date', 'Item_Name', 'Kategori', 'Prediksi_Jumlah_Terjual', 'Estimasi_Harga']]
        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"[ERROR] Prediction for '{item_to_predict}' failed: {e}")
        return pd.DataFrame()
def generate_final_forecast(start_date: str, end_date: str, items_to_forecast=None):
    if items_to_forecast is None:
        items_to_forecast = ['Ayam', 'Bayam', 'Beras', 'Jeruk', 'Telur', 'Tomat', 'Wortel']
    
    all_predictions = []
    for item in items_to_forecast:
        df = predict_from_saved_models(item, start_date, end_date)
        if not df.empty:
            all_predictions.append(df)

    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
    else:
        return pd.DataFrame()

import pandas as pd

def generate_forecast_json_from_df(forecast_df: pd.DataFrame) -> list[dict]:
    """
    Mengubah DataFrame hasil forecast menjadi format JSON-ready
    untuk visualisasi line chart berdasarkan Nama_Barang dan Kategori per Tanggal.

    Args:
        forecast_df (pd.DataFrame): DataFrame dengan kolom minimal:
            - Tanggal
            - Nama_Barang
            - Kategori
            - Prediksi_Jumlah_Terjual

    Returns:
        List[Dict]: Data dalam format JSON-friendly (list of dict).
    """
    # Pastikan kolom Tanggal dalam format datetime
    forecast_df['Tanggal'] = pd.to_datetime(forecast_df['Tanggal'])

    # Tambahkan kolom Hari (manual mapping karena locale ID bisa error)
    hari_mapping = {
        'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
        'Thursday': 'Kamis', 'Friday': 'Jumat',
        'Saturday': 'Sabtu', 'Sunday': 'Minggu'
    }
    forecast_df['Hari'] = forecast_df['Tanggal'].dt.day_name().map(hari_mapping)

    # Gabungkan Nama_Barang dan Kategori sebagai label
    forecast_df['Label'] = forecast_df['Nama_Barang'] + ' (' + forecast_df['Kategori'] + ')'

    # Buat pivot table (line chart style)
    pivot_df = forecast_df.pivot_table(
        index='Tanggal',
        columns='Label',
        values='Prediksi_Jumlah_Terjual',
        aggfunc='sum'
    ).reset_index()

    # Konversi tanggal ke string agar bisa dikonversi ke JSON
    pivot_df['Tanggal'] = pivot_df['Tanggal'].dt.strftime('%Y-%m-%d')

    # Konversi ke list of dict (JSON-friendly)
    json_result = pivot_df.to_dict(orient='records')

    return json_result


def get_top_predicted_items_json(forecast_df: pd.DataFrame, top_n: int = 5):
    print("Kolom forecast_df:", forecast_df.columns.tolist())

    if forecast_df.empty or 'Nama_Barang' not in forecast_df.columns or 'Prediksi_Jumlah_Terjual' not in forecast_df.columns:
        raise ValueError("DataFrame tidak valid. Harus berisi kolom 'Nama_Barang' dan 'Prediksi_Jumlah_Terjual'.")

    total_predicted_demand_df = forecast_df.groupby('Nama_Barang').agg(
        Prediksi_Kebutuhan_Total_kg=('Prediksi_Jumlah_Terjual', 'sum'),
        Prediksi_Harian_Rata2_kg=('Prediksi_Jumlah_Terjual', 'mean')
    ).reset_index()

    top_items_df = total_predicted_demand_df.sort_values(
        by='Prediksi_Kebutuhan_Total_kg', ascending=False
    ).head(top_n)[['Nama_Barang', 'Prediksi_Kebutuhan_Total_kg']]

    # Konversi ke list of dict supaya bisa jadi JSON
    return top_items_df.to_dict(orient='records')
