from secrets import choice
from typing import Callable, List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------
# Part 2: MongoDB connection
# -----------------------------
from pymongo.mongo_client import MongoClient

user = st.secrets["mongodb"]["user"]
pwd = st.secrets["mongodb"]["password"]
cluster = st.secrets["mongodb"]["cluster"]
dbname = st.secrets["mongodb"]["dbname"]

uri = f"mongodb+srv://{user}:{pwd}@{cluster}/?retryWrites=true&w=majority"
client = MongoClient(uri)
db = client[dbname]
collection = db['production_data']

# -----------------------------
# ERA5 API fetch function
# -----------------------------
import openmeteo_requests
import requests_cache
from retry_requests import retry

cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_era5_data(lat, lon, year):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m"
        ],
        "timezone": "Europe/Oslo",
        "wind_speed_unit": "ms",
        "models": "era5"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()
    data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "precipitation": hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
        "wind_gusts_10m": hourly.Variables(3).ValuesAsNumpy(),
        "wind_direction_10m": hourly.Variables(4).ValuesAsNumpy(),
    }
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("Europe/Oslo")
    df.set_index("time", inplace=True)
    return df

@st.cache_data
def load_weather_data(price_area: str, year: int = 2021) -> pd.DataFrame:
    row = df_price_areas[df_price_areas['price_area'] == price_area].iloc[0]
    lat, lon = row['latitude'], row['longitude']
    return fetch_era5_data(lat, lon, year)

# -----------------------------
# Price area representative cities
# -----------------------------
df_price_areas = pd.DataFrame([
    {"price_area": "NO1", "city": "Oslo", "latitude": 59.9127, "longitude": 10.7461},
    {"price_area": "NO2", "city": "Kristiansand", "latitude": 58.1467, "longitude": 7.9956},
    {"price_area": "NO3", "city": "Trondheim", "latitude": 63.4305, "longitude": 10.3951},
    {"price_area": "NO4", "city": "Tromsø", "latitude": 69.6489, "longitude": 18.9551},
    {"price_area": "NO5", "city": "Bergen", "latitude": 60.393, "longitude": 5.3242},
])

# -----------------------------
# Month names mapping
# -----------------------------
MONTH_NAMES = {i: pd.Timestamp(2020, i, 1).strftime("%b") for i in range(1,13)}

# -----------------------------
# Page: Home
# -----------------------------
def page_home():
    st.title("Home")
    st.write("Welcome to the app! Use the sidebar to navigate.")

# -----------------------------
# Page: Energy Production (old page 4)
# -----------------------------
def page_energy():
    st.title("Energy Production in 2021 by Price Area and Production Group")
    col1, col2 = st.columns(2)

    with col1:
        price_areas = collection.distinct("pricearea")
        selected_area = st.radio("Select Price Area", price_areas)
        st.session_state["selected_area"] = selected_area

        data = list(collection.find({"pricearea": selected_area}))
        if data:
            df_area = pd.DataFrame(data)
            df_grouped = df_area.groupby("productiongroup")["quantitykwh"].sum().reset_index()
            fig = px.pie(df_grouped, values="quantitykwh", names="productiongroup",
                         title=f"Total Production for {selected_area}")
            st.plotly_chart(fig)
        else:
            st.write("No data found for this price area.")

    with col2:
        production_groups = list(collection.distinct("productiongroup"))
        selected_groups = st.pills(
            label="Select Production Groups",
            options=production_groups,
            selection_mode="multi",
            default=production_groups
        )
        month = st.selectbox("Select Month", list(MONTH_NAMES.values()), index=0)
        month_num = [k for k,v in MONTH_NAMES.items() if v == month][0]

        query = {
            "pricearea": selected_area,
            "productiongroup": {"$in": selected_groups},
            "starttime": {
                "$gte": pd.Timestamp(2021, month_num, 1),
                "$lt": pd.Timestamp(2021, month_num+1, 1) if month_num < 12 else pd.Timestamp(2022,1,1)
            }
        }
        data_filtered = list(collection.find(query))
        if data_filtered:
            df_filtered = pd.DataFrame(data_filtered)
            df_filtered['starttime'] = pd.to_datetime(df_filtered['starttime'])
            df_grouped_time = df_filtered.groupby(['starttime','productiongroup'])['quantitykwh'].sum().reset_index()
            fig2 = px.line(df_grouped_time, x='starttime', y='quantitykwh', color='productiongroup',
                           title=f"Production in {selected_area} for {month}")
            st.plotly_chart(fig2)
        else:
            st.write("No data found for the selected filters.")

    with st.expander("Data Source"):
        st.write("The data displayed is sourced from MongoDB. The Elhub API provided hourly production data for all price areas in 2021.")

# -----------------------------
# Page newA: STL & Spectrogram (skeleton)
# -----------------------------
def page_newA():
    st.title("STL Decomposition & Spectrogram")
    st.write("To be implemented: STL decomposition & Spectrogram for selected_area weather data.")

# -----------------------------
# Page: Weather Data table
# -----------------------------
def page_data_table():
    st.title("Weather Data — January Overview")
    selected_area = st.session_state.get("selected_area", "NO1")
    df_weather = load_weather_data(selected_area)
    if df_weather.empty:
        st.warning(f"No weather data for {selected_area}")
        return

    df_jan = df_weather[df_weather.index.month == 1]
    series_rows = []
    for col in df_weather.columns:
        series_pd = df_jan[col]
        series_rows.append({
            "variable": col,
            "mean": series_pd.mean().round(1),
            "min": series_pd.min().round(1),
            "max": series_pd.max().round(1),
            "Jan": series_pd.tolist()
        })

    df_series = pd.DataFrame(series_rows).set_index("variable")
   
    column_config = {
        "mean": st.column_config.NumberColumn(label="Mean", format="%.1f", width=None),
        "min": st.column_config.NumberColumn(label="Min", format="%.1f", width=None),
        "max": st.column_config.NumberColumn(label="Max", format="%.1f", width=None),
        "Jan": st.column_config.LineChartColumn(label="January (hourly)", help="Hourly time series", width="large", y_min=None, y_max=None)
    }

    st.dataframe(df_series, column_config=column_config, use_container_width=True)

# -----------------------------
# Page: Interactive plots
# -----------------------------
def page_plots():
    st.title("Weather Plots")
    selected_area = st.session_state.get("selected_area", "NO1")
    df_weather = load_weather_data(selected_area)
    if df_weather.empty:
        st.warning(f"No weather data for {selected_area}")
        return

    choice = st.selectbox("Select variable", ["All"] + df_weather.columns.tolist(), index=0)
    month_range = st.select_slider("Select month range", options=list(range(1,13)), value=(1,1),
                                   format_func=lambda x: MONTH_NAMES[x])
    start_month, end_month = month_range

    df_sel = df_weather[(df_weather.index.month >= start_month) & (df_weather.index.month <= end_month)]
    fig, ax = plt.subplots(figsize=(12,6))
    if choice == "All":
        for col in df_sel.columns:
            ax.plot(df_sel.index, df_sel[col], label=col)
    else:
        ax.plot(df_sel.index, df_sel[choice], label=choice)

    ax.set_title(f"Weather in {selected_area} for months {MONTH_NAMES[start_month]} – {MONTH_NAMES[end_month]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# -----------------------------
# Page newB: Outlier & Anomaly (skeleton)
# -----------------------------
def page_newB():
    st.title("Outlier & Anomaly Detection")
    st.write("To be implemented: SPC and LOF analysis for selected_area weather data.")

# -----------------------------
# Navigation
# -----------------------------
pg_home = st.Page(page_home, title="Home", icon="🏠")
pg_energy = st.Page(page_energy, title="Energy Production", icon="⚙️")
pg_newA = st.Page(page_newA, title="STL & Spectrogram", icon="🌀")
pg_data = st.Page(page_data_table, title="Weather Data", icon="📋")
pg_plots = st.Page(page_plots, title="Weather Plots", icon="📈")
pg_newB = st.Page(page_newB, title="Outlier & Anomaly", icon="🚨")

nav = st.navigation(pages=[pg_home, pg_energy, pg_newA, pg_data, pg_plots, pg_newB])
nav.run()
