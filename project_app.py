from secrets import choice
from typing import Callable, List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
#from analysis import spc_outlier_plotly, lof_precipitation_plotly, matplotlib_spectrogram, stl_decomposition_plotly_subplots
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

# Collections
production_collection = db["production_data"]
consumption_collection = db["consumption_data"]

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
# Functions for outlier detection and plotting. Copied from notebook CA3.ipynb
# -----------------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.neighbors import LocalOutlierFactor



# --- Cell: SPC outlier detection (DCT SATV) + Plotly plotting ---
import numpy as np
import pandas as pd
from scipy.fftpack import dct, idct
import plotly.graph_objects as go

# --- DCT-based seasonal decomposition and SATV calculation ---
def dct_seasonal_and_satv(series: pd.Series, cutoff_frac: float = 0.05):
    """Return seasonal component (low-frequency DCT reconstruction) and SATV = series - seasonal."""
    x = series.values.astype(float)
    n = len(x)
    X = dct(x, norm='ortho')
    keep = int(np.floor(cutoff_frac * n))
    if keep < 1:
        keep = 1
    seasonal_coeffs = np.zeros_like(X)
    seasonal_coeffs[:keep] = X[:keep]
    seasonal = idct(seasonal_coeffs, norm='ortho')
    satv = x - seasonal
    return pd.Series(seasonal, index=series.index), pd.Series(satv, index=series.index)

# Median Absolute Deviation
def mad(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med))

# SPC outlier detection and Plotly plotting function
def spc_outlier_plotly(temp_series: pd.Series, cutoff_frac: float = 0.05, k: float = 3.0, title: str = None):
    """
    temp_series: pandas Series with datetime index and temperature values (°C).
    cutoff_frac: fraction of lowest DCT frequencies to KEEP as seasonal (default 0.05; 0.04–0.08 are sensible range).
    k: number of robust standard deviations to use for SPC boundaries (default 3).
    Returns: (plotly_fig, summary_dict)
    """
    if not isinstance(temp_series.index, pd.DatetimeIndex):
        temp_series = temp_series.copy()
        temp_series.index = pd.to_datetime(temp_series.index)

    temp_series = temp_series.sort_index()
    seasonal, satv = dct_seasonal_and_satv(temp_series, cutoff_frac=cutoff_frac)

    # robust stats on SATV
    med = float(np.median(satv.values))
    mad_val = float(mad(satv.values))
    sigma = float(1.4826 * mad_val) if mad_val > 0 else float(np.std(satv.values))

    # SATV thresholds (constant values) and convert to original scale by adding seasonal component
    lower_satv = med - k * sigma
    upper_satv = med + k * sigma
    lower_curve = seasonal + lower_satv
    upper_curve = seasonal + upper_satv

    outlier_mask = (satv < lower_satv) | (satv > upper_satv)
    n_points = len(temp_series)
    n_outliers = int(outlier_mask.sum())
    outlier_fraction = n_outliers / n_points if n_points > 0 else 0.0

    # Build Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temp_series.index, y=temp_series.values, mode='lines', name='Temperature'))
    fig.add_trace(go.Scatter(x=temp_series.index, y=upper_curve.values, mode='lines', name=f'Upper SPC (k={k})', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=temp_series.index, y=lower_curve.values, mode='lines', name=f'Lower SPC (k={k})', line=dict(dash='dash')))
    if n_outliers > 0:
        fig.add_trace(go.Scatter(x=temp_series.index[outlier_mask], y=temp_series.values[outlier_mask],
                                 mode='markers', name='Outliers', marker=dict(size=6, symbol='x')))

    fig.update_layout(
        title=title or 'Temperature and SPC outliers (DCT-based SATV)',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        template='plotly_white',
        height=520,
        width=1100
    )

    summary = {
        'n_points': n_points, # total number of data points
        'n_outliers': n_outliers, # number of detected outliers
        'outlier_fraction': outlier_fraction, # fraction of outliers
        'median_satv': med, # median of SATV
        'mad_satv': mad_val, # MAD of SATV
        'sigma_est': sigma, # robust std dev estimate of SATV
        'cutoff_frac': float(cutoff_frac), # DCT cutoff fraction
        'k': float(k), # SPC k parameter
    }

    return fig, summary



# --- Cell: LOF precipitation anomaly detection + Plotly plotting ---
def lof_precipitation_plotly(precip_series: pd.Series, contamination: float = 0.01, n_neighbors: int = 20, title: str = None):
    """
    Detect precipitation anomalies using the Local Outlier Factor (LOF) method and visualize with Plotly.

    Parameters
    ----------
    precip_series : pandas.Series
        Time series of precipitation (mm/hour or mm/day) with datetime index.
    contamination : float, optional
        Proportion of outliers (default 0.01 = 1%).
    n_neighbors : int, optional
        Number of neighbors to use for LOF (default 20).
    title : str, optional
        Custom plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure of precipitation with anomalies marked.
    summary : dict
        Summary with counts, proportions, and example outlier timestamps.
    """

    # Ensure datetime index
    if not isinstance(precip_series.index, pd.DatetimeIndex):
        precip_series = precip_series.copy()
        precip_series.index = pd.to_datetime(precip_series.index)
    precip_series = precip_series.sort_index()

    # Prepare data for LOF (2D input required)
    X = precip_series.values.reshape(-1, 1)

    # Fit LOF model
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    preds = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_

    # Identify outliers
    outlier_mask = preds == -1
    n_points = len(precip_series)
    n_outliers = int(outlier_mask.sum())
    outlier_fraction = n_outliers / n_points if n_points > 0 else 0.0

    # Build Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=precip_series.index, y=precip_series.values,
        mode='lines', name='Precipitation'
    ))

    if n_outliers > 0:
        fig.add_trace(go.Scatter(
            x=precip_series.index[outlier_mask],
            y=precip_series.values[outlier_mask],
            mode='markers',
            name='Anomalies (LOF)',
            marker=dict(color='red', size=6, symbol='x')
        ))

    fig.update_layout(
        title=title or f'Precipitation anomalies (LOF, contamination={contamination*100:.1f}%)',
        xaxis_title='Time',
        yaxis_title='Precipitation',
        hovermode='x unified',
        template='plotly_white',
        height=520,
        width=1100
    )

    summary = {
        'n_points': n_points,
        'n_outliers': n_outliers,
        'outlier_fraction': outlier_fraction,
        'contamination_param': contamination,
        'n_neighbors': n_neighbors,
        'lof_score_min': float(lof_scores.min()),
        'lof_score_max': float(lof_scores.max()) # max LOF score. LOF scores indicate the degree of outlierness; higher scores mean more anomalous.
        }

    return fig, summary



import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL

def stl_decomposition_plotly_subplots(
    df,
    price_area='NO1',
    production_group='hydro',
    period=24,
    seasonal=13,
    trend=25,
    robust=True
):
    """
    STL decomposition with four stacked subplots (Original, Trend, Seasonal, Residual) using Plotly.

    Returns the figure and STL results object.
    """
    # Case-insensitive filtering
    ts = df[(df['priceArea'].str.lower() == price_area.lower()) &
            (df['productionGroup'].str.lower() == production_group.lower())]['quantityKwh']
    
    if ts.empty:
        raise ValueError(f"No data found for price area '{price_area}' and production group '{production_group}'")
    
    # Fill missing values
    ts = ts.asfreq('h').ffill()
    
    # Fit STL
    stl = STL(ts, period=period, seasonal=seasonal, trend=trend, robust=robust)
    res = stl.fit()
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual']
    )
    
    # Original
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Original', line=dict(color='blue')), row=1, col=1)
    # Trend
    fig.add_trace(go.Scatter(x=ts.index, y=res.trend, mode='lines', name='Trend', line=dict(color='orange')), row=2, col=1)
    # Seasonal
    fig.add_trace(go.Scatter(x=ts.index, y=res.seasonal, mode='lines', name='Seasonal', line=dict(color='green')), row=3, col=1)
    # Residual
    fig.add_trace(go.Scatter(x=ts.index, y=res.resid, mode='lines', name='Residual', line=dict(color='red')), row=4, col=1)
    
    fig.update_layout(
        height=900,
        title_text=f"STL Decomposition: {production_group} in {price_area}",
        template='plotly_white'
    )
    
    fig.show()
    
    return fig, res




import numpy as np
import pandas as pd
from scipy.signal import stft
import plotly.graph_objects as go

def plotly_spectrogram(
    df,
    price_area='NO1',
    production_group='hydro',
    window_length=168,   # nperseg in STFT
    window_overlap=84,   # noverlap in STFT
    fs=1                 # sampling frequency (1 per hour)
):
    """
    Compute and plot a spectrogram for electricity production data using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ['priceArea','productionGroup','quantityKwh'] indexed by datetime.
    price_area : str
        Electricity price area to filter.
    production_group : str
        Production group to filter.
    window_length : int
        Number of samples per STFT window (nperseg).
    window_overlap : int
        Overlap between windows (noverlap).
    fs : float
        Sampling frequency. For hourly data, fs=1.

    Returns
    -------
    f : np.ndarray
        Frequency bins.
    t : np.ndarray
        Time bins.
    Zxx : np.ndarray
        STFT complex values.
    """
    # --- Filter data ---
    ts = df[
        (df['priceArea'].str.lower() == price_area.lower()) &
        (df['productionGroup'].str.lower() == production_group.lower())
    ]['quantityKwh']

    if ts.empty:
        raise ValueError(f"No data for price area '{price_area}' and production group '{production_group}'")

    ts = ts.asfreq('h').ffill()

    # --- Compute STFT ---
    f, t, Zxx = stft(ts.values, fs=fs, nperseg=window_length, noverlap=window_overlap)
    amplitude = np.abs(Zxx)

    # --- Create interactive spectrogram ---
    fig = go.Figure(
        data=go.Heatmap(
            z=amplitude,
            x=t,
            y=f,
            colorscale='Viridis',
            colorbar=dict(title='Amplitude'),
        )
    )

    fig.update_layout(
        title=f"Spectrogram: {production_group.capitalize()} in {price_area.upper()}",
        xaxis_title="Time [hours]",
        yaxis_title="Frequency [1/hour]",
        template="plotly_dark",
        height=500,
        width=1000,
    )

    fig.show()

    return f, t, Zxx




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

    # --- Fixed color mapping for production groups ---
    color_map = {
        "hydro": "#1f77b4",     # Dark Blue
        "wind": "#8ac9f3",      # Light Blue
        "solar": "#f25454",     # red
        "thermal": "#fead6b",   # Orange
        "other": "#2ecc71"      # Green
    }

    with col1:
        price_areas = production_collection.distinct("pricearea")
        selected_area = st.radio("Select Price Area", price_areas)
        st.session_state["selected_area"] = selected_area

        data = list(production_collection.find({"pricearea": selected_area}))
        if data:
            df_area = pd.DataFrame(data)
            df_grouped = df_area.groupby("productiongroup")["quantitykwh"].sum().reset_index()
            fig = px.pie(
                df_grouped,
                values="quantitykwh",
                names="productiongroup",
                title=f"Total Production for {selected_area}",
                color="productiongroup",
                color_discrete_map=color_map  # 👈 fixed consistent colors
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No data found for this price area.")

    with col2:
        production_groups = list(production_collection.distinct("productiongroup"))
        selected_groups = st.pills(
            label="Select Production Groups",
            options=production_groups,
            selection_mode="multi",
            default=production_groups
        )

        month = st.selectbox("Select Month", list(MONTH_NAMES.values()), index=0)
        month_num = [k for k, v in MONTH_NAMES.items() if v == month][0]

        query = {
            "pricearea": selected_area,
            "productiongroup": {"$in": selected_groups},
            "starttime": {
                "$gte": pd.Timestamp(2021, month_num, 1),
                "$lt": pd.Timestamp(2021, month_num + 1, 1) if month_num < 12 else pd.Timestamp(2022, 1, 1)
            }
        }
        data_filtered = list(production_collection.find(query))
        if data_filtered:
            df_filtered = pd.DataFrame(data_filtered)
            df_filtered["starttime"] = pd.to_datetime(df_filtered["starttime"])
            df_grouped_time = (
                df_filtered.groupby(["starttime", "productiongroup"])["quantitykwh"]
                .sum()
                .reset_index()
            )

            fig2 = px.line(
                df_grouped_time,
                x="starttime",
                y="quantitykwh",
                color="productiongroup",
                title=f"Production in {selected_area} for {month}",
                color_discrete_map=color_map  # 👈 fixed consistent colors
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("No data found for the selected filters.")

    with st.expander("Data Source"):
        st.write(
            "The data displayed is sourced from MongoDB. "
            "The Elhub API provided hourly production data for all price areas in 2021."
        )



# -----------------------------
# Page newA: STL & Spectrogram (skeleton)
# -----------------------------
def page_newA():
    st.title("STL Decomposition & Spectrogram Analysis (Elhub Production Data)")

    # --- Reuse the selected area from session_state (from Energy page) ---
    selected_area = st.session_state.get("selected_area", "NO1")

    # Get available production groups from MongoDB
    production_groups = production_collection.distinct("productiongroup")
    selected_group = st.selectbox("Select Production Group", production_groups, index=0)

    # --- Load Elhub data from MongoDB ---
    data = list(production_collection.find({"pricearea": selected_area, "productiongroup": selected_group}))
    if not data:
        st.warning(f"No production data found for {selected_area} ({selected_group}).")
        return

    df_prod = pd.DataFrame(data)
    df_prod["starttime"] = pd.to_datetime(df_prod["starttime"])
    df_prod = df_prod.sort_values("starttime").set_index("starttime")

    # 🔧 FIX: Handle duplicate timestamps by summing production
    if df_prod.index.duplicated().any():
        duplicates = df_prod.index.duplicated().sum()
        # st.info(f"Detected {duplicates} duplicate timestamps — summing production values.")
        df_prod = df_prod.groupby(df_prod.index).sum(numeric_only=True)

    # Re-add metadata columns that were dropped during aggregation
    df_prod["pricearea"] = selected_area
    df_prod["productiongroup"] = selected_group

    # Ensure index is unique and sorted for STL
    df_prod = df_prod[~df_prod.index.duplicated(keep="first")].sort_index()

    # Ensure regular hourly frequency
    df_prod = df_prod.asfreq("H")

    # Interpolate missing production values linearly
    df_prod["quantitykwh"] = df_prod["quantitykwh"].interpolate(method="linear")

    #st.write(
    #    f"Loaded **{len(df_prod)} hourly records** for {selected_group.upper()} production in {selected_area}."
    #)

    # --- Tabs for STL and Spectrogram ---
    tab_stl, tab_spec = st.tabs(["📊 STL Decomposition", "🌀 Spectrogram"])

    # ==================================================
    # TAB 1: STL Decomposition
    # ==================================================
    with tab_stl:
        st.subheader(f"STL Decomposition for {selected_group.upper()} in {selected_area}")

        # User parameters
        st.markdown("#### Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            period = st.number_input(
                "Period (hours per cycle)", min_value=1, max_value=1000, value=24,
                help="Typical daily pattern = 24 hours"
            )
        with col2:
            seasonal = st.number_input(
                "Seasonal smoothing length", min_value=3, max_value=101, value=13, step=2,
                help="Controls smoothness of the seasonal component"
            )
        with col3:
            trend = st.number_input(
                "Trend smoothing length", min_value=3, max_value=301, value=25, step=2,
                help="Controls smoothness of the trend component"
            )

        robust = st.checkbox("Use robust mode (less sensitive to outliers)", value=True)

        # Convert df to the column naming convention expected by STL function
        df_stl = df_prod.rename(
            columns={
                "pricearea": "priceArea",
                "productiongroup": "productionGroup",
                "quantitykwh": "quantityKwh"
            }
        )

        # Run STL decomposition
        try:
            fig_stl, res_stl = stl_decomposition_plotly_subplots(
                df_stl,
                price_area=selected_area,
                production_group=selected_group,
                period=period,
                seasonal=seasonal,
                trend=trend,
                robust=robust
            )
            st.plotly_chart(fig_stl, use_container_width=True)

            # Residual statistics
            residuals = res_stl.resid.dropna()
            st.markdown("#### Residual Summary")
            summary_df = pd.DataFrame({
                "Mean": [residuals.mean()],
                "Std Dev": [residuals.std()],
                "Min": [residuals.min()],
                "Max": [residuals.max()]
            }).T.rename(columns={0: "Value"})
            st.table(summary_df)

        except Exception as e:
            st.error(f"Error performing STL decomposition: {e}")

        with st.expander("ℹ️ About STL Decomposition"):
            st.markdown("""
            **STL (Seasonal-Trend decomposition using Loess)** splits a time series into:
            - **Trend:** long-term change  
            - **Seasonal:** repeating daily/weekly pattern  
            - **Residual:** short-term noise or anomalies
            """)

    # ==================================================
    # TAB 2: Spectrogram
    # ==================================================
    with tab_spec:
        st.subheader(f"Spectrogram for {selected_group.upper()} in {selected_area}")

        st.markdown("#### Parameters")
        col1, col2 = st.columns(2)
        with col1:
            window_length = st.slider(
                "Window length (hours per segment)",
                24, 500, 168, 12,
                help="Typical: 168 hours = 1 week"
            )
        with col2:
            overlap = st.slider(
                "Window overlap (hours)",
                0, window_length - 1, int(window_length / 2), 1,
                help="Overlap between segments"
            )

        # Fix column names for the spectrogram function
        df_spec = df_prod.rename(
            columns={
                "pricearea": "priceArea",
                "productiongroup": "productionGroup",
                "quantitykwh": "quantityKwh"
            }
        )

        try:
            # --- Use Plotly-based spectrogram ---
            f, t, Zxx = plotly_spectrogram(
                df_spec,
                price_area=selected_area,
                production_group=selected_group,
                window_length=window_length,
                window_overlap=overlap,
                fs=1
            )

            # Display interactive Plotly chart in Streamlit
            power = np.abs(Zxx)
            fig = go.Figure(
                data=go.Heatmap(
                    z=power,
                    x=t,
                    y=f,
                    colorscale='Viridis',
                    colorbar=dict(title='Amplitude'),
                )
            )

            fig.update_layout(
                title=f"Spectrogram: {selected_group.capitalize()} in {selected_area.upper()}",
                xaxis_title="Time [hours]",
                yaxis_title="Frequency [1/hour]",
                template="plotly_dark",
                height=500,
                width=1000,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Power summary
            st.markdown("#### Frequency Domain Summary")
            df_summary = pd.DataFrame({
                "Mean Power": [power.mean()],
                "Max Power": [power.max()],
                "Dominant Frequency": [f[np.argmax(power.mean(axis=1))]]
            }).T.rename(columns={0: "Value"})
            st.table(df_summary)

        except Exception as e:
            st.error(f"Error computing spectrogram: {e}")

        with st.expander("ℹ️ About Spectrogram"):
            st.markdown("""
            A **spectrogram** visualizes how the frequency content of production changes over time.  
            - Daily cycles → frequency ≈ 1/24  
            - Weekly cycles → frequency ≈ 1/168  
            Peaks highlight periodic production behavior.
            """)




# -----------------------------
# Page: Weather Data table
# -----------------------------
def page_data_table():
    st.title("Weather Data — January 2021 Overview")

    selected_area = st.session_state.get("selected_area", "NO1") # Default to NO1 if not set
    # Get the city corresponding to the selected area
    city_name = df_price_areas.loc[df_price_areas['price_area'] == selected_area, 'city'].values[0]

    st.write(
    f"This table shows the minimum, maximum and mean values for each variable in January. "
    f"The January (hourly) column contains the January time series for the selected area: {selected_area} ({city_name})."
    )

    
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
# Page: Interactive plots (Plotly version)
# -----------------------------
import plotly.graph_objects as go

def page_plots():
    st.title("Weather Plots 2021")
    selected_area = st.session_state.get("selected_area", "NO1")

    # Get city corresponding to the selected area
    city_name = df_price_areas.loc[
        df_price_areas['price_area'] == selected_area, 'city'
    ].values[0]

    df_weather = load_weather_data(selected_area)
    if df_weather.empty:
        st.warning(f"No weather data for {selected_area}")
        return

    # User controls
    choice = st.selectbox("Select variable", ["All"] + df_weather.columns.tolist(), index=0)
    month_range = st.select_slider(
        "Select month range", options=list(range(1, 13)), value=(1, 1),
        format_func=lambda x: MONTH_NAMES[x]
    )
    start_month, end_month = month_range

    # Filter by selected months
    df_sel = df_weather[
        (df_weather.index.month >= start_month) & (df_weather.index.month <= end_month)
    ]

    # Prepare figure
    fig = go.Figure()
    left_columns = [c for c in df_sel.columns if c != 'wind_direction_10m']

    # --- Left Y-axis (temp, wind speed, gusts) ---
    if choice == "All":
        for col in left_columns:
            fig.add_trace(go.Scatter(
                x=df_sel.index, y=df_sel[col],
                mode='lines',
                name=col
            ))
        yaxis_title_left = "Temperature / Wind Speed / Gusts"
    elif choice != "wind_direction_10m":
        fig.add_trace(go.Scatter(
            x=df_sel.index, y=df_sel[choice],
            mode='lines',
            name=choice
        ))
        yaxis_title_left = choice
    else:
        yaxis_title_left = "Temperature / Wind Speed / Gusts"

    # --- Right Y-axis (wind direction) ---
    if 'wind_direction_10m' in df_sel.columns and (choice == "All" or choice == 'wind_direction_10m'):
        fig.add_trace(go.Scatter(
            x=df_sel.index, y=df_sel['wind_direction_10m'],
            mode='lines',
            name='Wind Direction (°)',
            yaxis='y2',
            line=dict(color='lightgray', dash='dot')
        ))

    # Layout configuration
    fig.update_layout(
        title=f"Weather in {selected_area} ({city_name}) for months {MONTH_NAMES[start_month]} – {MONTH_NAMES[end_month]}",
        xaxis=dict(title="Time"),
        yaxis=dict(title=yaxis_title_left),
        yaxis2=dict(
            title="Wind Direction (°)",
            overlaying="y",
            side="right",
            range=[0, 360],
            showgrid=False
        ),
        legend=dict(orientation="h", y=-0.2),
        template="plotly_white",
        hovermode="x unified",
        height=600
    )

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)




# -----------------------------
# Page newB: Outlier & Anomaly (skeleton)
# -----------------------------
def page_newB():

    st.title("Outlier and Anomaly Detection")

    selected_area = st.session_state.get("selected_area", "NO1")
    # Get the city corresponding to the selected area
    city_name = df_price_areas.loc[df_price_areas['price_area'] == selected_area, 'city'].values[0]
    df_weather = load_weather_data(selected_area)

    if df_weather.empty:
        st.warning(f"No weather data for {selected_area}")
        return

    tabs = st.tabs(["SPC Outlier Detection", "Precipitation Anomaly Detection (LOF)"])

    # -------------------- Tab 1: SPC Outlier Detection --------------------
    with tabs[0]:
        st.subheader(f"SPC Outlier Detection for {selected_area} ({city_name})")
        # Let user select variable and parameters
        numeric_cols = df_weather.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            st.warning("No numeric weather variables available for SPC analysis.")
        else:
            var_choice = st.selectbox("Select variable", numeric_cols, index=0)
            cutoff_frac = st.slider("DCT cutoff fraction", 0.001, 0.2, 0.05, 0.001)
            k = st.slider("SPC k parameter (σ multiplier)", 0.5, 6.0, 3.0, 0.1)

            series = df_weather[var_choice].dropna()
            fig_spc, summary_spc = spc_outlier_plotly(series, cutoff_frac=cutoff_frac, k=k,
                                                      title=f"{var_choice} SPC Outlier Detection")
            st.plotly_chart(fig_spc, use_container_width=True)

            # present the summary as a table
            st.write("### SPC Summary Table")
            summary_df = pd.DataFrame.from_dict(summary_spc, orient='index', columns=['Value'])
            st.table(summary_df)

    # -------------------- Tab 2: Precipitation Anomaly Detection (LOF) --------------------
    with tabs[1]:
        st.subheader(f"Precipitation Anomaly Detection (LOF) for {selected_area} ({city_name})")
        if "precipitation" not in df_weather.columns:
            st.warning("No precipitation data available for LOF analysis.")
        else:
            contamination = st.slider(
                "Contamination (expected outlier fraction)",
                0.001, 0.05, 0.01, 0.001,
                help="Fraction of data expected to be outliers (1% is a common default)"
    )

            n_neighbors = st.slider(
                "LOF neighbors (local window size)",
                5, 60, 20, 1,
                help="Number of neighbors used to define the local density (smaller = more sensitive)"
        )


            series_precip = df_weather["precipitation"].dropna()
            fig_lof, summary_lof = lof_precipitation_plotly(series_precip, contamination=contamination,
                                                            n_neighbors=n_neighbors,
                                                            title="Precipitation Anomalies (LOF)")
            st.plotly_chart(fig_lof, use_container_width=True)

            # Present the summary as a table
            st.write("### LOF Summary Table")
            summary_df = pd.DataFrame.from_dict(summary_lof, orient='index', columns=['Value'])
            st.table(summary_df)



# -----------------------------
# Page: Price Areas Map
# -----------------------------
def page_map():
    import folium
    from streamlit_folium import st_folium
    import json
    import pandas as pd
    from shapely.geometry import shape, Point
    from datetime import datetime
    import datetime as dt

    st.title("Price Areas Map")

    # -----------------------------------------
    # 1. MongoDB Connections
    # -----------------------------------------
    production_col = db["production_data"]
    consumption_col = db["consumption_data"]

    # -----------------------------------------
    # 2. Load GeoJSON + helper mappings
    # -----------------------------------------
    @st.cache_data
    def load_geojson():
        with open("price_areas.geojson") as f:
            return json.load(f)

    geojson_data = load_geojson()

    @st.cache_data
    def build_id_to_name(gj):
        out = {}
        for f in gj.get("features", []):
            fid = f.get("id") or (f.get("properties") or {}).get("id")
            if fid is None:
                continue
            name = (f.get("properties") or {}).get("ElSpotOmr")
            if name:
                out[fid] = str(name)
        return out

    id_to_name = build_id_to_name(geojson_data)

    # Price area name → ID used in GeoJSON
    AREA_ID_MAP = {
        "NO1": 6,
        "NO2": 7,
        "NO3": 8,
        "NO4": 9,
        "NO5": 10
    }

    # -----------------------------------------
    # 3. Build polygons for coordinate lookup
    # -----------------------------------------
    if "polygons" not in st.session_state:
        polys = []
        for feat in geojson_data.get("features", []):
            fid = feat.get("id") or (feat.get("properties") or {}).get("id")
            if not fid:
                continue
            try:
                geom = shape(feat["geometry"])
            except Exception:
                continue
            polys.append((fid, geom))
        st.session_state.polygons = polys

    def find_feature_id(lon: float, lat: float):
        pt = Point(lon, lat)
        for fid, geom in st.session_state.polygons:
            if geom.covers(pt):
                return fid
        return None

    # -----------------------------------------
    # 4. UI – Dataset, Group, Date Selection
    # -----------------------------------------
    st.subheader("Choropleth Controls")

    data_type = st.radio(
        "Dataset",
        ["Production", "Consumption"],
        horizontal=True
    )

    # Select column names depending on dataset
    if data_type == "Production":
        groups = production_col.distinct("productiongroup")
        col_group = "productiongroup"
        col_time = "starttime"
        col_area = "pricearea"
        col_kwh = "quantitykwh"
    else:
        groups = consumption_col.distinct("groupname")
        col_group = "groupname"
        col_time = "starttime"
        col_area = "pricearea"
        col_kwh = "quantitykwh"

    group_select = st.selectbox("Select group", sorted(groups))

    # Date selection
    MIN_DATE = dt.date(2021, 1, 1)
    MAX_DATE = dt.date(2024, 12, 31)

    st.subheader("Select Time Interval")
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start date",
            value=MIN_DATE,
            min_value=MIN_DATE,
            max_value=MAX_DATE
        )

    with col2:
        end_date = st.date_input(
            "End date",
            value=MAX_DATE,
            min_value=MIN_DATE,
            max_value=MAX_DATE
        )

    if start_date > end_date:
        st.error("❌ Start date must be before end date.")
        st.stop()

    # -----------------------------------------
    # 5. Query MongoDB
    # -----------------------------------------
    @st.cache_data
    def query_data(data_type, group, start, end,
                   col_group, col_time, col_area, col_kwh):

        col = production_col if data_type == "Production" else consumption_col

        pipeline = [
            {"$match": {
                col_group: group,
                col_time: {
                    "$gte": datetime.combine(start, datetime.min.time()),
                    "$lte": datetime.combine(end, datetime.max.time())
                }
            }},
            {"$group": {
                "_id": f"${col_area}",
                "mean_value": {"$avg": f"${col_kwh}"}
            }}
        ]

        df = pd.DataFrame(list(col.aggregate(pipeline)))
        if df.empty:
            return pd.DataFrame({"id": [], "value": []})

        # Convert NO3 → 8 etc.
        df["id"] = df["_id"].map(AREA_ID_MAP)
        df["value"] = df["mean_value"]
        df = df.dropna(subset=["id"])

        return df[["id", "value"]]

    # Call query
    df_vals = query_data(
        data_type,
        group_select,
        start_date,
        end_date,
        col_group,
        col_time,
        col_area,
        col_kwh
    )


    # -----------------------------------------
    # 6. Initialize session state for selection
    # -----------------------------------------
    if "last_pin" not in st.session_state:
        st.session_state.last_pin = [66.32624933088354, 14.186465980232347]
    if "selected_feature_id" not in st.session_state:
        st.session_state.selected_feature_id = None

    if st.session_state.selected_feature_id is None:
        lat, lon = st.session_state.last_pin
        st.session_state.selected_feature_id = find_feature_id(lon, lat)

    # -----------------------------------------
    # 7. Layout: Map + Info
    # -----------------------------------------
    map_col, info_col = st.columns([2.2, 1])

    # ===== Map =====
    with map_col:
        m = folium.Map(
            location=st.session_state.last_pin,
            zoom_start=5,
            tiles="OpenStreetMap"
        )

        # Choropleth
        folium.Choropleth(
            geo_data=geojson_data,
            data=df_vals,
            columns=["id", "value"],
            key_on="feature.id",
            fill_color="YlGnBu",
            fill_opacity=0.4,
            line_opacity=0.8,
            line_color="white",
            nan_fill_opacity=0.1,
            legend_name=f"Mean {data_type} ({group_select})",
            highlight=True
        ).add_to(m)


        # Highlight the selected polygon
        sel_id = st.session_state.selected_feature_id
        if sel_id is not None:
            selected_feats = [
                f for f in geojson_data.get("features", [])
                if f.get("id") == sel_id
            ]
            if selected_feats:
                folium.GeoJson(
                    {"type": "FeatureCollection", "features": selected_feats},
                    style_function=lambda f: {
                        "fillOpacity": 0,
                        "color": "red",
                        "weight": 3
                    }
                ).add_to(m)

        # Pin marker
        folium.Marker(
            location=st.session_state.last_pin,
            icon=folium.Icon(color="red")
        ).add_to(m)

        # make the map a bit more zoomed out to see context
        m.fit_bounds(m.get_bounds(), padding=(15, 15))


        # Update map click
        out = st_folium(m, key="choropleth_map", height=600)

        if out and out.get("last_clicked"):
            lat = out["last_clicked"]["lat"]
            lon = out["last_clicked"]["lng"]
            new_coord = [lat, lon]
            if new_coord != st.session_state.last_pin:
                st.session_state.last_pin = new_coord
                st.session_state.selected_feature_id = find_feature_id(lon, lat)
                st.rerun()

    # ===== Info Box =====
    with info_col:
        st.subheader("Selection")
        st.write(f"Lat: {st.session_state.last_pin[0]:.6f}")
        st.write(f"Lon: {st.session_state.last_pin[1]:.6f}")

        fid = st.session_state.selected_feature_id
        if fid is None:
            st.write("Outside any price area.")
        else:
            area_name = id_to_name.get(fid, f"ID {fid}")
            value = df_vals.loc[df_vals["id"] == fid, "value"]

            value_display = float(value.iloc[0]) if len(value) else "No data"

            st.write(f"Area: {area_name}")
            st.write(f"Mean kWh: {value_display}")





# -----------------------------
import streamlit as st
import pandas as pd

def inspect_mongo():
    st.header("MongoDB Data Inspection")

    st.subheader("Production Data – Sample")
    prod_sample = list(production_collection.find().limit(5))
    st.json(prod_sample)

    st.subheader("Production Columns")
    if prod_sample:
        prod_df = pd.DataFrame(prod_sample)
        st.write(prod_df.columns.tolist())
        st.dataframe(prod_df)

    st.divider()

    st.subheader("Consumption Data – Sample")
    cons_sample = list(consumption_collection.find().limit(5))
    st.json(cons_sample)

    st.subheader("Consumption Columns")
    if cons_sample:
        cons_df = pd.DataFrame(cons_sample)
        st.write(cons_df.columns.tolist())
        st.dataframe(cons_df)

    st.divider()

    st.write("Record counts:")
    st.write("Production:", production_collection.count_documents({}))
    st.write("Consumption:", consumption_collection.count_documents({}))

    # What values does groupName take in consumption data and production data?
    st.subheader("Distinct groupName values in Consumption Data")
    cons_groups = consumption_collection.distinct("groupName")
    st.write(cons_groups)

    st.subheader("Distinct groupName values in Production Data")
    prod_groups = production_collection.distinct("groupName")
    st.write(prod_groups)



# -----------------------------
# Snow Drift Inspection Page
# -----------------------------
import Snow_drift as sd

def inspect_snow_drift():
    from datetime import datetime
    st.header("Snow Drift Analysis")

    # Check map selection
    if st.session_state.get("selected_feature_id") is None:
        st.warning("Please select a location on the map before calculating snow drift.")
        return

    lat, lon = st.session_state.last_pin
    st.subheader("Selected Location")
    st.write(f"Latitude: {lat:.6f}")
    st.write(f"Longitude: {lon:.6f}")

    # Year range selection. Can
    start_snowyear, end_snowyear = st.slider(
        "Select year range",
        min_value=2015,
        max_value=2023,
        value=(2020, 2023), # default range
        step=1
    )

    # Standard default values for snow drift calculation
    T = 3000      # Max transport distance [m]
    F = 30000     # Fetch distance [m]
    theta = 0.5   # Relocation coefficient

    # Fetch weather data
    # Assume lat, lon are defined, and start_snowyear, end_snowyear come from your slider
    calendar_years = list(range(start_snowyear, end_snowyear + 2))  # +2 to include last snowyear's next year

    dfs = []
    for y in calendar_years:
        df_year = fetch_era5_data(lat, lon, y)
        dfs.append(df_year)

    # Combine all years into one DataFrame
    df_weather = pd.concat(dfs)
    df_weather.sort_index(inplace=True)
    df_weather.reset_index(inplace=True)


    df_weather.rename(columns={
        "temperature_2m": "temperature_2m (°C)",
        "precipitation": "precipitation (mm)",
        "wind_speed_10m": "wind_speed_10m (m/s)",
        "wind_gusts_10m": "wind_gusts_10m (m/s)",
        "wind_direction_10m": "wind_direction_10m (°)"
    }, inplace=True)


    # Convert the 'time' column to datetime.
    df_weather['time'] = pd.to_datetime(df_weather['time'])

    df_weather['month'] = df_weather['time'].dt.month
    # show df_weather months for debugging
    st.write(df_weather[['time', 'month']])
    
    # Define season: if month >= 7, season = current year; otherwise, season = previous year.
    # Only rows from July onward get the season year
    df_weather['season'] = df_weather['time'].apply(lambda dt: dt.year if dt.month >= 7 else dt.year - 1)
    
    #OBS Show df_weather['season'] for debugging. the whole thing
    st.write(df_weather[['time', 'season']])

    # Riktig fram til hit.



        
    # Compute seasonal results (yearly averages for each season).
    yearly_df = sd.compute_yearly_results(df_weather, T, F, theta)
    overall_avg = yearly_df['Qt (kg/m)'].mean()
    st.write(f"Overall average Qt over all seasons: {overall_avg / 1000:.1f} tonnes/m")
    
    yearly_df_disp = yearly_df.copy()
    yearly_df_disp["Qt (tonnes/m)"] = yearly_df_disp["Qt (kg/m)"] / 1000
    st.write("Yearly average snow drift (Qt) per season (in tonnes/m):")
    st.dataframe(yearly_df_disp[['season', 'Qt (tonnes/m)']].style.format({"Qt (tonnes/m)": "{:.1f}"}))

    overall_avg_tonnes = overall_avg / 1000
    print(f"\nOverall average Qt over all seasons: {overall_avg_tonnes:.1f} tonnes/m")
    
    # Compute the average directional breakdown (average over all seasons).
    avg_sectors = sd.compute_average_sector(df_weather, T, F, theta)
    
    # Create the rose plot canvas with the average directional breakdown.
    sd.plot_rose(avg_sectors, overall_avg_tonnes)



    # -----------------------------
    # Monthly Snow Drift Calculation
    # -----------------------------
    st.subheader("Monthly Snow Drift")

    # Compute monthly results
    monthly_df = sd.compute_monthly_results(df_weather, T, F, theta)

    # Convert Qt to tonnes/m for readability
    monthly_df['Qt (tonnes/m)'] = monthly_df['Qt (kg/m)'] / 1000

    # Display monthly results
    st.write("Monthly snow drift (Qt) results (in tonnes/m):")
    st.dataframe(monthly_df[['season', 'month', 'Qt (tonnes/m)']].style.format({"Qt (tonnes/m)": "{:.1f}"}))



    # -----------------------------
    # 1. Ensure monthly timestamps
    # -----------------------------
    monthly_df["month_dt"] = monthly_df.apply(
        lambda row: pd.Timestamp(
            year=int(row["season"]) if row["month"] >= 7 else int(row["season"]) + 1,
            month=int(row["month"]),
            day=1
        ),
        axis=1
    )
    monthly_df["Qt_tonnes"] = monthly_df["Qt (kg/m)"] / 1000
    monthly_df["Type"] = "Monthly Qt"


    # -----------------------------
    # 2. Expand seasonal Qt to match months
    # -----------------------------
    seasonal_expanded = []

    for _, row in yearly_df.iterrows():
        # Extract starting year from "season" string like "2015-2016"
        season_str = row["season"]
        season_year = int(season_str.split("-")[0])  # first year
        qt_tonnes = row["Qt (kg/m)"] / 1000

        # Jul→Dec of season_year, Jan→Jun of season_year+1
        months = list(range(7, 13)) + list(range(1, 7))
        for m in months:
            year = season_year if m >= 7 else season_year + 1
            dt = pd.Timestamp(year=year, month=m, day=1)
            seasonal_expanded.append({
                "month_dt": dt,
                "Qt_tonnes": qt_tonnes,
                "Type": "Seasonal Qt"
            })

    seasonal_df = pd.DataFrame(seasonal_expanded)



    # -----------------------------
    # 3. Combine monthly + seasonal
    # -----------------------------
    plot_df = pd.concat([monthly_df[["month_dt", "Qt_tonnes", "Type"]], seasonal_df])
    plot_df.sort_values("month_dt", inplace=True)
    plot_df.reset_index(drop=True, inplace=True)




    import plotly.express as px

    # --- Add Month-Year labels ---
    plot_df["month_label"] = plot_df["month_dt"].dt.strftime("%b %Y")
    seasonal_df["month_label"] = seasonal_df["month_dt"].dt.strftime("%b %Y")
    monthly_df["month_label"] = monthly_df["month_dt"].dt.strftime("%b %Y")

    # --- Seasonal bar chart (no spacing, light blue transparent) ---
    fig = px.bar(
        seasonal_df,
        x="month_label",
        y="Qt_tonnes",
        color="Type",
        barmode="group",
    )

    # Color the seasonal bars only (light blue w/ transparency)
    fig.for_each_trace(
        lambda t: t.update(
            marker_color="rgba(135, 206, 250, 0.45)",  # light blue, transparent
            showlegend=True
        ) if t.name == "Seasonal Qt" else None
    )

    # Remove spacing between bars
    fig.update_layout(
        bargap=0,      # gap between bars in a group
        bargroupgap=0, # gap between groups
    )

    # --- Add monthly Qt line (dark blue) ---
    fig.add_scatter(
        x=monthly_df["month_label"],
        y=monthly_df["Qt_tonnes"],
        mode="lines+markers",   # add markers

        name="Monthly Qt",
        line=dict(width=3, color="rgba(0, 51, 153, 1)")  # dark blue
    )

    # --- Styling ---
    fig.update_layout(
        title="Monthly vs Seasonal Snow Drift (Qt)",
        xaxis_title="Month",
        yaxis_title="Qt (tonnes/m)",
        template="plotly_white",
        legend_title_text="Qt Type",
        xaxis_tickangle=45
    )

    st.plotly_chart(fig, use_container_width=True)




# -----------------------------
# Navigation
# -----------------------------
pg_home = st.Page(page_home, title="Home", icon="🏠")
pg_energy = st.Page(page_energy, title="Energy Production", icon="⚙️")
pg_newA = st.Page(page_newA, title="STL & Spectrogram", icon="🌀")
pg_data = st.Page(page_data_table, title="Weather Data", icon="📋")
pg_plots = st.Page(page_plots, title="Weather Plots", icon="📈")
pg_newB = st.Page(page_newB, title="Outlier & Anomaly", icon="🚨")
pg_map = st.Page(page_map, title="Price Areas Map", icon="🗺️")

pg_inspect = st.Page(inspect_mongo, title="Inspect MongoDB", icon="🔍")
pg_snow = st.Page(inspect_snow_drift, title="Snow Drift", icon="❄️")

nav = st.navigation(pages=[
    pg_home,
    pg_energy,
    pg_newA,
    pg_data,
    pg_plots,
    pg_newB,
    pg_map,
    pg_inspect,
    pg_snow
])
nav.run()

