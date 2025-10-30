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
import matplotlib.pyplot as plt
from scipy.signal import stft

def matplotlib_spectrogram(
    df,
    price_area='NO1',
    production_group='hydro',
    window_length=168,   # nperseg in STFT
    window_overlap=84,   # noverlap in STFT
    fs=1                 # sampling frequency (1 per hour)
):
    """
    Compute and plot a spectrogram for electricity production data using Matplotlib.

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
    # Filter data
    ts = df[(df['priceArea'].str.lower() == price_area.lower()) &
            (df['productionGroup'].str.lower() == production_group.lower())]['quantityKwh']
    
    if ts.empty:
        raise ValueError(f"No data for price area '{price_area}' and production group '{production_group}'")
    
    ts = ts.asfreq('h').ffill()
    
    # Compute STFT
    f, t, Zxx = stft(ts.values, fs=fs, nperseg=window_length, noverlap=window_overlap)
    
    # Plot with Matplotlib
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis', vmin=0, vmax=np.max(np.abs(Zxx)))
    plt.title(f'Spectrogram: {production_group} in {price_area}')
    plt.ylabel('Frequency [1/hour]')
    plt.xlabel('Time [hours]')
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.show()
    
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
        price_areas = collection.distinct("pricearea")
        selected_area = st.radio("Select Price Area", price_areas)
        st.session_state["selected_area"] = selected_area

        data = list(collection.find({"pricearea": selected_area}))
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
        production_groups = list(collection.distinct("productiongroup"))
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
        data_filtered = list(collection.find(query))
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
    st.title("STL Decomposition & Spectrogram")
    st.write("To be implemented: STL decomposition & Spectrogram for selected_area weather data.")

# -----------------------------
# Page: Weather Data table
# -----------------------------
def page_data_table():
    st.title("Weather Data — January Overview")

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
# Page: Interactive plots
# -----------------------------
def page_plots():
    st.title("Weather Plots")
    selected_area = st.session_state.get("selected_area", "NO1")
    # Get the city corresponding to the selected area
    city_name = df_price_areas.loc[df_price_areas['price_area'] == selected_area, 'city'].values[0]
    df_weather = load_weather_data(selected_area)
    if df_weather.empty:
        st.warning(f"No weather data for {selected_area}")
        return

    choice = st.selectbox("Select variable", ["All"] + df_weather.columns.tolist(), index=0)
    month_range = st.select_slider(
        "Select month range", options=list(range(1,13)), value=(1,1),
        format_func=lambda x: MONTH_NAMES[x]
    )
    start_month, end_month = month_range

    df_sel = df_weather[(df_weather.index.month >= start_month) & (df_weather.index.month <= end_month)]
    fig, ax1 = plt.subplots(figsize=(12,6))

    # Columns for left y-axis (all except wind_direction_10m)
    left_columns = [c for c in df_sel.columns if c != 'wind_direction_10m']

    if choice == "All":
        for col in left_columns:
            ax1.plot(df_sel.index, df_sel[col], label=col)
        ax1.set_ylabel("Temperature / Wind speed / Gusts")
    elif choice != 'wind_direction_10m':
        ax1.plot(df_sel.index, df_sel[choice], label=choice)
        ax1.set_ylabel(choice)

    # Right y-axis for wind direction. 
    if 'wind_direction_10m' in df_sel.columns and (choice == "All" or choice == 'wind_direction_10m'):
        ax2 = ax1.twinx()
        ax2.plot(df_sel.index, df_sel['wind_direction_10m'], color='lightgray', label='Wind Direction')
        ax2.set_ylabel("Wind Direction (°)")
        ax2.set_ylim(0, 360)
        ax2.legend(loc='upper right')

    ax1.set_title(f"Weather in {selected_area} ({city_name}) for months {MONTH_NAMES[start_month]} – {MONTH_NAMES[end_month]}")
    ax1.set_xlabel("Time")
    ax1.legend(loc='upper left')
    ax1.grid(True)
    fig.autofmt_xdate()
    st.pyplot(fig)


# -----------------------------
# Page newB: Outlier & Anomaly (skeleton)
# -----------------------------
def page_newB():

    st.title("Outlier and Anomaly Detection")

    selected_area = st.session_state.get("selected_area", "NO1")
    df_weather = load_weather_data(selected_area)

    if df_weather.empty:
        st.warning(f"No weather data for {selected_area}")
        return

    tabs = st.tabs(["SPC Outlier Detection", "Precipitation Anomaly Detection (LOF)"])

    # -------------------- Tab 1: SPC Outlier Detection --------------------
    with tabs[0]:
        st.subheader(f"SPC Outlier Detection for {selected_area}")
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
        st.subheader(f"Precipitation Anomaly Detection (LOF) for {selected_area}")
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
