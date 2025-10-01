from secrets import choice
from typing import Callable, List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data (cached)
# -----------------------------
@st.cache_data  # caches the loaded dataframe for app speed
def load_data(csv_path: str = "open-meteo-subset.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure a datetime column and a 'date' column for grouping
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time']) 
        df['date'] = df['time'].dt.date # extract date part for daily grouping
    return df

df = load_data()

# Identify data columns (exclude time and date)
data_columns = [c for c in df.columns if c not in ['time', 'date']]

# Helper: month name mapping
MONTH_NAMES = {i: pd.Timestamp(2020, i, 1).strftime("%b") for i in range(1,13)} # map 1->Jan, 2->Feb, etc.

# -----------------------------
# Page: Home
# -----------------------------
def page_home():
    st.title("Home")
    st.write("Welcome to my app! Use the sidebar to navigate.")


# -----------------------------
# Page: Data table (row-wise LineChartColumn for first month) with optional highlighting
# -----------------------------
def page_data_table() -> None:
    st.title("Data Table — January overview")
    st.write(
        "This table shows the minimum, maximum and mean values for each variable in January. The January (hourly) column contains the January time series as a line chart."
    )

    # Ensure 'time' column exists. Unnecessary if we know the CSV is correct.
    if 'time' not in df.columns:
        st.error("No 'time' column in CSV — this page requires a time column.")
        return

    df_jan = df[df['time'].dt.month == 1]

    series_rows = []
    for col in data_columns:
        series = df_jan[col].tolist()
        series_pd = pd.Series(series)
        series_rows.append({
            "variable": col,
            "mean": series_pd.mean().round(1) if pd.api.types.is_numeric_dtype(series_pd) else None,
            "min": series_pd.min().round(1) if pd.api.types.is_numeric_dtype(series_pd) else None,
            "max": series_pd.max().round(1) if pd.api.types.is_numeric_dtype(series_pd) else None,
            "Jan": series
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
def page_plots() -> None:
    st.title("Interactive Plots")
    st.write("Choose a variable (or All) and a month-range to plot")

    if 'time' not in df.columns:
        st.error("No 'time' column in CSV — this page requires a time column.")
        return

    # Column selectbox: allow "All" plus the data columns
    cols = [c for c in data_columns]
    choice = st.selectbox("Select variables to plot", ["All"] + cols, index=0)

    # Month selection slider
    month_options = list(range(1,13))
    month_range = st.select_slider(
        "Select month range (inclusive)",
        options=month_options,
        value=(1,1),
        format_func=lambda x: MONTH_NAMES.get(x, str(x))
    )
    if isinstance(month_range, int):
        month_range = (month_range, month_range)
    start_month, end_month = month_range

    # Filter dataframe for selected months
    df_sel = df[(df['time'].dt.month >= start_month) & (df['time'].dt.month <= end_month)].copy()
    if df_sel.empty:
        st.warning("No data for selected months.")
        return

    # Prepare figure
    fig, ax1 = plt.subplots(figsize=(12,7))

    # Columns for left y-axis (exclude wind_direction)
    left_columns = [c for c in data_columns if c != 'wind_direction_10m (°)']

    if choice == "All":
        for col in left_columns:
            ax1.plot(df_sel['time'], df_sel[col], label=col)
        ax1.set_ylabel("Temperature (°C) / Wind (m/s) / Wind Gusts (m/s)")
    elif choice != 'wind_direction_10m (°)':
        # Single numeric column
        if not pd.api.types.is_numeric_dtype(df_sel[choice]):
            st.warning(f"Column '{choice}' is not numeric. Showing value counts instead.")
            st.dataframe(df_sel[choice].value_counts().rename_axis(choice).reset_index(name="count"))
            return
        ax1.plot(df_sel['time'], df_sel[choice], linestyle='-')
        ax1.set_ylabel(choice)

    # Right y-axis for wind direction.
    if 'wind_direction_10m (°)' in df_sel.columns and (choice == "All" or choice == 'wind_direction_10m (°)'):
        ax2 = ax1.twinx()
        ax2.plot(df_sel['time'], df_sel['wind_direction_10m (°)'], color='lightgray', label='Wind Direction')
        ax2.set_ylabel("Wind Direction (°)")
        ax2.set_ylim(0, 360)  # wind direction in degrees
        ax2.legend(loc='upper right')


    # Final formatting
    ax1.set_title(f"Data for months {MONTH_NAMES[start_month]} – {MONTH_NAMES[end_month]}")
    ax1.set_xlabel("Time")
    ax1.legend(loc='upper left', fontsize='small')
    ax1.grid(True)
    fig.autofmt_xdate()
    st.pyplot(fig)

# -----------------------------
# Page: Extra placeholder
# -----------------------------
def page_extra() -> None:
    st.title("Extra / Placeholder")
    st.write("This is a placeholder page. Will be used for extra content later.")

# -----------------------------
# Create st.Page objects and navigation
# -----------------------------
pg_home = st.Page(page_home, title="Home", icon="🏠")
pg_data = st.Page(page_data_table, title="Data", icon="📋")
pg_plots = st.Page(page_plots, title="Plots", icon="📈")
pg_extra = st.Page(page_extra, title="Extra", icon="⚙️")

# The navigation object builds the UI and runs the selected page.
nav = st.navigation(pages=[pg_home, pg_data, pg_plots, pg_extra])
nav.run()
