# app.py
"""
IND320 Streamlit app entrypoint using st.Page + st.navigation
- Put this file in the root of your repository (same folder as open-meteo-subset.csv)
- Make sure requirements.txt includes streamlit>=1.32, pandas, matplotlib, numpy
"""

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
numeric_columns = df.select_dtypes(include='number').columns.tolist()

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
    st.title("Data Table — First month overview")
    st.write(
        "This page shows one row per variable. "
        "The `Jan` column contains the January time series (row-wise) as a sparkline / line chart."
    )

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
        "mean": st.column_config.NumberColumn(label="Mean", format="%.1f", width="small"),
        "min": st.column_config.NumberColumn(label="Min", format="%.1f", width="small"),
        "max": st.column_config.NumberColumn(label="Max", format="%.1f", width="small"),
        "Jan": st.column_config.LineChartColumn(label="January (hourly)", help="Hourly time series", width="large", y_min=None, y_max=None)
    }

    st.dataframe(df_series, column_config=column_config, use_container_width=True)


# -----------------------------
# Page: Interactive plots
# -----------------------------
def page_plots() -> None:
    st.title("Interactive Plots")
    st.write("Choose a column (or All) and a month-range to plot. Default month selection is January.")

    if 'time' not in df.columns:
        st.error("No 'time' column in CSV — this page requires a time column.")
        return

    # Column selectbox: allow "All" plus the data columns
    cols = [c for c in data_columns]
    choice = st.selectbox("Select column to plot", ["All"] + cols, index=0)

    # Month selection slider: allow selecting a range of months; default = first month (1)
    months = sorted(df['time'].dt.month.unique())
    # if months not covering full year, use 1..12 as options but pick the first present month
    if len(months) == 0:
        st.error("No months found in time column.")
        return

    # Build options list with month numbers and labels for nicer UI
    month_options = list(range(1,13))
    month_labels = [f"{m} — {MONTH_NAMES[m]}" for m in month_options]

    # Use a range slider (select_slider accepts a tuple) default to first month (1,1)
    default_val = (1, 1)
    month_range = st.select_slider(
        "Select month range (inclusive)",
        options=month_options,
        value=default_val,
        format_func=lambda x: MONTH_NAMES.get(x, str(x))
    )
    # If user chooses a single month, month_range will be an int — normalize to tuple
    if isinstance(month_range, int):
        month_range = (month_range, month_range)

    start_month, end_month = month_range

    # Filter dataframe for selected months
    df_sel = df[(df['time'].dt.month >= start_month) & (df['time'].dt.month <= end_month)].copy()
    if df_sel.empty:
        st.warning("No data for selected months.")
        return

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))

    # If All selected: normalize each numeric column for trend comparison, skip non-numeric
    if choice == "All":
        numeric_cols = [c for c in df_sel.columns if pd.api.types.is_numeric_dtype(df_sel[c]) and c != 'date']
        if not numeric_cols:
            st.error("No numeric columns to plot.")
            return

        # Normalize each series to 0-1 for trend comparison
        df_norm = df_sel.set_index('time')[numeric_cols].apply(lambda s: (s - s.min()) / (s.max() - s.min()))
        for col in df_norm.columns:
            ax.plot(df_norm.index, df_norm[col], label=col)
        ax.set_ylabel("Normalized value (0 - 1)")
    else:
        # Single column: plot actual values
        if not pd.api.types.is_numeric_dtype(df_sel[choice]):
            # If non-numeric (e.g. strings), display counts or a table instead
            st.warning(f"Column '{choice}' is not numeric. Showing value counts instead.")
            st.dataframe(df_sel[choice].value_counts().rename_axis(choice).reset_index(name="count"))
            return

        ax.plot(df_sel['time'], df_sel[choice], marker='o', linestyle='-')
        ax.set_ylabel(choice)

    ax.set_title(f"Data for months {start_month}–{end_month} ({MONTH_NAMES[start_month]} – {MONTH_NAMES[end_month]})")
    ax.set_xlabel("Time")
    ax.legend(loc='best', fontsize='small')
    ax.grid(True)
    fig.autofmt_xdate()
    st.pyplot(fig)

# -----------------------------
# Page: Extra placeholder
# -----------------------------
def page_extra() -> None:
    st.title("Extra / Placeholder")
    st.write("This is a placeholder page. Add your extra analysis, conclusions, or models here.")

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
