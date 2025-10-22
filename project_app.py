from secrets import choice
from typing import Callable, List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Part 2: MongoDB connection
# -----------------------------
import streamlit as st
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
# Page: Energy Production
# -----------------------------
# On page four, split the view into two columns using st.columns.
def page_extra() -> None:
    st.title("Energy Production in 2021 by Price Area and Production Group")

    col1, col2 = st.columns(2)

    # Left column: use radio buttons (st.radio) to select a price area and display a pie chart like in the Jupyter Notebook
    with col1:
        price_areas = collection.distinct("pricearea")  # get all unique price areas
        selected_area = st.radio("Select Price Area", price_areas)

        # show pie chart for selected price area in the left column
        data = list(collection.find({"pricearea": selected_area}))
        if data:
            df_area = pd.DataFrame(data)
            df_grouped = df_area.groupby("productiongroup")["quantitykwh"].sum().reset_index()

            import plotly.express as px
            fig = px.pie(
                df_grouped,
                values="quantitykwh",
                names="productiongroup",
                title=f"Total Production for {selected_area}"
            )
            st.plotly_chart(fig)
        else:
            st.write("No data found for this price area.")

    # On the right side, use pills (st.pills) to select which production groups to include and a selection element of your choice to select a month. Combine the price area, production group(s) and month, and display a line plot like in the Jupyter Notebook (but for any month).
    with col2:
        production_groups = collection.distinct("productiongroup")
        production_groups = [str(g) for g in production_groups]  # force strings
        selected_groups = st.pills("Select Production Groups", production_groups, default=production_groups)

        month = st.selectbox("Select Month", list(MONTH_NAMES.values()), index=0)
        month_num = [k for k,v in MONTH_NAMES.items() if v == month][0]

        # Query MongoDB with the selected filters
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
            df_filtered['starttime'] = pd.to_datetime(df_filtered['starttime'])
            df_grouped_time = df_filtered.groupby(['starttime', 'productiongroup'])['quantitykwh'].sum().reset_index()

            fig2 = px.line(
                df_grouped_time,
                x='starttime',
                y='quantitykwh',
                color='productiongroup',
                title=f"Production in {selected_area} for {month}"
            )
            st.plotly_chart(fig2)
        else:
            st.write("No data found for the selected filters.")

    # Below the columns, insert an expander (st.expander) where you briefly document the source of the data shown on the page.
    with st.expander("Data Source"):
        st.write("The data displayed on this page is sourced from the MongoDB database. The Elhub API was used to retrieve hourly production data for all price areas using PRODUCTION_PER_GROUP_MBA_HOUR for all days and hours of the year 2021.")

# -----------------------------
# Create st.Page objects and navigation
# -----------------------------
pg_home = st.Page(page_home, title="Home", icon="🏠")
pg_data = st.Page(page_data_table, title="Data", icon="📋")
pg_plots = st.Page(page_plots, title="Plots", icon="📈")
pg_extra = st.Page(page_extra, title="Energy Production", icon="⚙️")

# The navigation object builds the UI and runs the selected page.
nav = st.navigation(pages=[pg_home, pg_data, pg_plots, pg_extra])
nav.run()



