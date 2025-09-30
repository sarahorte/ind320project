import streamlit as st

# ---- Define your pages ----
# Each page can be a function or an external file
def home():
    st.title("Home")
    st.write("Welcome to my app! Use the sidebar to navigate.")

def data_page():
    st.title("Data Page")
    st.write("Here we'll show the imported CSV and charts in table form.")

def plot_page():
    st.title("Plot Page")
    st.write("Here we'll plot the imported data with dropdowns and sliders.")

def about_page():
    st.title("About Page")
    st.write("Some test content or info about the project.")

# ---- Page declarations ----
pg_home = st.Page(home, title="Home", icon="🏠")
pg_data = st.Page(data_page, title="Data", icon="📊")
pg_plot = st.Page(plot_page, title="Plots", icon="📈")
pg_about = st.Page(about_page, title="About", icon="ℹ️")

# ---- Navigation ----
# Sidebar menu with navigation
nav = st.navigation(pages=[pg_home, pg_data, pg_plot, pg_about])
nav.run()
