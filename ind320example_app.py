import streamlit as st
import pandas as pd

st.title("IND320 - Minimum Working Example 🚀")

st.write("Hello! This is my first Streamlit app for IND320.")

# Example: load a simple dataframe
data = {
    "Month": ["Jan", "Feb", "Mar", "Apr"],
    "Value": [10, 20, 15, 30]
}
df = pd.DataFrame(data)

st.write("Here’s a small test dataframe:")
st.dataframe(df)

# Make a simple line chart with the months from jan-apr on the x-axis and the values on the y-axis
st.write("And here’s a simple line chart:")
st.line_chart(df.set_index("Month"))
