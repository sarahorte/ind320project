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

st.line_chart(df.set_index("Month"))
