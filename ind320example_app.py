import streamlit as st
import pandas as pd

st.title("IND320 - Minimum Working Example 🚀")

# Create the dataframe
data = {
    "Month": ["Jan", "Feb", "Mar", "Apr"],
    "Value": [10, 20, 15, 30]
}
df = pd.DataFrame(data)

# Make 'Month' a categorical variable with the correct order
df['Month'] = pd.Categorical(df['Month'], categories=["Jan", "Feb", "Mar", "Apr"], ordered=True)

# Set Month as the index
df_sorted = df.set_index("Month").sort_index()

st.write("Here’s a small test dataframe:")
st.dataframe(df_sorted)

st.write("And here’s a simple line chart:")
st.line_chart(df_sorted)
