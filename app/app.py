import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Middle East Conflict Forecast", layout="wide")
st.title("Conflict Risk Forecast — Middle East")

parquet_path = "data/processed/latest_forecast.parquet"

if not os.path.exists(parquet_path):
    st.error(f"File not found at `{parquet_path}`. Run `python src/generate_dashboard_data.py` from your root directory first.")
    st.stop()

try:
    with st.spinner("Loading dataset..."):
        data = pd.read_parquet(parquet_path)
    st.success(f"Dataset loaded successfully! Shape: {data.shape}")
except Exception as e:
    st.error(f"Error reading Parquet file: {e}")
    st.stop()

tab_overview, tab_map, tab_country, tab_table = st.tabs(
    ["Overview", "Map", "Country Analysis", "Forecast Table"]
)

with tab_overview:
    col1, col2 = st.columns(2)
    col1.metric("Total Recent Fatalities", int(data["FATALITIES"].sum()))
    col2.metric("Total Recent Events", int(data["EVENTS"].sum()))
    st.plotly_chart(px.line(data, x="WEEK", y="FATALITIES", color="COUNTRY"), use_container_width=True)

with tab_map:
    latest_snapshot = data.groupby("COUNTRY", as_index=False)["predicted_fatalities"].last()
    fig = px.choropleth(
        latest_snapshot,
        locations="COUNTRY",
        locationmode="country names",
        color="predicted_fatalities",
        color_continuous_scale="Reds",
        title="Latest Predicted Fatalities by Country"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_country:
    country = st.selectbox("Select Country", sorted(data["COUNTRY"].unique()))
    cdf = data[data["COUNTRY"] == country]
    st.plotly_chart(px.line(cdf, x="WEEK", y=["FATALITIES", "predicted_fatalities"], title=f"Actual vs Predicted in {country}"), use_container_width=True)

with tab_table:
    st.dataframe(
        data.groupby("COUNTRY", as_index=False)
            .last()[["COUNTRY", "predicted_fatalities", "FATALITIES"]]
            .sort_values("predicted_fatalities", ascending=False),
        use_container_width=True
    )