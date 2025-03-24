import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# 🎨 Streamlit UI Styling
st.set_page_config(page_title="📈 AI Forecasting Tool", page_icon="📊", layout="wide")

st.title("📈 Forecasting with Prophet + AI Commentary")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your Excel file (with Date and Revenue columns)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        if 'Date' not in df.columns or 'Revenue' not in df.columns:
            st.error("❌ Excel file must contain 'Date' and 'Revenue' columns.")
            st.stop()

        # Preprocess data
        df = df[['Date', 'Revenue']].copy()
        df = df.rename(columns={'Date': 'ds', 'Revenue': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])

        # Forecasting
        st.subheader("📊 Prophet Forecast")
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig, use_container_width=True)

        # Show forecast table
        st.subheader("📋 Forecasted Values")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

        # 🎯 AI Commentary
        st.subheader("🧠 AI-Generated Forecast Commentary")

        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""
        You are the Head of FP&A at a SaaS company. Your task is to analyze the following revenue forecast data and provide:
        - Key insights from the projected values.
        - Areas of risk or opportunity.
        - A CFO-ready summary using the Pyramid Principle.
        - Actionable recommendations based on the trends.

        Forecast data:
        {forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).to_json(orient="records")}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial planning and analysis (FP&A) expert, specializing in SaaS companies."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )

        ai_commentary = response.choices[0].message.content
        st.markdown("### 📖 Forecast Commentary")
        st.write(ai_commentary)

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
