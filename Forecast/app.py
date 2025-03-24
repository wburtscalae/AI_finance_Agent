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
st.set_page_config(page_title="📈 Forecasting Agent by Customer & Product", page_icon="📊", layout="wide")
st.title("📈 Forecasting with Prophet + AI Commentary")
st.markdown("Upload a dataset with `Date`, `Revenue`, and optionally `Customer` and `Product` columns.")

uploaded_file = st.file_uploader("📤 Upload your Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Validate columns
        if 'Date' not in df.columns or 'Revenue' not in df.columns:
            st.error("❌ File must contain at least 'Date' and 'Revenue' columns.")
            st.stop()

        # Optional dimensions
        has_customer = 'Customer' in df.columns
        has_product = 'Product' in df.columns

        # Dimension selector
        group_options = ['Overall']
        if has_customer:
            group_options.append('Customer')
        if has_product:
            group_options.append('Product')
        if has_customer and has_product:
            group_options.append('Customer & Product')

        selected_group = st.selectbox("🔀 Select breakdown", group_options)

        # Group logic
        def get_group_keys(row):
            if selected_group == 'Overall':
                return ('All',)
            elif selected_group == 'Customer':
                return (row['Customer'],)
            elif selected_group == 'Product':
                return (row['Product'],)
            elif selected_group == 'Customer & Product':
                return (row['Customer'], row['Product'])

        df['GroupKey'] = df.apply(get_group_keys, axis=1)

        grouped = df.groupby('GroupKey')

        client = Groq(api_key=GROQ_API_KEY)

        for key, group_df in grouped:
            st.markdown(f"---\n### 📦 Forecast for {selected_group}: `{key}`")

            group_df = group_df[['Date', 'Revenue']].copy()
            group_df = group_df.rename(columns={'Date': 'ds', 'Revenue': 'y'})
            group_df['ds'] = pd.to_datetime(group_df['ds'])

            if len(group_df) < 3:
                st.warning("⚠️ Not enough data to forecast.")
                continue

            # Forecasting
            model = Prophet()
            model.fit(group_df)

            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)

            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig, use_container_width=True)

            # Forecast Table
            forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)
            st.dataframe(forecast_table)

            # AI Commentary
            st.subheader("🧠 AI Commentary")
            context = ""
            if selected_group == 'Customer':
                context = f"for Customer: {key[0]}"
            elif selected_group == 'Product':
                context = f"for Product: {key[0]}"
            elif selected_group == 'Customer & Product':
                context = f"for Customer: {key[0]} and Product: {key[1]}"

            prompt = f"""
            You are the Head of FP&A at a SaaS or retail company. Below is the forecasted revenue data {context}.
            Analyze this data and provide:
            - Key trends and changes vs. history
            - Risk areas or growth opportunities
            - A CFO-ready executive summary (use Pyramid Principle)
            - Recommendations to act on

            Forecast data:
            {forecast_table.to_json(orient="records")}
            """

            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a financial forecasting expert who writes sharp and concise summaries for executives."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
            )

            ai_commentary = response.choices[0].message.content
            st.markdown(ai_commentary)

    except Exception as e:
        st.error(f"❌ Error: {e}")
