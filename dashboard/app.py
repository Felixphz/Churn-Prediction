import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

API_URL = "http://api:8000"

st.title("Churn Prediction Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Customers", "Predictions", "Pipeline"])

if page == "Home":
    st.header("Overview")
    try:
        res = requests.get(f"{API_URL}/api/v1/dashboard/summary", timeout=10)
        if res.status_code == 200:
            data = res.json()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Customers", data["total_customers"])
            col2.metric("Churn Count", data["churn_count"])
            col3.metric("Churn Rate", f"{data['churn_rate']}%")
            col4.metric("Active Model", data["active_model"] or "N/A")
        else:
            st.error("Could not connect to API")
    except Exception as e:
        st.error(f"API connection error: {e}")

elif page == "Customers":
    st.header("Customers")
    try:
        res = requests.get(f"{API_URL}/api/v1/customers/", timeout=10)
        if res.status_code == 200:
            data = res.json()
            df = pd.DataFrame(data["customers"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Total: {data['total']} customers")
    except Exception as e:
        st.error(f"API connection error: {e}")

elif page == "Predictions":
    st.header("Make Prediction")
    customer_id = st.text_input("Customer ID (e.g., 7590-VHVEG)")
    threshold = st.slider("Threshold", 0.0, 1.0, 0.3, 0.05)

    if st.button("Predict"):
        if customer_id:
            try:
                res = requests.post(
                    f"{API_URL}/api/v1/predictions/predict",
                    json={"customer_id": customer_id, "threshold": threshold},
                    timeout=30,
                )
                if res.status_code == 200:
                    result = res.json()
                    pred = result["prediction"]
                    st.success(f"Churn: {'Yes' if pred['churn_predicted'] else 'No'}")
                    st.metric("Probability", f"{pred['churn_probability']:.4f}")
                    st.json(result["explainability"]["top_features"])
                else:
                    st.error(res.json().get("detail", "Prediction failed"))
            except Exception as e:
                st.error(f"Error: {e}")

elif page == "Pipeline":
    st.header("ML Pipeline")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Batch Predictions")
        if st.button("Run Batch"):
            with st.spinner("Running batch predictions..."):
                try:
                    res = requests.post(f"{API_URL}/api/v1/predictions/batch", timeout=120)
                    if res.status_code == 200:
                        st.success("Batch complete!")
                        st.json(res.json())
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.subheader("Retrain Model")
        if st.button("Trigger Retrain"):
            with st.spinner("Training in background..."):
                try:
                    res = requests.post(f"{API_URL}/api/v1/retrain/trigger", timeout=10)
                    if res.status_code == 200:
                        st.success("Retrain triggered!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.subheader("Retrain History")
    try:
        res = requests.get(f"{API_URL}/api/v1/retrain/history?limit=5", timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data["runs"]:
                df = pd.DataFrame(data["runs"])
                st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")
