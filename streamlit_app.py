import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import random

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Inventory SaaS", layout="wide", page_icon="📦")

st.markdown("### 🚀 AI Inventory Intelligence Platform")

# ---------------- NAVBAR ----------------
selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Analytics", "About"],
    icons=["bar-chart", "graph-up", "info-circle"],
    orientation="horizontal"
)

# ---------------- DATA ----------------
stores = {
    "Mumbai Central": 1,
    "Delhi NCR": 2,
    "Bangalore Hub": 3,
    "Hyderabad Zone": 4,
    "Chennai Store": 5
}

families = {
    "Beverages": 1,
    "Dairy": 2,
    "Snacks": 3,
    "Frozen Foods": 4,
    "Household": 5,
    "Personal Care": 6
}

products = {
    "Coca Cola 500ml": ("Beverages", 1),
    "Pepsi 1L": ("Beverages", 2),
    "Amul Milk 1L": ("Dairy", 3),
    "Lays Chips": ("Snacks", 4),
    "Frozen Peas": ("Frozen Foods", 5),
    "Surf Excel": ("Household", 6),
    "Shampoo": ("Personal Care", 7)
}

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Controls")

    store = st.selectbox("🏪 Store", list(stores.keys()))
    product = st.selectbox("🛒 Product", list(products.keys()))

    family_name, product_id = products[product]
    st.markdown(f"📦 Category: **{family_name}**")

    base_sales = st.slider("📊 Base Sales", 50, 500, 120)

    # 🔥 Lag generator
    def generate_lags(base):
        return {
            "lag_1": base + random.randint(-10, 10),
            "lag_2": base + random.randint(-15, 15),
            "lag_3": base + random.randint(-20, 20),
            "lag_7": base + random.randint(-25, 25),
            "lag_14": base + random.randint(-30, 30),
            "lag_21": base + random.randint(-35, 35),
            "lag_28": base + random.randint(-40, 40),
        }

    lags = generate_lags(base_sales)

    st.markdown("### 📉 Sales History")
    st.dataframe(pd.DataFrame([lags]))

    if st.button("🔄 Refresh Sales"):
        st.rerun()

    data = {
        "id": product_id,
        "store_nbr": stores[store],
        "family": families[family_name],
        "onpromotion": st.selectbox("🔥 Promotion", [0,1]),
        "day_of_week": st.slider("📅 Day", 1,7, 2),
        "month": st.slider("📆 Month", 1,12, 3),
        "week": st.slider("📅 Week", 1,52, 10),
        "is_weekend": 0,

        **lags,

        "rolling_mean_7": base_sales,
        "rolling_mean_14": base_sales + 5,
        "rolling_std_7": random.randint(5,15)
    }

    if data["day_of_week"] in [6,7]:
        data["is_weekend"] = 1

# ---------------- FEATURE ORDER ----------------
FEATURE_ORDER = [
    "id","store_nbr","family","onpromotion",
    "day_of_week","month","week","is_weekend",
    "lag_1","lag_2","lag_3","lag_7","lag_14","lag_21","lag_28",
    "rolling_mean_7","rolling_mean_14","rolling_std_7"
]

data = {k: data[k] for k in FEATURE_ORDER}

# ---------------- DASHBOARD ----------------
if selected == "Dashboard":

    st.markdown("## 📦 Inventory Forecasting Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Sales (Yesterday)", data["lag_1"])
    with c2:
        st.metric("Last Week Sales", data["lag_7"])
    with c3:
        st.metric("Store", store)
    with c4:
        st.metric("Promo", f"{data['onpromotion']*100}%")

    st.markdown("---")

    col1, col2 = st.columns([1,1.2])

    with col1:
        st.markdown("### 📥 Input Data")
        st.dataframe(pd.DataFrame([data]))

    with col2:
        st.markdown("### 🤖 Prediction Engine")

        if st.button("🚀 Predict Demand"):

            try:
                res = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json=data
                ).json()

                if "prediction" in res:
                    pred = res["prediction"]

                    st.success(f"Predicted Demand: {pred:.2f}")

                    # 🔥 Business Insight
                    if pred > data["lag_7"]:
                        st.success("🚀 Higher than last week")
                    else:
                        st.warning("⚠️ Lower than last week")

                    # 🔥 Trend
                    trend = "📈 Increasing" if pred > data["lag_1"] else "📉 Decreasing"
                    st.info(f"Trend: {trend}")

                    # 🔥 Confidence (UI only)
                    st.progress(0.85)
                    st.caption("Model Confidence: 85%")

                    # 🔥 Chart
                    labels = [
                        "Yesterday","2 Days Ago","3 Days Ago",
                        "Last Week","2 Weeks Ago","3 Weeks Ago","4 Weeks Ago"
                    ]

                    values = [
                        data["lag_1"], data["lag_2"], data["lag_3"],
                        data["lag_7"], data["lag_14"],
                        data["lag_21"], data["lag_28"]
                    ]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=labels, y=values, mode='lines+markers', name="Past Sales"))
                    fig.add_trace(go.Scatter(x=labels, y=[pred]*7, line=dict(dash="dash"), name="Prediction"))
                    fig.update_layout(template="plotly_dark")

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(res)

            except:
                st.error("⚠️ Start FastAPI backend")

# ---------------- ANALYTICS ----------------
elif selected == "Analytics":

    st.markdown("## 📊 Analytics")

    values = [
        data["lag_1"], data["lag_2"], data["lag_3"],
        data["lag_7"], data["lag_14"],
        data["lag_21"], data["lag_28"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Yesterday","2D","3D","1W","2W","3W","4W"],
        y=values
    ))

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

# ---------------- ABOUT ----------------
else:
    st.markdown("## ℹ️ About")
    st.write("AI-powered inventory forecasting system 🚀")