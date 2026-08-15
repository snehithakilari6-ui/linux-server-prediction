import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Linux Server Prediction",
    page_icon="🖥️",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------

st.title("🖥️ Linux Server Response Time Prediction")

st.write(
    "Machine Learning based Linux Server Response Time Prediction "
    "using Random Forest Regressor"
)

st.divider()

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("cleaned_linux_server_dataset.csv")

# -----------------------------
# Features and Target
# -----------------------------

features = [
    "CPU_Usage",
    "RAM_Usage",
    "Disk_Usage",
    "Network_Traffic",
    "Active_Users",
    "Running_Processes",
    "Server_Uptime",
    "Error_Count"
]

X = df[features]
y = df["Response_Time"]

# -----------------------------
# Train Model
# -----------------------------

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Server Details")

cpu = st.sidebar.number_input(
    "CPU Usage (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

ram = st.sidebar.number_input(
    "RAM Usage (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

disk = st.sidebar.number_input(
    "Disk Usage (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

network = st.sidebar.number_input(
    "Network Traffic",
    min_value=0.0,
    value=400.0
)

users = st.sidebar.number_input(
    "Active Users",
    min_value=0.0,
    value=50.0
)

processes = st.sidebar.number_input(
    "Running Processes",
    min_value=0.0,
    value=120.0
)

uptime = st.sidebar.number_input(
    "Server Uptime",
    min_value=0.0,
    value=300.0
)

errors = st.sidebar.number_input(
    "Error Count",
    min_value=0.0,
    value=5.0
)

# -----------------------------
# Prediction Button
# -----------------------------

if st.sidebar.button("Predict Response Time"):

    new_server = pd.DataFrame({
        "CPU_Usage": [cpu],
        "RAM_Usage": [ram],
        "Disk_Usage": [disk],
        "Network_Traffic": [network],
        "Active_Users": [users],
        "Running_Processes": [processes],
        "Server_Uptime": [uptime],
        "Error_Count": [errors]
    })

    prediction = model.predict(new_server)[0]

    # Server Status
    if prediction < 300:
        status = "🟢 Normal"
    elif prediction < 520:
        status = "🟡 Warning"
    else:
        status = "🔴 Critical"

    # -----------------------------
    # Results
    # -----------------------------

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Response Time",
            f"{prediction:.2f}"
        )

    with col2:
        st.metric(
            "Server Status",
            status
        )

    st.success("Prediction completed successfully!")

# -----------------------------
# Dataset Information
# -----------------------------

st.divider()

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Total Features", len(features))

with col3:
    st.metric("Target", "Response Time")

# -----------------------------
# Dataset Preview
# -----------------------------

st.subheader("Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)