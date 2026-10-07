import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

import matplotlib.pyplot as plt

# CONFIGURATION & ADVANCED PROFESSIONAL UI STYLING

st.set_page_config( page_title="IdleSpot",layout="wide")

# Custom High-End Corporate Dashboard CSS
st.markdown("""
<style>
    /* Main Background - Modern Slate Graphite (Not Black, Not Blue, Not White) */
    .stApp {
        background-color: #121418 !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Sidebar Styling & High-Contrast Visibility Fix */
    [data-testid="stSidebar"] {
        background-color: #1a1d24 !important;
        border-right: 1px solid #2d323e !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        background-color: #242832 !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        margin-bottom: 6px !important;
        border: 1px solid #323846 !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        border-color: #10B981 !important;
        background-color: #2d3340 !important;
    }

    /* Cards & Containers Styling */
    div[data-testid="stMetric"], .stDataFrame, .element-container {
        border-radius: 12px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1c2029 0%, #171a21 100%) !important;
        border: 1px solid #2d3342 !important;
        padding: 18px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
        border-left: 4px solid #10B981 !important;
    }

    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* Vibrant Modern Gradient Header Text */
    h1 {
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 50%, #8B5CF6 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        letter-spacing: -1px !important;
    }

    h2, h3 {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
    }

    /* Professional Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(16, 185, 129, 0.4) !important;
    }

    /* Input Boxes & Select Boxes */
    .stTextInput input, .stSelectbox select {
        background-color: #1c2029 !important;
        color: #F8FAFC !important;
        border: 1px solid #323846 !important;
        border-radius: 8px !important;
    }

    .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #10B981 !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2) !important;
    }

    /* Dividers */
    hr {
        border-color: #272b36 !important;
    }

    /* Dataframe Table Custom Styling */
    .stDataFrame {
        background-color: #1c2029 !important;
        border: 1px solid #2d3342 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD RANDOM FOREST MODEL
# =========================================================

model = joblib.load("model.pkl")


# =========================================================
# LOAD LIVE DEVICE DATA
# =========================================================

def load_devices():

    with open("devices.json", "r") as file:

        return json.load(file)


# =========================================================
# CALCULATE HOURS PAST SHUTDOWN
# =========================================================

def calculate_hours_past_shutdown(shutdown_time):

    now = datetime.now()

    current_minutes = (
        now.hour * 60
        + now.minute
    )

    shutdown_hour, shutdown_minute = map(
        int,
        shutdown_time.split(":")
    )

    shutdown_minutes = (
        shutdown_hour * 60
        + shutdown_minute
    )

    difference = (
        current_minutes
        - shutdown_minutes
    )

    return difference / 60


# FORMAT RUNTIME
def format_runtime(hours):

    if hours <= 0:

        return "0m"

    total_minutes = int(
        hours * 60
    )

    hours_part = total_minutes // 60

    minutes_part = (
        total_minutes % 60
    )

    if hours_part > 0:

        return (
            f"{hours_part}h "
            f"{minutes_part}m"
        )

    return f"{minutes_part}m"


# PREDICT DEVICE
def predict_device(device):

    occupancy = device["occupancy"]

    active = int(device["power"] == "ON")

    hours_past_shutdown = (
        calculate_hours_past_shutdown(
            device["typical_shutdown"]
        )
    )

    live_data = pd.DataFrame([
        {
            "room_occupancy": occupancy,

            "hours_past_shutdown":
                hours_past_shutdown,

            "device_active":
                active
        }
    ])

    # DEVICE OFF

    if active == 0:

        return {

            "prediction": 0,

            "risk": 0,

            "hours_past_shutdown":
                hours_past_shutdown
        }

    # RANDOM FOREST PREDICTION

    prediction = model.predict(
        live_data
    )[0]

    probability = model.predict_proba(
        live_data
    )[0][1]

    risk = round(
        probability * 100,
        1
    )

    return {

        "prediction":
            int(prediction),

        "risk":
            risk,

        "hours_past_shutdown":
            hours_past_shutdown
    }

# GET ALL DEVICE ANALYSIS
def get_device_analysis(devices):

    results = []

    for device in devices:

        result = predict_device(
            device
        )

        if device["power"] == "OFF":

            status = "⚪ OFF"

        elif result["prediction"] == 1:

            status = "🚨 POSSIBLY FORGOTTEN"

        else:

            status = "🟢 NORMAL"

        results.append({

            "Device":
                device["device"],

            "Location":
                device["room"],

            "Power":
                device["power"],

            "Occupancy":
                device["occupancy"],

            "Runtime":
                format_runtime(
                    max(
                        0,
                        result[
                            "hours_past_shutdown"
                        ]
                    )
                ),

            "ML Risk":
                result["risk"],

            "Status":
                status,

            "prediction":
                result["prediction"]
        })

    return pd.DataFrame(
        results
    )

# REAL RANDOM FOREST FEATURE INFLUENCE
def show_feature_importance():

    st.subheader(
        "Feature Influence"
    )

    st.caption(
        "Feature importance from the trained "
        "Random Forest model."
    )

    feature_names = [

        "room_occupancy",

        "hours_past_shutdown",

        "device_active"
    ]

    feature_importances = (
        model.feature_importances_
    )

    feature_data = pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            feature_importances

    })

    feature_data = (
        feature_data
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    for _, row in feature_data.iterrows():

        feature = row["Feature"]

        importance = float(
            row["Importance"]
        )

        percentage = round(
            importance * 100,
            1
        )

        st.write(
            f"**{feature}**"
        )

        st.progress(
            importance
        )

        st.caption(
            f"{percentage}%"
        )

# MODEL EVALUATION
def evaluate_model():

    np.random.seed(42)

    n = 2000

    room_occupancy = np.random.randint(
        0,
        31,
        n
    )

    hours_past_shutdown = np.random.uniform(
        -3,
        5,
        n
    )

    device_active = np.random.choice(
        [0, 1],
        n,
        p=[0.20, 0.80]
    )

    risk_score = (

        (room_occupancy == 0) * 4

        + (hours_past_shutdown > 0) * 3

        + (hours_past_shutdown > 1) * 2

        + (device_active == 1) * 2
    )

    noise = np.random.randint(
        -1,
        2,
        n
    )

    risk_score = (
        risk_score
        + noise
    )

    forgotten = (
        risk_score >= 7
    ).astype(int)

    evaluation_df = pd.DataFrame({

        "room_occupancy":
            room_occupancy,

        "hours_past_shutdown":
            hours_past_shutdown,

        "device_active":
            device_active,

        "forgotten":
            forgotten
    })

    X = evaluation_df.drop(
        columns=["forgotten"]
    )

    y = evaluation_df["forgotten"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    # Use already trained Random Forest

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        cm
    )


# CHATBOT
def chatbot(
    question,
    devices
):

    q = question.lower().strip()

    # DEVICE QUESTION
    for device in devices:

        device_name = (
            device["device"].lower()
        )

        if device_name in q:

            result = predict_device(
                device
            )

            if device["power"] == "OFF":

                return (

                    f"⚪ {device['device']} "
                    f"in {device['room']} "
                    f"is currently OFF.\n\n"

                    f"No action is required."
                )

            status = (

                "🚨 Possibly forgotten"

                if result["prediction"] == 1

                else "🟢 Normal"
            )

            return (

                f"🔌 {device['device']}\n\n"

                f"Location: "
                f"{device['room']}\n\n"

                f"Power: "
                f"{device['power']}\n\n"

                f"Room occupancy: "
                f"{device['occupancy']}\n\n"

                f"Runtime after shutdown: "
                f"{format_runtime(max(0, result['hours_past_shutdown']))}\n\n"

                f"ML risk: "
                f"{result['risk']}%\n\n"

                f"Status: "
                f"{status}"
            )

    # =====================================================
    # RUNNING DEVICES
    # =====================================================

    if (
        "what" in q
        and (
            "running" in q
            or "on" in q
            or "active" in q
        )
    ):

        running = []

        for device in devices:

            if device["power"] == "ON":

                result = predict_device(
                    device
                )

                status = (

                    "🚨 Risk"

                    if result["prediction"] == 1

                    else "🟢 Normal"
                )

                running.append(

                    f"• {device['device']} "
                    f"({device['room']}) — "
                    f"{result['risk']}% — "
                    f"{status}"
                )

        if not running:

            return (
                "No monitored devices are "
                "currently ON."
            )

        return (
            "Currently running devices:\n\n"
            + "\n".join(running)
        )

    # FORGOTTEN DEVICES
    if (
        "forgotten" in q
        or "alert" in q
        or "risk" in q
        or "danger" in q
    ):

        risky = []

        for device in devices:

            if device["power"] == "ON":

                result = predict_device(
                    device
                )

                if result["prediction"] == 1:

                    risky.append(

                        f"🚨 {device['device']} "
                        f"({device['room']}) — "
                        f"{result['risk']}%"
                    )

        if not risky:

            return (
                "✅ No devices currently "
                "appear to be forgotten."
            )

        return (
            "🚨 Potentially forgotten devices:\n\n"
            + "\n".join(risky)
        )

    # STATUS
    if (
        "status" in q
        or "summary" in q
    ):

        total = len(devices)

        running = sum(

            1

            for device in devices

            if device["power"] == "ON"
        )

        off = (
            total
            - running
        )

        empty = sum(

            1

            for device in devices

            if device["occupancy"] == 0
        )

        risky = 0

        for device in devices:

            result = predict_device(
                device
            )

            if result["prediction"] == 1:

                risky += 1

        return (

            f"📊 IdleSpot Summary\n\n"

            f"Devices monitored: "
            f"{total}\n\n"

            f"Currently ON: "
            f"{running}\n\n"

            f"Currently OFF: "
            f"{off}\n\n"

            f"Empty rooms: "
            f"{empty}\n\n"

            f"Potentially forgotten: "
            f"{risky}"
        )

    return (

        "Ask me something like:\n\n"

        "• Is the AC ON?\n"

        "• Is the projector ON?\n"

        "• What devices are running?\n"

        "• Which devices may be forgotten?\n"

        "• Are there any alerts?\n"

        "• Give me the current status."
    )


# HEADER

st.title("⚡ IdleSpot Enterprise")

st.subheader("Detect it before it gets forgotten.")

st.caption(
    "🟢 Live · Simulated telemetry · "
    "ML: Random Forest"
)

# LOAD DEVICES
devices = load_devices()

# DASHBOARD ANALYSIS
analysis_df = get_device_analysis(devices)

devices_monitored = len(devices)

rooms = set(
    device["room"]
    for device in devices
)

at_risk_devices = len(
    analysis_df[
        analysis_df["prediction"] == 1
    ]
)

empty_rooms = len(
    set(
        device["room"]
        for device in devices
        if device["occupancy"] == 0
    )
)

active_devices = sum(

    1

    for device in devices

    if device["power"] == "ON"
)

active_alerts = at_risk_devices


# TOP METRICS

metric1, metric2, metric3, metric4 = (
    st.columns(4)
)


with metric1:

    st.metric(
        "Devices Monitored",
        devices_monitored,
        f"across {len(rooms)} rooms"
    )


with metric2:

    st.metric(
        "At-Risk Devices",
        at_risk_devices,
        "needs attention"
    )


with metric3:

    st.metric(
        "Empty Rooms",
        empty_rooms,
        "occupancy = 0"
    )


with metric4:

    st.metric(
        "Active Alerts",
        active_alerts,
        "unacknowledged"
    )


st.divider()


# SIDEBAR

st.sidebar.header("Monitoring Mode")


mode = st.sidebar.radio(
    label="",
    options=[
        "Dashboard",
        "Manual Prediction",
        "Automatic Live Monitoring",
        "Model Evaluation & Live Analytics"
    ],
    label_visibility="collapsed",
)


st.sidebar.divider()


st.sidebar.write("### System")

st.sidebar.write("🟢 Model: Random Forest")

st.sidebar.write("🟢 Telemetry")

st.sidebar.write(
    f"🟢 Devices ON: "
    f"{active_devices}"
)

st.sidebar.write(
    f"⚪ Devices OFF: "
    f"{devices_monitored - active_devices}"
)

st.sidebar.write(
    f"🕐 "
    f"{datetime.now().strftime('%I:%M:%S %p')}"
)

# DASHBOARD
if mode == "Dashboard":

    st.header(
        "Device telemetry"
    )

    st.caption(
        f"Telemetry · "
        f"{datetime.now().strftime('%I:%M:%S %p')}"
    )

    # LIVE DEVICE TABLE
    display_df = analysis_df[
        [
            "Device",
            "Location",
            "Power",
            "Occupancy",
            "Runtime",
            "ML Risk",
            "Status"
        ]
    ].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # DEVICE RISK ANALYSIS

    st.header("📊 Device Risk Analysis")

    show_feature_importance()

    st.divider()

    # CURRENT DEVICE STATUS
    st.subheader("Device Status")

    current_devices = load_devices()

    on_devices = [

        device

        for device in current_devices

        if device["power"] == "ON"
    ]

    off_devices = [

        device

        for device in current_devices

        if device["power"] == "OFF"
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "### 🟢 Currently ON"
        )

        if on_devices:

            for device in on_devices:

                result = predict_device(
                    device
                )

                if result["prediction"] == 1:

                    st.error(

                        f"🚨 {device['device']} · "
                        f"{device['room']} · "
                        f"Risk {result['risk']}%"
                    )

                else:

                    st.success(

                        f"🟢 {device['device']} · "
                        f"{device['room']} · "
                        f"Risk {result['risk']}%"
                    )

        else:

            st.info(
                "No monitored devices are ON."
            )

    with col2:

        st.write(
            "### ⚪ Currently OFF"
        )

        if off_devices:

            for device in off_devices:

                st.info(

                    f"⚪ {device['device']} · "
                    f"{device['room']} · OFF"
                )

        else:

            st.info(
                "No monitored devices are OFF."
            )

# AUTOMATIC LIVE MONITORING
elif mode == "Automatic Live Monitoring":

    st.header("🔴 Automatic Live Monitoring")

    st.write("IdleSpot reads the Telemetry "
        "from devices.json and predicts every device."
    )

    auto_refresh = st.checkbox(
        "Enable automatic refresh",
        value=False
    )

    if st.button(
        "🔄 CHECK ALL DEVICES",
        use_container_width=True
    ):

        live_devices = load_devices()

        live_analysis = get_device_analysis(
            live_devices
        )

        display_df = live_analysis[
            [
                "Device",
                "Location",
                "Power",
                "Occupancy",
                "Runtime",
                "ML Risk",
                "Status"
            ]
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        current_on = sum(

            1

            for device in live_devices

            if device["power"] == "ON"
        )

        current_off = (

            len(live_devices)
            - current_on
        )

        current_risk = len(

            live_analysis[
                live_analysis["prediction"] == 1
            ]
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "🟢 Devices ON",
                current_on
            )

        with col2:

            st.metric(
                "⚪ Devices OFF",
                current_off
            )

        with col3:

            st.metric(
                "🚨 ML Risk",
                current_risk
            )

        if current_risk > 0:

            st.error(

                f"🚨 {current_risk} device(s) "
                "may be forgotten."
            )

            risky_devices = (
                live_analysis[
                    live_analysis["prediction"] == 1
                ]
            )

            for _, row in risky_devices.iterrows():

                st.warning(

                    f"🚨 {row['Device']} · "
                    f"{row['Location']} · "
                    f"Risk {row['ML Risk']}%"
                )

        else:

            st.success(

                "✅ No devices currently "
                "appear to be forgotten."
            )

        st.caption(

            "Last prediction: "

            f"{datetime.now().strftime('%I:%M:%S %p')}"
        )

    if auto_refresh:

        st.info(

            "🟢 Telemetry monitoring active — "
            "checking latest data every 5 seconds."
        )

        time.sleep(5)

        st.rerun()


# MANUAL PREDICTION
elif mode == "Manual Prediction":

    st.header("🎛 Manual Device Check")

    latest_devices = load_devices()

    device_names = [

        device["device"]

        for device in latest_devices
    ]

    selected_device = st.selectbox(
        "Select Device",
        device_names
    )

    selected = next(

        device

        for device in latest_devices

        if device["device"]
        == selected_device
    )

    st.write(
        f"**Location:** "
        f"{selected['room']}"
    )

    st.write(
        f"**Power:** "
        f"{selected['power']}"
    )

    st.write(
        f"**Occupancy:** "
        f"{selected['occupancy']}"
    )

    st.write(
        f"**Typical shutdown:** "
        f"{selected['typical_shutdown']}"
    )

    if st.button(
        "🔍 PREDICT DEVICE",
        use_container_width=True
    ):

        latest_devices = load_devices()

        selected = next(

            device

            for device in latest_devices

            if device["device"]
            == selected_device
        )

        result = predict_device(
            selected
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Power",
                selected["power"]
            )

        with col2:

            st.metric(
                "ML Risk",
                f"{result['risk']}%"
            )

        with col3:

            st.metric(
                "Runtime",
                format_runtime(
                    max(
                        0,
                        result[
                            "hours_past_shutdown"
                        ]
                    )
                )
            )

        if selected["power"] == "OFF":

            st.success(

                f"⚪ {selected['device']} "
                "is OFF. "
                "No action required."
            )

        elif result["prediction"] == 1:

            st.error(

                f"🚨 {selected['device']} "
                "may be forgotten!"
            )

            st.warning(

                "Recommended action: "
                "Check or switch OFF the device."
            )

        else:

            st.success(

                f"🟢 {selected['device']} "
                "appears to be in normal use."
            )

# MODEL EVALUATION & LIVE ANALYTICS
else:

    st.header("📈 Model Evaluation & Live Analytics")

    st.caption(
        "Separate evaluation page for the trained "
        "Random Forest model."
    )

    # EVALUATE MODEL

    (
        accuracy,
        precision,
        recall,
        f1,
        cm
    ) = evaluate_model()

    # METRICS
    st.subheader("Random Forest Evaluation")

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    st.divider()
    # CONFUSION MATRIX HEATMAP
    st.subheader("Confusion Matrix")

    st.caption("0 = Normal device, 1 = Possibly forgotten")

    fig, ax = plt.subplots(figsize=(6, 4))

    image = ax.imshow(cm,interpolation="nearest")

    ax.set_title("Random Forest Confusion Matrix")

    ax.set_xlabel("Predicted Label")

    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])

    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Normal", "Forgotten"])

    ax.set_yticklabels(["Normal", "Forgotten"])

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    # METRIC LINE PLOT
    st.divider()

    st.subheader("Model Performance Line Plot")

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    metric_values = [

        accuracy * 100,

        precision * 100,

        recall * 100,

        f1 * 100
    ]

    fig2, ax2 = plt.subplots(
        figsize=(8, 4)
    )

    ax2.plot(
        metric_names,
        metric_values,
        marker="o",
        linewidth=2
    )

    ax2.set_ylim(
        0,
        100
    )

    ax2.set_ylabel(
        "Score (%)"
    )

    ax2.set_title(
        "Random Forest Evaluation Metrics"
    )

    ax2.grid(
        True,
        alpha=0.3
    )

    for i, value in enumerate(
        metric_values
    ):

        ax2.text(
            i,
            value + 2,
            f"{value:.2f}%",
            ha="center"
        )

    st.pyplot(
        fig2,
        use_container_width=True
    )

    plt.close(fig2)
    # LIVE DEVICE RISK LINE
    st.divider()

    st.subheader("🔴 Live Device Risk")

    st.caption(
        "Current risk calculated from the latest "
        "devices.json telemetry."
    )

    live_devices = load_devices()

    live_results = []

    for device in live_devices:

        result = predict_device(
            device
        )

        live_results.append({

            "Device":
                device["device"],

            "Risk":
                result["risk"]
        })

    live_risk_df = pd.DataFrame(
        live_results
    )

    live_risk_df = (
        live_risk_df
        .set_index("Device")
    )

    st.line_chart(
        live_risk_df
    )

    st.caption(
        "The chart updates when the application "
        "reruns and reads the latest telemetry."
    )

    # LIVE ON / OFF STATUS
    st.divider()

    st.subheader("⚡ Current Device State")

    current_on = sum(

        1

        for device in live_devices

        if device["power"] == "ON"
    )

    current_off = (
        len(live_devices)
        - current_on
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        st.metric(
            "🟢 Devices ON",
            current_on
        )

    with col2:

        st.metric(
            "⚪ Devices OFF",
            current_off
        )

    st.caption(
        "Telemetry read: "
        f"{datetime.now().strftime('%I:%M:%S %p')}"
    )

# CHATBOT
st.divider()

st.header("🤖 Ask IdleSpot")

st.caption("Ask about the current monitored device state.")

question = st.text_input(
    "Ask your question",
    placeholder="Example: Is the AC ON?"
)

if question:

    answer = chatbot(
        question,
        load_devices()
    )

    st.info(answer)