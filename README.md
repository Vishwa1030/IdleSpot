# IdleSpot AI

> **Detect it before it gets forgotten.**

IdleSpot is a machine-learning-based device monitoring system that analyzes live device telemetry and predicts whether an active device is likely to be forgotten.

The system monitors parameters such as:

- Room occupancy
- Device power state
- Hours past the typical shutdown time

A **Random Forest Classifier** analyzes these features and predicts whether a device is:

- 🟢 Normal
- 🚨 Possibly Forgotten
- ⚪ OFF

The application provides a real-time Streamlit dashboard, automatic monitoring, manual device prediction, ML risk scores, feature importance, alerts, and a chatbot for querying the current device state.

---

## Project Overview

In offices, laboratories, classrooms, and meeting rooms, devices such as ACs, projectors, PCs, monitors, and chargers may remain powered on after people leave.

This can result in:

- Unnecessary electricity consumption
- Increased operational costs
- Hardware wear
- Energy wastage
- Manual monitoring effort

IdleSpot aims to identify these situations automatically.

Instead of simply checking whether a device is ON, the system considers multiple signals:

``
Device ON
     +
Room Empty
     +
Past Typical Shutdown Time
     ↓
Machine Learning Analysis
     ↓
Forgotten Device Risk

## System Architecture

                  ┌──────────────────────────┐
                  │     Device Telemetry     │
                  │                          │
                  │ AC / PC / Projector /    │
                  │ Monitor / Charger etc.  │
                  └────────────┬─────────────┘
                               │
                               │ JSON Telemetry
                               ▼
                  ┌──────────────────────────┐
                  │      devices.json        │
                  │                          │
                  │ Device ID                │
                  │ Power                    │
                  │ Occupancy                │
                  │ Room                     │
                  │ Shutdown Time            │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │      Feature Engine      │
                  │                          │
                  │ Room Occupancy           │
                  │ Device Active            │
                  │ Hours Past Shutdown      │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │   Random Forest Model    │
                  │                          │
                  │ model.pkl                │
                  │                          │
                  │ Prediction               │
                  │ Risk Probability         │
                  └────────────┬─────────────┘
                               │
             ┌─────────────────┼──────────────────┐
             │                 │                  │
             ▼                 ▼                  ▼
     ┌──────────────┐  ┌──────────────┐  ┌────────────────┐
     │   Dashboard  │  │    Alerts    │  │    Chatbot     │
     │              │  │              │  │                │
     │ Live Status  │  │ Risk Devices │  │ Device Queries │
     │ ML Risk      │  │ Notifications│  │ Status Queries │
     │ Metrics      │  │              │  │ Risk Queries   │
     └──────────────┘  └──────────────┘  └────────────────┘

## Input Features
IdleSpot uses three main ML features.

Feature	         Description
room_occupancy	   Number of people currently detected in the room
hours_past_shutdown	   Number of hours after the device's typical shutdown time
device_active	          1 if the device is ON, otherwise 0

The target variable is:

forgotten

where:

0 → Normal

1 → Possibly Forgotten

## Example Prediction

# Suppose:

Device        : Projector
Room          : Lab B
Power         : ON
Occupancy     : 0
Shutdown Time : 18:00
Current Time  : 19:30

# The system calculates:

room_occupancy      = 0
device_active       = 1
hours_past_shutdown = 1.5

These values are passed to the Random Forest model.

## Example:

ML Risk: 91.4%

Prediction:
🚨 POSSIBLY FORGOTTEN

The dashboard then displays an alert.

## Technologies Used
# Programming
Python
# Machine Learning
Scikit-learn

Random Forest Classifier

Classification Metrics
# Data Processing
Pandas

NumPy
# Application
Streamlit
# Model Storage
Joblib
# Telemetry
JSON
# Visualization
Streamlit Metrics

Streamlit DataFrame

Feature Importance Visualization

Model Evaluation Visualization

## Train the Machine Learning Model

# Run:

python train_model.py

# Example output:

Historical data:

   room_occupancy  hours_past_shutdown  device_active  forgotten
0              6              2.43              1          0
1             19             -1.24              1          0
2              0              3.18              1          1
3             11              0.72              1          0
4              0              1.54              1          1

## Run IdleSpot

# Start the Streamlit application:

streamlit run app.py

The application opens in the browser.

## Application Features

# Dashboard

The dashboard provides an overview of monitored devices.

# Sample Dashboard Output

<img width="1897" height="866" alt="image" src="https://github.com/user-attachments/assets/e151c1c7-8e3c-41df-9d4f-2ae2c19f903f" />

## Live Monitoring

IdleSpot provides an Automatic Live Monitoring mode.

The application repeatedly reads:

## Sample Live Monitoring Output
<img width="1892" height="867" alt="image" src="https://github.com/user-attachments/assets/241f5d54-8a3b-44fc-982f-13e58c6a1bda" />

## Manual Device Prediction

The user can select a specific device.

Example:

Select Device:

AC

IdleSpot displays:

Location: Lab A
Power: ON
Occupancy: 0
Typical shutdown: 18:00

After clicking:

🔍 PREDICT DEVICE

The application displays:

Power       ON
ML Risk     92.4%
Runtime     2h 15m

And:

🚨 AC may be forgotten!

Recommended action:
Check or switch OFF the device.

## Sample Manual Device Prediction Output

<img width="1897" height="855" alt="image" src="https://github.com/user-attachments/assets/0575be88-8714-4c98-9bf4-7bef2dd1ec88" />

## Model Evaluation

The model can be evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Accuracy

Measures the percentage of predictions that are correct.

Accuracy =
Correct Predictions / Total Predictions
Precision

Measures how many devices predicted as forgotten were actually forgotten.

Precision =
True Positive / (True Positive + False Positive)
Recall

Measures how many actual forgotten devices were successfully detected.

Recall =
True Positive / (True Positive + False Negative)
F1 Score

F1 combines precision and recall.

F1 =
2 × Precision × Recall /
(Precision + Recall)
🔥 Confusion Matrix

The confusion matrix contains four prediction categories:

                    Predicted
                    
                               Normal   Forgotten

              Actual Normal       TN         FP

              Actual Forgotten    FN         TP

Where:

TN = Correctly predicted normal devices

TP = Correctly predicted forgotten devices

FP = Normal device incorrectly flagged

FN = Forgotten device incorrectly missed

A heatmap can be used to visualize the confusion matrix.

## Sample Model Evaluation Output

<img width="1905" height="840" alt="image" src="https://github.com/user-attachments/assets/6cdaeac8-0f2a-4590-8dfc-922d3d544a8b" />

## IdleSpot Chatbot

The application also includes a simple device-status chatbot.

Example questions:

Is the AC ON?
What devices are running?
Which devices may be forgotten?
Are there any alerts?
Give me the current status.

## Sample chatbot output

<img width="1901" height="852" alt="image" src="https://github.com/user-attachments/assets/6a7b50ea-5f70-46df-87d8-5d6d1d3c658a" />

