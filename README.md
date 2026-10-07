# ⚡ IdleSpot — AI-Based Forgotten Device Detection System

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

## 📌 Project Overview

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

