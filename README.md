# ⚡ IdleSpot AI

### Real-Time Forgotten Device Detection Using Machine Learning

IdleSpot AI is a machine-learning-based monitoring system that detects devices that may have been accidentally left ON after working hours.

The system analyzes device telemetry such as:

- Room occupancy
- Device power status
- Hours past the typical shutdown time

A Random Forest classifier predicts whether an active device is potentially forgotten.

---

## 🚀 Key Features

- Real-time device monitoring
- Random Forest ML prediction
- Device ON/OFF status detection
- Forgotten-device risk prediction
- ML risk probability
- Automatic monitoring mode
- Manual device prediction
- Live dashboard
- Device risk analysis
- Feature importance visualization
- Interactive chatbot
- JSON-based telemetry
- Automatic refresh
- Real-time alerts

---

## 🧠 How It Works

```text
Device Telemetry
       ↓
devices.json
       ↓
Feature Extraction
       ↓
Room Occupancy
Device Active
Hours Past Shutdown
       ↓
Random Forest Model
       ↓
Forgotten Risk Prediction
       ↓
Dashboard / Alert / Chatbot
