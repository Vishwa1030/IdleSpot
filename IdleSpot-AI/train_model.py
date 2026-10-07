import pandas as pd 
import numpy as np 
import joblib 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score,classification_report 
 
 
np.random.seed(42) 
 
# Create historical device telemetry 
n = 2000 
 
room_occupancy = np.random.randint(0, 31, n) 
 
hours_past_shutdown = np.random.uniform(-3, 5, n) 
 
device_active = np.random.choice([0, 1],n,p=[0.20, 0.80]) 
 
# Create realistic historical labels 
 
risk_score = ( 
    (room_occupancy == 0) * 4 
    + (hours_past_shutdown > 0) * 3 
    + (hours_past_shutdown > 1) * 2 
    + (device_active == 1) * 2 
) 
 
# Small noise 
noise = np.random.randint(-1, 2, n) 
 
risk_score = risk_score + noise 
 
forgotten = (risk_score >= 7).astype(int) 
 
 
df = pd.DataFrame({ 
 
    "room_occupancy": room_occupancy, 
 
    "hours_past_shutdown": hours_past_shutdown, 
 
    "device_active": device_active, 
 
    "forgotten": forgotten 
}) 
 
 
print("\nHistorical data:") 
print(df.head()) 
 
print("\nClass distribution:") 
print(df["forgotten"].value_counts()) 
 
 
#Features and target 
X = df.drop(columns=["forgotten"]) 
y = df["forgotten"] 
 
# Train / test split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42) 
 
 
# RandomForestClassifier model 
model = RandomForestClassifier(n_estimators=100,random_state=42) 
 
# Train the model 
model.fit(X_train,y_train) 
y_pred = model.predict(X_test) 
 
# Evaluate the model 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("\nClassification Report:\n", classification_report(y_test, y_pred)) 
 
 
 
# Save model 
joblib.dump(model,"model.pkl") 
print("\nRandomForest Model successfully saved as model.pkl") 
