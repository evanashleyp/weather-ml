import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ===========================
#  LOAD MODELS
# ===========================
rf_model = joblib.load("models/rain_rf.pkl")
lstm_model = load_model("models/lstm_rain.h5")

print("Models loaded successfully!")


# ==============================================================
# RANDOM FOREST PREDICTION (1 record manual input)
# ==============================================================

def predict_rf(temp, humidity, pressure, light, rain):
    """
    Predict rain (0/1) using RandomForest classifier
    """
    X = np.array([[temp, humidity, pressure, light, rain]])
    pred = rf_model.predict(X)[0]
    return int(pred)


# ==============================================================
# LSTM SEQUENCE PREDICTION
# ==============================================================

# LSTM butuh window (urutan data)
# Jadi kita load min. 60 data terakhir dari sensor.csv

def load_recent_sequence(csv_path="data/sensor.csv", window=60):
    df = pd.read_csv(csv_path)

    features = df[['humidity', 'pressure', 'rain', 'light']].values

    if len(features) < window:
        raise ValueError(f"Dataset terlalu pendek! butuh {window} baris terakhir.")

    return features[-window:]


def predict_lstm_next(csv_path="data/sensor.csv"):
    """
    Predict next rain_binary (0/1) using trained LSTM
    """
    seq = load_recent_sequence(csv_path)

    seq = np.expand_dims(seq, axis=0)   # bentuk: (1, 60, 4)

    pred = lstm_model.predict(seq)
    pred_class = 1 if pred[0][0] >= 0.5 else 0

    return pred_class

def predict_lstm_1_hour(csv_path="data/sensor.csv", window=60, future_steps=120):
    """
    Predict next 1 hour (assuming 1 reading per 30 sec).
    Returns list of 0/1 values.
    """

    seq = load_recent_sequence(csv_path, window)
    seq = np.array(seq)

    predictions = []

    for _ in range(future_steps):
        input_seq = np.expand_dims(seq, axis=0)  # (1,60,4)
        pred = lstm_model.predict(input_seq, verbose=0)
        pred_class = 1 if pred[0][0] >= 0.5 else 0
        predictions.append(pred_class)

        # Append predicted rain, but feature shape is (humidity, pressure, rain, light)
        # We keep humidity/pressure/light same as last known sample
        new_row = seq[-1].copy()
        new_row[2] = pred_class  # update rain column only

        # Shift window
        seq = np.vstack([seq[1:], new_row])

    return predictions


# ==============================================================
# DEMO
# ==============================================================

if __name__ == "__main__":
    print("\n=== MANUAL RF INPUT TEST ===")
    p = predict_rf(temp=28.3, humidity=82, pressure=1010, light=350, rain=0)
    print("RF Prediction:", p, "(Hujan)" if p else "(Tidak Hujan)")

    print("\n=== LSTM NEXT-STEP PREDICTION ===")
    p2 = predict_lstm_next()
    print("Next Rain (LSTM):", p2)

    print("\n=== LSTM PREDIKSI 1 JAM KE DEPAN ===")
    pred_next_hour = predict_lstm_1_hour()
    print(pred_next_hour)
    print("Jumlah hujan diprediksi:", sum(pred_next_hour), "menit hujan")

