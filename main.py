from src.load_data import load_sensor_data
from src.preprocess import fill_missing, add_zscores, scale_features
from src.eda import plot_temperature, plot_humidity_hist, plot_corr
from src.classical_models import train_rain_classifier
from src.lstm_model import create_window_dataset, train_lstm
from sklearn.metrics import accuracy_score
import os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# ================== LOAD DATA ==================
df = load_sensor_data("data/sensor.csv")

# ================== PREPROCESS ==================
df = fill_missing(df)
df = add_zscores(df)

# Binary target: rain or not
df["rain_binary"] = df["rain_level"].apply(lambda x: 1 if x >= 2 else 0)
print(df["rain_binary"].value_counts())

# ================== EDA ==================
plot_temperature(df)
plot_humidity_hist(df)
corr = plot_corr(df)
print(corr)

# ================== CLASSICAL MODEL ==================
rf_model, Xtest_rf, ytest_rf, rf_preds = train_rain_classifier(df)

# RF Accuracy
rf_acc = accuracy_score(ytest_rf, rf_preds)
print("\nRandomForest Test Accuracy:", rf_acc)

# ================== LSTM INPUT SCALING ==================
cols = ['humidity', 'pressure', 'rain', 'light', 'rain_binary']

scaled, scaler = scale_features(df, cols)

# ---- FIX: scaled = numpy array, so index by position ----
features = scaled[:, :4]           # humidity, pressure, rain, light
target = scaled[:, 4]              # rain_binary

# ================== LSTM DATASET ==================
X, y = create_window_dataset(
    features,
    target,
    window=60
)

# ================== TRAIN LSTM ==================
lstm_model, lstm_preds = train_lstm(X, y)

# ================== CONFUSION MATRIX ==================
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y[-len(lstm_preds):], lstm_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - LSTM Rain Classifier")
plt.show()

# ================== LSTM ACCURACY ==================
lstm_acc = accuracy_score(y[-len(lstm_preds):], lstm_preds.reshape(-1))
print("\nLSTM Test Accuracy:", lstm_acc)

print("LSTM rain classification training completed!")
