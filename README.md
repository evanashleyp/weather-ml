# 🌧 Weather ML – Rain Prediction System

Sistem Machine Learning dan Deep Learning untuk memprediksi apakah akan terjadi hujan berdasarkan data sensor cuaca.  
Project ini mencakup *EDA*, *preprocessing*, *model training*, dan *model saving*.

---

## 📁 Project Structure

weather-ml/
│── main.py
│── README.md
│── requirements.txt
│── models/
│ ├── rain_rf.pkl
│ └── lstm_rain.h5
│── data/
│ └── sensor.csv
│── src/
│ ├── classical_models.py
│ ├── eda.py
│ ├── load_data.py
│ ├── lstm_model.py
│ ├── preprocess.py
│ └── utils.py


---

## 📌 Features

### ✔ **1. EDA (Exploratory Data Analysis)**
- Correlation heatmap  
- Feature distribution  
- Time-series analysis  
- Missing value check  

### ✔ **2. Classical Machine Learning (RandomForest)**
- Input features:
  - `temp, humidity, pressure, light, rain`
- Output label:
  - `0 = tidak hujan`
  - `1 = hujan`
- Handling imbalance: `class_weight="balanced"`
- Saved model: `models/rain_rf.pkl`

### ✔ **3. LSTM Deep Learning for Rain Prediction**
- Sequence-based binary rain classification
- Sliding window generator
- Saved model: `models/lstm_rain.h5`

---

## 🌧 Rain Labeling Rule

Sensor memiliki nilai rain level `0–5`.  
Project ini mengonversi ke label biner:



rain_binary = 1 → hujan (rain level >= 2)
rain_binary = 0 → tidak hujan (rain level <= 1)


---

## 📄 Dataset Description (`sensor.csv`)

| Column     | Description                  |
|------------|------------------------------|
| temp       | Temperature (°C)             |
| humidity   | Humidity (%)                 |
| pressure   | Atmospheric pressure (hPa)   |
| light      | Light sensor reading         |
| rain       | Raw rain sensor level (0–5)  |
| timestamp  | (opsional) Time of reading   |

---

## 🚀 How to Run

### **1. Create Virtual Environment**
```bash
python -m venv venv

2. Activate Environment

Windows:

venv\Scripts\activate

3. Install Requirements
pip install -r requirements.txt

4. Run the Training
python main.py


Model akan otomatis:

melakukan EDA

preprocess data

train RandomForest

train LSTM

menyimpan model ke folder /models/

📊 Model Evaluation
RandomForest

Accuracy

Classification Report

Confusion Matrix

LSTM

Training accuracy

Test accuracy

Predict based on time sequence

🔄 Load Models for Future Use
1. RandomForest
import joblib
clf = joblib.load("models/rain_rf.pkl")

2. LSTM
from keras.models import load_model
model = load_model("models/lstm_rain.h5")

🔧 Troubleshooting

TensorFlow sangat lambat?
Gunakan: pip install tensorflow-intel

Model tidak tersimpan?
Pastikan folder /models sudah dibuat.

Dataset tidak ditemukan?
Pastikan file berada di: data/sensor.csv

👨‍💻 Authors
Name	Student ID
Yoel Jonathan Lee	1123008
Evan Ashley Pringadi	1124012
Garry Alexander Chandra	1124055

📜 License
Project for academic purposes ( Microprocessor and Embedded Systems ML Project).