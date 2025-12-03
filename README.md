🌧 Weather ML – Rain Prediction System

Sistem Machine Learning dan Deep Learning untuk memprediksi apakah akan terjadi hujan berdasarkan data sensor cuaca.
Project ini mencakup EDA, preprocessing, training RandomForest, training LSTM, dan model inference.

📁 Project Structure
weather-ml/
│── main.py
│── predict.py
│── README.md
│── requirements.txt
│── models/
│   ├── rain_rf.pkl
│   └── lstm_rain.h5
│── data/
│   └── sensor.csv
│── src/
│   ├── classical_models.py
│   ├── eda.py
│   ├── load_data.py
│   ├── lstm_model.py
│   ├── preprocess.py
│   └── utils.py

📌 Features
✔ 1. Exploratory Data Analysis (EDA)

Correlation heatmap

Feature distribution

Time-series visualization

Missing value handling

✔ 2. RandomForest Rain Classifier

Input features:
temp, humidity, pressure, light, rain

Output:
0 = tidak hujan
1 = hujan

Handling imbalance

Model: models/rain_rf.pkl

✔ 3. LSTM Deep Learning (Time-Series)

Memproses 60 data terakhir sebagai sequence

Prediksi rain_binary di step berikutnya

Model: models/lstm_rain.h5

🌧 Rain Labeling Rule
rain_binary = 1 → hujan        (rain level >= 2)
rain_binary = 0 → tidak hujan  (rain level <= 1)

📄 Dataset Description (sensor.csv)
Column	Description
temp	Temperature (°C)
humidity	Humidity (%)
pressure	Atmospheric pressure (hPa)
light	Light intensity
rain	Rain sensor level (0–5)
timestamp	Time of reading (optional)
🚀 How to Run (Training)
1. Create Virtual Environment
python -m venv venv

2. Activate Environment

Windows:

venv\Scripts\activate

3. Install Requirements
pip install -r requirements.txt

4. Run Training
python main.py


Model akan otomatis:

melakukan EDA

melakukan preprocessing

training RandomForest

training LSTM

menyimpan model ke folder /models/

🧪 How to Predict (Without Training)

Gunakan script:

predict.py

▶ Run
python predict.py

Script melakukan dua hal:
1. Manual Single-Record Prediction (RandomForest)

Contoh input:

temp=28.3, humidity=82, pressure=1010, light=350, rain=0


Output contoh:

RF Prediction: 1 (Hujan)

2. LSTM Next-Step Prediction (Sequence)

Menggunakan 60 data terakhir di sensor.csv

Output contoh:

Next Rain (LSTM): 0


Interpretasi:

0 → Tidak hujan

1 → Hujan

🧠 predict.py (Summary)

Load RandomForest & LSTM dari folder models/

RF: menerima input manual (1 baris)

LSTM: membaca 60 baris terakhir dari sensor.csv

Mengembalikan prediksi Rain (0/1)

📜 Example Output (Real Result)
Models loaded successfully!

=== MANUAL RF INPUT TEST ===
RF Prediction: 1 (Hujan)

=== LSTM NEXT-STEP PREDICTION ===
Next Rain (LSTM): 0

🔧 Troubleshooting
Problem	Solution
TensorFlow slow	pip install tensorflow-intel
Model not saved	Pastikan folder /models ada
Dataset error	Letakkan file di data/sensor.csv
Sequence too short	LSTM butuh >= 60 baris
👨‍💻 Authors
Name	Student ID
Yoel Jonathan Lee	1123008
Evan Ashley Pringadi	1124012
Garry Alexander Chandra	1124055
📜 License

Academic project for Microprocessor and Embedded Systems.