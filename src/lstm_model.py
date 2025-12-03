import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

def create_window_dataset(features, target, window=60):
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features[i:i+window])
        y.append(target[i+window])  # 0/1

    return np.array(X), np.array(y)

def build_lstm(window, features=5):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, features)),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


from keras.models import load_model

def train_lstm(X, y, epochs=20, batch=64):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = build_lstm(X.shape[1], X.shape[2])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch)

    preds = model.predict(X_test)
    preds_binary = (preds > 0.5).astype(int)

    acc = (preds_binary.reshape(-1) == y_test).mean()
    print("LSTM Accuracy:", acc)

    model.save("models/lstm_rain.h5")
    print("LSTM saved to models/lstm_rain.h5")

    return model, preds_binary

