from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def train_linear_reg(df):
    X = df[['humidity']]
    y = df['temp']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)

    return model, mse, X_test, y_test

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_rain_classifier(df):
    features = ["temp", "humidity", "pressure", "light", "rain"]
    
    X = df[features]
    y = df["rain_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_depth=12
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n=== Rain Classifier Results ===")
    print("Accuracy:", acc)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - RandomForest Rain Classifier")
    plt.show()

    print(classification_report(y_test, preds))

    # SAVE MODEL
    joblib.dump(clf, "models/rain_rf.pkl")
    print("RandomForest saved to models/rain_rf.pkl")

    return clf, X_test, y_test, preds
