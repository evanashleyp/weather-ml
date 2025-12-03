import matplotlib.pyplot as plt
import seaborn as sns

def plot_temperature(df):
    plt.plot(df['timestamp'], df['temp'])
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.show()

def plot_humidity_hist(df):
    plt.hist(df['humidity'])
    plt.xlabel("Humidity")
    plt.ylabel("Count")
    plt.show()

def plot_corr(df):
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True)
    plt.show()

    return corr
