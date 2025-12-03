import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

def fill_missing(df):
    return df.fillna(df.mean(numeric_only=True))

def add_zscores(df):
    df['z_temp'] = zscore(df['temp'])
    return df

def create_temp_category(df):
    df['temp_category'] = pd.cut(
        df['temp'],
        bins=[-999, 20, 28, 999],
        labels=['Cold','Normal','Hot']
    )
    return df

def scale_features(df, cols):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[cols])
    return scaled, scaler
