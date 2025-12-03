import pandas as pd

def load_sensor_data(path: str):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df = df.drop_duplicates()
    return df
