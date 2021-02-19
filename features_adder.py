import pandas as pd
import ta
import numpy as np

class AddFeatures:

    def __init__(self, df, lookback_window_size):
        self.df = df
        self.lookback_window_size = lookback_window_size
        self.features = None

    def _add_features(self, df):
        df_final =  ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True)
        self.features = [col for col in df_final.columns if col not in ['Timestamp']]
        return df_final

    def _add_lagged_features(self, df):
        df_final = df.copy()
        for ohlcv in self.features:
            for lag in range(1, self.lookback_window_size+1):
                df_final[f'{ohlcv}_lag_{lag}'] = df_final[ohlcv].shift(lag)
        return df_final

    def _finalize_remove_na(self, df):
        df.dropna(axis=0, inplace=True)
        df = df.reset_index()
        return df

    def get_features(self):
        return self.features

    def forward(self):
        df_out = self._add_features(self.df)
        df_out = self._add_lagged_features(df_out)
        df_out = self._finalize_remove_na(df_out)
        return df_out

df = pd.read_csv('data/train/btc_price_yahoo_140917_191231.csv')
df.dropna(axis=0, inplace=True)
feature_adder = AddFeatures(df, 10)
df_out = feature_adder.forward()
print(df_out.describe().columns)
print(feature_adder.get_features())