import pandas as pd
import ta
import numpy as np
import talib

class FeaturesAdder:

    def __init__(self, df, lookback_window_size, all_features=False):
        self.lookback_window_size = lookback_window_size
        self.features = None
        self.complete_features_with_lags = []
        self.all_features = all_features
        self.df_out = self._forward(df)

    def _add_all_features(self, df):
        df_final =  ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=False)
        df_final.dropna(axis=0, inplace=True)
        df_final = df_final.reset_index()
        self.features = [col for col in df_final.columns if col not in ['Timestamp', 'index']]
        return df_final

    def _add_CCI_RSI_features(self, df):
        df_final = df.copy()
        cci = talib.CCI(df['High'], df['Low'], df['Close'])
        rsi = talib.RSI(df['Close'])
        roc = talib.ROC(df['Close'])
        df['avg_cci_rsi_roc'] = (cci + rsi + roc)/3
        df_final.dropna(axis=0, inplace=True)
        df_final = df_final.reset_index()
        self.features = [col for col in df_final.columns if col not in ['Timestamp', 'index']]
        return df_final

    def _add_lagged_features(self, df):
        df_final = df.copy()
        for feature in self.features:
            for lag in range(1, self.lookback_window_size+1):
                df_final[f'{feature}_lag_{lag}'] = df_final[feature].shift(lag)
                self.complete_features_with_lags.append(f'{feature}_lag_{lag}')
        self.complete_features_with_lags = self.features + self.complete_features_with_lags
        df_final = ta.utils.dropna(df_final)
        return df_final

    def get_features(self):
        return self.features

    def get_complete_features_with_lags(self):
        return self.complete_features_with_lags

    def _forward(self, df):
        if self.all_features:
            df_out = self._add_all_features(df)
        else:
            df_out = self._add_CCI_RSI_features(df)
        df_out = self._add_lagged_features(df_out)
        return df_out

    def get_processed_df(self):
        return self.df_out

# df = pd.read_csv('data/train/btc_price_yahoo_140917_191231.csv')
# df.dropna(axis=0, inplace=True)
# feature_adder = FeaturesAdder(df, 10)
#
# print(feature_adder.get_processed_df())
# print(feature_adder.get_features())
# print(len(feature_adder.get_complete_features_with_lags()))