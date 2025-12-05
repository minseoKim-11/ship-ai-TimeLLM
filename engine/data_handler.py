import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FINAL_MASTER_FILE = os.path.join(DATA_DIR, "final_master_table_v2.csv")
MASTER_TABLE_PATH = os.path.join(DATA_DIR, "final_master_table_v2.csv")

# ============================================================
# 표준화(Standardization)와 zfill(6)이 적용된 DataHandler
# ============================================================
class DataHandler:
    def __init__(self, file_path, train_end_date='2022-12-31'):
        self.file_path = file_path
        self.train_end_date = pd.to_datetime(train_end_date)
        self.data_by_ticker = {}   # 원본 데이터
        self.scalers_by_ticker = {} # Ticker별 Scaler
        self.tickers = []
        
        self._load_and_process_data()
        self._fit_scalers()
        
    def _load_and_process_data(self):
        try:
            # 1. dtype=str로 읽기
            df = pd.read_csv(
                self.file_path, 
                parse_dates=['date'],
                dtype={'ticker': str} 
            )
            # 2. zfill(6)로 '0' 채우기
            df['ticker'] = df['ticker'].str.zfill(6)
            df = df.set_index('date')
            
            self.tickers = df['ticker'].unique()
            
            for ticker in self.tickers:
                ticker_df = df[df['ticker'] == ticker].copy()
                channel_cols = [col for col in ticker_df.columns if col not in ['ticker']]
                self.data_by_ticker[ticker] = ticker_df[channel_cols]
            
            print(f"[DataHandler V2] Success: Loaded {len(self.tickers)} tickers.")
            print(f"[DataHandler V2] Available tickers: {self.tickers}")

        except Exception as e:
            print(f"[DataHandler V2] Error loading data: {e}")

    def _fit_scalers(self):
        """
        [Data Leakage 방지] 훈련 데이터로만 Scaler를 학습(fit)
        """
        print(f"[DataHandler V2] Fitting scalers using data up to {self.train_end_date.date()}...")
        for ticker in self.tickers:
            train_data = self.data_by_ticker[ticker].loc[:self.train_end_date]
            if train_data.empty:
                print(f"  > Warning: No training data for {ticker}.")
                continue
            
            scaler = StandardScaler()
            scaler.fit(train_data) # 'fit'은 훈련 데이터로만!
            self.scalers_by_ticker[ticker] = scaler
        print("[DataHandler V2] Scalers fitted.")

    def get_scaled_data_by_ticker(self, ticker):
        """
        'transform'은 전체 데이터에 적용하여 표준화된 DF 반환
        """
        if ticker not in self.scalers_by_ticker:
            print(f"[DataHandler V2] Error: No scaler for {ticker}")
            return None
        
        original_data = self.data_by_ticker[ticker]
        scaler = self.scalers_by_ticker[ticker]
        
        scaled_data_np = scaler.transform(original_data)
        
        scaled_df = pd.DataFrame(
            scaled_data_np, 
            index=original_data.index, 
            columns=original_data.columns
        )
        return scaled_df

    def get_all_tickers(self):
        return self.tickers

# ============================================================
# 슬라이딩 윈도우 함수 + Dataset 정의
# ============================================================
def create_sliding_windows(data, input_seq_len, output_seq_len):
    """
    DataFrame(2D: [time, features]) -> (X, y) 3D numpy 배열로 변환
    X: (N, input_seq_len, C)
    y: (N, output_seq_len, C)
    """
    data_np = data.values
    n_samples = len(data_np)
    X, y = [], []

    total_len = input_seq_len + output_seq_len
    for i in range(n_samples - total_len + 1):
        x_win = data_np[i : i + input_seq_len]
        y_win = data_np[i + input_seq_len : i + total_len]
        X.append(x_win)
        y.append(y_win)

    return np.array(X), np.array(y)


class ShipDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
