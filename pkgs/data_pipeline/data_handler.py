import pandas as pd
import numpy as np
import os
from datetime import datetime

# 1단계에서 만든 STL 진단기 클래스를 import합니다.
from .stl_analyzer import Stl_decomposition

# (가정) pykrx, dart-fss 등 Raw 데이터 수집기는 별도로 구현되어 있다고 가정
# from .raw_collectors import KrxCollector, DartCollector, MacroCollector

class DataHandler:
    """
    설계안 v2.0에 명시된 데이터 파이프라인을 실행합니다.
    1. Raw 데이터 수집 (placeholder)
    2. 12채널 피처 엔지니어링
    3. STL 진단기를 이용한 메타데이터(QS 점수) 생성
    """
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        # ... (pykrx, dart API 세션 등 초기화) ...
        print("DataHandler initialized.")

    def _load_raw_data(self, ticker: str) -> dict:
        """
        (Placeholder) Phase 1-1. 각 소스로부터 Raw 데이터를 수집합니다.
        실제로는 pykrx, DART API, 크롤러 등을 여기서 호출해야 합니다.
        """
        print(f"Loading raw data for {ticker}...")
        # (예시) 실제로는 이 부분에 pykrx, dart-fss 호출 로직이 들어갑니다.
        # 지금은 data/silver/prices_daily/ 에서 불러오는 것으로 대체합니다.
        # (참고) wrapper.py의 load_minimal_df를 이곳으로 가져와도 좋습니다.
        price_df_path = os.path.join(self.raw_data_path, f"prices_daily/{ticker}.csv")
        if not os.path.exists(price_df_path):
             raise FileNotFoundError(f"Raw price data not found: {price_df_path}")
             
        price_df = pd.read_csv(price_df_path, parse_dates=['date'])
        
        # (임시) 다른 채널들의 Raw 데이터가 있다고 가정
        financial_df = pd.DataFrame() # DART API 결과
        order_df = pd.DataFrame()     # 뉴스/공시 크롤링 결과
        macro_df = pd.DataFrame()     # BDI, WTI 등
        
        return {
            "price": price_df,
            "financial": financial_df,
            "order": order_df,
            "macro": macro_df
        }

    def _engineer_features(self, raw_data: dict) -> pd.DataFrame:
        """
        (Placeholder) Phase 1-2. 12채널 피처 엔지니어링을 수행합니다.
        설계안의 모든 파생 변수 로직이 여기에 구현되어야 합니다.
        """
        print("Engineering 12 channels...")
        
        # 기준 캘린더 생성 (한국 거래일 기준)
        price_df = raw_data['price'].set_index('date')
        
        # 1. close_log, ret_1d, trading_volume_log
        df = pd.DataFrame(index=price_df.index)
        df['close_log'] = np.log(price_df['close'])
        df['ret_1d'] = np.log(price_df['close'] / price_df['close'].shift(1))
        df['trading_volume_log'] = np.log1p(price_df['trading_volume'])
        
        # (Placeholder) 2~12번 채널 생성 로직
        # ... (roe_asof, real_debt_ratio_asof의 'as-of' ffill 로직 구현) ...
        # ... (new_order_amount_impulse, new_order_amount_stair 로직 구현) ...
        # ... (bdi, wti, newbuilding_price_index 리샘플 및 ffill 로직 구현) ...
        # ... (imo_event_impulse, imo_event_decay 로직 구현) ...
        
        # (임시) 12개 채널이 모두 생성되었다고 가정하고, 빈 값은 0으로 채움
        channel_names = [
            'close_log', 'ret_1d', 'trading_volume_log', 'roe_asof', 
            'real_debt_ratio_asof', 'new_order_amount_impulse', 
            'new_order_amount_stair', 'bdi', 'wti', 'newbuilding_price_index', 
            'imo_event_impulse', 'imo_event_decay'
        ]
        for col in channel_names:
            if col not in df.columns:
                df[col] = 0.0 # 임시로 0으로 채움
        
        df = df[channel_names].dropna() # 모든 채널이 준비된 시점부터 사용
        return df.reset_index() # date를 다시 컬럼으로

    def _run_stl_diagnostics(self, df: pd.DataFrame) -> dict:
        """
        (Core) Phase 2-2. STL 진단기를 실행하여 QS 점수를 계산합니다.
        """
        print("Running STL diagnostics...")
        metadata = {'diagnostics': {}}
        
        # 진단 대상 채널 목록
        # (주기성이 중요해 보이는 외부 매크로 변수들)
        channels_to_diagnose = ['bdi', 'wti', 'newbuilding_price_index']
        
        # (참고) 거래일 기준 1년은 약 252일, 1분기는 약 63일입니다.
        # 데이터의 특성에 맞춰 period를 설정해야 합니다.
        # 예: WTI (일별 데이터, 1년 주기성) -> period=252
        # 예: newbuilding_price_index (월별 데이터, 1년 주기성) -> period=12 (월별 ffill 후 기준)
        period_map = {
            'bdi': 252, # 1년 (거래일 기준)
            'wti': 252, # 1년 (거래일 기준)
            'newbuilding_price_index': 252 # 월별 데이터를 일별 ffill 했다고 가정
        }

        for channel in channels_to_diagnose:
            if channel not in df.columns:
                continue
                
            data_series = df.set_index('date')[channel]
            period = period_map.get(channel)

            # STL 분석기 인스턴스 생성 및 실행
            analyzer = Stl_decomposition(data=data_series, period=period)
            trend_strength, seas_strength = analyzer.strength()
            
            # 메타데이터에 저장
            metadata['diagnostics'][channel] = {
                'QS_Score': seas_strength,
                'QT_Score': trend_strength,
                'period_used': period
            }
            print(f"  - Channel '{channel}': QS={seas_strength:.4f}, QT={trend_strength:.4f}")

        return metadata

    def get_processed_data_with_diagnostics(self, ticker: str, force_process: bool = False) -> tuple[pd.DataFrame, dict]:
        """
        메인 실행 함수.
        1. 전처리된 데이터가 있으면 로드, 없으면 생성
        2. STL 진단 실행
        3. 최종 데이터프레임과 진단 메타데이터 반환
        """
        
        # (TBD) 설계안에 따라 Parquet 파일로 저장/로드하는 로직 구현
        # processed_file_path = os.path.join(self.processed_data_path, f"{ticker}_processed.parquet")
        
        # (임시) 지금은 항상 새로 생성
        if force_process: # or not os.path.exists(processed_file_path):
            raw_data = self._load_raw_data(ticker)
            processed_df = self._engineer_features(raw_data)
            # (TBD) processed_df.to_parquet(processed_file_path)
        else:
            # (TBD) processed_df = pd.read_parquet(processed_file_path)
            pass
        
        # (임시) 위에서 바로 생성한 df 사용
        processed_df = self._engineer_features(self._load_raw_data(ticker))

        # 2. STL 진단 실행
        metadata = self._run_stl_diagnostics(processed_df)
        
        print(f"Data processing and diagnostics complete for {ticker}.")
        
        # 3. 결과 반환
        return processed_df, metadata

#
# 이 파일을 직접 실행할 때 테스트하는 코드
#
if __name__ == "__main__":
    # (주의) 경로를 본인 환경에 맞게 수정해야 합니다.
    RAW_PATH = "../../data/silver" # (예시) ship-ai/data/silver
    PROCESSED_PATH = "../../data/processed" # (예시) ship-ai/data/processed
    
    # (주의) data/silver/prices_daily/에 HHI.csv (예시) 파일이 있어야 합니다.
    TICKER_SYMBOL = "HHI" 
    
    try:
        handler = DataHandler(raw_data_path=RAW_PATH, processed_data_path=PROCESSED_PATH)
        
        # 메인 함수 실행
        final_dataframe, diagnostic_metadata = handler.get_processed_data_with_diagnostics(TICKER_SYMBOL)
        
        print("\n--- Final DataFrame (Tail 5) ---")
        print(final_dataframe.tail())
        
        print("\n--- Diagnostic Metadata ---")
        print(diagnostic_metadata)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please check if the data file (e.g., data/silver/prices_daily/HHI.csv) exists and paths are correct.")
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please ensure 'statsmodels' is installed (`pip install statsmodels`)")