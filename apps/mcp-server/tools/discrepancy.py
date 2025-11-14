import pandas as pd
import numpy as np
import os, sys

# --- 프로젝트 루트 경로 설정 ---
# 이 파일(discrepancy.py)은 ship-ai/apps/mcp-server/tools/ 에 있습니다.
# pkgs/ 폴더를 import하기 위해 ship-ai/ 까지 경로를 올려야 합니다.
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_PATH, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ---------------------------------

# 이제 pkgs/ 에 있는 모듈을 import 할 수 있습니다.
from pkgs.data_pipeline.data_handler import DataHandler
from pkgs.timellm_wrapper.wrapper import TimeLLMWrapper

class DiscrepancyCalculator:
    """
    설계안의 '괴리 분석 Tool' 로직을 수행합니다.
    DataHandler와 TimeLLMWrapper에 의존합니다.
    """
    def __init__(self, data_handler: DataHandler, model_wrapper: TimeLLMWrapper):
        self.data_handler = data_handler
        self.model_wrapper = model_wrapper
        print("DiscrepancyCalculator initialized.")

    def _calculate_fundamental_price(self, roe: float, real_debt_ratio: float) -> float:
        """
        (Placeholder) Phase 3-1. 펀더멘탈 적정 주가를 계산합니다.
        실제로는 더 복잡한 가치 평가 모델(e.g., RIM, DCF)이 필요합니다.
        여기서는 ROE가 높고 부채비율이 낮을수록 높게 평가하는 임시 공식을 사용합니다.
        """
        # (임시 공식) ROE 10%, 부채비율 100%를 기준으로 PBR 1배로 가정
        base_pbr = 1.0
        roe_factor = roe / 10.0  # ROE가 10%면 1.0
        debt_factor = 100.0 / max(real_debt_ratio, 1.0) # 부채비율 100%면 1.0
        
        # (임시) BPS(주당순자산)가 50000원이라고 가정
        bps = 50000 
        
        fundamental_price = bps * base_pbr * roe_factor * debt_factor
        return fundamental_price

    def calculate_score(self, ticker: str, as_of_date: str) -> dict:
        """
        특정 시점(as_of_date) 기준으로 괴리도 점수를 계산합니다.
        """
        try:
            # 1. DataHandler에서 12채널 데이터와 진단 점수를 가져옵니다.
            processed_df, diagnostics = self.data_handler.get_processed_data_with_diagnostics(ticker)
            
            # --- (선택적) 2단계 '지능형 라우터' 로직 ---
            qs_score = diagnostics['diagnostics'].get('bdi', {}).get('QS_Score', 0)
            if qs_score < 0.5: # 예: BDI 계절성이 50% 미만이면
                print(f"Warning: Low seasonality score (QS={qs_score:.2f}). LLM prediction might be unreliable.")
                # (향후) 여기에서 stats_model.predict() 등으로 분기할 수 있습니다.
            
            # 2. TimeLLMWrapper에서 기술적 예측치를 가져옵니다.
            # (주의) wrapper가 12채널을 입력받도록 수정되었다고 가정합니다. (로드맵 3단계)
            # (주의) 지금은 wrapper가 1채널(close)만 받으므로, 임시로 close 데이터만 넘깁니다.
            
            # wrapper.py의 predict_band는 내부에서 데이터를 로드합니다.
            # → DataHandler와 중복 로드를 피하기 위해 wrapper 수정이 필요하지만,
            # → 지금은 wrapper의 기존 코드를 그대로 활용합니다.
            
            # (수정) wrapper가 df를 직접 받도록 수정하는 것이 가장 좋습니다.
            # (임시) wrapper.py의 predict_band가 ticker와 as_of_date만 받으므로 그대로 호출
            
            prediction_result = self.model_wrapper.predict_band(ticker=ticker, as_of_date=as_of_date, horizon_days=20)
            technical_price = prediction_result['expected_price']

            # 3. 펀더멘탈 데이터 조회 (가장 최신 값)
            latest_data = processed_df[processed_df['date'] <= as_of_date].iloc[-1]
            roe = latest_data.get('roe_asof', 10.0) # 기본값 10%
            debt_ratio = latest_data.get('real_debt_ratio_asof', 100.0) # 기본값 100%
            
            # 4. 펀더멘탈 적정 주가 계산
            fundamental_price = self._calculate_fundamental_price(roe, debt_ratio)
            
            # 5. 괴리도 점수 계산
            # (기술적 기대주가 / 펀더멘탈 적정주가) - 1
            if fundamental_price == 0:
                discrepancy_score = 0.0
            else:
                discrepancy_score = (technical_price / fundamental_price) - 1
            
            return {
                "ticker": ticker,
                "as_of_date": as_of_date,
                "discrepancy_score": round(discrepancy_score, 4),
                "technical_price": round(technical_price, 2),
                "fundamental_price": round(fundamental_price, 2),
                "details": {
                    "roe_asof": roe,
                    "real_debt_ratio_asof": debt_ratio,
                    "bdi_qs_score": qs_score
                }
            }

        except Exception as e:
            print(f"Error in DiscrepancyCalculator: {e}")
            return {"error": str(e)}