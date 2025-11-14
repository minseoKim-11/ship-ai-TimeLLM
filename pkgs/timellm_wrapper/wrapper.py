import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from types import SimpleNamespace
from .io import load_minimal_df

# Time-LLM 레포 경로를 sys.path에 추가
TLLM_ROOT = "/workspace/Time-LLM/models"
if TLLM_ROOT not in sys.path:
    sys.path.insert(0, TLLM_ROOT)

# 이제 모델 import
from TimeLLM import Model as TimeLLMModel

class TimeLLMWrapper:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # <<< 1. 여기에 '조선업 특화 브리핑 프롬프트'를 정의합니다. >>>
        # (이것이 바로 EKE이자 PaP+ 입니다.)
        SHIPBUILDING_PROMPT = """
        This data represents key financial and operational metrics for a shipbuilding company.
        The channels include stock prices, financial health, new business orders,
        and external macroeconomic factors like the Baltic Dry Index (bdi).
        Forecast the stock's log return for the next {pred_len} days,
        considering the complex interplay between these factors.
        """
        # 기본 하이퍼 (네 Model에 필요한 필드와 일치해야 함)
        # - enc_in: 입력 채널 수 (우선 close 한 채널만 -> 1)
        # - seq_len/pred_len/patch_len/stride는 네가 쓰던 값으로 맞춰도 됨
        self.cfg = SimpleNamespace(
            task_name="long_term_forecast",
            pred_len=20, # <<< 2. 우선 20일로 고정 (나중에 predict_band에서 동적으로 바꿀지 결정) >>>
            seq_len=120,
            d_ff=1536,
            llm_dim=768,         # GPT-2 hidden size
            patch_len=16,
            stride=8,
            enc_in=1,            # 입력 채널 수 # <<< 3. 아직은 1채널 유지 (3단계에서 12로 변경) >>>
            llm_model="GPT2",    # 'GPT2' / 'LLAMA' / 'BERT' 중 선택
            llm_layers=4,        # 가벼운 실험용 레이어 수
            dropout=0.1,
            d_model=512,         # PatchEmbedding 쪽 내부 차원
            n_heads=8,           # d_model % n_heads == 0 권장
            # <<< 4. EKE/PaP+ 기능을 활성화합니다. >>>
            prompt_domain=True, # True면 configs.content 사용
            content=SHIPBUILDING_PROMPT.format(pred_len=20) # "" -> 정의한 프롬프트로 변경
        )

        # 모델 생성 및 디바이스 배치
        self.model = TimeLLMModel(self.cfg, patch_len=self.cfg.patch_len, stride=self.cfg.stride)
        self.model.to(self.device).eval()

    # 요청 날짜 기준으로 윈도우 자르기
    def _slice_by_date(self, df: pd.DataFrame, as_of_date: str, seq_len: int) -> pd.DataFrame:
        d = pd.to_datetime(as_of_date)
        win = df[df["date"] <= d].tail(seq_len)
        if len(win) < max(30, seq_len // 2):
            raise ValueError("입력 날짜 이전 시계열이 충분하지 않습니다.")
        return win

    # 간단 밴드 계산 유틸 (예측 + 과거 변동성 기반 CI)
    def _band_from_sequence(self, last_price: float, y_seq: np.ndarray) -> tuple[float,float,float,float]:
        """
        y_seq: (pred_len,) 예측 가격 시퀀스
        반환: (expected, lower, upper, conf)
        """
        expected = float(y_seq[-1])  # 마지막 시점 예측가
        # 과거 변동성으로 대략적 밴드 (너무 작거나 크면 클램프)
        # 더 좋은 방법: y_seq의 표준편차/예측 오차분산이 있다면 그걸 사용
        # 여기서는 간단히 과거 20일 수익률 표준편차를 사용
        conf = 0.68
        return expected, None, None, conf

    def predict_band(self, ticker: str, as_of_date: str, horizon_days: int = 20):
        # 1) 데이터 로드 & 윈도우
        df = load_minimal_df(ticker)  # date, close 필수
        win = self._slice_by_date(df, as_of_date, self.cfg.seq_len)
        close = win["close"].to_numpy(dtype=np.float32)  # [T]
        last = float(close[-1])

        # 2) 입력 텐서 구성
        # x_enc: (B, T, N)  ── 여기서 B=1, N=1(close만)
        x_enc = torch.from_numpy(close).view(1, -1, 1).to(self.device)  # [1, seq_len, 1]

        # x_mark_enc/x_dec/x_mark_dec: 모델 시그니처상 필요하지만 forecast에선 안 쓰이므로 0 텐서로 채움
        # 모양만 맞춰주면 됨. x_dec 길이는 pred_len로.
        B = 1
        T = x_enc.shape[1]
        N = x_enc.shape[2]
        pred_len = int(horizon_days)  # 요청 horizon에 맞춤
        # 모델 내부는 self.cfg.pred_len을 사용하므로 일치시켜줌
        self.cfg.pred_len = pred_len

        # (중요) pred_len이 바뀌었으면 head 등 내부가 pred_len을 참조할 수 있으므로
        # 안전하게 모델을 재생성하는 편이 낫다.
        # 비용을 줄이려면 pred_len을 고정값으로 쓰고 요청과 다르면 마지막 시점만 사용해도 됨.
        self.model = TimeLLMModel(self.cfg, patch_len=self.cfg.patch_len, stride=self.cfg.stride).to(self.device).eval()

        x_mark_enc  = torch.zeros((B, T, 4), dtype=torch.float32, device=self.device)       # dummy
        x_dec       = torch.zeros((B, pred_len, N), dtype=torch.float32, device=self.device) # dummy
        x_mark_dec  = torch.zeros((B, pred_len, 4), dtype=torch.float32, device=self.device) # dummy

        # 3) 추론
        with torch.no_grad():
            # forward는 (B, T, N) → (B, pred_len, N)를 반환
            y_pred = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [1, pred_len, 1]
            y_seq = y_pred.squeeze().detach().float().cpu().numpy()    # (pred_len,)

        # 4) 밴드 계산
        # 기본 전략: 최근 20일 수익률 표준편차로 밴드폭 산출
        ret = pd.Series(close).pct_change().dropna()
        if len(ret) >= 20:
            sigma_r = float(ret.tail(20).std())
        elif len(ret) >= 5:
            sigma_r = float(ret.tail(5).std())
        else:
            sigma_r = 0.02  # 최소 폭 가드

        expected = float(y_seq[-1])
        # 밴드는 예측가 주변의 ± (최근 변동성 × 예측가)로 구성
        lower = expected * (1.0 - sigma_r)
        upper = expected * (1.0 + sigma_r)
        confidence = 0.68  # 대략 1σ

        return {
            "pred_horizon_days": pred_len,
            "expected_price": round(expected, 4),
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "confidence": confidence
        }
