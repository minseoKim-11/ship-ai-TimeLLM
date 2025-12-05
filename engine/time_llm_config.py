import os
import torch
import sys
import importlib.util

# ============================================================
# 경로 정의
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPT2_PATH = os.path.join(PROJECT_ROOT, "pretrained_models", "gpt2")
CKPT_PATH = os.path.join(PROJECT_ROOT, "models", "ship_time_llm_tmp6_ft_h3_e5.pth")

TIME_LLM_ROOT = os.path.join(PROJECT_ROOT, "external", "time-llm")

print("[Time-LLM ROOT]", TIME_LLM_ROOT)
print("[has models?] ", os.path.isdir(os.path.join(TIME_LLM_ROOT, "models")))
print("[has layers?] ", os.path.isdir(os.path.join(TIME_LLM_ROOT, "layers")))

# -----------------------------
# 2) 'layers' 패키지를 직접 로드해서 등록
# -----------------------------
LAYERS_INIT = os.path.join(TIME_LLM_ROOT, "layers", "__init__.py")
if not os.path.exists(LAYERS_INIT):
    raise FileNotFoundError(f"layers/__init__.py not found at: {LAYERS_INIT}")

# 혹시 이전 실패한 import가 sys.modules에 남아 있으면 제거
if "layers" in sys.modules:
    del sys.modules["layers"]

layers_spec = importlib.util.spec_from_file_location("layers", LAYERS_INIT)
layers_module = importlib.util.module_from_spec(layers_spec)
sys.modules["layers"] = layers_module  # ★ 여기서 공식 등록
layers_spec.loader.exec_module(layers_module)
print("[INFO] 'layers' 패키지 로드 성공")

# ============================================================
# TimeLLM 모델 import
# ============================================================
TIME_LLM_FILE = os.path.join(TIME_LLM_ROOT, "models", "TimeLLM.py")
if not os.path.exists(TIME_LLM_FILE):
    raise FileNotFoundError(f"TimeLLM.py not found at: {TIME_LLM_FILE}")

spec = importlib.util.spec_from_file_location("time_llm_custom", TIME_LLM_FILE)
time_llm_module = module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time_llm_module)

TimeLLM = time_llm_module.Model
print("[INFO] TimeLLM 모델 임포트 성공:", TIME_LLM_FILE)

# ============================================================
# Cofigs 정의
# ============================================================
class Configs:
    def __init__(self):
        # 기본 세팅
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'Stock_Prediction'
        self.model = 'TimeLLM'

        # 데이터 차원
        self.seq_len   = 120
        self.label_len = 60
        self.pred_len  = 10
        self.enc_in = 12
        self.dec_in = 12
        self.c_out = 12

        # LLM 설정 (학습 때와 동일하게!)
        self.llm_model      = 'GPT2'
        self.llm_model_path = GPT2_PATH
        self.llm_dim    = 768
        self.llm_layers = 8

        # Patch 설정
        self.patch_len = 8
        self.stride    = 4

        # 모델 차원
        self.d_model = 512
        self.d_ff    = 512
        self.n_heads = 12
        self.dropout = 0.00

        # Prompt
        self.prompt_domain = 1
        self.content = (
            "Task: Forecast daily closing prices for Korean shipbuilding companies. "
            "Input Data: 12 channels including OHLC prices, trading volume, "
            "and macro-indicators such as Brent oil price, USD/KRW exchange rate, "
            "interest rate, and BDI (Baltic Dry Index). "
            "Context: Shipbuilding stocks are sensitive to oil prices and BDI. "
            "Analyze the 120-day trend, focusing on volatility and correlations, "
            "and predict the next 10 days."
        )

        # 기타
        self.embed   = 'timeF'
        self.freq    = 'd'
        self.factor  = 1
        self.moving_avg = 25
        self.e_layers = 2
        self.d_layers = 1
        self.top_k    = 5

def load_time_llm_model(device):
    configs = Configs()  
    model = TimeLLM(configs).to(device).float()

    ckpt_path = os.path.join(PROJECT_ROOT, "models", "ship_time_llm_tmp6_ft_h3_e5.pth")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, configs