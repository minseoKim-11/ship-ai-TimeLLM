# 🚢 ship-ai: Time-LLM 기반 조선업 주가·재무 분석 엔진

> **목표**  
> 이 레포는 조선업 종목(한화오션, 삼성중공업, 현대중공업, HD한국조선해양 등)에 대해  
> **시계열 예측(Time-LLM)** + **워렌 버핏식 퀄리티/밸류에이션 분석**을 수행하는  
> 백엔드 엔진(API 서버) 입니다.
>
> 자연어 리포트는 별도 sLLM 서비스가 담당하고,  
> 이 엔진은 숫자·스코어·라벨만 반환하는 “정답 JSON 서버” 역할을 합니다.

---

## 📁 프로젝트 구조

```bash
ship-ai/
├─ .venv/                        # 가상환경 (VSCode 자동 생성)
│apps/
│├─ api/
│   ├─ __init__.py
│   ├─ serve.py                # FastAPI 진입점 (main 역할)
│   │
│   ├─ tool1_routes.py         # TOOL1 API 엔드포인트
│   ├─ tool1_schemas.py        # TOOL1 요청/응답 Pydantic schema
│   │
│   ├─ tool2_routes.py         # TOOL2 API 엔드포인트
│   ├─ tool2_schemas.py        # TOOL2 요청/응답 Pydantic schema
│   │
│   ├─ tool3_routes.py         # TOOL3 API 엔드포인트
│   ├─ tool3_schemas.py        # TOOL3 요청/응답 Pydantic schema
│   │
│   ├─ tool4_routes.py         # TOOL4 API 엔드포인트
│   ├─ tool4_schemas.py        # TOOL4 요청/응답 Pydantic schema
│ 
│└─ __pycache__/               # 자동 생성 캐시
│
├─ data/
│  └─processed/                # 원본/전처리 데이터 저장 위치
│
├─ engine/
│  ├─ __init__.py
│  ├─ build_financials_mapped.py # Tool4 재무 스냅샷 ETL
│  ├─ data_handler.py            # Time-LLM 입력 스케일링/윈도우 처리
│  ├─ time_llm_config.py         # Time-LLM Config 로더
│  ├─ tool1_stock_info.py        # TOOL1 엔진: 기본 시계열 조회
│  ├─ tool2_macro_pulse.py       # TOOL2 엔진: 업황 스냅샷 / Macro Pulse
│  ├─ tool3_forecast_gap.py      # TOOL3 엔진: 예측 대비 괴리 분석
│  └─ tool4_buffett_score.py     # TOOL4 엔진: 버핏식 퀄리티/밸류 평가
│
├─ external/
│  └─ ...                        # ( Time-LLM 서브모듈, 외부 의존 코드)
│
├─ logs/
│  └─ ...                        # 엔진 로그 (필요시)
│
├─ models/
│  └─ ship_time_llm_tmp6.pth     # 학습된 Time-LLM 체크포인트
│
├─ notebooks/
│  └─ ...                        # Jupyter 실험 노트북
│
├─ pkgs/
│  └─ ...                        # 별도 패키지, 실험 코드 등
│
├─ pretrained_models/
│  └─ ...                        # 외부 모델, 임베딩 등 (필요 시)
│
├─ .gitmodules                   # 외부 repo를 submodule로 포함한 경우
├─ .gitignore
└─ README.md
````

---

## 🧠 전체 아키텍처 개요

* **Time-LLM 엔진**

  * PatchTST + LLM 기반 다변량 시계열 예측 모델
  * 입력: 12채널 (log close, ret_1d, trading_volume_log, ROE, debt_ratio, BDI proxy, WTI 등)
  * 출력: 향후 10영업일 주가/수익률 예측 시계열

* **엔진 API (해당 레포)**

  * TOOL1 ~ TOOL4 숫자 전용 REST API
  * 모든 계산·스코어링은 여기서 수행
  * 자연어 문장은 생성하지 않음

* **sLLM 서비스 (별도 레포)**

  * Ollama / Llama / Qwen 등으로 동작
  * 엔진에서 반환한 JSON을 받아  리포트로 변환

---

## 🚀 실행 방법

### 1) 환경 준비

```bash
# (예시) conda 환경
conda create -n ship-ai python=3.11
conda activate ship-ai

# 의존성 설치
pip install -r requirements.txt
```

GPU + PyTorch 버전은 서버 환경에 맞게 직접 설치하는 걸 추천합니다.

---

### 2) 데이터 준비

아래 파일들이 `data/processed/`에 존재해야 합니다.

* `final_master_table_v2.csv`

  * Time-LLM 학습에 사용한 12채널 마스터 테이블 (as-of 처리 완료)
* `master.csv`

  * 원본 일별 OHLCV (date, open, high, low, close, trading_volume, ticker)
* (이후 스크립트로 생성)

  * `master_table_denorm.csv`
  * `master_table_for_t4.csv`

#### 2-1) master_table_denorm 생성 (TOOL1/2/3용)

```bash
cd /workspace/ship-ai
python engine/build_master_denorm.py
```

* `final_master_table_v2.csv` + `master.csv`를 조인해서
* 사용자 친화적인 값(원래 가격, 원래 ROE/부채비율 등)이 들어 있는
  `master_table_denorm.csv`를 생성합니다.
* 주요 컬럼:

  * `date`, `ticker`
  * `close`, `ret_1d`
  * `roe`, `debt_ratio`
  * `bdi_proxy`, `newbuild_proxy`, `wti` 등

#### 2-2) master_table_for_t4 생성 (TOOL4용)

```bash
python engine/build_financials_mapped.py
```

* 입력: `summary_quarter2.csv` (또는 기존 재무 요약 테이블)
* pykrx `get_market_fundamental`을 이용해 EPS, PER 수집
* 추가 가공:

  * `industry_per` (동일 분기 내 업종 평균 PER)
  * `per_3y_min` / `per_3y_max` (최근 3년 PER 밴드)
  * `roe_5y_avg` / `roe_5y_std` (최근 5년 ROE 평균/표준편차)
* 출력: `data/processed/master_table_for_t4.csv`

---

### 3) Time-LLM 체크포인트 준비

`models/ship_time_llm_tmp6.pth`에 학습된 체크포인트를 둡니다.
(경로를 바꾸고 싶다면 `engine/tool3_forecast_gap.py`에서 경로만 수정)

예시:

```python
# engine/tool3_forecast_gap.py 등
CKPT_PATH = os.path.join(PROJECT_ROOT, "models", "ship_time_llm_tmp6.pth")
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
```

---

### 4) FastAPI 서버 실행

```bash
cd /workspace/ship-ai
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

---

## 🔧 TOOL별 API 명세

### 1️⃣ TOOL1 — 기본 시계열 조회 (Stock Info)

> 특정 종목에 대해 **기간별 시계열 데이터**를 반환.
> 현재는 프론트에서 표로 보여줄 4개 채널만 강제 제공:

* 종가(`close`)
* 일일 수익률(`ret_1d`)
* ROE(`roe`)
* 부채비율(`debt_ratio`)

#### Endpoint

```http
GET /api/tool1/stock-info
```

#### Request Body
```json
{
  "ticker": "string"
}
```

#### Response (예시)

```json
{
  "ticker": "010620",
  "date_range": {
    "start": "2022-01-01",
    "end": "2022-12-31"
  },
  "time_series": {
    "close": [
      ["2022-01-03", 12950.0],
      ["2022-01-04", 13100.0]
    ],
    "ret_1d": [
      ["2022-01-04", 0.0123],
      ["2022-01-05", -0.0041]
    ],
    "roe": [
      ["2022-03-31", 0.158],
      ["2022-06-30", 0.161]
    ],
    "debt_ratio": [
      ["2022-03-31", 132.4],
      ["2022-06-30", 131.9]
    ]
  },
  "meta": {
    "source": "master_table_denorm.csv",
    "note": null
  }
}
```

---

### 2️⃣ TOOL2 — 조선업 업황 스냅샷 (Market Snapshot)

> BDI proxy, 신조선가 proxy 등 **조선업 사이클 지표**를 기반으로
> “지금 업황이 어느 국면인가?”를 정량화한 도구.

* 사용자는 별도 파라미터를 보내지 않고,
* 엔진이 가장 최신 as-of date를 자동 선택해 스냅샷을 반환.

#### Endpoint

```http
GET /api/tool2/market-snapshot
```

#### Response (예시)

```json
{
  "as_of_date": "2025-08-14",
  "inputs": {
    "bdi_proxy": 0.93,
    "newbuild_proxy": 1.24,
    "backlog_years": 2.38
  },
  "scores": {
    "cycle_phase": "LATE_UPCYCLE",
    "cycle_score": 72,
    "momentum_score": 65
  },
  "meta": {
    "source": "master_table_denorm.csv",
    "note": "최근 12개월 기준으로 상단 25% 수준의 BDI proxy"
  }
}
```

sLLM은 이 JSON을 보고 “업황 상단에 근접한 상승 사이클 후반부” 같은 자연어 리포트를 만들어주는 역할을 합니다.

---

### 3️⃣ TOOL3 — Time-LLM 주가 괴리 분석 (Forecast Gap)

> Time-LLM이 예측한 **향후 10영업일 주가/수익률**과
> 현재 실제 주가를 비교해서,
> “현재 가격이 LLM이 보는 공정가 대비 얼마나 괴리되어 있는지”를 스코어링합니다.

#### Endpoint

```http
POST /api/tool3/forecast-gap
```

#### Request Body 

```json
{
  "ticker": "string",
  "as_of_date": null,
  "horizon_days": 10
}
```

* `ticker`: 대상 종목
* `as_of_date`: 예측 기준일 ( 서버 측에서 가장 최근 일자로 사용)
* `horizon_days`: 예측 기간 (engine v1.0의 구조상 10으로 고정)

#### Response (예시)

```json
{
  "ticker": "010620",
  "as_of_date": "2025-08-14",
  "horizon_days": 10,
  "actual": {
    "last_close": 32500.0
  },
  "forecast": {
    "expected_path": [
      ["2025-08-15", 32710.0],
      ["2025-08-18", 32980.0]
    ],
    "expected_mean_price": 33050.0
  },
  "gap_analysis": {
    "price_gap_pct": 1.7,
    "gap_score": 68,
    "label": "SLIGHT_UNDERVALUE"
  },
  "meta": {
    "model_checkpoint": "ship_time_llm_tmp6.pth",
    "input_window_days": 120
  }
}
```

---

### 4️⃣ TOOL4 — Buffett Quality & Valuation Score

> **워렌 버핏의 퀄리티(ROE) + 밸류에이션(PER) 원칙**을
> 엔진 레벨에서 수치화한 도구입니다.

* ROE 레벨/안정성
* PER vs 업종 평균
* PER vs 최근 3년 밴드
* ROE-PER 밸런스
* 부채비율 기반 리스크 가중치

#### Endpoint

```http
GET /api/tool4/buffett-score
```

#### Request Body
```json
{
  "ticker": "string"
}
```

#### Response (실제 예시 포맷)

```json
{
  "ticker": "010140",
  "as_of_date": "2025-08-14",
  "inputs": {
    "roe": 5.883636535,
    "roe_5y_avg": -6.297899503150001,
    "roe_5y_std": 12.553151447958756,
    "debt_ratio": 286.3087918,
    "eps": 75,
    "per": 251.87,
    "industry_per": 95.864,
    "per_3y_min": 0,
    "per_3y_max": 251.87
  },
  "scores": {
    "quality": {
      "roe_level_score": 30,
      "roe_stability_score": 0,
      "debt_ratio_score": 5,
      "total_quality_score": 35
    },
    "valuation": {
      "per_vs_industry_score": 0,
      "per_vs_history_score": 0,
      "roe_per_balance_score": 10,
      "total_valuation_score": 10
    },
    "total_buffett_score": 45,
    "grade": "D"
  },
  "meta": {
    "data_source": "master_table_for_t4.csv",
    "missing_fields": [],
    "notes": [
      "ROE 15% 이상으로 질적으로 우수한 수익성을 보입니다."
    ]
  }
}
```
> ⚠️ 위 예시의 `notes` 내용은 실제 수치(ROE 5.88%)와 맞지 않으며,
> 엔진 로직과 일관되게 수정해 사용해야 합니다.
> (`notes`는 엔진이 간단한 내부 메모용으로 남기거나, 필요시 제거해도 됩니다.)

sLLM은 이 JSON을 받아서:

* 등급 요약 (`grade`, `total_buffett_score`)
* 퀄리티 설명 (`quality` 블록)
* 밸류에이션 설명 (`valuation` 블록)
* 리스크 설명 (`debt_ratio`, 필요시 Tool4 + Safety Guard 연계)
* 종합 코멘트

형태의 자연어 리포트를 만들어 사용자에게 보여줍니다.

---

## 📌 사용 종목 (조선업 6개 티커)
해당 프로젝트는 프로토타입이므로 스코프를 제한하여 조선업 상위 6개 기업을 대상으로 데이터를 수집하여 진행하였습니다.
이 데이터의 범위는 고도화 과정에서 확장 할 계획입니다.
이번 프로젝트에서 사용한 기업은 아래와 같습니다.

| 티커     | 기업명                          |
| ------ | ---------------------------- |
| 009540 | 현대중공업지주                      |
| 010140 | 삼성중공업                        |
| 010620 | 한화오션 (구 대우조선해양)              |
| 042660 | (조선/해양 기자재 관련 종목, 데이터 기준 포함) |
| 329180 | 현대중공업                        |
| 443060 | HD한국조선해양                     |

---

## 🧾 License

* 2025년 9월 - 12월 진행된 가천대학교 금융분석 수업을 계기로 컴퓨터공학과 김민서가 제작하게 된 엔진입니다.
* Time-LLM을 끌어와 파인튜닝을 진행하여 12월 5일 기준 체크포인트와 임계값을 지정하여 engine v1.0 생성
* 위에 적힌 TOOL 4가지를 설계 및 제작하였습니다.
* 2026년 1월부터 sLLM단의 프롬프트 설계 및 engine 2.0 제작 예정에 있습니다. 
