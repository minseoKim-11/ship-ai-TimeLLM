import os
import pandas as pd
from typing import Dict, Any, List, Optional

PROJECT_ROOT = "/workspace/ship-ai" 
FIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_table_for_t4.csv")

if os.path.exists(FIN_PATH):
    _fin_df = pd.read_csv(
        FIN_PATH,
        parse_dates=["as_of_date"],
        dtype={"ticker": str},
    )
else:
    print(f"[WARN] {FIN_PATH} not found. Loading empty DataFrame.")
    _fin_df = pd.DataFrame(columns=["ticker", "as_of_date"])


def _pick_latest_snapshot(ticker: str) -> Dict[str, Any]:
    """
    특정 ticker에 대해 가장 최신(날짜가 가장 큰) 스냅샷 한 줄 선택.
    데이터가 없으면 ValueError 발생.
    """
    t = str(ticker).zfill(6)

    # 해당 티커의 모든 데이터를 가져옴
    df_sub = _fin_df[
        _fin_df["ticker"] == t
    ].sort_values("as_of_date")

    if df_sub.empty:
        raise ValueError(f"No financial data found for ticker: {t}")

    # 가장 마지막 행(가장 최신 날짜) 선택
    row = df_sub.iloc[-1]

    # 있으면 쓰고, 없으면 None
    def get(col):
        return row[col] if col in df_sub.columns and pd.notnull(row[col]) else None

    return {
        "ticker": t,
        "as_of_date": row["as_of_date"],
        "roe": get("roe"),
        "roe_5y_avg": get("roe_5y_avg"),
        "roe_5y_std": get("roe_5y_std"),
        "debt_ratio": get("debt_ratio"),
        "eps": get("eps"),
        "per": get("per"),
        "industry_per": get("industry_per"),
        "per_3y_min": get("per_3y_min"),
        "per_3y_max": get("per_3y_max"),
    }


# ---------- Scoring helpers ----------

def _score_roe_level(roe: float | None) -> int:
    if roe is None:
        return 0
    if roe >= 0.20:
        return 30
    if roe >= 0.15:
        return 25
    if roe >= 0.10:
        return 18
    if roe >= 0.05:
        return 10
    return 0


def _score_roe_stability(roe_std: float | None) -> int:
    if roe_std is None:
        return 0
    if roe_std < 0.02:
        return 15
    if roe_std < 0.04:
        return 10
    if roe_std < 0.07:
        return 5
    return 0


def _score_debt_ratio(debt_ratio: float | None) -> int:
    if debt_ratio is None:
        return 0
    if debt_ratio <= 100:
        return 15
    if debt_ratio <= 200:
        return 10
    if debt_ratio <= 300:
        return 5
    return 0


def _score_per_vs_industry(per: float | None, industry_per: float | None, eps: float | None) -> int:
    if per is None or industry_per is None or industry_per <= 0:
        return 0
    if eps is None or eps <= 0 or per <= 0:
        # 의미 없는 PER
        return 0

    ratio = per / industry_per
    if ratio <= 0.7:
        return 20
    if ratio <= 1.0:
        return 15
    if ratio <= 1.3:
        return 8
    return 0


def _score_per_vs_history(per: float | None, per_min: float | None, per_max: float | None) -> int:
    if per is None or per_min is None or per_max is None:
        return 0
    if per_max <= per_min:
        return 0

    rng = per_max - per_min
    thresh1 = per_min + 0.3 * rng
    thresh2 = per_min + 0.6 * rng
    thresh3 = per_min + 0.9 * rng

    if per <= thresh1:
        return 10
    if per <= thresh2:
        return 6
    if per <= thresh3:
        return 3
    return 0


def _score_roe_per_balance(roe: float | None, per: float | None) -> int:
    if roe is None or per is None or per <= 0:
        return 0
    # ROE: 0.xx -> %
    growth_yield = (roe * 100.0) / per  # e.g. ROE 15%, PER 10 -> 1.5
    if growth_yield >= 1.0:
        return 10
    if growth_yield >= 0.6:
        return 6
    if growth_yield >= 0.3:
        return 3
    return 0


def _grade_from_total(total: int) -> str:
    if total >= 80:
        return "A"
    if total >= 65:
        return "B"
    if total >= 50:
        return "C"
    return "D"


# ---------- Public Engine API ----------

def compute_buffett_score_engine(ticker: str) -> Dict[str, Any]:
    """
    Tool4 엔진 메인 함수.
    - 입력: ticker
    - 로직: DB에 있는 해당 Ticker의 가장 최신 데이터를 조회하여 점수 계산
    - 출력: 버핏식 Quality + Valuation score 및 등급
    """

    # 날짜 인자 없이 호출 -> 내부에서 가장 최신 날짜 데이터 픽
    snap = _pick_latest_snapshot(ticker)

    roe = snap["roe"]
    roe_5y_std = snap["roe_5y_std"]
    debt_ratio = snap["debt_ratio"]
    per = snap["per"]
    eps = snap["eps"]
    industry_per = snap["industry_per"]
    per_3y_min = snap["per_3y_min"]
    per_3y_max = snap["per_3y_max"]

    # --- Quality ---
    roe_level_score = _score_roe_level(roe)
    roe_stab_score = _score_roe_stability(roe_5y_std)
    debt_score = _score_debt_ratio(debt_ratio)

    total_quality = roe_level_score + roe_stab_score + debt_score

    # --- Valuation ---
    per_vs_industry_score = _score_per_vs_industry(per, industry_per, eps)
    per_vs_hist_score = _score_per_vs_history(per, per_3y_min, per_3y_max)
    roe_per_balance_score = _score_roe_per_balance(roe, per)

    total_valuation = per_vs_industry_score + per_vs_hist_score + roe_per_balance_score

    # --- Total ---
    total_score = int(total_quality + total_valuation)
    grade = _grade_from_total(total_score)

    # 어떤 필드가 비어 있었는지 기록 (sLLM이 설명할 때 사용 가능)
    missing_fields: List[str] = []
    for key in ["roe", "roe_5y_avg", "roe_5y_std", "debt_ratio",
                "eps", "per", "industry_per", "per_3y_min", "per_3y_max"]:
        if snap.get(key) is None:
            missing_fields.append(key)

    response = {
        "ticker": snap["ticker"],
        "as_of_date": snap["as_of_date"].strftime("%Y-%m-%d"), # 조회된 최신 날짜 반환

        "inputs": {
            "roe": roe,
            "roe_5y_avg": snap["roe_5y_avg"],
            "roe_5y_std": roe_5y_std,
            "debt_ratio": debt_ratio,
            "eps": eps,
            "per": per,
            "industry_per": industry_per,
            "per_3y_min": per_3y_min,
            "per_3y_max": per_3y_max,
        },

        "scores": {
            "quality": {
                "roe_level_score": roe_level_score,
                "roe_stability_score": roe_stab_score,
                "debt_ratio_score": debt_score,
                "total_quality_score": total_quality,
            },
            "valuation": {
                "per_vs_industry_score": per_vs_industry_score,
                "per_vs_history_score": per_vs_hist_score,
                "roe_per_balance_score": roe_per_balance_score,
                "total_valuation_score": total_valuation,
            },
            "total_buffett_score": total_score,
            "grade": grade,
        },

        "meta": {
            "data_source": os.path.basename(FIN_PATH),
            "missing_fields": missing_fields,
            "notes": [],
        }
    }

    # 간단한 note 예시
    notes = []
    if per is not None and industry_per is not None and per < industry_per:
        notes.append("PER가 업종 평균보다 낮아 상대적으로 저평가 구간입니다.")
    if roe is not None and roe >= 0.15:
        notes.append("ROE 15% 이상으로 질적으로 우수한 수익성을 보입니다.")
    if debt_ratio is not None and debt_ratio > 300:
        notes.append("부채비율이 매우 높아 재무 리스크가 클 수 있습니다.")

    response["meta"]["notes"] = notes

    return response