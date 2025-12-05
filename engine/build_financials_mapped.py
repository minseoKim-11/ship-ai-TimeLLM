import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
from pykrx import stock

PROJECT_ROOT = "/workspace/ship-ai"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SUMMARY_PATH = os.path.join(PROCESSED_DIR, "financials_mapped.csv")
OUT_PATH = os.path.join(PROCESSED_DIR, "master_table_for_t4.csv")

def load_base_summary() -> pd.DataFrame:
    """
    기존 summary_quarter2.csv를 로드해서
    - ticker: str (zfill(6))
    - as_of_date: datetime
    컬럼 정리.
    """
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(f"[ERROR] 파일이 없습니다: {SUMMARY_PATH}")

    df = pd.read_csv(
        SUMMARY_PATH,
        parse_dates=["as_of_date"],
        dtype={"ticker": str},
    )

    # 티커 6자리 zero-fill
    df["ticker"] = df["ticker"].str.zfill(6)

    # 컬럼 이름 통일: real_debt_ratio -> debt_ratio
    if "real_debt_ratio" in df.columns and "debt_ratio" not in df.columns:
        df = df.rename(columns={"real_debt_ratio": "debt_ratio"})

    # 필요 컬럼만 남기고 정리
    expected_cols = ["ticker", "as_of_date", "roe", "debt_ratio"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"[ERROR] SUMMARY 파일에 '{col}' 컬럼이 없습니다.")

    df_base = df[expected_cols].copy()

    # 날짜 & 티커 기준 정렬
    df_base = df_base.sort_values(["ticker", "as_of_date"]).reset_index(drop=True)

    print(f"[LOAD] financials_mapped.csv -> {df_base.shape} rows")
    return df_base


def fetch_eps_per_for_row(row) -> Tuple[float, float]:
    """
    [수정됨] 단일 행(row)에 대해 pykrx에서 EPS, PER를 조회.
    - 기존: stock.get_market_fundamental(ymd, ticker) -> 에러 발생 가능
    - 변경: stock.get_market_fundamental(ymd, ymd, ticker) -> 기간 조회 방식으로 에러 우회
    """
    date = row["as_of_date"]
    ticker = str(row["ticker"]).zfill(6)
    ymd = date.strftime("%Y%m%d")

    try:
        # [핵심 수정] 시작일과 종료일을 동일하게 넣어 '기간 조회' 로직을 태웁니다.
        # 이렇게 하면 반환되는 df_fund는 Index가 날짜, Columns가 [BPS, PER, PBR, EPS, DIV, DPS] 형태가 됩니다.
        df_fund = stock.get_market_fundamental(ymd, ymd, ticker)
    except Exception as e:
        print(f"[WARN] get_market_fundamental 호출 실패: {ticker}, {ymd} -> {e}")
        return np.nan, np.nan

    # 데이터가 비어있는 경우 (휴장일이거나 데이터 없음)
    if df_fund is None or df_fund.empty:
        # print(f"[INFO] 데이터 없음(휴장일 등): {ticker}, {ymd}") # 로그 너무 많으면 주석 처리
        return np.nan, np.nan

    # 기간 조회 결과는 항상 DataFrame 형태이며, 정상적이라면 1개의 행이 존재합니다.
    try:
        # 첫 번째 행(해당 날짜) 가져오기
        row_data = df_fund.iloc[0]
        
        # 컬럼 존재 여부 확인 후 값 추출 (없으면 NaN)
        eps_val = row_data['EPS'] if 'EPS' in df_fund.columns else np.nan
        per_val = row_data['PER'] if 'PER' in df_fund.columns else np.nan

        # 0인 경우도 데이터가 있는 것으로 간주할지, NaN 처리할지는 정책에 따름
        # 여기서는 단순히 float 변환만 수행
        eps = float(eps_val)
        per = float(per_val)
        
        return eps, per

    except Exception as e:
        print(f"[WARN] 데이터 파싱 에러: {ticker}, {ymd} -> {e}")
        return np.nan, np.nan


def add_eps_per(df: pd.DataFrame) -> pd.DataFrame:
    """
    pykrx로 EPS, PER를 붙여 넣기.
    """
    eps_list = []
    per_list = []

    print(f"[STEP] EPS, PER 수집 시작 (rows={len(df)})")

    for idx, row in df.iterrows():
        eps, per = fetch_eps_per_for_row(row)
        eps_list.append(eps)
        per_list.append(per)

        if (idx + 1) % 20 == 0:
            print(f"  - processed {idx+1}/{len(df)} rows...")
            time.sleep(0.1) # 서버 부하 방지용 딜레이

    df["eps"] = eps_list
    df["per"] = per_list

    print("[STEP] EPS, PER 수집 완료")
    return df


def add_industry_per(df: pd.DataFrame) -> pd.DataFrame:
    """
    industry_per = 같은 as_of_date 내에서 조선업 6개 ticker PER 평균
    """
    df["industry_per"] = (
        df.groupby("as_of_date")["per"]
          .transform(lambda s: s.mean(skipna=True))
    )
    print("[STEP] industry_per (동일 분기 내 평균 PER) 계산 완료")
    return df


def add_per_3y_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    per_3y_min, per_3y_max: 과거 3년(=12분기) rolling window
    """
    df = df.sort_values(["ticker", "as_of_date"]).reset_index(drop=True)

    def _per_rolling_min(s: pd.Series) -> pd.Series:
        return s.rolling(window=12, min_periods=3).min()

    def _per_rolling_max(s: pd.Series) -> pd.Series:
        return s.rolling(window=12, min_periods=3).max()

    df["per_3y_min"] = (
        df.groupby("ticker")["per"]
          .transform(_per_rolling_min)
    )
    df["per_3y_max"] = (
        df.groupby("ticker")["per"]
          .transform(_per_rolling_max)
    )

    print("[STEP] per_3y_min / per_3y_max 계산 완료")
    return df


def add_roe_5y_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    roe_5y_avg, roe_5y_std: 과거 5년(=20분기) rolling window
    """
    df = df.sort_values(["ticker", "as_of_date"]).reset_index(drop=True)

    def _roe_rolling_avg(s: pd.Series) -> pd.Series:
        return s.rolling(window=20, min_periods=4).mean()

    def _roe_rolling_std(s: pd.Series) -> pd.Series:
        return s.rolling(window=20, min_periods=4).std()

    df["roe_5y_avg"] = (
        df.groupby("ticker")["roe"]
          .transform(_roe_rolling_avg)
    )
    df["roe_5y_std"] = (
        df.groupby("ticker")["roe"]
          .transform(_roe_rolling_std)
    )

    print("[STEP] roe_5y_avg / roe_5y_std 계산 완료")
    return df


def main():
    # 1) 베이스 요약 로드 (ROE, 부채비율)
    df = load_base_summary()

    # 2) EPS / PER 추가 (pykrx) -> [수정됨] 기간 조회 방식 적용
    df = add_eps_per(df)

    # 3) 업종 평균 PER (우리 프로젝트 내 6개 ticker 평균)
    df = add_industry_per(df)

    # 4) 3년 rolling PER min/max
    df = add_per_3y_stats(df)

    # 5) 5년 rolling ROE avg/std
    df = add_roe_5y_stats(df)

    # 6) 저장
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[SAVE] master_table_for_t4.csv 저장 완료 -> {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()