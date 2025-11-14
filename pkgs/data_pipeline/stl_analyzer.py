import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

class Stl_decomposition:
    """
    시계열 데이터에서 추세(Trend)와 계절성(Seasonality)의 강도를 계산합니다.
    'Understanding and Enhancing...' 논문의 핵심 진단 로직입니다.
    """
    def __init__(self, data: pd.Series, period: int = None, whether_plot: bool = False):
        # NaN 값이 있으면 STL이 오류를 일으키므로, ffill/bfill로 채웁니다.
        self.train_data = data.ffill().bfill()
        self.period = period
        self.whether_plot = whether_plot
    
    def stl_decoposition(self):
        """
        STL을 실행하여 시계열을 3가지 요소로 분해합니다.
        """
        # 데이터 길이가 period의 2배수보다 작으면 오류가 발생하므로, 최소 길이를 보장합니다.
        if self.period and len(self.train_data) < 2 * self.period:
            print(f"Warning: Data length ({len(self.train_data)}) is too short for period ({self.period}). Skipping STL.")
            return None, None, None

        try:
            res = STL(self.train_data, period=self.period).fit()
            if self.whether_plot:
                res.plot()
            
            seas_data = res.seasonal
            trend_data = res.trend
            resid_data = res.resid
            return seas_data, trend_data, resid_data
        
        except ValueError as e:
            print(f"STL decomposition failed: {e}")
            return None, None, None
    
    def strength(self) -> tuple[float, float]:
        """
        추세 강도(QT)와 계절성 강도(QS)를 계산합니다.
        """
        seas_data, trend_data, resid_data = self.stl_decoposition()

        # 분해에 실패한 경우
        if resid_data is None:
            return 0.0, 0.0 # 강도 0으로 반환

        # 분산 계산 (ddof=1은 샘플 분산을 의미)
        resid_var = np.var(resid_data, ddof=1)
        
        # 0으로 나누는 것을 방지
        trend_resid_var = np.var(resid_data + trend_data, ddof=1)
        if trend_resid_var == 0:
            trend_strength = 1.0 # 분모가 0이면, 잔차가 0이라는 뜻이므로 강도 1
        else:
            trend_strength = max(0.0, 1 - resid_var / trend_resid_var)

        seas_resid_var = np.var(resid_data + seas_data, ddof=1)
        if seas_resid_var == 0:
            seas_strength = 1.0
        else:
            seas_strength = max(0.0, 1 - resid_var / seas_resid_var)
            
        return trend_strength, seas_strength