# -*- coding: utf-8 -*-
"""07. 2027년 수요 예측 → 행정동 배분

구조
  서울 2027 연간 물동량 = 전국 택배 물동량(2027 예측) × 서울 착지 비중
  자치구 비중            = 생활물류(CJ) 2023 자치구 비중 × 장래인구 증감 보정 (2023 검증으로 방법 선택)
  하루 수요              = 연간 ÷ 365 × (평일/연평균 비율)          → 평상일
                           × 피크계수(평일 상위 5% / 평일 평균)       → 피크일
  행정동                 = 자치구 수요 × 구내_야간비중 (행정동 생활인구)
  신선식품               = 자치구별 '식품' 비중 × 신선 비율(온라인 식품 중 농축수산물 26.7%)

입력  전처리결과/demand/생활물류_일별_구_품목.csv (CJ 착지 일별, 2018~2023)
      장래인구_자치구_2022_2042.csv, 행정동_야간인구_202607.csv
출력  demand/수요_2027_행정동.csv, 수요_2027_자치구.csv, 수요예측_검증.csv, forecast_chart.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, RES, proc, res  # noqa: E402
import os
import numpy as np, pandas as pd

OUT = proc("demand")
SRC = os.path.join(OUT, "생활물류_일별_구_품목.csv")   # 07a_life_logistics.py 결과
TARGET = 2027

# ---------- 외부 수치 ----------
# 전국 택배 물동량(억 개). 국가물류통합정보센터·한국통합물류협회 생활물류실적
NAT = pd.Series({2018: 25.4, 2019: 27.9, 2020: 33.7, 2021: 36.3, 2022: 42.1,
                 2023: 51.58, 2024: 59.56, 2025: 64.18})
# CJ 전국 대비 서울 착지 비중(%) — CJ 공시 물량 대비 생활물류 데이터 서울 합계
SEOUL_SHARE = pd.Series({2018: 20.8, 2019: 20.7, 2023: 19.5})
FRESH_IN_FOOD = 10.85 / 40.69   # 2023 온라인 식품 거래액 중 농축수산물 (통계청 온라인쇼핑동향)

A = pd.read_csv(SRC)
A["d"] = pd.to_datetime(A.배송년월일.astype(str))
CAT = list(A.columns[3:14])
A["tot"] = A[CAT].sum(axis=1)
A["yr"] = A.d.dt.year
gu_nm = A.drop_duplicates("수하인_구코드").set_index("수하인_구코드").수하인_구명

V = []  # 검증 기록

# ---------- 1. 전국 물동량 예측 ----------
def holt_damped(y, h, phi=0.9):
    """감쇠 추세 Holt(로그). α,β 격자 탐색으로 1-step 오차제곱합 최소."""
    ly = np.log(y.values); best = None
    for a in np.linspace(0.1, 1, 19):
        for b in np.linspace(0.05, 1, 20):
            l, t, sse = ly[0], ly[1] - ly[0], 0
            for v in ly[1:]:
                f = l + phi * t; sse += (v - f) ** 2
                ln = a * v + (1 - a) * f; t = b * (ln - l) + (1 - b) * phi * t; l = ln
            if best is None or sse < best[0]:
                best = (sse, l, t)
    _, l, t = best
    return np.exp([l + sum(phi ** k for k in range(1, i + 1)) * t for i in range(1, h + 1)])

def loglin(y, h):
    b, a = np.polyfit(y.index, np.log(y.values), 1)
    return np.exp(a + b * np.arange(y.index[-1] + 1, y.index[-1] + h + 1))

def cagr3(y, h):
    g = (y.iloc[-1] / y.iloc[-4]) ** (1 / 3)
    return y.iloc[-1] * g ** np.arange(1, h + 1)

MODELS = {"감쇠추세(Holt)": holt_damped, "로그선형추세": loglin, "최근3년 CAGR": cagr3}
tr, te = NAT.loc[:2023], NAT.loc[2024:]
for nm, f in MODELS.items():
    p = f(tr, len(te))
    mape = np.mean(np.abs(p - te.values) / te.values) * 100
    V.append({"단계": "전국 물동량", "학습": "2018~2023", "검증": "2024~2025", "방법": nm,
              "MAPE(%)": round(mape, 2), "예측": " / ".join("%.1f" % v for v in p)})
# 검증(2024~25) MAPE는 세 모형 모두 3%대로 추세모형은 2025년까지 잘 맞았다.
# 다만 2025년 성장률이 7.8%로 급감(2023 22.5% → 2024 15.5% → 2025 7.8%)해서, 2018~2025 전체로 다시
# 적합한 감쇠추세도 연 11.7%를 내놓는다 → 이것을 '고성장'으로, 2025 성장률 유지를 '기준'으로 둔다.
best_nat = "감쇠추세(Holt)"
h = TARGET - NAT.index[-1]
g_last = NAT.iloc[-1] / NAT.iloc[-2] - 1                     # 2025 성장률 7.75%
NAT_SC = {"저성장": NAT.iloc[-1] * 1.04 ** h,                  # 연 4% (2025 성장률의 절반)
          "기준": NAT.iloc[-1] * (1 + g_last) ** h,            # 2025 성장률 유지
          "고성장": MODELS[best_nat](NAT, h)[-1]}               # 감쇠추세 모형
NAT_G = {k: (v / NAT.iloc[-1]) ** (1 / h) - 1 for k, v in NAT_SC.items()}

# ---------- 2. 서울 비중 ----------
b = np.polyfit(SEOUL_SHARE.index, SEOUL_SHARE.values, 1)
share_trend = np.polyval(b, TARGET)
SEOUL_PCT = (SEOUL_SHARE.loc[2023] + share_trend) / 2          # 2023 유지와 추세의 중간
SEOUL = {k: v * 1e8 * SEOUL_PCT / 100 for k, v in NAT_SC.items()}

# ---------- 3. 자치구 비중 (2018~2022 학습 → 2023 검증) ----------
Y = A.pivot_table(index="yr", columns="수하인_구코드", values="tot", aggfunc="sum")
S = Y.div(Y.sum(axis=1), axis=0)
P = pd.read_csv(os.path.join(OUT, "장래인구_자치구_2022_2042.csv")).set_index("자치구")
code_of = {v: k for k, v in gu_nm.items()}
P = P.drop(index="서울시").rename(index=code_of); P.columns = P.columns.astype(int)

def by_pop(s0, y0, y1):
    s = s0 * P[y1] / P[y0]
    return s / s.sum()

cand = {
    "직전년도 비중 유지": S.loc[2022],
    "비중 선형추세(2018~22)": pd.Series({c: np.polyval(np.polyfit(range(2018, 2023), S.loc[2018:2022, c], 1), 2023)
                                       for c in S.columns}),
    "직전년도 비중×인구증감": by_pop(S.loc[2022], 2022, 2023),
}
for nm, s in cand.items():
    s = s / s.sum()
    V.append({"단계": "자치구 비중", "학습": "2018~2022", "검증": "2023", "방법": nm,
              "MAPE(%)": round(float(np.mean(np.abs(s - S.loc[2023]) / S.loc[2023]) * 100), 2), "예측": ""})
best_gu = min([v for v in V if v["단계"] == "자치구 비중"], key=lambda v: v["MAPE(%)"])["방법"]
if best_gu == "비중 선형추세(2018~22)":
    sh = pd.Series({c: np.polyval(np.polyfit(range(2018, 2024), S.loc[2018:2023, c], 1), TARGET) for c in S.columns})
    GU_SHARE = (sh.clip(lower=0) / sh.clip(lower=0).sum())
elif best_gu == "직전년도 비중×인구증감":
    GU_SHARE = by_pop(S.loc[2023], 2023, TARGET)
else:
    GU_SHARE = S.loc[2023]

# ---------- 4. 하루 단위: 평일 비율, 피크계수 ----------
D = A[A.yr == 2023].groupby("d").tot.sum()
wd = D[D.index.dayofweek < 5]
wd = wd[wd > 0.4 * wd.median()]                                # 공휴일 제외
R_WD = wd.mean() / (D.sum() / 365)                             # 평일 하루 = 연평균 하루 × R_WD
PEAK = wd.quantile(0.95) / wd.mean()
DG = A[A.yr == 2023].pivot_table(index="d", columns="수하인_구코드", values="tot", aggfunc="sum").loc[wd.index]
PEAK_GU = DG.quantile(0.95) / DG.mean()

# 품목: 식품 비중(2023)
F = A[A.yr == 2023].groupby("수하인_구코드")[["식품", "tot"]].sum()
FOOD = F.식품 / F.tot

# ---------- 5. 자치구 → 행정동 ----------
G = pd.DataFrame({"구코드": GU_SHARE.index, "gu_nm": gu_nm.reindex(GU_SHARE.index).values,
                  "비중": GU_SHARE.values, "식품비중": FOOD.reindex(GU_SHARE.index).values,
                  "피크계수": PEAK_GU.reindex(GU_SHARE.index).values})
for sc, tot in SEOUL.items():
    G["연간_" + sc] = np.round(tot * G.비중, 0)
    G["평상일_" + sc] = np.round(tot / 365 * R_WD * G.비중, 0)
    G["피크일_" + sc] = np.round(G["평상일_" + sc] * G.피크계수, 0)
G.to_csv(os.path.join(OUT, "수요_2027_자치구.csv"), index=False, encoding="utf-8-sig")

H = pd.read_csv(os.path.join(OUT, "행정동_야간인구_202607.csv"))
H = H.merge(G, on="구코드", how="left", suffixes=("", "_g"))
Z = H[["행정동코드", "gu_nm", "hjd_nm", "구코드", "rep_x", "rep_y", "area_km2", "구내_야간비중"]].copy()
for sc in SEOUL:
    for dt in ["평상일", "피크일"]:
        Z["d_%s_%s" % (sc, dt)] = np.round(H["%s_%s" % (dt, sc)] * H.구내_야간비중, 1)
    Z["fresh_%s_평상일" % sc] = np.round(Z["d_%s_평상일" % sc] * H.식품비중 * FRESH_IN_FOOD, 1)
    Z["fresh_%s_피크일" % sc] = np.round(Z["d_%s_피크일" % sc] * H.식품비중 * FRESH_IN_FOOD, 1)
Z.to_csv(os.path.join(OUT, "수요_2027_행정동.csv"), index=False, encoding="utf-8-sig")

# 시나리오 표 (2단계 확률모형 입력)
SC = []
for sc, pg in [("저성장", .25), ("기준", .5), ("고성장", .25)]:
    for dt, pd_ in [("평상일", .95), ("피크일", .05)]:
        SC.append({"시나리오": "%s_%s" % (sc, dt), "성장": sc, "날유형": dt, "확률": pg * pd_,
                   "서울_하루합계": round(Z["d_%s_%s" % (sc, dt)].sum()),
                   "신선_하루합계": round(Z["fresh_%s_%s" % (sc, dt)].sum())})
pd.DataFrame(SC).to_csv(os.path.join(OUT, "시나리오_2027.csv"), index=False, encoding="utf-8-sig")

V.append({"단계": "결과", "학습": "", "검증": "", "방법": "선택: 전국=저4%%/기준%.2f%%/고=%s, 자치구=%s" % (g_last * 100, best_nat, best_gu),
          "MAPE(%)": np.nan, "예측": ""})
pd.DataFrame(V).to_csv(os.path.join(OUT, "수요예측_검증.csv"), index=False, encoding="utf-8-sig")

summ = {
    "전국 2027(억 개)": {k: round(v, 1) for k, v in NAT_SC.items()},
    "연 성장률(%)": {k: round(v * 100, 1) for k, v in NAT_G.items()},
    "서울 비중(%)": round(SEOUL_PCT, 2), "평일/연평균 비율": round(R_WD, 3), "피크계수(서울)": round(PEAK, 3),
    "신선/식품": round(FRESH_IN_FOOD, 3), "서울 식품비중": round(F.식품.sum() / F.tot.sum(), 3),
    "서울 연간(억 개)": {k: round(v / 1e8, 2) for k, v in SEOUL.items()},
}
print(pd.DataFrame(V).to_string(index=False)); print(summ); print(pd.DataFrame(SC).to_string(index=False))

# ---------- 그림 ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "Nanum" in f or "Noto" in f and "CJK" in f:
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = [n.name for n in fm.fontManager.ttflist if "Nanum" in n.name or "CJK" in n.name][:1] or ["sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
yrs = list(range(2026, TARGET + 1))
ax[0].plot(NAT.index, NAT.values, "o-", color="#222", label="실적")
for sc, c in [("저성장", "#999"), ("기준", "#1f77b4"), ("고성장", "#d62728")]:
    r = (NAT_SC[sc] / NAT.iloc[-1]) ** (1 / h)
    ax[0].plot([2025] + yrs, [NAT.iloc[-1] * r ** i for i in range(h + 1)], "--", color=c, label="%s %.1f" % (sc, NAT_SC[sc]))
ax[0].set_title("전국 택배 물동량 (억 개)"); ax[0].legend(); ax[0].grid(alpha=.3)
g = G.sort_values("평상일_기준")
ax[1].barh(g.gu_nm, g.평상일_기준 / 1e3, color="#1f77b4", label="평상일")
ax[1].barh(g.gu_nm, (g.피크일_기준 - g.평상일_기준) / 1e3, left=g.평상일_기준 / 1e3, color="#ff7f0e", label="피크 추가분")
ax[1].set_title("2027 기준 시나리오 자치구별 하루 수요 (천 개)"); ax[1].legend(); ax[1].tick_params(axis="y", labelsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "forecast_chart.png"), dpi=130)
