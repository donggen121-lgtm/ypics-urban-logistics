# -*- coding: utf-8 -*-
"""07c. 팀원 구별 수요예측(Colab, 데이터 단위)을 실제 시장 단위로 환산해 2018~2027 경로를 만든다.

팀원 노트북 「구별_수요예측」 요약
  · 입력: 서울 생활물류(CJ대한통운 착지) 2018~2023, 구 × 일
  · 2022 선정 · 2023 평가로 SNAIVE 채택 → 2027 구 × 월 = 2023 구 × 월 (서울 연 309,723,575, 하루 848,558)
  · 단위는 '데이터 단위'(한 택배사 물량). 시장 전체가 아니고, 성장도 0으로 고정됨

환산 방법
  시장_서울(y) = 전국 택배 물동량(y) × 서울 비중(y)          ← 07과 같은 근거
  포착률 c(y) = 데이터_서울(y) / 시장_서울(y)                ← 2018~2023 실측, 2024~2027은 팀원 예측으로 계산
  시장_구·월(2027) = 팀원_구·월(2027) × 시장_서울(2027) / 팀원_서울(2027)
  → 총량은 전국 통계, 구 · 월 분포는 팀원 예측(2023 패턴)
  평상일 · 피크일은 07과 같은 정의(2023 평일 평균 / 연평균, 구별 평일 95분위)

입력  data/processed/demand/생활물류_일별_구_품목.csv, 수요_2027_자치구.csv (07 결과)
출력  data/processed/demand_teammate/서울_연도별_시장환산.csv, 구별_2027_시장환산.csv, 구월별_2027_시장환산_기준.csv,
      환산_그림.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import PROC, proc  # noqa: E402
import os
import numpy as np, pandas as pd

SRC = os.path.join(PROC, "demand", "생활물류_일별_구_품목.csv")
OUT = proc("demand_teammate")
NAT = pd.Series({2018: 25.4, 2019: 27.9, 2020: 33.7, 2021: 36.3, 2022: 42.1,
                 2023: 51.58, 2024: 59.56, 2025: 64.18})           # 억 개, 07과 동일
SHARE_OBS = {2018: 20.8, 2019: 20.7, 2023: 19.5}                     # 서울 비중(%), 07과 동일
TEAM_2027_TOTAL = 309_723_575                                         # 팀원 노트북 셀 20 결과 (검산용)

# ---------- 1. 팀원 예측 재현 (SNAIVE: 2027 구×월 = 2023 구×월) ----------
A = pd.read_csv(SRC)
CAT = list(A.columns[3:14])                                          # 11개 품목 (서울발 열 제외)
A["tot"] = A[CAT].sum(axis=1)
A["d"] = pd.to_datetime(A.배송년월일.astype(str)); A["yr"] = A.d.dt.year; A["m"] = A.d.dt.month
assert int(A.tot.sum()) == 1_874_355_796, "팀원 노트북의 원자료 합계와 다름"
DATA_Y = A.groupby("yr").tot.sum()
GM = A[A.yr == 2023].groupby(["수하인_구코드", "수하인_구명", "m"]).tot.sum().rename("team_2027").reset_index()
assert int(GM.team_2027.sum()) == TEAM_2027_TOTAL, "팀원 2027 서울 합계와 다름"

# ---------- 2. 시장 단위 서울 총량 (연도별) ----------
G07 = pd.read_csv(os.path.join(PROC, "demand", "수요_2027_자치구.csv"))
SEOUL27 = {sc: G07["연간_" + sc].sum() for sc in ["저성장", "기준", "고성장"]}   # 07의 2027 서울 시장 총량
SHARE27 = SEOUL27["기준"] / (NAT.loc[2025] * (1 + (NAT.loc[2025] / NAT.loc[2024] - 1)) ** 2 * 1e8) * 100
share = pd.Series(SHARE_OBS, dtype=float).reindex(range(2018, 2028))
share.loc[2027] = SHARE27
share = share.interpolate()

rows = []
for y in range(2018, 2028):
    days = 366 if y % 4 == 0 else 365
    data = DATA_Y.get(y, TEAM_2027_TOTAL if y >= 2024 else np.nan)  # 2024~2027 데이터 단위 = 팀원 예측(2023 유지)
    for sc in (["실적"] if y <= 2025 else ["저성장", "기준", "고성장"]):
        if y <= 2025:
            mk = NAT.loc[y] * 1e8 * share.loc[y] / 100
        else:  # 2026은 2025 실적과 2027 시나리오의 기하평균
            mk27 = SEOUL27[sc]; mk25 = NAT.loc[2025] * 1e8 * share.loc[2025] / 100
            mk = mk27 if y == 2027 else np.sqrt(mk25 * mk27)
        rows.append({"연도": y, "시나리오": sc, "전국(억 개)": round(mk / share.loc[y] * 100 / 1e8, 2),
                     "서울 비중(%)": round(share.loc[y], 2), "서울_시장_연간": round(mk), "서울_시장_하루평균": round(mk / days),
                     "데이터단위_연간": int(data), "데이터단위_하루평균": round(data / days),
                     "포착률(%)": round(data / mk * 100, 1), "환산계수": round(mk / data, 3),
                     "구분": "실측" if y <= 2023 else ("팀원 예측(데이터 2023 유지)")})
Y = pd.DataFrame(rows)
Y.to_csv(os.path.join(OUT, "서울_연도별_시장환산.csv"), index=False, encoding="utf-8-sig")

# ---------- 3. 2027 구별 · 구월별 시장 환산 ----------
D23 = A[A.yr == 2023]
Dd = D23.groupby("d").tot.sum(); wd = Dd[Dd.index.dayofweek < 5]; wd = wd[wd > 0.4 * wd.median()]
R_WD = wd.mean() / (Dd.sum() / 365)                                  # 07과 같은 정의
DG = D23.pivot_table(index="d", columns="수하인_구코드", values="tot", aggfunc="sum").loc[wd.index]
PEAK_GU = DG.quantile(0.95) / DG.mean()

g = GM.groupby(["수하인_구코드", "수하인_구명"]).team_2027.sum().reset_index()
g["팀원_비중"] = g.team_2027 / g.team_2027.sum()
g = g.merge(G07[["구코드", "비중", "평상일_기준"]].rename(columns={"구코드": "수하인_구코드", "비중": "07_비중", "평상일_기준": "07_평상일_기준"}),
            on="수하인_구코드")
g["피크계수"] = g.수하인_구코드.map(PEAK_GU)
for sc, tot in SEOUL27.items():
    k = tot / TEAM_2027_TOTAL
    g["연간_" + sc] = np.round(g.team_2027 * k)
    g["평상일_" + sc] = np.round(g.team_2027 * k / 365 * R_WD)
    g["피크일_" + sc] = np.round(g["평상일_" + sc] * g.피크계수)
g["평상일_차이(07대비,%)"] = np.round((g.평상일_기준 / g["07_평상일_기준"] - 1) * 100, 2)
g = g.rename(columns={"수하인_구코드": "구코드", "수하인_구명": "gu_nm", "team_2027": "팀원_2027_데이터단위"})
g.to_csv(os.path.join(OUT, "구별_2027_시장환산.csv"), index=False, encoding="utf-8-sig")

k_base = SEOUL27["기준"] / TEAM_2027_TOTAL
GM["시장_기준"] = np.round(GM.team_2027 * k_base)
GM.rename(columns={"수하인_구코드": "구코드", "수하인_구명": "gu_nm", "m": "월", "team_2027": "팀원_2027_데이터단위"}) \
  .to_csv(os.path.join(OUT, "구월별_2027_시장환산_기준.csv"), index=False, encoding="utf-8-sig")

print(Y.to_string(index=False))
print("\n평일/연평균 %.3f · 환산계수(기준) %.3f" % (R_WD, k_base))
print("서울 2027 평상일(기준): 팀원 환산 %s / 07 %s" % (format(int(g.평상일_기준.sum()), ","), format(int(g["07_평상일_기준"].sum()), ",")))
print(g[["gu_nm", "팀원_비중", "07_비중", "평상일_기준", "평상일_차이(07대비,%)"]].sort_values("평상일_차이(07대비,%)").to_string(index=False))

# ---------- 4. 그림 ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f or "Nanum" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = [f.name for f in font_manager.fontManager.ttflist
                               if "Sans CJK KR" in f.name or "Nanum" in f.name][:1] or ["sans-serif"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8f98", "#e6e5e1", "#ffffff"
S1, S2, S3 = "#1E4FB4", "#eb6834", "#9FB6E8"
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), dpi=140, facecolor=SURF, gridspec_kw={"width_ratios": [1.05, 1]})
a = ax[0]
h = Y[Y.시나리오.isin(["실적", "기준"])].set_index("연도")
a.plot(h.index, h.서울_시장_하루평균 / 1e4, "o-", color=S1, lw=2.2, label="시장 단위 (전국 × 서울 비중)")
for sc, ls in [("저성장", ":"), ("고성장", "--")]:
    s = Y[(Y.시나리오 == sc)].set_index("연도").서울_시장_하루평균 / 1e4
    a.plot([2025] + list(s.index), [h.loc[2025, "서울_시장_하루평균"] / 1e4] + list(s.values), ls, color=S1, lw=1.3, alpha=.7)
a.plot(h.index, h.데이터단위_하루평균 / 1e4, "o-", color=S2, lw=2.2, label="데이터 단위 (팀원 예측, 2024~ = 2023 유지)")
a.axvspan(2023.5, 2027.5, color=GRID, alpha=.5, lw=0)
for y in [2018, 2023, 2027]:
    a.annotate("포착률 %.0f%%" % h.loc[y, "포착률(%)"], (y, h.loc[y, "데이터단위_하루평균"] / 1e4), textcoords="offset points",
               xytext=(0, -18), ha="center", fontsize=9.5, color=S2)
    a.annotate("%.0f만" % (h.loc[y, "서울_시장_하루평균"] / 1e4), (y, h.loc[y, "서울_시장_하루평균"] / 1e4), textcoords="offset points",
               xytext=(0, 8), ha="center", fontsize=10, color=S1, fontweight="bold")
a.set_title("서울 택배 하루 평균 (만 개): 데이터 단위 → 시장 단위", loc="left", fontsize=13, color=INK)
a.set_xticks(range(2018, 2028)); a.set_ylim(0, 480); a.grid(axis="y", color=GRID); a.legend(frameon=False, fontsize=10, loc="upper left")
a.text(2025.5, 20, "데이터 단위는 팀원 예측 · 2026~27 시장은 시나리오", ha="center", color=MUTED, fontsize=8.5)
for s_ in ["top", "right"]: a.spines[s_].set_visible(False)
b = ax[1]
gg = g.sort_values("평상일_기준")
b.barh(gg.gu_nm, gg.평상일_기준 / 1e4, color=S1, height=.62, label="팀원 분포 × 시장 총량")
b.scatter(gg["07_평상일_기준"] / 1e4, gg.gu_nm, marker="|", s=160, color=S2, lw=2.2, zorder=3, label="현재 모형 (07, 인구 증감 반영)")
b.set_title("2027 구별 평상일 수요 (만 개/일, 기준 시나리오)", loc="left", fontsize=13, color=INK)
b.tick_params(axis="y", labelsize=9.5); b.grid(axis="x", color=GRID); b.legend(frameon=False, fontsize=10, loc="lower right")
for s_ in ["top", "right"]: b.spines[s_].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "환산_그림.png"), facecolor=SURF)
print(os.path.join(OUT, "환산_그림.png"))
