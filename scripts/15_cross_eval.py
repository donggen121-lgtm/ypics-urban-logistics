# -*- coding: utf-8 -*-
"""15. 교차 평가 — 논문 방식 해와 우리 해를 서로의 지표로 평가

비교 대상 (모두 기존 택배센터 27곳은 그대로 운영한다고 본다)
  현상유지   기존 27곳만
  A          우리 모형, 후보 = 영업 주유소            (12_run_compare.py 결과)
  C          우리 모형, 후보 = 주유소 전체 + 공영주차장 (12_run_compare.py 결과, 기준안)
  논문 P50   논문 Method 3, 50개 권역 → 주유소 50곳   (14_paper_replication.py 결과)
  논문 P92   논문 Method 3, 92개 권역 → 주유소 최대 92곳 (C안 신규 센터 수와 맞춤)

지표
  우리 지표  12번 모형으로 개설을 고정하고 배정·냉장만 다시 풀어 연간 총비용, 서울 처리 비율, 20분 커버를 계산
  논문 지표  각 행정동 수요를 가장 가까운 센터가 처리한다고 볼 때(용량 무시) 수요 가중 평균 이동시간
             View-T 2021 6개 시간대 평균(논문 조건)과 2025 평일 11시(우리 조건)
출력  results/paper_compare/cross_eval.csv, cross_eval.png
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE)); _sys.path.insert(0, _HERE)
from config import PROC, res  # noqa: E402
import os, importlib
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

m12 = importlib.import_module("12_run_compare")
OUT = res("paper_compare")
FIN = res("final")
SLOTS21 = ["tt21_h07", "tt21_h08", "tt21_h11", "tt21_h12", "tt21_h17", "tt21_h18"]

# ---------- 해 목록 ----------
selA = pd.read_csv(os.path.join(FIN, "selected_A.csv"))
selC = pd.read_csv(os.path.join(FIN, "selected_C.csv"))
PM3 = pd.read_csv(os.path.join(OUT, "paper_m3_selected.csv"))
SOL = {
    "현상유지 (기존 27곳)": [],
    "논문 P50 (주유소 50)": sorted(set(PM3[PM3.권역수 == 50].cand_id)),
    "논문 P92 (주유소 92)": sorted(set(PM3[PM3.권역수 == 92].cand_id)),
    "우리 A (영업 주유소 후보)": selA[~selA.existing].cand_id.tolist(),
    "우리 C (주유소+공영주차장, 기준안)": selC[~selC.existing].cand_id.tolist(),
}

# ---------- 우리 지표: 개설 고정 후 12번 모형으로 평가 ----------
F = m12.make_F(m12.SETS["C"])
ours, sels = {}, {}
for k, ids in SOL.items():
    if not ids:   # 현상유지: 12번과 같은 방식 (커버 제약 없음)
        ours[k], sels[k] = m12.solve(m12.make_F([]), statusquo=True, label=k[:8]); continue
    Fk = F[F.existing | F.cand_id.isin(ids)].reset_index(drop=True)   # 개설한 센터만 남겨 배정 후보로 쓴다
    ours[k], sels[k] = m12.solve(Fk, label=k[:8], fixed_new=ids)

# ---------- 논문 지표: 가장 가까운 센터까지 수요 가중 평균 이동시간 ----------
N = pd.read_csv(os.path.join(PROC, "network", "nodes.csv"))
E = pd.read_csv(os.path.join(PROC, "network", "edges.csv"), usecols=["u", "v", "tt25_h11"] + SLOTS21)
Z = pd.read_csv(os.path.join(PROC, "demand", "수요_2027_행정동.csv"))
ZN = pd.read_csv(os.path.join(PROC, "model", "zone_nodes.csv"))
idx = pd.Series(np.arange(len(N)), index=N.node_id)
zn = idx[Z[["행정동코드"]].merge(ZN, on="행정동코드").node_id].values
w = Z["d_기준_평상일"].values
G = {c: sp.csr_matrix((E[c].values, (idx[E.u].values, idx[E.v].values)), shape=(len(N), len(N)))
     for c in SLOTS21 + ["tt25_h11"]}
paper = {}
for k, s in sels.items():
    src = np.unique(idx[s.node_id.astype(int)].values)
    t21 = np.mean([dijkstra(G[c], directed=True, indices=src)[:, zn].min(0) for c in SLOTS21], axis=0)
    t25 = dijkstra(G["tt25_h11"], directed=True, indices=src)[:, zn].min(0)
    paper[k] = {"[논문 지표] 평균 이동시간, 2021 6개 시간대 (분)": w @ t21 / w.sum(),
                "[논문 지표] 평균 이동시간, 2025 11시 (분)": w @ t25 / w.sum(),
                "[논문 지표] 10분 넘는 수요 비율 (%)": w[t25 > 10].sum() / w.sum() * 100}

keep = ["신규 센터 수", "  공영주차장", "  주유소(영업)", "  주유소(휴업·폐업)", "총비용(억/년)", "서울 센터 처리 비율(%)",
        "신선 서울 처리 비율(%)", "20분 커버 불가 동", "1개당 비용(원)"]
X = pd.concat([pd.DataFrame(ours).loc[keep].rename(index=lambda r: "[우리 지표] " + r.strip()),
               pd.DataFrame(paper)])
sq = X.loc["[우리 지표] 총비용(억/년)", "현상유지 (기존 27곳)"]
X.loc["[우리 지표] 현상유지 대비 절감(억/년)"] = sq - X.loc["[우리 지표] 총비용(억/년)"]
X = X.astype(float).round(2)
X.to_csv(os.path.join(OUT, "cross_eval.csv"), encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(X.to_string())

# ---------- 그림 ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f or "Nanum" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = [f.name for f in font_manager.fontManager.ttflist
                               if "Sans CJK KR" in f.name or "Nanum" in f.name][:1] or ["sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, S1, S2, MUTE = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#eb6834", "#b9b8b3"
names = ["현상유지\n(기존 27곳)", "논문 P50\n(주유소 50)", "논문 P92\n(주유소 92)", "우리 A\n(영업 주유소)", "우리 C\n(기준안)"]
col = [MUTE, S2, S2, S1, S1]
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), dpi=130)
for a_, row, title, fmt in [
        (ax[0], "[논문 지표] 평균 이동시간, 2021 6개 시간대 (분)", "논문 지표: 가장 가까운 센터까지 평균 이동시간 (분)", "%.1f분"),
        (ax[1], "[우리 지표] 서울 센터 처리 비율(%)", "우리 지표: 서울 안 센터가 처리하는 물량 (%)", "%.0f%%"),
        (ax[2], "[우리 지표] 현상유지 대비 절감(억/년)", "우리 지표: 현상유지 대비 연간 절감액 (억 원)", "%.0f억")]:
    v = X.loc[row].values
    b = a_.bar(range(5), v, color=col, edgecolor="white", lw=2, width=.62)
    for i, x in enumerate(v):
        a_.text(i, x, fmt % x, ha="center", va="bottom" if x >= 0 else "top", fontsize=11, color=INK)
    a_.set_xticks(range(5)); a_.set_xticklabels(names, fontsize=10, color=INK2)
    a_.set_title(title, fontsize=12.5, color=INK, loc="left")
    a_.grid(axis="y", color=GRID); a_.set_axisbelow(True)
    for s_ in ["top", "right", "left"]: a_.spines[s_].set_visible(False)
    a_.tick_params(axis="y", colors=INK2, length=0); a_.spines["bottom"].set_color(MUTE)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "cross_eval.png"))
print("ok")
