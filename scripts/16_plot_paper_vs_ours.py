# -*- coding: utf-8 -*-
"""16. 논문 방식(Method 3, 50개 권역)과 우리 기준안(C)의 신규 MFC 위치 비교 지도"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import PROC, res  # noqa: E402
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from pyproj import Transformer
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f or "Nanum" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = [f.name for f in font_manager.fontManager.ttflist
                               if "Sans CJK KR" in f.name or "Nanum" in f.name][:1] or ["sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#b9b8b3", "#e6e5e1", "#fcfcfb"
S1, S2 = "#2a78d6", "#eb6834"
LEVEL = int(_sys.argv[1]) if len(_sys.argv) > 1 else 50

C = pd.read_csv(os.path.join(PROC, "candidates", "candidates.csv"))
PM = pd.read_csv(res("paper_compare", "paper_m3_selected.csv"))
SC = pd.read_csv(res("final", "selected_C.csv"))
X = pd.read_csv(res("paper_compare", "cross_eval.csv"), index_col=0)
Z = pd.read_csv(os.path.join(PROC, "demand", "수요_2027_행정동.csv"))
gu = json.load(open(os.path.join(PROC, "boundary", "gu.geojson"), encoding="utf-8"))
zlon, zlat = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True).transform(Z.rep_x.values, Z.rep_y.values)
ex = SC[SC.existing]
paper = C[C.cand_id.isin(PM[PM.권역수 == LEVEL].cand_id)]
ours = SC[~SC.existing]
both = set(paper.cand_id) & set(ours.cand_id)
col_p = [c for c in X.columns if "P%d" % LEVEL in c][0]; col_c = [c for c in X.columns if "우리 C" in c][0]


def base(ax):
    for f in gu["features"]:
        g = f["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polys:
            r = np.array(p[0]); ax.plot(r[:, 0], r[:, 1], color=MUTED, lw=.6, zorder=1)
    ax.scatter(zlon, zlat, s=Z["d_기준_평상일"] / 500, color=GRID, zorder=2)
    ax.scatter(ex.lon, ex.lat, marker="^", s=60, color=INK2, edgecolor=SURF, lw=1.2, zorder=4)
    ax.set_aspect(1.25); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(SURF)
    for s_ in ax.spines.values(): s_.set_visible(False)


def stat(col):
    t = X.loc["[논문 지표] 평균 이동시간, 2021 6개 시간대 (분)", col]
    r = X.loc["[우리 지표] 서울 센터 처리 비율(%)", col]
    s = X.loc["[우리 지표] 현상유지 대비 절감(억/년)", col]
    return "평균 이동시간 %.1f분 · 서울 처리 %.0f%% · 연 %s억 원 %s" % (
        t, r, format(abs(round(s)), ","), "절감" if s > 0 else "증가")


fig = plt.figure(figsize=(18, 8.2), dpi=130, facecolor=SURF)
a1 = fig.add_axes([0.01, 0.1, 0.48, 0.78]); a2 = fig.add_axes([0.51, 0.1, 0.48, 0.78])
base(a1); base(a2)
a1.scatter(paper.lon, paper.lat, marker="o", s=70, color=S2, edgecolor=SURF, lw=1.5, zorder=5)
a1.set_title("논문 방식: %d개 권역마다 주유소 1곳 (Method 3)\n신규 %d곳 · %s" % (LEVEL, len(paper), stat(col_p)),
             loc="left", fontsize=13, color=INK)
po = ours[ours.type == "공영주차장"]; go = ours[ours.type != "공영주차장"]
a2.scatter(po.lon, po.lat, marker="s", s=55, color=S1, edgecolor=SURF, lw=1.5, zorder=5)
a2.scatter(go.lon, go.lat, marker="o", s=70, color=S1, edgecolor=SURF, lw=1.5, zorder=5)
a2.set_title("우리 기준안 C: 비용·용량·20분 서비스 수준 최적화\n신규 %d곳 (공영주차장 %d · 주유소 %d) · %s" % (
    len(ours), len(po), len(go), stat(col_c)), loc="left", fontsize=13, color=INK)
leg = [Line2D([], [], marker="^", ls="", color=INK2, ms=9, label="기존 택배센터 (27, 두 안 공통)"),
       Line2D([], [], marker="o", ls="", color=S2, ms=9, label="논문 방식 신규: 주유소"),
       Line2D([], [], marker="s", ls="", color=S1, ms=8, label="우리 신규: 공영주차장"),
       Line2D([], [], marker="o", ls="", color=S1, ms=9, label="우리 신규: 주유소"),
       Line2D([], [], marker="o", ls="", color=GRID, ms=9, label="행정동 수요 (원 크기)")]
fig.legend(handles=leg, loc="lower center", ncol=5, frameon=False, fontsize=11, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
fig.text(0.5, 0.075, "두 안이 같이 고른 주유소: %d곳" % len(both), ha="center", fontsize=10.5, color=INK2)
out = res("paper_compare", "map_paper_vs_ours_P%d.png" % LEVEL)
plt.savefig(out, facecolor=SURF); print(out, "공통", len(both))
print(paper.gu.value_counts().head(5).to_dict(), ours.gu.value_counts().head(5).to_dict())
