# -*- coding: utf-8 -*-
"""04. 검증 지도: 도로망(새벽/퇴근 속도) + 자치구 경계 + 후보지"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, RES, proc, res  # noqa: E402
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import font_manager

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f or "NotoSerifCJK" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = [f.name for f in font_manager.fontManager.ttflist if "CJK KR" in f.name][:1] or ["Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

OUT = PROC
N = pd.read_csv(os.path.join(OUT, "network/nodes.csv")).set_index("node_id")
E = pd.read_csv(os.path.join(OUT, "network/edges.csv"), usecols=["u", "v", "sp25_h18", "sp25_r02_05", "road_rank"])
C = pd.read_csv(os.path.join(OUT, "candidates/candidates.csv"))
gu = json.load(open(os.path.join(OUT, "boundary/gu.geojson"), encoding="utf-8"))

seg = np.stack([N.loc[E.u, ["lon", "lat"]].values, N.loc[E.v, ["lon", "lat"]].values], axis=1)


def draw_gu(ax):
    for f in gu["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = np.array(poly[0])
            ax.plot(ring[:, 0], ring[:, 1], color="#666", lw=0.6, zorder=3)


fig, axes = plt.subplots(1, 3, figsize=(24, 8.6), dpi=110)
for ax, col, title in [(axes[0], "sp25_r02_05", "새벽(02~05시) 평균속도"),
                       (axes[1], "sp25_h18", "퇴근(18~19시) 평균속도")]:
    lc = LineCollection(seg, array=E[col].clip(5, 80).values, cmap="RdYlGn", lw=0.5, zorder=2)
    ax.add_collection(lc)
    draw_gu(ax)
    ax.set_title(title + " (km/h, 2025 평일)", fontsize=14)
    fig.colorbar(lc, ax=ax, shrink=0.6)

ax = axes[2]
ax.add_collection(LineCollection(seg, color="#ddd", lw=0.3, zorder=1))
draw_gu(ax)
style = {"주유소_영업": ("#1f77b4", "o"), "주유소_폐업": ("#d62728", "X"), "주유소_휴업": ("#ff7f0e", "X"),
         "공영주차장": ("#2ca02c", "s"), "기존창고": ("#9467bd", "^")}
for t, (c, m) in style.items():
    d = C[C.type == t]
    real = ~d.coord_approx.astype(bool)
    ax.scatter(d[real].lon, d[real].lat, s=18, c=c, marker=m, label="%s (%d)" % (t, len(d)), zorder=4,
               edgecolors="white", linewidths=0.3)
    ax.scatter(d[~real].lon, d[~real].lat, s=22, facecolors="none", edgecolors=c, marker=m, zorder=4,
               linewidths=0.9)
ax.set_title("후보지 914곳 (빈 기호 = 지오코딩 실패, 임시 좌표 5곳)", fontsize=14)
ax.legend(loc="lower left", fontsize=10, frameon=True)
for a in axes:
    a.set_xlim(126.76, 127.19); a.set_ylim(37.42, 37.71); a.set_aspect(1 / np.cos(np.radians(37.56)))
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
p = res("check_map.png"); plt.savefig(p, bbox_inches="tight"); print(p)
