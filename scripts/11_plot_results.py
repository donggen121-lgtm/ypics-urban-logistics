# -*- coding: utf-8 -*-
"""11. 최적화 결과 그림: 기준안 입지 지도 + U자 곡선 + 민감도"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, RES, proc, res  # noqa: E402
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f or "Nanum" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = [f.name for f in font_manager.fontManager.ttflist if "CJK KR" in f.name or "Nanum" in f.name][:1] or ["sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = PROC
R = res("exploratory")
S = pd.read_csv(os.path.join(R, "selected_base.csv"))
U = pd.read_csv(os.path.join(R, "ucurve_sensitivity.csv"))
Z = pd.read_csv(os.path.join(BASE, "demand/수요_2027_행정동.csv"))
gu = json.load(open(os.path.join(BASE, "boundary/gu.geojson"), encoding="utf-8"))
from pyproj import Transformer
ll = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
zlon, zlat = ll.transform(Z.rep_x.values, Z.rep_y.values)

fig = plt.figure(figsize=(20, 8.5), dpi=110)
ax = fig.add_axes([0.01, 0.04, 0.5, 0.9])
for f in gu["features"]:
    g = f["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    for p in polys:
        r = np.array(p[0]); ax.plot(r[:, 0], r[:, 1], color="#888", lw=.6)
ax.scatter(zlon, zlat, s=Z.d_기준_평상일 / 400, color="#cfd8e3", label="행정동 수요(기준 평상일)", zorder=1)
ex = S[S.existing]; rp = S[(~S.existing) & (S.RP_open > 0)]
mk = {"공영주차장": "s", "주유소_영업": "o", "주유소_폐업": "X", "주유소_휴업": "D"}
ax.scatter(ex.lon, ex.lat, marker="^", s=90, color="#555", edgecolor="k", label="기존 센터 (%d)" % len(ex), zorder=4)
for t, m in mk.items():
    d = rp[rp.type == t]
    if len(d) == 0: continue
    ax.scatter(d[d.RP_cold == 0].lon, d[d.RP_cold == 0].lat, marker=m, s=40, color="#1f77b4", edgecolor="k", lw=.4, zorder=5,
               label="신규 %s (%d)" % (t, len(d)))
    ax.scatter(d[d.RP_cold > 0].lon, d[d.RP_cold > 0].lat, marker=m, s=60, color="#d62728", edgecolor="k", lw=.4, zorder=6)
ax.scatter([], [], color="#d62728", s=60, label="빨강 = 냉장 설비 (%d)" % int(rp.RP_cold.sum()))
ax.set_title("2단계 확률모형 기준안: 신규 %d곳 (T=20분, 11시)" % len(rp), fontsize=14)
ax.legend(loc="lower left", fontsize=9); ax.set_aspect(1.25); ax.set_xticks([]); ax.set_yticks([])

ax2 = fig.add_axes([0.56, 0.55, 0.42, 0.38])
u = U[U.분석 == "U곡선"]
ax2.plot(u.신규센터수, u.연간비용_억, "o-", color="#1f77b4")
b = u.loc[u.연간비용_억.idxmin()]
ax2.annotate("최소 %d곳 · %.0f억원/년" % (b.신규센터수, b.연간비용_억), (b.신규센터수, b.연간비용_억),
             xytext=(b.신규센터수 - 60, b.연간비용_억 + 180), arrowprops=dict(arrowstyle="->"))
sq = json.load(open(os.path.join(R, "run_base.json")))["objs_억"]["statusquo"]
ax2.axhline(sq, color="#999", ls="--"); ax2.text(150, sq - 40, "현상유지(기존 27곳) %s억원/年".replace("年", "년") % format(round(sq), ","), color="#666")
ax2.set_xlabel("신규 센터 수"); ax2.set_ylabel("연간 총비용 (억원)"); ax2.set_title("센터 수별 최소 연간비용 (평균수요)"); ax2.grid(alpha=.3)

ax3 = fig.add_axes([0.56, 0.07, 0.42, 0.36])
s = U[U.분석 == "민감도"]
for fx, c in zip([0, 1, 2], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
    d = s[s.고정운영비_억 == fx]
    ax3.plot(d.외곽편도_분, d.신규센터수, "o-", color=c, label="센터당 고정운영비 %d억/년" % fx)
    for _, r in d.iterrows(): ax3.text(r.외곽편도_분 + .5, r.신규센터수 + 4, "%d" % r.신규센터수, fontsize=8, color=c)
ax3.set_xlabel("외곽 터미널 편도 시간 가정(분) → 벌점"); ax3.set_ylabel("최적 신규 센터 수")
ax3.set_title("민감도: 가정에 따라 최적 개수가 크게 달라짐"); ax3.legend(fontsize=9); ax3.grid(alpha=.3)
plt.savefig(os.path.join(R, "result_base.png"))
print("ok")
