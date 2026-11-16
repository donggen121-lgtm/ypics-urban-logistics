# -*- coding: utf-8 -*-
"""13. 후보 집합 A vs C 결과 그림 (발표용): 입지 지도 2장 + 연간 비용 구성"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, RES, proc, res  # noqa: E402
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
plt.rcParams["font.family"] = [f.name for f in font_manager.fontManager.ttflist if "Sans CJK KR" in f.name or "Nanum" in f.name][:1] or ["sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = PROC
R = res("final")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#b9b8b3", "#e6e5e1", "#fcfcfb"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # 기준 팔레트 1~3번 (고정 순서)

S = pd.read_csv(os.path.join(R, "compare_A_C.csv"), index_col=0)
gu = json.load(open(os.path.join(BASE, "boundary/gu.geojson"), encoding="utf-8"))
Z = pd.read_csv(os.path.join(BASE, "demand/수요_2027_행정동.csv"))
zlon, zlat = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True).transform(Z.rep_x.values, Z.rep_y.values)

fig = plt.figure(figsize=(18, 6.9), dpi=130, facecolor=SURF)


def draw_map(ax, k, col):
    s = pd.read_csv(os.path.join(R, "selected_%s.csv" % k))
    for f in gu["features"]:
        g = f["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polys:
            r = np.array(p[0]); ax.plot(r[:, 0], r[:, 1], color=MUTED, lw=.6, zorder=1)
    ax.scatter(zlon, zlat, s=Z["d_기준_평상일"] / 500, color=GRID, zorder=2)
    e = s[s.existing]; n = s[~s.existing]
    gas = n[n.type.str.startswith("주유소")]; park = n[n.type == "공영주차장"]
    cold = n[n.cold_open > 0]
    ax.scatter(e.lon, e.lat, marker="^", s=70, color=INK2, edgecolor=SURF, lw=1.5, zorder=4)
    ax.scatter(gas.lon, gas.lat, marker="o", s=48, color=S1, edgecolor=SURF, lw=1.5, zorder=5)
    ax.scatter(park.lon, park.lat, marker="s", s=44, color=S2, edgecolor=SURF, lw=1.5, zorder=5)
    ax.scatter(cold.lon, cold.lat, marker="o", s=170, facecolor="none", edgecolor=S3, lw=2, zorder=6)
    nn = int(S.loc["신규 센터 수", col]); tot = S.loc["총비용(억/년)", col]; rt = S.loc["서울 센터 처리 비율(%)", col]
    ax.set_title("%s\n신규 %d곳 · 연 %s억 원 · 서울 처리 %.0f%%" % (
        "A. 후보 = 영업 주유소만 (논문 조건)" if k == "A" else "C. 후보 = 주유소 전체 + 공영주차장",
        nn, format(round(tot), ","), rt), fontsize=13, color=INK, loc="left")
    ax.set_aspect(1.25); ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    ax.set_facecolor(SURF)


cols = list(S.columns)
draw_map(fig.add_axes([0.015, 0.09, 0.32, 0.8]), "A", cols[1])
draw_map(fig.add_axes([0.345, 0.09, 0.32, 0.8]), "C", cols[2])
leg = [Line2D([], [], marker="^", ls="", color=INK2, ms=9, label="기존 택배센터 (27)"),
       Line2D([], [], marker="o", ls="", color=S1, ms=8, label="신규: 주유소"),
       Line2D([], [], marker="s", ls="", color=S2, ms=8, label="신규: 공영주차장"),
       Line2D([], [], marker="o", ls="", mfc="none", mec=S3, mew=2, ms=12, label="냉장 설비 설치"),
       Line2D([], [], marker="o", ls="", color=GRID, ms=8, label="행정동 수요 (원 크기)")]
fig.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=5, frameon=False, fontsize=10.5,
           labelcolor=INK2)

# 비용 구성
ax = fig.add_axes([0.72, 0.24, 0.26, 0.6]); ax.set_facecolor(SURF)
names = ["현상유지\n(기존 27곳)", "A. 영업\n주유소만", "C. 주유소 전체\n+ 공영주차장"]
fac = S.loc[["임대료(억/년)", "고정 운영비(억/년)", "냉장 추가 임대료(억/년)"]].sum().values
ship = S.loc["서울 센터 배송비(억/년)"].values; out = S.loc["외곽 배송비(억/년)"].values
y = np.arange(3)[::-1]
parts = [(fac, S1, "센터 비용 (임대·고정·냉장)"), (ship, S3, "서울 센터 배송비"), (out, S2, "외곽 터미널 배송비")]
left = np.zeros(3)
for v, c, lab in parts:
    ax.barh(y, v, left=left, color=c, edgecolor=SURF, lw=2, height=.55, label=lab)
    left += v
for yi, t, sv in zip(y, left, S.loc["현상유지 대비 절감(억/년)"].values):
    txt = "%s억" % format(round(t), ",") + ("" if sv == 0 else "\n(−%s억)" % format(round(sv), ","))
    ax.text(t + 40, yi, txt, va="center", fontsize=11, color=INK)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=11, color=INK)
ax.set_xlim(0, 2900); ax.set_xlabel("연간 총비용 (억 원)", color=INK2)
ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
for sp_ in ["top", "right", "left"]: ax.spines[sp_].set_visible(False)
ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=INK2, length=0)
ax.set_title("연간 비용 구성 (2027 기준 평상일)", fontsize=13, color=INK, loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=1, frameon=False, fontsize=10, labelcolor=INK2)
plt.savefig(os.path.join(R, "compare_A_C.png"), facecolor=SURF)
print("ok")
