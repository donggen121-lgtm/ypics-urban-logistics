# -*- coding: utf-8 -*-
"""17. 논문 Figure 7·8(10·20개 권역 MFC 선정 결과)을 재현 결과로 같은 형식으로 그려 나란히 비교

14번과 같은 설계 수요점(시드 0)·K-Means(random_state 0)로 권역을 다시 만들고,
Method 1(권역 볼록 껍질의 지리적 중심, 붉은 x), Method 2(수요 무게중심, 녹색 o),
Method 3(14번이 고른 주유소, 파란 ▼)를 권역 볼록 껍질 위에 그린다.
논문 그림은 results/paper_compare/_paper_fig78.png (논문 13쪽 Figure 7·8을 잘라 낸 것)가 있으면 위에 붙인다.
출력  results/paper_compare/fig78_paper_vs_replication.png, fig78_shift.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import PROC, res  # noqa: E402
import os
import numpy as np, pandas as pd
import shapely
from shapely import wkt as _wkt
from sklearn.cluster import KMeans
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f or "Nanum" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = [f.name for f in font_manager.fontManager.ttflist
                               if "Sans CJK KR" in f.name or "Nanum" in f.name][:1] or ["sans-serif"]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#ffffff"

N_DESIGN = 20000
SOUTH = {"강서구", "양천구", "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"}
C = pd.read_csv(os.path.join(PROC, "candidates", "candidates.csv"))
Zb = pd.read_csv(os.path.join(PROC, "boundary", "hjd_5179.wkt.csv"), encoding="utf-8-sig")
Zd = pd.read_csv(os.path.join(PROC, "demand", "수요_2027_행정동.csv"))
Z = Zb.merge(Zd[["행정동코드", "d_기준_평상일"]], left_on="zone", right_on="행정동코드")
Z["geom"] = Z.wkt.apply(_wkt.loads); Z["south"] = Z.gu_nm.isin(SOUTH)
SEL = pd.read_csv(res("paper_compare", "paper_m3_selected.csv"))
gu = pd.read_csv(os.path.join(PROC, "boundary", "gu_5179.wkt.csv"), encoding="utf-8-sig")


def sample_points(n, rng):   # 14번과 같은 절차 (같은 시드 → 같은 점)
    w = Z["d_기준_평상일"].values / Z["d_기준_평상일"].sum()
    cnt = rng.multinomial(n, w); xs, ys, sth = [], [], []
    for g, c, s in zip(Z.geom.values, cnt, Z.south.values):
        if c == 0: continue
        x0, y0, x1, y1 = g.bounds; got = 0; bx, by = [], []
        while got < c:
            m = max(4 * (c - got), 20)
            px, py = rng.uniform(x0, x1, m), rng.uniform(y0, y1, m)
            ok = shapely.contains_xy(g, px, py); bx.append(px[ok]); by.append(py[ok]); got += ok.sum()
        xs.append(np.concatenate(bx)[:c]); ys.append(np.concatenate(by)[:c]); sth.append(np.full(c, s))
    return pd.DataFrame({"x": np.concatenate(xs), "y": np.concatenate(ys), "south": np.concatenate(sth)})


D0 = sample_points(N_DESIGN, np.random.default_rng(0))
OUTROWS = []


def regions(L):
    ns = D0.south.sum(); kn = int(round(L * (len(D0) - ns) / len(D0))); ks = L - kn
    lab = np.empty(len(D0), int); s = D0.south.values
    lab[~s] = KMeans(kn, n_init=10, random_state=0).fit_predict(D0.loc[~s, ["x", "y"]].values)
    lab[s] = KMeans(ks, n_init=10, random_state=0).fit_predict(D0.loc[s, ["x", "y"]].values) + kn
    out = []
    for c in range(L):
        pts = D0.loc[lab == c, ["x", "y"]].values
        hull = shapely.convex_hull(shapely.multipoints(pts))
        m1 = hull.centroid; m2 = pts.mean(0)
        r = SEL[(SEL.권역수 == L) & (SEL.권역 == c)].iloc[0]
        st = C.loc[C.cand_id == r.cand_id].iloc[0]
        out.append((hull, (m1.x, m1.y), tuple(m2), (st.x, st.y)))
        OUTROWS.append({"권역수": L, "권역": c, "M1→M2_m": round(float(np.hypot(*(m2 - [m1.x, m1.y])))),
                        "M1→M3_m": round(float(np.hypot(st.x - m1.x, st.y - m1.y))), "M3_주유소": r["name"], "M3_구": r.gu})
    return out, kn


def draw(ax, L):
    reg, kn = regions(L)
    for g in gu.wkt.apply(_wkt.loads):
        for p in getattr(g, "geoms", [g]):
            ax.plot(*p.exterior.xy, color="#e6e5e1", lw=.6, zorder=0)
    for hull, m1, m2, m3 in reg:
        ax.plot(*hull.exterior.xy, color="black", lw=.7, zorder=1)
        ax.scatter(*m1, marker="x", s=40, color="red", lw=1, zorder=3)
        ax.scatter(*m2, marker="o", s=34, color="green", zorder=3)
        ax.scatter(*m3, marker="v", s=55, color="blue", zorder=4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("재현: %d개 권역 (이북 %d · 이남 %d)" % (L, kn, L - kn), fontsize=13, color=INK, loc="left")


pap = res("paper_compare", "_paper_fig78.png")
has = os.path.exists(pap)
fig = plt.figure(figsize=(15, 13.5 if has else 7), dpi=130, facecolor=SURF)
if has:
    a0 = fig.add_axes([0.02, 0.53, 0.96, 0.42]); a0.imshow(plt.imread(pap)); a0.axis("off")
    a0.set_title("논문 Figure 7·8 (강명구 외 2024, 13쪽): 10개 권역(왼쪽) · 20개 권역(오른쪽)", fontsize=13, color=INK, loc="left")
    a1 = fig.add_axes([0.02, 0.06, 0.47, 0.43]); a2 = fig.add_axes([0.51, 0.06, 0.47, 0.43])
else:
    a1 = fig.add_axes([0.02, 0.1, 0.47, 0.8]); a2 = fig.add_axes([0.51, 0.1, 0.47, 0.8])
draw(a1, 10); draw(a2, 20)
leg = [Line2D([], [], marker="x", ls="", color="red", ms=8, label="Method 1: 권역의 지리적 중심"),
       Line2D([], [], marker="o", ls="", color="green", ms=7, label="Method 2: 수요 무게중심"),
       Line2D([], [], marker="v", ls="", color="blue", ms=8, label="Method 3: 이동시간 최소 주유소"),
       Line2D([], [], color="#e6e5e1", lw=2, label="자치구 경계 (재현 그림만)")]
fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False, fontsize=11, labelcolor=INK2)
plt.savefig(res("paper_compare", "fig78_paper_vs_replication.png"), facecolor=SURF)
S = pd.DataFrame(OUTROWS); S.to_csv(res("paper_compare", "fig78_shift.csv"), index=False, encoding="utf-8-sig")
print(S.groupby("권역수")[["M1→M2_m", "M1→M3_m"]].median())
