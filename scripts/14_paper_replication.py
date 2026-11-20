# -*- coding: utf-8 -*-
"""14. 대상 논문(강명구 외 2024) 방법 재현 — K-Means 권역 × Method 1~3, 2021 속도 6개 시간대

논문 절차를 우리 데이터로 그대로 따라 한다.
  1) 수요점: 행정동 수요(2027 기준 평상일) 비중대로 행정동 경계 안에 무작위 점 20,000개 (설계용)
  2) 권역: 한강 이북·이남을 따로 K-Means (권역 수 10·20·30·40·50, 교차평가용 92)
  3) 권역별 MFC 입지
       Method 1  권역(볼록 껍질)의 지리적 중심
       Method 2  권역 수요점의 평균 좌표 (수요 무게중심)
       Method 3  권역 안 영업 주유소 중 설계 수요점까지 이동시간 합(6개 시간대 평균)이 가장 작은 곳
  4) 평가: 새 무작위 수요점 5,000개 × 시드 100개 × 시간대 6개(7·8·11·12·17·18시, View-T 2021)
           권역별 '이동시간 총합'과 점당 평균 이동시간
  5) 검정: 프리드먼 검정 + Conover 사후검정 (블록 = 시드 × 시간대 × 권역)

논문과 다른 점 (데이터 제약)
  - 논문 수요는 생활물류 블록(68,216개) 단위, 여기서는 행정동(424개) 단위 수요를 경계 안에 고르게 뿌림
  - 후보 주유소: 논문 396곳, 여기서는 좌표·도로 연결(200m)·공시지가가 확인된 영업 주유소 385곳
    (교차평가에서 같은 후보로 비용을 계산하기 위해)
  - 권역 소속은 가장 가까운 K-Means 중심으로 판정(논문은 볼록 껍질)
출력  results/paper_compare/  replication_table.csv, replication_tests.csv, paper_m3_selected.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import PROC, res  # noqa: E402
import os, time
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy import stats
from sklearn.cluster import KMeans
import shapely
from shapely import wkt as _wkt

OUT = res("paper_compare")
LEVELS = [10, 20, 30, 40, 50]
EXTRA = [92]                       # 교차평가용: C안 신규 센터 수와 같은 권역 수
SLOTS = ["tt21_h07", "tt21_h08", "tt21_h11", "tt21_h12", "tt21_h17", "tt21_h18"]
N_DESIGN, N_SIM, N_SEED = 20000, 5000, 100
SOUTH = {"강서구", "양천구", "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"}
PAPER = {  # 논문 Table 3 (권역별 이동시간 총합의 평균)
    10: (157.14, 154.15, 146.95), 20: (61.67, 58.27, 53.12), 30: (35.87, 32.06, 32.02),
    40: (23.22, 21.67, 21.17), 50: (16.47, 15.38, 15.21)}
t0 = time.time()

# ---------- 데이터 ----------
N = pd.read_csv(os.path.join(PROC, "network", "nodes.csv"))
E = pd.read_csv(os.path.join(PROC, "network", "edges.csv"), usecols=["u", "v"] + SLOTS)
C = pd.read_csv(os.path.join(PROC, "candidates", "candidates.csv"))
Zb = pd.read_csv(os.path.join(PROC, "boundary", "hjd_5179.wkt.csv"), encoding="utf-8-sig")
Zd = pd.read_csv(os.path.join(PROC, "demand", "수요_2027_행정동.csv"))
Z = Zb.merge(Zd[["행정동코드", "d_기준_평상일"]], left_on="zone", right_on="행정동코드")
assert len(Z) == 424
Z["geom"] = Z.wkt.apply(_wkt.loads)
Z["south"] = Z.gu_nm.isin(SOUTH)
idx = pd.Series(np.arange(len(N)), index=N.node_id)
tree = cKDTree(N[["x", "y"]].values)
ST = C[(C.type == "주유소_영업") & ~C.coord_approx.astype(bool) & C.road_ok.astype(bool) & C.in_seoul.astype(bool)
       & (C.area_m2 > 0) & C.rent_year.notna()].reset_index(drop=True)
st_node = idx[ST.node_id.astype(int)].values
st_south = ST.gu.isin(SOUTH).values
print("주유소 후보 %d곳, 행정동 %d" % (len(ST), len(Z)))


def sample_points(n, rng):
    """행정동 수요 비중대로 n개 점을 행정동 경계 안에 고르게 뿌린다."""
    w = Z["d_기준_평상일"].values / Z["d_기준_평상일"].sum()
    cnt = rng.multinomial(n, w)
    xs, ys, sth = [], [], []
    for g, c, s in zip(Z.geom.values, cnt, Z.south.values):
        if c == 0:
            continue
        x0, y0, x1, y1 = g.bounds; got = 0; bx, by = [], []
        while got < c:
            m = max(4 * (c - got), 20)
            px, py = rng.uniform(x0, x1, m), rng.uniform(y0, y1, m)
            ok = shapely.contains_xy(g, px, py)
            bx.append(px[ok]); by.append(py[ok]); got += ok.sum()
        xs.append(np.concatenate(bx)[:c]); ys.append(np.concatenate(by)[:c]); sth.append(np.full(c, s))
    P = pd.DataFrame({"x": np.concatenate(xs), "y": np.concatenate(ys), "south": np.concatenate(sth)})
    P["node"] = tree.query(P[["x", "y"]].values)[1]
    return P


# ---------- 1. 설계 수요점과 K-Means 권역 ----------
rng = np.random.default_rng(0)
D0 = sample_points(N_DESIGN, rng)
REG = {}   # level -> dict(km_n, km_s, k_n, k_s)
for L in LEVELS + EXTRA:
    ns = D0.south.sum(); kn = int(round(L * (len(D0) - ns) / len(D0))); ks = L - kn
    kmn = KMeans(kn, n_init=10, random_state=0).fit(D0.loc[~D0.south, ["x", "y"]].values)
    kms = KMeans(ks, n_init=10, random_state=0).fit(D0.loc[D0.south, ["x", "y"]].values)
    REG[L] = dict(kmn=kmn, kms=kms, kn=kn, ks=ks)


def label(L, x, y, south):
    r = REG[L]; lab = np.empty(len(x), int)
    s = np.asarray(south, bool)
    if (~s).any(): lab[~s] = r["kmn"].predict(np.c_[x[~s], y[~s]])
    if s.any(): lab[s] = r["kms"].predict(np.c_[x[s], y[s]]) + r["kn"]
    return lab


def graph(col):
    return sp.csr_matrix((E[col].values, (idx[E.u].values, idx[E.v].values)), shape=(len(N), len(N)))


G = {c: graph(c) for c in SLOTS}

# ---------- 2. 권역별 Method 1~3 입지 ----------
D0lab = {L: label(L, D0.x.values, D0.y.values, D0.south.values) for L in LEVELS + EXTRA}
st_lab = {L: label(L, ST.x.values, ST.y.values, st_south) for L in LEVELS + EXTRA}
# Method 3 목적: 주유소 s → 자기 권역 설계점까지 이동시간 합 (6개 시간대 평균)
obj = {L: np.zeros(len(ST)) for L in LEVELS + EXTRA}
for col in SLOTS:
    d = dijkstra(G[col], directed=True, indices=st_node)[:, D0.node.values]       # (주유소, 설계점)
    for L in LEVELS + EXTRA:
        own = st_lab[L][:, None] == D0lab[L][None, :]
        obj[L] += (d * own).sum(1) / len(SLOTS)
SITE = {}
sel_rows = []
for L in LEVELS + EXTRA:
    m1, m2, m3 = [], [], []
    for c in range(L):
        pts = D0.loc[D0lab[L] == c, ["x", "y"]].values
        hull = shapely.convex_hull(shapely.multipoints(pts))
        cen = hull.centroid if hull.area > 0 else shapely.Point(pts.mean(0))
        m1.append(tree.query([cen.x, cen.y])[1])
        m2.append(tree.query(pts.mean(0))[1])
        cand = np.where(st_lab[L] == c)[0]
        fb = len(cand) == 0
        if fb:   # 권역 안에 주유소가 없으면 권역 무게중심에서 가장 가까운 주유소
            cand = np.array([cKDTree(ST[["x", "y"]].values).query(pts.mean(0))[1]])
        s = cand[np.argmin(obj[L][cand])] if not fb else cand[0]
        m3.append(st_node[s])
        sel_rows.append({"권역수": L, "권역": c, "cand_id": ST.cand_id[s], "name": ST.name[s], "gu": ST.gu[s],
                         "권역안_주유소수": int((st_lab[L] == c).sum()), "대체선정": fb})
    SITE[L] = {"M1": np.array(m1), "M2": np.array(m2), "M3": np.array(m3)}
SEL = pd.DataFrame(sel_rows)
SEL.to_csv(os.path.join(OUT, "paper_m3_selected.csv"), index=False, encoding="utf-8-sig")
print("입지 선정 완료 %.0fs, 주유소 없는 권역: %s" % (time.time() - t0, SEL.groupby("권역수").대체선정.sum().to_dict()))

# ---------- 3. 무작위 시뮬레이션 (시드 100 × 시간대 6) ----------
SIMS = [sample_points(N_SIM, np.random.default_rng(1000 + s)) for s in range(N_SEED)]
SP = pd.concat([p.assign(seed=s) for s, p in enumerate(SIMS)], ignore_index=True)
SPlab = {L: label(L, SP.x.values, SP.y.values, SP.south.values) for L in LEVELS}
rows = []
for col in SLOTS:
    for L in LEVELS:
        lab = SPlab[L]
        for mth in ["M1", "M2", "M3"]:
            src = SITE[L][mth]
            d = dijkstra(G[col], directed=True, indices=src)                       # (L, 노드)
            t = d[lab, SP.node.values]
            key = SP.seed.values * L + lab
            tot = np.bincount(key, weights=t, minlength=N_SEED * L)
            cnt = np.bincount(key, minlength=N_SEED * L)
            rows.append(pd.DataFrame({"slot": col, "level": L, "method": mth,
                                      "seed": np.repeat(np.arange(N_SEED), L), "cluster": np.tile(np.arange(L), N_SEED),
                                      "total_min": tot, "n": cnt}))
R = pd.concat(rows, ignore_index=True)
R.to_csv(os.path.join(OUT, "replication_raw.csv.gz"), index=False)
print("시뮬레이션 완료 %.0fs, %d행" % (time.time() - t0, len(R)))

# ---------- 4. 기초통계 (논문 Table 3 형식) ----------
T = R.groupby(["level", "method"]).agg(mean=("total_min", "mean"), sd=("total_min", "std"),
                                         min=("total_min", "min"), max=("total_min", "max"),
                                         tot=("total_min", "sum"), n=("n", "sum")).reset_index()
T["점당평균_분"] = T.tot / T.n
T["M1대비_감소율(%)"] = T.groupby("level")["mean"].transform(lambda s: (s.iloc[0] - s) / s.iloc[0] * 100)
T["논문_mean"] = [PAPER[l][int(m[1]) - 1] for l, m in zip(T.level, T.method)]
T["논문_M1대비_감소율(%)"] = T.groupby("level")["논문_mean"].transform(lambda s: (s.iloc[0] - s) / s.iloc[0] * 100)
T = T.drop(columns=["tot", "n"]).round(2)
T.to_csv(os.path.join(OUT, "replication_table.csv"), index=False, encoding="utf-8-sig")


# ---------- 5. 프리드먼 + Conover 사후검정 ----------
def conover_friedman(X):
    """X: (블록 n, 처리 k). Conover(1999) 다중비교. 반환 {(i,j): p}"""
    n, k = X.shape
    r = stats.rankdata(X, axis=1)
    Rj = r.sum(0); A1 = (r ** 2).sum()
    den = np.sqrt(2 * (n * A1 - (Rj ** 2).sum()) / ((n - 1) * (k - 1)))
    df = (n - 1) * (k - 1)
    return {(i, j): 2 * stats.t.sf(abs(Rj[i] - Rj[j]) / den, df) for i in range(k) for j in range(i + 1, k)}


tests = []
for L in LEVELS:
    W = R[R.level == L].pivot_table(index=["slot", "seed", "cluster"], columns="method", values="total_min")
    W = W[(W > 0).all(axis=1)]
    fr = stats.friedmanchisquare(W.M1, W.M2, W.M3)
    pc = conover_friedman(W[["M1", "M2", "M3"]].values)
    tests.append({"level": L, "blocks": len(W), "friedman_stat": round(fr.statistic, 2), "friedman_p": fr.pvalue,
                  "p_M1_M2": pc[(0, 1)], "p_M1_M3": pc[(0, 2)], "p_M2_M3": pc[(1, 2)]})
TS = pd.DataFrame(tests)
TS.to_csv(os.path.join(OUT, "replication_tests.csv"), index=False, encoding="utf-8-sig")
print(T.to_string(index=False)); print(TS.to_string(index=False)); print("총 %.0fs" % (time.time() - t0))

# ---------- 그림: Method 1 대비 Method 3 이동시간 감소율, 논문 vs 재현 ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f or "Nanum" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = [f.name for f in font_manager.fontManager.ttflist
                               if "Sans CJK KR" in f.name or "Nanum" in f.name][:1] or ["sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, S1, MUTE = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#b9b8b3"
M3 = T[T.method == "M3"].set_index("level")
fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=130)
xx = np.arange(len(LEVELS)); bw = 0.36
for off, colr, vals, lab in [(-bw / 2, MUTE, M3["논문_M1대비_감소율(%)"], "논문 (강명구 외 2024)"),
                             (bw / 2, S1, M3["M1대비_감소율(%)"], "재현 (우리 데이터)")]:
    ax.bar(xx + off, vals.values, bw, color=colr, edgecolor="white", lw=2, label=lab)
    for x_, v_ in zip(xx + off, vals.values):
        ax.text(x_, v_ + 0.2, "%.1f%%" % v_, ha="center", fontsize=10, color=INK)
ax.set_xticks(xx); ax.set_xticklabels(["%d개 권역" % l for l in LEVELS], color=INK2)
ax.set_ylabel("Method 1 대비 이동시간 감소율 (%)", color=INK2)
ax.set_title("Method 3(주유소 + 실제 이동시간)의 이동시간 감소 효과: 논문과 재현 비교", loc="left", fontsize=12, color=INK)
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
for s_ in ["top", "right", "left"]: ax.spines[s_].set_visible(False)
ax.tick_params(colors=INK2, length=0); ax.legend(frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "replication.png"))
