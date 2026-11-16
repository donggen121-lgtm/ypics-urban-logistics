# -*- coding: utf-8 -*-
"""01. 도로망(상세도로망 Level6) + View-T 평균속도(2025, 2021) → 시간대별 방향 그래프

입력  data/raw/lev6/*.shp, View-T 2025 정리본 csv, 2021 원본 xlsx
출력  out/network/nodes.csv, edges.csv, speed_fill_ratio.csv, network_summary.txt
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, RES, proc, res  # noqa: E402
import os, re, sys, warnings
import numpy as np, pandas as pd, shapefile
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pyproj import Transformer
warnings.filterwarnings("ignore")

UP = RAW
OUT = proc("network")
LOG = []
def log(m): print(m); LOG.append(m)

# ---------------- 1. 도로망 → 방향 엣지 ----------------
r = shapefile.Reader(os.path.join(RAW, "lev6", "seoul_link_lev6_2025"), encoding="cp949")
L = pd.DataFrame([x.as_dict() for x in r.records()])
rn = shapefile.Reader(os.path.join(RAW, "lev6", "seoul_node_lev6_2025"), encoding="cp949")
N = pd.DataFrame([x.as_dict() for x in rn.records()])[["node_id", "x", "y", "node_name", "tra_light"]]
N["x"] = pd.to_numeric(N.x); N["y"] = pd.to_numeric(N.y)

up = pd.DataFrame({"link_id": L.link_id, "dir": "up", "u": L.up_f_node, "v": L.up_t_node,
                   "its_id": L.up_its_id, "lanes_dir": np.where(L.oneway == 1, L.lanes, L.up_lanes)})
dw = L[L.oneway == 0]
dw = pd.DataFrame({"link_id": dw.link_id, "dir": "dw", "u": dw.dw_f_node, "v": dw.dw_t_node,
                   "its_id": dw.dw_its_id, "lanes_dir": dw.dw_lanes})
E = pd.concat([up, dw], ignore_index=True)
E = E.merge(L[["link_id", "length", "max_speed", "road_rank", "road_name", "link_type", "oneway"]],
            on="link_id", how="left").rename(columns={"length": "length_km"})
E["u"] = E.u.astype("int64"); E["v"] = E.v.astype("int64")
E["its_id"] = pd.to_numeric(E.its_id, errors="coerce").fillna(0).astype("int64")
E["lanes_dir"] = pd.to_numeric(E.lanes_dir, errors="coerce").fillna(1).clip(lower=1)
log("링크 %d개 → 방향 엣지 %d개 (일방통행 %d, 양방향 %d×2)" %
    (len(L), len(E), int((L.oneway == 1).sum()), int((L.oneway == 0).sum())))

# ---------------- 2. 속도 읽기 ----------------
SLOT_MAP = {"연평균 일": "avg", "6시~9시": "r06_09", "13시~17시": "r13_17", "2시~5시": "r02_05",
            "17시~20시": "r17_20"}
for h in range(24):
    SLOT_MAP["%d~%d시" % (h, h + 1)] = "h%02d" % h
SLOTS = ["avg"] + ["h%02d" % h for h in range(24)] + ["r06_09", "r13_17", "r02_05", "r17_20"]

def read_viewt(xlsx=None, csv=None):
    if csv:
        d = pd.read_csv(csv, dtype={"L55_ID": str, "ITS_ID": str})
    else:
        raw = pd.read_excel(xlsx, header=None)
        d = raw.iloc[7:].reset_index(drop=True)
        d.columns = ["L55_ID", "ITS_ID", "도로등급", "도로명", "권역", "연장_km", "차선수"] + \
                    [str(c) for c in raw.iloc[6, 7:].tolist()]
        d["L55_ID"] = d.L55_ID.astype(str); d["ITS_ID"] = d.ITS_ID.astype(str)
    d = d.rename(columns=SLOT_MAP)
    for s in SLOTS: d[s] = pd.to_numeric(d[s], errors="coerce")
    # ITS ID 쉼표 분리 → ITS 1개당 행 1개 (여러 Level5.5 구간에 걸치면 평균)
    m = d[["ITS_ID"] + SLOTS].assign(ITS_ID=d.ITS_ID.str.split(",")).explode("ITS_ID")
    m["ITS_ID"] = pd.to_numeric(m.ITS_ID.str.strip(), errors="coerce")
    m = m[m.ITS_ID.notna() & (m.ITS_ID != 0)]
    m["ITS_ID"] = m.ITS_ID.astype("int64")
    return m.groupby("ITS_ID")[SLOTS].mean()

sp25 = read_viewt(csv=os.path.join(UP, "View-T_평균속도_2025_링크.csv"))
sp21 = read_viewt(xlsx=os.path.join(UP, "2021년 차량 네트워크.xlsx"))
log("View-T ITS 링크 수: 2025 %d, 2021 %d" % (len(sp25), len(sp21)))

# ---------------- 3. 조인 + 이상치 + 결측 보정 (논문 Figure 3 방식) ----------------
E["lane_grp"] = pd.cut(E.lanes_dir, [0, 1, 2, 99], labels=["1", "2", "3+"]).astype(str)
ratio_rows = []

def attach(sp, tag, slots):
    global E
    j = E[["its_id"]].join(sp[slots], on="its_id")
    obs = j[slots[0]].notna() & (E.its_id != 0)
    E["src_" + tag] = np.where(obs, "관측", "보정")
    ncap = 0
    for s in slots:
        v = j[s].where(E.its_id != 0)
        # 상한: max(제한속도×1.4, 제한속도+15), 최대 130km/h (저속 구간에 큰 도로 속도가 붙는 경우 완화)
        ub = np.minimum(130, np.maximum(E.max_speed * 1.4, E.max_speed + 15))
        ncap += int((v > ub).sum())
        v = v.clip(lower=3).where(~(v > ub), ub)
        rat = (v / E.max_speed)
        g1 = rat.groupby([E.road_rank, E.lane_grp]).median()
        g2 = rat.groupby(E.road_rank).median()
        g0 = rat.median()
        r1 = pd.Series(list(zip(E.road_rank, E.lane_grp))).map(g1).values
        r2 = E.road_rank.map(g2).values
        rr = np.where(np.isnan(r1), np.where(np.isnan(r2), g0, r2), r1)
        E["sp%s_%s" % (tag, s)] = np.round(np.where(v.isna(), E.max_speed * rr, v), 2)
        if s in ("avg", "h08", "h18", "r02_05"):
            for (rk, lg), val in g1.items():
                ratio_rows.append({"연도": tag, "시간대": s, "도로등급": rk, "차로": lg,
                                   "실주행/제한속도 비율(중앙값)": round(val, 3)})
    log("[%s] 속도 관측 엣지 %d / %d (%.1f%%), 상한 초과로 자른 칸 %d" %
        (tag, int(obs.sum()), len(E), 100 * obs.mean(), ncap))

attach(sp25, "25", SLOTS)
PAPER = ["avg", "h07", "h08", "h11", "h12", "h17", "h18"]
attach(sp21, "21", PAPER)

# ---------------- 4. 최대 강연결 요소만 남기기 (논문 3.2절) ----------------
nodes = pd.Index(pd.unique(np.concatenate([E.u.values, E.v.values])))
ui = nodes.get_indexer(E.u); vi = nodes.get_indexer(E.v)
G = csr_matrix((np.ones(len(E)), (ui, vi)), shape=(len(nodes), len(nodes)))
n, lab = connected_components(G, directed=True, connection="strong")
big = np.bincount(lab).argmax()
keep_nodes = set(nodes[lab == big])
before = len(E)
E = E[E.u.isin(keep_nodes) & E.v.isin(keep_nodes)].reset_index(drop=True)
log("강연결 요소 %d개 중 최대 요소: 노드 %d / %d (%.1f%%), 엣지 %d / %d" %
    (n, len(keep_nodes), len(nodes), 100 * len(keep_nodes) / len(nodes), len(E), before))

# ---------------- 5. 이동시간(분) + 저장 ----------------
for s in ["avg", "h07", "h08", "h11", "h12", "h17", "h18", "r02_05"]:
    E["tt25_" + s] = np.round(E.length_km / E["sp25_" + s] * 60, 4)
for s in PAPER:
    E["tt21_" + s] = np.round(E.length_km / E["sp21_" + s] * 60, 4)
E.insert(0, "edge_id", np.arange(len(E)))
E = E.drop(columns=["lane_grp"])

N = N[N.node_id.isin(keep_nodes)].copy()
N["coord_src"] = "노드파일"
# 노드 파일에 없는 노드는 링크 형상의 시작·끝점에서 좌표를 가져온다
miss = keep_nodes - set(N.node_id)
if miss:
    shp = {rec["link_id"]: s.points for rec, s in zip(r.records(), r.shapes())}
    add = {}
    for _, e in E[E.u.isin(miss) | E.v.isin(miss)].iterrows():
        pts = shp.get(e.link_id)
        if not pts:
            continue
        a, b = (pts[0], pts[-1]) if e.dir == "up" else (pts[-1], pts[0])
        if e.u in miss: add.setdefault(e.u, a)
        if e.v in miss: add.setdefault(e.v, b)
    N = pd.concat([N, pd.DataFrame([{"node_id": k, "x": v[0], "y": v[1], "coord_src": "링크형상"}
                                    for k, v in add.items()])], ignore_index=True)
    log("노드 파일에 없는 노드 %d개 → 링크 형상으로 좌표 보완 %d개" % (len(miss), len(add)))
tr = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
N["lon"], N["lat"] = tr.transform(N.x.values, N.y.values)
N.to_csv(os.path.join(OUT, "nodes.csv"), index=False, encoding="utf-8-sig")
E.to_csv(os.path.join(OUT, "edges.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(ratio_rows).to_csv(os.path.join(OUT, "speed_fill_ratio.csv"), index=False,
                                encoding="utf-8-sig")

log("\n[속도 요약 2025, km/h 중앙값] 관측 엣지 / 전체")
for s in ["avg", "h08", "h18", "r02_05"]:
    o = E[E.src_25 == "관측"]["sp25_" + s].median(); a = E["sp25_" + s].median()
    log("  %-7s 관측 %.1f / 전체 %.1f" % (s, o, a))
log("[도로등급별 엣지 수]\n" + E.road_rank.value_counts().sort_index().to_string())
open(os.path.join(OUT, "network_summary.txt"), "w", encoding="utf-8").write("\n".join(LOG))
print("saved", OUT)
