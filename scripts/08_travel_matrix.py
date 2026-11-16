# -*- coding: utf-8 -*-
"""08. 이동시간 행렬: 후보지(914) → 행정동(424) 최단 이동시간(분), 다익스트라

- 센터 → 동 방향(배송 출발 방향) 방향그래프
- 동 대표점(rep_x, rep_y)과 후보지를 각각 가장 가까운 도로 노드에 연결 (후보지는 03에서 계산한 node_id)
- 시간대: tt25_h11(기준), tt25_avg, tt25_h18
출력  model/tt_{col}.csv.gz  (행: cand_id, 열: 행정동코드), model/zone_nodes.csv, model/coverage_T.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, RES, proc, res  # noqa: E402
import os
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

O = PROC; M = proc("model")
N = pd.read_csv(os.path.join(O, "network/nodes.csv"))
E = pd.read_csv(os.path.join(O, "network/edges.csv"),
                usecols=["u", "v", "tt25_h11", "tt25_avg", "tt25_h18"])
C = pd.read_csv(os.path.join(O, "candidates/candidates.csv"))
Z = pd.read_csv(os.path.join(O, "demand/수요_2027_행정동.csv"))

idx = pd.Series(np.arange(len(N)), index=N.node_id)
tree = cKDTree(N[["x", "y"]].values)
d, i = tree.query(Z[["rep_x", "rep_y"]].values)
Z["node_id"] = N.node_id.values[i]; Z["snap_m"] = d.round(1)
Z[["행정동코드", "gu_nm", "hjd_nm", "node_id", "snap_m"]].to_csv(os.path.join(M, "zone_nodes.csv"), index=False, encoding="utf-8-sig")

src = idx[C.node_id.astype(int)].values
dst = idx[Z.node_id].values
cov = []
for col in ["tt25_h11", "tt25_avg", "tt25_h18"]:
    G = csr_matrix((E[col].values, (idx[E.u].values, idx[E.v].values)), shape=(len(N), len(N)))
    us, inv = np.unique(src, return_inverse=True)
    T = dijkstra(G, directed=True, indices=us)[:, dst][inv]          # (914, 424) 분
    T = pd.DataFrame(np.round(T, 2), index=C.cand_id, columns=Z.행정동코드)
    T.to_csv(os.path.join(M, "tt_%s.csv.gz" % col), encoding="utf-8")
    use = (~C.coord_approx).values
    for t in [15, 20, 25, 30]:
        reach = (T.values[use] <= t)
        cov.append({"시간대": col, "T(분)": t,
                    "후보 1곳 이상으로 닿는 동": int(reach.any(0).sum()),
                    "기존 27곳만으로 닿는 동": int((T.values[(C.type == "기존창고").values & (C.parcel_biz == True).values] <= t).any(0).sum()),
                    "후보당 평균 커버 동 수": round(reach.sum(1).mean(), 1),
                    "inf(도달불가)": int(np.isinf(T.values).sum())})
    print(col, "중앙값 %.1f분" % np.median(T.values[np.isfinite(T.values)]))
cov = pd.DataFrame(cov); cov.to_csv(os.path.join(M, "coverage_T.csv"), index=False, encoding="utf-8-sig")
print(cov.to_string(index=False)); print(Z.snap_m.describe().round())
