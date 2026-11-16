# -*- coding: utf-8 -*-
"""09. 도심 물류센터 입지 최적화 — 결정론(EV) + 2단계 확률계획(RP) + VSS

실행
    python scripts/09_optimize.py                     # HiGHS (pip install highspy)
    python scripts/09_optimize.py --solver gurobi     # Gurobi (학생 라이선스)
    python scripts/09_optimize.py --T 15 --tt tt25_h18 --far 2   # 민감도

모형 (원/년)
  집합  I = 기존센터(27, 개설 고정) ∪ 신규 후보,  J = 행정동(424),  S = 시나리오(성장 3 × 날 2)
        A_j = {T분 안 기존센터 전부} ∪ {T분 안 신규 후보 중 가까운 K곳}
  1단계 y_i ∈ {0,1} 센터 개설,  z_i ∈ {0,1} 냉장 설비
  2단계 x_ijs, w_ijs ≥ 0 일반·신선 배정 물량(개/일),  u_js, v_js ≥ 0 서울 센터로 못 받아 외곽 터미널이 배송하는 물량
  min  Σ_i R_i y_i + Σ_i (κ-1) R_i z_i
     + Σ_s p_s · DAYS · [ Σ_ij c·θ·t_ij (x_ijs + w_ijs) + P Σ_j u_js + P_f Σ_j v_js ]
  s.t. Σ_{i∈A_j} x_ijs + u_js = d_js                      일반 수요
       Σ_{i∈A_j} w_ijs + v_js = f_js                      신선 수요
       Σ_j (x_ijs + w_ijs) ≤ Cap_i y_i                     처리 용량
       Σ_j w_ijs ≤ Cap_i z_i,   z_i ≤ y_i                  신선은 냉장 센터만
       Σ_{i: t_ij ≤ T} y_i ≥ 1      ∀j                     서비스 수준: 모든 동이 T분 안 센터 커버
       Σ_{i: t_ij ≤ T} z_i ≥ 1      ∀j                     신선 서비스 수준: T분 안 냉장 센터
  R_i = 면적×공시지가×5.5% (기존센터 0),  κ = 1.87,  Cap_i = 면적×far×밀도
  c·θ = 420원/분 × 2/170,  DAYS = 365/1.401 (평일 환산 연 일수)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, RES, proc, res  # noqa: E402
import os, sys, json, time, argparse
import numpy as np, pandas as pd
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--solver", default="highs", choices=["highs", "gurobi"])
ap.add_argument("--T", type=float, default=20)            # 서비스 수준(분)
ap.add_argument("--tt", default="tt25_h11")               # 이동시간 시간대
ap.add_argument("--far", type=float, default=1.0)          # 부지→처리면적 배수
ap.add_argument("--dens", type=float, default=4.0)         # 개/㎡·일
ap.add_argument("--K", type=int, default=30)               # 동별 배정 후보 수
ap.add_argument("--penalty_min", type=float, default=45)   # 외곽 터미널 편도 분
ap.add_argument("--fresh_mult", type=float, default=1.5)   # 신선 외곽배송 벌점 배수
ap.add_argument("--fresh", type=int, default=1)            # 0이면 신선 제외
ap.add_argument("--fixed", type=float, default=0)        # 신규센터 고정 운영비(억원/년)
ap.add_argument("--nnew", type=int, default=-1)            # 신규센터 수 고정(U자 곡선용), -1이면 자유
ap.add_argument("--only", default="all")                   # ev 이면 EV만 풀기
ap.add_argument("--general", type=float, default=1.0)     # 일반 물량 반영 비율(0이면 신선 전용 센터망)
ap.add_argument("--gap", type=float, default=0.005)
ap.add_argument("--tlim", type=float, default=600)
ap.add_argument("--tag", default="base")
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = PROC
M = os.path.join(BASE, "model"); RES = res("exploratory")

# ---------- 파라미터 ----------
C_MIN, Q = 420, 170
CT = C_MIN * 2 / Q                     # 원/개·분 (편도 1분당)
P = CT * a.penalty_min                 # 외곽 배송 벌점 원/개
PF = P * a.fresh_mult
KAPPA = 1.87
DAYS = 365 / 1.401

# ---------- 데이터 ----------
C = pd.read_csv(os.path.join(BASE, "candidates/candidates.csv"))
T_all = pd.read_csv(os.path.join(M, "tt_%s.csv.gz" % a.tt), index_col=0)
Z = pd.read_csv(os.path.join(BASE, "demand/수요_2027_행정동.csv"))
SC = pd.read_csv(os.path.join(BASE, "demand/시나리오_2027.csv"))

ex = (C.type == "기존창고") & (C.parcel_biz == True)
new = (C.type != "기존창고") & ~C.coord_approx.astype(bool) & C.road_ok.astype(bool) & C.in_seoul.astype(bool) \
      & (C.area_m2 > 0) & C.rent_year.notna()
F = pd.concat([C[ex].assign(existing=True), C[new].assign(existing=False)], ignore_index=True)
F["cap"] = F.area_m2 * a.far * a.dens
F.loc[F.existing, "cap"] = F.area_m2 * a.dens                 # 기존센터는 현재 면적 그대로
F["R"] = np.where(F.existing, 0.0, F.rent_year)
F["cold0"] = F.cold.fillna(False).astype(bool)
nI, nJ = len(F), len(Z)
T = T_all.loc[F.cand_id, Z.행정동코드.astype(str)].values      # (nI, nJ) 분

# 배정 가능 호(arc): 동별 T분 이내 중 가까운 K곳
arcs_i, arcs_j = [], []
for j in range(nJ):
    ok = np.where(T[:, j] <= a.T)[0]
    exi = ok[F.existing.values[ok]]                       # 기존센터는 T분 안이면 항상 포함
    nw = ok[~F.existing.values[ok]]
    ok = np.r_[exi, nw[np.argsort(T[nw, j])][:a.K]]
    arcs_i += list(ok); arcs_j += [j] * len(ok)
arcs_i, arcs_j = np.array(arcs_i), np.array(arcs_j); nA = len(arcs_i)
cover = sp.csr_matrix((T <= a.T).T.astype(float))             # (nJ, nI)


def build(scen):
    """scen: [(prob, d(nJ), f(nJ))] → 행렬형 MIP."""
    nS = len(scen)
    # 변수 배치: y(nI) z(nI) | s별 x(nA) w(nA) u(nJ) v(nJ)
    blk = 2 * nA + 2 * nJ
    nV = 2 * nI + nS * blk
    oy, oz = 0, nI
    ox = lambda s: 2 * nI + s * blk
    c = np.zeros(nV); lb = np.zeros(nV); ub = np.full(nV, np.inf); integ = np.zeros(nV, int)
    c[oy:oy + nI] = F.R.values + np.where(F.existing.values, 0, a.fixed * 1e8); c[oz:oz + nI] = (KAPPA - 1) * F.R.values
    ub[:2 * nI] = 1; integ[:2 * nI] = 1
    lb[oy:oy + nI][F.existing.values] = 1                                   # 기존센터 개설 고정
    ex_ = F.existing.values
    lb[oz:oz + nI][ex_] = F.cold0.values[ex_]; ub[oz:oz + nI][ex_] = F.cold0.values[ex_]  # 기존 냉장 여부 고정
    if not a.fresh:
        ub[oz:oz + nI] = 0
    rows, cols, vals, rlo, rhi = [], [], [], [], []
    r = 0

    def add(rr, cc, vv, lo, hi):
        nonlocal r
        rows.extend(np.asarray(rr) + r); cols.extend(cc); vals.extend(vv)
        n = int(np.max(rr)) + 1 if len(rr) else 0
        rlo.extend(lo); rhi.extend(hi); r += n

    tarc = T[arcs_i, arcs_j]
    for s, (p, d, f) in enumerate(scen):
        b = ox(s); xo, wo, uo, vo = b, b + nA, b + 2 * nA, b + 2 * nA + nJ
        c[xo:xo + nA] = p * DAYS * CT * tarc
        c[wo:wo + nA] = p * DAYS * CT * tarc
        c[uo:uo + nJ] = p * DAYS * P
        c[vo:vo + nJ] = p * DAYS * PF
        if not a.fresh:
            ub[wo:wo + nA] = 0
        # 일반 수요
        add(np.r_[arcs_j, np.arange(nJ)], np.r_[xo + np.arange(nA), uo + np.arange(nJ)], np.ones(nA + nJ), d, d)
        # 신선 수요
        add(np.r_[arcs_j, np.arange(nJ)], np.r_[wo + np.arange(nA), vo + np.arange(nJ)], np.ones(nA + nJ), f, f)
        # 용량: Σ x + Σ w − Cap y ≤ 0
        add(np.r_[arcs_i, arcs_i, np.arange(nI)], np.r_[xo + np.arange(nA), wo + np.arange(nA), oy + np.arange(nI)],
            np.r_[np.ones(2 * nA), -F.cap.values], np.full(nI, -np.inf), np.zeros(nI))
        # 냉장 용량: Σ w − Cap z ≤ 0
        add(np.r_[arcs_i, np.arange(nI)], np.r_[wo + np.arange(nA), oz + np.arange(nI)],
            np.r_[np.ones(nA), -F.cap.values], np.full(nI, -np.inf), np.zeros(nI))
    # z ≤ y
    add(np.r_[np.arange(nI), np.arange(nI)], np.r_[oz + np.arange(nI), oy + np.arange(nI)],
        np.r_[np.ones(nI), -np.ones(nI)], np.full(nI, -np.inf), np.zeros(nI))
    # 커버: Σ y ≥ 1, Σ z ≥ 1
    cj, ci = cover.nonzero()
    add(cj, oy + ci, np.ones(len(ci)), np.ones(nJ), np.full(nJ, np.inf))
    if a.fresh:
        add(cj, oz + ci, np.ones(len(ci)), np.ones(nJ), np.full(nJ, np.inf))
    if a.nnew >= 0:  # 신규센터 수 = nnew
        nw = np.where(~F.existing.values)[0]
        add(np.zeros(len(nw), int), oy + nw, np.ones(len(nw)), [a.nnew], [a.nnew])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nV))
    return dict(c=c, lb=lb, ub=ub, integ=integ, A=A, rlo=np.array(rlo, float), rhi=np.array(rhi, float),
                nS=nS, ox=ox, blk=blk)


def solve(m, fix_y=None, fix_z=None, label=""):
    lb, ub = m["lb"].copy(), m["ub"].copy()
    if fix_y is not None:
        lb[:nI] = ub[:nI] = fix_y; lb[nI:2 * nI] = ub[nI:2 * nI] = fix_z
    t0 = time.time()
    if a.solver == "highs":
        import highspy
        h = highspy.Highs(); h.setOptionValue("output_flag", False)
        h.setOptionValue("mip_rel_gap", a.gap); h.setOptionValue("time_limit", a.tlim)
        lp = highspy.HighsLp(); A = m["A"].tocsc()
        lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = m["c"], lb, ub
        lp.row_lower_, lp.row_upper_ = m["rlo"], m["rhi"]
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in m["integ"]]
        h.passModel(lp); h.run()
        sol = np.array(h.getSolution().col_value); obj = h.getInfo().objective_function_value
        gap = h.getInfo().mip_gap if fix_y is None else 0.0
    else:
        import gurobipy as gp
        g = gp.Model(); g.Params.OutputFlag = 0; g.Params.MIPGap = a.gap; g.Params.TimeLimit = a.tlim
        vt = np.where(m["integ"] == 1, gp.GRB.BINARY, gp.GRB.CONTINUOUS)
        X = g.addMVar(len(m["c"]), lb=lb, ub=ub, vtype=vt, obj=m["c"])
        A = m["A"]; fin_lo = np.isfinite(m["rlo"]); fin_hi = np.isfinite(m["rhi"])
        eq = fin_lo & fin_hi & (m["rlo"] == m["rhi"])
        if eq.any(): g.addMConstr(A[eq], X, "=", m["rlo"][eq])
        if (fin_lo & ~eq).any(): g.addMConstr(A[fin_lo & ~eq], X, ">", m["rlo"][fin_lo & ~eq])
        if (fin_hi & ~eq).any(): g.addMConstr(A[fin_hi & ~eq], X, "<", m["rhi"][fin_hi & ~eq])
        g.optimize(); sol = X.X; obj = g.ObjVal; gap = g.MIPGap if fix_y is None else 0.0
    print("  %-10s 목적함수 %.1f억원/년  gap %.2f%%  %.0fs" % (label, obj / 1e8, gap * 100, time.time() - t0))
    return sol, obj, gap


def breakdown(m, sol, scen):
    y, z = np.round(sol[:nI]), np.round(sol[nI:2 * nI])
    out = {"임대료": float(F.R.values @ y), "고정운영비": float(a.fixed * 1e8 * y[~F.existing.values].sum()), "냉장추가임대료": float((KAPPA - 1) * F.R.values @ z)}
    tarc = T[arcs_i, arcs_j]; ship = out_ = outf = 0; served = tot = 0
    for s, (p, d, f) in enumerate(scen):
        b = m["ox"](s); x = sol[b:b + nA]; w = sol[b + nA:b + 2 * nA]
        u = sol[b + 2 * nA:b + 2 * nA + nJ]; v = sol[b + 2 * nA + nJ:b + 2 * nA + 2 * nJ]
        ship += p * DAYS * CT * tarc @ (x + w); out_ += p * DAYS * P * u.sum(); outf += p * DAYS * PF * v.sum()
        served += p * (x.sum() + w.sum()); tot += p * (d.sum() + f.sum())
    out.update({"서울센터 배송비": ship, "외곽배송(일반)": out_, "외곽배송(신선)": outf})
    out["합계"] = sum(out.values()); out["서울센터 처리비율"] = served / tot
    return y, z, out


def scen_list(rows):
    L = []
    for _, r in rows.iterrows():
        k = r.시나리오
        dtot = Z["d_%s" % k].values; f = Z["fresh_%s" % k].values if a.fresh else np.zeros(nJ)
        L.append((r.확률, (dtot - f) * a.general, f))
    return L

S6 = scen_list(SC)
pr = SC.확률.values / SC.확률.sum()
EVd = (sum(p * d for p, (_, d, _) in zip(pr, S6)), sum(p * f for p, (_, _, f) in zip(pr, S6)))
EV1 = [(1.0, EVd[0], EVd[1])]
print("후보 %d곳(기존 %d, 신규 %d), 동 %d, 배정호 %d, T=%g분, %s, far=%g, 밀도=%g, 벌점 %.0f원/개"
      % (nI, F.existing.sum(), (~F.existing).sum(), nJ, nA, a.T, a.tt, a.far, a.dens, P))

# 1) 현상유지: 기존센터만 (커버 제약 없이)
_nn = a.nnew; a.nnew = -1
m0 = build(EV1); m0["rlo"][-nJ * (2 if a.fresh else 1):] = -np.inf     # 커버 제약 해제
a.nnew = _nn
y0 = F.existing.values.astype(float); z0 = (F.existing & F.cold0).values.astype(float)
s0, o0, _ = solve(m0, y0, z0, "현상유지")
_, _, B0 = breakdown(m0, s0, EV1)
# 2) EV: 평균수요 결정론
m1 = build(EV1); s1, o1, g1 = solve(m1, label="EV(결정론)")
y1, z1, B1 = breakdown(m1, s1, EV1)
# 3) RP: 2단계 확률
if a.only == "ev":
    print("EV_ONLY %.3f %d %.3f" % (o1 / 1e8, newsel(y1) if False else int(y1[~F.existing.values].sum()), B1["서울센터 처리비율"]))
    sys.exit()
m2 = build(S6); s2, o2, g2 = solve(m2, label="RP(2단계)")
y2, z2, B2 = breakdown(m2, s2, S6)
# 4) EEV: EV 입지를 6개 시나리오에 적용
s3, o3, _ = solve(m2, y1, z1, "EEV")
_, _, B3 = breakdown(m2, s3, S6)

newsel = lambda y: int(y[~F.existing.values].sum())
R = pd.DataFrame({"현상유지(평균수요)": B0, "EV 결정론": B1, "RP 2단계": B2, "EEV(EV입지×6시나리오)": B3}).T / 1e8
R.loc[:, "서울센터 처리비율"] *= 1e8
R["신규센터 수"] = [0, newsel(y1), newsel(y2), newsel(y1)]
R["냉장센터 수(신규)"] = [0, int(z1[~F.existing.values].sum()), int(z2[~F.existing.values].sum()), int(z1[~F.existing.values].sum())]
R["MIP gap(%)"] = [0, g1 * 100, g2 * 100, 0]
VSS = o3 - o2
print(R.round(2).to_string()); print("VSS = EEV − RP = %.2f억원/년 (%.2f%%)" % (VSS / 1e8, VSS / o3 * 100))
print("현상유지 대비 절감(EV 기준): %.1f억원/년" % ((o0 - o1) / 1e8))

tag = a.tag
R.round(3).to_csv(os.path.join(RES, "summary_%s.csv" % tag), encoding="utf-8-sig")
sel = F[["cand_id", "type", "name", "gu", "dong", "lon", "lat", "area_m2", "cap", "R", "existing"]].copy()
sel["EV_open"], sel["EV_cold"], sel["RP_open"], sel["RP_cold"] = y1, z1, y2, z2
# RP 기준 시나리오별 처리량
for s, k in enumerate(SC.시나리오):
    b = m2["ox"](s); x = s2[b:b + nA] + s2[b + nA:b + 2 * nA]
    sel["load_" + k] = np.bincount(arcs_i, weights=x, minlength=nI).round()
sel["util_기준_평상일"] = (sel["load_기준_평상일"] / sel.cap).round(3)
sel[(sel.EV_open > 0) | (sel.RP_open > 0)].to_csv(os.path.join(RES, "selected_%s.csv" % tag), index=False, encoding="utf-8-sig")
json.dump({"args": vars(a), "VSS_억": VSS / 1e8, "saving_vs_statusquo_억": (o0 - o1) / 1e8,
           "objs_억": {"statusquo": o0 / 1e8, "EV": o1 / 1e8, "RP": o2 / 1e8, "EEV": o3 / 1e8}},
          open(os.path.join(RES, "run_%s.json" % tag), "w"), ensure_ascii=False, indent=1)
