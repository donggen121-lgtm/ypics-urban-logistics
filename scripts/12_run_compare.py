# -*- coding: utf-8 -*-
"""12. 확정 파라미터로 결정론 모형 실행: 후보 집합 A(영업 주유소만) vs C(전체)

확정값 (2026-11-16 팀 결정)
  설계 수요   2027 기준 시나리오 평상일 (서울 하루 542만 개, 신선 35만 개)
  이동시간     평일 11시 (tt25_h11), 서비스 수준 T = 20분
  운행비       380원/분, Q = 170개/회전 → 1개·1분당 4.47원
  외곽 배송    편도 45분 → 201원/개 (신선 1.5배)
  고정 운영비  신규 센터 처리용량 1만 개/일 미만 1억, 이상 2억 원/년
  복층 없음    처리용량 = 부지면적 × 4개/㎡·일
  역할         전체 택배(일반 + 신선)
후보 집합
  A  기존 센터 27 + 영업 주유소       (논문과 같은 후보 유형)
  C  기존 센터 27 + 영업·휴업·폐업 주유소 + 공영주차장 (기준안)
  (현상유지 = 기존 센터 27곳만, 비교 기준)

모형 (원/년)
  min Σ_i (R_i + g_i) y_i + Σ_i 0.87 R_i z_i + D·[ Σ_ij cθ t_ij (x_ij + w_ij) + P Σ_j u_j + 1.5P Σ_j v_j ]
  s.t. Σ_{i∈A_j} x_ij + u_j = d_j,  Σ_{i∈A_j} w_ij + v_j = f_j
       Σ_j (x_ij + w_ij) ≤ Cap_i y_i,  Σ_j w_ij ≤ Cap_i z_i,  z_i ≤ y_i
       Σ_{i: t_ij ≤ T} y_i ≥ 1,  Σ_{i: t_ij ≤ T} z_i ≥ 1   (닿는 후보가 있는 동만)
       기존 센터 y=1, z=현재 냉장 여부
실행  python scripts/12_run_compare.py [--solver gurobi]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, RES, proc, res  # noqa: E402
import os, json, time, argparse
import numpy as np, pandas as pd
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--solver", default="highs", choices=["highs", "gurobi"])
ap.add_argument("--gap", type=float, default=0.001)
ap.add_argument("--tlim", type=float, default=600)
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = PROC
M = os.path.join(BASE, "model"); RES = res("final")

# ---------- 확정 파라미터 ----------
SCEN = "기준_평상일"
TT = "tt25_h11"; T_SVC = 20; K = 30
C_MIN, Q = 380, 170
CT = C_MIN * 2 / Q                 # 4.47 원/개·분
P = CT * 45                        # 201 원/개
PF = P * 1.5
KAPPA = 1.87
DAYS = 365 / 1.401                 # 평상일 수요 × DAYS = 연간 물량
DENS, FAR = 4.0, 1.0
CAP_SMALL = 10000                  # 개/일
G_SMALL, G_LARGE = 1e8, 2e8        # 원/년
PARAMS = dict(설계수요=SCEN, 시간대=TT, T분=T_SVC, 운행비_원분=C_MIN, Q=Q, 원_개분=round(CT, 3),
              외곽벌점_원개=round(P, 1), 신선벌점_원개=round(PF, 1), 고정비="용량<1만 1억 / 이상 2억",
              처리밀도=DENS, 복층배수=FAR, 저온배수=KAPPA, 연환산일수=round(DAYS, 1))

# ---------- 데이터 ----------
C = pd.read_csv(os.path.join(BASE, "candidates/candidates.csv"))
TA = pd.read_csv(os.path.join(M, "tt_%s.csv.gz" % TT), index_col=0)
Z = pd.read_csv(os.path.join(BASE, "demand/수요_2027_행정동.csv"))
nJ = len(Z)
f_all = Z["fresh_%s" % SCEN].values
d_all = Z["d_%s" % SCEN].values - f_all
ex = (C.type == "기존창고") & (C.parcel_biz == True)
ok_new = (C.type != "기존창고") & ~C.coord_approx.astype(bool) & C.road_ok.astype(bool) \
         & C.in_seoul.astype(bool) & (C.area_m2 > 0) & C.rent_year.notna()
SETS = {"A": ["주유소_영업"], "C": ["주유소_영업", "주유소_휴업", "주유소_폐업", "공영주차장"]}


def make_F(types):
    nw = ok_new & C.type.isin(types)
    F = pd.concat([C[ex].assign(existing=True), C[nw].assign(existing=False)], ignore_index=True)
    F["cap"] = F.area_m2 * DENS * np.where(F.existing, 1.0, FAR)
    F["R"] = np.where(F.existing, 0.0, F.rent_year)
    F["g"] = np.where(F.existing, 0.0, np.where(F.cap < CAP_SMALL, G_SMALL, G_LARGE))
    F["cold0"] = F.cold.fillna(False).astype(bool)
    return F


def solve(F, statusquo=False, label=""):
    nI = len(F)
    T = TA.loc[F.cand_id, Z.행정동코드.astype(str)].values
    ai, aj = [], []
    for j in range(nJ):
        ok = np.where(T[:, j] <= T_SVC)[0]
        e = ok[F.existing.values[ok]]; n = ok[~F.existing.values[ok]]
        ok = np.r_[e, n[np.argsort(T[n, j])][:K]]
        ai += list(ok); aj += [j] * len(ok)
    ai, aj = np.array(ai), np.array(aj); nA = len(ai); tarc = T[ai, aj]
    # 변수: y(nI) z(nI) x(nA) w(nA) u(nJ) v(nJ)
    oy, oz, ox, ow, ou, ov = 0, nI, 2 * nI, 2 * nI + nA, 2 * nI + 2 * nA, 2 * nI + 2 * nA + nJ
    nV = ov + nJ
    c = np.zeros(nV); lb = np.zeros(nV); ub = np.full(nV, np.inf); integ = np.zeros(nV, int)
    c[oy:oz] = F.R.values + F.g.values; c[oz:ox] = (KAPPA - 1) * F.R.values
    c[ox:ow] = DAYS * CT * tarc; c[ow:ou] = DAYS * CT * tarc
    c[ou:ov] = DAYS * P; c[ov:] = DAYS * PF
    ub[:ox] = 1; integ[:ox] = 1
    E = F.existing.values
    lb[oy:oz][E] = 1
    lb[oz:ox][E] = F.cold0.values[E]; ub[oz:ox][E] = F.cold0.values[E]
    if statusquo:
        ub[oy:oz][~E] = 0; ub[oz:ox][~E] = 0
    R, Cc, V, lo, hi = [], [], [], [], []
    r = 0
    def add(rr, cc, vv, l, h):
        nonlocal r
        R.extend(np.asarray(rr) + r); Cc.extend(cc); V.extend(vv); lo.extend(l); hi.extend(h); r += len(l)
    J = np.arange(nJ); I = np.arange(nI); A_ = np.arange(nA)
    add(np.r_[aj, J], np.r_[ox + A_, ou + J], np.ones(nA + nJ), d_all, d_all)
    add(np.r_[aj, J], np.r_[ow + A_, ov + J], np.ones(nA + nJ), f_all, f_all)
    add(np.r_[ai, ai, I], np.r_[ox + A_, ow + A_, oy + I], np.r_[np.ones(2 * nA), -F.cap.values],
        np.full(nI, -np.inf), np.zeros(nI))
    add(np.r_[ai, I], np.r_[ow + A_, oz + I], np.r_[np.ones(nA), -F.cap.values], np.full(nI, -np.inf), np.zeros(nI))
    add(np.r_[I, I], np.r_[oz + I, oy + I], np.r_[np.ones(nI), -np.ones(nI)], np.full(nI, -np.inf), np.zeros(nI))
    cov = (T <= T_SVC)
    reach = cov.any(0)
    if not statusquo:
        cj, ci = sp.csr_matrix(cov.T[reach].astype(float)).nonzero()
        nr = int(reach.sum())
        add(cj, oy + ci, np.ones(len(ci)), np.ones(nr), np.full(nr, np.inf))
        add(cj, oz + ci, np.ones(len(ci)), np.ones(nr), np.full(nr, np.inf))
    Am = sp.csr_matrix((V, (R, Cc)), shape=(r, nV))
    lo, hi = np.array(lo, float), np.array(hi, float)
    t0 = time.time()
    if a.solver == "highs":
        import highspy
        h = highspy.Highs(); h.setOptionValue("output_flag", False)
        h.setOptionValue("mip_rel_gap", a.gap); h.setOptionValue("time_limit", a.tlim)
        lp = highspy.HighsLp(); Ac = Am.tocsc()
        lp.num_col_, lp.num_row_ = nV, r
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, lb, ub
        lp.row_lower_, lp.row_upper_ = lo, hi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = Ac.indptr, Ac.indices, Ac.data
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integ]
        h.passModel(lp); h.run()
        s = np.array(h.getSolution().col_value); obj = h.getInfo().objective_function_value; gap = h.getInfo().mip_gap
    else:
        import gurobipy as gp
        g = gp.Model(); g.Params.OutputFlag = 0; g.Params.MIPGap = a.gap; g.Params.TimeLimit = a.tlim
        X = g.addMVar(nV, lb=lb, ub=ub, vtype=np.where(integ == 1, gp.GRB.BINARY, gp.GRB.CONTINUOUS), obj=c)
        eq = np.isfinite(lo) & (lo == hi); ge = np.isfinite(lo) & ~eq; le = np.isfinite(hi) & ~eq
        g.addMConstr(Am[eq], X, "=", lo[eq]); g.addMConstr(Am[ge], X, ">", lo[ge]); g.addMConstr(Am[le], X, "<", hi[le])
        g.optimize(); s = X.X; obj = g.ObjVal; gap = g.MIPGap
    print("  %-8s 목적함수 %.1f억원/년  gap %.3f%%  %.0fs" % (label, obj / 1e8, gap * 100, time.time() - t0))
    y = np.round(s[oy:oz]); z = np.round(s[oz:ox]); x = s[ox:ow]; w = s[ow:ou]; u = s[ou:ov]; v = s[ov:]
    load = np.bincount(ai, weights=x + w, minlength=nI)
    served = x.sum() + w.sum(); tot = d_all.sum() + f_all.sum()
    out = {
        "후보 수(신규)": int((~E).sum()),
        "신규 센터 수": int(y[~E].sum()),
        "  공영주차장": int(y[(~E) & (F.type == "공영주차장").values].sum()),
        "  주유소(영업)": int(y[(~E) & (F.type == "주유소_영업").values].sum()),
        "  주유소(휴업·폐업)": int(y[(~E) & F.type.isin(["주유소_휴업", "주유소_폐업"]).values].sum()),
        "냉장 설비 신규": int(z[~E].sum()),
        "임대료(억/년)": F.R.values @ y / 1e8,
        "고정 운영비(억/년)": F.g.values @ y / 1e8,
        "냉장 추가 임대료(억/년)": (KAPPA - 1) * F.R.values @ z / 1e8,
        "서울 센터 배송비(억/년)": DAYS * CT * tarc @ (x + w) / 1e8,
        "외곽 배송비(억/년)": DAYS * (P * u.sum() + PF * v.sum()) / 1e8,
        "총비용(억/년)": obj / 1e8,
        "서울 센터 처리 비율(%)": served / tot * 100,
        "신선 서울 처리 비율(%)": w.sum() / f_all.sum() * 100,
        "서울 처리분 평균 이동시간(분)": tarc @ (x + w) / served if served > 0 else np.nan,
        "1개당 비용(원)": obj / (tot * DAYS),
        "20분 커버 불가 동": int((~reach).sum()),
        "MIP gap(%)": gap * 100,
    }
    sel = F.assign(open=y, cold_open=z, load=np.round(load), util=np.where(F.cap > 0, load / F.cap, 0))
    return out, sel[sel.open > 0]


print("파라미터:", PARAMS)
res, sels = {}, {}
F0 = make_F([])
res["현상유지(기존 27곳)"], _ = solve(F0, statusquo=True, label="현상유지")
for k, types in SETS.items():
    F = make_F(types)
    res["%s: %s" % (k, "영업 주유소만" if k == "A" else "주유소 전체+공영주차장")], sels[k] = solve(F, label=k)
S = pd.DataFrame(res)
base = S.loc["총비용(억/년)", "현상유지(기존 27곳)"]
S.loc["현상유지 대비 절감(억/년)"] = base - S.loc["총비용(억/년)"]
S = S.round(2)
print(S.to_string())
S.to_csv(os.path.join(RES, "compare_A_C.csv"), encoding="utf-8-sig")
cols = ["cand_id", "type", "name", "gu", "dong", "addr", "lon", "lat", "area_m2", "cap", "R", "g", "existing",
        "cold_open", "load", "util"]
for k, s in sels.items():
    s[cols].sort_values(["existing", "type"], ascending=[False, True]).to_csv(
        os.path.join(RES, "selected_%s.csv" % k), index=False, encoding="utf-8-sig")
json.dump(PARAMS, open(os.path.join(RES, "params_final.json"), "w"), ensure_ascii=False, indent=1)
