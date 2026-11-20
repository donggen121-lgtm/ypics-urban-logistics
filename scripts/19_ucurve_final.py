# -*- coding: utf-8 -*-
"""19. 최종 모형 기준 U자 곡선: 신규 센터 수 k를 고정했을 때 최소 연간비용

- 후보: 18번과 같은 C안 집합 (주유소 = 팀원 공간 필터 450곳, 공영주차장 전체)
- 12번 모형에 제약 Σ_{신규 i} y_i = k 를 추가해 k별로 푼다 (20분 커버·냉장 커버 유지)
- 각 해는 18번과 같은 방식(개설 위치 고정 후 재평가)으로 비용을 다시 계산해 슬라이드 숫자와 맞춘다
- 손익분기: 현상유지(기존 27곳) 비용보다 비싸지는 k

출력  results/slides/ucurve_final.csv, ucurve_final.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, res  # noqa: E402
import os, types, argparse, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ks", default="15,30,50,70,110,130,160,200,250,300,400,500")
ap.add_argument("--gap", default="0.002")
ap.add_argument("--tlim", default="300")
args = ap.parse_known_args()[0]

HERE = os.path.dirname(os.path.abspath(__file__))
_sys.argv = _sys.argv[:1] + ["--gap", args.gap, "--tlim", args.tlim]
src = open(os.path.join(HERE, "12_run_compare.py"), encoding="utf-8").read()
# 신규 센터 수 고정 제약을 끼워 넣는다
src = src.replace(
    "    Am = sp.csr_matrix((V, (R, Cc)), shape=(r, nV))",
    "    if NNEW is not None and fixed_new is None and not statusquo:\n"
    "        idx = np.where(~E)[0]\n"
    "        add(np.zeros(len(idx), int), oy + idx, np.ones(len(idx)), [NNEW], [NNEW])\n"
    "    Am = sp.csr_matrix((V, (R, Cc)), shape=(r, nV))")
src = src.replace('if __name__ == "__main__":', "if False:")
m = types.ModuleType("m12"); m.__file__ = os.path.join(HERE, "12_run_compare.py"); m.NNEW = None
exec(compile(src, "m12", "exec"), m.__dict__)

C = m.C
GAS = pd.read_csv(os.path.join(RAW, "주유소_최종후보.csv"), encoding="utf-8-sig", dtype={"관리번호": str})
ok_all = m.ok_new.copy()
ok_450 = ok_all & (~C.type.str.startswith("주유소") | C.src_id.astype(str).isin(set(GAS.관리번호)))

KEEP = ["신규 센터 수", "  공영주차장", "  주유소(영업)", "  주유소(휴업·폐업)", "냉장 설비 신규",
        "임대료(억/년)", "고정 운영비(억/년)", "냉장 추가 임대료(억/년)", "서울 센터 배송비(억/년)",
        "외곽 배송비(억/년)", "총비용(억/년)", "서울 센터 처리 비율(%)", "1개당 비용(원)", "MIP gap(%)"]


def evaluate(ids, label):
    """18번과 같은 재평가: 개설 위치 고정, 배정·냉장만 다시 푼다."""
    m.ok_new = ok_all; m.NNEW = None
    F = m.make_F(m.SETS["C"] + ["주유소_영업"])
    Fk = F[F.existing | F.cand_id.isin(ids)].reset_index(drop=True)
    out, _ = m.solve(Fk, label=label, fixed_new=ids)
    return out


rows = []
m.ok_new = ok_all
sq, _ = m.solve(m.make_F([]), statusquo=True, label="현상유지")
BASE = sq["총비용(억/년)"]
rows.append(dict(k=0, 방식="현상유지", **{c: sq[c] for c in KEEP}))

ks = [None] + [int(x) for x in args.ks.split(",")]
for k in ks:
    t0 = time.time()
    m.ok_new = ok_450; m.NNEW = k
    out, sel = m.solve(m.make_F(m.SETS["C"]), label="k=%s" % k)
    ids = sel[~sel.existing.astype(bool)].cand_id.tolist()
    ev = evaluate(ids, "  재평가")
    rows.append(dict(k=len(ids), 방식="자유(최적)" if k is None else "k 고정", 최적화목적_억=out["총비용(억/년)"],
                     **{c: ev[c] for c in KEEP}))
    print("k=%s → 신규 %d곳, 총비용 %.1f억 (%.0fs)" % (k, len(ids), ev["총비용(억/년)"], time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(res("slides", "ucurve_final.csv"), index=False, encoding="utf-8-sig")

D = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
D["센터 비용(억/년)"] = D[["임대료(억/년)", "고정 운영비(억/년)", "냉장 추가 임대료(억/년)"]].sum(axis=1)
D["배송비(억/년)"] = D[["서울 센터 배송비(억/년)", "외곽 배송비(억/년)"]].sum(axis=1)
D["현상유지 대비(억/년)"] = D["총비용(억/년)"] - BASE
D.round(2).to_csv(res("slides", "ucurve_final.csv"), index=False, encoding="utf-8-sig")
print(D[["k", "방식", "센터 비용(억/년)", "배송비(억/년)", "총비용(억/년)", "현상유지 대비(억/년)",
         "서울 센터 처리 비율(%)"]].round(1).to_string(index=False))
