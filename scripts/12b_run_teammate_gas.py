# -*- coding: utf-8 -*-
"""12b. 팀원 주유소 후보(주유소_최종후보.csv, 후보지_추천 = O)로 12번 모형을 다시 돌려 비교

팀원 파일 기준
  · 고속도로 인접 · 어린이보호구역 근처 영업 주유소 43곳은 파일에서 이미 빠져 있음
  · 후보지_추천 X: 면적 하위 25% · 면적 없음 · 좌표 없음 · 2020년 이전 폐업
  → O 299곳 (영업 234 · 휴업 10 · 폐업 55)
공시지가 · 물류창고는 팀원 파일과 우리 데이터가 같아(아래 검산) 그대로 쓴다.

사용  python scripts/12b_run_teammate_gas.py --gas 경로/주유소_최종후보.csv
출력  results/final_teamgas/compare_A_C.csv, selected_A.csv, selected_C.csv, compare_vs_before.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, res  # noqa: E402
import os, argparse, importlib.util, json
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--gas", default=os.path.join(RAW, "주유소_최종후보.csv"))
ap.add_argument("--set", default="O", choices=["O", "all"], help="O: 후보지_추천=O만, all: 파일에 있는 주유소 전부(공간 필터만 적용)")
args = ap.parse_known_args()[0]

spec = importlib.util.spec_from_file_location("m12", os.path.join(os.path.dirname(__file__), "12_run_compare.py"))
m12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m12)

G = pd.read_csv(args.gas, encoding="utf-8-sig", dtype={"관리번호": str})
keep = set(G.loc[G.후보지_추천 == "O", "관리번호"]) if args.set == "O" else set(G.관리번호)
C = m12.C
is_gas = C.type.str.startswith("주유소")
in_team = C.src_id.astype(str).isin(keep)
m12.ok_new = m12.ok_new & (~is_gas | in_team)          # 주유소는 팀원 O 목록만, 공영주차장은 그대로
print("팀원 목록(%s) %d곳" % (args.set, 0) if False else "팀원 목록 %d곳 중 우리 후보와 매칭 %d곳" % (len(keep), int((is_gas & in_team).sum())))
print("모형에 들어가는 신규 후보:", C[m12.ok_new].type.value_counts().to_dict())

OUT = res("final_teamgas" if args.set == "O" else "final_teamgas_spatial")
out, sels = {}, {}
out["현상유지(기존 27곳)"], _ = m12.solve(m12.make_F([]), statusquo=True, label="현상유지")
for k, types in m12.SETS.items():
    out["%s: %s" % (k, "영업 주유소만" if k == "A" else "주유소 전체+공영주차장")], sels[k] = m12.solve(m12.make_F(types), label=k)
S = pd.DataFrame(out)
base = S.loc["총비용(억/년)", "현상유지(기존 27곳)"]
S.loc["현상유지 대비 절감(억/년)"] = base - S.loc["총비용(억/년)"]
S = S.round(2)
S.to_csv(os.path.join(OUT, "compare_A_C.csv"), encoding="utf-8-sig")
cols = ["cand_id", "type", "name", "gu", "dong", "addr", "lon", "lat", "area_m2", "cap", "R", "g", "existing",
        "cold_open", "load", "util"]
for k, s in sels.items():
    s[cols].sort_values(["existing", "type"], ascending=[False, True]).to_csv(
        os.path.join(OUT, "selected_%s.csv" % k), index=False, encoding="utf-8-sig")
json.dump(dict(m12.PARAMS, 주유소후보="팀원 주유소_최종후보.csv (후보지_추천=O)"),
          open(os.path.join(OUT, "params.json"), "w"), ensure_ascii=False, indent=1)

# 기존 결과와 비교
B = pd.read_csv(os.path.join(res("final"), "compare_A_C.csv"), index_col=0)
rows = ["신규 개설", "총비용(억/년)", "현상유지 대비 절감(억/년)", "서울 처리 비율(%)"]
rows = [r for r in rows if r in S.index and r in B.index] or list(S.index)
V = pd.concat({"기존(우리 주유소)": B.loc[rows], "팀원 주유소": S.loc[rows]}, axis=1)
V.to_csv(os.path.join(OUT, "compare_vs_before.csv"), encoding="utf-8-sig")
print(S.to_string()); print(); print(V.to_string())
for k in sels:
    a = pd.read_csv(os.path.join(res("final"), "selected_%s.csv" % k)); b = sels[k]
    na, nb = set(a[~a.existing].cand_id), set(b[~b.existing].cand_id)
    print("%s안 신규: 기존 %d곳 → 팀원 주유소 %d곳, 공통 %d곳" % (k, len(na), len(nb), len(na & nb)))
