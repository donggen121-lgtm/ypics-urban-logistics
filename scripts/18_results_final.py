# -*- coding: utf-8 -*-
"""18. 발표용 결과 정리 (주유소 후보 = 팀원 공간 필터 450곳 기준)

① 비용: 현상유지 · A · C(최적화) + 논문 P50 · P92(위치 고정 후 우리 비용 모형으로 평가)
② 서비스 수준: 동별 20분 커버(일반 · 냉장) · 가장 가까운 센터까지 시간
   - 논문 방식은 냉장 개념이 없으므로 기존 냉장 센터만 냉장으로 본다
③ 20분 제약의 역할: 외곽 배송비를 바꿔 가며 제약 유무 비교
④ 그림: 서비스 수준 지도 3장, 재현 그림(우리 쪽만), 비용 막대
입력  results/final_teamgas_spatial/selected_*.csv (12b --set all), results/paper_compare/paper_m3_selected.csv
출력  results/slides/*.csv, *.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, res  # noqa: E402
import os, json, types
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_sys.argv = _sys.argv[:1]
src = open(os.path.join(HERE, "12_run_compare.py"), encoding="utf-8").read()
src = src.replace("elif not statusquo:", "elif not statusquo and not NO_COVER:").replace('if __name__ == "__main__":', "if False:")
m = types.ModuleType("m12"); m.__file__ = os.path.join(HERE, "12_run_compare.py"); m.NO_COVER = False
exec(compile(src, "m12", "exec"), m.__dict__)
OUT = res("slides")

GAS = pd.read_csv(os.path.join(RAW, "주유소_최종후보.csv"), encoding="utf-8-sig", dtype={"관리번호": str})
C = m.C
ok_all = m.ok_new.copy()
ok_450 = ok_all & (~C.type.str.startswith("주유소") | C.src_id.astype(str).isin(set(GAS.관리번호)))
SEL = {k: pd.read_csv(os.path.join(res("final_teamgas_spatial"), "selected_%s.csv" % k)) for k in "AC"}
PM = pd.read_csv(res("paper_compare", "paper_m3_selected.csv"))
SOL = {"현상유지": [], "A안": SEL["A"][~SEL["A"].existing].cand_id.tolist(), "C안": SEL["C"][~SEL["C"].existing].cand_id.tolist(),
       "논문 50곳": PM[PM.권역수 == 50].cand_id.tolist(), "논문 92곳": PM[PM.권역수 == 92].cand_id.tolist()}
D = m.Z["d_%s" % m.SCEN].values


def cover(ids, cold_ids):
    """열린 센터(기존+ids)로 동별 20분 커버 · 최근접 시간. cold_ids: 냉장 센터로 볼 cand_id."""
    ex = C[(C.type == "기존창고") & (C.parcel_biz == True)]
    open_ids = list(ex.cand_id) + list(ids)
    T = m.TA.loc[open_ids, m.Z.행정동코드.astype(str)].values
    tc = m.TA.loc[list(cold_ids), m.Z.행정동코드.astype(str)].values
    near = T.min(0); near_c = tc.min(0)
    return near, near_c


# ---------- ① 비용 (모두 위치 고정 후 같은 방식으로 평가) ----------
m.ok_new = ok_all
F = m.make_F(m.SETS["C"] + ["주유소_영업"])
rows, cold = {}, {}
for k, ids in SOL.items():
    if not ids:
        out, sel = m.solve(m.make_F([]), statusquo=True, label=k)
    else:
        Fk = F[F.existing | F.cand_id.isin(ids)].reset_index(drop=True)
        out, sel = m.solve(Fk, label=k, fixed_new=ids)
    cold[k] = sel[sel.cold_open > 0].cand_id.tolist()
    rows[k] = out
R = pd.DataFrame(rows)
R.loc["센터 비용(억/년)"] = R.loc[["임대료(억/년)", "고정 운영비(억/년)", "냉장 추가 임대료(억/년)"]].sum()
R.loc["현상유지 대비(억/년)"] = R.loc["총비용(억/년)"] - R.loc["총비용(억/년)", "현상유지"]
R.round(2).to_csv(os.path.join(OUT, "cost_compare.csv"), encoding="utf-8-sig")

# ---------- ② 서비스 수준 ----------
ex_cold = C[(C.type == "기존창고") & (C.parcel_biz == True) & C.cold.fillna(False).astype(bool)].cand_id.tolist()
SV = {}
NEAR = {}
for k, ids in [("현상유지", []), ("논문 50곳", SOL["논문 50곳"]), ("C안", SOL["C안"])]:
    cids = ex_cold if k != "C안" else cold["C안"]      # 논문 방식: 냉장 개념 없음 → 기존 냉장 센터만
    near, near_c = cover(ids, cids)
    NEAR[k] = (near, near_c)
    SV[k] = {"신규 센터": len(ids), "일반 20분 밖 동": int((near > 20).sum()), "냉장 20분 밖 동": int((near_c > 20).sum()),
             "20분 밖 수요(%)": round(D[near > 20].sum() / D.sum() * 100, 1),
             "최근접 평균(분)": round(np.average(near, weights=D), 1), "최근접 최대(분)": round(near.max(), 1),
             "냉장 최근접 평균(분)": round(np.average(near_c, weights=D), 1)}
SV = pd.DataFrame(SV); SV.to_csv(os.path.join(OUT, "service_level.csv"), encoding="utf-8-sig")

# ---------- ③ 20분 제약의 역할 ----------
m.ok_new = ok_450
SN = {}
for mins in [45, 20, 10]:
    m.P = m.CT * mins; m.PF = m.P * 1.5
    for nc in [False, True]:
        m.NO_COVER = nc
        out, sel = m.solve(m.make_F(m.SETS["C"]), label="P%d%s" % (mins, "-" if nc else "+"))
        new = sel[~sel.existing].cand_id.tolist(); nr, nrc = cover(new, sel[sel.cold_open > 0].cand_id.tolist())
        SN[(mins, "없음" if nc else "있음")] = {"신규 센터": out["신규 센터 수"], "총비용(억/년)": round(out["총비용(억/년)"], 1),
                                            "일반 20분 밖 동": int((nr > 20).sum()), "냉장 20분 밖 동": int((nrc > 20).sum())}
m.NO_COVER = False; m.P = m.CT * 45; m.PF = m.P * 1.5
SN = pd.DataFrame(SN); SN.to_csv(os.path.join(OUT, "cover_constraint_role.csv"), encoding="utf-8-sig")
print(R.loc[["신규 센터 수", "센터 비용(억/년)", "서울 센터 배송비(억/년)", "외곽 배송비(억/년)", "총비용(억/년)", "현상유지 대비(억/년)", "서울 센터 처리 비율(%)", "1개당 비용(원)"]].round(1).to_string())
print(SV.to_string()); print(SN.to_string())

# ---------- ④ 그림 ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely import wkt as _wkt
for f in [os.path.expanduser("~/.fonts/Pretendard-Regular.otf"), os.path.expanduser("~/.fonts/Pretendard-Bold.otf")]:
    if os.path.exists(f): font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Pretendard", "Noto Sans CJK KR", "sans-serif"]
DARK, BRIGHT, INK2 = "#004AAD", "#3A70FC", "#4A4F57"
OKC, ONLYG, BAD = "#CCDFFF", "#F6C26B", "#E5484D"
hjd = pd.read_csv(os.path.join(PROC, "boundary", "hjd_5179.wkt.csv"), encoding="utf-8-sig")
geom = hjd.set_index("zone").wkt.apply(_wkt.loads).reindex(m.Z.행정동코드.values)
gu = pd.read_csv(os.path.join(PROC, "boundary", "gu_5179.wkt.csv"), encoding="utf-8-sig").wkt.apply(_wkt.loads)


def poly(ax, g, **kw):
    for p in getattr(g, "geoms", [g]):
        ax.fill(*p.exterior.xy, **kw)


fig, axs = plt.subplots(1, 3, figsize=(18, 6.2), dpi=150)
ex = C[(C.type == "기존창고") & (C.parcel_biz == True)]
for ax, (k, ids, col) in zip(axs, [("현상유지", [], None), ("논문 50곳", SOL["논문 50곳"], "#8A9BB8"), ("C안", SOL["C안"], BRIGHT)]):
    near, near_c = NEAR[k]
    for g, a, b in zip(geom.values, near, near_c):
        c = BAD if a > 20 else (ONLYG if b > 20 else OKC)
        poly(ax, g, facecolor=c, edgecolor="white", lw=.25)
    for g in gu:
        for p in getattr(g, "geoms", [g]):
            ax.plot(*p.exterior.xy, color="#7d8794", lw=.5)
    ax.scatter(ex.x, ex.y, marker="^", s=38, color="#2b2f36", edgecolor="white", lw=.8, zorder=5)
    if ids:
        s = C[C.cand_id.isin(ids)]
        ax.scatter(s.x, s.y, s=22, color=col, edgecolor="white", lw=.7, zorder=6)
    ax.set_aspect("equal"); ax.axis("off")
plt.subplots_adjust(left=.01, right=.99, top=.99, bottom=.08, wspace=.02)
leg = [Patch(color=OKC, label="일반 · 냉장 모두 20분 내"), Patch(color=ONLYG, label="일반만 20분 내 (냉장 20분 밖)"),
       Patch(color=BAD, label="일반도 20분 밖"), Line2D([], [], marker="^", ls="", color="#2b2f36", ms=8, label="기존 택배센터 27곳"),
       Line2D([], [], marker="o", ls="", color="#8A9BB8", ms=7, label="논문 방식 신규"), Line2D([], [], marker="o", ls="", color=BRIGHT, ms=7, label="우리 C안 신규")]
fig.legend(handles=leg, loc="lower center", ncol=6, frameon=False, fontsize=12, labelcolor=INK2)
plt.savefig(os.path.join(OUT, "service_maps.png"), facecolor="white", transparent=False)
plt.close(fig)

# 재현 그림 (우리 쪽만, 논문 Figure 7·8과 같은 기호)
src17 = open(os.path.join(HERE, "17_plot_paper_fig78.py"), encoding="utf-8").read().split("pap = res(")[0]
ns = {"__file__": os.path.join(HERE, "17_plot_paper_fig78.py")}
exec(compile(src17, "s17", "exec"), ns)
plt.rcParams["font.family"] = ["Pretendard", "Noto Sans CJK KR", "sans-serif"]
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150)
for ax, L in zip(axs, (10, 20)):
    reg, kn = ns["regions"](L)
    for hull, m1, m2, m3 in reg:
        ax.plot(*hull.exterior.xy, color="black", lw=.8)
        ax.scatter(*m1, marker="x", s=46, color="red", lw=1.2, zorder=3)
        ax.scatter(*m2, marker="o", s=34, color="green", zorder=3)
        ax.scatter(*m3, marker="v", s=60, color="blue", zorder=4)
    ax.set_aspect("equal"); ax.axis("off")
plt.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01, wspace=.04)
plt.savefig(os.path.join(OUT, "replication_ours.png"), facecolor="white"); plt.close(fig)
json.dump({k: v for k, v in SOL.items()}, open(os.path.join(OUT, "solutions.json"), "w"), ensure_ascii=False)
print("saved", OUT)
