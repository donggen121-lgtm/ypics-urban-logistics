# -*- coding: utf-8 -*-
"""06. 행정동 경계(admdongkor ver20260701) ↔ 생활인구(2026.07) 행정동 코드 맞추기

생활인구 데이터는 옛 행정동 코드를 써서 최신 경계와 9개 동이 어긋난다.
→ 대응표로 경계를 '생활인구 코드 기준 구역(zone)'으로 묶는다.
   - 강북구 번1~3동·수유1~3동: 코드 체계만 다름 → 순서대로 1:1 (가정)
   - 강남구 개포3동 ← 옛 코드 11680740 (1:1)
   - 동대문구 신설동+용두동 → 옛 코드 11230536 하나로 합침
   - 강동구 상일1동+상일2동 → 옛 코드 11740520 하나로 합침
   - 구로구 항동(신설) → 경계를 가장 길게 맞댄 이웃 동에 합침
출력  out/boundary/hjd.geojson, hjd_5179.wkt.csv (zone 단위)
      out/demand/행정동_야간인구_202607.csv (zone 대표점·면적 추가)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, RES, proc, res  # noqa: E402
import os, json, warnings
import pandas as pd
from shapely.geometry import shape, mapping
from shapely.ops import transform, unary_union
from pyproj import Transformer
warnings.filterwarnings("ignore")

SRC = os.path.join(RAW, "HangJeongDong_ver20260701.geojson")   # github.com/vuski/admdongkor
POP = proc("demand", "행정동_야간인구_202607.csv")            # 05a_living_population.py 결과
to5179 = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True).transform
to4326 = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True).transform

XWALK = {  # 경계 코드(2026) → 생활인구 코드
    "11305595": "11305590", "11305603": "11305600", "11305608": "11305606",
    "11305615": "11305610", "11305625": "11305620", "11305635": "11305630",
    "11680675": "11680740",
    "11230515": "11230536", "11230533": "11230536",
    "11740525": "11740520", "11740526": "11740520",
}

gj = json.load(open(SRC, encoding="utf-8"))
rows = []
for f in gj["features"]:
    p = f["properties"]
    if str(p.get("sido")) != "11":
        continue
    rows.append({"hjd_cd8": str(p["adm_cd2"])[:8], "gu_nm": p["sggnm"], "hjd_nm": p["adm_nm"].split()[-1],
                 "geom": transform(to5179, shape(f["geometry"])).buffer(0)})
H = pd.DataFrame(rows)
pop = pd.read_csv(POP, dtype={"행정동코드": str, "구코드": str})
popset = set(pop.행정동코드)
H["zone"] = H.hjd_cd8.map(lambda c: c if c in popset else XWALK.get(c))

# 대응표에도 없는 신설 동(항동 등) → 같은 구 안에서 경계를 가장 길게 맞댄 이웃 zone
for i in H.index[H.zone.isna()]:
    g = H.at[i, "geom"]
    cand = H[(H.gu_nm == H.at[i, "gu_nm"]) & H.zone.notna()]
    shared = cand.geom.apply(lambda h: g.boundary.intersection(h.boundary).length)
    j = shared.idxmax()
    H.at[i, "zone"] = H.at[j, "zone"]
    print("신설 동 %s %s → %s %s 에 합침 (맞댄 경계 %.0fm)" %
          (H.at[i, "gu_nm"], H.at[i, "hjd_nm"], H.at[j, "hjd_nm"], H.at[j, "zone"], shared.max()))

Z = H.groupby("zone").agg(gu_nm=("gu_nm", "first"), hjd_nm=("hjd_nm", lambda s: "+".join(s)),
                          n_dong=("hjd_cd8", "size"),
                          geom=("geom", lambda s: unary_union(list(s)).buffer(0))).reset_index()
Z["rep_x"] = Z.geom.apply(lambda g: g.representative_point().x)
Z["rep_y"] = Z.geom.apply(lambda g: g.representative_point().y)
Z["area_km2"] = Z.geom.apply(lambda g: g.area / 1e6)
miss_pop = popset - set(Z.zone); miss_geo = set(Z.zone) - popset
print("zone %d개 (경계 동 %d개 → 합침), 생활인구 코드 %d개 | 인구만 %d, 경계만 %d" %
      (len(Z), len(H), len(popset), len(miss_pop), len(miss_geo)))

Z.assign(wkt=Z.geom.apply(lambda g: g.wkt)).drop(columns="geom") \
 .to_csv(proc("boundary", "hjd_5179.wkt.csv"), index=False, encoding="utf-8-sig")
feats = [{"type": "Feature", "properties": {"zone": r.zone, "hjd_nm": r.hjd_nm, "gu_nm": r.gu_nm},
          "geometry": mapping(transform(to4326, r.geom).simplify(0.00002))} for r in Z.itertuples()]
json.dump({"type": "FeatureCollection", "features": feats},
          open(proc("boundary", "hjd.geojson"), "w", encoding="utf-8"), ensure_ascii=False)

out = pop.merge(Z[["zone", "gu_nm", "hjd_nm", "n_dong", "rep_x", "rep_y", "area_km2"]],
                left_on="행정동코드", right_on="zone", how="left").drop(columns="zone")
out["야간인구밀도"] = (out.야간평균 / out.area_km2).round(0)
out.to_csv(POP, index=False, encoding="utf-8-sig")
print("구내 비중 합 검증(구별 1.0):", out.groupby("구코드").구내_야간비중.sum().round(6).unique())
print(out[out.n_dong > 1][["행정동코드", "gu_nm", "hjd_nm", "야간평균"]].to_string(index=False))
