# -*- coding: utf-8 -*-
"""02. 법정동 경계(EPSG:5186) → EPSG:5179, 자치구·서울 경계 생성

출력  out/boundary/emd.geojson, gu.geojson, seoul.geojson (WGS84, 지도용)
      out/boundary/emd_5179.wkt.csv, gu_5179.wkt.csv, seoul_5179.wkt (분석용)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, RES, proc, res  # noqa: E402
import os, json, warnings
warnings.filterwarnings("ignore")
import pandas as pd, shapefile
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform
from pyproj import Transformer

OUT = proc("boundary")
GU_NAMES = {"11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구", "11215": "광진구",
            "11230": "동대문구", "11260": "중랑구", "11290": "성북구", "11305": "강북구", "11320": "도봉구",
            "11350": "노원구", "11380": "은평구", "11410": "서대문구", "11440": "마포구", "11470": "양천구",
            "11500": "강서구", "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
            "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구", "11740": "강동구"}

r = shapefile.Reader(os.path.join(RAW, "umd", "LSMD_ADM_SECT_UMD_11_202610"), encoding="cp949")
to5179 = Transformer.from_crs("EPSG:5186", "EPSG:5179", always_xy=True).transform
to4326 = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True).transform

rows = []
for rec, shp in zip(r.records(), r.shapes()):
    g = transform(to5179, shape(shp.__geo_interface__)).buffer(0)
    d = rec.as_dict()
    rows.append({"emd_cd": d["EMD_CD"], "emd_nm": d["EMD_NM"], "gu_cd": d["COL_ADM_SE"],
                 "gu_nm": GU_NAMES.get(d["COL_ADM_SE"], ""), "geom": g})
emd = pd.DataFrame(rows)
gu = emd.groupby(["gu_cd", "gu_nm"])["geom"].apply(lambda s: unary_union(list(s)).buffer(0)).reset_index()
seoul = unary_union(list(gu.geom)).buffer(0)


def to_geojson(df, props, path):
    feats = [{"type": "Feature", "properties": {k: r[k] for k in props},
              "geometry": mapping(transform(to4326, r["geom"]).simplify(0.00002))}
             for _, r in df.iterrows()]
    json.dump({"type": "FeatureCollection", "features": feats}, open(path, "w", encoding="utf-8"),
              ensure_ascii=False)


to_geojson(emd, ["emd_cd", "emd_nm", "gu_cd", "gu_nm"], os.path.join(OUT, "emd.geojson"))
to_geojson(gu, ["gu_cd", "gu_nm"], os.path.join(OUT, "gu.geojson"))
to_geojson(pd.DataFrame([{"name": "서울특별시", "geom": seoul}]), ["name"], os.path.join(OUT, "seoul.geojson"))
emd.assign(area_km2=emd.geom.apply(lambda g: g.area / 1e6), wkt=emd.geom.apply(lambda g: g.wkt)).drop(columns="geom") \
   .to_csv(os.path.join(OUT, "emd_5179.wkt.csv"), index=False, encoding="utf-8-sig")
gu.assign(area_km2=gu.geom.apply(lambda g: g.area / 1e6), wkt=gu.geom.apply(lambda g: g.wkt)).drop(columns="geom") \
  .to_csv(os.path.join(OUT, "gu_5179.wkt.csv"), index=False, encoding="utf-8-sig")
open(os.path.join(OUT, "seoul_5179.wkt"), "w").write(seoul.wkt)
print("법정동 %d, 자치구 %d (이름 매칭 %d), 서울 면적 %.1f km²" %
      (len(emd), len(gu), int((gu.gu_nm != "").sum()), seoul.area / 1e6))
