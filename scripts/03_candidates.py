# -*- coding: utf-8 -*-
"""03. 후보지 정리: 주유소(영업·휴업·폐업) + 공영주차장(노외) + 기존 물류창고

- 좌표계 EPSG:5174 / WGS84 → EPSG:5179 통일
- 좌표가 없으면 법정동 대표점으로 임시 대체 (coord_src 로 표시, 지오코딩으로 교체 예정)
- 공시지가(2026) 지번 매칭 → 연 임대료 = 면적 × 공시지가 × 5.5%
- 가장 가까운 도로 노드 연결, 서울 경계 안 여부 확인
출력  out/candidates/candidates.csv, candidates_summary.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, RES, proc, res  # noqa: E402
import os, re, warnings
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import wkt
from shapely.geometry import Point
from shapely.prepared import prep
warnings.filterwarnings("ignore")

UP = RAW
OUT = proc("candidates")
CAP_RATE = 0.055          # 연 임대수익률 (JLL 수도권 물류센터 캡레이트)
M2_PER_SPACE = 30         # 주차면당 면적(통로 포함) 가정
MIN_SPACES = 50           # 공영주차장 최소 면수
CLOSED_SINCE = "2020-01-01"
MAX_SNAP_M = 200          # 도로 접근성 기준

t5174 = Transformer.from_crs("EPSG:5174", "EPSG:5179", always_xy=True)
t4326 = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
to_ll = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

# ---------- 공시지가 ----------
g = pd.read_csv(os.path.join(RAW, "gsj", "공시지가_2026년.csv"), encoding="cp949", dtype=str)
g["P"] = pd.to_numeric(g["공시지가(원/㎡)"])
san = g.필지구분명.map({"산": "1", "임야": "1"}).fillna("0")
g["key"] = g.시군구명 + "|" + g.법정동명 + "|" + san + "|" + g.본번.str.lstrip("0") + "|" + g.부번.str.lstrip("0")
PK = g.drop_duplicates("key").set_index("key").P
PD = g.groupby(g.시군구명 + "|" + g.법정동명).P.median()

ADDR = re.compile(r"(?:서울(?:특별)?시\s+)?(\S+구)\s+(\S+?(?:동|가)\d*(?:가)?)\s+(산\s*)?(\d+)(?:-(\d+))?")


def parse(a):
    if not isinstance(a, str):
        return None
    m = ADDR.search(a)
    if not m:
        return None
    gu, dong, sn, b, s = m.groups()
    return gu, dong, "1" if sn else "0", str(int(b)), str(int(s)) if s else ""


def price(p):
    if p is None:
        return np.nan, "주소해석실패"
    k = "|".join(p)
    if k in PK.index:
        return PK[k], "필지"
    kd = p[0] + "|" + p[1]
    if kd in PD.index:
        return PD[kd], "법정동중앙값"
    return np.nan, "매칭실패"

# ---------- 경계 ----------
emd = pd.read_csv(os.path.join(PROC, "boundary", "emd_5179.wkt.csv"), dtype=str)
emd["geom"] = emd.wkt.apply(wkt.loads)
EMD_PT = {(r.gu_nm, r.emd_nm): r.geom.representative_point() for r in emd.itertuples()}
SEOUL = prep(wkt.loads(open(os.path.join(PROC, "boundary", "seoul_5179.wkt")).read()).buffer(300))  # 경계선 오차 300m 허용

# ---------- 도로 노드 ----------
N = pd.read_csv(os.path.join(PROC, "network", "nodes.csv"))
TREE = cKDTree(N[["x", "y"]].values)


_kg = os.path.join(OUT, "kakao_geocode.csv")   # 05_geocode_kakao.py 결과 (없으면 지오코딩 없이 진행)
KG = pd.read_csv(_kg, dtype={"src_id": str}) if os.path.exists(_kg) else pd.DataFrame(columns=["type", "src_id", "geo_lon", "geo_lat", "geo_method"])
KAKAO = {(t, i): (a, b, g) for t, i, a, b, g in zip(KG.type, KG.src_id, KG.geo_lon, KG.geo_lat, KG.geo_method)}


def finalize(df, typ):
    """좌표 보완·공시지가·임대료·도로연결 공통 처리."""
    p = df["addr"].map(parse)
    df["gu"] = p.map(lambda t: t[0] if t else None)
    df["dong"] = p.map(lambda t: t[1] if t else None)
    # 카카오 지오코딩 결과(05_geocode_kakao.py)로 임시·누락·서울밖 좌표 교체
    key = list(zip(df["type"], df["src_id"].astype(str)))
    hit = [KAKAO.get(k) for k in key]
    m = np.array([h is not None for h in hit])
    if m.any():
        xx, yy = t4326.transform([h[0] for h in hit if h], [h[1] for h in hit if h])
        df.loc[m, "x"] = xx; df.loc[m, "y"] = yy
        df.loc[m, "coord_src"] = ["카카오(%s)" % h[2].replace("검색", "") for h in hit if h]
    pr = p.map(price)
    df["land_price"] = pr.map(lambda t: t[0]); df["price_src"] = pr.map(lambda t: t[1])
    # 좌표 없는 곳 → 법정동 대표점
    need = df.x.isna()
    fb = [EMD_PT.get((a, b)) if n else None for a, b, n in zip(df.gu, df.dong, need)]
    df.loc[need, "x"] = [q.x if q is not None else np.nan for q, n in zip(fb, need) if n]
    df.loc[need, "y"] = [q.y if q is not None else np.nan for q, n in zip(fb, need) if n]
    df.loc[need & df.x.notna(), "coord_src"] = "법정동대표점(임시)"
    df.loc[df.x.isna(), "coord_src"] = "없음"
    ok = df.x.notna()
    df["in_seoul"] = False
    df.loc[ok, "in_seoul"] = [SEOUL.contains(Point(a, b)) for a, b in zip(df.x[ok], df.y[ok])]
    d, i = TREE.query(df.loc[ok, ["x", "y"]].values)
    df.loc[ok, "node_id"] = N.node_id.values[i]; df.loc[ok, "snap_m"] = np.round(d, 1)
    df["road_ok"] = df.snap_m <= MAX_SNAP_M
    lon, lat = to_ll.transform(df.x.values, df.y.values)
    df["lon"] = np.round(lon, 6); df["lat"] = np.round(lat, 6)
    df["rent_year"] = np.round(df.area_m2 * df.land_price * CAP_RATE, -3)
    df["rent_per_m2_year"] = np.round(df.land_price * CAP_RATE, 0)
    df["type"] = df.get("type", typ)
    return df

# ---------- 1. 주유소 ----------
o = pd.read_csv(os.path.join(UP, "자원환경_석유판매업_서울특별시.csv"), encoding="cp949", low_memory=False)
o = o[o.업태구분명 == "주유소"].copy()
o["폐업일"] = pd.to_datetime(o.폐업일자, errors="coerce")
st = np.select([o.영업상태명 == "영업/정상", o.영업상태명 == "휴업",
                (o.영업상태명 == "폐업") & (o.폐업일 >= CLOSED_SINCE)],
               ["주유소_영업", "주유소_휴업", "주유소_폐업"], default="")
o["type"] = st
o = o[o.type != ""]
gas = pd.DataFrame({
    "type": o.type, "src_id": o.관리번호.astype(str), "name": o.사업장명,
    "addr": o.지번주소.fillna(o.도로명주소),
    "x": pd.to_numeric(o["좌표정보(X)"], errors="coerce"), "y": pd.to_numeric(o["좌표정보(Y)"], errors="coerce"),
    "area_m2": pd.to_numeric(o.소재지면적, errors="coerce"), "closed_date": o.폐업일.dt.strftime("%Y-%m-%d"),
})
ok = gas.x.notna() & gas.y.notna()
gas.loc[ok, "x"], gas.loc[ok, "y"] = t5174.transform(gas.x[ok].values, gas.y[ok].values)
gas["coord_src"] = np.where(ok, "인허가좌표", None)
gas["area_src"] = np.where(gas.area_m2 > 0, "인허가면적", "유형중앙값")
gas.loc[~(gas.area_m2 > 0), "area_m2"] = gas.loc[gas.area_m2 > 0, "area_m2"].median()
gas = finalize(gas, None)

# ---------- 2. 공영주차장 (노외, 50면 이상) ----------
p = pd.read_csv(os.path.join(UP, "서울시 공영주차장 안내 정보.csv"), encoding="cp949", low_memory=False)
p = p[p["주차장 종류명"] == "노외 주차장"].drop_duplicates("주차장코드")
p = p[p["총 주차면"] >= MIN_SPACES]
park = pd.DataFrame({
    "type": "공영주차장", "src_id": p.주차장코드.astype(str), "name": p.주차장명,
    "addr": p.주소.astype(str), "spaces": p["총 주차면"],
    "x": np.where(p.위도.fillna(0) > 0, p.경도, np.nan), "y": np.where(p.위도.fillna(0) > 0, p.위도, np.nan),
})
ok = park.x.notna()
park.loc[ok, "x"], park.loc[ok, "y"] = t4326.transform(park.x[ok].values, park.y[ok].values)
park["coord_src"] = np.where(ok, "제공좌표", None)

# 좌표 없는 곳 → 전국주차장정보표준데이터 (지번주소 → 같은 구 안 이름 순으로 매칭)
std = pd.read_csv(os.path.join(UP, "전국주차장정보표준데이터.csv"), encoding="cp949", low_memory=False)
std = std[(std.소재지지번주소.fillna("") + std.소재지도로명주소.fillna("") + std.제공기관명.fillna(""))
          .str.contains("서울") & (std.위도.fillna(0) > 30)]


def norm_addr(a):
    if not isinstance(a, str):
        return None
    a = re.sub(r"^서울(특별)?시\s+", "", a.strip())
    m = re.match(r"(\S+구)\s+(\S+)\s+(산\s*)?(\d+)(?:-(\d+))?", a)
    if not m:
        return None
    s = m.group(5)
    return "%s %s %s%d" % (m.group(1), m.group(2), "산" if m.group(3) else "", int(m.group(4))) + \
        ("-%d" % int(s) if s and int(s) > 0 else "")


def norm_name(s):
    return re.sub(r"\(.*?\)|공영주차장|주차장|공영|\s", "", str(s))


std = std.assign(k=std.소재지지번주소.map(norm_addr), nm=std.주차장명.map(norm_name),
                 gu=std.소재지지번주소.fillna(std.소재지도로명주소).str.extract(r"(\S+구)")[0])
by_addr = std.drop_duplicates("k").set_index("k")[["위도", "경도"]]
by_name = std.drop_duplicates(["gu", "nm"]).set_index(["gu", "nm"])[["위도", "경도"]]
pk = park.addr.map(norm_addr)
pgu = park.addr.str.extract(r"(\S+구)")[0]
pnm = park.name.map(norm_name)
bad = park.x.isna()
if bad.any():  # 제공좌표가 서울 밖이면 버리고 다시 찾는다
    pass
for i in park.index:
    if park.at[i, "coord_src"] == "제공좌표":
        if SEOUL.contains(Point(park.at[i, "x"], park.at[i, "y"])):
            continue
        park.at[i, "x"] = np.nan; park.at[i, "y"] = np.nan; park.at[i, "coord_src"] = None
    hit, how = None, None
    if pk[i] in by_addr.index:
        hit, how = by_addr.loc[pk[i]], "표준데이터(주소)"
    elif (pgu[i], pnm[i]) in by_name.index:
        hit, how = by_name.loc[(pgu[i], pnm[i])], "표준데이터(이름)"
    if hit is not None:
        park.at[i, "x"], park.at[i, "y"] = t4326.transform(hit["경도"], hit["위도"])
        park.at[i, "coord_src"] = how
park["area_m2"] = park.spaces * M2_PER_SPACE; park["area_src"] = "면수×30㎡"
park = finalize(park, None)

# ---------- 3. 기존 물류창고 (영업 중) ----------
w = pd.read_csv(os.path.join(UP, "서울시 물류창고업체 인허가 정보.csv"), encoding="cp949", low_memory=False)
w = w[w.영업상태명 == "등록"].copy()
num = lambda c: pd.to_numeric(w[c], errors="coerce").fillna(0)
wh = pd.DataFrame({
    "type": "기존창고", "src_id": w.관리번호.astype(str), "name": w.사업장명,
    "addr": w.지번주소.fillna(w.도로명주소),
    "x": pd.to_numeric(w["좌표정보(X)"], errors="coerce"), "y": pd.to_numeric(w["좌표정보(Y)"], errors="coerce"),
    "area_m2": num("일반창고_면적") + num("냉동,냉장창고_면적") + num("보관장소_면적"),
    "cold": num("냉동,냉장창고_면적") > 0, "parcel_biz": num("업태_운송및택배업") > 0,
})
ok = wh.x.notna() & wh.y.notna()
wh.loc[ok, "x"], wh.loc[ok, "y"] = t5174.transform(wh.x[ok].values, wh.y[ok].values)
wh["coord_src"] = np.where(ok, "인허가좌표", None); wh["area_src"] = "인허가면적"
wh = finalize(wh, None)

# ---------- 합치기 ----------
C = pd.concat([gas, park, wh], ignore_index=True)
C = C[C.coord_src != "없음"].reset_index(drop=True)   # 주소 자체가 깨진 곳 제외
C["coord_approx"] = C.coord_src == "법정동대표점(임시)"  # 지오코딩 실패 → 최적화 기본 집합에서 제외
C.insert(0, "cand_id", ["C%04d" % i for i in range(len(C))])
cols = ["cand_id", "type", "name", "addr", "gu", "dong", "x", "y", "lon", "lat", "coord_src", "in_seoul",
        "node_id", "snap_m", "road_ok", "area_m2", "area_src", "land_price", "price_src", "rent_year",
        "rent_per_m2_year", "coord_approx", "spaces", "cold", "parcel_biz", "closed_date", "src_id"]
C = C[cols]
C.to_csv(os.path.join(OUT, "candidates.csv"), index=False, encoding="utf-8-sig")

S = C.groupby("type").agg(
    개수=("cand_id", "size"),
    좌표_원본=("coord_src", lambda s: int(s.isin(["인허가좌표", "제공좌표"]).sum())),
    좌표_카카오=("coord_src", lambda s: int(s.str.startswith("카카오", na=False).sum())),
    좌표_표준데이터=("coord_src", lambda s: int(s.str.startswith("표준데이터", na=False).sum())),
    좌표_임시=("coord_src", lambda s: int((s == "법정동대표점(임시)").sum())),
    좌표_없음=("coord_src", lambda s: int((s == "없음").sum())),
    서울밖=("in_seoul", lambda s: int((~s.astype(bool)).sum())),
    도로200m초과=("road_ok", lambda s: int((~s.astype(bool)).sum())),
    공시지가_필지=("price_src", lambda s: int((s == "필지").sum())),
    면적중앙값_m2=("area_m2", "median"),
    공시지가중앙값_만원=("land_price", lambda s: round(s.median() / 1e4, 0)),
    연임대료중앙값_억원=("rent_year", lambda s: round(s.median() / 1e8, 2)),
).reset_index()
S.to_csv(os.path.join(OUT, "candidates_summary.csv"), index=False, encoding="utf-8-sig")
print(S.to_string(index=False))
