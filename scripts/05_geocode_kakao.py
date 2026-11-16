# -*- coding: utf-8 -*-
"""05. 카카오 로컬 API로 좌표가 없는(임시) 후보지 지오코딩

사용법 (저장소 최상위 폴더에서, 03_candidates.py 를 한 번 돌린 뒤):
  1) python scripts/05_geocode_kakao.py
  2) REST API 키를 물어보면 붙여넣기 (또는 환경변수 KAKAO_REST_KEY 로 지정). 키는 절대 커밋하지 않는다.
  3) 결과: data/processed/candidates/candidates_geocoded.csv, kakao_geocode.csv
  4) python scripts/03_candidates.py 를 다시 돌리면 kakao_geocode.csv 좌표가 반영된다
     - geo_lon, geo_lat : 찾은 좌표 (WGS84)
     - geo_method       : 주소검색 / 키워드검색 / 실패
     - geo_addr         : 카카오가 돌려준 주소 (맞게 찾았는지 확인용)

필요한 것: Python 3.6+ 와 pandas (그 외는 표준 라이브러리만 사용)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import PROC  # noqa: E402
import os, sys, json, time, getpass
import urllib.parse, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(PROC, "candidates", "candidates.csv")
DST = os.path.join(PROC, "candidates", "candidates_geocoded.csv")

KEY = os.environ.get("KAKAO_REST_KEY") or getpass.getpass("카카오 REST API 키: ").strip()
HEAD = {"Authorization": "KakaoAK " + KEY}
SEOUL_RECT = "126.76,37.42,127.19,37.71"   # 키워드 검색을 서울 범위로 제한


def call(url, params):
    q = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(q, headers=HEAD)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            if e.code in (401, 403):
                sys.exit("인증 오류(%d): 키 또는 '카카오맵 사용 설정'을 확인하세요.\n%s" % (e.code, body))
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    return {"documents": []}


def by_address(addr):
    a = addr if addr.startswith("서울") else "서울특별시 " + addr
    d = call("https://dapi.kakao.com/v2/local/search/address.json", {"query": a, "size": 1})["documents"]
    if d:
        return float(d[0]["x"]), float(d[0]["y"]), "주소검색", d[0].get("address_name", "")
    return None


def by_keyword(name, gu):
    q = ("%s %s" % (gu or "", name)).strip()
    d = call("https://dapi.kakao.com/v2/local/search/keyword.json",
             {"query": q, "rect": SEOUL_RECT, "size": 1})["documents"]
    if d:
        return float(d[0]["x"]), float(d[0]["y"]), "키워드검색", d[0].get("place_name", "") + " / " + \
            d[0].get("address_name", "")
    return None


C = pd.read_csv(SRC, encoding="utf-8-sig")
need = C.coord_src.isin(["법정동대표점(임시)", "없음"]) | ~C.in_seoul.astype(bool)
print("지오코딩 대상 %d곳 (전체 %d곳)" % (need.sum(), len(C)))
for c in ["geo_lon", "geo_lat", "geo_method", "geo_addr"]:
    C[c] = None
for n, i in enumerate(C.index[need], 1):
    row = C.loc[i]
    res = by_address(str(row.addr)) if isinstance(row.addr, str) else None
    if res is None:
        res = by_keyword(str(row["name"]), row.gu if isinstance(row.gu, str) else "")
    if res is None:
        C.loc[i, "geo_method"] = "실패"
    else:
        C.loc[i, ["geo_lon", "geo_lat", "geo_method", "geo_addr"]] = list(res)
    if n % 50 == 0:
        print("  %d / %d" % (n, need.sum()))
    time.sleep(0.05)
C.to_csv(DST, index=False, encoding="utf-8-sig")
print("완료:", DST)
print(C.loc[need, "geo_method"].value_counts().to_string())
# 03_candidates.py 가 읽는 성공분만 따로 저장
ok = C[C.geo_method.isin(["주소검색", "키워드검색"])]
ok[["type", "src_id", "name", "geo_lon", "geo_lat", "geo_method", "geo_addr"]].to_csv(
    os.path.join(PROC, "candidates", "kakao_geocode.csv"), index=False, encoding="utf-8-sig")
