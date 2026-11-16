# -*- coding: utf-8 -*-
"""05a. 행정동 단위 서울 생활인구(내국인, 2026년 7월) → 행정동별 야간·주간 평균과 구 안 비중

입력  data/raw/LOCAL_PEOPLE_DONG_202607.zip (서울 열린데이터광장 OA-14991, 압축 그대로)
출력  data/processed/demand/행정동_야간인구_202607.csv
      행정동코드, 야간평균(19~08시), 주간평균(09~18시), 구코드, 구내_야간비중, 야간/주간
      → 07_demand_forecast.py 가 자치구 수요를 행정동으로 나눌 때 구내_야간비중을 쓴다
      → 06_hjd_boundary.py 가 경계 구역(zone) 대표점·면적을 이 파일에 붙인다
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, proc  # noqa: E402
import os, zipfile
import pandas as pd

src = os.path.join(RAW, "LOCAL_PEOPLE_DONG_202607.zip")
with zipfile.ZipFile(src) as z:
    with z.open(z.namelist()[0]) as f:
        d = pd.read_csv(f, encoding="utf-8-sig", index_col=False,   # 행 끝 쉼표 때문에 index_col=False
                        usecols=["기준일ID", "시간대구분", "행정동코드", "총생활인구수"], dtype={"행정동코드": str})
night = d.시간대구분.isin(list(range(19, 24)) + list(range(0, 9)))
day = d.시간대구분.between(9, 18)
P = pd.DataFrame({
    "야간평균": d[night].groupby("행정동코드").총생활인구수.mean(),
    "주간평균": d[day].groupby("행정동코드").총생활인구수.mean(),
}).reset_index()
P["구코드"] = P.행정동코드.str[:5]
P["구내_야간비중"] = (P.야간평균 / P.groupby("구코드").야간평균.transform("sum")).round(4)
P["야간/주간"] = (P.야간평균 / P.주간평균).round(3)
P[["야간평균", "주간평균"]] = P[["야간평균", "주간평균"]].round(4)
out = proc("demand", "행정동_야간인구_202607.csv")
P.to_csv(out, index=False, encoding="utf-8-sig")
print("행정동 %d개, 일수 %d, 서울 야간 평균 합 %.0f명 → %s" % (len(P), d.기준일ID.nunique(), P.야간평균.sum(), out))
