# -*- coding: utf-8 -*-
"""07a. 서울 생활물류(CJ대한통운 송장 기반) 월별 파일 72개 → 일 × 자치구 × 품목 표

입력  data/raw/서울시 생활물류/DWC_KXLCLS_OD_DAY_SIDO_SEOULGU_YYYYMM.csv (2018.01~2023.12, cp949)
      출지 = 전국 시도, 착지 = 서울 자치구, 11개 대분류 착지 물동량
출력  data/processed/demand/생활물류_일별_구_품목.csv
      배송년월일, 수하인_구코드, 수하인_구명, 품목 11개, 서울발(출지가 서울인 물량)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, proc  # noqa: E402
import glob, os
import pandas as pd

fs = sorted(glob.glob(os.path.join(RAW, "서울시 생활물류", "DWC_KXLCLS_*.csv")))
assert fs, "data/raw/서울시 생활물류/ 에 DWC_KXLCLS_*.csv 파일을 넣어 주세요"
L = []
for f in fs:
    d = pd.read_csv(f, encoding="cp949")
    cat = [c for c in d.columns if c.startswith("대분류_")]
    key = ["배송년월일", "수하인_구코드", "수하인_구명"]
    g = d.groupby(key)[cat].sum()
    gi = d[d.송하인_시코드 == 11].groupby(key)[cat].sum().sum(axis=1).rename("서울발")
    L.append(g.join(gi))
A = pd.concat(L).reset_index()
A.columns = [c.replace("대분류_착지물동량 ", "") for c in A.columns]
out = proc("demand", "생활물류_일별_구_품목.csv")
A.to_csv(out, index=False, encoding="utf-8-sig")
print("%d개 파일 → %s행, %s ~ %s, 자치구 %d개" % (len(fs), len(A), A.배송년월일.min(), A.배송년월일.max(),
                                            A.수하인_구코드.nunique()))
print(out)
