# -*- coding: utf-8 -*-
"""
빅데이터캠퍼스 'CJ대한통운 택배 유형별 (행정동 단위)' 반출용 집계 스크립트
=====================================================================

[무엇을 하나]
캠퍼스 원본(일별 출지·착지, 월별 OD)을 읽어서, 반출 심사를 받기 쉬운
'월 × 행정동 × 품목' 수준의 집계표와 피크 통계, 품질 리포트를 만든다.

[사용법]
1) 아래 '설정'에서 INPUT_DIR(원본 CSV가 있는 폴더)와 OUTPUT_DIR을 바꾼다.
   - 하위 폴더까지 *.csv 를 모두 찾고, 헤더를 보고 파일 종류를 자동 판별한다.
2) 데이터 정의서(품목 코드표)를 보고 FRESH_SUBCODES 에 신선 품목 세부코드를 적는다.
   - 예: 세부코드 '07_01' 이 농산물이면 "07_01" (접두어 s_c_/e_c_ 없이)
   - 비워두면 신선 집계만 건너뛰고 나머지는 모두 만든다.
3) 실행:  python bigdata_campus_aggregate.py
4) OUTPUT_DIR 의 CSV 들을 반출 신청한다. (원본 일별 데이터는 출력하지 않음)

[필요 환경] Python 3.6+, pandas, numpy (인터넷 연결 불필요)

[출력 파일]
  T1_착지_월별_행정동.csv        년월 × 행정동 × 대분류 (전체일 / 정상평일 합계, 정상평일 수)
  T2_신선_월별_행정동.csv        년월 × 행정동 × 신선·사용자 그룹 (FRESH_SUBCODES 설정 시)
  T3_피크통계_행정동.csv         연도 × 행정동 × 품목: 정상평일 평균·상위분위·최대·피크계수·주말비
  T3_피크통계_자치구.csv         같은 통계를 자치구 단위로 (동 단위 반출이 거절될 때 대안)
  T5_출지_월별_행정동.csv        T1 과 같은 형식의 출지 버전
  T6a_OD_서울에서_전국_월별.csv  년월 × 보낸 서울 행정동 × 목적지 권역(서울/수도권/지방)
  T6b_OD_전국에서_서울_월별.csv  년월 × 받은 서울 행정동 × 출발 권역
  T6c_OD_서울내_구간_월별.csv    년월 × 보낸 구 × 받은 구 (서울 안 이동)
  T7_서울전체_일별_품목.csv      일 × 대분류 서울 합계 (25개 구 합, 식별위험 낮음)
  Q0_달력_정상평일판정.csv       날짜별 공휴일·비정상일 판정 근거
  Q1_품질리포트.txt              파일별 행 수·기간·동 수·0 비율·대분류-세부분류 합계 일치 여부
  Q2_행정동코드_이력.csv         동 코드별 첫/마지막 등장일 (통폐합 확인용)
  Q3_코드표_작성용.csv           데이터에 있는 품목 코드 목록 (품목명은 정의서 보고 채우기)
"""

import os
import re
import sys
import glob
import datetime as dt

import numpy as np
import pandas as pd

# =====================================================================
# 설정
# =====================================================================
INPUT_DIR = r"./input"
OUTPUT_DIR = r"./output"

# 신선 품목 세부코드 (예: ["07_01", "07_02"]). 코드표 확인 후 채우기.
FRESH_SUBCODES = []

# 추가로 묶어서 보고 싶은 품목 그룹 (선택). 예: {"냉동식품": ["07_05"]}
CUSTOM_GROUPS = {}

PEAK_QUANTILE = 0.95     # 피크 = 정상평일 상위 5% 값
ABNORMAL_RATIO = 0.6     # 같은 요일 주변(±4주) 중앙값의 60% 미만이면 비정상일(연휴 등)로 판정
MASK_BELOW = None        # 반출 규정상 소수 셀을 가려야 하면 숫자 지정 (예: 10 → 1~9 를 빈칸 처리)
CHUNKSIZE = 200000       # 큰 파일을 나눠 읽는 행 수

SEOUL_CD = 11
METRO_CDS = {28, 41}     # 인천, 경기 → '수도권'

# 한국 공휴일 (대체·임시공휴일, 선거일 포함). 데이터 기반 판정과 함께 쓰이므로 일부 누락돼도 보완됨.
KR_HOLIDAYS = {
    2020: ["01-01", "01-24", "01-25", "01-26", "01-27", "03-01", "04-15", "04-30", "05-05",
           "06-06", "08-15", "08-17", "09-30", "10-01", "10-02", "10-03", "10-09", "12-25"],
    2021: ["01-01", "02-11", "02-12", "02-13", "03-01", "05-05", "05-19", "06-06", "08-15",
           "08-16", "09-20", "09-21", "09-22", "10-03", "10-04", "10-09", "10-11", "12-25"],
    2022: ["01-01", "01-31", "02-01", "02-02", "03-01", "03-09", "05-05", "05-08", "06-01",
           "06-06", "08-15", "09-09", "09-10", "09-11", "09-12", "10-03", "10-09", "10-10", "12-25"],
    2023: ["01-01", "01-21", "01-22", "01-23", "01-24", "03-01", "05-05", "05-27", "05-29",
           "06-06", "08-15", "09-28", "09-29", "09-30", "10-02", "10-03", "10-09", "12-25"],
    2024: ["01-01", "02-09", "02-10", "02-11", "02-12", "03-01", "04-10", "05-05", "05-06",
           "05-15", "06-06", "08-15", "09-16", "09-17", "09-18", "10-01", "10-03", "10-09", "12-25"],
    2025: ["01-01", "01-27", "01-28", "01-29", "01-30", "03-01", "03-03", "05-05", "05-06",
           "06-03", "06-06", "08-15", "10-03", "10-05", "10-06", "10-07", "10-08", "10-09", "12-25"],
    2026: ["01-01", "02-16", "02-17", "02-18", "03-01", "03-02", "05-05", "05-24", "05-25",
           "06-03", "06-06", "08-15", "08-17", "09-24", "09-25", "09-26", "10-03", "10-05",
           "10-09", "12-25"],
}

# =====================================================================
# 공통 유틸
# =====================================================================
TOP_RE = re.compile(r"^([se])_c_(\d{2})$")
SUB_RE = re.compile(r"^([se])_c_(\d{2})_(\d{2})$")
ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]

LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


def detect_encoding(path):
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                f.readline()
                f.read(65536)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError("인코딩을 알 수 없음: %s" % path)


def read_columns(path, enc):
    return list(pd.read_csv(path, encoding=enc, nrows=0).columns)


def classify(cols):
    """헤더로 파일 종류 판별."""
    has_s = any(TOP_RE.match(c) and c.startswith("s_") for c in cols)
    has_e = any(TOP_RE.match(c) and c.startswith("e_") for c in cols)
    if "dl_ymd" in cols and "hjdong_cd" in cols:
        if has_s and not has_e:
            return "daily_send"
        if has_e and not has_s:
            return "daily_rec"
    if "dl_ym" in cols and "send_hjdong_cd" in cols and "rec_hjdong_cd" in cols:
        return "od"
    return None


def value_columns(cols):
    tops = [c for c in cols if TOP_RE.match(c)]
    subs = [c for c in cols if SUB_RE.match(c)]
    return tops, subs


def cat_code(col):
    """'s_c_07' → '07', 's_c_07_01' → '07_01'."""
    return col.split("_c_", 1)[1]


def group_columns(subs):
    """신선·사용자 그룹 → 실제 컬럼명 목록."""
    groups = {}
    wanted = {}
    if FRESH_SUBCODES:
        wanted["신선"] = FRESH_SUBCODES
    wanted.update(CUSTOM_GROUPS)
    sub_by_code = {cat_code(c): c for c in subs}
    for name, codes in wanted.items():
        cols = [sub_by_code[c] for c in codes if c in sub_by_code]
        missing = [c for c in codes if c not in sub_by_code]
        if missing:
            log("  [경고] 그룹 '%s' 의 코드 %s 가 데이터에 없음" % (name, missing))
        if cols:
            groups[name] = cols
    return groups


def mask_small(df, cols):
    if MASK_BELOW is None:
        return df
    df = df.copy()
    for c in cols:
        s = df[c]
        df[c] = s.where(~((s > 0) & (s < MASK_BELOW)), np.nan)
    return df


def save(df, name):
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log("  저장: %s (%d행 × %d열)" % (name, len(df), df.shape[1]))


# =====================================================================
# 1. 일별 출지·착지 읽기 → 동 × 일 × (대분류 + 합계 + 그룹)
# =====================================================================
ID_DAILY = ["dl_ymd", "ctgg_cd", "ctgg_nm", "hjdong_cd", "hjdong_nm"]


def load_daily(files, kind, quality):
    frames = []
    for path, enc, cols in files:
        tops, subs = value_columns(cols)
        groups = group_columns(subs)
        usecols = [c for c in ID_DAILY if c in cols] + tops + subs
        n_rows, n_zero, n_cells = 0, 0, 0
        mismatch = {}
        for ch in pd.read_csv(path, encoding=enc, usecols=usecols, chunksize=CHUNKSIZE,
                              low_memory=False):
            vals = ch[tops + subs].apply(pd.to_numeric, errors="coerce").fillna(0)
            n_rows += len(ch)
            n_zero += int((vals == 0).values.sum())
            n_cells += vals.size
            # 대분류 = 세부분류 합 인지 점검
            for t in tops:
                code = cat_code(t)
                kids = [s for s in subs if cat_code(s).startswith(code + "_")]
                if kids:
                    diff = int((vals[t] != vals[kids].sum(axis=1)).sum())
                    mismatch[code] = mismatch.get(code, 0) + diff
            out = ch[[c for c in ID_DAILY if c in ch.columns]].copy()
            for t in tops:
                out["c_" + cat_code(t)] = vals[t].astype("int64")
            out["합계"] = vals[tops].sum(axis=1).astype("int64")
            for gname, gcols in groups.items():
                out[gname] = vals[gcols].sum(axis=1).astype("int64")
            frames.append(out)
        quality.append({
            "파일": os.path.basename(path), "종류": kind, "행수": n_rows,
            "0 비율": round(n_zero / max(n_cells, 1), 3),
            "대분류≠세부합 행수": mismatch,
        })
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["dl_ymd"].astype(str).str[:8], format="%Y%m%d")
    # 같은 동·날짜가 여러 파일에 중복되면 합산하지 않고 하나만 남긴다
    key = ["date", "hjdong_cd"]
    dup = int(df.duplicated(key).sum())
    if dup:
        log("  [경고] %s: 동·날짜 중복 %d행 → 첫 행만 사용" % (kind, dup))
        df = df.drop_duplicates(key)
    return df


# =====================================================================
# 2. 달력: 공휴일 + 데이터 기반 비정상일 → 정상평일
# =====================================================================
def build_calendar(daily):
    seoul = daily.groupby("date")["합계"].sum().sort_index()
    cal = seoul.rename("서울합계").to_frame()
    cal["요일"] = cal.index.dayofweek  # 0=월
    hol = set()
    for y, days in KR_HOLIDAYS.items():
        for d in days:
            hol.add(pd.Timestamp("%d-%s" % (y, d)))
    cal["공휴일"] = cal.index.isin(hol)
    # 같은 요일 ±4주 중앙값 대비 비율
    ratio = []
    for d, row in cal.iterrows():
        same = cal[(cal["요일"] == row["요일"]) &
                   (abs((cal.index - d).days) <= 28) & (cal.index != d)]["서울합계"]
        med = same.median() if len(same) else np.nan
        ratio.append(row["서울합계"] / med if med and med > 0 else np.nan)
    cal["주변중앙값대비"] = np.round(ratio, 3)
    cal["비정상일"] = cal["주변중앙값대비"] < ABNORMAL_RATIO
    cal["평일"] = cal["요일"] < 5
    cal["정상평일"] = cal["평일"] & ~cal["공휴일"] & ~cal["비정상일"]

    def reason(r):
        if not r["평일"]:
            return "주말"
        why = []
        if r["공휴일"]:
            why.append("공휴일")
        if r["비정상일"]:
            why.append("물량급감")
        return ",".join(why) if why else "정상"
    cal["판정"] = cal.apply(reason, axis=1)
    extra = cal[cal["평일"] & cal["비정상일"] & ~cal["공휴일"]]
    if len(extra):
        log("  공휴일 목록에 없는데 물량이 급감한 평일 %d일: %s" %
            (len(extra), ", ".join(d.strftime("%Y-%m-%d") for d in extra.index[:20])))
    return cal


# =====================================================================
# 3. 월별 집계 (T1, T2, T5)
# =====================================================================
def attach_calendar(daily, cal):
    d = daily.merge(cal[["정상평일"]], left_on="date", right_index=True, how="left")
    d["정상평일"] = d["정상평일"].fillna(False).astype(bool)
    d["요일"] = d["date"].dt.dayofweek
    return d


def monthly_table(daily, cal, value_cols):
    d = attach_calendar(daily, cal)
    d["년월"] = d["date"].dt.strftime("%Y%m")
    keys = ["년월", "ctgg_cd", "ctgg_nm", "hjdong_cd", "hjdong_nm"]
    keys = [k for k in keys if k in d.columns]
    a = d.groupby(keys)[value_cols].sum().reset_index()
    a.insert(len(keys), "구분", "전체일")
    w = d[d["정상평일"]].groupby(keys)[value_cols].sum().reset_index()
    w.insert(len(keys), "구분", "정상평일")
    out = pd.concat([a, w], ignore_index=True)
    cal_m = cal.copy()
    cal_m["년월"] = cal_m.index.strftime("%Y%m")
    nwd = cal_m.groupby("년월").agg(일수=("서울합계", "size"), 정상평일수=("정상평일", "sum"))
    out = out.merge(nwd, left_on="년월", right_index=True, how="left")
    out = out.sort_values(keys + ["구분"]).reset_index(drop=True)
    return mask_small(out, value_cols)


# =====================================================================
# 4. 피크 통계 (T3)
# =====================================================================
def peak_stats(daily, cal, value_cols, level):
    d = attach_calendar(daily, cal)
    d["연도"] = d["date"].dt.year
    if level == "자치구":
        keys = ["ctgg_cd", "ctgg_nm"]
        d = d.groupby(["date", "연도", "정상평일", "요일"] + keys)[value_cols].sum().reset_index()
    else:
        keys = ["ctgg_cd", "ctgg_nm", "hjdong_cd", "hjdong_nm"]
    rows = []
    for (yr, *k), g in d.groupby(["연도"] + keys):
        wd = g[g["정상평일"]]
        we = g[g["요일"] >= 5]
        for c in value_cols:
            m = wd[c].mean() if len(wd) else np.nan
            q = wd[c].quantile(PEAK_QUANTILE) if len(wd) else np.nan
            mx = wd[c].max() if len(wd) else np.nan
            wem = we[c].mean() if len(we) else np.nan
            rows.append(dict(zip(["연도"] + keys, [yr] + k), **{
                "품목": c,
                "정상평일수": len(wd),
                "정상평일_평균": round(m, 2) if pd.notna(m) else np.nan,
                "정상평일_상위%d%%" % round((1 - PEAK_QUANTILE) * 100): q,
                "정상평일_최대": mx,
                "피크계수": round(q / m, 3) if m and m > 0 else np.nan,
                "주말평균/평일평균": round(wem / m, 3) if m and m > 0 and pd.notna(wem) else np.nan,
            }))
    out = pd.DataFrame(rows)
    return out


# =====================================================================
# 5. OD 월별 (T6)
# =====================================================================
def region_of(cd):
    cd = int(cd)
    if cd == SEOUL_CD:
        return "서울"
    if cd in METRO_CDS:
        return "수도권(인천·경기)"
    return "지방"


def load_od(files, quality):
    send_parts, rec_parts, intra_parts = [], [], []
    for path, enc, cols in files:
        tops, _ = value_columns(cols)
        idc = ["dl_ym", "send_ctpv_cd", "send_ctgg_cd", "send_ctgg_nm", "send_hjdong_cd",
               "send_hjdong_nm", "rec_ctpv_cd", "rec_ctgg_cd", "rec_ctgg_nm", "rec_hjdong_cd",
               "rec_hjdong_nm"]
        idc = [c for c in idc if c in cols]
        n_rows = 0
        direction = None
        for ch in pd.read_csv(path, encoding=enc, usecols=idc + tops, chunksize=CHUNKSIZE,
                              low_memory=False):
            n_rows += len(ch)
            vals = ch[tops].apply(pd.to_numeric, errors="coerce").fillna(0)
            v = pd.DataFrame({"c_" + cat_code(t): vals[t].astype("int64") for t in tops})
            v["합계"] = vals.sum(axis=1).astype("int64")
            base = pd.concat([ch[idc].reset_index(drop=True), v.reset_index(drop=True)], axis=1)
            base["년월"] = base["dl_ym"].astype(str).str[:6]
            all_rec_seoul = (base["rec_ctpv_cd"] == SEOUL_CD).all()
            all_send_seoul = (base["send_ctpv_cd"] == SEOUL_CD).all()
            if all_send_seoul and not all_rec_seoul:
                direction = "서울→전국"
            elif all_rec_seoul and not all_send_seoul:
                direction = "전국→서울"
            elif all_rec_seoul and all_send_seoul:
                direction = direction or "서울→서울"
            vcols = list(v.columns)
            s = base[base["send_ctpv_cd"] == SEOUL_CD].copy()
            if len(s) and direction in ("서울→전국", "서울→서울"):
                s["목적지권역"] = s["rec_ctpv_cd"].map(region_of)
                send_parts.append(s.groupby(["년월", "send_ctgg_cd", "send_ctgg_nm",
                                             "send_hjdong_cd", "send_hjdong_nm", "목적지권역"]
                                            )[vcols].sum().reset_index())
            r = base[base["rec_ctpv_cd"] == SEOUL_CD].copy()
            if len(r) and direction in ("전국→서울", "서울→서울"):
                r["출발권역"] = r["send_ctpv_cd"].map(region_of)
                rec_parts.append(r.groupby(["년월", "rec_ctgg_cd", "rec_ctgg_nm",
                                            "rec_hjdong_cd", "rec_hjdong_nm", "출발권역"]
                                           )[vcols].sum().reset_index())
            # 서울 안 이동은 한 방향 파일에서만 집계 (두 파일에 같은 행이 있을 수 있으므로)
            if direction in ("전국→서울", "서울→서울"):
                ii = base[(base["send_ctpv_cd"] == SEOUL_CD) & (base["rec_ctpv_cd"] == SEOUL_CD)]
                if len(ii):
                    intra_parts.append(ii.groupby(["년월", "send_ctgg_cd", "send_ctgg_nm",
                                                   "rec_ctgg_cd", "rec_ctgg_nm"]
                                                  )[vcols].sum().reset_index())
        quality.append({"파일": os.path.basename(path), "종류": "OD(%s)" % direction,
                        "행수": n_rows, "0 비율": "-", "대분류≠세부합 행수": "-"})

    def combine(parts, keys):
        if not parts:
            return None
        df = pd.concat(parts, ignore_index=True)
        vcols = [c for c in df.columns if c.startswith("c_")] + ["합계"]
        df = df.groupby(keys)[vcols].sum().reset_index()
        return mask_small(df, vcols)

    t6a = combine(send_parts, ["년월", "send_ctgg_cd", "send_ctgg_nm", "send_hjdong_cd",
                               "send_hjdong_nm", "목적지권역"])
    t6b = combine(rec_parts, ["년월", "rec_ctgg_cd", "rec_ctgg_nm", "rec_hjdong_cd",
                              "rec_hjdong_nm", "출발권역"])
    t6c = combine(intra_parts, ["년월", "send_ctgg_cd", "send_ctgg_nm", "rec_ctgg_cd",
                                "rec_ctgg_nm"])
    return t6a, t6b, t6c


# =====================================================================
# 6. 품질: 행정동 코드 이력, 코드표 템플릿
# =====================================================================
def dong_history(daily, kind):
    g = daily.groupby(["ctgg_cd", "ctgg_nm", "hjdong_cd", "hjdong_nm"])["date"]
    h = g.agg(첫등장="min", 마지막등장="max", 일수="nunique").reset_index()
    h["데이터"] = kind
    h["첫등장"] = h["첫등장"].dt.strftime("%Y-%m-%d")
    h["마지막등장"] = h["마지막등장"].dt.strftime("%Y-%m-%d")
    return h


# =====================================================================
# main
# =====================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*.csv"), recursive=True))
    log("입력 파일 %d개 (%s)" % (len(paths), os.path.abspath(INPUT_DIR)))
    groups = {"daily_send": [], "daily_rec": [], "od": []}
    all_codes = set()
    for p in paths:
        try:
            enc = detect_encoding(p)
            cols = read_columns(p, enc)
        except Exception as e:
            log("  [건너뜀] %s: %s" % (os.path.basename(p), e))
            continue
        kind = classify(cols)
        if kind is None:
            log("  [건너뜀] %s: 형식을 알 수 없음" % os.path.basename(p))
            continue
        groups[kind].append((p, enc, cols))
        for c in cols:
            if TOP_RE.match(c) or SUB_RE.match(c):
                all_codes.add(cat_code(c))
        log("  %-12s %s (%s)" % (kind, os.path.basename(p), enc))

    quality = []
    rec = load_daily(groups["daily_rec"], "착지(일별)", quality) if groups["daily_rec"] else None
    snd = load_daily(groups["daily_send"], "출지(일별)", quality) if groups["daily_send"] else None

    base = rec if rec is not None else snd
    hist = []
    if base is not None:
        log("\n[달력] 정상평일 판정")
        cal = build_calendar(base)
        cal_out = cal.reset_index().rename(columns={"index": "날짜", "date": "날짜"})
        cal_out["날짜"] = pd.to_datetime(cal_out["날짜"]).dt.strftime("%Y-%m-%d")
        save(cal_out, "Q0_달력_정상평일판정.csv")

        for df, kind, mname in [(rec, "착지", "T1_착지_월별_행정동.csv"),
                                (snd, "출지", "T5_출지_월별_행정동.csv")]:
            if df is None:
                continue
            log("\n[%s] 월별·피크 집계" % kind)
            cats = sorted([c for c in df.columns if re.match(r"^c_\d{2}$", c)])
            grp = [g for g in (["신선"] + list(CUSTOM_GROUPS)) if g in df.columns]
            save(monthly_table(df, cal, cats + ["합계"]), mname)
            if kind == "착지" and grp:
                save(monthly_table(df, cal, grp), "T2_신선_월별_행정동.csv")
            elif kind == "착지":
                log("  FRESH_SUBCODES 가 비어 있어 T2(신선) 는 건너뜀")
            pk = cats + ["합계"] + grp
            suffix = "" if kind == "착지" else "_출지"
            save(peak_stats(df, cal, pk, "행정동"), "T3_피크통계_행정동%s.csv" % suffix)
            save(peak_stats(df, cal, pk, "자치구"), "T3_피크통계_자치구%s.csv" % suffix)
            hist.append(dong_history(df, kind))
            if kind == "착지":
                t7 = df.groupby("date")[cats + ["합계"] + grp].sum().reset_index()
                t7["date"] = t7["date"].dt.strftime("%Y-%m-%d")
                save(t7.rename(columns={"date": "날짜"}), "T7_서울전체_일별_품목.csv")

    if groups["od"]:
        log("\n[OD] 월별 권역 집계")
        t6a, t6b, t6c = load_od(groups["od"], quality)
        if t6a is not None:
            save(t6a, "T6a_OD_서울에서_전국_월별.csv")
        if t6b is not None:
            save(t6b, "T6b_OD_전국에서_서울_월별.csv")
        if t6c is not None:
            save(t6c, "T6c_OD_서울내_구간_월별.csv")

    if hist:
        save(pd.concat(hist, ignore_index=True), "Q2_행정동코드_이력.csv")
    codes = pd.DataFrame({"코드": sorted(all_codes)})
    codes["구분"] = np.where(codes["코드"].str.len() == 2, "대분류", "세부분류")
    codes["품목명(정의서 보고 입력)"] = ""
    save(codes, "Q3_코드표_작성용.csv")

    with open(os.path.join(OUTPUT_DIR, "Q1_품질리포트.txt"), "w", encoding="utf-8") as f:
        f.write("생성: %s\n\n" % dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
        f.write("[설정] FRESH_SUBCODES=%s, PEAK_QUANTILE=%s, ABNORMAL_RATIO=%s, MASK_BELOW=%s\n\n"
                % (FRESH_SUBCODES, PEAK_QUANTILE, ABNORMAL_RATIO, MASK_BELOW))
        for q in quality:
            f.write("%s\n" % q)
        for d, kind in [(rec, "착지"), (snd, "출지")]:
            if d is not None:
                f.write("\n%s: 기간 %s ~ %s, 날짜 %d일, 행정동 %d개, 자치구 %d개\n" % (
                    kind, d["date"].min().date(), d["date"].max().date(), d["date"].nunique(),
                    d["hjdong_cd"].nunique(), d["ctgg_cd"].nunique()))
        f.write("\n[실행 로그]\n" + "\n".join(LOG))
    log("\n완료 → %s" % os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        INPUT_DIR = sys.argv[1]
    if len(sys.argv) >= 3:
        OUTPUT_DIR = sys.argv[2]
    main()
