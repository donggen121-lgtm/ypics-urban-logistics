# -*- coding: utf-8 -*-
"""경로 설정. 모든 스크립트는 이 파일 기준 상대 경로를 쓴다.

data/raw/        원본 데이터 (저장소에 올리지 않음, docs/데이터출처.md 대로 받아서 넣기)
data/processed/  전처리 결과 (저장소에 포함 → 원본 없이도 09~13 최적화 실행 가능)
results/         최적화 결과·그림
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")


def proc(*parts):
    """data/processed 아래 경로 (폴더가 없으면 만든다)."""
    p = os.path.join(PROC, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def res(*parts):
    """results 아래 경로 (폴더가 없으면 만든다)."""
    p = os.path.join(RES, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p
