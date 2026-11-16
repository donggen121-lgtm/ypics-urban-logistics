# -*- coding: utf-8 -*-
"""10. 신규센터 수 k별 최소 연간비용(EV) → U자 곡선 + 민감도(벌점·고정비)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW, PROC, RES, proc, res  # noqa: E402
import os, subprocess, sys, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
BASE = PROC
def ev(*args):
    o = subprocess.run([PY, os.path.join(HERE, "09_optimize.py"), "--only", "ev", "--gap", "0.002", *args],
                       capture_output=True, text=True).stdout
    l = [x for x in o.splitlines() if x.startswith("EV_ONLY")][0].split()
    return float(l[1]), int(l[2]), float(l[3])
rows = []
for k in [10, 20, 40, 70, 100, 140, 187, 240, 300]:
    c, n, r = ev("--nnew", str(k)); rows.append({"분석": "U곡선", "신규센터수": n, "연간비용_억": c, "서울처리비율": r})
for pm in [30, 45, 60]:
    for fx in [0, 1, 2]:
        c, n, r = ev("--penalty_min", str(pm), "--fixed", str(fx))
        rows.append({"분석": "민감도", "외곽편도_분": pm, "고정운영비_억": fx, "신규센터수": n, "연간비용_억": c, "서울처리비율": r})
D = pd.DataFrame(rows); D.to_csv(res("exploratory", "ucurve_sensitivity.csv"), index=False, encoding="utf-8-sig")
print(D.to_string(index=False))
