# -*- coding: utf-8 -*-
"""00. data/raw 의 압축 파일을 스크립트가 읽는 폴더로 푼다.

  seoul_lev6_2025.zip         → data/raw/lev6/   (View-T 상세도로망 Level6)
  LSMD_ADM_SECT_UMD_서울.zip  → data/raw/umd/    (브이월드 법정동 경계)
  공시지가_2026년.zip          → data/raw/gsj/    (서울시 개별공시지가)
LOCAL_PEOPLE_DONG_202607.zip 은 압축을 풀지 않고 05a 가 바로 읽는다.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RAW  # noqa: E402
import os, zipfile

JOBS = {"seoul_lev6_2025.zip": "lev6", "LSMD_ADM_SECT_UMD_서울.zip": "umd", "공시지가_2026년.zip": "gsj"}

for z, sub in JOBS.items():
    src = os.path.join(RAW, z)
    if not os.path.exists(src):
        print("없음(건너뜀):", src); continue
    dst = os.path.join(RAW, sub); os.makedirs(dst, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        for info in zf.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:          # UTF-8 표시가 없는 이름은 cp949 로 복원
                try:
                    name = name.encode("cp437").decode("cp949")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            if name.endswith("/"):
                continue
            with zf.open(info) as f, open(os.path.join(dst, os.path.basename(name)), "wb") as g:
                g.write(f.read())
    print("풀기 완료:", z, "→", dst, sorted(os.listdir(dst))[:6])
