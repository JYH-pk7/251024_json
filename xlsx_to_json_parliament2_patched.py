#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xlsx_to_json_parliament2.py
---------------------------
Improved extraction for Meetings: derives `number_of_meetings` (회수) and `chasu` (차수)
by scanning multiple columns (회의록구분/회의구분/위원회/기타 정보/안건 등) with regex
when explicit columns ('회수', '차수') are empty or missing.

Usage:
  pip install pandas openpyxl
  python xlsx_to_json_parliament2.py --excel "./제21대 국회 소위원회 법제사법위원회 회의록 데이터셋.xlsx" --outdir "./out"

Outputs:
  <base>_speeches.json
  <base>_meetings.json
"""

# === Bills filter: keep only lines that contain a valid bill number (의안번호 ####) ===
import re as _re_bills_filter

_RE_BILLNO = _re_bills_filter.compile(r"의\s*안\s*번\s*호\s*[:\s\-]*\d{4,}", _re_bills_filter.IGNORECASE)

def _bf_has_bill_number(text: str) -> bool:
    if not text:
        return False
    return bool(_RE_BILLNO.search(str(text)))

def _bf_filter_bills_lines(bills: str):
    """
    - If bills has multiple lines, keep only lines that contain a bill number (의안번호 ####).
    - If none remain, return None to avoid recording bills.
    """
    if not bills:
        return None
    s = str(bills).replace("\\r", "\\n")
    lines = [ln.strip() for ln in s.split("\\n") if ln.strip()]
    kept = [ln for ln in lines if _bf_has_bill_number(ln)]
    return "\\n".join(kept) if kept else None


import argparse
import hashlib
import json
import math
import os
import re
from typing import Optional, List, Dict, Any

import pandas as pd


# ---------- utils ----------
def _coerce_int(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isnan(x):
            return None
        return int(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(s))
    except Exception:
        # Try pure digits extraction
        m = re.search(r"\d+", s)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass
        return None


def _coerce_str(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return str(x)


def _mk_speech_id(meeting_no, order_no, member_id, row_index: int) -> int:
    key_parts = [meeting_no, order_no, member_id]
    if any(p is not None and str(p) != "" for p in key_parts):
        base = f"{meeting_no}|{order_no}|{member_id}"
    else:
        base = f"row|{row_index}"
    h = hashlib.sha1(str(base).encode("utf-8")).hexdigest()
    val = int(h[:16], 16) & ((1 << 63) - 1)
    return val


def _non_empty_first(series: pd.Series):
    for v in series:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        s = str(v).strip()
        if s != "" and s.lower() not in {"nan", "none", "null"}:
            return v
    return None


def _safe_join_lines(values: List[Any]) -> Optional[str]:
    parts: List[str] = []
    for v in values:
        s = _coerce_str(v)
        if s and s.strip():
            parts.append(s.strip())
    return "\n".join(parts) if parts else None


# ---------- speeches ----------
def build_speeches(df: pd.DataFrame) -> List[Dict[str, Any]]:
    required = ["회의번호", "발언자", "발언순번"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"엑셀에 필수 컬럼이 없습니다: {required} / 현재: {list(df.columns)}")

    speech_cols = [f"발언내용{i}" for i in range(1, 8) if f"발언내용{i}" in df.columns]

    speeches: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        meeting_no = row.get("회의번호")
        member_name = row.get("발언자")
        order_no = row.get("발언순번")
        member_id = row.get("의원ID", None)
        bills = row.get("안건", None)

        speech_text = _safe_join_lines([row.get(c) for c in speech_cols]) if speech_cols else None

        speeches.append({
            "speech_id": _mk_speech_id(meeting_no, order_no, member_id, idx),
            "meeting_id": _coerce_int(meeting_no),
            "bills": _coerce_str(bills),
            "member_id": _coerce_str(member_id),
            "member_name": _coerce_str(member_name),
            "speech_order": _coerce_int(order_no),
            "speech_text": speech_text,
        })
    return speeches


# ---------- meetings ----------
RE_MEETING = re.compile(r"제\s*(\d{1,5})\s*회")   # e.g., 제403회
RE_CHASU   = re.compile(r"제\s*(\d{1,5})\s*차")   # e.g., 제1차

def _derive_meeting_numbers_from_texts(texts: List[str]) -> Dict[str, Optional[int]]:
    number_of_meetings = None
    chasu = None
    for t in texts:
        if not t:
            continue
        m1 = RE_MEETING.search(t)
        if m1 and number_of_meetings is None:
            try:
                number_of_meetings = int(m1.group(1))
            except Exception:
                pass
        m2 = RE_CHASU.search(t)
        if m2 and chasu is None:
            try:
                chasu = int(m2.group(1))
            except Exception:
                pass
        if number_of_meetings is not None and chasu is not None:
            break
    return {"number_of_meetings": number_of_meetings, "chasu": chasu}


def build_meetings(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols_map = {
        "meeting_category": "회의록구분",
        "deasu": "대수",
        "meeting_specification": "회의구분",
        "commitee": "위원회",
        "number_of_meetings": "회수",
        "chasu": "차수",
        "other_info": "기타정보",
        "meeting_date": "회의일자",
    }
    present = {k: v for k, v in cols_map.items() if v in df.columns}

    meetings: List[Dict[str, Any]] = []

    if "회의번호" in df.columns:
        grouped = df.groupby("회의번호", dropna=False)

        # Columns to scan with regex if explicit fields are empty/missing
        scan_cols = [c for c in ["회의록구분", "회의구분", "위원회", "기타정보", "안건"] if c in df.columns]

        for _, g in grouped:
            rec: Dict[str, Any] = {}

            # Fill straightforward fields from first non-empty
            for out_key, in_col in present.items():
                first_val = _non_empty_first(g[in_col]) if in_col in g.columns else None
                if out_key in {"number_of_meetings", "chasu"}:
                    rec[out_key] = _coerce_int(first_val)
                else:
                    rec[out_key] = _coerce_str(first_val)

            # If number_of_meetings or chasu still None, derive via regex from multiple text columns
            if rec.get("number_of_meetings") is None or rec.get("chasu") is None:
                concat_texts: List[str] = []
                for c in scan_cols:
                    # Join unique non-empty strings from this column in the group
                    vals = [str(v) for v in g[c].dropna().astype(str).unique() if str(v).strip()]
                    concat_texts.extend(vals)
                derived = _derive_meeting_numbers_from_texts(concat_texts)
                if rec.get("number_of_meetings") is None and derived["number_of_meetings"] is not None:
                    rec["number_of_meetings"] = derived["number_of_meetings"]
                if rec.get("chasu") is None and derived["chasu"] is not None:
                    rec["chasu"] = derived["chasu"]

            meetings.append(rec)
    else:
        # Fallback: no meeting key, produce per-row (best-effort)
        for _, row in df.iterrows():
            rec: Dict[str, Any] = {}
            for out_key, in_col in present.items():
                val = row.get(in_col)
                if out_key in {"number_of_meetings", "chasu"}:
                    rec[out_key] = _coerce_int(val)
                else:
                    rec[out_key] = _coerce_str(val)
            # Try derive from text columns
            scan_texts = [str(row.get(c)) for c in ["회의록구분", "회의구분", "위원회", "기타 정보", "안건"] if c in df.columns]
            derived = _derive_meeting_numbers_from_texts([t for t in scan_texts if t and t != "None"])
            if rec.get("number_of_meetings") is None and derived["number_of_meetings"] is not None:
                rec["number_of_meetings"] = derived["number_of_meetings"]
            if rec.get("chasu") is None and derived["chasu"] is not None:
                rec["chasu"] = derived["chasu"]
            meetings.append(rec)

    # ---------- meeting_id 생성 ----------
    for rec in meetings:
        # 회의번호가 있으면 그걸 그대로 사용
        base_no = rec.get("number_of_meetings")
        if base_no is not None:
            rec["meeting_id"] = int(base_no)
        else:
            import hashlib
            seed = json.dumps(rec, ensure_ascii=False, sort_keys=True)
            rec["meeting_id"] = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16)
    return meetings


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="회의록 엑셀을 speeches/meetings JSON으로 변환 (regex 보강판)")
    parser.add_argument("--excel", required=True, help="입력 엑셀(.xlsx) 경로")
    parser.add_argument("--outdir", default=".", help="출력 디렉토리 (기본: 현재 폴더)")
    args = parser.parse_args()

    if not os.path.exists(args.excel):
        raise FileNotFoundError(f"엑셀 파일이 없습니다: {args.excel}")

    df = pd.read_excel(args.excel, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    speeches = build_speeches(df)
    meetings = build_meetings(df)

    base = os.path.splitext(os.path.basename(args.excel))[0]
    out_speeches = os.path.join(args.outdir, f"{base}_speeches_patched.json")
    out_meetings_patched = os.path.join(args.outdir, f"{base}_meetings_patched.json")

    
    # __BILLS_FILTER_PRE_SAVE__: sanitize `bills` based on presence of a bill number
    if isinstance(speeches, list):
        changed_cnt = 0
        for __rec in speeches:
            if isinstance(__rec, dict) and "bills" in __rec:
                __new = _bf_filter_bills_lines(__rec.get("bills"))
                if __new != __rec.get("bills"):
                    __rec["bills"] = __new
                    changed_cnt += 1
        print(f"[Bills filter] Cleaned/updated {changed_cnt} records (removed bills without 의안번호).")

    os.makedirs(args.outdir, exist_ok=True)
    with open(out_speeches, "w", encoding="utf-8") as f:
        json.dump(speeches, f, ensure_ascii=False, indent=2)
    with open(out_meetings_patched, "w", encoding="utf-8") as f:
        json.dump(meetings, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_speeches}")
    print(f"Saved: {out_meetings_patched}")


if __name__ == "__main__":
    main()
