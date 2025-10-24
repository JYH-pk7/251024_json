#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xlsx_to_json_parliament.py  (v2: robust meetings aggregation)
----------------------------------------------------------------
Fix for missing `number_of_meetings` and `chasu` in meetings JSON by grouping on meeting id
and taking the first non-empty value per field.
"""
import argparse
import hashlib
import json
import math
import os
from typing import Optional, List, Dict, Any

import pandas as pd


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


def build_meetings(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols_map = {
        "meeting_category": "회의록구분",
        "deasu": "대수",
        "meeting_specification": "회의구분",
        "commitee": "위원회",
        "number_of_meetings": "회수",
        "chasu": "차수",
        "other_info": "기타 정보",
        "meeting_date": "회의일자",
    }
    present = {k: v for k, v in cols_map.items() if v in df.columns}

    meetings: List[Dict[str, Any]] = []
    if "회의번호" in df.columns:
        grouped = df.groupby("회의번호", dropna=False)
        for _, g in grouped:
            rec: Dict[str, Any] = {}
            for out_key, in_col in present.items():
                first_val = _non_empty_first(g[in_col])
                if out_key in {"number_of_meetings", "chasu"}:
                    rec[out_key] = _coerce_int(first_val)
                else:
                    rec[out_key] = _coerce_str(first_val)
            meetings.append(rec)
    else:
        for _, row in df.iterrows():
            rec: Dict[str, Any] = {}
            for out_key, in_col in present.items():
                val = row.get(in_col)
                if out_key in {"number_of_meetings", "chasu"}:
                    rec[out_key] = _coerce_int(val)
                else:
                    rec[out_key] = _coerce_str(val)
            meetings.append(rec)

    return meetings


def main():
    parser = argparse.ArgumentParser(description="회의록 엑셀을 speeches/meetings JSON으로 변환 (v2)")
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
    out_speeches = os.path.join(args.outdir, f"{base}_speeches.json")
    out_meetings = os.path.join(args.outdir, f"{base}_meetings.json")

    os.makedirs(args.outdir, exist_ok=True)
    with open(out_speeches, "w", encoding="utf-8") as f:
        json.dump(speeches, f, ensure_ascii=False, indent=2)
    with open(out_meetings, "w", encoding="utf-8") as f:
        json.dump(meetings, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_speeches}")
    print(f"Saved: {out_meetings}")


if __name__ == "__main__":
    main()
