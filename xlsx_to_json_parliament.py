#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xlsx_to_json_parliament.py
--------------------------
엑셀(회의록 데이터셋)을 읽어 아래 두 JSON을 생성합니다.
  1) speeches JSON (발언 단위)
  2) meetings JSON (회의 메타데이터)

입력 엑셀의 컬럼(예시):
- 회의번호, 회의록구분, 대수, 회의구분, 위원회, 회수, 차수, 기타 정보, 회의일자, 안건,
  발언자, 의원ID, 발언순번, 발언내용1~발언내용7

출력 스키마

A. speeches (발언 단위)
- speech_id (pk)                         ← sha1("회의번호|발언순번|의원ID") 기반 63bit 정수 (fallback: 행 인덱스)
- meeting_id (FK from meetings)          ← 엑셀 '회의번호'
- bills (nullable, 안건)                 ← 엑셀 '안건' (1차 저장만)
- member_id                              ← 엑셀 '의원ID' (없으면 None)
- member_name (발언자 이름)              ← 엑셀 '발언자'
- speech_order (회의에서 발언된 순서)    ← 엑셀 '발언순번'
- speech_text (발언 전문)                ← 엑셀 '발언내용1'~'발언내용7' 줄바꿈 연결 (빈 값/NaN 제외)

C. meetings
- meeting_category (회의록 구분)         ← 엑셀 '회의록구분'
- deasu (대수)                           ← 엑셀 '대수'
- meeting_specification (회의구분)       ← 엑셀 '회의구분'
- commitee (위원회)                      ← 엑셀 '위원회'     (원문 철자 유지)
- number_of_meetings (회수)              ← 엑셀 '회수'
- chasu (차수)                           ← 엑셀 '차수'
- other_info (기타 정보)                 ← 엑셀 '기타 정보'
- meeting_date (회의일자)                ← 엑셀 '회의일자'

출력 파일명:
- <입력파일베이스>_speeches.json
- <입력파일베이스>_meetings.json

사용법:
  python xlsx_to_json_parliament.py --excel "/path/to/회의록.xlsx" --outdir "./out"

필요 패키지:
  pip install pandas openpyxl
"""
import argparse
import hashlib
import json
import math
import os
from typing import Optional, List, Dict, Any

import pandas as pd


# ---------- 유틸 ----------
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
    s = str(x)
    return s


def _mk_speech_id(meeting_no, order_no, member_id, row_index: int) -> int:
    """
    sha1 기반 63-bit 정수 ID (재실행시 안정적으로 동일)
    입력 값이 부족하면 row_index 기반으로 대체
    """
    key_parts = [meeting_no, order_no, member_id]
    if any(p is not None and str(p) != "" for p in key_parts):
        base = f"{meeting_no}|{order_no}|{member_id}"
    else:
        base = f"row|{row_index}"
    h = hashlib.sha1(str(base).encode("utf-8")).hexdigest()
    val = int(h[:16], 16) & ((1 << 63) - 1)  # 상위 63비트만 사용
    return val


def _safe_join_lines(values: List[Any]) -> Optional[str]:
    parts: List[str] = []
    for v in values:
        s = _coerce_str(v)
        if s and s.strip():
            parts.append(s.strip())
    return "\n".join(parts) if parts else None


# ---------- 매핑/생성 ----------
def build_speeches(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # 한국어 컬럼명 존재 여부 확인
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

        speech = {
            "speech_id": _mk_speech_id(meeting_no, order_no, member_id, idx),
            "meeting_id": _coerce_int(meeting_no),
            "bills": _coerce_str(bills),
            "member_id": _coerce_str(member_id),
            "member_name": _coerce_str(member_name),
            "speech_order": _coerce_int(order_no),
            "speech_text": speech_text,
        }
        speeches.append(speech)
    return speeches


def build_meetings(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # meetings는 회의 행(회의번호 기준)으로 중복이 많을 수 있어, 회의번호 단위로 unique 처리
    cols_map = {
        "meeting_id": "회의번호",
        "meeting_category": "회의록구분",
        "deasu": "대수",
        "meeting_specification": "회의구분",
        "commitee": "위원회",
        "number_of_meetings": "회수",
        "chasu": "차수",
        "other_info": "기타 정보",
        "meeting_date": "회의일자",
    }
    # 존재하는 컬럼만 사용
    available = {k: v for k, v in cols_map.items() if v in df.columns}

    # 회의번호가 있으면 그 기준으로 unique, 없으면 전체에서 한 줄씩 추출
    meeting_key_col = "회의번호" if "회의번호" in df.columns else None

    if meeting_key_col:
        # 최신/첫번째 행 기준으로 하나만 선택 (여기서는 첫번째)
        unique_meetings = df.drop_duplicates(subset=[meeting_key_col], keep="first")
    else:
        unique_meetings = df  # fallback

    meetings: List[Dict[str, Any]] = []
    for _, row in unique_meetings.iterrows():
        rec = {}
        for out_key, in_col in available.items():
            val = row.get(in_col)
            # if out_key in {"number_of_meetings", "chasu"}:
            #     rec[out_key] = _coerce_int(val)
            # else:
            #     rec[out_key] = _coerce_str(val)
        meetings.append(rec)
    return meetings


# ---------- 메인 ----------
def main():
    parser = argparse.ArgumentParser(description="회의록 엑셀을 speeches/meetings JSON으로 변환")
    parser.add_argument("--excel", required=True, help="입력 엑셀(.xlsx) 경로")
    parser.add_argument("--outdir", default=".", help="출력 디렉토리 (기본: 현재 폴더)")
    args = parser.parse_args()

    if not os.path.exists(args.excel):
        raise FileNotFoundError(f"엑셀 파일이 없습니다: {args.excel}")

    # 엑셀 읽기
    df = pd.read_excel(args.excel, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]  # 헤더 공백 제거

    # 스피치/미팅 구조 생성
    speeches = build_speeches(df)
    meetings = build_meetings(df)

    # 출력 파일명 결정
    base = os.path.splitext(os.path.basename(args.excel))[0]
    out_speeches = os.path.join(args.outdir, f"{base}_speeches.json")
    out_meetings = os.path.join(args.outdir, f"{base}_meetings.json")

    os.makedirs(args.outdir, exist_ok=True)

    # JSON 저장
    with open(out_speeches, "w", encoding="utf-8") as f:
        json.dump(speeches, f, ensure_ascii=False, indent=2)
    with open(out_meetings, "w", encoding="utf-8") as f:
        json.dump(meetings, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_speeches}")
    print(f"Saved: {out_meetings}")


if __name__ == "__main__":
    main()
