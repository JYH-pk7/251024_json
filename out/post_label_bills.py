#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_label_bills.py

사용법:
  python post_label_bills.py speeches_meeting_50848.json -o out_dir

출력:
  out_dir/bills_summary.json
  out_dir/speeches_meeting_50848_labeled.json
"""

import argparse, json, os, re, unicodedata
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

# -------- 공통 유틸 --------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return s

def clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# 회의 안건 문자열에서 번호/괄호/부가설명 제거
RE_PREFIX_NUM = re.compile(r"^\s*\d+\s*[.\)]\s*")
RE_PARENS = re.compile(r"\([^()]*\)")
RE_DANGLING_MARKS = re.compile(r"[·•\-–—]+")
STOP_PHRASES = [
    "일부개정법률안","법률안","대안","정부 제출","의원 대표발의","대표발의",
    "위원장 제출","의안번호","정부제출","제출","소위자료","안건"
]

def normalize_bill_title(raw: str) -> str:
    if not raw: 
        return ""
    s = normalize_text(raw)
    s = s.replace("\n", " ")
    s = RE_PREFIX_NUM.sub("", s)         # "1. ..." 제거
    s = RE_PARENS.sub("", s)             # 괄호내용 제거
    s = RE_DANGLING_MARKS.sub(" ", s)    # 중간점류 정리
    s = clean_spaces(s)
    # 불용구 제거
    for sp in STOP_PHRASES:
        s = s.replace(sp, "")
    s = clean_spaces(s)
    return s

# bill 키워드 추출: 한글/영문/숫자 단어 + 관용 불용어 제거
BILL_STOP_TOKENS = {
    "일부개정","개정","법률","법","대책","에","관한","및","의","등","관련","사항",
    "제정","일괄","추진","보고","자료","소위","위원회","위원","정부","제출","위원장",
    "대표발의","대안","의안","번호","검토","안","개요","기타"
}

def extract_keywords_from_bill(title: str, topk: int = 6) -> List[str]:
    t = normalize_text(title)
    # 한글/영문/숫자 토큰
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", t)
    kept = []
    for tok in tokens:
        if len(tok) <= 1: 
            continue
        if tok in BILL_STOP_TOKENS:
            continue
        kept.append(tok)
    # 빈도 상위
    cnt = Counter(kept)
    return [w for w,_ in cnt.most_common(topk)]

# 스코어링을 위한 정규식(공백/개행/중간점 허용)
def spaced_pat(word: str) -> re.Pattern:
    chars = list(word)
    pat = r"\s*".join(map(re.escape, chars))
    return re.compile(pat, re.IGNORECASE|re.DOTALL)

def build_keyword_patterns(words: List[str]) -> List[re.Pattern]:
    return [spaced_pat(w) for w in words if w]

# bills 필드 자체에도 여러 줄/여러 안건이 섞여있을 수 있음
def split_raw_bills_field(bills: str) -> List[str]:
    if not bills: 
        return []
    s = normalize_text(bills)
    parts = re.split(r"[\n;]+", s)
    parts = [clean_spaces(p) for p in parts if clean_spaces(p)]
    return parts

# -------- 핵심 로직 --------
def collect_bills_and_keywords(speeches: List[Dict[str, Any]]) -> Tuple[List[Dict[str,Any]], Dict[str,Dict[str,Any]]]:
    """
    returns:
      unique_bills: [{raw:..., canonical:..., keywords:[...]}, ...]
      bill_map: {canonical: {raw_variants:set(), keywords:[], patterns:[...]}}
    """
    bill_variants = []
    for rec in speeches:
        for piece in split_raw_bills_field(rec.get("bills")):
            bill_variants.append(piece)

    canonical_set = {}
    for raw in bill_variants:
        canon = normalize_bill_title(raw)
        if not canon:
            continue
        if canon not in canonical_set:
            canonical_set[canon] = {"raw_variants": set()}
        canonical_set[canon]["raw_variants"].add(raw)

    unique_bills = []
    bill_map = {}
    for canon,info in canonical_set.items():
        kws = extract_keywords_from_bill(canon, topk=8)
        pats = build_keyword_patterns([canon] + kws)  # 제목 전체 + 키워드
        unique_bills.append({
            "canonical": canon,
            "raw_variants": sorted(list(info["raw_variants"])),
            "keywords": kws
        })
        bill_map[canon] = {"raw_variants": info["raw_variants"], "keywords": kws, "patterns": pats}
    return unique_bills, bill_map

def score_text_for_bill(text: str, bill_entry: Dict[str,Any]) -> Tuple[int,int]:
    """
    returns: (hard_hits, soft_hits)
      hard_hits: canonical 전체 패턴 일치 수
      soft_hits: 키워드 패턴 일치 수
    """
    if not text:
        return (0,0)
    T = normalize_text(text)
    pats = bill_entry["patterns"]
    # 첫 패턴은 canonical 전체, 나머지는 키워드라고 가정
    hard = 0
    soft = 0
    for i,pat in enumerate(pats):
        matches = list(pat.finditer(T))
        if not matches:
            continue
        if i == 0:
            hard += len(matches)
        else:
            soft += len(matches)
    return (hard, soft)

def assign_bills_to_speeches(speeches: List[Dict[str,Any]], bill_map: Dict[str,Dict[str,Any]]) -> List[Dict[str,Any]]:
    canon_list = list(bill_map.keys())

    def choose_by_score(text: str) -> Tuple[str,float,str]:
        best = None
        best_score = (-1,-1)
        for c in canon_list:
            h,s = score_text_for_bill(text, bill_map[c])
            if (h,s) > best_score:
                best_score = (h,s); best = c
        if best is None:
            return None, 0.0, "no-match"
        h,s = best_score
        conf = 0.9 if h>0 else (0.6 if s>=2 else (0.4 if s==1 else 0.0))
        reason = f"hard={h}, soft={s}"
        return best, conf, reason

    labeled = []
    last_agenda_canon = None   # 직전 안건 흐름 승계
    last_by_speaker = {}

    for rec in speeches:
        text = rec.get("speech_text") or ""
        raw_bills = split_raw_bills_field(rec.get("bills") or "")
        # 1) bills 필드에 명시된 후보를 정규화
        candidates = [normalize_bill_title(b) for b in raw_bills if normalize_bill_title(b)]
        candidates = [c for c in candidates if c in bill_map]

        picked = None; conf=0.0; why=""

        # 2) 텍스트 스코어링 우선
        t_pick, t_conf, t_reason = choose_by_score(text)
        if t_pick:
            picked, conf, why = t_pick, t_conf, f"text-match({t_reason})"

        # 3) 후보가 있고 텍스트 근거가 약하면 후보와 교차검증
        if candidates and (conf < 0.6):
            # 후보 중 텍스트 스코어가 가장 높은 것 선택
            best_cand = None; best_cscore=(-1,-1)
            for c in candidates:
                h,s = score_text_for_bill(text, bill_map[c])
                if (h,s) > best_cscore:
                    best_cscore=(h,s); best_cand=c
            if best_cand:
                picked = best_cand
                conf = max(conf, 0.6 if best_cscore==(0,0) else (0.9 if best_cscore[0]>0 else 0.7))
                why = f"bills-field-candidate + text({best_cscore[0]},{best_cscore[1]})"

        # 4) 여전히 미정이면 직전 안건 흐름 승계
        if not picked and last_agenda_canon:
            picked = last_agenda_canon; conf = 0.5; why="agenda-flow"

        # 5) 그래도 없으면 같은 화자의 직전 값
        spk = rec.get("member_name")
        if not picked and spk in last_by_speaker:
            picked = last_by_speaker[spk]; conf = 0.45; why="speaker-history"

        # 6) 최종 확정 및 상태 갱신
        if picked:
            last_agenda_canon = picked
            last_by_speaker[spk] = picked

        newrec = dict(rec)
        newrec["bill_assigned"] = picked
        newrec["bill_confidence"] = round(conf,2)
        newrec["bill_reason"] = why
        labeled.append(newrec)

    return labeled

# -------- 메인 --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", help="speeches_meeting_XXXX.json")
    ap.add_argument("-o","--outdir", default=".", help="출력 디렉토리")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        speeches = json.load(f)
    # 예: 이 파일은 meeting 50848이고, 같은 파일 안에서 여러 법률안이 등장합니다
    # - 자연재해대책법 일부개정법률안 상정/토론 구간 :contentReference[oaicite:2]{index=2}
    # - 고향사랑/기부금품 법안 일괄 심사 구간 :contentReference[oaicite:3]{index=3}
    # - 경찰공무원법 일부개정법률안 구간 :contentReference[oaicite:4]{index=4}

    # 1) bills 수집 및 키워드 생성
    unique_bills, bill_map = collect_bills_and_keywords(speeches)

    os.makedirs(args.outdir, exist_ok=True)
    bills_out = os.path.join(args.outdir, "bills_summary.json")
    with open(bills_out, "w", encoding="utf-8") as f:
        json.dump(unique_bills, f, ensure_ascii=False, indent=2)

    # 2) 발언별 bill 할당
    labeled = assign_bills_to_speeches(speeches, bill_map)
    labeled_out = os.path.join(args.outdir, os.path.basename(args.input_json).replace(".json","_labeled.json"))
    with open(labeled_out, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)

    # 간단 리포트
    cnt = Counter([rec.get("bill_assigned") for rec in labeled if rec.get("bill_assigned")])
    print("[bill 할당 분포]")
    for k,v in cnt.most_common():
        print(f"  {k}: {v}")
    print(f"\n저장: {bills_out}")
    print(f"저장: {labeled_out}")

if __name__ == "__main__":
    main()
