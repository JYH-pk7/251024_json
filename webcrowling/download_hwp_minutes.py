#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_hwp_minutes.py
- 대상: https://record.assembly.go.kr/assembly/mnts/total/21.do
- 목표: 제21대, 상임위원회, '법제사법위원회', 검색구분 '소위원회' 결과에서 HWP 파일 일괄 다운로드
- 방법: Playwright(Chromium)로 목록 → 상세 진입 → .hwp 링크/버튼 클릭 시 download 이벤트로 저장
- 주의: 사이트 개편/라벨명이 조금씩 다를 수 있으므로, 실행 중 셀렉터가 안 맞으면 --no-headless --debug로 실제 라벨/버튼 텍스트 확인 후 아래 선택자 테이블만 손보세요.
"""

from __future__ import annotations
import re
import os
import time
import argparse
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout, Download

START_URL = "https://record.assembly.go.kr/assembly/mnts/total/21.do"

# 라벨/버튼/테이블에 대한 '유연한' 선택자 후보들
SELECTORS = {
    # 검색 필터 라벨 (사이트 구조에 따라 name이나 label 텍스트가 다를 수 있어서 후보를 둠)
    "assembly_label":    [r"제\d+대", r"대수", r"국회대수", r"대수 선택"],
    "type_label":        [r"구분", r"검색구분", r"회의구분"],   # 본회의/상임위원회/소위원회 등
    "committee_label":   [r"위원회명", r"위원회", r"상임위원회"],
    "search_button":     [r"검색", r"조회", r"찾기", r"Search", r"조회하기"],
    # 결과 테이블/행
    "table_rows":        ["table tbody tr", "table.list tbody tr", "tbody tr"],
    # 페이지네이션
    "next_button":       [r"다음", r"Next", r">", r"▶"],
    # 상세 페이지 내 한글/HWP 버튼 키워드
    "hwp_keywords":      [r"\.hwp\b", r"HWP", r"한글", r"한글파일", r"한글문서"]
}

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_by_label_loose(page, label_patterns: List[str]):
    """라벨 텍스트(정규식)로 input/select를 느슨하게 찾는다."""
    for patt in label_patterns:
        # 1) role 기반
        try:
            loc = page.get_by_label(re.compile(patt))
            if loc and loc.count() > 0:
                return loc
        except Exception:
            pass
        # 2) 라벨 텍스트 근처의 select/input을 CSS로 추정
        try:
            el = page.locator(f"text={patt}").first
            if el and el.count() > 0:
                # label의 형제/근처 select 또는 input 찾기
                neighbor = el.locator("xpath=following::*[self::select or self::input][1]")
                if neighbor and neighbor.count() > 0:
                    return neighbor
        except Exception:
            pass
    return None

def click_by_text_candidates(page, text_patterns: List[str]) -> bool:
    """버튼/링크 텍스트 후보 중 하나라도 클릭."""
    for patt in text_patterns:
        try:
            btn = page.get_by_role("button", name=re.compile(patt))
            if btn and btn.count() > 0:
                btn.first.click()
                return True
        except Exception:
            pass
        # a 태그도 시도
        try:
            a = page.get_by_role("link", name=re.compile(patt))
            if a and a.count() > 0:
                a.first.click()
                return True
        except Exception:
            pass
        # 일반 텍스트 일치도 시도
        try:
            page.get_by_text(re.compile(patt)).first.click()
            return True
        except Exception:
            pass
    return False

def fill_filters(page, debug=False):
    """
    제21대 / (상임위원회) / 위원회명=법제사법위원회 / 검색구분=소위원회
    형태로 필터를 채우고 검색.
    """
    # 대수
    assem = get_by_label_loose(page, SELECTORS["assembly_label"])
    if assem:
        try:
            # select인 경우
            assem.select_option(re.compile(r"21|제?21대"))
        except Exception:
            try:
                assem.fill("21")
            except Exception:
                if debug: print("[DEBUG] 대수 입력 실패 - 수동 선택 필요")
    # 구분(상임위원회)
    typ = get_by_label_loose(page, SELECTORS["type_label"])
    if typ:
        try:
            typ.select_option(re.compile(r"상임위원회"))
        except Exception:
            try:
                typ.fill("상임위원회")
            except Exception:
                if debug: print("[DEBUG] 구분 입력 실패")
    # 위원회명(법제사법위원회)
    comm = get_by_label_loose(page, SELECTORS["committee_label"])
    if comm:
        try:
            comm.select_option(re.compile(r"법제사법위원회"))
        except Exception:
            try:
                comm.fill("법제사법위원회")
            except Exception:
                if debug: print("[DEBUG] 위원회명 입력 실패")
    # 검색구분(소위원회)가 별도 라디오/체크박스/드롭다운일 수 있음 → 페이지 내 텍스트로 체크
    try:
        # 라디오/체크박스 형태?
        page.get_by_label(re.compile(r"소위원회")).check()
    except Exception:
        # 드롭다운 안에 '소위원회'가 있을 수 있음
        try:
            dd = page.get_by_role("combobox").filter(has_text=re.compile(r"소위원회"))
            if dd and dd.count() > 0:
                dd.first.click()
        except Exception:
            pass

    # 검색 버튼 클릭
    if not click_by_text_candidates(page, SELECTORS["search_button"]):
        if debug: print("[DEBUG] 검색 버튼 자동 클릭 실패 - 수동 확인 필요")

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4),
       retry=retry_if_exception_type(PWTimeout))
def collect_list_links(page, debug=False) -> List[str]:
    """결과 테이블에서 상세 링크들을 수집."""
    links = []
    for css in SELECTORS["table_rows"]:
        try:
            page.wait_for_selector(css, timeout=8000)
            rows = page.query_selector_all(css)
            if not rows:
                continue
            for tr in rows:
                a = tr.query_selector("a[href]")
                if a:
                    href = a.get_attribute("href") or ""
                    if href:
                        # 상대경로 보정
                        if href.startswith("/"):
                            href = page.url.split("/assembly")[0] + href
                        elif href.startswith("http"):
                            pass
                        else:
                            # 상대경로 (현재 URL 기준)
                            base = page.url.rsplit("/", 1)[0]
                            href = base + "/" + href
                        links.append(href)
            if links:
                break
        except PWTimeout:
            continue
    if debug: print(f"[DEBUG] 목록에서 상세 링크 {len(links)}건")
    return links

def find_hwp_links_in_detail(page) -> List[str]:
    """상세 페이지 내에서 HWP 관련 다운로드 링크 후보를 모두 찾는다."""
    cand = []
    anchors = page.query_selector_all("a[href]")
    for a in anchors:
        href = a.get_attribute("href") or ""
        text = (a.inner_text() or "").strip()
        blob = f"{href} {text}".upper()
        # 키워드/확장자 휴리스틱
        if any(re.search(p, blob, re.I) for p in SELECTORS["hwp_keywords"]):
            # 절대경로화
            if href.startswith("/"):
                href = page.url.split("/assembly")[0] + href
            elif not href.startswith("http"):
                base = page.url.rsplit("/", 1)[0]
                href = base + "/" + href
            cand.append(href)
    return list(dict.fromkeys(cand))  # 중복 제거, 순서 유지

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:200]

def derive_title_date(page) -> tuple[str, str]:
    """상세 상단에서 제목/일자 비슷한 문자열을 찾아 파일명에 보태기(실패해도 무관)."""
    title = ""
    date = ""
    try:
        # 페이지 타이틀/헤더 텍스트
        h = page.locator("h1, h2, .title, .subject").first
        if h and h.count() > 0:
            title = h.inner_text().strip()
    except Exception:
        pass
    try:
        body = page.inner_text("body")
        m = re.search(r"(\d{4}[.\-\/년 ]\s*\d{1,2}[.\-\/월 ]\s*\d{1,2}일?)", body)
        if m:
            date = m.group(1)
            date = re.sub(r"[^\d]", "", date)
            if len(date) >= 8:  # YYYYMMDD
                date = f"{date[:4]}{date[4:6]}{date[6:8]}"
    except Exception:
        pass
    return title, date

def save_with_playwright_download(page, href: str, out_dir: Path, prefix: str):
    """
    href가 단순 파일이면 request로 받는 것보다, Playwright의 download 이벤트로 받는 게
    세션/쿠키/POST 다운로드까지 안전하게 커버됨.
    """
    # 새 탭 이동/다운로드 모두 대응
    with page.expect_download(timeout=30000) as dl_info:
        page.evaluate("(url) => window.open(url, '_self')", href)
    dl: Download = dl_info.value
    # 파일명 결정
    suggested = dl.suggested_filename or os.path.basename(href.split("?")[0])
    filename = sanitize_filename(f"{prefix}_{suggested}")
    dl.save_as(str(out_dir / filename))
    return filename

def crawl(out_dir: Path, max_pages: int, detail_timeout: int, headless: bool, debug: bool):
    out_dir = ensure_dir(out_dir)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        ctx = browser.new_context(accept_downloads=True, locale="ko-KR")
        page = ctx.new_page()
        page.goto(START_URL, wait_until="domcontentloaded", timeout=45000)

        # 필터 채우고 검색
        fill_filters(page, debug=debug)
        # 네트워크 유휴까지 대기
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        page_no = 1
        total_downloads = 0

        while page_no <= max_pages:
            # 목록에서 상세 링크 수집
            links = collect_list_links(page, debug=debug)
            if not links and debug:
                print(f"[DEBUG] {page_no}페이지 상세 링크 없음")

            for detail_url in tqdm(links, desc=f"[{page_no}p] 상세 처리"):
                try:
                    sub = ctx.new_page()
                    sub.goto(detail_url, wait_until="domcontentloaded", timeout=45000)
                    try:
                        sub.wait_for_load_state("networkidle", timeout=detail_timeout * 1000)
                    except Exception:
                        pass

                    title, date = derive_title_date(sub)
                    prefix = sanitize_filename("_".join(x for x in [date, title] if x))

                    # 상세에서 HWP 후보 링크 찾기
                    hwp_links = find_hwp_links_in_detail(sub)

                    for href in hwp_links:
                        try:
                            fname = save_with_playwright_download(sub, href, out_dir, prefix or "document")
                            total_downloads += 1
                            if debug:
                                print(f"[OK] {fname}")
                            time.sleep(0.6)
                        except Exception as e:
                            if debug:
                                print("[WARN] 다운로드 실패:", href, e)
                    sub.close()
                    time.sleep(0.5)
                except Exception as e:
                    if debug:
                        print("[WARN] 상세 진입 실패:", detail_url, e)

            # 다음 페이지 이동
            moved = False
            for patt in SELECTORS["next_button"]:
                try:
                    # role 기반 → 링크/버튼 시도
                    if click_by_text_candidates(page, [patt]):
                        moved = True
                        break
                except Exception:
                    continue
            if not moved:
                # 페이지네이션이 숫자형일 수 있으므로 '다음'이 없으면 종료
                break
            # 다음 페이지 로딩 대기
            try:
                page.wait_for_load_state("networkidle", timeout=12000)
            except Exception:
                pass
            page_no += 1

        browser.close()
        print(f"[DONE] 총 다운로드: {total_downloads}개, 저장 위치: {out_dir.resolve()}")

def parse_args():
    ap = argparse.ArgumentParser(description="국회 회의록 기록시스템 HWP 일괄 다운로드")
    ap.add_argument("--out-dir", type=str, default="downloads_hwp", help="HWP 저장 폴더")
    ap.add_argument("--max-pages", type=int, default=10, help="목록에서 순회할 최대 페이지 수")
    ap.add_argument("--detail-timeout", type=int, default=15, help="상세페이지 네트워크 유휴 대기(초)")
    ap.add_argument("--no-headless", action="store_true", help="브라우저 UI 표시(디버그 시 유용)")
    ap.add_argument("--debug", action="store_true", help="디버그 로그")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    crawl(
        out_dir=Path(args.out_dir),
        max_pages=args.max_pages,
        detail_timeout=args.detail_timeout,
        headless=not args.no_headless,
        debug=args.debug
    )
