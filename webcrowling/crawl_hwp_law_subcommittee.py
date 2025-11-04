#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crawl_hwp_law_subcommittee.py

목표
- 제21대 / 상임위원회 / 위원회명: 법제사법위원회(value=1525) / 검색구분: 소위원회(value=sub_cmit)
- 결과에서 모든 "상세내용 보기"를 펼친 뒤, 노출되는 모든 HWP 다운로드 링크
  (/assembly/viewer/minutes/download/hwp.do?id=...)를 일괄 다운로드

전략
1) /assembly/mnts/total/21.do 접속
2) 좌측/상단의 '상임위원회' 링크 클릭 (href에 class_id_sch=2)
3) 위원회 select에서 option[value="1525"] 선택 (법제사법위원회)
4) 검색구분 select에서 option[value="sub_cmit"] 선택 (소위원회)
5) '검색' 버튼(.btn.blue[type=submit]) 클릭
6) 결과 페이지에서 ".btn_tit.cmit" (상세내용 보기) 버튼들을 모두 클릭하여 확장
7) 확장된 영역 안의 hwp 링크들을 모두 수집 & 다운로드 (expect_download)
8) 페이지네이션(다음/숫자) 반복 (기본 20페이지)

사용
    python crawl_hwp_law_subcommittee.py --out downloads_hwp --max-pages 20 --no-headless --delay 2.0
"""
from __future__ import annotations
import re
import os
import time
import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout, Download

BASE = "https://record.assembly.go.kr"
START = f"{BASE}/assembly/mnts/total/21.do"

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def click_anchor_contains(page, href_sub: str) -> bool:
    """href에 특정 쿼리(예: class_id_sch=2)가 들어간 a를 클릭."""
    loc = page.locator(f"a[href*='{href_sub}']").first
    if loc and loc.count() > 0:
        loc.click()
        return True
    return False

def select_by_option_value(page, value: str) -> bool:
    """
    option[value=...]를 가진 <select>를 찾아 value 선택.
    - 보이지 않거나 custom UI여도 select_option 우선 시도
    - 실패 시 JS로 value 세팅 + change 이벤트 디스패치
    - 여러 개 있으면 순차 시도
    """
    # 후보 select 모두 찾기
    selects = page.locator(f"select:has(option[value='{value}'])")
    count = selects.count()
    if count == 0:
        return False

    for i in range(count):
        sel = selects.nth(i)
        try:
            # 보이게 만들기(가능한 경우)
            try:
                sel.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                pass

            # 1) 정석: select_option
            sel.select_option(value=value)
            # 선택 반영 대기 (이 사이트는 종종 ajax로 필터 반영)
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            return True
        except Exception:
            # 2) Fallback: parent select에 JS로 값 주입 + change 이벤트
            try:
                # select 핸들을 얻어 JS 실행
                page.evaluate(
                    """(sel, val) => {
                        sel.value = val;
                        const ev1 = new Event('change', { bubbles: true });
                        const ev2 = new Event('input',  { bubbles: true });
                        sel.dispatchEvent(ev1);
                        sel.dispatchEvent(ev2);
                    }""",
                    sel, value
                )
                try:
                    page.wait_for_load_state("networkidle", timeout=3000)
                except Exception:
                    pass
                return True
            except Exception:
                continue
    return False

def click_search(page) -> bool:
    """검색 버튼 .btn.blue[type=submit] 클릭."""
    btn = page.locator("button.btn.blue[type='submit']").first
    if btn and btn.count() > 0:
        btn.click()
        return True
    return False

def expand_all_details(page, delay: float):
    """
    결과 목록에서 '상세내용 보기' 버튼(.btn_tit.cmit)을 모두 클릭해 펼친다.
    ex) <button class="btn_tit cmit" ...>상세내용 보기</button>
    """
    # 동적으로 늘어날 수 있어 안전하게 반복
    while True:
        btns = page.locator("button.btn_tit.cmit").filter(has_text=re.compile(r"상세내용\s*보기"))
        n = btns.count()
        if n == 0:
            break
        clicked = 0
        for i in range(n):
            try:
                btns.nth(i).click()
                clicked += 1
                time.sleep(max(delay, 0.3))
            except Exception:
                pass
        # 이미 모두 펼쳤으면 루프 종료
        if clicked == 0:
            break

def absolute_url(current_url: str, href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return BASE + href
    base = current_url.rsplit("/", 1)[0]
    return base + "/" + href

def collect_hwp_links(page) -> List[str]:
    """
    펼쳐진 상세들에서 HWP 다운로드 a를 수집.
    예: <a href="/assembly/viewer/minutes/download/hwp.do?id=45139" ... class="btn_ico"><img ... alt="hwp..."></a>
    """
    links = []
    anchors = page.locator("a[href*='/assembly/viewer/minutes/download/hwp.do']")
    for i in range(anchors.count()):
        try:
            a = anchors.nth(i)
            href = a.get_attribute("href") or ""
            if not href:
                continue
            links.append(absolute_url(page.url, href))
        except Exception:
            pass
    # 중복 제거 (순서 유지)
    seen = set()
    uniq = []
    for u in links:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq

def save_download(page, url: str, out: Path, prefix: str = "") -> str:
    # Playwright download 이벤트를 사용해 저장(세션/쿠키/POST 대응)
    with page.expect_download(timeout=30000) as dl_info:
        page.evaluate("(u)=>window.open(u,'_self')", url)
    dl: Download = dl_info.value
    suggested = dl.suggested_filename or os.path.basename(url.split("?")[0]) or "file.hwp"
    fname = suggested
    if prefix:
        fname = f"{prefix}_{suggested}"
    # 파일명 안전화
    fname = re.sub(r"[\\/:*?\"<>|]", "_", fname)
    fname = re.sub(r"\s+", " ", fname).strip()
    dl.save_as(str(out / fname))
    return fname

def goto_next_page(page) -> bool:
    """
    페이지네이션:
      1) '다음' 있는 경우 클릭
      2) 없으면 숫자형 페이지네이션을 찾아 '현재+1' 클릭 시도
    실패하면 False 반환.
    """
    # 1) '다음'
    next_btn = page.get_by_role("link", name=re.compile(r"다음|Next|▶")).first
    if next_btn and next_btn.count() > 0:
        try:
            next_btn.click()
            return True
        except Exception:
            pass
    # 2) 숫자형 (현재 활성 페이지 찾아 +1)
    # 활성 페이지는 보통 <strong> 이거나 .on, .active 같은 클래스
    try:
        current = page.locator("nav, .pagination, .paginate, .paging").locator("strong, .on, .active").first
        if current and current.count() > 0:
            txt = current.inner_text().strip()
            cur = int(re.sub(r"\D", "", txt)) if re.search(r"\d", txt) else None
            if cur:
                # 다음 숫자 클릭
                cand = page.get_by_role("link", name=str(cur+1)).first
                if cand and cand.count() > 0:
                    cand.click()
                    return True
    except Exception:
        pass
    return False

def run(out_dir: Path, max_pages: int, no_headless: bool, delay: float):
    out_dir = ensure_dir(out_dir)
    total = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not no_headless)
        ctx = browser.new_context(accept_downloads=True, locale="ko-KR")
        page = ctx.new_page()

        # 1) 시작
        page.goto(START, wait_until="domcontentloaded", timeout=45000)

        # 2) '상임위원회' 앵커 클릭 (1번 힌트)
        # 2) '상임위원회' 링크 클릭
        clicked = click_anchor_contains(page, "class_id_sch=2")
        if clicked:
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            time.sleep(0.5)

        # 3) 위원회=법제사법위원회 (value=1525) (2번 힌트)
        select_by_option_value(page, "1525")
        time.sleep(0.5)  # ★ 추가


        # 4) 검색구분=소위원회 (value=sub_cmit) (3번 힌트)
        select_by_option_value(page, "sub_cmit")
        time.sleep(0.5)  # ★ 추가

        # 5) '검색' 버튼 클릭 (4번 힌트)
        if click_search(page):
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            time.sleep(delay)

        # 페이지 루프
        page_idx = 1
        while page_idx <= max_pages:
            # 6) '상세내용 보기' 모두 펼치기 (5번 힌트: .btn_tit.cmit)
            try:
                expand_all_details(page, delay=delay)
            except Exception:
                pass

            # 7) HWP 링크 수집 (7번 힌트: /assembly/viewer/minutes/download/hwp.do?id=...)
            links = collect_hwp_links(page)

            # 8) 다운로드
            for url in tqdm(links, desc=f"[p{page_idx}] HWP"):
                try:
                    fname = save_download(page, url, out_dir)
                    total += 1
                    time.sleep(delay)
                except Exception as e:
                    print("[WARN] 다운로드 실패:", url, e)

            # 다음 페이지 이동
            moved = False
            try:
                moved = goto_next_page(page)
                if moved:
                    try:
                        page.wait_for_load_state("networkidle", timeout=12000)
                    except Exception:
                        pass
                    time.sleep(delay)
            except Exception:
                moved = False

            if not moved:
                break
            page_idx += 1

        browser.close()

    print(f"[DONE] 총 HWP 다운로드: {total}개  →  {out_dir.resolve()}")
    if total == 0:
        print("※ 결과가 0개라면: 조건(대수/위원회/소위원회) 여부, '상세내용 보기'가 실제로 열리는지, 페이지네이션 유무를 점검하세요.")
        print("※ --no-headless 로 화면을 보면서 선택자에 변화가 없는지 확인하세요.")

def main():
    ap = argparse.ArgumentParser(description="국회 회의록(법제사법위 소위원회) HWP 일괄 다운로드")
    ap.add_argument("--out", type=str, default="downloads_hwp", help="저장 폴더")
    ap.add_argument("--max-pages", type=int, default=20, help="순회할 최대 페이지 수")
    ap.add_argument("--no-headless", action="store_true", help="브라우저 창 표시(디버그용)")
    ap.add_argument("--delay", type=float, default=0.6, help="클릭/다운 간 대기(초)")
    args = ap.parse_args()

    run(
        out_dir=Path(args.out),
        max_pages=args.max_pages,
        no_headless=args.no_headless,
        delay=args.delay
    )

if __name__ == "__main__":
    main()
