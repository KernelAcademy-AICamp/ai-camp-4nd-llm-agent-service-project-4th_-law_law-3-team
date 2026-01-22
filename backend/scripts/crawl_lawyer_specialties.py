#!/usr/bin/env python3
"""
대한변호사협회에서 전문분야별 변호사 목록을 크롤링하는 스크립트

사용법:
    # Playwright 설치 필요
    uv add playwright
    uv run playwright install chromium

    # 크롤링 실행
    uv run python scripts/crawl_lawyer_specialties.py

    # 특정 전문분야만 크롤링 (디버깅용)
    uv run python scripts/crawl_lawyer_specialties.py --specialty 25

    # 헤드리스 모드 비활성화 (디버깅용)
    uv run python scripts/crawl_lawyer_specialties.py --no-headless
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from playwright.sync_api import Page, sync_playwright

# stdout 버퍼링 비활성화 (실시간 출력용)
sys.stdout.reconfigure(line_buffering=True)

# 프로젝트 경로
BACKEND_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "lawyers_with_coords.json"
OUTPUT_FILE = DATA_DIR / "lawyers_with_specialties.json"

# 전문분야 코드 매핑 (대한변호사협회 '신' 전문분야 기준)
SPECIALTIES: dict[int, str] = {
    1: "민사법",
    2: "부동산",
    3: "건설",
    4: "재개발·재건축",
    5: "의료",
    6: "교통사고",
    7: "등기·경매",
    8: "손해배상",
    9: "임대차관련법",
    10: "국가계약",
    11: "민사집행",
    12: "채권추심",
    13: "상사법",
    14: "회사법",
    15: "인수합병",
    16: "도산",
    17: "증권",
    18: "금융",
    19: "보험",
    20: "해상",
    21: "무역",
    22: "조선",
    23: "중재",
    24: "IT",
    25: "형사법",
    26: "군형법",
    27: "가사법",
    28: "이혼",
    29: "상속",
    30: "소년법",
    31: "행정법",
    32: "공정거래",
    33: "언론·방송통신",
    34: "헌법재판",
    35: "환경",
    36: "에너지",
    37: "식품·의약",
    38: "수용 및 보상",
    39: "노동법",
    40: "산재",
    41: "조세법",
    42: "관세",
    43: "법인세",
    44: "상속증여세",
    45: "국제조세",
    46: "상표",
    47: "특허",
    48: "저작권",
    49: "영업비밀",
    50: "엔터테인먼트",
    51: "국제관계법",
    52: "국제거래",
    53: "국제중재",
    54: "해외투자",
    55: "이주 및 비자",
    56: "스포츠",
    57: "종교",
    58: "성년후견",
    59: "스타트업",
    60: "학교폭력",
    61: "입법",
    62: "지식재산권법",
    63: "신탁",
    64: "기타",
}


def parse_lawyer_row(row_text: str) -> Optional[dict]:
    """
    검색 결과 테이블의 행 텍스트를 파싱

    테이블 구조 (탭으로 구분):
    No | 사진(빈값) | 성명\\n개업\\n[상세보기] | 전문분야\\n(등록일) | 출생년도 | 사무소명 | 주소\\n전화 | Map

    Args:
        row_text: 테이블 행의 텍스트 내용

    Returns:
        파싱된 변호사 정보 또는 None
    """
    # 탭으로 주요 컬럼 분리
    columns = row_text.split("\t")

    if len(columns) < 6:
        return None

    # 첫 번째 컬럼이 숫자(순번)인지 확인
    first_col = columns[0].strip()
    try:
        int(first_col)
    except ValueError:
        return None  # 헤더 행 스킵

    # 성명 컬럼 (인덱스 2, 사진 컬럼이 빈값이므로)
    # 형식: "남하경\n개업\n[상세보기]"
    name_col = columns[2].strip() if len(columns) > 2 else ""

    # 줄바꿈으로 분리하여 이름과 상태 추출
    name_parts = [p.strip() for p in name_col.split("\n") if p.strip()]

    if not name_parts:
        return None

    name = name_parts[0]  # 첫 줄이 이름
    status = name_parts[1] if len(name_parts) > 1 else ""  # 둘째 줄이 상태

    # 이름 유효성 체크 (한글 2글자 이상)
    if not name or len(name) < 2:
        return None

    # 사무소명 (인덱스 5 또는 6)
    office_name = ""
    for idx in [5, 6]:
        if idx < len(columns):
            col = columns[idx].strip()
            # 사무소 관련 키워드 확인
            if any(keyword in col for keyword in ["법률사무소", "법무법인", "변호사"]):
                office_name = col.split("\n")[0].strip()  # 첫 줄만
                break

    return {
        "name": name,
        "status": status,
        "office_name": office_name,
    }


def crawl_specialty_page(
    page: Page,
    specialty_code: int,
    specialty_name: str,
    max_pages: int = 100
) -> list[dict]:
    """
    특정 전문분야의 변호사 목록 크롤링

    Args:
        page: Playwright 페이지 객체
        specialty_code: 전문분야 코드 (1-64)
        specialty_name: 전문분야 이름
        max_pages: 최대 페이지 수

    Returns:
        변호사 정보 리스트
    """
    lawyers = []

    try:
        # 검색 페이지 접속
        page.goto(
            "https://www.koreanbar.or.kr/pages/search/search1.asp",
            wait_until="networkidle",
            timeout=30000
        )

        # 전문분야 select 요소 대기
        page.wait_for_selector("#special1", timeout=10000)

        # '신(新)' 전문분야 선택 (value="1")
        page.select_option("#special1", "1")

        # JavaScript가 special1_1을 동적으로 채울 때까지 대기
        page.wait_for_function(
            """() => {
                const sel = document.querySelector('#special1_1');
                return sel && sel.options.length > 1;
            }""",
            timeout=5000
        )
        page.wait_for_timeout(300)

        # 세부 전문분야 선택
        page.select_option("#special1_1", str(specialty_code))
        page.wait_for_timeout(200)

        # 지역 선택 (서울) - 선택 사항
        page.select_option("#sido1", "서울")
        page.wait_for_timeout(200)

        # 전문분야 검색 버튼 클릭 (fnc_goSubmit(2))
        # 폼 제출 시 페이지 네비게이션 발생
        with page.expect_navigation(timeout=30000):
            page.evaluate("fnc_goSubmit(2)")
        page.wait_for_load_state("domcontentloaded", timeout=15000)
        page.wait_for_timeout(500)

        # 결과 테이블 파싱
        current_page = 1

        while current_page <= max_pages:
            # 결과 테이블 찾기 (table_style4 클래스)
            table = page.locator("table.table_style4")
            if table.count() == 0:
                break

            # 테이블 행 추출 (tbody가 없을 수 있음)
            rows = table.locator("tr")
            row_count = rows.count()

            if row_count <= 1:  # 헤더만 있는 경우
                break

            found_lawyers = 0
            for i in range(1, row_count):  # 헤더 스킵
                row = rows.nth(i)
                row_text = row.inner_text()

                # "검색 결과가 없습니다" 또는 빈 결과 체크
                if "검색 결과가 없습니다" in row_text:
                    break
                if "결과가 없습니다" in row_text:
                    break

                lawyer_info = parse_lawyer_row(row_text)
                if lawyer_info and lawyer_info.get("name"):
                    lawyers.append(lawyer_info)
                    found_lawyers += 1

            if found_lawyers == 0:
                break

            # 페이지네이션: 다음 페이지 확인
            next_page_num = current_page + 1

            # 현재 페이지에서 goPage 함수가 정의되어 있는지 확인
            has_gopage = page.evaluate("typeof goPage === 'function'")
            if not has_gopage:
                break

            # 다음 페이지 링크 존재 확인
            next_link = page.locator(f"a[href*='goPage({next_page_num})']")
            if next_link.count() == 0:
                # 마지막 페이지 도달
                break

            # goPage 함수 호출로 페이지 이동
            page.evaluate(f"goPage({next_page_num})")
            page.wait_for_load_state("domcontentloaded", timeout=10000)
            page.wait_for_timeout(500)
            current_page += 1

        print(f"  [{specialty_code:02d}] {specialty_name}: {len(lawyers)}명")

    except Exception as e:
        print(f"  [{specialty_code:02d}] {specialty_name}: 오류 - {e}")

    return lawyers


def crawl_all_specialties(
    headless: bool = True,
    specialty_filter: Optional[int] = None
) -> dict[str, list[str]]:
    """
    모든 전문분야 크롤링

    Args:
        headless: 헤드리스 모드 사용 여부
        specialty_filter: 특정 전문분야만 크롤링 (디버깅용)

    Returns:
        {변호사명: [전문분야 리스트]} 형태의 딕셔너리
    """
    lawyer_specialties: dict[str, list[str]] = defaultdict(list)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        specialties_to_crawl = (
            {specialty_filter: SPECIALTIES[specialty_filter]}
            if specialty_filter
            else SPECIALTIES
        )

        print(f"\n전문분야 {len(specialties_to_crawl)}개 크롤링 시작...\n")

        for code, name in specialties_to_crawl.items():
            lawyers = crawl_specialty_page(page, code, name)

            for lawyer in lawyers:
                lawyer_name = lawyer["name"]
                if name not in lawyer_specialties[lawyer_name]:
                    lawyer_specialties[lawyer_name].append(name)

            # API 부하 방지
            time.sleep(1)

        browser.close()

    return dict(lawyer_specialties)


def merge_with_existing_data(
    crawled_specialties: dict[str, list[str]]
) -> dict:
    """
    크롤링한 전문분야 정보를 기존 변호사 데이터와 병합

    Args:
        crawled_specialties: {변호사명: [전문분야 리스트]}

    Returns:
        병합된 데이터
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    lawyers = data.get("lawyers", [])
    matched_count = 0

    for lawyer in lawyers:
        name = lawyer.get("name", "")
        if name in crawled_specialties:
            lawyer["specialties"] = crawled_specialties[name]
            matched_count += 1
        else:
            lawyer["specialties"] = []

    # 메타데이터 업데이트
    metadata = data.get("metadata", {})
    metadata["specialty_crawl_date"] = datetime.now().isoformat()
    metadata["with_specialties"] = matched_count
    metadata["specialty_source"] = "koreanbar.or.kr"

    return {
        "metadata": metadata,
        "lawyers": lawyers
    }


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="대한변호사협회 전문분야별 변호사 크롤링"
    )
    parser.add_argument(
        "--specialty",
        type=int,
        help="특정 전문분야 코드만 크롤링 (1-64)"
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="브라우저 화면 표시 (디버깅용)"
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="기존 데이터 병합 생략 (크롤링 결과만 저장)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("대한변호사협회 전문분야 변호사 크롤러")
    print("=" * 60)

    # 크롤링 실행
    headless = not args.no_headless
    crawled_specialties = crawl_all_specialties(
        headless=headless,
        specialty_filter=args.specialty
    )

    print(f"\n크롤링 완료: {len(crawled_specialties)}명의 변호사 전문분야 수집")

    if args.skip_merge:
        # 크롤링 결과만 저장
        output_file = DATA_DIR / "specialty_crawl_result.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(crawled_specialties, f, ensure_ascii=False, indent=2)
        print(f"크롤링 결과 저장: {output_file}")
        return

    # 기존 데이터와 병합
    print("\n기존 데이터와 병합 중...")
    merged_data = merge_with_existing_data(crawled_specialties)

    # 결과 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    # 통계 출력
    metadata = merged_data["metadata"]
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"총 변호사 수: {metadata.get('total_count', len(merged_data['lawyers']))}명")
    print(f"전문분야 있는 변호사: {metadata.get('with_specialties', 0)}명")
    print(f"출력 파일: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
