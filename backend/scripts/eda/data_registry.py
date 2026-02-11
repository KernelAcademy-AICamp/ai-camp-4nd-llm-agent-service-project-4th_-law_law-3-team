"""[DONE] 파일 목록 + 카테고리 매핑 레지스트리.

48개 [DONE] JSON 파일을 카테고리별로 분류하고,
각 카테고리의 주요 필드, 텍스트 필드, 날짜 필드, ID 필드를 정의합니다.

Usage:
    from scripts.eda.data_registry import CATEGORIES, get_all_files, get_category_files
"""

from __future__ import annotations

from typing import Any

# ── 카테고리별 파일 매핑 ──────────────────────────────────────
# streaming: True면 200MB 이상이라 ijson 스트리밍 필요

CATEGORIES: dict[str, dict[str, Any]] = {
    "precedent": {
        "label": "판례",
        "files": ["[DONE]precedents-4.json"],
        "id_field": "판례정보일련번호",
        "text_fields": ["판례내용", "판결요지", "판시사항", "이유"],
        "date_field": "선고일자",
        "streaming": True,
    },
    "law": {
        "label": "법령",
        "files": ["[DONE]law-2.json"],
        "id_field": "법령ID",
        "text_fields": ["조문"],
        "date_field": None,
        "streaming": True,
    },
    "constitutional": {
        "label": "헌재결정례",
        "files": ["[DONE]constitutional.json"],
        "id_field": "헌재결정례일련번호",
        "text_fields": ["판시사항", "결정요지", "이유"],
        "date_field": "선고일자",
        "streaming": True,
    },
    "administration": {
        "label": "행정심판례",
        "files": ["[DONE]administration.json"],
        "id_field": "행정심판례일련번호",
        "text_fields": ["주문", "이유"],
        "date_field": "의결일자",
        "streaming": True,
    },
    "special_tribunal": {
        "label": "특별행정심판",
        "files": [
            "[DONE]kmstSpecialDecc.json",
            "[DONE]ttSpecialDecc.json",
        ],
        "id_field": "특별행정심판재결례일련번호",
        "text_fields": ["주문", "이유", "청구취지"],
        "date_field": "의결일자",
        "streaming": True,
    },
    "legislation": {
        "label": "법령해석례",
        "files": ["[DONE]legislation.json"],
        "id_field": "법령해석례일련번호",
        "text_fields": ["질의요지", "회답", "이유"],
        "date_field": "해석일자",
        "streaming": False,
    },
    "committee": {
        "label": "위원회 결정문",
        "files": [
            "[DONE]acr.json",       # 국민권익위원회
            "[DONE]ecc.json",       # 선거관리위원회
            "[DONE]eiac.json",      # 환경분쟁조정위원회
            "[DONE]fsc.json",       # 금융위원회
            "[DONE]ftc.json",       # 공정거래위원회
            "[DONE]iaciac.json",    # 정보공개위원회
            "[DONE]nhrck.json",     # 국가인권위원회
            "[DONE]nlrc.json",      # 노동위원회
            "[DONE]ppc.json",       # 개인정보보호위원회
            "[DONE]sfc.json",       # 소청심사위원회
        ],
        "id_field": "결정문일련번호",
        "text_fields": ["이유", "결정요지", "주문"],
        "date_field": "의결일",
        "streaming": False,
    },
    "cgm_expc": {
        "label": "부처 해석례",
        "files": [
            "[DONE]dapaCgmExpc.json",
            "[DONE]kcsCgmExpc.json",
            "[DONE]kfsCgmExpc.json",
            "[DONE]kipoCgmExpc.json",
            "[DONE]kmaCgmExpc.json",
            "[DONE]mafraCgmExpc.json",
            "[DONE]mcstCgmExpc.json",
            "[DONE]meCgmExpc.json",
            "[DONE]mfdsCgmExpc.json",
            "[DONE]mndCgmExpc.json",
            "[DONE]moeCgmExpc.json",
            "[DONE]moelCgmExpc.json",
            "[DONE]mofaCgmExpc.json",
            "[DONE]mofCgmExpc.json",
            "[DONE]mogefCgmExpc.json",
            "[DONE]mohwCgmExpc.json",
            "[DONE]moisCgmExpc.json",
            "[DONE]mojCgmExpc.json",
            "[DONE]molitCgmExpc.json",
            "[DONE]motieCgmExpc.json",
            "[DONE]mouCgmExpc.json",
            "[DONE]mpmCgmExpc.json",
            "[Done]mpvaCgmExpc.json",  # Note: [Done] 대소문자
            "[DONE]msitCgmExpc.json",
            "[DONE]mssCgmExpc.json",
            "[DONE]nfaCgmExpc.json",
            "[DONE]npaCgmExpc.json",
            "[DONE]rdaCgmExpc.json",
        ],
        "id_field": "법령해석일련번호",
        "text_fields": ["질의요지", "회답"],
        "date_field": "안건일자",
        "streaming": False,
    },
    "law_term": {
        "label": "법률용어사전",
        "files": ["[DONE]lawterms.json"],
        "id_field": "법령용어ID",
        "text_fields": ["법령용어정의"],
        "date_field": None,
        "streaming": False,
    },
    "treaty": {
        "label": "조약",
        "files": ["[DONE]treaty.json"],
        "id_field": "조약일련번호",
        "text_fields": ["조약내용"],
        "date_field": "서명일자",
        "streaming": False,
    },
    "school": {
        "label": "행정규칙",
        "files": ["[DONE]school.json"],
        "id_field": "행정규칙ID",
        "text_fields": ["조문내용"],
        "date_field": None,
        "streaming": False,
    },
}

# ── 부처 해석례 세부 매핑 (파일명 → 기관명) ──────────────────
CGM_EXPC_AGENCIES: dict[str, str] = {
    "dapaCgmExpc": "방위사업청",
    "kcsCgmExpc": "관세청",
    "kfsCgmExpc": "산림청",
    "kipoCgmExpc": "특허청",
    "kmaCgmExpc": "기상청",
    "mafraCgmExpc": "농림축산식품부",
    "mcstCgmExpc": "문화체육관광부",
    "meCgmExpc": "환경부",
    "mfdsCgmExpc": "식품의약품안전처",
    "mndCgmExpc": "국방부",
    "moeCgmExpc": "교육부",
    "moelCgmExpc": "고용노동부",
    "mofaCgmExpc": "외교부",
    "mofCgmExpc": "기획재정부",
    "mogefCgmExpc": "여성가족부",
    "mohwCgmExpc": "보건복지부",
    "moisCgmExpc": "행정안전부",
    "mojCgmExpc": "법무부",
    "molitCgmExpc": "국토교통부",
    "motieCgmExpc": "산업통상자원부",
    "mouCgmExpc": "통일부",
    "mpmCgmExpc": "인사혁신처",
    "mpvaCgmExpc": "국가보훈부",
    "msitCgmExpc": "과학기술정보통신부",
    "mssCgmExpc": "중소벤처기업부",
    "nfaCgmExpc": "새만금개발청",
    "npaCgmExpc": "경찰청",
    "rdaCgmExpc": "농촌진흥청",
}

# ── 위원회 세부 매핑 (파일명 → 기관명) ──────────────────────
COMMITTEE_AGENCIES: dict[str, str] = {
    "acr": "국민권익위원회",
    "ecc": "선거관리위원회",
    "eiac": "환경분쟁조정위원회",
    "fsc": "금융위원회",
    "ftc": "공정거래위원회",
    "iaciac": "정보공개위원회",
    "nhrck": "국가인권위원회",
    "nlrc": "노동위원회",
    "ppc": "개인정보보호위원회",
    "sfc": "소청심사위원회",
}


def get_all_files() -> list[dict[str, str]]:
    """모든 카테고리의 파일 목록을 플랫하게 반환.

    Returns:
        [{category, file, label}, ...]
    """
    result = []
    for cat_key, cat_info in CATEGORIES.items():
        for filename in cat_info["files"]:
            result.append({
                "category": cat_key,
                "file": filename,
                "label": cat_info["label"],
            })
    return result


def get_category_files(category: str) -> list[str]:
    """특정 카테고리의 파일명 리스트 반환."""
    if category not in CATEGORIES:
        msg = f"Unknown category: {category}. Available: {list(CATEGORIES.keys())}"
        raise ValueError(msg)
    return CATEGORIES[category]["files"]


def get_agency_name(filename: str) -> str:
    """파일명에서 기관명 추출.

    [DONE]moelCgmExpc.json → "고용노동부"
    [DONE]ftc.json → "공정거래위원회"
    """
    # [DONE] 또는 [Done] 접두사 제거, .json 제거
    stem = filename.split("]", 1)[-1].replace(".json", "")

    if stem in CGM_EXPC_AGENCIES:
        return CGM_EXPC_AGENCIES[stem]
    if stem in COMMITTEE_AGENCIES:
        return COMMITTEE_AGENCIES[stem]
    return stem


def get_total_file_count() -> int:
    """전체 파일 수 반환."""
    return sum(len(cat["files"]) for cat in CATEGORIES.values())
