"""EDA 데이터 파일 목록 + 카테고리 매핑 레지스트리.

48개 JSON 파일을 카테고리별로 분류하고,
각 카테고리의 주요 필드, 텍스트 필드, 날짜 필드, ID 필드를 정의합니다.

Usage:
    from scripts.eda.data_registry import CATEGORIES, get_all_files, get_category_files
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

# ── 카테고리별 파일 매핑 ──────────────────────────────────────
# streaming: True면 200MB 이상이라 ijson 스트리밍 필요

_THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = _THIS_DIR.parent.parent  # backend/
PROJECT_ROOT = BACKEND_DIR.parent  # law-3-team/
DATA_DIR = PROJECT_ROOT / "data"

CATEGORIES: dict[str, dict[str, Any]] = {
    "precedent": {
        "label": "판례",
        "files": ["precedents_v1.json"],
        "id_field": "판례정보일련번호",
        "text_fields": ["판례내용", "판결요지", "판시사항", "이유"],
        "summary_field": "판례요약",
        "date_field": "선고일자",
        "streaming": True,
    },
    "law": {
        "label": "법령",
        "files": ["law_v1.json"],
        "id_field": "법령ID",
        "text_fields": ["조문"],
        "summary_field": "법령 요약",
        "date_field": None,
        "streaming": True,
    },
    "constitutional": {
        "label": "헌재결정례",
        "files": ["constitutional_v1.json"],
        "id_field": "헌재결정례일련번호",
        "text_fields": ["판시사항", "결정요지", "이유"],
        "summary_field": "심판례요약",
        "date_field": "선고일자",
        "streaming": True,
    },
    "administration": {
        "label": "행정심판례",
        "files": ["administration_v1.json"],
        "id_field": "행정심판례일련번호",
        "text_fields": ["주문", "이유"],
        "summary_field": "심판례요약",
        "date_field": "의결일자",
        "streaming": True,
    },
    "special_tribunal": {
        "label": "특별행정심판",
        "files": [
            "special_admin_appeal/sadm_case_조세심판원_v1.json",
            "special_admin_appeal/sadm_case_해양안전심판원_v1.json",
        ],
        "id_field": "특별행정심판재결례일련번호",
        "text_fields": ["주문", "이유", "청구취지"],
        "summary_field": "심판례요약",
        "date_field": "의결일자",
        "streaming": True,
    },
    "legislation": {
        "label": "법령해석례",
        "files": ["legislation_v1.json"],
        "id_field": "법령해석례일련번호",
        "text_fields": ["질의요지", "회답", "이유"],
        "summary_field": "해석례요약",
        "date_field": "해석일자",
        "streaming": False,
    },
    "committee": {
        "label": "위원회 결정문",
        "files": [
            "decisions_committee/dec_comm_개인정보보호위원회_v1.json",
            "decisions_committee/dec_comm_고용보험심사위원회_v1.json",
            "decisions_committee/dec_comm_공정거래위원회_v1.json",
            "decisions_committee/dec_comm_국가인권위원회_v1.json",
            "decisions_committee/dec_comm_국민권익위원회_v1.json",
            "decisions_committee/dec_comm_금융위원회_v1.json",
            "decisions_committee/dec_comm_노동위원회_v1.json",
            "decisions_committee/dec_comm_산업재해보상위험재심사위원회_v1.json",
            "decisions_committee/dec_comm_중앙환경분쟁조정위원회_v1.json",
            "decisions_committee/dec_comm_증권선물위원회_v1.json",
        ],
        "id_field": "결정문일련번호",
        "text_fields": ["이유", "결정요지", "주문"],
        "summary_field": "결정문요약",
        "date_field": "의결일",
        "streaming": False,
    },
    "cgm_expc": {
        "label": "부처 해석례",
        "files": [
            "interpretation_ministry/intp_min_경찰청_v1.json",
            "interpretation_ministry/intp_min_고용노동부_v1.json",
            "interpretation_ministry/intp_min_과학기술정보통신부_v1.json",
            "interpretation_ministry/intp_min_관세청_v1.json",
            "interpretation_ministry/intp_min_교육부_v1.json",
            "interpretation_ministry/intp_min_국가보훈부_v1.json",
            "interpretation_ministry/intp_min_국방부_v1.json",
            "interpretation_ministry/intp_min_국토교통부_v1.json",
            "interpretation_ministry/intp_min_기상청_v1.json",
            "interpretation_ministry/intp_min_기후에너지환경부_v1.json",
            "interpretation_ministry/intp_min_농림축산식품부_v1.json",
            "interpretation_ministry/intp_min_농촌진흥청_v1.json",
            "interpretation_ministry/intp_min_문화체육관광부_v1.json",
            "interpretation_ministry/intp_min_방위사업청_v1.json",
            "interpretation_ministry/intp_min_법무부_v1.json",
            "interpretation_ministry/intp_min_보건복지부_v1.json",
            "interpretation_ministry/intp_min_산림청_v1.json",
            "interpretation_ministry/intp_min_산업통상자원부_v1.json",
            "interpretation_ministry/intp_min_성평등가족부_v1.json",
            "interpretation_ministry/intp_min_소방청_v1.json",
            "interpretation_ministry/intp_min_식품의약품안전처_v1.json",
            "interpretation_ministry/intp_min_외교부_v1.json",
            "interpretation_ministry/intp_min_인사혁신처_v1.json",
            "interpretation_ministry/intp_min_중소벤처기업부_v1.json",
            "interpretation_ministry/intp_min_지식재산처_v1.json",
            "interpretation_ministry/intp_min_통일부_v1.json",
            "interpretation_ministry/intp_min_해양수산부_v1.json",
            "interpretation_ministry/intp_min_행정안전부_v1.json",
        ],
        "id_field": "법령해석일련번호",
        "text_fields": ["질의요지", "회답"],
        "summary_field": "해석요약",
        "date_field": "안건일자",
        "streaming": False,
    },
    "law_term": {
        "label": "법률용어사전",
        "files": ["lawterms_v1.json"],
        "id_field": "법령용어ID",
        "text_fields": ["법령용어정의"],
        "summary_field": None,
        "date_field": None,
        "streaming": False,
    },
    "treaty": {
        "label": "조약",
        "files": ["treaty_v1.json"],
        "id_field": "조약일련번호",
        "text_fields": ["조약내용"],
        "summary_field": "조약요약",
        "date_field": "서명일자",
        "streaming": False,
    },
    "school": {
        "label": "행정규칙",
        "files": ["admin_rule_v1.json"],
        "id_field": "행정규칙ID",
        "text_fields": ["조문내용"],
        "summary_field": "행정규칙요약",
        "date_field": None,
        "streaming": False,
    },
}


_VERSIONED_FILE_RE = re.compile(r"^(?P<base>.+)_v(?P<ver>\d+)\.json$")


def _resolve_latest_version(rel_path: str) -> str:
    """_vN 파일 경로를 data 디렉터리에서 최신 버전으로 해석."""
    rel = Path(rel_path)
    m = _VERSIONED_FILE_RE.match(rel.name)
    if not m:
        return rel_path

    base = m.group("base")
    norm_base = unicodedata.normalize("NFC", base)
    parent_dir = DATA_DIR / rel.parent
    if not parent_dir.exists():
        return rel_path

    best_ver = -1
    best_name: str | None = None
    pattern = re.compile(r"^(?P<base>.+)_v(?P<ver>\d+)\.json$")
    for cand in parent_dir.glob("*_v*.json"):
        cand_name = unicodedata.normalize("NFC", cand.name)
        mm = pattern.match(cand_name)
        if not mm:
            continue
        if mm.group("base") != norm_base:
            continue
        ver = int(mm.group("ver"))
        if ver > best_ver:
            best_ver = ver
            best_name = cand.name

    if best_name is None:
        return rel_path
    return str((rel.parent / best_name).as_posix()) if rel.parent != Path(".") else best_name


def _normalize_category_files() -> None:
    """CATEGORIES의 파일 목록을 최신 버전(_vN) 기준으로 정규화."""
    dynamic_patterns: dict[str, str] = {
        "committee": "decisions_committee/dec_comm_*_v*.json",
        "cgm_expc": "interpretation_ministry/intp_min_*_v*.json",
        "special_tribunal": "special_admin_appeal/sadm_case_*_v*.json",
    }

    for cat_info in CATEGORIES.values():
        files = cat_info.get("files", [])
        resolved = [_resolve_latest_version(p) for p in files]

        deduped: list[str] = []
        seen: set[str] = set()
        for path in resolved:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        cat_info["files"] = deduped

    # 디렉터리 기반 카테고리는 신규 파일도 자동 포함
    for cat_key, glob_pattern in dynamic_patterns.items():
        if cat_key not in CATEGORIES:
            continue
        matches: list[str] = []
        for p in sorted(DATA_DIR.glob(glob_pattern)):
            if not p.is_file():
                continue
            matches.append(str(p.relative_to(DATA_DIR).as_posix()))
        if matches:
            CATEGORIES[cat_key]["files"] = matches


_normalize_category_files()

# ── 부처 해석례 세부 매핑 (파일명 → 기관명) ──────────────────
CGM_EXPC_AGENCIES: dict[str, str] = {
    "경찰청": "경찰청",
    "고용노동부": "고용노동부",
    "과학기술정보통신부": "과학기술정보통신부",
    "관세청": "관세청",
    "교육부": "교육부",
    "국가보훈부": "국가보훈부",
    "국방부": "국방부",
    "국토교통부": "국토교통부",
    "기상청": "기상청",
    "기후에너지환경부": "기후에너지환경부",
    "농림축산식품부": "농림축산식품부",
    "농촌진흥청": "농촌진흥청",
    "문화체육관광부": "문화체육관광부",
    "방위사업청": "방위사업청",
    "법무부": "법무부",
    "보건복지부": "보건복지부",
    "산림청": "산림청",
    "산업통상자원부": "산업통상자원부",
    "성평등가족부": "성평등가족부",
    "소방청": "소방청",
    "식품의약품안전처": "식품의약품안전처",
    "외교부": "외교부",
    "인사혁신처": "인사혁신처",
    "중소벤처기업부": "중소벤처기업부",
    "지식재산처": "지식재산처",
    "통일부": "통일부",
    "해양수산부": "해양수산부",
    "행정안전부": "행정안전부",
}

# ── 위원회 유효 기관명 집합 (파일명에서 직접 추출 가능) ────
COMMITTEE_AGENCIES: dict[str, str] = {
    "개인정보보호위원회": "개인정보보호위원회",
    "고용보험심사위원회": "고용보험심사위원회",
    "공정거래위원회": "공정거래위원회",
    "국가인권위원회": "국가인권위원회",
    "국민권익위원회": "국민권익위원회",
    "금융위원회": "금융위원회",
    "노동위원회": "노동위원회",
    "산업재해보상위험재심사위원회": "산업재해보상위험재심사위원회",
    "중앙환경분쟁조정위원회": "중앙환경분쟁조정위원회",
    "증권선물위원회": "증권선물위원회",
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

    dec_comm_공정거래위원회_v1.json → "공정거래위원회"
    intp_min_고용노동부_v1.json → "고용노동부"
    """
    # 서브디렉토리 경로 제거 (decisions_committee/dec_comm_... → dec_comm_...)
    basename = Path(filename).name
    stem = basename.replace(".json", "")

    # dec_comm_{한국어명}_v1 패턴 (위원회)
    if stem.startswith("dec_comm_"):
        name = unicodedata.normalize(
            "NFC",
            re.sub(r"_v\d+$", "", stem.removeprefix("dec_comm_")),
        )
        if name in COMMITTEE_AGENCIES:
            return name

    # sadm_case_{한국어명}_v1 패턴 (특별행정심판)
    if stem.startswith("sadm_case_"):
        return unicodedata.normalize(
            "NFC",
            re.sub(r"_v\d+$", "", stem.removeprefix("sadm_case_")),
        )

    # intp_min_{한국어명}_v1 패턴 (부처 해석례)
    if stem.startswith("intp_min_"):
        name = unicodedata.normalize(
            "NFC",
            re.sub(r"_v\d+$", "", stem.removeprefix("intp_min_")),
        )
        if name in CGM_EXPC_AGENCIES:
            return name

    # 기존 [DONE] 패턴 (하위 호환)
    stem = filename.split("]", 1)[-1].replace(".json", "")
    if stem in CGM_EXPC_AGENCIES:
        return CGM_EXPC_AGENCIES[stem]
    return stem


def get_total_file_count() -> int:
    """전체 파일 수 반환."""
    return sum(len(cat["files"]) for cat in CATEGORIES.values())
