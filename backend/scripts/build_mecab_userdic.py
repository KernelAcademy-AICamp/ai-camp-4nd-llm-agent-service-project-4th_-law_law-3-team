"""
MeCab 사용자 사전(userdic) 빌드 스크립트

법률 용어를 MeCab userdic에 등록하여 복합명사를 올바르게 인식하고,
FTS 부분검색을 위한 분해맵(decomposition_map.json)을 생성한다.

사전 조건:
  - mecab, libmecab-dev, mecab-ko-dic 시스템 패키지 설치
  - PostgreSQL legal_terms 테이블 존재 (uv run alembic upgrade head)
  - 법률 용어 데이터 로드 (uv run python scripts/load_legal_terms_data.py)

Usage:
    cd backend

    # 기본 빌드 (DB에서 용어 로드)
    uv run python scripts/build_mecab_userdic.py

    # JSON fallback (DB 미사용)
    uv run python scripts/build_mecab_userdic.py --from-json

    # 통계만 (빌드 안 함)
    uv run python scripts/build_mecab_userdic.py --dry-run

    # 빌드 후 검증
    uv run python scripts/build_mecab_userdic.py --verify
"""

import argparse
import asyncio
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# 출력 경로
OUTPUT_DIR = BACKEND_DIR / "data" / "mecab_userdic"
CSV_PATH = OUTPUT_DIR / "legal_terms.csv"
DIC_PATH = OUTPUT_DIR / "legal_terms.dic"
DECOMP_MAP_PATH = OUTPUT_DIR / "decomposition_map.json"
PRIORITY_TERMS_PATH = OUTPUT_DIR / "priority_terms.json"
MANUAL_TERMS_PATH = OUTPUT_DIR / "manual_terms.json"

# MeCab 시스템 경로
MECAB_DICT_INDEX_CANDIDATES = [
    "/usr/lib/mecab/mecab-dict-index",
    "/usr/local/lib/mecab/mecab-dict-index",
    "/usr/local/libexec/mecab/mecab-dict-index",
    "/opt/homebrew/lib/mecab/mecab-dict-index",
    "/opt/homebrew/libexec/mecab/mecab-dict-index",
    "/opt/homebrew/Cellar/mecab-ko/0.996-ko-0.9.2/libexec/mecab/mecab-dict-index",
]
SYS_DICT_CANDIDATES = [
    "/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ko-dic",
    "/usr/local/lib/mecab/dic/mecab-ko-dic",
    "/usr/lib/mecab/dic/mecab-ko-dic",
    "/opt/homebrew/lib/mecab/dic/mecab-ko-dic",
]

# 법률 용어 JSON fallback 경로 (우선순위 순)
LAWTERMS_V1_JSON = PROJECT_ROOT / "data" / "lawterms_v1.json"
LEGAL_TERMS_JSON = PROJECT_ROOT / "data" / "lawterms_full.json"

# userdic CSV 비용 (낮을수록 우선 선택)
USERDIC_COST = 100
USERDIC_PRIORITY_COST = -3000  # 회귀 방지 대상 용어의 비용 (Viterbi 강력 우선)


def _find_path(candidates: list[str]) -> Optional[str]:
    """후보 경로에서 존재하는 첫 번째 반환"""
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def _has_final_consonant(char: str) -> bool:
    """마지막 문자의 받침(종성) 여부 판별

    한글: 종성 코드로 판별
    영문/숫자: 음절 연결 시 받침처럼 동작하는지 기준
      - 받침 있음(T): l, m, n, r, ng, 1, 3, 6, 7, 8, 0 등
      - 받침 없음(F): 나머지
    """
    if "\uAC00" <= char <= "\uD7A3":
        # 한글
        return ((ord(char) - 0xAC00) % 28) != 0
    # 영문/숫자: 보수적으로 T(받침 있음) 처리
    # MeCab 연결 비용에서 안전한 쪽
    return True


def _load_module(name: str, path: Path) -> object:
    """파일 경로에서 Python 모듈 동적 로드"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"모듈 로드 실패: {path}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ================================================================
# 용어 로드
# ================================================================


def _extract_bracket_core_terms(rows_all: list[str]) -> set[str]:
    """괄호 포함 용어에서 userdic 후보 추출 (두 가지 변형)

    Variant A: 괄호+내용 제거 → 접두어+접미어
        "전자(세금)계산서" → "전자계산서"
    Variant B: 괄호 내용+접미어 → 괄호 앞 접두어를 괄호 내용으로 대체
        "전자(세금)계산서" → "세금계산서"

    한자 괄호는 Variant B가 한글전용 필터에서 자동 제외:
        "가처분(假處分)" → A: "가처분" (OK), B: "假處分" (제외)

    예시:
        "전자(세금)계산서" → {"전자계산서", "세금계산서"}
        "가처분(假處分)"  → {"가처분"}
        "인지(印紙)대"    → {"인지대"}
    """
    import re

    # 괄호 매칭: (), （）, [] - 캡처 그룹으로 내용 추출
    bracket_pattern = re.compile(r"[（(]([^)）]*)[)）]|\[([^\]]*)\]")
    # 괄호+내용 전체 제거용
    bracket_remove = re.compile(r"[（(][^)）]*[)）]|\[[^\]]*\]")
    korean_only = re.compile(r"^[가-힣]+$")

    def _clean(s: str) -> str:
        return s.replace(" ", "").replace("ㆍ", "").replace("·", "")

    def _add(candidate: str, term: str) -> None:
        if not candidate or candidate == term:
            return
        if not korean_only.match(candidate):
            return
        if len(candidate) < 2 or len(candidate) > 15:
            return
        cores.add(candidate)

    cores: set[str] = set()
    for term in rows_all:
        matches = list(bracket_pattern.finditer(term))
        if not matches:
            continue

        # Variant A: 괄호+내용 전체 제거
        variant_a = _clean(bracket_remove.sub("", term).strip())
        _add(variant_a, term)

        # Variant B: 각 괄호에 대해 (괄호 내용 + 괄호 뒤 텍스트)
        for match in matches:
            content = (match.group(1) or match.group(2) or "").strip()
            suffix = term[match.end():]
            # suffix에서 추가 괄호가 있으면 제거
            suffix = bracket_remove.sub("", suffix).strip()
            variant_b = _clean(content + suffix)
            _add(variant_b, term)

    return cores


async def load_terms_from_db() -> set[str]:
    """PostgreSQL legal_terms 테이블에서 용어 로드

    대상:
    1. 한글 전용 (2-15자): 복합명사 포함 (예: 개인신용정보처리시스템)
    2. 혼합 단어 (2-15자): 영문/숫자+한글 복합어 (공백/괄호/가운뎃점 없음)
       예: A1해역, 제1심, DB서버, IP주소
    3. 괄호 포함 용어에서 핵심어 추출 (예: 전자(세금)계산서 → 전자계산서)
    """
    # app 패키지 임포트를 위해 경로 추가
    sys.path.insert(0, str(BACKEND_DIR))

    from sqlalchemy import (
        or_,  # type: ignore[import-untyped]
        select,  # type: ignore[import-untyped]
    )

    from app.core.database import async_session_factory  # type: ignore[import-untyped]
    from app.models.legal_term import LegalTerm  # type: ignore[import-untyped]

    query = select(LegalTerm.term).where(
        or_(
            # 1) 한글 전용 (2-15자)
            (
                LegalTerm.is_korean_only.is_(True)
                & (LegalTerm.term_length >= 2)
                & (LegalTerm.term_length <= 15)
            ),
            # 2) 혼합 단어 (2-15자, 한글 포함, 공백/괄호/가운뎃점 없음)
            (
                LegalTerm.is_korean_only.is_(False)
                & (LegalTerm.term_length >= 2)
                & (LegalTerm.term_length <= 15)
                & LegalTerm.term.regexp_match(r"[가-힣]")
                & ~LegalTerm.term.regexp_match(r"[ （()）)·ㆍ]")
            ),
        ),
    )

    async with async_session_factory() as session:
        result = await session.execute(query)
        rows = result.scalars().all()

        # 괄호 핵심어 추출을 위해 괄호 포함 용어도 별도 로드
        bracket_query = select(LegalTerm.term).where(
            LegalTerm.term.regexp_match(r"[（()）)]")
            & (LegalTerm.term_length >= 3)
            & (LegalTerm.term_length <= 30)
        )
        bracket_result = await session.execute(bracket_query)
        all_rows = bracket_result.scalars().all()

    import re
    # 혼합 단어 중 영문/숫자+한글만 (슬래시, 특수기호 등 제외)
    mixed_pattern = re.compile(r"^[가-힣a-zA-Z0-9]+$")
    terms = {t for t in rows if mixed_pattern.match(t)}

    # 3) 괄호 포함 용어에서 한글 핵심어 추출
    bracket_terms = _extract_bracket_core_terms(rows_all=all_rows)
    new_cores = bracket_terms - terms
    if new_cores:
        print(f"  괄호 핵심어 추가: {len(new_cores):,}개")
    terms.update(new_cores)

    return terms


def load_terms_from_json() -> set[str]:
    """JSON 파일에서 법률 용어 로드 (fallback, 리스트 평탄화 포함)

    한글 전용(2-15자) + 혼합 단어(영문/숫자+한글, 2-15자) 모두 포함.
    """
    import re

    # 한글 전용 또는 영문/숫자+한글 혼합
    valid_pattern = re.compile(r"^[가-힣a-zA-Z0-9]+$")
    has_korean = re.compile(r"[가-힣]")

    # 우선순위: lawterms_v1.json > lawterms_full.json
    if LAWTERMS_V1_JSON.exists():
        json_path = LAWTERMS_V1_JSON
    else:
        json_path = LEGAL_TERMS_JSON
    if not json_path.exists():
        print(f"[ERROR] 법률 용어 JSON 파일 없음: {json_path}")
        return set()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    terms: set[str] = set()
    all_raw_terms: list[str] = []  # 괄호 핵심어 추출용
    for item in data:
        raw_val = item.get("법령용어명_한글", "")

        # 리스트 타입 레코드 평탄화 (lawterms_v1.json의 ~15%가 리스트)
        if isinstance(raw_val, list):
            term_list = raw_val
        else:
            term_list = [raw_val]

        for term_raw in term_list:
            if not isinstance(term_raw, str):
                continue
            term = term_raw.strip()
            if not term:
                continue
            if not has_korean.search(term):
                continue
            all_raw_terms.append(term)
            if not valid_pattern.match(term):
                continue
            term_len = len(term)
            if term_len < 2 or term_len > 15:
                continue
            terms.add(term)

    # 괄호 포함 용어에서 한글 핵심어 추출
    bracket_rows = [t for t in all_raw_terms if "(" in t or "（" in t or ")" in t or "）" in t]
    bracket_cores = _extract_bracket_core_terms(bracket_rows)
    new_cores = bracket_cores - terms
    if new_cores:
        print(f"  괄호 핵심어 추가: {len(new_cores):,}개")
    terms.update(new_cores)

    return terms


# ================================================================
# CSV 생성
# ================================================================


def generate_csv(
    terms: set[str],
    csv_path: Path,
    priority_terms: Optional[set[str]] = None,
) -> int:
    """
    userdic CSV 생성

    형식: surface,1780,{3534 if 받침 else 3533},cost,NNG,*,{T/F},surface,*,*,*,*

    priority_terms에 포함된 용어는 USERDIC_PRIORITY_COST로 비용을 낮추어
    Viterbi 경합에서 우선하도록 한다.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _priority = priority_terms or set()

    lines: list[str] = []
    priority_count = 0
    for term in sorted(terms):
        last_char = term[-1]
        # 받침 여부: T(종성 있음) / F(종성 없음)
        has_jongseong = _has_final_consonant(last_char)
        right_id = 3534 if has_jongseong else 3533
        t_f = "T" if has_jongseong else "F"

        if term in _priority:
            cost = USERDIC_PRIORITY_COST
            priority_count += 1
        else:
            cost = USERDIC_COST

        line = f"{term},1780,{right_id},{cost},NNG,*,{t_f},{term},*,*,*,*"
        lines.append(line)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if priority_count > 0:
        print(f"  우선 비용 적용: {priority_count:,}개 (cost={USERDIC_PRIORITY_COST})")

    return len(lines)


# ================================================================
# 사전 컴파일
# ================================================================


def compile_dic(csv_path: Path, dic_path: Path) -> bool:
    """mecab-dict-index로 CSV → .dic 컴파일

    기존 .dic 파일이 있으면 백업 후 컴파일. 실패 시 백업에서 복원.
    컴파일 성공 후 .dic 파일 크기/무결성을 검증.
    """
    dict_index = _find_path(MECAB_DICT_INDEX_CANDIDATES)
    sys_dict = _find_path(SYS_DICT_CANDIDATES)

    if not dict_index:
        print("[ERROR] mecab-dict-index를 찾을 수 없습니다")
        print("  후보:", MECAB_DICT_INDEX_CANDIDATES)
        return False

    if not sys_dict:
        print("[ERROR] mecab-ko-dic 시스템 사전을 찾을 수 없습니다")
        print("  후보:", SYS_DICT_CANDIDATES)
        return False

    # 기존 .dic 백업
    backup_path = dic_path.with_suffix(".dic.bak")
    had_existing = dic_path.exists()
    if had_existing:
        import shutil
        shutil.copy2(dic_path, backup_path)
        print(f"[INFO] 기존 .dic 백업: {backup_path}")

    cmd = [
        dict_index,
        "-d", sys_dict,
        "-u", str(dic_path),
        "-f", "utf-8",
        "-t", "utf-8",
        str(csv_path),
    ]

    print(f"[INFO] 컴파일 중: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("[ERROR] mecab-dict-index 타임아웃 (120초)")
        _restore_backup(dic_path, backup_path, had_existing)
        return False

    if result.returncode != 0:
        print(f"[ERROR] mecab-dict-index 실패 (returncode={result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}")
        if result.stdout:
            print(f"  stdout: {result.stdout.strip()}")
        _restore_backup(dic_path, backup_path, had_existing)
        return False

    # 성공 시에도 stderr 경고 출력 (경고 메시지가 있을 수 있음)
    if result.stderr and result.stderr.strip():
        print(f"  [WARN] mecab-dict-index stderr: {result.stderr.strip()}")

    # .dic 파일 무결성 검증
    if not dic_path.exists():
        print("[ERROR] 컴파일 완료했으나 .dic 파일이 생성되지 않음")
        _restore_backup(dic_path, backup_path, had_existing)
        return False

    dic_size = dic_path.stat().st_size
    if dic_size == 0:
        print("[ERROR] .dic 파일 크기가 0바이트 (빈 파일)")
        _restore_backup(dic_path, backup_path, had_existing)
        return False

    # CSV 행 수와 .dic 크기의 합리성 검증 (행당 최소 ~20바이트 기대)
    csv_line_count = sum(1 for _ in open(csv_path, encoding="utf-8"))
    min_expected_size = csv_line_count * 20
    if dic_size < min_expected_size:
        print(
            f"[WARN] .dic 크기({dic_size:,}B)가 예상 최소({min_expected_size:,}B)보다 작음. "
            f"CSV {csv_line_count:,}행 대비 비정상적일 수 있음"
        )

    # 백업 정리 (성공)
    if backup_path.exists():
        backup_path.unlink()

    print(f"[INFO] .dic 생성 완료: {dic_path} ({dic_size:,} bytes)")
    return True


def _restore_backup(dic_path: Path, backup_path: Path, had_existing: bool) -> None:
    """컴파일 실패 시 기존 .dic 백업에서 복원"""
    if had_existing and backup_path.exists():
        import shutil
        shutil.copy2(backup_path, dic_path)
        backup_path.unlink()
        print(f"[INFO] 기존 .dic 복원 완료: {dic_path}")
    elif backup_path.exists():
        backup_path.unlink()


# ================================================================
# 분해맵 생성
# ================================================================


def build_decomposition_map(
    terms: set[str],
    dict_mod: object,
) -> dict[str, list[str]]:
    """
    복합어 분해맵 생성

    각 용어에 대해 사전 내 서브 용어를 탐색하여 분해맵을 구성.
    예: {"법정이율": ["법정", "이율"], "소멸시효": ["소멸", "시효"]}
    """
    ld = dict_mod.LegalTermDictionary()  # type: ignore[attr-defined]
    ld.load_from_terms(terms)

    decomp_map: dict[str, list[str]] = {}
    for term in sorted(terms):
        sub_terms = ld.find_terms_in_text(term)
        # 자기 자신 제외
        sub_terms = [st for st in sub_terms if st != term]
        if sub_terms:
            decomp_map[term] = sub_terms

    return decomp_map


# ================================================================
# 검증
# ================================================================


def verify_userdic(dic_path: Path, sample_terms: list[str]) -> bool:
    """userdic 로드 후 샘플 용어 인식 검증"""
    try:
        import MeCab
    except ImportError:
        print("[ERROR] MeCab Python 패키지 미설치")
        return False

    sys_dict = _find_path(SYS_DICT_CANDIDATES)
    if not sys_dict:
        print("[ERROR] 시스템 사전 경로를 찾을 수 없습니다")
        return False

    try:
        tagger = MeCab.Tagger(f"-d {sys_dict} -u {dic_path}")
    except RuntimeError as e:
        print(f"[ERROR] MeCab Tagger 초기화 실패: {e}")
        return False

    print("\n검증 결과:")
    print("-" * 60)
    all_ok = True
    for term in sample_terms:
        parsed = tagger.parse(term)
        lines = [ln for ln in parsed.strip().split("\n") if ln and ln != "EOS"]

        # 단일 토큰으로 인식되었는지 확인
        surfaces = []
        for line in lines:
            parts = line.split("\t")
            surfaces.append(parts[0])

        is_single = len(surfaces) == 1 and surfaces[0] == term
        status = "OK" if is_single else "SPLIT"
        if not is_single:
            all_ok = False
        print(f"  [{status}] {term} → {surfaces}")

    print("-" * 60)
    return all_ok


# ================================================================
# 회귀 탐지
# ================================================================


PARTICLES = ["이", "가", "은", "는", "을", "의", "에", "도", "로", "와", "과", "에서", "으로"]


def _parse_surfaces(tagger: object, text: str) -> list[str]:
    """MeCab parseToNode로 표면형 리스트 반환"""
    node = tagger.parseToNode(text)  # type: ignore[attr-defined]
    node = node.next  # BOS 건너뛰기
    surfaces: list[str] = []
    while node and node.next:  # EOS 건너뛰기
        surfaces.append(node.surface)
        node = node.next
    return surfaces


def detect_regression_terms(
    dic_path: Path,
    terms: set[str],
) -> set[str]:
    """
    회귀 발생 용어 탐지

    userdic으로 인해 조사 결합 시 잘못 분해되는 상위 용어(parent term)를 반환.
    기본 MeCab에서는 정상이지만 userdic에서 깨지는 케이스만 수집.
    """
    try:
        import MeCab
    except ImportError:
        print("[ERROR] MeCab Python 패키지 미설치")
        return set()

    sys_dict = _find_path(SYS_DICT_CANDIDATES)
    if not sys_dict:
        print("[ERROR] 시스템 사전 경로를 찾을 수 없습니다")
        return set()

    default_tagger = MeCab.Tagger(f"-d {sys_dict}")
    userdic_tagger = MeCab.Tagger(f"-d {sys_dict} -u {dic_path}")

    # 3자 이상 용어만 대상 (2자는 서브 용어이므로 회귀 대상이 아님)
    long_terms = sorted(t for t in terms if len(t) >= 3)
    print(f"  테스트 대상: {len(long_terms):,}개 (3자 이상)")

    # 1단계: 단독 테스트 통과하는 용어만 필터
    solo_ok: list[str] = []
    solo_regression: list[str] = []
    for term in long_terms:
        userdic_surfaces = _parse_surfaces(userdic_tagger, term)
        if term in userdic_surfaces:
            solo_ok.append(term)
        else:
            default_surfaces = _parse_surfaces(default_tagger, term)
            if term in default_surfaces:
                solo_regression.append(term)

    print(f"  단독 정상: {len(solo_ok):,}개")
    if solo_regression:
        print(f"  단독 회귀: {len(solo_regression):,}개")

    # 2단계: 조사 결합 시 회귀 탐지
    regression_parents: set[str] = set()
    regression_details: list[tuple[str, str, list[str], list[str]]] = []

    for i, term in enumerate(solo_ok):
        if (i + 1) % 500 == 0:
            print(f"  진행: {i + 1:,}/{len(solo_ok):,}")

        for particle in PARTICLES:
            text = term + particle
            default_surfaces = _parse_surfaces(default_tagger, text)
            userdic_surfaces = _parse_surfaces(userdic_tagger, text)

            default_has = term in default_surfaces
            userdic_has = term in userdic_surfaces

            if default_has and not userdic_has:
                regression_parents.add(term)
                regression_details.append(
                    (term, particle, default_surfaces, userdic_surfaces),
                )

    # 단독 회귀 용어도 추가
    regression_parents.update(solo_regression)

    print(f"\n  회귀 용어 합계: {len(regression_parents):,}개")
    print(f"  회귀 조사 쌍: {len(regression_details):,}개")

    return regression_parents


# ================================================================
# 메인
# ================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MeCab userdic 빌드 (법률 용어 사전)",
    )
    parser.add_argument("--dry-run", action="store_true", help="통계만 출력 (빌드 안 함)")
    parser.add_argument("--verify", action="store_true", help="빌드 후 검증")
    parser.add_argument("--from-json", action="store_true", help="JSON에서 용어 로드 (DB 대신)")
    parser.add_argument(
        "--fix-regression",
        action="store_true",
        help="회귀 탐지 후 해당 용어의 비용을 낮춰 재빌드",
    )
    args = parser.parse_args()

    t0 = time.time()

    # 1. 용어 로드
    print("=" * 60)
    print("MeCab userdic 빌드")
    print("=" * 60)
    print()

    if args.from_json:
        print("[1/4] JSON에서 법률 용어 로드 중...")
        terms = load_terms_from_json()
    else:
        print("[1/4] PostgreSQL에서 법률 용어 로드 중...")
        terms = asyncio.run(load_terms_from_db())

    print(f"  로드 완료: {len(terms):,}개")

    # 수동 추가 용어 병합 (manual_terms.json)
    if MANUAL_TERMS_PATH.exists():
        with open(MANUAL_TERMS_PATH, encoding="utf-8") as f:
            manual_terms = set(json.load(f))
        new_terms = manual_terms - terms
        if new_terms:
            terms.update(new_terms)
            print(f"  수동 추가: {len(new_terms):,}개 ({', '.join(sorted(new_terms))})")

    if not terms:
        print("[ERROR] 로드된 용어가 없습니다")
        sys.exit(1)

    # 통계 출력
    len_dist: dict[int, int] = {}
    for t in terms:
        length = len(t)
        len_dist[length] = len_dist.get(length, 0) + 1

    print("\n  길이 분포:")
    for length in sorted(len_dist):
        print(f"    {length}글자: {len_dist[length]:,}개")

    if args.dry_run:
        print(f"\n총 소요: {time.time() - t0:.1f}초")
        return

    # priority terms 로드 (기존 파일이 있으면)
    priority_terms: Optional[set[str]] = None
    if PRIORITY_TERMS_PATH.exists() and not args.fix_regression:
        with open(PRIORITY_TERMS_PATH, encoding="utf-8") as f:
            priority_terms = set(json.load(f))
        print(f"\n  기존 priority_terms 로드: {len(priority_terms):,}개")

    # 2. CSV 생성
    print(f"\n[2/4] userdic CSV 생성: {CSV_PATH}")
    csv_count = generate_csv(terms, CSV_PATH, priority_terms)
    print(f"  {csv_count:,}개 엔트리")

    # 3. .dic 컴파일
    print(f"\n[3/4] .dic 컴파일: {DIC_PATH}")
    if not compile_dic(CSV_PATH, DIC_PATH):
        sys.exit(1)

    # 4. 분해맵 생성
    print(f"\n[4/4] 분해맵 생성: {DECOMP_MAP_PATH}")
    vectorstore_dir = BACKEND_DIR / "app" / "tools" / "vectorstore"
    dict_mod = _load_module("legal_term_dict", vectorstore_dir / "legal_term_dict.py")
    decomp_map = build_decomposition_map(terms, dict_mod)

    with open(DECOMP_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(decomp_map, f, ensure_ascii=False, indent=2)

    print(f"  {len(decomp_map):,}개 복합어 분해 등록")

    # 요약
    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"빌드 완료 ({elapsed:.1f}초)")
    print(f"  용어:   {len(terms):,}개")
    print(f"  CSV:    {CSV_PATH}")
    print(f"  DIC:    {DIC_PATH} ({DIC_PATH.stat().st_size:,} bytes)")
    print(f"  분해맵: {DECOMP_MAP_PATH} ({len(decomp_map):,}개)")
    if priority_terms:
        print(f"  우선비용: {len(priority_terms):,}개 (cost={USERDIC_PRIORITY_COST})")
    print("=" * 60)

    # 회귀 수정 모드
    if args.fix_regression:
        print()
        print("=" * 60)
        print("회귀 탐지 및 수정")
        print("=" * 60)

        # Pass 1: 현재 userdic으로 회귀 탐지
        print("\n[Pass 1] 회귀 용어 탐지 중...")
        regression_terms = detect_regression_terms(DIC_PATH, terms)

        if not regression_terms:
            print("\n회귀 용어 없음. 수정 불필요.")
        else:
            # priority_terms.json 저장
            with open(PRIORITY_TERMS_PATH, "w", encoding="utf-8") as f:
                json.dump(sorted(regression_terms), f, ensure_ascii=False, indent=2)
            print(f"\n  {PRIORITY_TERMS_PATH} 저장: {len(regression_terms):,}개")

            # Pass 2: 낮은 비용으로 CSV 재생성 + 재컴파일
            print(f"\n[Pass 2] 우선 비용(cost={USERDIC_PRIORITY_COST}) 적용 재빌드...")
            csv_count = generate_csv(terms, CSV_PATH, regression_terms)
            print(f"  {csv_count:,}개 엔트리")

            if not compile_dic(CSV_PATH, DIC_PATH):
                sys.exit(1)

            # Pass 3: 재검증
            print("\n[Pass 3] 수정 후 회귀 재검증 중...")
            remaining = detect_regression_terms(DIC_PATH, terms)

            resolved = len(regression_terms) - len(remaining)
            print(f"\n  수정 전 회귀: {len(regression_terms):,}개")
            print(f"  수정 후 회귀: {len(remaining):,}개")
            print(f"  해소:         {resolved:,}개 ({resolved/len(regression_terms)*100:.1f}%)")

            if remaining:
                print("\n  미해소 용어 (상위 10개):")
                for t in sorted(remaining)[:10]:
                    print(f"    {t}")

        elapsed2 = time.time() - t0
        print(f"\n총 소요: {elapsed2:.1f}초")
        return

    # 검증
    if args.verify:
        print()
        sample_terms = [
            t for t in sorted(terms) if len(t) >= 3
        ][:10]
        verify_userdic(DIC_PATH, sample_terms)


if __name__ == "__main__":
    main()
