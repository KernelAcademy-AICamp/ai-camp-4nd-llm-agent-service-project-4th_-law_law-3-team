#!/usr/bin/env python3
"""
들어오는 법률 데이터 파일을 프로젝트 네이밍 규칙에 맞게 변환하는 스크립트.

네이밍 규칙:
  - 단일 파일: <타입>_v<버전>.json  (예: law_v3.json, precedents_v2.json)
  - 부처 해석례: intp_min_<부처명>_v<버전>.json
  - 위원회 결정문: dec_comm_<위원회명>_v<버전>.json
  - 특별행정심판: sadm_case_<기관명>_v<버전>.json

사용법:
    cd backend

    # data/incoming/ 의 모든 파일 미리보기 (기본: dry-run)
    uv run python scripts/rename_incoming_data.py

    # 실제 복사 실행
    uv run python scripts/rename_incoming_data.py --execute

    # 복사 대신 이동
    uv run python scripts/rename_incoming_data.py --execute --move

    # 특정 파일만 처리
    uv run python scripts/rename_incoming_data.py --file ppc.json --execute

    # 매핑 테이블 전체 출력
    uv run python scripts/rename_incoming_data.py --show-mapping
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
INCOMING_DIR = DATA_DIR / "incoming"


@dataclass(frozen=True)
class FileMapping:
    """파일 매핑 정보

    Attributes:
        target_prefix: 대상 파일명 접두사 (예: "precedents", "dec_comm_개인정보보호위원회")
        target_subdir: data/ 하위 서브디렉토리 (빈 문자열이면 data/ 루트)
        korean_name: 한국어 데이터명 (리포트용)
    """

    target_prefix: str
    target_subdir: str
    korean_name: str


# ──────────────────────────────────────────────
# 매핑 테이블
# 키: 들어오는 파일의 normalized name (접미사 제거 후)
# 값: FileMapping(대상 접두사, 서브디렉토리, 한국어명)
#
# 새 데이터 타입 추가 시 여기에 항목을 추가하면 됩니다.
# ──────────────────────────────────────────────
MAPPING: dict[str, FileMapping] = {
    # ─── 단일 파일 (data/) ───
    "precedents": FileMapping("precedents", "", "판례"),
    "administration": FileMapping("administration", "", "행정심판례"),
    "constitutional": FileMapping("constitutional", "", "헌재결정례"),
    "legislation": FileMapping("legislation", "", "법령해석례"),
    "law": FileMapping("law", "", "법령"),
    "law_full": FileMapping("law", "", "법령 (전체)"),
    "administrative_rules": FileMapping("admin_rule", "", "행정규칙"),
    "treaty": FileMapping("treaty", "", "조약"),
    "lawterms": FileMapping("lawterms", "", "법정 용어"),
    "local_rules": FileMapping("local_rules", "", "자치법규"),
    "school": FileMapping("school", "", "학칙공단"),
    # ─── 위원회 결정문 (data/decisions_committee/) ───
    "ppc": FileMapping(
        "dec_comm_개인정보보호위원회", "decisions_committee", "개인정보보호위원회 결정문"
    ),
    "eiac": FileMapping(
        "dec_comm_고용보험심사위원회", "decisions_committee", "고용보험심사위원회 결정문"
    ),
    "ftc": FileMapping(
        "dec_comm_공정거래위원회", "decisions_committee", "공정거래위원회 결정문"
    ),
    "acr": FileMapping(
        "dec_comm_국민권익위원회", "decisions_committee", "국민권익위원회 결정문"
    ),
    "fsc": FileMapping(
        "dec_comm_금융위원회", "decisions_committee", "금융위원회 결정문"
    ),
    "nlrc": FileMapping(
        "dec_comm_노동위원회", "decisions_committee", "노동위원회 결정문"
    ),
    "kcc": FileMapping(
        "dec_comm_방송미디어통신위원회",
        "decisions_committee",
        "방송미디어통신위원회 결정문",
    ),
    "iaciac": FileMapping(
        "dec_comm_산업재해보상위험재심사위원회",
        "decisions_committee",
        "산업재해보상위험재심사위원회 결정문",
    ),
    "ecc": FileMapping(
        "dec_comm_중앙환경분쟁조정위원회",
        "decisions_committee",
        "중앙환경분쟁조정위원회 결정문",
    ),
    "sfc": FileMapping(
        "dec_comm_증권선물위원회", "decisions_committee", "증권선물위원회 결정문"
    ),
    "nhrck": FileMapping(
        "dec_comm_국가인권위원회", "decisions_committee", "국가인권위원회 결정문"
    ),
    # ─── 부처별 해석례 (data/interpretation_ministry/) ───
    "moelCgmExpc": FileMapping(
        "intp_min_고용노동부", "interpretation_ministry", "고용노동부 법령 해석"
    ),
    "molitCgmExpc": FileMapping(
        "intp_min_국토교통부", "interpretation_ministry", "국토교통부 법령 해석"
    ),
    "mofCgmExpc": FileMapping(
        "intp_min_해양수산부", "interpretation_ministry", "해양수산부 법령 해석"
    ),
    "moisCgmExpc": FileMapping(
        "intp_min_행정안전부", "interpretation_ministry", "행정안전부 법령 해석"
    ),
    "meCgmExpc": FileMapping(
        "intp_min_기후에너지환경부", "interpretation_ministry", "기후에너지환경부 법령 해석"
    ),
    "kcsCgmExpc": FileMapping(
        "intp_min_관세청", "interpretation_ministry", "관세청 법령 해석"
    ),
    "moeCgmExpc": FileMapping(
        "intp_min_교육부", "interpretation_ministry", "교육부 법령 해석"
    ),
    "msitCgmExpc": FileMapping(
        "intp_min_과학기술정보통신부",
        "interpretation_ministry",
        "과학기술정보통신부 법령 해석",
    ),
    "mpvaCgmExpc": FileMapping(
        "intp_min_국가보훈부", "interpretation_ministry", "국가보훈부 법령 해석"
    ),
    "mndCgmExpc": FileMapping(
        "intp_min_국방부", "interpretation_ministry", "국방부 법령 해석"
    ),
    "mafraCgmExpc": FileMapping(
        "intp_min_농림축산식품부", "interpretation_ministry", "농림축산식품부 법령 해석"
    ),
    "mcstCgmExpc": FileMapping(
        "intp_min_문화체육관광부", "interpretation_ministry", "문화체육관광부 법령 해석"
    ),
    "mojCgmExpc": FileMapping(
        "intp_min_법무부", "interpretation_ministry", "법무부 법령 해석"
    ),
    "mohwCgmExpc": FileMapping(
        "intp_min_보건복지부", "interpretation_ministry", "보건복지부 법령 해석"
    ),
    "motieCgmExpc": FileMapping(
        "intp_min_산업통상자원부", "interpretation_ministry", "산업통상자원부 법령 해석"
    ),
    "mogefCgmExpc": FileMapping(
        "intp_min_성평등가족부", "interpretation_ministry", "성평등가족부 법령 해석"
    ),
    "mofaCgmExpc": FileMapping(
        "intp_min_외교부", "interpretation_ministry", "외교부 법령 해석"
    ),
    "mssCgmExpc": FileMapping(
        "intp_min_중소벤처기업부", "interpretation_ministry", "중소벤처기업부 법령 해석"
    ),
    "mouCgmExpc": FileMapping(
        "intp_min_통일부", "interpretation_ministry", "통일부 법령 해석"
    ),
    "molegCgmExpc": FileMapping(
        "intp_min_법제처", "interpretation_ministry", "법제처 법령 해석"
    ),
    "mfdsCgmExpc": FileMapping(
        "intp_min_식품의약품안전처", "interpretation_ministry", "식품의약품안전처 법령 해석"
    ),
    "mpmCgmExpc": FileMapping(
        "intp_min_인사혁신처", "interpretation_ministry", "인사혁신처 법령 해석"
    ),
    "kmaCgmExpc": FileMapping(
        "intp_min_기상청", "interpretation_ministry", "기상청 법령 해석"
    ),
    "khsCgmExpc": FileMapping(
        "intp_min_국가유산청", "interpretation_ministry", "국가유산청 법령 해석"
    ),
    "rdaCgmExpc": FileMapping(
        "intp_min_농촌진흥청", "interpretation_ministry", "농촌진흥청 법령 해석"
    ),
    "npaCgmExpc": FileMapping(
        "intp_min_경찰청", "interpretation_ministry", "경찰청 법령 해석"
    ),
    "dapaCgmExpc": FileMapping(
        "intp_min_방위사업청", "interpretation_ministry", "방위사업청 법령 해석"
    ),
    "mmaCgmExpc": FileMapping(
        "intp_min_병무청", "interpretation_ministry", "병무청 법령 해석"
    ),
    "kfsCgmExpc": FileMapping(
        "intp_min_산림청", "interpretation_ministry", "산림청 법령 해석"
    ),
    "nfaCgmExpc": FileMapping(
        "intp_min_소방청", "interpretation_ministry", "소방청 법령 해석"
    ),
    "okaCgmExpc": FileMapping(
        "intp_min_재외동포청", "interpretation_ministry", "재외동포청 법령 해석"
    ),
    "ppsCgmExpc": FileMapping(
        "intp_min_조달청", "interpretation_ministry", "조달청 법령 해석"
    ),
    "kdcaCgmExpc": FileMapping(
        "intp_min_질병관리청", "interpretation_ministry", "질병관리청 법령 해석"
    ),
    "kostatCgmExpc": FileMapping(
        "intp_min_국가데이터처", "interpretation_ministry", "국가데이터처 법령 해석"
    ),
    "kipoCgmExpc": FileMapping(
        "intp_min_지식재산처", "interpretation_ministry", "지식재산처 법령 해석"
    ),
    "kcgCgmExpc": FileMapping(
        "intp_min_해양경찰청", "interpretation_ministry", "해양경찰청 법령 해석"
    ),
    "naaccCgmExpc": FileMapping(
        "intp_min_행정중심복합도시건설청",
        "interpretation_ministry",
        "행정중심복합도시건설청 법령 해석",
    ),
    # ─── 특별행정심판례 (data/special_admin_appeal/) ───
    "ttSpecialDecc": FileMapping(
        "sadm_case_조세심판원", "special_admin_appeal", "조세심판원 특별행정심판례"
    ),
    "kmstSpecialDecc": FileMapping(
        "sadm_case_해양안전심판원", "special_admin_appeal", "해양안전심판원 특별행정심판례"
    ),
    "acrSpecialDecc": FileMapping(
        "sadm_case_국민권익위원회", "special_admin_appeal", "국민권익위원회 특별행정심판례"
    ),
    "adapSpecialDecc": FileMapping(
        "sadm_case_인사혁신처",
        "special_admin_appeal",
        "인사혁신처 소청심사위원 특별행정심판례",
    ),
}


# ──────────────────────────────────────────────
# 핵심 함수
# ──────────────────────────────────────────────


def find_mapping_key(filename: str) -> str | None:
    """파일명에서 접미사를 단계적으로 제거하며 매핑 키를 탐색.

    탐색 순서:
      1. 숫자 접미사 제거 (-5, -28 등) 후 매핑 확인
      2. -full 접미사 추가 제거 후 매핑 확인
      3. _full 접미사 추가 제거 후 매핑 확인

    예:
      precedents-5.json    → "precedents"
      fsc-full.json        → "fsc"
      law_full-30.json     → "law_full"
      adapSpecialDecc_full.json → "adapSpecialDecc"
    """
    stem = Path(filename).stem

    # [DONE] 접두사 제거 (혹시 파일명에 포함된 경우)
    if stem.startswith("[DONE]"):
        stem = stem[6:]

    # 1단계: 숫자 접미사 제거 (-5, -28, -30 등)
    normalized = re.sub(r"-\d+$", "", stem)
    if normalized in MAPPING:
        return normalized

    # 2단계: -full 접미사 제거
    without_dash_full = re.sub(r"-full$", "", normalized)
    if without_dash_full != normalized and without_dash_full in MAPPING:
        return without_dash_full

    # 3단계: _full 접미사 제거
    without_under_full = re.sub(r"_full$", "", normalized)
    if without_under_full != normalized and without_under_full in MAPPING:
        return without_under_full

    return None


def detect_max_version(target_dir: Path, prefix: str) -> int:
    """대상 디렉토리에서 {prefix}_v{N}.json 패턴의 최대 버전 번호를 반환.

    파일이 없으면 0을 반환합니다.

    Note:
        macOS APFS에서 한글 파일명이 NFD로 저장될 수 있어서
        NFC 정규화 후 비교합니다.
    """
    if not target_dir.exists():
        return 0

    nfc_prefix = unicodedata.normalize("NFC", prefix)
    pattern = re.compile(rf"^{re.escape(nfc_prefix)}_v(\d+)\.json$")
    max_version = 0

    for filepath in target_dir.iterdir():
        # macOS NFD → NFC 정규화
        normalized_name = unicodedata.normalize("NFC", filepath.name)
        match = pattern.match(normalized_name)
        if match:
            version = int(match.group(1))
            max_version = max(max_version, version)

    return max_version


@dataclass
class RenameResult:
    """개별 파일 처리 결과"""

    source: Path
    target: Path
    mapping_key: str
    korean_name: str
    old_version: int
    new_version: int
    success: bool = False
    error: str = ""


def process_file(
    source_path: Path,
    *,
    dry_run: bool = True,
    move: bool = False,
) -> RenameResult:
    """단일 파일을 처리하여 대상 경로로 복사/이동.

    Args:
        source_path: 입력 파일 경로
        dry_run: True이면 실제 파일 작업 없이 결과만 반환
        move: True이면 복사 대신 이동
    """
    filename = source_path.name
    mapping_key = find_mapping_key(filename)

    if mapping_key is None:
        return RenameResult(
            source=source_path,
            target=Path(""),
            mapping_key="",
            korean_name="",
            old_version=0,
            new_version=0,
            error=f"매핑 없음: {filename}",
        )

    mapping = MAPPING[mapping_key]

    # 대상 디렉토리 결정
    if mapping.target_subdir:
        target_dir = DATA_DIR / mapping.target_subdir
    else:
        target_dir = DATA_DIR

    # 현재 최대 버전 감지 → 새 버전 = 최대 + 1
    old_version = detect_max_version(target_dir, mapping.target_prefix)
    new_version = old_version + 1

    # 대상 파일 경로
    target_filename = f"{mapping.target_prefix}_v{new_version}.json"
    target_path = target_dir / target_filename

    result = RenameResult(
        source=source_path,
        target=target_path,
        mapping_key=mapping_key,
        korean_name=mapping.korean_name,
        old_version=old_version,
        new_version=new_version,
    )

    if dry_run:
        result.success = True
        return result

    # 실제 파일 작업
    target_dir.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        result.error = f"대상 파일이 이미 존재: {target_path}"
        return result

    if move:
        shutil.move(str(source_path), str(target_path))
    else:
        shutil.copy2(str(source_path), str(target_path))

    result.success = True
    return result


def scan_incoming(incoming_dir: Path) -> list[Path]:
    """incoming 디렉토리에서 JSON 파일 목록을 반환."""
    if not incoming_dir.exists():
        return []
    return sorted(
        f for f in incoming_dir.iterdir() if f.suffix == ".json" and f.is_file()
    )


# ──────────────────────────────────────────────
# 리포트 출력
# ──────────────────────────────────────────────

# ANSI 색상 (터미널 출력용)
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_report(results: list[RenameResult], *, dry_run: bool, move: bool) -> None:
    """처리 결과 리포트 출력."""
    action = "이동" if move else "복사"
    mode = f"{YELLOW}[DRY-RUN 미리보기]{RESET}" if dry_run else f"{GREEN}[실행 완료]{RESET}"

    print(f"\n{'='*70}")
    print(f"  법률 데이터 파일명 변환 결과 {mode}")
    print(f"{'='*70}\n")

    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    if successes:
        print(f"{GREEN}--- 성공 ({len(successes)}건) ---{RESET}\n")
        for r in successes:
            version_info = (
                f"v{r.old_version} → v{r.new_version}"
                if r.old_version > 0
                else f"v{r.new_version} (신규)"
            )
            rel_source = r.source.name
            rel_target = r.target.relative_to(DATA_DIR)
            print(f"  {CYAN}{r.korean_name}{RESET}")
            print(f"    {DIM}{rel_source}{RESET} → {BOLD}{rel_target}{RESET}  [{version_info}]")
            print()

    if failures:
        print(f"{RED}--- 실패 ({len(failures)}건) ---{RESET}\n")
        for r in failures:
            print(f"  {RED}✗{RESET} {r.source.name}: {r.error}")
        print()

    # 요약
    print(f"{'─'*70}")
    print(f"  총 {len(results)}건 | {GREEN}성공 {len(successes)}{RESET} | {RED}실패 {len(failures)}{RESET}")
    if dry_run:
        print(f"\n  {YELLOW}미리보기 모드입니다. 실제 {action}하려면 --execute 옵션을 추가하세요.{RESET}")
    print(f"{'─'*70}\n")


def print_mapping_table() -> None:
    """전체 매핑 테이블 출력."""
    print(f"\n{'='*90}")
    print(f"  파일명 매핑 테이블 ({len(MAPPING)}개 항목)")
    print(f"{'='*90}\n")

    # 카테고리별 그룹핑
    categories: dict[str, list[tuple[str, FileMapping]]] = {
        "단일 파일 (data/)": [],
        "위원회 결정문 (data/decisions_committee/)": [],
        "부처별 해석례 (data/interpretation_ministry/)": [],
        "특별행정심판례 (data/special_admin_appeal/)": [],
    }

    for key, m in MAPPING.items():
        if m.target_subdir == "decisions_committee":
            categories["위원회 결정문 (data/decisions_committee/)"].append((key, m))
        elif m.target_subdir == "interpretation_ministry":
            categories["부처별 해석례 (data/interpretation_ministry/)"].append((key, m))
        elif m.target_subdir == "special_admin_appeal":
            categories["특별행정심판례 (data/special_admin_appeal/)"].append((key, m))
        else:
            categories["단일 파일 (data/)"].append((key, m))

    for category_name, items in categories.items():
        if not items:
            continue
        print(f"  {BOLD}{category_name}{RESET}")
        print(f"  {'─'*86}")
        print(f"  {'들어오는 키':<30} {'대상 파일명':<40} {'한국어명'}")
        print(f"  {'─'*86}")
        for key, m in items:
            target = f"{m.target_prefix}_v{{N}}.json"
            print(f"  {key:<30} {target:<40} {m.korean_name}")
        print()

    print(f"  {DIM}접미사 자동 제거: -숫자 (-5, -28), -full, _full{RESET}")
    print(f"  {DIM}예: precedents-5.json → 키 'precedents' → precedents_v{{N}}.json{RESET}")
    print(f"  {DIM}예: fsc-full.json     → 키 'fsc'        → dec_comm_금융위원회_v{{N}}.json{RESET}\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="들어오는 법률 데이터 파일을 프로젝트 네이밍 규칙에 맞게 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  uv run python scripts/rename_incoming_data.py                  # 미리보기
  uv run python scripts/rename_incoming_data.py --execute        # 실제 복사
  uv run python scripts/rename_incoming_data.py --execute --move # 실제 이동
  uv run python scripts/rename_incoming_data.py --show-mapping   # 매핑 테이블 출력
  uv run python scripts/rename_incoming_data.py --file ppc.json  # 특정 파일만
        """,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="실제 파일 작업 실행 (기본: dry-run 미리보기)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="복사 대신 이동 (--execute와 함께 사용)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="특정 파일만 처리 (data/incoming/ 기준 파일명 또는 절대경로)",
    )
    parser.add_argument(
        "--incoming-dir",
        type=str,
        default=str(INCOMING_DIR),
        help=f"입력 디렉토리 경로 (기본: {INCOMING_DIR})",
    )
    parser.add_argument(
        "--show-mapping",
        action="store_true",
        help="전체 매핑 테이블 출력 후 종료",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # 매핑 테이블 출력
    if args.show_mapping:
        print_mapping_table()
        return 0

    dry_run = not args.execute
    incoming_dir = Path(args.incoming_dir)

    # 처리할 파일 목록 결정
    if args.file:
        file_path = Path(args.file)
        # 상대 경로면 incoming_dir 기준으로 해석
        if not file_path.is_absolute():
            file_path = incoming_dir / file_path
        if not file_path.exists():
            print(f"{RED}파일을 찾을 수 없습니다: {file_path}{RESET}")
            return 1
        files = [file_path]
    else:
        files = scan_incoming(incoming_dir)
        if not files:
            print(f"\n{YELLOW}data/incoming/ 디렉토리에 JSON 파일이 없습니다.{RESET}")
            print(f"  처리할 파일을 {incoming_dir}/ 에 넣어주세요.\n")
            return 0

    # 파일 처리
    results: list[RenameResult] = []
    for filepath in files:
        result = process_file(filepath, dry_run=dry_run, move=args.move)
        results.append(result)

    # 리포트 출력
    print_report(results, dry_run=dry_run, move=args.move)

    # 실패가 있으면 exit code 1
    failures = [r for r in results if not r.success]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
