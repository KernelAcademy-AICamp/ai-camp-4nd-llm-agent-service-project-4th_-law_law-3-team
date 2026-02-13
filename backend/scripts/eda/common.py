"""EDA 공유 유틸리티.

노트북에서 사용하는 스트리밍 로드, 샘플링, 파일 I/O 함수 모음.

Usage (노트북에서):
    import sys
    sys.path.insert(0, str(Path.cwd().parent.parent))
    from scripts.eda.common import smart_load, reservoir_sample, save_result
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Generator

import ijson

# ── 경로 상수 ──────────────────────────────────────────────
# 이 파일 기준: backend/scripts/eda/common.py
_THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = _THIS_DIR.parent.parent  # backend/
PROJECT_ROOT = BACKEND_DIR.parent  # law-3/
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = BACKEND_DIR / "eda_output"

# OUTPUT_DIR이 없으면 생성
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 스트리밍 사용 기준 (MB)
STREAMING_THRESHOLD_MB = 200


def get_file_size_mb(path: Path) -> float:
    """파일 크기를 MB 단위로 반환."""
    return path.stat().st_size / (1024 * 1024)


def stream_json(path: Path) -> Generator[dict[str, Any], None, None]:
    """ijson 기반 스트리밍 제너레이터 (대용량 파일용).

    JSON 배열의 각 아이템을 하나씩 yield합니다.
    """
    with open(path, "rb") as f:
        for item in ijson.items(f, "item"):
            yield item


def load_json(path: Path) -> list[dict[str, Any]]:
    """json.load로 전체 로드 (소용량 파일용)."""
    with open(path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def smart_load(
    path: Path,
    streaming_threshold_mb: float = STREAMING_THRESHOLD_MB,
) -> list[dict[str, Any]] | Generator[dict[str, Any], None, None]:
    """파일 크기에 따라 전체 로드 또는 스트리밍 제너레이터 반환.

    Args:
        path: JSON 파일 경로
        streaming_threshold_mb: 이 크기 이상이면 스트리밍 사용

    Returns:
        소용량: list[dict], 대용량: Generator[dict]
    """
    size_mb = get_file_size_mb(path)
    if size_mb >= streaming_threshold_mb:
        return stream_json(path)
    return load_json(path)


def load_all(path: Path) -> list[dict[str, Any]]:
    """파일의 모든 레코드를 리스트로 로드.

    소용량 파일은 json.load, 대용량 파일은 스트리밍 후 리스트 변환합니다.
    대용량 파일은 메모리 사용이 클 수 있으므로 주의하세요.
    """
    size_mb = get_file_size_mb(path)
    if size_mb < STREAMING_THRESHOLD_MB:
        return load_json(path)
    return list(stream_json(path))


def reservoir_sample(
    iterable: Generator[dict[str, Any], None, None] | list[dict[str, Any]],
    k: int = 10000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Reservoir sampling으로 k개 샘플 추출.

    메모리에 k개만 유지하므로 대용량 스트리밍에도 안전합니다.
    """
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []

    for i, item in enumerate(iterable):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item

    return reservoir


def head_sample(
    path: Path,
    n: int = 5000,
) -> list[dict[str, Any]]:
    """파일에서 처음 n개 레코드만 추출 (O(n) 시간).

    reservoir_sample과 달리 파일 전체를 읽지 않으므로 대용량 파일에서 훨씬 빠릅니다.
    단, 데이터가 정렬되어 있으면 편향될 수 있습니다.
    편향 없는 샘플이 필요하면 get_sample(fast=False) 또는 cached_sample()을 사용하세요.
    """
    size_mb = get_file_size_mb(path)
    if size_mb < STREAMING_THRESHOLD_MB:
        data = load_json(path)
        return data[:n]
    records: list[dict[str, Any]] = []
    for item in stream_json(path):
        records.append(item)
        if len(records) >= n:
            break
    return records


def _sample_cache_path(path: Path, n: int, seed: int) -> Path:
    """샘플 캐시 파일 경로를 생성."""
    # 파일 경로 + 크기 + n + seed로 고유 키 생성
    key = f"{path.name}:{path.stat().st_size}:{n}:{seed}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]  # noqa: S324
    return OUTPUT_DIR / f"_cache_sample_{path.stem}_{digest}.json"


def cached_sample(
    path: Path,
    n: int = 5000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """디스크 캐시 기반 reservoir sampling.

    첫 실행: reservoir sampling (전체 파일 스캔) → 결과를 JSON 캐시로 저장.
    이후 실행: 캐시에서 즉시 로드.

    무작위 샘플이 필요하면서 반복 실행 속도도 중요한 경우 사용합니다.
    """
    cache_path = _sample_cache_path(path, n, seed)

    # 캐시 히트
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    # 캐시 미스: reservoir sampling 실행
    sample = get_sample(path, n=n, seed=seed, fast=False)

    # 캐시 저장
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False)

    return sample


def count_records(path: Path) -> int:
    """전체 레코드 수 카운트 (스트리밍).

    대용량 파일에도 메모리 안전하게 동작합니다.
    """
    count = 0
    with open(path, "rb") as f:
        for _ in ijson.items(f, "item"):
            count += 1
    return count


def count_records_fast(path: Path) -> int:
    """빠른 레코드 수 카운트 (소용량 파일용).

    json.load 후 len() 사용. 200MB 미만 파일에 적합합니다.
    """
    size_mb = get_file_size_mb(path)
    if size_mb >= STREAMING_THRESHOLD_MB:
        return count_records(path)
    data = load_json(path)
    return len(data)


def save_result(name: str, data: Any) -> Path:
    """eda_output/에 JSON 결과 저장.

    Args:
        name: 파일명 (확장자 없이, 예: "phase1_inventory")
        data: JSON 직렬화 가능한 데이터

    Returns:
        저장된 파일 경로
    """
    output_path = OUTPUT_DIR / f"{name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return output_path


def load_result(name: str) -> Any:
    """이전 단계 결과 로드.

    Args:
        name: 파일명 (확장자 없이, 예: "phase1_inventory")

    Returns:
        로드된 JSON 데이터

    Raises:
        FileNotFoundError: 결과 파일이 없을 때
    """
    output_path = OUTPUT_DIR / f"{name}.json"
    if not output_path.exists():
        msg = (
            f"결과 파일 없음: {output_path}\n"
            f"이전 노트북을 먼저 실행해주세요."
        )
        raise FileNotFoundError(msg)
    with open(output_path, encoding="utf-8") as f:
        return json.load(f)


def discover_done_files() -> list[dict[str, Any]]:
    """DATA_DIR에서 EDA 대상 JSON 파일 목록 반환.

    매칭 패턴: *_v1.json, dec_*, intp_* 접두사
    (하위 호환: [DONE]/[Done] 접두사도 매칭)

    Returns:
        각 파일의 {name, path, size_mb} 딕셔너리 리스트.
        name은 DATA_DIR 기준 상대 경로 (예: "decisions_committee/dec_comm_...json").
    """
    files = []
    all_json_paths = sorted(DATA_DIR.rglob("*.json"))
    for p in all_json_paths:
        if not p.is_file():
            continue
        name_lower = p.name.lower()
        # _v2 등 데이터 폴더 구조 지원을 위한 로직 추가
        rel_path = p.relative_to(DATA_DIR)

        target_dirs = {
            "decisions_committee",
            "interpretation_ministry",
            "special_admin_appeal",
            "trial_statistics_data",
        }
        # 상위 디렉토리가 target_dirs에 포함되는지 확인
        is_in_target = (len(rel_path.parts) > 1 and rel_path.parts[0] in target_dirs)

        is_match = (
            # _v*.json 패턴 매칭 (정규식 사용)
            bool(re.search(r"_v\d+\.json$", name_lower))
            or is_in_target
        )
        if not is_match:
            continue
        # DATA_DIR 기준 상대 경로 (서브디렉토리 포함)
        rel = str(p.relative_to(DATA_DIR))
        files.append({
            "name": rel,
            "path": str(p),
            "size_mb": round(get_file_size_mb(p), 2),
        })
    return files


def get_sample(
    path: Path,
    n: int = 1000,
    seed: int = 42,
    *,
    fast: bool = False,
) -> list[dict[str, Any]]:
    """파일에서 n개 샘플 추출 (크기에 따라 전략 자동 선택).

    Args:
        path: JSON 파일 경로
        n: 샘플 수
        seed: 랜덤 시드
        fast: True이면 head sampling 사용 (전체 파일 스캔 없이 빠름, EDA 권장)

    Returns:
        소용량: random.sample, 대용량(fast=False): reservoir_sample,
        대용량(fast=True): head_sample
    """
    if fast:
        return head_sample(path, n)
    size_mb = get_file_size_mb(path)
    if size_mb < STREAMING_THRESHOLD_MB:
        data = load_json(path)
        if len(data) <= n:
            return data
        rng = random.Random(seed)
        return rng.sample(data, n)
    return reservoir_sample(stream_json(path), k=n, seed=seed)


def detect_root_type(path: Path) -> str:
    """JSON 파일의 루트 타입 감지 (array 또는 object)."""
    with open(path, "rb") as f:
        # 첫 번째 비공백 바이트 확인
        while True:
            byte = f.read(1)
            if not byte:
                return "unknown"
            if byte in (b" ", b"\n", b"\r", b"\t", b"\xef", b"\xbb", b"\xbf"):
                continue
            if byte == b"[":
                return "array"
            if byte == b"{":
                return "object"
            return "unknown"


def infer_field_types(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """레코드 목록에서 필드별 타입, null률, 존재율 추론.

    Returns:
        {필드명: {types: set, null_count, present_count, total, presence_rate}}
    """
    total = len(records)
    if total == 0:
        return {}

    field_stats: dict[str, dict[str, Any]] = {}

    for record in records:
        for key, value in record.items():
            if key not in field_stats:
                field_stats[key] = {
                    "types": set(),
                    "null_count": 0,
                    "empty_count": 0,
                    "present_count": 0,
                }

            stats = field_stats[key]
            stats["present_count"] += 1

            if value is None:
                stats["null_count"] += 1
                stats["types"].add("null")
            elif isinstance(value, str):
                stats["types"].add("string")
                if value.strip() == "":
                    stats["empty_count"] += 1
            elif isinstance(value, bool):
                stats["types"].add("boolean")
            elif isinstance(value, int):
                stats["types"].add("integer")
            elif isinstance(value, float):
                stats["types"].add("float")
            elif isinstance(value, list):
                stats["types"].add("array")
            elif isinstance(value, dict):
                stats["types"].add("object")
            else:
                stats["types"].add(type(value).__name__)

    # 비율 계산
    result: dict[str, dict[str, Any]] = {}
    for key, stats in field_stats.items():
        presence_rate = stats["present_count"] / total if total > 0 else 0
        null_rate = stats["null_count"] / stats["present_count"] if stats["present_count"] > 0 else 0
        empty_rate = stats["empty_count"] / stats["present_count"] if stats["present_count"] > 0 else 0
        result[key] = {
            "types": sorted(stats["types"]),
            "null_count": stats["null_count"],
            "empty_count": stats["empty_count"],
            "present_count": stats["present_count"],
            "total": total,
            "presence_rate": round(presence_rate, 4),
            "null_rate": round(null_rate, 4),
            "empty_rate": round(empty_rate, 4),
        }

    return result


# ── 법령 인용 추출 ────────────────────────────────────────

# 「법령명」 제N조 패턴 (가장 정확)
# 제2조의2 같은 조의N 패턴도 캡처
_CITATION_BRACKET_RE = re.compile(
    r"「([^」]+)」\s*제(\d+)\s*조(?:의(\d+))?"
)

# 법령명 제N조 패턴 (꺾쇠 없이)
_CITATION_PLAIN_RE = re.compile(
    r"((?:[가-힣]+법|[가-힣]+령|[가-힣]+규칙|[가-힣]+조례)(?:\s*시행[령규칙])?)"
    r"\s+제(\d+)\s*조(?:의(\d+))?"
)

# 법령명만 추출 (조문 번호 없이)
_LAW_NAME_RE = re.compile(
    r"「([^」]+)」"
)


def extract_citations(text: str) -> list[str]:
    """텍스트에서 법령 인용을 추출.

    「법령명」 제N조 패턴과 법령명 제N조 패턴을 모두 매칭합니다.
    중복 제거 후 발견 순서로 반환합니다.

    Args:
        text: 분석할 텍스트

    Returns:
        추출된 인용 문자열 리스트 (예: ["민법 제750조", "형법 제250조"])
    """
    if not text or not isinstance(text, str):
        return []

    seen: set[str] = set()
    citations: list[str] = []

    # 패턴 1: 「법령명」 제N조
    for match in _CITATION_BRACKET_RE.finditer(text):
        law_name = match.group(1).strip()
        article = match.group(2)
        suffix = f"의{match.group(3)}" if match.group(3) else ""
        citation = f"{law_name} 제{article}조{suffix}"
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)

    # 패턴 2: 법령명 제N조 (꺾쇠 없이)
    for match in _CITATION_PLAIN_RE.finditer(text):
        law_name = match.group(1).strip()
        article = match.group(2)
        suffix = f"의{match.group(3)}" if match.group(3) else ""
        citation = f"{law_name} 제{article}조{suffix}"
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)

    return citations


def extract_law_names(text: str) -> list[str]:
    """텍스트에서 「법령명」 패턴으로 법령명만 추출.

    Args:
        text: 분석할 텍스트

    Returns:
        추출된 법령명 리스트 (중복 제거)
    """
    if not text or not isinstance(text, str):
        return []

    seen: set[str] = set()
    names: list[str] = []

    for match in _LAW_NAME_RE.finditer(text):
        name = match.group(1).strip()
        if name not in seen:
            seen.add(name)
            names.append(name)

    return names


# ── 사건번호 추출 ─────────────────────────────────────────

# 사건번호 패턴: 연도(2-4자리) + 사건종류(한글 1-3자) + 번호
# 예: 80다268, 2023도1234, 99다12345
# build_graph.py:298 패턴과 동일
_CASE_NUMBER_RE = re.compile(
    r"(\d{2,4})"    # 연도 (2-4자리)
    r"([가-힣]{1,3})"  # 사건종류 (다, 도, 누, 카, 마 등)
    r"(\d+)"         # 번호
)


def extract_case_numbers(text: str) -> list[str]:
    """텍스트에서 사건번호 패턴을 추출.

    ``2022다12345``, ``80도268`` 등의 사건번호를 찾습니다.
    build_graph.py의 판례 인용 추출 패턴과 동일합니다.

    Args:
        text: 분석할 텍스트

    Returns:
        추출된 사건번호 리스트 (중복 제거, 발견 순서)
    """
    if not text or not isinstance(text, str):
        return []

    seen: set[str] = set()
    case_numbers: list[str] = []

    for match in _CASE_NUMBER_RE.finditer(text):
        year, case_type, number = match.groups()
        case_number = f"{year}{case_type}{number}"
        if case_number not in seen:
            seen.add(case_number)
            case_numbers.append(case_number)

    return case_numbers


# ── 꺾쇠 없는 법령명 추출 ─────────────────────────────────

# 꺾쇠(「」) 없이 본문에서 법령명 추출
# build_graph.py:244 패턴과 동일 계열
# 한글+ "법"으로 매칭, 단독 "법" 오매칭 방지 (최소 결과 2글자)
# build_graph.py:244와 동일: r"([가-힣]+법(?:시행령|시행규칙)?)"
_STATUTE_NAME_PLAIN_RE = re.compile(
    r"([가-힣]+법(?:\s*시행[령규칙])?)"
)


def extract_statute_names_plain(text: str) -> list[str]:
    """텍스트에서 꺾쇠 없는 법령명을 추출.

    ``민법``, ``형사소송법 시행령`` 등 본문에 직접 언급된 법령명을 찾습니다.
    기존 ``extract_law_names``는 「법령명」 패턴만 매칭하므로,
    꺾쇠 없이 나타나는 경우를 보완합니다.

    Args:
        text: 분석할 텍스트

    Returns:
        추출된 법령명 리스트 (중복 제거, 발견 순서)
    """
    if not text or not isinstance(text, str):
        return []

    seen: set[str] = set()
    names: list[str] = []

    for match in _STATUTE_NAME_PLAIN_RE.finditer(text):
        name = match.group(1).strip()
        if name not in seen:
            seen.add(name)
            names.append(name)

    return names
