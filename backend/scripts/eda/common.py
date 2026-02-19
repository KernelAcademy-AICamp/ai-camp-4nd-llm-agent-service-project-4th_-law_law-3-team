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

# ── scripts.common 에서 JSON I/O 함수 re-export (하위 호환) ──
from scripts.common.citation import (  # noqa: F401
    extract_case_numbers,
    extract_citations,
    extract_law_names,
    extract_statute_names_plain,
)
from scripts.common.json_loader import (  # noqa: F401
    get_file_size_mb,
    smart_load,
    stream_json,
)
from scripts.common.json_loader import (
    load_json_file as load_json,
)

# ── 경로 상수 ──────────────────────────────────────────────
# 이 파일 기준: backend/scripts/eda/common.py
_THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = _THIS_DIR.parent.parent  # backend/
PROJECT_ROOT = BACKEND_DIR.parent  # law-3/
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = BACKEND_DIR / "eda_output"

# OUTPUT_DIR이 없으면 생성
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 스트리밍 사용 기준 (MB) — common.json_loader.STREAMING_THRESHOLD_MB와 동일
STREAMING_THRESHOLD_MB = 200


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
    if get_file_size_mb(path) < STREAMING_THRESHOLD_MB:
        return load_json(path)[:n]
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
