"""EDA 공유 유틸리티.

노트북에서 사용하는 스트리밍 로드, 샘플링, 파일 I/O 함수 모음.

Usage (노트북에서):
    import sys
    sys.path.insert(0, str(Path.cwd().parent.parent))
    from scripts.eda.common import smart_load, reservoir_sample, save_result
"""

from __future__ import annotations

import json
import random
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
    """DATA_DIR에서 [DONE] 또는 [Done] 접두사 JSON 파일 목록 반환.

    Returns:
        각 파일의 {name, path, size_mb} 딕셔너리 리스트
    """
    files = []
    for p in sorted(DATA_DIR.iterdir()):
        if not p.is_file():
            continue
        if not p.suffix.lower() == ".json":
            continue
        name_lower = p.name.lower()
        if not (name_lower.startswith("[done]") or name_lower.startswith("[done]")):
            continue
        files.append({
            "name": p.name,
            "path": str(p),
            "size_mb": round(get_file_size_mb(p), 2),
        })
    return files


def get_sample(
    path: Path,
    n: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """파일에서 n개 샘플 추출 (크기에 따라 전략 자동 선택).

    소용량: random.sample, 대용량: reservoir_sample
    """
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
