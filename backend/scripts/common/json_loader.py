"""JSON 로딩 유틸리티.

단일 파일/디렉토리 로드, 스트리밍, 스마트 로드, 경로 해석을 통합합니다.

Usage:
    from scripts.common.json_loader import load_items, load_json_file, smart_load
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

# 스트리밍 사용 기준 (MB)
STREAMING_THRESHOLD_MB: float = 200


def get_file_size_mb(path: Path) -> float:
    """파일 크기를 MB 단위로 반환."""
    return path.stat().st_size / (1024 * 1024)


def load_json_file(path: Path) -> list[dict[str, Any]]:
    """단일 JSON 파일 로드.

    지원 형식:
      - JSON 배열: [{"key": "val"}, ...]
      - dict with "items": {"items": [...]}
      - dict with "lawyers": {"lawyers": [...]}

    Args:
        path: JSON 파일 경로

    Returns:
        레코드 리스트
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    # dict 형식: "items", "lawyers" 등 첫 번째 리스트 값 반환
    for key in ("items", "lawyers"):
        if key in data and isinstance(data[key], list):
            return data[key]

    return [data] if isinstance(data, dict) else []


def load_json_directory(
    dir_path: Path,
    *,
    group_key: str | None = None,
) -> list[dict[str, Any]]:
    """디렉토리 내 모든 .json 파일 합산 로드.

    Args:
        dir_path: JSON 파일들이 있는 디렉토리
        group_key: 설정 시 각 item에 소스 파일명 기반 그룹 키 추가

    Returns:
        합산된 레코드 리스트

    Raises:
        FileNotFoundError: 디렉토리에 .json 파일이 없는 경우
    """
    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"디렉토리에 .json 파일이 없습니다: {dir_path}")

    all_items: list[dict[str, Any]] = []
    skipped = 0
    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning("JSON 파싱 실패, 건너뜀: %s (%s)", json_file.name, e)
            skipped += 1
            continue

        items: list[dict[str, Any]]
        if isinstance(data, list):
            items = data
        else:
            items = data.get("items", [])

        if group_key:
            group_name = _extract_group_from_filename(json_file.name)
            for item in items:
                item[group_key] = group_name

        all_items.extend(items)

    if skipped > 0:
        logger.warning("JSON 파싱 실패 파일: %d개 (건너뜀)", skipped)

    return all_items


def load_items(source_path: Path) -> list[dict[str, Any]]:
    """파일이면 load_json_file, 디렉토리면 load_json_directory.

    Args:
        source_path: JSON 파일 또는 디렉토리 경로

    Returns:
        레코드 리스트

    Raises:
        FileNotFoundError: 경로가 존재하지 않는 경우
    """
    if not source_path.exists():
        raise FileNotFoundError(f"소스를 찾을 수 없습니다: {source_path}")

    if source_path.is_dir():
        return load_json_directory(source_path)

    return load_json_file(source_path)


def stream_json(path: Path) -> Generator[dict[str, Any], None, None]:
    """ijson 기반 스트리밍 제너레이터 (대용량 파일용).

    JSON 배열의 각 아이템을 하나씩 yield합니다.

    Args:
        path: JSON 파일 경로

    Yields:
        각 JSON 아이템 (dict)
    """
    import ijson

    with open(path, "rb") as f:
        for item in ijson.items(f, "item"):
            yield item


def smart_load(
    path: Path,
    streaming_threshold_mb: float = STREAMING_THRESHOLD_MB,
) -> list[dict[str, Any]] | Generator[dict[str, Any], None, None]:
    """파일 크기에 따라 전체 로드 또는 스트리밍 제너레이터 반환.

    Args:
        path: JSON 파일 경로
        streaming_threshold_mb: 이 크기 이상이면 스트리밍 사용 (기본 200MB)

    Returns:
        소용량: list[dict], 대용량: Generator[dict]
    """
    size_mb = get_file_size_mb(path)
    if size_mb >= streaming_threshold_mb:
        return stream_json(path)
    return load_json_file(path)


def should_stream(source_path: Path) -> bool:
    """파일 크기 기반 스트리밍 여부 결정.

    Args:
        source_path: 소스 경로

    Returns:
        스트리밍 사용 여부
    """
    if source_path.is_dir():
        return False
    return get_file_size_mb(source_path) >= STREAMING_THRESHOLD_MB


def resolve_source_path(
    source_path: Path,
    data_dir: Path | None,
    default_base_dir: Path,
) -> Path:
    """소스 경로 해석 (--data-dir 재매핑 지원).

    IngestConfig의 source_path가 기본 data/ 하위를 가리킵니다.
    --data-dir로 다른 디렉토리를 지정하면 상대 경로를 재매핑합니다.
    파일명이 _v1 → _v2 등으로 바뀐 경우도 glob으로 탐색합니다.

    Args:
        source_path: config에서 가져온 기본 소스 경로
        data_dir: 사용자 지정 데이터 디렉토리 (None이면 기본값 사용)
        default_base_dir: config가 가리키는 기본 베이스 디렉토리 (DATA_DIR)

    Returns:
        해석된 소스 경로
    """
    if data_dir is None:
        return source_path

    try:
        relative = source_path.relative_to(default_base_dir)
    except ValueError:
        return source_path

    candidate = data_dir / relative
    if candidate.exists():
        return candidate

    # 파일명 버전 차이 대응: law_v1.json → law_v2.json 등
    if not candidate.is_dir():
        stem = candidate.stem
        parent = candidate.parent
        if parent.exists():
            base_stem = re.sub(r"_v\d+$", "", stem)
            matches = sorted(parent.glob(f"{base_stem}_v*.json"))
            if matches:
                return matches[-1]

    return candidate


def _extract_group_from_filename(filename: str) -> str:
    """파일명에서 그룹명 추출 (위원회/부처 이름).

    예: 'dec_comm_공정거래위원회_v3.json' → '공정거래위원회'
        'intp_min_경찰청_v2.json' → '경찰청'
    """
    m = re.search(r"(?:dec_comm|intp_min|sadm_case)_(.+?)_v\d+\.json", filename)
    return m.group(1) if m else Path(filename).stem
