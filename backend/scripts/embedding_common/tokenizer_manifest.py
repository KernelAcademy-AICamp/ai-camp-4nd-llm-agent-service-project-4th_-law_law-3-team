"""
content_tokenized 버전 추적 매니페스트

토크나이저 설정(userdic hash, legal_dict 용어 수, 생성 날짜)을 기록하여
content_tokenized와 검색 시 토크나이저 간 불일치를 탐지한다.

매니페스트 파일: {LANCEDB_PATH}/tokenizer_manifest.json
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "tokenizer_manifest.json"


def _compute_file_hash(path: Path) -> str:
    """파일의 SHA-256 해시 반환 (상위 16자)"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_manifest(
    *,
    legal_dict_count: int = 0,
    userdic_path: Optional[str] = None,
    decomp_map_path: Optional[str] = None,
    total_rows: int = 0,
    tokenized_rows: int = 0,
) -> dict[str, Any]:
    """토크나이저 매니페스트 딕셔너리 생성"""
    manifest: dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "legal_dict_term_count": legal_dict_count,
        "userdic_hash": None,
        "userdic_path": userdic_path,
        "decomp_map_hash": None,
        "decomp_map_path": str(decomp_map_path) if decomp_map_path else None,
        "total_rows": total_rows,
        "tokenized_rows": tokenized_rows,
    }

    if userdic_path:
        p = Path(userdic_path)
        if p.exists():
            manifest["userdic_hash"] = _compute_file_hash(p)
            manifest["userdic_size_bytes"] = p.stat().st_size

    if decomp_map_path:
        p = Path(decomp_map_path)
        if p.exists():
            manifest["decomp_map_hash"] = _compute_file_hash(p)

    return manifest


def save_manifest(lancedb_path: Path, manifest: dict[str, Any]) -> Path:
    """매니페스트를 LanceDB 데이터 디렉토리에 저장"""
    out_path = lancedb_path / MANIFEST_FILENAME
    out_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("토크나이저 매니페스트 저장: %s", out_path)
    return out_path


def load_manifest(lancedb_path: Path) -> Optional[dict[str, Any]]:
    """매니페스트 로드 (없으면 None)"""
    path = lancedb_path / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def check_manifest_consistency(
    lancedb_path: Path,
    *,
    current_legal_dict_count: int = 0,
    current_userdic_path: Optional[str] = None,
) -> tuple[bool, str]:
    """현재 토크나이저 설정과 매니페스트의 일치 여부 확인

    Returns:
        (일치 여부, 설명 메시지)
    """
    manifest = load_manifest(lancedb_path)
    if manifest is None:
        return False, "매니페스트 파일 없음 (content_tokenized 버전 추적 불가)"

    issues: list[str] = []

    # 법률 용어 수 비교
    stored_count = manifest.get("legal_dict_term_count", 0)
    if stored_count != current_legal_dict_count:
        issues.append(
            f"법률용어사전 불일치: 저장={stored_count:,}, 현재={current_legal_dict_count:,}"
        )

    # userdic 해시 비교
    if current_userdic_path:
        p = Path(current_userdic_path)
        if p.exists():
            current_hash = _compute_file_hash(p)
            stored_hash = manifest.get("userdic_hash")
            if stored_hash and current_hash != stored_hash:
                issues.append(
                    f"userdic 해시 불일치: 저장={stored_hash}, 현재={current_hash}"
                )
    elif manifest.get("userdic_hash"):
        issues.append("userdic 미사용이나 매니페스트에는 userdic 기록 존재")

    if issues:
        return False, "; ".join(issues)
    return True, "일치"
