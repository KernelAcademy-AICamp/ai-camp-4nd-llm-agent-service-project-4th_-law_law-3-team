"""변호사시험 기록형 연습 - 파일 읽기 서비스"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

# 프로젝트 루트 기준 data/bar_exam_raw/
# __file__ → service/__init__.py → law_study/ → modules/ → app/ → backend/ → project root
DATA_DIR = Path(__file__).resolve().parents[5] / "data" / "bar_exam_raw"

CATEGORY_LABELS: dict[str, str] = {
    "CIVIL": "민사법",
    "CRIMINAL": "형사법",
    "PUBLIC": "공법",
}

# 파일명 패턴: [CATEGORY]N.md
_FILENAME_PATTERN = re.compile(r"^\[([A-Z]+)\](\d+)\.md$")


def _parse_filename(filename: str) -> tuple[str, int] | None:
    """파일명에서 (category, session) 추출. 실패 시 None."""
    m = _FILENAME_PATTERN.match(filename)
    if not m:
        return None
    return m.group(1), int(m.group(2))


@lru_cache(maxsize=1)
def list_exam_files() -> list[dict[str, Any]]:
    """시험 문제 파일 목록 반환. 결과를 캐싱."""
    if not DATA_DIR.is_dir():
        return []

    exams: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        parsed = _parse_filename(path.name)
        if not parsed:
            continue
        category, session = parsed
        year = 2011 + session
        label = CATEGORY_LABELS.get(category, category)
        exams.append(
            {
                "category": category,
                "year": year,
                "session": session,
                "filename": path.name,
                "title": f"제{session}회 변호사시험 {label} 기록형 ({year}년)",
            }
        )
    return exams


def get_exam_content(category: str, session: int) -> str | None:
    """시험 문제 전문 읽기. 파일 없으면 None."""
    filename = f"[{category.upper()}]{session}.md"
    filepath = DATA_DIR / filename
    if not filepath.is_file():
        return None
    return filepath.read_text(encoding="utf-8")
