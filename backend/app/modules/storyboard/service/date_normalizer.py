"""한국 법률 문서 날짜 정규화 유틸리티

한국 법률 문서에서 사용되는 다양한 날짜 표현을 정렬 가능한 형식으로 정규화한다.

입력 예시:
    "2012. 1. 5."     → ("2012-01-05", "exact")
    "2010. 5."         → ("2010-05", "month")
    "1995년경"          → ("1995", "approximate")
    "2000년 여름"       → ("2000-07", "approximate")
    "2000년 초"         → ("2000-02", "approximate")
    "2000년 말"         → ("2000-11", "approximate")
"""

import re
from dataclasses import dataclass

# 계절/시기별 정렬용 중간값 매핑 (월)
_SEASON_MONTH_MAP: dict[str, str] = {
    "초": "02",
    "초순": "02",
    "상반기": "04",
    "봄": "04",
    "여름": "07",
    "중순": "06",
    "하반기": "09",
    "가을": "10",
    "말": "11",
    "하순": "11",
    "겨울": "01",
}


@dataclass(frozen=True)
class NormalizedDate:
    """정규화된 날짜 결과"""

    normalized: str  # 정렬용 정규화 값 (예: "2012-01-05")
    precision: str  # exact, month, year, approximate, relative
    raw: str  # 원본 표현


# YYYY. M. D. 패턴 (선택적 시간 포함)
_PATTERN_YMD_DOT = re.compile(
    r"(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?"
)

# YYYY년 M월 D일 패턴
_PATTERN_YMD_KR = re.compile(
    r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"
)

# YYYY. M. 패턴 (일 없음)
_PATTERN_YM_DOT = re.compile(
    r"(\d{4})\s*\.\s*(\d{1,2})\s*\.(?!\s*\d)"
)

# YYYY년 M월 패턴 (일 없음)
_PATTERN_YM_KR = re.compile(
    r"(\d{4})\s*년\s*(\d{1,2})\s*월(?!\s*\d)"
)

# YYYY년경 패턴
_PATTERN_YEAR_APPROX = re.compile(r"(\d{4})\s*년\s*경")

# YYYY년 + 계절/시기 패턴
_PATTERN_YEAR_SEASON = re.compile(
    r"(\d{4})\s*년\s*(초|초순|상반기|봄|여름|중순|하반기|가을|말|하순|겨울)"
)

# YYYY년 패턴 (단독)
_PATTERN_YEAR_ONLY = re.compile(r"(\d{4})\s*년(?!\s*경|\s*\d|\s*[초상봄여중하가말겨])")


def normalize_date(raw: str) -> NormalizedDate:
    """날짜 표현을 정규화한다.

    Args:
        raw: 원본 날짜 표현

    Returns:
        NormalizedDate 객체 (normalized, precision, raw)
    """
    text = raw.strip()

    # 1. YYYY. M. D. 또는 YYYY년 M월 D일
    match = _PATTERN_YMD_DOT.search(text)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3)
        return NormalizedDate(
            normalized=f"{year}-{month.zfill(2)}-{day.zfill(2)}",
            precision="exact",
            raw=raw,
        )

    match = _PATTERN_YMD_KR.search(text)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3)
        return NormalizedDate(
            normalized=f"{year}-{month.zfill(2)}-{day.zfill(2)}",
            precision="exact",
            raw=raw,
        )

    # 2. YYYY년경
    match = _PATTERN_YEAR_APPROX.search(text)
    if match:
        year = match.group(1)
        return NormalizedDate(
            normalized=year,
            precision="approximate",
            raw=raw,
        )

    # 3. YYYY년 + 계절/시기
    match = _PATTERN_YEAR_SEASON.search(text)
    if match:
        year = match.group(1)
        season = match.group(2)
        month = _SEASON_MONTH_MAP.get(season, "06")
        return NormalizedDate(
            normalized=f"{year}-{month}",
            precision="approximate",
            raw=raw,
        )

    # 4. YYYY. M. (일 없음)
    match = _PATTERN_YM_DOT.search(text)
    if match:
        year, month = match.group(1), match.group(2)
        return NormalizedDate(
            normalized=f"{year}-{month.zfill(2)}",
            precision="month",
            raw=raw,
        )

    match = _PATTERN_YM_KR.search(text)
    if match:
        year, month = match.group(1), match.group(2)
        return NormalizedDate(
            normalized=f"{year}-{month.zfill(2)}",
            precision="month",
            raw=raw,
        )

    # 5. YYYY년 (단독)
    match = _PATTERN_YEAR_ONLY.search(text)
    if match:
        year = match.group(1)
        return NormalizedDate(
            normalized=year,
            precision="year",
            raw=raw,
        )

    # 6. 4자리 연도만 있는 경우
    match = re.search(r"(\d{4})", text)
    if match:
        return NormalizedDate(
            normalized=match.group(1),
            precision="approximate",
            raw=raw,
        )

    # 7. 상대적 표현 (그 뒤에도, 며칠 후 등)
    return NormalizedDate(
        normalized="",
        precision="relative",
        raw=raw,
    )


def sort_key_from_normalized(normalized: str) -> str:
    """정규화된 날짜로부터 정렬 키를 생성한다.

    빈 문자열(relative)은 마지막에 정렬되도록 큰 값을 반환한다.
    연도만 있으면 -06(중간)을 추가하여 해당 연도 중간에 위치시킨다.

    Args:
        normalized: normalize_date()의 결과 normalized 필드

    Returns:
        정렬 가능한 문자열 키
    """
    if not normalized:
        return "9999-99-99"

    # "2012" → "2012-06-15" (연도만, 중간값)
    if len(normalized) == 4:
        return f"{normalized}-06-15"

    # "2012-01" → "2012-01-15" (월만, 중간값)
    if len(normalized) == 7:
        return f"{normalized}-15"

    # "2012-01-05" → 그대로
    return normalized
