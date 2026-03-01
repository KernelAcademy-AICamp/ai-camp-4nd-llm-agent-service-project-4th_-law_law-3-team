"""스토리보드 헬퍼 함수 단위 테스트

대상: doc_type_detector, batch_analyzer 헬퍼, timeline_merger 헬퍼
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.storyboard.service.batch_analyzer import (
    MAX_CONTEXT_LENGTH,
    _sanitize_context,
    _truncate_to_token_limit,
)
from app.modules.storyboard.service.doc_type_detector import (
    DOC_TYPE_PATTERNS,
    detect_document_type,
)
from app.modules.storyboard.service.timeline_merger import (
    _dates_overlap,
    _parse_date,
)


# ── doc_type_detector ──


class TestDetectDocumentType:
    """문서 유형 감지 테스트"""

    def test_criminal(self) -> None:
        text = "피고인 홍길동은 공소사실에 기재된 범행 일시에 범죄사실을 저질렀다"
        assert detect_document_type(text) == "criminal"

    def test_civil(self) -> None:
        text = "원고는 피고에 대하여 손해배상 청구취지 금 1000만원을 청구한다"
        assert detect_document_type(text) == "civil"

    def test_public(self) -> None:
        text = "행정처분 취소소송을 제기하며 처분청의 등록취소 처분에 대해"
        assert detect_document_type(text) == "public"

    def test_kakaotalk(self) -> None:
        text = (
            "카카오톡 대화\n"
            "저장한 날짜 : 2024년 1월 15일\n"
            "2024년 1월 15일 오후 3:42, 홍길동 : 안녕하세요"
        )
        assert detect_document_type(text) == "kakaotalk"

    def test_general_fallback(self) -> None:
        text = "오늘 날씨가 좋습니다. 산책을 나가고 싶네요."
        assert detect_document_type(text) == "general"

    def test_empty_text(self) -> None:
        assert detect_document_type("") == "general"

    def test_all_categories_have_patterns(self) -> None:
        for doc_type, patterns in DOC_TYPE_PATTERNS.items():
            assert len(patterns) > 0, f"{doc_type} 패턴이 비어있음"


# ── batch_analyzer helpers ──


class TestSanitizeContext:
    """_sanitize_context 테스트"""

    def test_empty(self) -> None:
        assert _sanitize_context("") == ""

    def test_short_context(self) -> None:
        result = _sanitize_context("간단한 컨텍스트")
        assert "간단한 컨텍스트" in result
        assert "사용자 입력 컨텍스트" in result

    def test_truncation(self) -> None:
        long_text = "가" * 1000
        result = _sanitize_context(long_text)
        # MAX_CONTEXT_LENGTH 이하로 잘려야 함
        content_part = result.split("] ")[1] if "] " in result else result
        assert len(content_part) <= MAX_CONTEXT_LENGTH

    def test_prefix_warning(self) -> None:
        result = _sanitize_context("테스트")
        assert result.startswith("[사용자 입력 컨텍스트")


class TestTruncateToTokenLimit:
    """_truncate_to_token_limit 테스트"""

    def test_empty(self) -> None:
        assert _truncate_to_token_limit("") == ""

    def test_short_text_unchanged(self) -> None:
        text = "짧은 텍스트"
        assert _truncate_to_token_limit(text) == text

    def test_long_text_truncated(self) -> None:
        # 매우 긴 텍스트 생성 (토큰 초과 보장)
        long_text = "법률 상담 내용입니다. " * 10000
        result = _truncate_to_token_limit(long_text, max_tokens=100)
        assert len(result) < len(long_text)

    def test_custom_max_tokens(self) -> None:
        text = "Hello world " * 100
        result = _truncate_to_token_limit(text, max_tokens=10)
        assert len(result) < len(text)


# ── timeline_merger helpers ──


class TestParseDate:
    """_parse_date 테스트"""

    def test_yyyy_mm_dd_dash(self) -> None:
        dt = _parse_date("2024-01-15")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_yyyy_mm_dd_dot(self) -> None:
        dt = _parse_date("2024.03.20")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 20

    def test_yyyy_mm_dd_slash(self) -> None:
        dt = _parse_date("2024/06/01")
        assert dt is not None
        assert dt.month == 6

    def test_yyyy_mm_only(self) -> None:
        dt = _parse_date("2024-03")
        assert dt is not None
        assert dt.day == 15  # 일자 없으면 15일 기본값

    def test_unknown_date(self) -> None:
        assert _parse_date("날짜 미상") is None

    def test_empty(self) -> None:
        assert _parse_date("") is None

    def test_invalid_date(self) -> None:
        assert _parse_date("2024-13-32") is None


class TestDatesOverlap:
    """_dates_overlap 테스트"""

    def test_same_date(self) -> None:
        assert _dates_overlap("2024-01-15", "2024-01-15") is True

    def test_within_range(self) -> None:
        # 30일 이내
        assert _dates_overlap("2024-01-01", "2024-01-25") is True

    def test_outside_range(self) -> None:
        # 30일 초과
        assert _dates_overlap("2024-01-01", "2024-06-01") is False

    def test_one_unknown(self) -> None:
        assert _dates_overlap("날짜 미상", "2024-01-01") is False

    def test_both_unknown(self) -> None:
        assert _dates_overlap("날짜 미상", "날짜 미상") is False

    def test_boundary_30_days(self) -> None:
        # 정확히 30일 차이 → True (<=)
        assert _dates_overlap("2024-01-01", "2024-01-31") is True

    def test_boundary_31_days(self) -> None:
        # 31일 차이 → False
        assert _dates_overlap("2024-01-01", "2024-02-01") is False
