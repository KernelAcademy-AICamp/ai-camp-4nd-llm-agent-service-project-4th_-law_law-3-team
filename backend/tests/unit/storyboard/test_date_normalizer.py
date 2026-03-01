"""날짜 정규화 유틸리티 단위 테스트"""

import pytest

from app.modules.storyboard.service.date_normalizer import (
    NormalizedDate,
    normalize_date,
    sort_key_from_normalized,
)


class TestNormalizeDateExact:
    """YYYY. M. D. 및 YYYY년 M월 D일 패턴"""

    def test_dot_format_full(self) -> None:
        result = normalize_date("2012. 1. 5.")
        assert result.normalized == "2012-01-05"
        assert result.precision == "exact"

    def test_dot_format_no_trailing_dot(self) -> None:
        result = normalize_date("2012. 1. 5")
        assert result.normalized == "2012-01-05"
        assert result.precision == "exact"

    def test_dot_format_two_digit(self) -> None:
        result = normalize_date("2008. 10. 24.")
        assert result.normalized == "2008-10-24"
        assert result.precision == "exact"

    def test_korean_format(self) -> None:
        result = normalize_date("2012년 1월 5일")
        assert result.normalized == "2012-01-05"
        assert result.precision == "exact"

    def test_preserves_raw(self) -> None:
        raw = "  2012. 1. 5.  "
        result = normalize_date(raw)
        assert result.raw == raw


class TestNormalizeDateMonth:
    """YYYY. M. 및 YYYY년 M월 패턴"""

    def test_dot_format_month(self) -> None:
        result = normalize_date("2010. 5.")
        assert result.normalized == "2010-05"
        assert result.precision == "month"

    def test_korean_format_month(self) -> None:
        result = normalize_date("2010년 5월")
        assert result.normalized == "2010-05"
        assert result.precision == "month"


class TestNormalizeDateApproximate:
    """년경, 계절 등 근사 패턴"""

    def test_year_approx(self) -> None:
        result = normalize_date("1995년경")
        assert result.normalized == "1995"
        assert result.precision == "approximate"

    def test_summer(self) -> None:
        result = normalize_date("2000년 여름")
        assert result.normalized == "2000-07"
        assert result.precision == "approximate"

    def test_early(self) -> None:
        result = normalize_date("2000년 초")
        assert result.normalized == "2000-02"
        assert result.precision == "approximate"

    def test_late(self) -> None:
        result = normalize_date("2000년 말")
        assert result.normalized == "2000-11"
        assert result.precision == "approximate"

    def test_spring(self) -> None:
        result = normalize_date("2005년 봄")
        assert result.normalized == "2005-04"
        assert result.precision == "approximate"

    def test_fall(self) -> None:
        result = normalize_date("2010년 가을")
        assert result.normalized == "2010-10"
        assert result.precision == "approximate"

    def test_winter(self) -> None:
        result = normalize_date("2010년 겨울")
        assert result.normalized == "2010-01"
        assert result.precision == "approximate"

    def test_first_half(self) -> None:
        result = normalize_date("2015년 상반기")
        assert result.normalized == "2015-04"
        assert result.precision == "approximate"

    def test_second_half(self) -> None:
        result = normalize_date("2015년 하반기")
        assert result.normalized == "2015-09"
        assert result.precision == "approximate"


class TestNormalizeDateYear:
    """연도만 있는 경우"""

    def test_year_only(self) -> None:
        result = normalize_date("2005년")
        assert result.normalized == "2005"
        assert result.precision == "year"


class TestNormalizeDateRelative:
    """상대적 표현"""

    def test_relative_expression(self) -> None:
        result = normalize_date("그 뒤에도")
        assert result.normalized == ""
        assert result.precision == "relative"

    def test_relative_few_days(self) -> None:
        result = normalize_date("며칠 후")
        assert result.normalized == ""
        assert result.precision == "relative"


class TestNormalizeDateEdgeCases:
    """엣지 케이스"""

    def test_embedded_year(self) -> None:
        """텍스트 속 연도 추출"""
        result = normalize_date("약 2003 무렵")
        assert result.normalized == "2003"
        assert result.precision == "approximate"

    def test_empty_string(self) -> None:
        result = normalize_date("")
        assert result.precision == "relative"

    def test_whitespace_only(self) -> None:
        result = normalize_date("   ")
        assert result.precision == "relative"


class TestSortKeyFromNormalized:
    """정렬 키 생성"""

    def test_full_date(self) -> None:
        assert sort_key_from_normalized("2012-01-05") == "2012-01-05"

    def test_month_only(self) -> None:
        assert sort_key_from_normalized("2012-01") == "2012-01-15"

    def test_year_only(self) -> None:
        assert sort_key_from_normalized("2012") == "2012-06-15"

    def test_empty(self) -> None:
        assert sort_key_from_normalized("") == "9999-99-99"

    def test_sorting_order(self) -> None:
        """정렬이 올바른 시간순을 만드는지 확인"""
        dates = ["2000-07", "2012-01-05", "1995", "2005-04", ""]
        sorted_dates = sorted(dates, key=sort_key_from_normalized)
        assert sorted_dates == ["1995", "2000-07", "2005-04", "2012-01-05", ""]
