"""카카오톡 파서 단위 테스트"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.storyboard.service.kakao_parser import (
    KakaoTalkParser,
    _convert_time,
    _make_date,
)


class TestConvertTime:
    """_convert_time 단위 테스트"""

    def test_am_normal(self) -> None:
        assert _convert_time("오전", "9", "30") == "09:30"

    def test_pm_normal(self) -> None:
        assert _convert_time("오후", "3", "42") == "15:42"

    def test_pm_12(self) -> None:
        """오후 12시는 12시 그대로"""
        assert _convert_time("오후", "12", "00") == "12:00"

    def test_am_12(self) -> None:
        """오전 12시는 0시"""
        assert _convert_time("오전", "12", "00") == "00:00"

    def test_midnight(self) -> None:
        assert _convert_time("오전", "12", "30") == "00:30"

    def test_noon(self) -> None:
        assert _convert_time("오후", "12", "30") == "12:30"


class TestMakeDate:
    """_make_date 단위 테스트"""

    def test_normal(self) -> None:
        assert _make_date("2024", "1", "5") == "2024-01-05"

    def test_two_digit(self) -> None:
        assert _make_date("2024", "12", "31") == "2024-12-31"


class TestKakaoTalkParserParse:
    """KakaoTalkParser.parse() 테스트"""

    def setup_method(self) -> None:
        self.parser = KakaoTalkParser()

    def test_pc_format_single(self) -> None:
        text = "2024년 1월 15일 오후 3:42, 홍길동 : 안녕하세요"
        messages = self.parser.parse(text)
        assert len(messages) == 1
        msg = messages[0]
        assert msg.sender == "홍길동"
        assert msg.date == "2024-01-15"
        assert msg.time == "15:42"
        assert msg.content == "안녕하세요"

    def test_pc_format_multiple(self) -> None:
        text = (
            "2024년 1월 15일 오후 3:42, 홍길동 : 안녕하세요\n"
            "2024년 1월 15일 오후 3:43, 김철수 : 네 안녕하세요\n"
            "2024년 1월 15일 오후 3:45, 홍길동 : 오늘 법원 다녀왔어요"
        )
        messages = self.parser.parse(text)
        assert len(messages) == 3
        assert messages[0].sender == "홍길동"
        assert messages[1].sender == "김철수"
        assert messages[2].content == "오늘 법원 다녀왔어요"

    def test_mobile_format(self) -> None:
        text = (
            "--------------- 2024년 1월 15일 월요일 ---------------\n"
            "[홍길동] [오후 3:42] 안녕하세요\n"
            "[김철수] [오후 3:43] 네 안녕하세요"
        )
        messages = self.parser.parse(text)
        assert len(messages) == 2
        assert messages[0].sender == "홍길동"
        assert messages[0].date == "2024-01-15"
        assert messages[0].time == "15:42"

    def test_mobile_format_no_date_header(self) -> None:
        """날짜 헤더 없는 모바일 메시지는 '날짜 미상'"""
        text = "[홍길동] [오전 9:00] 아침이에요"
        messages = self.parser.parse(text)
        assert len(messages) == 1
        assert messages[0].date == "날짜 미상"

    def test_empty_text(self) -> None:
        assert self.parser.parse("") == []

    def test_non_message_lines_skipped(self) -> None:
        text = (
            "카카오톡 대화\n"
            "저장한 날짜 : 2024년 1월 15일\n"
            "2024년 1월 15일 오후 3:42, 홍길동 : 안녕하세요\n"
            "이 메시지는 일반 텍스트입니다\n"
        )
        messages = self.parser.parse(text)
        assert len(messages) == 1

    def test_sorted_by_time(self) -> None:
        text = (
            "2024년 1월 15일 오후 5:00, 김철수 : 늦은 메시지\n"
            "2024년 1월 15일 오전 9:00, 홍길동 : 이른 메시지"
        )
        messages = self.parser.parse(text)
        assert messages[0].time == "09:00"
        assert messages[1].time == "17:00"


class TestKakaoTalkParserToTimelineText:
    """KakaoTalkParser.to_timeline_text() 테스트"""

    def setup_method(self) -> None:
        self.parser = KakaoTalkParser()

    def test_basic(self) -> None:
        text = "2024년 1월 15일 오후 3:42, 홍길동 : 안녕하세요"
        messages = self.parser.parse(text)
        result = self.parser.to_timeline_text(messages)
        assert "2024-01-15 15:42 홍길동: 안녕하세요" in result

    def test_empty(self) -> None:
        assert self.parser.to_timeline_text([]) == ""


class TestKakaoTalkParserToChunks:
    """KakaoTalkParser.to_chunks() 테스트"""

    def setup_method(self) -> None:
        self.parser = KakaoTalkParser()

    def test_small_input_single_chunk(self) -> None:
        text = "2024년 1월 15일 오후 3:42, 홍길동 : 짧은 메시지"
        messages = self.parser.parse(text)
        chunks = self.parser.to_chunks(messages)
        assert len(chunks) == 1

    def test_empty_messages(self) -> None:
        assert self.parser.to_chunks([]) == []

    def test_large_input_multiple_chunks(self) -> None:
        """대량 메시지 시 복수 청크 생성 확인"""
        lines = []
        for i in range(500):
            h = (i % 12) + 1
            ampm = "오전" if i % 24 < 12 else "오후"
            lines.append(
                f"2024년 1월 15일 {ampm} {h}:{i % 60:02d}, 발신자 : "
                + "테스트 메시지 " * 10
            )
        text = "\n".join(lines)
        messages = self.parser.parse(text)
        chunks = self.parser.to_chunks(messages)
        assert len(chunks) > 1
        # 모든 청크가 비어있지 않아야 함
        for chunk in chunks:
            assert len(chunk) > 0
