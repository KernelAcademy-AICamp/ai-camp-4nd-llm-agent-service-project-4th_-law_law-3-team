"""카카오톡 대화 내보내기 .txt 파일 파서 (FR-02)"""

import re
from dataclasses import dataclass


@dataclass
class KakaoMessage:
    """파싱된 카카오톡 메시지"""
    sender: str
    datetime_str: str   # "YYYY-MM-DD HH:MM"
    date: str           # "YYYY-MM-DD"
    time: str           # "HH:MM"
    content: str


# 카카오톡 내보내기 형식 패턴 (다중 형식 지원)
KAKAO_PATTERNS: list[re.Pattern[str]] = [
    # 형식 1 (PC): "2024년 1월 15일 오후 3:42, 홍길동 : 내용"
    re.compile(
        r"^(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*(오전|오후)\s*(\d{1,2}):(\d{2}),\s*(.+?)\s*:\s*(.+)$"
    ),
    # 형식 2 (모바일): "[홍길동] [오후 3:42] 내용" (날짜 헤더 별도)
    re.compile(
        r"^\[(.+?)\]\s*\[(오전|오후)\s*(\d{1,2}):(\d{2})\]\s*(.+)$"
    ),
]

# 날짜 헤더 패턴 (모바일 형식): "--------------- 2024년 1월 15일 화요일 ---------------"
DATE_HEADER_PATTERN: re.Pattern[str] = re.compile(
    r"^-+\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*\w+요일\s*-+$"
)

# 슬라이딩 윈도우 청크 크기 (문자 수 기준)
KAKAO_CHUNK_SIZE = 12_000
KAKAO_CHUNK_OVERLAP = 500


def _convert_time(ampm: str, hour_str: str, minute_str: str) -> str:
    """오전/오후 + 시/분 → 24시간 HH:MM 변환."""
    hour = int(hour_str)
    minute = int(minute_str)
    if ampm == "오후" and hour != 12:
        hour += 12
    elif ampm == "오전" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


def _make_date(year: str, month: str, day: str) -> str:
    """연월일 → YYYY-MM-DD."""
    return f"{year}-{int(month):02d}-{int(day):02d}"


class KakaoTalkParser:
    """
    카카오톡 .txt 파일 → KakaoMessage 리스트

    지원 형식:
    1. PC 카카오톡 내보내기 (날짜 + 시간 한 줄)
    2. 모바일 카카오톡 내보내기 (날짜 헤더 + [발신자] [시간] 형식)
    """

    def parse(self, text: str) -> list[KakaoMessage]:
        """텍스트 → 메시지 리스트 (시간순 정렬)."""
        messages: list[KakaoMessage] = []
        current_date: str = ""

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # 날짜 헤더 감지 (모바일 형식)
            date_header_match = DATE_HEADER_PATTERN.match(line)
            if date_header_match:
                year, month, day = date_header_match.group(1, 2, 3)
                current_date = _make_date(year, month, day)
                continue

            # 형식 1 (PC): 날짜 + 시간 + 발신자 + 내용
            pc_match = KAKAO_PATTERNS[0].match(line)
            if pc_match:
                year, month, day, ampm, hour, minute, sender, content = pc_match.groups()
                date = _make_date(year, month, day)
                time = _convert_time(ampm, hour, minute)
                messages.append(
                    KakaoMessage(
                        sender=sender.strip(),
                        datetime_str=f"{date} {time}",
                        date=date,
                        time=time,
                        content=content.strip(),
                    )
                )
                continue

            # 형식 2 (모바일): [발신자] [시간] 내용 (current_date 필요)
            if current_date:
                mobile_match = KAKAO_PATTERNS[1].match(line)
                if mobile_match:
                    sender, ampm, hour, minute, content = mobile_match.groups()
                    time = _convert_time(ampm, hour, minute)
                    messages.append(
                        KakaoMessage(
                            sender=sender.strip(),
                            datetime_str=f"{current_date} {time}",
                            date=current_date,
                            time=time,
                            content=content.strip(),
                        )
                    )

        # 시간순 정렬
        messages.sort(key=lambda m: m.datetime_str)
        return messages

    def to_timeline_text(self, messages: list[KakaoMessage]) -> str:
        """
        메시지 리스트 → LLM 입력용 텍스트.

        각 메시지를 "YYYY-MM-DD HH:MM 발신자: 내용" 형식으로 변환.
        슬라이딩 윈도우 분할 대상으로 사용 (NFR-06).
        """
        lines: list[str] = []
        for msg in messages:
            lines.append(f"{msg.datetime_str} {msg.sender}: {msg.content}")
        return "\n".join(lines)

    def to_chunks(self, messages: list[KakaoMessage]) -> list[str]:
        """
        대용량 카카오톡 대화를 슬라이딩 윈도우로 분할.

        KAKAO_CHUNK_SIZE 문자 기준으로 분할하며,
        KAKAO_CHUNK_OVERLAP 문자 겹침을 유지하여 경계 이벤트 누락 방지.
        """
        full_text = self.to_timeline_text(messages)
        if len(full_text) <= KAKAO_CHUNK_SIZE:
            return [full_text]

        chunks: list[str] = []
        start = 0
        while start < len(full_text):
            end = start + KAKAO_CHUNK_SIZE
            chunk = full_text[start:end]
            chunks.append(chunk)
            if end >= len(full_text):
                break
            start = end - KAKAO_CHUNK_OVERLAP

        return chunks
