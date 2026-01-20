"""스토리보드 모듈 - 서비스 레이어"""
import json
import uuid
from typing import List

from openai import OpenAI

from app.core.config import settings
from ..schema import TimelineItem, ExtractTimelineResponse, TimelineData


EXTRACTION_SYSTEM_PROMPT = """당신은 법률 사건 분석 전문가입니다.
사용자가 입력한 사건 내용에서 시간순으로 중요한 이벤트들을 추출합니다.

각 이벤트에 대해 다음 정보를 추출하세요:
1. date: 날짜 (가능한 경우 YYYY-MM-DD 형식, 불가능하면 "2024년 초", "약 1개월 전" 등 자유형식)
2. title: 짧고 명확한 이벤트 제목 (20자 이내)
3. description: 이벤트에 대한 상세 설명 (100자 이내)
4. participants: 해당 이벤트에 관련된 인물/기관 목록

반드시 다음 JSON 형식으로만 응답하세요:
{
  "timeline": [
    {"date": "날짜", "title": "제목", "description": "설명", "participants": ["관련자1", "관련자2"]},
    ...
  ],
  "summary": "전체 사건 요약 (한 문장)"
}

추가 설명 없이 JSON만 출력합니다."""


async def extract_timeline_from_text(text: str) -> ExtractTimelineResponse:
    """텍스트에서 타임라인 추출 (OpenAI API 사용)"""
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"다음 사건 내용에서 타임라인을 추출해주세요:\n\n{text}"},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    # 응답 파싱
    content = response.choices[0].message.content
    if not content:
        return ExtractTimelineResponse(success=False, timeline=[], summary=None)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return ExtractTimelineResponse(success=False, timeline=[], summary=None)

    # TimelineItem 리스트로 변환
    timeline_items: List[TimelineItem] = []
    raw_timeline = data.get("timeline", [])

    for idx, item in enumerate(raw_timeline):
        timeline_items.append(
            TimelineItem(
                id=str(uuid.uuid4()),
                date=item.get("date", "날짜 미상"),
                title=item.get("title", "제목 없음"),
                description=item.get("description", ""),
                participants=item.get("participants", []),
                order=idx,
            )
        )

    return ExtractTimelineResponse(
        success=True,
        timeline=timeline_items,
        summary=data.get("summary"),
    )


def validate_timeline_data(data: dict) -> bool:
    """타임라인 데이터 유효성 검사"""
    try:
        TimelineData(**data)
        return True
    except Exception:
        return False
