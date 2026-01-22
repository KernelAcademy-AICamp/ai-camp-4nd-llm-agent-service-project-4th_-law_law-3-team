"""스토리보드 모듈 - 서비스 레이어"""
import json
import uuid
from typing import List

from openai import OpenAI

from app.core.config import settings
from ..schema import TimelineItem, ExtractTimelineResponse, TimelineData, Participant, ParticipantRole


EXTRACTION_SYSTEM_PROMPT = """당신은 법률 사건 분석 전문가이자 영화 스토리보드 작가입니다.
사용자가 입력한 사건 내용에서 시간순으로 중요한 이벤트들을 **영화 스토리보드 형식**으로 추출합니다.

## 핵심 원칙
1. **완전한 이해 가능**: 스토리보드만 보고도 제3자가 상황을 완벽히 파악 가능해야 함
2. **역할 명확화**: 피해자와 가해자를 명확히 구분하여 표시
3. **법적 맥락**: 각 장면의 법적 의미와 중요성 포함
4. **영화적 표현**: 장면의 분위기, 감정, 시각적 요소 표현

## 각 장면(이벤트)에서 추출할 정보

1. **date**: 날짜 (YYYY-MM-DD 또는 "2024년 초", "약 1개월 전" 등)
2. **time_of_day**: 시간대 ("아침", "낮", "저녁", "밤", "새벽" 중 하나, 추정 가능 시)
3. **time**: 구체적 시간 (HH:MM 형식, 예: "14:30", "오후 3시" 등, 언급된 경우에만)
4. **location**: 장소 (사무실, 회의실, 거리, 카페 등)
4. **title**: 짧은 이벤트 제목 (20자 이내, 핵심 행위 중심)
5. **description_short**: 한 줄 요약 (50자 이내)
6. **description_detailed**: 상세 설명 (300자 이내, 5W1H 포함)
7. **participants_detailed**: 참여자 배열, 각 참여자는:
   - name: 이름/호칭 (예: "A씨", "B 과장", "경찰관")
   - role: 역할 ("victim", "perpetrator", "witness", "bystander", "authority", "other")
   - action: 해당 장면에서의 행동 (예: "폭언을 함", "맞고 있음")
   - emotion: 감정 상태 (예: "분노", "두려움", "무관심")
8. **key_dialogue**: 핵심 대사나 발언 (있는 경우, 인용부호 포함)
9. **legal_significance**: 법적 의미/중요성 (예: "직장 내 괴롭힘 구성 요건", "상해죄 성립 가능")
10. **evidence_items**: 관련 증거물 배열 (예: ["CCTV 영상", "진단서", "목격자 증언"])
11. **mood**: 장면 분위기 (예: "긴장감", "두려움", "혼란")

## 역할(role) 구분 기준
- **victim (피해자)**: 불법 행위나 부당한 행위를 당하는 사람
- **perpetrator (가해자)**: 불법 행위나 부당한 행위를 하는 사람
- **witness (증인)**: 사건을 직접 목격한 사람
- **bystander (방관자)**: 현장에 있었지만 개입하지 않은 사람
- **authority (공권력)**: 경찰, 검찰, 법원 등 공적 기관/인물
- **other (기타)**: 위에 해당하지 않는 관련자

## JSON 응답 형식 (반드시 준수)
{
  "timeline": [
    {
      "date": "2024-01-15",
      "time_of_day": "낮",
      "time": "15:00",
      "location": "사무실 회의실",
      "title": "B 과장의 폭언 시작",
      "description_short": "B 과장이 팀 회의 중 A씨에게 공개적으로 폭언",
      "description_detailed": "2024년 1월 15일 오후 3시경, 마케팅팀 주간 회의 중 B 과장이 A씨의 보고서 오류를 지적하며 '이런 것도 못하면 왜 월급 받아?'라며 10분간 폭언을 퍼부음. 동료 5명이 현장에서 목격함.",
      "participants_detailed": [
        {"name": "A씨", "role": "victim", "action": "보고서 발표 중 폭언을 당함", "emotion": "수치심, 두려움"},
        {"name": "B 과장", "role": "perpetrator", "action": "보고서 오류를 빌미로 폭언", "emotion": "분노"},
        {"name": "동료들", "role": "witness", "action": "침묵하며 지켜봄", "emotion": "불편함"}
      ],
      "key_dialogue": "이런 것도 못하면 왜 월급 받아? 너 같은 애 때문에 팀이 망하는 거야!",
      "legal_significance": "직장 내 괴롭힘 구성요건 중 '우월적 지위를 이용한 업무상 적정 범위 초과 행위'에 해당 가능",
      "evidence_items": ["동료 증언", "회의 참석자 명단"],
      "mood": "긴장감, 수치심"
    }
  ],
  "summary": "B 과장의 지속적인 직장 내 괴롭힘으로 인한 A씨의 피해 사례"
}

추가 설명 없이 JSON만 출력합니다. 모든 필드를 가능한 상세하게 채워주세요."""


def _parse_participant(participant_data: dict) -> Participant:
    """참여자 데이터를 Participant 모델로 변환"""
    role_str = participant_data.get("role", "other")
    try:
        role = ParticipantRole(role_str)
    except ValueError:
        role = ParticipantRole.OTHER

    return Participant(
        name=participant_data.get("name", "미상"),
        role=role,
        action=participant_data.get("action"),
        emotion=participant_data.get("emotion"),
    )


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

    content = response.choices[0].message.content
    if not content:
        return ExtractTimelineResponse(success=False, timeline=[], summary=None)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return ExtractTimelineResponse(success=False, timeline=[], summary=None)

    timeline_items: List[TimelineItem] = []
    raw_timeline = data.get("timeline", [])

    for idx, item in enumerate(raw_timeline):
        participants_detailed_raw = item.get("participants_detailed", [])
        participants_detailed = [
            _parse_participant(p) for p in participants_detailed_raw
        ]

        participant_names = [p.name for p in participants_detailed]
        if not participant_names:
            participant_names = item.get("participants", [])

        description_detailed = item.get("description_detailed", "")
        description_short = item.get("description_short", "")
        legacy_description = item.get("description", "")

        description = description_short or description_detailed or legacy_description

        timeline_items.append(
            TimelineItem(
                id=str(uuid.uuid4()),
                date=item.get("date", "날짜 미상"),
                title=item.get("title", "제목 없음"),
                description=description,
                participants=participant_names,
                order=idx,
                scene_number=idx + 1,
                location=item.get("location"),
                time_of_day=item.get("time_of_day"),
                time=item.get("time"),
                description_short=description_short,
                description_detailed=description_detailed,
                participants_detailed=participants_detailed,
                key_dialogue=item.get("key_dialogue"),
                legal_significance=item.get("legal_significance"),
                evidence_items=item.get("evidence_items", []),
                mood=item.get("mood"),
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
