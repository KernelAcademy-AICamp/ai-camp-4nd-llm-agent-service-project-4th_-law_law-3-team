"""스토리보드 모듈 - 서비스 레이어"""

import json
import uuid
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.core.config import settings

from ..schema import (
    ExtractTimelineResponse,
    Participant,
    ParticipantRole,
    TimelineData,
    TimelineItem,
)
from .doc_type_detector import detect_document_type

# --- 장문 처리 임계값 ---
LONG_TEXT_THRESHOLD = 15_000

# --- 문서 유형별 추가 프롬프트 ---

CIVIL_EXTRA = """
## 민사 기록 추가 지시
- 상담일지는 이슈 중심 서술 → 날짜를 추출하여 시간순 재배열
- "YYYY. M." 형식은 "YYYY-MM"으로 정규화
- 계약 체결, 대금 지급, 등기 이전, 소제기 등이 핵심 이벤트
- 대여, 차용, 보증, 연대보증, 어음 발행, 근저당권 설정/말소도 핵심 이벤트
- "한편", "그런데", "나." 등으로 시작하는 부수적 서술도 별개 법률관계의 사실이므로 반드시 포함
- 원고/피고/소외 등 소송 당사자 관계를 역할로 명확히 구분
"""

CRIMINAL_EXTRA = """
## 형사 기록 추가 지시
- 공소사실은 대체로 시간순 기술
- "YYYY. M. D. HH:MM경" 형식을 정확히 파싱
- 피고인=perpetrator, 피해자=victim으로 분류
- 범행일시, 체포일, 기소일, 판결선고일을 핵심 이벤트로 포함
"""

PUBLIC_EXTRA = """
## 공법 기록 추가 지시
- 행정처분 경위를 시간순 재구성
- 처분청=authority, 처분 대상자=other 또는 victim
- 사전통지→의견제출→처분→소제기 흐름 추적
- 인허가, 등록, 취소, 영업정지 등이 핵심 이벤트
"""

_DOC_TYPE_EXTRA: dict[str, str] = {
    "civil": CIVIL_EXTRA,
    "criminal": CRIMINAL_EXTRA,
    "public": PUBLIC_EXTRA,
}

# --- 메인 프롬프트 ---

EXTRACTION_SYSTEM_PROMPT = """당신은 법률 사건 분석 전문가이자 영화 스토리보드 작가입니다.
사용자가 입력한 사건 내용에서 시간순으로 중요한 이벤트들을 **영화 스토리보드 형식**으로 추출합니다.

## 핵심 원칙
1. **완전한 이해 가능**: 스토리보드만 보고도 제3자가 상황을 완벽히 파악 가능해야 함
2. **역할 명확화**: 피해자와 가해자를 명확히 구분하여 표시
3. **법적 맥락**: 각 장면의 법적 의미와 중요성 포함
4. **영화적 표현**: 장면의 분위기, 감정, 시각적 요소 표현
5. **시간순 정렬**: 서술 순서가 시간 순서와 다르면, 날짜 기준으로 재배열할 것
6. **빠짐없이 추출**: "한편", "그런데", "나." 등으로 시작하는 부수적 서술도 별개 법률관계이므로 반드시 포함. 날짜가 있는 사실은 모두 이벤트로 추출할 것
7. **날짜 정확성**: 입력 텍스트에 명시된 날짜를 우선 사용. 명시적 날짜가 없지만 문맥상 시점을 추론할 수 있으면("그 사이에", "그 후", "이후" 등), 인접한 알려진 날짜를 date에 사용하고 date_raw에 "추정"을 표기 (예: date="2012", date_raw="2012년 이후 추정"). 시점을 전혀 추론할 수 없는 경우에만 "날짜 미상" 사용. 입력에 없는 날짜를 임의로 만들어내는 것은 금지

## 각 장면(이벤트)에서 추출할 정보

1. **date**: 정렬용 정규화 날짜. 반드시 시간순 오름차순 정렬.
   - "YYYY. M. D." → "YYYY-MM-DD"
   - "YYYY. M." → "YYYY-MM"
   - "YYYY년경" → "YYYY"
   - "YYYY년 여름" → "YYYY-07" (정렬용 중간값)
   - "YYYY년 초" → "YYYY-02", "YYYY년 말" → "YYYY-11"
   - 서술 순서가 시간 순서와 다르면 날짜 기준으로 재배열할 것
2. **date_raw**: 원본 날짜 표현 그대로 보존 (예: "2000년 여름", "1995년경", "2012. 1. 5.")
3. **time_of_day**: 시간대 ("아침", "낮", "저녁", "밤", "새벽" 중 하나, 추정 가능 시)
4. **time**: 구체적 시간 (HH:MM 형식, 예: "14:30", "오후 3시" 등, 언급된 경우에만)
5. **location**: 장소 (사무실, 회의실, 거리, 카페 등)
6. **title**: 짧은 이벤트 제목 (20자 이내, 핵심 행위 중심)
7. **description_short**: 한 줄 요약 (50자 이내)
8. **description_detailed**: 상세 설명 (300자 이내, 5W1H 포함)
9. **participants_detailed**: 참여자 배열, 각 참여자는:
   - name: 이름/호칭 (예: "A씨", "B 과장", "경찰관")
   - role: 역할 ("victim", "perpetrator", "witness", "bystander", "authority", "other")
   - action: 해당 장면에서의 행동 (예: "폭언을 함", "맞고 있음")
   - emotion: 감정 상태 (예: "분노", "두려움", "무관심")
10. **key_dialogue**: 핵심 대사나 발언 (있는 경우, 인용부호 포함)
11. **legal_significance**: 법적 의미/중요성 (예: "직장 내 괴롭힘 구성 요건", "상해죄 성립 가능")
12. **evidence_items**: 관련 증거물 배열 (예: ["CCTV 영상", "진단서", "목격자 증언"])
13. **mood**: 장면 분위기 (예: "긴장감", "두려움", "혼란")

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
      "date_raw": "2024. 1. 15.",
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

GANTT_CHART_FIELDS_INSTRUCTION = """
## 간트차트 추가 필드 (신규)

각 이벤트에 아래 필드를 추가로 추출하세요:

- **topic**: 사건의 법적 주제 (한국어, 예: "폭행", "협박", "금전 갈취")
  - 전체 타임라인에서 3~8개 범위로 그룹핑
  - 하나의 이벤트에 하나의 topic만

- **date_start**: 사건 시작일 (YYYY-MM-DD 또는 YYYY-MM 또는 YYYY)
  - 단발성이면 date와 동일
  - 지속적이면 시작일

- **date_end**: 사건 종료일 (같은 형식)
  - 단발성이면 date_start와 동일
  - 현재 진행 중이면 null

- **confidence**: 추출 신뢰도 (0.0~1.0)
  - 1.0: 날짜/내용이 원문에 명확히 기재
  - 0.7~0.9: 문맥에서 추론
  - 0.5 미만: 추정이 많음
"""

# --- 장문 1단계 프롬프트: 날짜 문장 추출 ---

_DATE_EXTRACTION_PROMPT = """다음 법률 문서에서 **날짜가 포함된 문장**과 그 전후 2줄을 추출해주세요.
날짜가 없더라도 시간적 순서를 나타내는 표현("그 뒤에도", "이후", "며칠 후" 등)이 있는 문장도 포함합니다.

출력 형식: 추출된 문장들을 원본 순서대로 나열. 추가 설명 없이 텍스트만 출력합니다."""


def _parse_participant(participant_data: dict[str, Any]) -> Participant:
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


def _build_system_prompt(doc_type: str) -> str:
    """문서 유형에 맞는 시스템 프롬프트를 조합한다."""
    extra = _DOC_TYPE_EXTRA.get(doc_type, "")
    base = EXTRACTION_SYSTEM_PROMPT + GANTT_CHART_FIELDS_INSTRUCTION
    if extra:
        return base + "\n" + extra
    return base


def _date_sort_key(date: str) -> str:
    """date 필드를 정렬 가능한 문자열로 변환한다.

    "YYYY-MM-DD" → "YYYY-MM-DD"
    "YYYY-MM"    → "YYYY-MM-00"
    "YYYY"       → "YYYY-00-00"
    "날짜 미상"   → "9999-99-99" (맨 뒤로)
    """
    if not date or date == "날짜 미상":
        return "9999-99-99"
    parts = date.split("-")
    year = parts[0] if len(parts) >= 1 else "9999"
    month = parts[1] if len(parts) >= 2 else "00"
    day = parts[2] if len(parts) >= 3 else "00"
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


def _parse_timeline_response(data: dict[str, Any]) -> ExtractTimelineResponse:
    """LLM 응답 JSON을 ExtractTimelineResponse로 변환한다."""
    timeline_items: list[TimelineItem] = []
    raw_timeline = data.get("timeline", [])

    for idx, item in enumerate(raw_timeline):
        participants_detailed_raw = item.get("participants_detailed", [])

        # participants_detailed 정규화: None이거나 리스트가 아니면 빈 리스트로
        if participants_detailed_raw is None or not isinstance(
            participants_detailed_raw, list
        ):
            participants_detailed_raw = []

        # 리스트 요소가 문자열이면 {"name": value} 형태로 변환
        normalized_participants = []
        for p in participants_detailed_raw:
            if p is None:
                continue
            if isinstance(p, str):
                normalized_participants.append({"name": p})
            elif isinstance(p, dict):
                normalized_participants.append(p)

        participants_detailed = [
            _parse_participant(p) for p in normalized_participants
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
                date_raw=item.get("date_raw"),
                title=item.get("title", "제목 없음"),
                description=description,
                participants=participant_names,
                order=idx,
                image_url=None,
                image_prompt=None,
                image_status=None,
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
                topic=item.get("topic"),
                date_start=item.get("date_start"),
                date_end=item.get("date_end"),
                evidence_ids=item.get("evidence_ids", []),
                confidence=item.get("confidence"),
            )
        )

    # date 필드 기준 시간순 정렬 (LLM이 순서를 보장하지 않으므로)
    timeline_items.sort(key=lambda item: _date_sort_key(item.date))

    # 정렬 후 order, scene_number 재할당
    for idx, item in enumerate(timeline_items):
        item.order = idx
        item.scene_number = idx + 1

    return ExtractTimelineResponse(
        success=True,
        timeline=timeline_items,
        summary=data.get("summary"),
    )


async def _call_llm(
    client: AsyncOpenAI,
    system_prompt: str,
    user_content: str,
    *,
    response_format: dict[str, str] | None = None,
) -> str | None:
    """OpenAI LLM 호출 공통 함수"""
    kwargs: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
    }
    if response_format:
        kwargs["response_format"] = response_format

    response = await client.chat.completions.create(**kwargs)

    if not response.choices:
        return None
    content: str | None = response.choices[0].message.content
    return content


async def _extract_single_pass(
    text: str,
    system_prompt: str,
) -> ExtractTimelineResponse:
    """단일 패스 타임라인 추출"""
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    content = await _call_llm(
        client,
        system_prompt,
        f"다음 사건 내용에서 타임라인을 추출해주세요:\n\n{text}",
        response_format={"type": "json_object"},
    )

    if not content:
        return ExtractTimelineResponse(success=False, timeline=[], summary=None)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return ExtractTimelineResponse(success=False, timeline=[], summary=None)

    return _parse_timeline_response(data)


async def _extract_two_pass(
    text: str,
    system_prompt: str,
) -> ExtractTimelineResponse:
    """2단계 장문 타임라인 추출

    1단계: 날짜 포함 문장 추출 (컨텍스트 축소)
    2단계: 축소된 텍스트로 타임라인 추출
    """
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # 1단계: 날짜 문장 추출
    date_sentences = await _call_llm(
        client,
        _DATE_EXTRACTION_PROMPT,
        text,
    )

    if not date_sentences:
        # 1단계 실패 시 원본 텍스트로 단일 패스 시도
        return await _extract_single_pass(text, system_prompt)

    # 2단계: 축소된 컨텍스트로 타임라인 추출
    content = await _call_llm(
        client,
        system_prompt,
        f"다음 사건 내용에서 타임라인을 추출해주세요:\n\n{date_sentences}",
        response_format={"type": "json_object"},
    )

    if not content:
        return ExtractTimelineResponse(success=False, timeline=[], summary=None)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return ExtractTimelineResponse(success=False, timeline=[], summary=None)

    return _parse_timeline_response(data)


async def extract_timeline_from_text(text: str) -> ExtractTimelineResponse:
    """텍스트에서 타임라인 추출 (OpenAI API 사용)

    문서 유형을 감지하여 유형별 프롬프트를 적용하고,
    장문(15,000자 초과)은 2단계 처리로 컨텍스트를 축소한다.
    """
    doc_type = detect_document_type(text)
    system_prompt = _build_system_prompt(doc_type)

    if len(text) > LONG_TEXT_THRESHOLD:
        return await _extract_two_pass(text, system_prompt)
    return await _extract_single_pass(text, system_prompt)


def validate_timeline_data(data: dict[str, Any]) -> bool:
    """타임라인 데이터 유효성 검사"""
    try:
        TimelineData(**data)
        return True
    except (ValidationError, ValueError):
        return False
