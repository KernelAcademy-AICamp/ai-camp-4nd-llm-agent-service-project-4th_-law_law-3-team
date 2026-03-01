"""Vision 서비스 - Gemini Vision API를 사용한 이미지 분석"""
import json
import logging
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)

# 지원되는 이미지 포맷
SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "gif", "webp", "bmp"}

VISION_SYSTEM_PROMPT = """당신은 법률 문서 및 이미지 분석 전문가이자 영화 스토리보드 작가입니다.
업로드된 이미지(문서, 스크린샷, 사진 등)를 분석하여 시간순으로 중요한 이벤트들을 추출합니다.

각 이벤트에 대해 다음 정보를 추출하세요:
1. date: 날짜 (가능한 경우 YYYY-MM-DD 형식, 불가능하면 "2024년 초", "약 1개월 전" 등 자유형식)
2. time_of_day: 시간대 ("아침", "낮", "저녁", "밤", "새벽" 중 하나, 추정 가능 시)
3. time: 구체적 시간 (HH:MM 형식, 언급된 경우에만)
4. location: 장소 (문서/이미지에서 확인 가능한 경우)
5. title: 짧고 명확한 이벤트 제목 (20자 이내)
6. description_short: 한 줄 요약 (50자 이내)
7. description_detailed: 상세 설명 (300자 이내)
8. participants_detailed: 참여자 배열, 각 참여자는:
   - name: 이름/호칭
   - role: 역할 ("victim", "perpetrator", "witness", "bystander", "authority", "other")
   - action: 해당 장면에서의 행동
   - emotion: 감정 상태
9. key_dialogue: 핵심 대사나 발언 (있는 경우)
10. legal_significance: 법적 의미/중요성 (있는 경우)
11. evidence_items: 관련 증거물 배열
12. mood: 장면 분위기

반드시 다음 JSON 형식으로만 응답하세요:
{
  "timeline": [
    {
      "date": "날짜",
      "time_of_day": "시간대",
      "time": "구체적 시간",
      "location": "장소",
      "title": "제목",
      "description_short": "한 줄 요약",
      "description_detailed": "상세 설명",
      "participants_detailed": [
        {"name": "이름", "role": "역할", "action": "행동", "emotion": "감정"}
      ],
      "key_dialogue": "핵심 대사",
      "legal_significance": "법적 의미",
      "evidence_items": ["증거물1", "증거물2"],
      "mood": "분위기"
    }
  ],
  "summary": "전체 내용 요약 (한 문장)"
}

추가 설명 없이 JSON만 출력합니다."""


@lru_cache(maxsize=1)
def _get_genai_client() -> genai.Client:
    """Gemini 클라이언트 싱글톤 반환 (지연 초기화)"""
    return genai.Client(api_key=settings.GOOGLE_API_KEY)


async def analyze_image(
    image_file: BinaryIO,
    filename: str,
    additional_context: str = "",
) -> dict[str, Any]:
    """
    이미지를 분석하여 타임라인 추출

    Args:
        image_file: 이미지 파일 객체
        filename: 원본 파일명
        additional_context: 추가 컨텍스트 설명

    Returns:
        타임라인 추출 결과 dict
    """
    # 파일 확장자 확인
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(f"지원하지 않는 이미지 포맷입니다: {extension}")

    # 클라이언트 싱글톤 사용
    client = _get_genai_client()

    # 이미지 읽기
    image_data = image_file.read()

    # MIME 타입 설정
    mime_type = f"image/{extension}"
    if extension == "jpg":
        mime_type = "image/jpeg"

    # 프롬프트 생성
    user_prompt = "이 이미지를 분석하여 타임라인을 추출해주세요."
    if additional_context:
        user_prompt += f"\n\n추가 정보: {additional_context}"

    # Gemini Vision API 비동기 호출
    try:
        contents: list[types.Part] = [
            types.Part.from_text(text=VISION_SYSTEM_PROMPT),
            types.Part.from_bytes(data=image_data, mime_type=mime_type),
            types.Part.from_text(text=user_prompt),
        ]
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,  # type: ignore[arg-type]
            config=types.GenerateContentConfig(
                temperature=0.3,
            ),
        )
    except Exception as e:
        # 로깅용 컨텍스트 정보
        image_size_kb = len(image_data) / 1024
        prompt_summary = user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt
        logger.error(
            "Gemini Vision API 호출 실패: %s | mime_type=%s, image_size=%.1fKB, prompt=%s",
            str(e),
            mime_type,
            image_size_kb,
            prompt_summary,
        )
        return {
            "success": False,
            "error": f"이미지 분석 API 오류: {str(e)}",
            "timeline": [],
            "summary": None,
        }

    # 응답 파싱
    content = response.text
    if not content:
        return {"success": False, "timeline": [], "summary": None}

    # JSON 추출 (markdown 코드 블록 처리)
    if "```json" in content:
        parts = content.split("```json")
        if len(parts) > 1:
            inner_parts = parts[1].split("```")
            content = inner_parts[0] if inner_parts else parts[1]
    elif "```" in content:
        parts = content.split("```")
        if len(parts) > 1:
            content = parts[1]

    try:
        data = json.loads(content.strip())
    except json.JSONDecodeError:
        return {"success": False, "timeline": [], "summary": None}

    # TimelineItem 리스트로 변환
    timeline_items: list[dict[str, Any]] = []
    raw_timeline = data.get("timeline", [])

    for idx, item in enumerate(raw_timeline):
        description_short = item.get("description_short", "")
        description_detailed = item.get("description_detailed", "")
        legacy_description = item.get("description", "")
        description = description_short or description_detailed or legacy_description

        participants_detailed_raw = item.get("participants_detailed", [])
        if not isinstance(participants_detailed_raw, list):
            participants_detailed_raw = []

        participant_names = [
            p.get("name", "") if isinstance(p, dict) else str(p)
            for p in participants_detailed_raw
            if p is not None
        ] or item.get("participants", [])

        timeline_items.append({
            "id": str(uuid.uuid4()),
            "date": item.get("date", "날짜 미상"),
            "title": item.get("title", "제목 없음"),
            "description": description,
            "participants": participant_names,
            "order": idx,
            "location": item.get("location"),
            "time_of_day": item.get("time_of_day"),
            "time": item.get("time"),
            "description_short": description_short,
            "description_detailed": description_detailed,
            "participants_detailed": participants_detailed_raw,
            "key_dialogue": item.get("key_dialogue"),
            "legal_significance": item.get("legal_significance"),
            "evidence_items": item.get("evidence_items", []),
            "mood": item.get("mood"),
        })

    return {
        "success": True,
        "timeline": timeline_items,
        "summary": data.get("summary"),
    }
