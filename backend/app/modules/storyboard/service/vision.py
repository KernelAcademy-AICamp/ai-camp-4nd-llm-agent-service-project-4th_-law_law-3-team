"""Vision 서비스 - Gemini Vision API를 사용한 이미지 분석"""
import base64
import json
import logging
import uuid
from pathlib import Path
from typing import BinaryIO, List

import google.generativeai as genai

logger = logging.getLogger(__name__)

from app.core.config import settings
from ..schema import TimelineItem

# 지원되는 이미지 포맷
SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "gif", "webp", "bmp"}

VISION_SYSTEM_PROMPT = """당신은 법률 문서 및 이미지 분석 전문가입니다.
업로드된 이미지(문서, 스크린샷, 사진 등)를 분석하여 시간순으로 중요한 이벤트들을 추출합니다.

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
  "summary": "전체 내용 요약 (한 문장)"
}

추가 설명 없이 JSON만 출력합니다."""


async def analyze_image(
    image_file: BinaryIO,
    filename: str,
    additional_context: str = "",
) -> dict:
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

    # Gemini 설정
    genai.configure(api_key=settings.GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # 이미지 읽기
    image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    # MIME 타입 설정
    mime_type = f"image/{extension}"
    if extension == "jpg":
        mime_type = "image/jpeg"

    # 프롬프트 생성
    user_prompt = "이 이미지를 분석하여 타임라인을 추출해주세요."
    if additional_context:
        user_prompt += f"\n\n추가 정보: {additional_context}"

    # Gemini Vision API 호출
    try:
        response = model.generate_content(
            [
                VISION_SYSTEM_PROMPT,
                {"mime_type": mime_type, "data": image_base64},
                user_prompt,
            ],
            generation_config=genai.types.GenerationConfig(
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
    timeline_items: List[dict] = []
    raw_timeline = data.get("timeline", [])

    for idx, item in enumerate(raw_timeline):
        timeline_items.append({
            "id": str(uuid.uuid4()),
            "date": item.get("date", "날짜 미상"),
            "title": item.get("title", "제목 없음"),
            "description": item.get("description", ""),
            "participants": item.get("participants", []),
            "order": idx,
        })

    return {
        "success": True,
        "timeline": timeline_items,
        "summary": data.get("summary"),
    }
