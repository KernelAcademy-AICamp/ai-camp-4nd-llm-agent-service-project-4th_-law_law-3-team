"""이미지 생성 서비스 - Google Gemini 2.0 Flash 사용 (나노바나나 프로)"""
import uuid
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from PIL import Image

from app.core.config import settings
from ..schema import Participant, ParticipantRole

# 미디어 디렉토리 경로
MEDIA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "media"
IMAGES_DIR = MEDIA_DIR / "storyboard" / "images"

# 역할별 한글 라벨
ROLE_LABELS = {
    ParticipantRole.VICTIM: "피해자",
    ParticipantRole.PERPETRATOR: "가해자",
    ParticipantRole.WITNESS: "증인",
    ParticipantRole.BYSTANDER: "방관자",
    ParticipantRole.AUTHORITY: "공권력",
    ParticipantRole.OTHER: "기타",
}


def _ensure_dirs() -> None:
    """디렉토리 존재 확인 및 생성"""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _build_image_prompt(
    title: str,
    description: str,
    participants: list[str],
    location: Optional[str] = None,
    time_of_day: Optional[str] = None,
    participants_detailed: Optional[list[Participant]] = None,
    mood: Optional[str] = None,
) -> str:
    """
    타임라인 항목에서 스토리보드 이미지 생성 프롬프트 생성 (한글 최적화)

    새 필드가 있으면 더 상세한 프롬프트를 생성합니다.
    """
    # 장소/시간대 정보
    setting_parts = []
    if location:
        setting_parts.append(f"장소: {location}")
    if time_of_day:
        setting_parts.append(f"시간대: {time_of_day}")
    setting_text = ", ".join(setting_parts) if setting_parts else ""

    # 참여자 상세 정보
    participants_section = ""
    if participants_detailed:
        participant_lines = []
        for p in participants_detailed:
            role_label = ROLE_LABELS.get(p.role, "기타")
            line = f"- {p.name} ({role_label})"
            if p.action:
                line += f": {p.action}"
            if p.emotion:
                line += f" (감정: {p.emotion})"
            participant_lines.append(line)
        participants_section = "## 등장인물 (역할별)\n" + "\n".join(participant_lines)
    elif participants:
        participants_section = f"등장인물: {', '.join(participants)}"

    # 분위기 정보
    mood_text = f"분위기: {mood}" if mood else ""

    prompt = f"""Create a professional storyboard illustration for the following scene.

## Scene Information
- Title: {title}
- Situation: {description}
{f"- Setting: {setting_text}" if setting_text else ""}

{participants_section}

## Visual Expression Guide
- **Victim**: Defensive posture, cowering expression, anxious/fearful look
- **Perpetrator**: Aggressive stance, intimidating presence, angry/contemptuous expression
- **Witness/Bystander**: Observing position, uncomfortable or indifferent expression
- **Authority**: Uniform, authoritative stance

## Style Requirements
- Professional storyboard illustration style
- Black and white sketch with clean lines and minimal shading
- Cinematic composition and visual storytelling
- 16:9 aspect ratio, high quality frame
- Clear facial expressions and body language
- Composition that visually shows the relationship between victim and perpetrator
- Dynamic camera angles (low angle for intimidation, high angle for vulnerability)

## IMPORTANT: NO TEXT
- Do NOT include any text, letters, words, or writing in the image
- No speech bubbles, captions, signs, or any written content
- Focus purely on visual storytelling through expressions, poses, and composition

Generate a high-quality storyboard image that clearly conveys the situation through visuals only."""

    return prompt


async def generate_image(
    item_id: str,
    title: str,
    description: str,
    participants: list[str],
    location: Optional[str] = None,
    time_of_day: Optional[str] = None,
    participants_detailed: Optional[list[Participant]] = None,
    mood: Optional[str] = None,
) -> dict:
    """
    타임라인 항목에 대한 스토리보드 이미지 생성 (Google Gemini 2.0 Flash)

    Args:
        item_id: 타임라인 항목 ID
        title: 이벤트 제목
        description: 이벤트 설명
        participants: 관련자 목록
        location: 장소
        time_of_day: 시간대
        participants_detailed: 참여자 상세 정보 (역할 포함)
        mood: 장면 분위기

    Returns:
        생성 결과 dict (image_url, image_prompt 포함)
    """
    _ensure_dirs()

    prompt = _build_image_prompt(
        title=title,
        description=description,
        participants=participants,
        location=location,
        time_of_day=time_of_day,
        participants_detailed=participants_detailed,
        mood=mood,
    )

    try:
        # Gemini 클라이언트 초기화
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)

        # Gemini 2.0 Flash 이미지 생성
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        if not response.candidates or not response.candidates[0].content.parts:
            raise ValueError("이미지 생성 결과가 없습니다")

        # 이미지 데이터 추출
        image_data = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_data = part.inline_data.data
                break

        if not image_data:
            raise ValueError("생성된 이미지를 찾을 수 없습니다")

        # 이미지 저장
        image_filename = f"{item_id}_{uuid.uuid4().hex[:8]}.png"
        image_path = IMAGES_DIR / image_filename

        with open(image_path, "wb") as f:
            f.write(image_data)

        local_url = f"/media/storyboard/images/{image_filename}"

        return {
            "success": True,
            "image_url": local_url,
            "image_prompt": prompt,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "image_url": None,
            "image_prompt": prompt,
        }


async def generate_image_fallback(
    item_id: str,
    title: str,
    description: str,
    participants: list[str],
    location: Optional[str] = None,
    time_of_day: Optional[str] = None,
    participants_detailed: Optional[list[Participant]] = None,
    mood: Optional[str] = None,
) -> dict:
    """Gemini 실패 시 플레이스홀더 이미지 생성"""
    _ensure_dirs()

    prompt = _build_image_prompt(
        title=title,
        description=description,
        participants=participants,
        location=location,
        time_of_day=time_of_day,
        participants_detailed=participants_detailed,
        mood=mood,
    )

    # 플레이스홀더 이미지 생성
    img = Image.new("RGB", (768, 432), color=(30, 41, 59))

    image_filename = f"{item_id}_{uuid.uuid4().hex[:8]}_placeholder.png"
    image_path = IMAGES_DIR / image_filename
    img.save(image_path, "PNG")

    image_url = f"/media/storyboard/images/{image_filename}"

    return {
        "success": True,
        "image_url": image_url,
        "image_prompt": prompt,
        "is_placeholder": True,
    }
