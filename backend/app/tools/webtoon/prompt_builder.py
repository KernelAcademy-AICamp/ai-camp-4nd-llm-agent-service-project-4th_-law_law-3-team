"""Chain 2: 장면 → 이미지 프롬프트 (LLM 불필요, 텍스트 조합)

WebtoonPanel 데이터를 Gemini 이미지 생성용 영문 프롬프트로 변환합니다.
"""

from dataclasses import dataclass

from app.modules.content_marketing.schema import WebtoonPanel


@dataclass(frozen=True)
class LawyerCharacterProfile:
    """변호사 캐릭터 프로필 (일관성 유지용)"""

    gender: str = "male"
    age_range: str = "35-45"
    hair: str = "neat black hair, side-parted"
    suit: str = "dark navy pinstripe suit, white dress shirt, burgundy tie"
    build: str = "lean, professional posture"
    expression_default: str = "calm, authoritative, reassuring"


DEFAULT_LAWYER = LawyerCharacterProfile()


def get_character_profile(persona_id: str | None = None) -> LawyerCharacterProfile:
    """페르소나 기반 캐릭터 프로필 조회

    persona_id가 제공되면 향후 DB에서 조회하여 동적 프로필 생성,
    없거나 조회 실패 시 DEFAULT_LAWYER 반환.
    """
    if not persona_id:
        return DEFAULT_LAWYER
    # TODO: persona_db_service에서 persona 조회 → 프로필 변환
    return DEFAULT_LAWYER


# 섹션별 색상/분위기 오버라이드
SECTION_STYLE: dict[str, str] = {
    "hooking": (
        "bold high-contrast lighting, red accent highlights, "
        "dramatic tension, urgent atmosphere"
    ),
    "analysis": (
        "clean informative composition, navy blue dominant, "
        "authoritative professional mood"
    ),
    "advice_cta": (
        "warm golden light, hopeful resolution atmosphere, "
        "comforting professional tone"
    ),
}

# 카메라 앵글 → 영문 설명 매핑
CAMERA_ANGLE_MAP: dict[str, str] = {
    "close-up": "close-up shot focusing on facial expressions",
    "medium": "medium shot showing upper body and gestures",
    "wide": "wide establishing shot showing full environment",
    "over-shoulder": "over-the-shoulder perspective creating intimacy",
    "low-angle": "low angle shot conveying authority and power",
    "high-angle": "high angle shot showing vulnerability",
    "dutch-angle": "dutch angle creating unease and tension",
}

BASE_STYLE = (
    "Korean webtoon manhwa art style, professional legal drama aesthetic, "
    "cel shading with clean line art, cinematic widescreen 16:9 panel, "
    "full-bleed illustration, no text no speech bubbles no captions"
)


def build_image_prompt(
    panel: WebtoonPanel,
    has_reference: bool = False,
) -> str:
    """WebtoonPanel → Gemini 전용 영문 이미지 프롬프트"""
    lawyer = get_character_profile(None)
    section_style = SECTION_STYLE.get(panel.section, SECTION_STYLE["analysis"])
    camera = CAMERA_ANGLE_MAP.get(panel.camera_angle, "medium shot")

    # 캐릭터 설명
    character_desc = (
        f"Main character: {lawyer.gender}, {lawyer.age_range} years old, "
        f"{lawyer.hair}, wearing {lawyer.suit}, {lawyer.build}. "
        f"Expression: {panel.emotion}."
    )
    if has_reference:
        character_desc += (
            " IMPORTANT: Match the character appearance "
            "exactly from the reference image."
        )

    characters_str = (
        ", ".join(panel.characters) if panel.characters else "none"
    )

    prompt = f"""{BASE_STYLE}

Scene: {panel.scene_description}
Location: {panel.location}, {panel.time_of_day}
Visual Focus: {panel.visual_focus}
Camera: {camera}
Mood: {section_style}

{character_desc}

Additional characters: {characters_str}

Legal context keyword: {panel.legal_keyword}

Generate a single high-quality webtoon panel illustration. NO TEXT in the image."""

    return prompt.strip()
