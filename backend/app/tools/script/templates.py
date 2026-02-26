"""대본 생성 프롬프트 템플릿

v2.0: PersonaTone 4종 지원 (PersonaType 하위호환 유지)
"""

from typing import Any

from app.modules.content_marketing.schema import PersonaTone, PersonaType, SectionType

# v1.0 (하위호환)
PERSONA_DESC: dict[PersonaType, str] = {
    PersonaType.PROFESSIONAL: "전문 변호사",
    PersonaType.CASUAL: "친근한 법률 유튜버",
}

TONE_GUIDE: dict[PersonaType, str] = {
    PersonaType.PROFESSIONAL: "경어체, 전문 용어 사용, 신뢰감 있는 톤",
    PersonaType.CASUAL: "반말+존댓말 혼합, 쉬운 비유 활용, 친근하고 편안한 톤",
}

# v2.0 PersonaTone (4종)
PERSONA_TONE_DESC: dict[PersonaTone, str] = {
    PersonaTone.PROFESSIONAL: "전문 변호사",
    PersonaTone.CASUAL: "친근한 법률 유튜버",
    PersonaTone.STORYTELLING: "사건 스토리텔러",
    PersonaTone.EDUCATIONAL: "법률 교육 강사",
}

TONE_GUIDE_V2: dict[PersonaTone, str] = {
    PersonaTone.PROFESSIONAL: "경어체, 전문 용어 사용, 신뢰감 있는 톤",
    PersonaTone.CASUAL: "반말+존댓말 혼합, 쉬운 비유 활용, 친근하고 편안한 톤",
    PersonaTone.STORYTELLING: "극적 구성, 사건 전개 중심, 흥미를 유발하는 서술 톤",
    PersonaTone.EDUCATIONAL: "교과서적 설명, 단계별 해설, 쉽고 정확한 교육 톤",
}

HOOKING_PROMPT = """당신은 법률 유튜브 채널의 {persona_desc}입니다.
다음 주제에 대한 도입부(Hooking)를 작성하세요.

## 요구사항
- 시청자의 관심을 끄는 강렬한 사례/질문으로 시작
- 이 주제가 왜 중요한지 간략히 설명
- {target_words}자 내외로 작성
- {tone_guide}

## 주제
{topic}

## 관련 법령/판례 (참고용)
{rag_context}
"""

ANALYSIS_PROMPT = """당신은 법률 유튜브 채널의 {persona_desc}입니다.
다음 주제에 대한 본론(Legal Analysis)을 작성하세요.

## 요구사항
- 관련 법 조항을 정확히 인용 (예: "민법 제750조에 따르면...")
- 관련 판례를 구체적으로 언급 (예: "대법원 2024다12345 판결에서...")
- 인용 시 [📋 인용: 출처명] 형식으로 마크업
- 법적 쟁점을 명확히 분석
- {target_words}자 내외로 작성
- {tone_guide}
- **중요**: 아래 제공된 법령/판례만 인용하세요. 제공되지 않은 판례를 만들어내지 마세요.

## 주제
{topic}

## 관련 법령 (인용 대상)
{law_context}

## 관련 판례 (인용 대상)
{case_context}
"""

ADVICE_CTA_PROMPT = """당신은 법률 유튜브 채널의 {persona_desc}입니다.
다음 주제에 대한 결론(Advice & CTA)을 작성하세요.

## 요구사항
- 시청자에게 실질적인 조언 제공
- 법률 상담 유도 CTA 문구 포함
- 면책 고지 자연스럽게 포함 ("다만, 개별 사안에 따라 다를 수 있으므로...")
- {target_words}자 내외로 작성
- {tone_guide}

## 주제
{topic}
"""

METADATA_PROMPT = """다음 법률 유튜브 대본의 메타데이터를 생성하세요.

## 대본 주제
{topic}

## 대본 내용 (요약)
{script_summary}

## 요구사항
다음 JSON 형식으로만 응답하세요:
{{
  "description": "영상 설명문 (200~300자, 핵심 내용 요약)",
  "tags": ["SEO 태그 10~15개"],
  "cta_text": "상담 유도 문구 (1문장)",
  "hashtags": ["#해시태그 5개"]
}}
"""

_SECTION_PROMPTS: dict[SectionType, str] = {
    SectionType.HOOKING: HOOKING_PROMPT,
    SectionType.ANALYSIS: ANALYSIS_PROMPT,
    SectionType.ADVICE_CTA: ADVICE_CTA_PROMPT,
}


def _format_rag_context(rag_context: dict[str, Any]) -> str:
    """RAG 검색 결과를 프롬프트용 텍스트로 변환"""
    parts: list[str] = []
    for doc in rag_context.get("laws", [])[:5]:
        title = doc.get("title", "")
        content = doc.get("content", "")[:200]
        parts.append(f"[법령] {title}: {content}")
    for doc in rag_context.get("cases", [])[:5]:
        title = doc.get("title", "")
        case_number = doc.get("case_number", "")
        content = doc.get("content", "")[:200]
        parts.append(f"[판례] {case_number} {title}: {content}")
    return "\n".join(parts) if parts else "(검색 결과 없음)"


def get_section_prompt(
    section_type: SectionType,
    topic: str,
    persona: PersonaType,
    target_words: int,
    rag_context: dict[str, Any],
    persona_tone: PersonaTone | None = None,
) -> str:
    """섹션별 프롬프트 생성

    v2.0: persona_tone이 지정되면 PersonaTone 기반 설명/톤 사용
    """
    template = _SECTION_PROMPTS[section_type]

    if persona_tone is not None:
        persona_desc = PERSONA_TONE_DESC[persona_tone]
        tone_guide = TONE_GUIDE_V2[persona_tone]
    else:
        persona_desc = PERSONA_DESC[persona]
        tone_guide = TONE_GUIDE[persona]

    format_kwargs: dict[str, str | int] = {
        "persona_desc": persona_desc,
        "topic": topic,
        "target_words": target_words,
        "tone_guide": tone_guide,
    }

    if section_type == SectionType.HOOKING:
        format_kwargs["rag_context"] = _format_rag_context(rag_context)
    elif section_type == SectionType.ANALYSIS:
        format_kwargs["law_context"] = _format_rag_context({"laws": rag_context.get("laws", [])})
        format_kwargs["case_context"] = _format_rag_context({"cases": rag_context.get("cases", [])})

    return template.format(**format_kwargs)
