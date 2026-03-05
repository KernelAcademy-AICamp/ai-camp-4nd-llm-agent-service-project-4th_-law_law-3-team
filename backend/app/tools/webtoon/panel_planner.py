"""Chain 1: 대본 → 장면 분할 (Solar Pro2)

대본 텍스트를 웹툰 스토리보드 패널로 분할합니다.
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from app.modules.content_marketing.schema import (
    SectionType,
    WebtoonPanel,
    WebtoonSceneType,
)
from app.tools.llm import get_chat_model

logger = logging.getLogger(__name__)

# 섹션별 패널 수 범위
PANEL_RANGES: dict[str, tuple[int, int]] = {
    "hooking": (2, 3),
    "analysis": (4, 8),
    "advice_cta": (2, 3),
}

# 프롬프트 인젝션 방지
INJECTION_PATTERNS = re.compile(
    r"(ignore previous|system prompt|you are|act as|forget|disregard)",
    re.IGNORECASE,
)


def sanitize_input(text: str, max_length: int = 500) -> str:
    """사용자 입력에서 프롬프트 인젝션 패턴 제거"""
    text = text[:max_length]
    text = INJECTION_PATTERNS.sub("", text)
    return text.strip()


SCENE_SPLIT_PROMPT = """당신은 법률 유튜브 영상 스토리보드 전문가입니다.
다음 대본을 웹툰 스토리보드 패널로 분할해주세요.

## 주제
{topic}

## 대본

### 도입 (Hooking)
{hooking}

### 본론 (Analysis)
{analysis}

### 결론 (Advice & CTA)
{advice_cta}

## 규칙
- 도입: {hooking_min}~{hooking_max}패널
- 본론: {analysis_min}~{analysis_max}패널
- 결론: {cta_min}~{cta_max}패널
- 전체: {total_min}~{total_max}패널

## 씬 타입 (섹션별)
- hooking: hook_shock, hook_question
- analysis: legal_explanation, case_example, conflict_drama, document_closeup
- advice_cta: lawyer_advice, cta_subscribe

## 출력 형식 (JSON 배열)
각 패널에 다음 필드를 포함:
- panel_number, section, scene_type, script_excerpt, scene_description,
  location, time_of_day, characters, emotion, visual_focus, camera_angle, legal_keyword

JSON 배열만 출력하세요. 다른 텍스트는 포함하지 마세요.
"""

# 유효한 씬 타입 집합
VALID_SCENE_TYPES = {e.value for e in WebtoonSceneType}
VALID_SECTIONS = {e.value for e in SectionType}

# 섹션별 허용 씬 타입
SECTION_SCENE_TYPES: dict[str, list[str]] = {
    "hooking": ["hook_shock", "hook_question"],
    "analysis": [
        "legal_explanation",
        "case_example",
        "conflict_drama",
        "document_closeup",
    ],
    "advice_cta": ["lawyer_advice", "cta_subscribe"],
}


def _repair_json(text: str) -> str:
    """LLM 출력의 일반적인 JSON 문법 오류 복구"""
    # 트레일링 쉼표 제거: ,] 또는 ,}
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # 객체 사이 누락된 쉼표: }\n  { → },\n  {
    text = re.sub(r"}\s*\n(\s*)\{", r"},\n\1{", text)
    # 문자열 값 뒤 쉼표 누락: "value"\n  "next_key" → "value",\n  "next_key"
    # (배열 내 객체 프로퍼티 사이에서만 발생)
    text = re.sub(r'(")\s*\n(\s*"[a-z_]+"\s*:)', r'\1,\n\2', text)
    return text


def _extract_individual_panels(text: str) -> list[dict[str, Any]]:
    """개별 패널 객체를 하나씩 추출하는 폴백 파서

    전체 JSON 배열 파싱이 실패해도 유효한 패널을 최대한 복구합니다.
    """
    panels: list[dict[str, Any]] = []
    depth = 0
    start = -1

    for i, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                fragment = text[start : i + 1]
                try:
                    obj = json.loads(fragment)
                    if isinstance(obj, dict) and ("scene_type" in obj or "section" in obj):
                        panels.append(obj)
                except json.JSONDecodeError:
                    # 개별 객체도 깨진 경우 → 복구 시도
                    try:
                        panels.append(json.loads(_repair_json(fragment)))
                    except json.JSONDecodeError:
                        logger.debug("패널 객체 복구 실패 (pos %d-%d)", start, i)
                start = -1

    return panels


def _parse_panels_json(content: str) -> list[dict[str, Any]]:
    """LLM 응답에서 JSON 배열 추출 및 파싱 (3단계 폴백)"""
    text = content.strip()

    # ```json ... ``` 블록 추출
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    # [ ... ] 배열 추출
    if not text.startswith("["):
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            text = bracket_match.group(0)

    # Stage 1: 직접 파싱
    try:
        data = json.loads(text)
        if isinstance(data, list) and data:
            return data
    except json.JSONDecodeError:
        pass

    # Stage 2: JSON 복구 후 재파싱
    try:
        repaired = _repair_json(text)
        data = json.loads(repaired)
        if isinstance(data, list) and data:
            logger.info("JSON 복구 성공 (Stage 2)")
            return data
    except json.JSONDecodeError:
        pass

    # Stage 3: 개별 패널 객체 추출
    panels = _extract_individual_panels(text)
    if panels:
        logger.info("개별 패널 추출 성공 (Stage 3): %d개", len(panels))
        return panels

    msg = "장면 분할 결과 파싱 실패: 유효한 JSON을 추출할 수 없습니다."
    raise ValueError(msg)


_FIELD_MAX_LENGTHS: dict[str, int] = {
    "script_excerpt": 500,
    "scene_description": 300,
    "location": 100,
    "time_of_day": 50,
    "emotion": 50,
    "visual_focus": 100,
    "camera_angle": 50,
    "legal_keyword": 100,
}

_MAX_CHARACTERS = 5


def _validate_and_fix_panel(panel: dict[str, Any], idx: int) -> dict[str, Any]:
    """패널 데이터 검증 및 보정"""
    # panel_number 보정
    panel.setdefault("panel_number", idx + 1)
    try:
        panel["panel_number"] = max(1, min(14, int(panel["panel_number"])))
    except (ValueError, TypeError):
        panel["panel_number"] = idx + 1

    # section 보정
    section = panel.get("section", "analysis")
    if section not in VALID_SECTIONS:
        section = "analysis"
    panel["section"] = section

    # scene_type 보정
    scene_type = panel.get("scene_type", "")
    allowed = SECTION_SCENE_TYPES.get(section, [])
    if scene_type not in VALID_SCENE_TYPES or scene_type not in allowed:
        panel["scene_type"] = allowed[0] if allowed else "legal_explanation"

    # 필수 문자열 필드: 기본값 + max_length 초과분 자르기
    for field, max_len in _FIELD_MAX_LENGTHS.items():
        value = panel.get(field)
        if not value or not isinstance(value, str):
            panel[field] = "미정"
        elif len(value) > max_len:
            panel[field] = value[:max_len]

    # characters: 리스트 보정 + 최대 개수 제한
    chars = panel.get("characters")
    if not isinstance(chars, list):
        panel["characters"] = []
    else:
        panel["characters"] = [
            str(c) for c in chars[:_MAX_CHARACTERS] if c
        ]

    return panel


_MAX_RETRIES = 2


async def _try_split_once(
    llm: Any,
    prompt: str,
) -> list[WebtoonPanel]:
    """단일 장면 분할 시도 (LLM 호출 → 파싱 → 검증)"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    content = response.content if isinstance(response.content, str) else str(response.content)
    panels_data = _parse_panels_json(content)

    panels: list[WebtoonPanel] = []
    for idx, raw in enumerate(panels_data):
        fixed = _validate_and_fix_panel(raw, idx)
        try:
            panels.append(WebtoonPanel(**fixed))
        except Exception as e:
            logger.warning("패널 %d 생성 실패: %s | 데이터: %s", idx + 1, e, fixed)

    return panels


async def split_script_to_scenes(
    topic: str,
    sections: dict[str, str],
    target_panels: int | None = None,
    on_progress: Any | None = None,
) -> list[WebtoonPanel]:
    """대본을 웹툰 패널로 분할 (Solar Pro2)"""
    # 입력 sanitize
    safe_topic = sanitize_input(topic, max_length=500)
    safe_sections = {
        k: sanitize_input(v, max_length=2000)
        for k, v in sections.items()
        if k in ("hooking", "analysis", "advice_cta")
    }

    # 패널 수 계산
    if target_panels:
        total_min = total_max = target_panels
    else:
        total_min = sum(r[0] for r in PANEL_RANGES.values())  # 8
        total_max = sum(r[1] for r in PANEL_RANGES.values())  # 14

    llm = get_chat_model(provider="upstage", temperature=0.3)
    prompt = SCENE_SPLIT_PROMPT.format(
        topic=safe_topic,
        hooking=safe_sections.get("hooking", ""),
        analysis=safe_sections.get("analysis", ""),
        advice_cta=safe_sections.get("advice_cta", ""),
        hooking_min=PANEL_RANGES["hooking"][0],
        hooking_max=PANEL_RANGES["hooking"][1],
        analysis_min=PANEL_RANGES["analysis"][0],
        analysis_max=PANEL_RANGES["analysis"][1],
        cta_min=PANEL_RANGES["advice_cta"][0],
        cta_max=PANEL_RANGES["advice_cta"][1],
        total_min=total_min,
        total_max=total_max,
    )

    # 재시도 로직: LLM 응답이 간헐적으로 불완전할 수 있음
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            if on_progress:
                on_progress("llm_start", attempt + 1)
            panels = await _try_split_once(llm, prompt)
            if on_progress:
                on_progress("llm_done", attempt + 1)
            if panels:
                if on_progress:
                    on_progress("validate", attempt + 1)
                # panel_number 재정렬
                for i, panel in enumerate(panels):
                    panel.panel_number = i + 1
                logger.info("장면 분할 완료: %d패널 (시도 %d/%d)", len(panels), attempt + 1, _MAX_RETRIES + 1)
                return panels
            logger.warning("유효한 패널 0개 (시도 %d/%d), 재시도합니다.", attempt + 1, _MAX_RETRIES + 1)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning("장면 분할 실패 (시도 %d/%d): %s", attempt + 1, _MAX_RETRIES + 1, e)

    msg = f"유효한 패널이 생성되지 않았습니다. (총 {_MAX_RETRIES + 1}회 시도, 마지막 오류: {last_error})"
    raise ValueError(msg)
