# 대본 + 웹툰 스토리보드 설계서

> **기능명**: 대본 생성 시 웹툰 스토리보드 동시 생성
> **PDCA Phase**: Design
> **기획서**: `docs/01-plan/features/script-storyboard.plan.md` (v1.0)
> **작성일**: 2026-02-28
> **상태**: v1.1 — Gemini CLI + Codex CLI 설계 리뷰 반영 완료

---

## 1. Backend 상세 설계

### 1.1 Pydantic 스키마 (`backend/app/modules/content_marketing/schema/__init__.py` 추가)

```python
# ── Webtoon Storyboard Enums ──

class WebtoonSceneType(str, Enum):
    """웹툰 씬 타입 (8종)"""
    HOOK_SHOCK = "hook_shock"
    HOOK_QUESTION = "hook_question"
    LEGAL_EXPLANATION = "legal_explanation"
    CASE_EXAMPLE = "case_example"
    CONFLICT_DRAMA = "conflict_drama"
    DOCUMENT_CLOSEUP = "document_closeup"
    LAWYER_ADVICE = "lawyer_advice"
    CTA_SUBSCRIBE = "cta_subscribe"

class ImageStatus(str, Enum):
    """이미지 생성 상태 (Codex 리뷰 반영: retrying 추가)"""
    PENDING = "pending"
    GENERATING = "generating"
    RETRYING = "retrying"  # 재시도 중 (Codex 리뷰)
    COMPLETED = "completed"
    ERROR = "error"

# ── Webtoon Panel ──

class WebtoonPanel(BaseModel):
    """웹툰 패널 (장면 분할 결과 + 이미지 생성 결과)"""
    panel_number: int = Field(ge=1, le=14)
    section: SectionType
    scene_type: WebtoonSceneType
    script_excerpt: str = Field(max_length=500)
    scene_description: str = Field(max_length=300)
    location: str = Field(max_length=100)
    time_of_day: str = Field(max_length=50)
    characters: list[str] = Field(max_length=5)
    emotion: str = Field(max_length=50)
    visual_focus: str = Field(max_length=100)
    camera_angle: str = Field(max_length=50)
    legal_keyword: str = Field(max_length=100)
    image_prompt: str | None = None
    image_url: str | None = None
    image_status: ImageStatus = ImageStatus.PENDING
    error_message: str | None = None
    # 메타데이터 (운영 추적)
    model_version: str | None = None
    prompt_version: str | None = None
    generation_cost_ms: int | None = None
    safety_flags: list[str] = Field(default_factory=list)

# ── Request / Response ──

class WebtoonGenerateRequest(BaseModel):
    """웹툰 스토리보드 생성 요청"""
    topic: str = Field(min_length=2, max_length=500)
    sections: dict[str, str]  # {"hooking": "...", "analysis": "...", "advice_cta": "..."}
    persona: PersonaType = PersonaType.PROFESSIONAL
    persona_id: str | None = None
    panel_count: int | None = Field(default=None, ge=4, le=14)

    @model_validator(mode="after")
    def validate_sections(self) -> "WebtoonGenerateRequest":
        """필수 섹션 키 검증"""
        required = {"hooking", "analysis", "advice_cta"}
        if not required.issubset(self.sections.keys()):
            missing = required - self.sections.keys()
            msg = f"필수 섹션 누락: {missing}"
            raise ValueError(msg)
        # 빈 섹션 검증
        for key in required:
            if not self.sections[key].strip():
                msg = f"섹션 '{key}'의 내용이 비어있습니다."
                raise ValueError(msg)
        return self

class WebtoonJobResponse(BaseModel):
    """웹툰 Job 생성 응답"""
    job_id: str
    status: str = "accepted"
    estimated_panels: int

class WebtoonStreamEvent(BaseModel):
    """SSE 스트리밍 이벤트"""
    event: str  # scene_split_start | scene_split_done | panel_start | panel_complete | panel_failed | all_done | error
    panel_number: int | None = None
    total_panels: int | None = None
    section: str | None = None
    caption: str | None = None
    scene_description: str | None = None
    image_url: str | None = None
    error: str | None = None

class WebtoonJobStatusResponse(BaseModel):
    """웹툰 Job 상태 폴링 응답"""
    job_id: str
    status: str  # pending | processing | completed | failed
    progress: int  # 0-100
    panels: list[WebtoonPanel] = Field(default_factory=list)
    error: str | None = None

class WebtoonRegenerateRequest(BaseModel):
    """개별 패널 재생성 요청"""
    panel_number: int = Field(ge=1, le=14)
```

### 1.2 API 엔드포인트 상세 (`backend/app/modules/content_marketing/router/__init__.py` 추가)

```python
# ── Webtoon Storyboard API (신규) ──

@router.post("/script/webtoon", response_model=WebtoonJobResponse)
async def create_webtoon_job(
    request: WebtoonGenerateRequest,
) -> WebtoonJobResponse:
    """웹툰 스토리보드 Job 생성 → 백그라운드 파이프라인 시작"""
    job_id = webtoon_job_manager.create_job(total_steps=1)
    asyncio.create_task(
        run_webtoon_pipeline(job_id, request)
    )
    return WebtoonJobResponse(
        job_id=job_id,
        status="accepted",
        estimated_panels=request.panel_count or 10,
    )

@router.get("/script/webtoon/{job_id}/stream")
async def stream_webtoon_progress(job_id: str) -> StreamingResponse:
    """SSE로 패널별 생성 진행률 스트리밍 (Codex 리뷰: 30초 heartbeat 추가)"""
    if not webtoon_job_manager.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job을 찾을 수 없습니다.")
    return StreamingResponse(
        _webtoon_sse_generator(job_id),  # 내부에서 30초 간격 `:heartbeat\n\n` 전송
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

@router.get("/script/webtoon/{job_id}", response_model=WebtoonJobStatusResponse)
async def get_webtoon_job_status(job_id: str) -> WebtoonJobStatusResponse:
    """Job 상태 폴링 조회"""
    job = webtoon_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job을 찾을 수 없습니다.")
    return WebtoonJobStatusResponse(
        job_id=job_id,
        status=job.status.value,
        progress=job.progress,
        panels=job.result.get("panels", []) if job.result else [],
        error=job.error,
    )

@router.post("/script/webtoon/{job_id}/regenerate")
async def regenerate_webtoon_panel(
    job_id: str,
    request: WebtoonRegenerateRequest,
) -> WebtoonStreamEvent:
    """개별 패널 이미지 재생성"""
    job = webtoon_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job을 찾을 수 없습니다.")
    result = await regenerate_single_panel(job_id, request.panel_number)
    return result
```

### 1.3 서비스 레이어 (`backend/app/services/service_function/webtoon_service.py` 신규)

```python
"""웹툰 스토리보드 파이프라인 서비스"""

import asyncio
import logging
import time
from pathlib import Path

from app.modules.content_marketing.schema import (
    ImageStatus,
    WebtoonGenerateRequest,
    WebtoonPanel,
    WebtoonStreamEvent,
)
from app.modules.storyboard.service.job_manager import JobManager, JobStatus
from app.tools.webtoon.panel_planner import split_script_to_scenes
from app.tools.webtoon.prompt_builder import build_image_prompt
from app.tools.webtoon.image_generator import generate_webtoon_image, generate_placeholder

logger = logging.getLogger(__name__)

# storyboard 모듈의 JobManager를 재사용 (별도 인스턴스)
# → Gemini 리뷰: 공통 모듈 분리 검토했으나, 현재 동일 클래스 import로 충분
webtoon_job_manager = JobManager()

# 동시 이미지 생성 제한
_semaphore = asyncio.Semaphore(3)

async def run_webtoon_pipeline(
    job_id: str,
    request: WebtoonGenerateRequest,
) -> None:
    """웹툰 스토리보드 전체 파이프라인 실행 (백그라운드)

    Chain 1: 대본 → 장면 분할 (Solar Pro2)
    Chain 2: 장면 → 이미지 프롬프트 (텍스트 조합)
    Chain 3: 이미지 생성 (Gemini 3 Pro Image)
    """
    try:
        await webtoon_job_manager.update_progress(
            job_id, status=JobStatus.PROCESSING, message="장면 분할 중..."
        )

        # Chain 1: 대본 → 장면 분할
        panels: list[WebtoonPanel] = await split_script_to_scenes(
            topic=request.topic,
            sections=request.sections,
            target_panels=request.panel_count,
        )

        total = len(panels)
        await webtoon_job_manager.update_progress(
            job_id, message=f"장면 분할 완료 ({total}패널). 이미지 생성 시작..."
        )

        # Chain 2 + Chain 3: 순차 이미지 생성 (캐릭터 일관성)
        # Gemini 리뷰 반영: 첫 패널 실패 시 2번째 성공 패널을 레퍼런스로 사용
        reference_image: bytes | None = None

        for idx, panel in enumerate(panels):
            panel.image_status = ImageStatus.GENERATING

            # Chain 2: 이미지 프롬프트 빌드
            panel.image_prompt = build_image_prompt(panel, reference_image is not None)
            panel.prompt_version = "1.0"

            # Chain 3: 이미지 생성
            start_ms = time.monotonic_ns() // 1_000_000
            async with _semaphore:
                result = await generate_webtoon_image(
                    panel=panel,
                    reference_image=reference_image,
                    job_id=job_id,
                )

            elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms
            panel.generation_cost_ms = elapsed_ms

            if result["success"]:
                panel.image_url = result["image_url"]
                panel.image_status = ImageStatus.COMPLETED
                panel.model_version = result.get("model_version", "gemini-3-pro-image-preview")
                panel.safety_flags = result.get("safety_flags", [])
                # 첫 성공 패널을 레퍼런스로 저장 (첫 패널 실패 시 다음 성공 패널 사용)
                if reference_image is None and result.get("image_data"):
                    reference_image = result["image_data"]
            else:
                panel.image_status = ImageStatus.ERROR
                panel.error_message = result.get("error", "이미지 생성 실패")

            await webtoon_job_manager.update_progress(
                job_id,
                current_step=idx + 1,
                message=f"이미지 생성 중... ({idx + 1}/{total})",
            )

        # 완료
        await webtoon_job_manager.complete_job(
            job_id,
            result={"panels": [p.model_dump() for p in panels]},
        )

    except Exception as e:
        logger.error(f"웹툰 파이프라인 오류 (job_id={job_id}): {e}", exc_info=True)
        await webtoon_job_manager.fail_job(job_id, str(e))
```

### 1.4 AI 파이프라인 상세

#### Chain 1: 장면 분할 (`backend/app/tools/webtoon/panel_planner.py` 신규)

```python
"""Chain 1: 대본 → 장면 분할 (Solar Pro2)"""

from app.modules.content_marketing.schema import SectionType, WebtoonPanel, WebtoonSceneType
from app.tools.llm import get_chat_model

# 섹션별 패널 수 범위
PANEL_RANGES: dict[str, tuple[int, int]] = {
    "hooking": (2, 3),
    "analysis": (4, 8),
    "advice_cta": (2, 3),
}

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

async def split_script_to_scenes(
    topic: str,
    sections: dict[str, str],
    target_panels: int | None = None,
) -> list[WebtoonPanel]:
    """대본을 웹툰 패널로 분할 (Solar Pro2)"""
    # 패널 수 계산
    if target_panels:
        total_min = total_max = target_panels
    else:
        total_min = sum(r[0] for r in PANEL_RANGES.values())  # 8
        total_max = sum(r[1] for r in PANEL_RANGES.values())  # 14

    llm = get_chat_model(provider="upstage")
    prompt = SCENE_SPLIT_PROMPT.format(
        topic=topic,
        hooking=sections["hooking"],
        analysis=sections["analysis"],
        advice_cta=sections["advice_cta"],
        hooking_min=PANEL_RANGES["hooking"][0],
        hooking_max=PANEL_RANGES["hooking"][1],
        analysis_min=PANEL_RANGES["analysis"][0],
        analysis_max=PANEL_RANGES["analysis"][1],
        cta_min=PANEL_RANGES["advice_cta"][0],
        cta_max=PANEL_RANGES["advice_cta"][1],
        total_min=total_min,
        total_max=total_max,
    )

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    # JSON 파싱 + WebtoonPanel 변환
    panels_data = _parse_panels_json(response.content)
    return [WebtoonPanel(**p) for p in panels_data]
```

#### Chain 2: 이미지 프롬프트 빌더 (`backend/app/tools/webtoon/prompt_builder.py` 신규)

```python
"""Chain 2: 장면 → 이미지 프롬프트 (LLM 불필요, 텍스트 조합)"""

from dataclasses import dataclass
from app.modules.content_marketing.schema import SectionType, WebtoonPanel

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
    """페르소나 기반 캐릭터 프로필 조회 (Gemini 리뷰 반영)

    persona_id가 제공되면 DB에서 조회하여 동적 프로필 생성,
    없거나 조회 실패 시 DEFAULT_LAWYER 반환.
    """
    if not persona_id:
        return DEFAULT_LAWYER
    # TODO: persona_db_service에서 persona 조회 → 프로필 변환
    # persona = await get_persona_by_id(persona_id)
    # return LawyerCharacterProfile(gender=..., ...)
    return DEFAULT_LAWYER

# 섹션별 색상/분위기 오버라이드
SECTION_STYLE: dict[str, str] = {
    "hooking": "bold high-contrast lighting, red accent highlights, dramatic tension, urgent atmosphere",
    "analysis": "clean informative composition, navy blue dominant, authoritative professional mood",
    "advice_cta": "warm golden light, hopeful resolution atmosphere, comforting professional tone",
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
    lawyer = get_character_profile(getattr(panel, '_persona_id', None))
    section_style = SECTION_STYLE.get(panel.section, SECTION_STYLE["analysis"])
    camera = CAMERA_ANGLE_MAP.get(panel.camera_angle, "medium shot")

    # 캐릭터 설명
    character_desc = (
        f"Main character: {lawyer.gender}, {lawyer.age_range} years old, "
        f"{lawyer.hair}, wearing {lawyer.suit}, {lawyer.build}. "
        f"Expression: {panel.emotion}."
    )
    if has_reference:
        character_desc += " IMPORTANT: Match the character appearance exactly from the reference image."

    prompt = f"""{BASE_STYLE}

Scene: {panel.scene_description}
Location: {panel.location}, {panel.time_of_day}
Visual Focus: {panel.visual_focus}
Camera: {camera}
Mood: {section_style}

{character_desc}

Additional characters: {', '.join(panel.characters) if panel.characters else 'none'}

Legal context keyword: {panel.legal_keyword}

Generate a single high-quality webtoon panel illustration. NO TEXT in the image."""

    return prompt.strip()
```

#### Chain 3: 이미지 생성 (`backend/app/tools/webtoon/image_generator.py` 신규)

```python
"""Chain 3: 나노바나나(Gemini 3 Pro Image) 이미지 생성"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image

from app.core.config import settings
from app.modules.content_marketing.schema import WebtoonPanel

logger = logging.getLogger(__name__)

IMAGES_DIR = Path(__file__).parent.parent.parent.parent / "data" / "media" / "webtoon" / "images"
MAX_RETRIES = 2

def _ensure_dirs(job_id: str) -> Path:
    """Job별 이미지 디렉토리 생성"""
    job_dir = IMAGES_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir

def _get_cache_key(panel: WebtoonPanel, job_id: str) -> str:
    """해시 기반 캐시 키 (동일 대본 재생성 방지)"""
    content = f"{job_id}:{panel.panel_number}:{panel.scene_description}:{panel.image_prompt}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]

async def generate_webtoon_image(
    panel: WebtoonPanel,
    reference_image: bytes | None = None,
    job_id: str = "",
) -> dict[str, Any]:
    """패널 이미지 생성 (재시도 2회, 실패 시 플레이스홀더)"""
    job_dir = _ensure_dirs(job_id)

    for attempt in range(MAX_RETRIES + 1):
        try:
            client = genai.Client(api_key=settings.GOOGLE_API_KEY)

            # 프롬프트 + 레퍼런스 이미지 구성
            contents: list[Any] = []
            if reference_image:
                contents.append(types.Part.from_bytes(
                    data=reference_image,
                    mime_type="image/png",
                ))
                contents.append(
                    "Using the character appearance from this reference image, "
                    f"generate the following scene:\n\n{panel.image_prompt}"
                )
            else:
                contents.append(panel.image_prompt or "")

            model_name = getattr(settings, "STORYBOARD_IMAGE_MODEL", "gemini-3-pro-image-preview")

            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )

            # 이미지 추출
            if not response.candidates or not response.candidates[0].content:
                raise ValueError("이미지 생성 결과 없음")

            image_data = None
            safety_flags: list[str] = []
            content = response.candidates[0].content
            assert content is not None and content.parts is not None
            for part in content.parts:
                if part.inline_data and part.inline_data.data:
                    image_data = part.inline_data.data
                    break

            if not image_data:
                raise ValueError("생성된 이미지를 찾을 수 없음")

            # WebP 변환 + 저장
            filename = f"panel_{panel.panel_number:03d}.webp"
            image_path = job_dir / filename
            _save_as_webp(image_data, image_path)

            return {
                "success": True,
                "image_url": f"/media/webtoon/images/{job_id}/{filename}",
                "image_data": image_data,
                "model_version": model_name,
                "safety_flags": safety_flags,
            }

        except Exception as e:
            logger.warning(
                f"이미지 생성 시도 {attempt + 1}/{MAX_RETRIES + 1} 실패 "
                f"(panel={panel.panel_number}): {e}"
            )
            if attempt == MAX_RETRIES:
                # 최종 실패 → 플레이스홀더
                return await generate_placeholder(panel, job_dir, job_id)

    return await generate_placeholder(panel, job_dir, job_id)

async def generate_placeholder(
    panel: WebtoonPanel,
    job_dir: Path,
    job_id: str,
) -> dict[str, Any]:
    """PIL 플레이스홀더 이미지 생성"""
    filename = f"panel_{panel.panel_number:03d}_placeholder.webp"
    image_path = job_dir / filename
    img = Image.new("RGB", (768, 432), color=(30, 41, 59))
    img.save(image_path, "WEBP", quality=80)
    return {
        "success": False,
        "image_url": f"/media/webtoon/images/{job_id}/{filename}",
        "error": "이미지 생성 실패 (플레이스홀더)",
        "is_placeholder": True,
    }

def _save_as_webp(image_data: bytes, path: Path) -> None:
    """PNG/JPEG → WebP 변환 저장"""
    import io
    img = Image.open(io.BytesIO(image_data))
    img.save(path, "WEBP", quality=85)
```

### 1.5 환경변수 추가 (`backend/app/core/config.py`)

```python
# Webtoon Storyboard (콘텐츠 마케팅)
STORYBOARD_IMAGE_MODEL: str = "gemini-3-pro-image-preview"
STORYBOARD_MAX_PANELS: int = 14
STORYBOARD_IMAGE_FORMAT: str = "webp"
STORYBOARD_CACHE_TTL: int = 604800  # 7일 (초)
STORYBOARD_MAX_CONCURRENT: int = 3
```

### 1.6 StaticFiles 마운트 (`backend/app/main.py` 수정)

```python
# 기존 storyboard 마운트와 별도로 webtoon 전용 마운트 추가
from fastapi.staticfiles import StaticFiles

webtoon_media_dir = Path(__file__).parent.parent / "data" / "media" / "webtoon"
webtoon_media_dir.mkdir(parents=True, exist_ok=True)
app.mount("/media/webtoon", StaticFiles(directory=str(webtoon_media_dir)), name="webtoon-media")
```

---

## 2. Frontend 상세 설계

### 2.1 TypeScript 타입 (`frontend/src/features/content-marketing/types/index.ts` 추가)

```typescript
// ── Webtoon Storyboard Types ──

export type WebtoonSceneType =
  | 'hook_shock' | 'hook_question'
  | 'legal_explanation' | 'case_example' | 'conflict_drama' | 'document_closeup'
  | 'lawyer_advice' | 'cta_subscribe'

export type WebtoonImageStatus = 'pending' | 'generating' | 'retrying' | 'completed' | 'error'

export interface WebtoonPanel {
  panel_number: number
  section: SectionType
  scene_type: WebtoonSceneType
  script_excerpt: string
  scene_description: string
  location: string
  time_of_day: string
  characters: string[]
  emotion: string
  visual_focus: string
  camera_angle: string
  legal_keyword: string
  image_prompt: string | null
  image_url: string | null
  image_status: WebtoonImageStatus
  error_message: string | null
  model_version: string | null
  prompt_version: string | null
  generation_cost_ms: number | null
  safety_flags: string[]
}

export interface WebtoonGenerateRequest {
  topic: string
  sections: Record<string, string>
  persona: PersonaType
  persona_id?: string | null
  panel_count?: number | null
}

export interface WebtoonJobResponse {
  job_id: string
  status: string
  estimated_panels: number
}

export interface WebtoonStreamEvent {
  event: 'scene_split_start' | 'scene_split_done' | 'panel_start' | 'panel_complete' | 'panel_failed' | 'all_done' | 'error'
  panel_number: number | null
  total_panels: number | null
  section: string | null
  caption: string | null
  scene_description: string | null
  image_url: string | null
  error: string | null
}

export interface WebtoonJobStatus {
  job_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  panels: WebtoonPanel[]
  error: string | null
}
```

### 2.2 서비스 함수 (`frontend/src/features/content-marketing/services/index.ts` 추가)

```typescript
// ── Webtoon Storyboard API ──

/** 웹툰 스토리보드 Job 생성 */
export async function createWebtoonJob(
  request: WebtoonGenerateRequest,
): Promise<WebtoonJobResponse> {
  const { data } = await api.post<WebtoonJobResponse>(
    `${BASE}/script/webtoon`,
    request,
  )
  return data
}

/** 웹툰 Job 상태 폴링 */
export async function getWebtoonJobStatus(
  jobId: string,
): Promise<WebtoonJobStatus> {
  const { data } = await api.get<WebtoonJobStatus>(
    `${BASE}/script/webtoon/${jobId}`,
  )
  return data
}

/** 웹툰 SSE 스트리밍 */
export function streamWebtoonProgress(
  jobId: string,
  onEvent: (event: WebtoonStreamEvent) => void,
  onError: (error: string) => void,
  onDone: () => void,
): AbortController {
  const controller = new AbortController()
  const url = `/api${BASE}/script/webtoon/${jobId}/stream`

  fetch(url, {
    method: 'GET',
    headers: { Accept: 'text/event-stream' },
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        onError(`서버 오류: ${response.status}`)
        return
      }
      const reader = response.body?.getReader()
      if (!reader) { onError('스트리밍 불가'); return }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6)) as WebtoonStreamEvent
              if (event.event === 'all_done') { onDone() }
              else if (event.event === 'error') { onError(event.error || '알 수 없는 오류') }
              else { onEvent(event) }
            } catch { /* ignore */ }
          }
        }
      }
    })
    .catch((error) => {
      if (error instanceof Error && error.name !== 'AbortError') {
        // Codex 리뷰 반영: SSE 연결 끊김 시 자동 재연결 (최대 3회)
        if (retryCount < MAX_SSE_RETRIES) {
          retryCount++
          setTimeout(() => connectSSE(), 2000 * retryCount)  // 지수 백오프
        } else {
          onError(error.message)
        }
      }
    })

  return controller
}

/** 개별 패널 재생성 */
export async function regenerateWebtoonPanel(
  jobId: string,
  panelNumber: number,
): Promise<WebtoonStreamEvent> {
  const { data } = await api.post<WebtoonStreamEvent>(
    `${BASE}/script/webtoon/${jobId}/regenerate`,
    { panel_number: panelNumber },
  )
  return data
}
```

### 2.3 커스텀 훅 (`frontend/src/features/content-marketing/hooks/useStoryboardStream.ts` 신규)

```typescript
'use client'

import { useCallback, useRef, useState } from 'react'
import type {
  WebtoonGenerateRequest,
  WebtoonJobStatus,
  WebtoonPanel,
  WebtoonStreamEvent,
  SectionType,
} from '../types'
import {
  createWebtoonJob,
  getWebtoonJobStatus,
  streamWebtoonProgress,
  regenerateWebtoonPanel,
} from '../services'

export type StoryboardPhase =
  | 'idle'           // 초기 상태
  | 'scene_split'    // Chain 1: 장면 분할 중
  | 'generating'     // Chain 2+3: 이미지 생성 중
  | 'completed'      // 전체 완료
  | 'error'          // 오류

export interface StoryboardProgress {
  percent: number
  label: string
  currentPanel: number
  totalPanels: number
}

export function useStoryboardStream() {
  const [panels, setPanels] = useState<WebtoonPanel[]>([])
  const [phase, setPhase] = useState<StoryboardPhase>('idle')
  const [progress, setProgress] = useState<StoryboardProgress>({
    percent: 0, label: '', currentPanel: 0, totalPanels: 0,
  })
  const [error, setError] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)

  const controllerRef = useRef<AbortController | null>(null)

  const startGeneration = useCallback(async (request: WebtoonGenerateRequest) => {
    // 이전 작업 정리
    controllerRef.current?.abort()
    setPanels([])
    setPhase('scene_split')
    setError(null)
    setProgress({ percent: 5, label: '장면 분할 중...', currentPanel: 0, totalPanels: 0 })

    try {
      // Job 생성
      const job = await createWebtoonJob(request)
      setJobId(job.job_id)

      // SSE 스트리밍 시작
      const controller = streamWebtoonProgress(
        job.job_id,
        (event: WebtoonStreamEvent) => {
          if (event.event === 'scene_split_done') {
            setPhase('generating')
            setProgress(prev => ({
              ...prev, percent: 15, label: '이미지 생성 시작...',
              totalPanels: event.total_panels || 0,
            }))
          } else if (event.event === 'panel_complete' && event.panel_number != null) {
            setPanels(prev => {
              const updated = [...prev]
              const idx = updated.findIndex(p => p.panel_number === event.panel_number)
              if (idx >= 0) {
                updated[idx] = {
                  ...updated[idx],
                  image_url: event.image_url || null,
                  image_status: 'completed',
                }
              }
              return updated
            })
            const total = event.total_panels || 1
            const pct = 15 + (85 * (event.panel_number / total))
            setProgress({
              percent: Math.round(pct),
              label: `이미지 생성 중... (${event.panel_number}/${total})`,
              currentPanel: event.panel_number,
              totalPanels: total,
            })
          } else if (event.event === 'panel_failed' && event.panel_number != null) {
            setPanels(prev => {
              const updated = [...prev]
              const idx = updated.findIndex(p => p.panel_number === event.panel_number)
              if (idx >= 0) {
                updated[idx] = {
                  ...updated[idx],
                  image_status: 'error',
                  error_message: event.error || '생성 실패',
                }
              }
              return updated
            })
          }
        },
        (errorMsg: string) => {
          setError(errorMsg)
          setPhase('error')
        },
        () => {
          setPhase('completed')
          setProgress(prev => ({ ...prev, percent: 100, label: '스토리보드 완료!' }))
        },
      )
      controllerRef.current = controller

    } catch (err) {
      setError(err instanceof Error ? err.message : '스토리보드 생성 실패')
      setPhase('error')
    }
  }, [])

  const regeneratePanel = useCallback(async (panelNumber: number) => {
    if (!jobId) return
    setPanels(prev => prev.map(p =>
      p.panel_number === panelNumber
        ? { ...p, image_status: 'generating' as const, error_message: null }
        : p
    ))
    try {
      const result = await regenerateWebtoonPanel(jobId, panelNumber)
      setPanels(prev => prev.map(p =>
        p.panel_number === panelNumber
          ? { ...p, image_url: result.image_url, image_status: 'completed' as const }
          : p
      ))
    } catch {
      setPanels(prev => prev.map(p =>
        p.panel_number === panelNumber
          ? { ...p, image_status: 'error' as const, error_message: '재생성 실패' }
          : p
      ))
    }
  }, [jobId])

  const stopGeneration = useCallback(() => {
    controllerRef.current?.abort()
    setPhase('idle')
  }, [])

  const reset = useCallback(() => {
    controllerRef.current?.abort()
    setPanels([])
    setPhase('idle')
    setError(null)
    setJobId(null)
    setProgress({ percent: 0, label: '', currentPanel: 0, totalPanels: 0 })
  }, [])

  return {
    panels, phase, progress, error, jobId,
    startGeneration, regeneratePanel, stopGeneration, reset,
  }
}
```

### 2.4 컴포넌트 설계

#### ScriptStoryboardSplitView.tsx (최상위 분할 뷰)

```
┌──────────────────────────────────────────────────────────────────┐
│ ScriptStoryboardSplitView                                        │
│                                                                  │
│  Desktop (md+):                                                  │
│  ┌─────────── 45% ──────────┬──────────── 55% ──────────────┐  │
│  │ ScriptSectionList        │ StoryboardPanel                │  │
│  │  ├─ ref="hooking"        │  ├─ StoryboardPanelCard #1     │  │
│  │  │  대본 도입 텍스트      │  │  [이미지 / 스켈레톤]        │  │
│  │  ├─ ref="analysis"       │  ├─ StoryboardPanelCard #2     │  │
│  │  │  대본 본론 텍스트      │  │  [이미지 / 스켈레톤]        │  │
│  │  └─ ref="advice_cta"     │  ├─ ...                        │  │
│  │     대본 결론 텍스트      │  └─ StoryboardPanelCard #N     │  │
│  └──────────────────────────┴────────────────────────────────┘  │
│                                                                  │
│  Tablet (<md):  [대본] | [스토리보드]  탭 전환                    │
│  Mobile (<sm):  스택형 + StoryboardPanel 하단 슬라이드업           │
└──────────────────────────────────────────────────────────────────┘
```

**Props:**
```typescript
interface ScriptStoryboardSplitViewProps {
  // 대본 관련 (useScript에서 전달)
  sections: Record<SectionType, string>
  currentSection: SectionType | null
  isGenerating: boolean
  stageInfo: StageInfo | null
  // 스토리보드 관련 (useStoryboardStream에서 전달)
  panels: WebtoonPanel[]
  storyboardPhase: StoryboardPhase
  storyboardProgress: StoryboardProgress
  onRegeneratePanel: (panelNumber: number) => void
}
```

**동작:**
1. `useScript.generate()` → 대본 SSE 완료 시 → 자동으로 `useStoryboardStream.startGeneration()` 트리거
2. 대본 생성 중에는 우측에 "대본 생성 완료 후 스토리보드가 표시됩니다" 메시지
3. 스토리보드 생성 중에는 스켈레톤 + 순차 이미지 표시

#### StoryboardPanelCard.tsx (개별 패널 카드)

```typescript
interface StoryboardPanelCardProps {
  panel: WebtoonPanel
  isHighlighted: boolean  // 현재 스크롤 위치에 해당하는 패널
  onRegenerate: () => void
  onImageClick: () => void  // 확대 보기
}
```

**상태별 렌더링:**
| image_status | 표시 |
|------|------|
| `pending` | 회색 스켈레톤 + 패널 번호 |
| `generating` | 펄스 애니메이션 스켈레톤 + 스피너 |
| `completed` | 웹툰 이미지 + 캡션 + 재생성 버튼 |
| `error` | 에러 메시지 + 다시 시도 버튼 |

#### 스크롤 동기화 로직

```typescript
// ScriptStoryboardSplitView 내부
const sectionRefs = useRef<Record<SectionType, HTMLDivElement | null>>({
  hooking: null, analysis: null, advice_cta: null,
})

// IntersectionObserver: 대본 섹션 가시성 → 스토리보드 패널 하이라이트
useEffect(() => {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          const section = entry.target.getAttribute('data-section') as SectionType
          setActiveSection(section)
          // 해당 섹션의 첫 패널로 우측 스크롤
          const firstPanel = panels.find(p => p.section === section)
          if (firstPanel) {
            document.getElementById(`panel-${firstPanel.panel_number}`)
              ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
          }
        }
      }
    },
    { threshold: 0.3 }
  )
  // ...observe sectionRefs
}, [panels])
```

---

## 3. 파일 변경 매트릭스

### 3.1 Backend 신규 파일 (5개)

| 파일 | 역할 | 예상 줄 수 |
|------|------|----------|
| `backend/app/tools/webtoon/__init__.py` | 패키지 init | 5 |
| `backend/app/tools/webtoon/panel_planner.py` | Chain 1: 장면 분할 | ~120 |
| `backend/app/tools/webtoon/prompt_builder.py` | Chain 2: 프롬프트 빌더 | ~100 |
| `backend/app/tools/webtoon/image_generator.py` | Chain 3: 이미지 생성 | ~150 |
| `backend/app/services/service_function/webtoon_service.py` | 파이프라인 서비스 | ~120 |

### 3.2 Backend 수정 파일 (3개)

| 파일 | 변경 내용 | 영향도 |
|------|----------|--------|
| `backend/app/modules/content_marketing/schema/__init__.py` | Webtoon 스키마 6개 추가 | 중 |
| `backend/app/modules/content_marketing/router/__init__.py` | 4개 엔드포인트 추가 | 중 |
| `backend/app/core/config.py` | STORYBOARD_* 환경변수 5개 추가 | 저 |
| `backend/app/main.py` | StaticFiles 마운트 1줄 추가 | 저 |

### 3.3 Frontend 신규 파일 (5개)

| 파일 | 역할 | 예상 줄 수 |
|------|------|----------|
| `frontend/src/features/content-marketing/hooks/useStoryboardStream.ts` | SSE 스트리밍 훅 | ~160 |
| `frontend/src/features/content-marketing/components/ScriptStoryboardSplitView.tsx` | 분할 뷰 | ~200 |
| `frontend/src/features/content-marketing/components/StoryboardPanel.tsx` | 우측 패널 | ~120 |
| `frontend/src/features/content-marketing/components/StoryboardPanelCard.tsx` | 패널 카드 | ~100 |
| `frontend/src/features/content-marketing/components/StoryboardPanelSkeleton.tsx` | 스켈레톤 | ~40 |

### 3.4 Frontend 수정 파일 (3개)

| 파일 | 변경 내용 | 영향도 |
|------|----------|--------|
| `frontend/src/features/content-marketing/types/index.ts` | Webtoon 타입 7개 추가 | 저 |
| `frontend/src/features/content-marketing/services/index.ts` | API 함수 4개 추가 | 저 |
| `frontend/src/features/content-marketing/components/ScriptGenerator.tsx` | SplitView로 전환 | 고 |

---

## 4. SSE 이벤트 흐름 시퀀스

```
Client                           Backend
  │                                │
  ├─ POST /script/webtoon ────────→│ Job 생성, job_id 반환
  │←── {job_id, status: accepted} ─┤
  │                                │
  ├─ GET  /script/webtoon/{id}/stream →│ SSE 연결
  │                                │
  │←── event: scene_split_start ───┤ Chain 1 시작
  │←── event: scene_split_done ────┤ 패널 목록 확정 (total_panels)
  │                                │
  │←── event: panel_start(1) ──────┤ Chain 2+3: 패널 1 생성 시작
  │←── event: panel_complete(1) ───┤ image_url 포함
  │                                │
  │←── event: panel_start(2) ──────┤
  │←── event: panel_complete(2) ───┤
  │                                │
  │←── event: panel_failed(3) ─────┤ 생성 실패 (재시도 소진)
  │                                │
  │←── event: panel_start(4) ──────┤ 다음 패널 계속
  │←── event: panel_complete(4) ───┤
  │     ...                        │
  │←── event: all_done ────────────┤ 전체 완료
  │                                │
  ├─ GET  /script/webtoon/{id} ───→│ 최종 결과 폴링 (panels 전체)
  │←── {status, panels, progress} ─┤
```

---

## 5. 보안 설계

### 5.1 입력 검증

| 필드 | 검증 |
|------|------|
| `topic` | 2~500자, strip 후 재검증 |
| `sections` | 3개 키 필수, 빈 값 거부 |
| `panel_count` | 4~14 범위 또는 None |
| `job_id` | UUID v4 형식 검증 (라우터 레벨) |
| `panel_number` | 1~14 정수 범위 |

### 5.2 프롬프트 인젝션 방지

```python
# panel_planner.py에 입력 sanitize 함수
import re

INJECTION_PATTERNS = re.compile(
    r"(ignore previous|system prompt|you are|act as|forget|disregard)",
    re.IGNORECASE,
)

def sanitize_input(text: str, max_length: int = 500) -> str:
    """사용자 입력에서 프롬프트 인젝션 패턴 제거"""
    text = text[:max_length]
    text = INJECTION_PATTERNS.sub("", text)
    return text.strip()
```

### 5.3 Path Traversal 방지

```python
def _ensure_dirs(job_id: str) -> Path:
    """UUID 형식 검증 + 경로 안전성 확인"""
    # UUID v4 형식만 허용
    try:
        uuid.UUID(job_id, version=4)
    except ValueError:
        raise ValueError(f"유효하지 않은 job_id: {job_id}")

    job_dir = IMAGES_DIR / job_id
    # resolve 후 기준 디렉토리 하위인지 확인
    resolved = job_dir.resolve()
    if not str(resolved).startswith(str(IMAGES_DIR.resolve())):
        raise ValueError("경로 탈출 시도 감지")

    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir
```

---

## 6. 구현 순서 (16단계)

| # | 단계 | 파일 | 의존성 |
|---|------|------|--------|
| 1 | 환경변수 추가 | `config.py` | 없음 |
| 2 | Pydantic 스키마 추가 | `schema/__init__.py` | 없음 |
| 3 | Chain 1: 장면 분할 | `tools/webtoon/panel_planner.py` | #2 |
| 4 | Chain 2: 프롬프트 빌더 | `tools/webtoon/prompt_builder.py` | #2 |
| 5 | Chain 3: 이미지 생성 | `tools/webtoon/image_generator.py` | #1 |
| 6 | 파이프라인 서비스 | `webtoon_service.py` | #3, #4, #5 |
| 7 | 라우터 엔드포인트 | `router/__init__.py` | #2, #6 |
| 8 | StaticFiles 마운트 | `main.py` | 없음 |
| 9 | Backend 정적 검증 | `ruff check` + `mypy` | #1~#8 |
| 10 | Frontend 타입 추가 | `types/index.ts` | 없음 |
| 11 | Frontend 서비스 추가 | `services/index.ts` | #10 |
| 12 | useStoryboardStream 훅 | `hooks/useStoryboardStream.ts` | #10, #11 |
| 13 | StoryboardPanelCard | `components/StoryboardPanelCard.tsx` | #10 |
| 14 | StoryboardPanel | `components/StoryboardPanel.tsx` | #13 |
| 15 | ScriptStoryboardSplitView | `components/ScriptStoryboardSplitView.tsx` | #12, #14 |
| 16 | ScriptGenerator 수정 | `components/ScriptGenerator.tsx` | #15 |

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-02-28 | v0.1 | 초안 작성 (기획서 v1.0 기반) |
| 2026-02-28 | v1.0 | Gemini CLI 설계 리뷰 반영: 페르소나 동적 연동, 레퍼런스 이미지 폴백, JobManager 재사용 명시 |
| 2026-02-28 | v1.1 | Codex CLI 컨설팅 리뷰 반영: `retrying` 상태 추가, SSE heartbeat/자동 재연결, 프롬프트 캐시 활용 |
