"""Chain 3: Gemini 3 Pro Image 이미지 생성

패널별 이미지 프롬프트로 웹툰 이미지를 생성합니다.
재시도 2회, 실패 시 PIL 플레이스홀더를 생성합니다.
"""

import io
import logging
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import settings
from app.modules.content_marketing.schema import WebtoonPanel

logger = logging.getLogger(__name__)

IMAGES_DIR = (
    Path(__file__).parent.parent.parent.parent / "data" / "media" / "webtoon" / "images"
)
MAX_RETRIES = 2


def _ensure_dirs(job_id: str) -> Path:
    """Job별 이미지 디렉토리 생성 (Path Traversal 방지)"""
    # UUID v4 형식만 허용
    try:
        uuid.UUID(job_id, version=4)
    except ValueError as e:
        msg = f"유효하지 않은 job_id: {job_id}"
        raise ValueError(msg) from e

    job_dir = IMAGES_DIR / job_id
    # resolve 후 기준 디렉토리 하위인지 확인
    resolved = job_dir.resolve()
    if not str(resolved).startswith(str(IMAGES_DIR.resolve())):
        msg = "경로 탈출 시도 감지"
        raise ValueError(msg)

    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _save_as_webp(image_data: bytes, path: Path) -> None:
    """PNG/JPEG → WebP 변환 저장"""
    img = Image.open(io.BytesIO(image_data))
    img.save(path, "WEBP", quality=85)


async def generate_webtoon_image(
    panel: WebtoonPanel,
    reference_image: bytes | None = None,
    job_id: str = "",
) -> dict[str, Any]:
    """패널 이미지 생성 (재시도 2회, 실패 시 플레이스홀더)"""
    job_dir = _ensure_dirs(job_id)

    for attempt in range(MAX_RETRIES + 1):
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=settings.GOOGLE_API_KEY)

            # 프롬프트 + 레퍼런스 이미지 구성
            contents: list[Any] = []
            if reference_image:
                contents.append(
                    types.Part.from_bytes(
                        data=reference_image,
                        mime_type="image/png",
                    )
                )
                contents.append(
                    "Using the character appearance from this reference image, "
                    f"generate the following scene:\n\n{panel.image_prompt}"
                )
            else:
                contents.append(panel.image_prompt or "")

            model_name = settings.STORYBOARD_IMAGE_MODEL

            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )

            # 이미지 추출
            if not response.candidates or not response.candidates[0].content:
                msg = "이미지 생성 결과 없음"
                raise ValueError(msg)

            image_data: bytes | None = None
            safety_flags: list[str] = []
            content = response.candidates[0].content
            if content.parts:
                for part in content.parts:
                    if part.inline_data and part.inline_data.data:
                        image_data = part.inline_data.data
                        break

            if not image_data:
                msg = "생성된 이미지를 찾을 수 없음"
                raise ValueError(msg)

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
                "이미지 생성 시도 %d/%d 실패 (panel=%d): %s",
                attempt + 1,
                MAX_RETRIES + 1,
                panel.panel_number,
                e,
            )
            if attempt == MAX_RETRIES:
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
