"""영상 생성 서비스 - moviepy를 사용한 이미지 → 영상 변환"""
import uuid
from pathlib import Path
from typing import Literal, Optional

from moviepy import ImageClip, concatenate_videoclips, vfx
import requests

# 전환 효과 타입
TransitionType = Literal["fade", "slide", "zoom", "none"]

# 미디어 디렉토리 경로
MEDIA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "media"
IMAGES_DIR = MEDIA_DIR / "storyboard" / "images"
VIDEOS_DIR = MEDIA_DIR / "storyboard" / "videos"

# 기본 설정
DEFAULT_DURATION_PER_IMAGE = 6  # 초
DEFAULT_FPS = 24
DEFAULT_RESOLUTION = (1280, 720)
DEFAULT_TRANSITION_DURATION = 1  # 초


def _ensure_dirs() -> None:
    """디렉토리 존재 확인 및 생성"""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def _download_image(url: str, local_path: Path) -> bool:
    """
    이미지 URL에서 로컬 파일로 다운로드

    Args:
        url: 이미지 URL (로컬 경로 또는 원격 URL)
        local_path: 저장할 로컬 경로

    Returns:
        성공 여부
    """
    try:
        if url.startswith("/media/"):
            # 로컬 파일인 경우
            source_path = MEDIA_DIR / url.replace("/media/", "")
            if source_path.exists():
                import shutil
                shutil.copy(source_path, local_path)
                return True
        elif url.startswith("http"):
            # 원격 URL인 경우
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
            return True
        return False
    except Exception:
        return False


def _create_clip_with_transition(
    image_path: Path,
    duration: float,
    transition: TransitionType,
    transition_duration: float,
    resolution: tuple[int, int],
    is_first: bool = False,
    is_last: bool = False,
) -> ImageClip:
    """
    전환 효과가 적용된 이미지 클립 생성

    Args:
        image_path: 이미지 파일 경로
        duration: 표시 시간 (초)
        transition: 전환 효과 타입
        transition_duration: 전환 효과 시간 (초)
        resolution: 해상도 (width, height)
        is_first: 첫 번째 클립 여부
        is_last: 마지막 클립 여부

    Returns:
        ImageClip 객체
    """
    clip = ImageClip(str(image_path), duration=duration)
    clip = clip.resized(resolution)

    if transition == "none":
        return clip

    if transition == "fade":
        if not is_first:
            clip = clip.with_effects([vfx.CrossFadeIn(transition_duration)])
        if not is_last:
            clip = clip.with_effects([vfx.CrossFadeOut(transition_duration)])

    return clip


async def generate_video(
    timeline_id: str,
    image_urls: list[str],
    duration_per_image: float = DEFAULT_DURATION_PER_IMAGE,
    transition: TransitionType = "fade",
    transition_duration: float = DEFAULT_TRANSITION_DURATION,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    fps: int = DEFAULT_FPS,
) -> dict:
    """
    이미지들을 결합하여 영상 생성

    Args:
        timeline_id: 타임라인 ID
        image_urls: 이미지 URL 목록
        duration_per_image: 이미지당 표시 시간 (초)
        transition: 전환 효과
        transition_duration: 전환 효과 시간 (초)
        resolution: 영상 해상도
        fps: 프레임 레이트

    Returns:
        생성 결과 dict (video_url 포함)
    """
    _ensure_dirs()

    if len(image_urls) < 2:
        return {
            "success": False,
            "error": "최소 2개 이상의 이미지가 필요합니다",
            "video_url": None,
        }

    # 임시 이미지 디렉토리
    temp_dir = VIDEOS_DIR / f"temp_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(exist_ok=True)

    try:
        # 이미지 다운로드
        local_images: list[Path] = []
        for idx, url in enumerate(image_urls):
            ext = Path(url).suffix or ".png"
            local_path = temp_dir / f"img_{idx:03d}{ext}"
            if _download_image(url, local_path):
                local_images.append(local_path)

        if len(local_images) < 2:
            return {
                "success": False,
                "error": "이미지 다운로드에 실패했습니다",
                "video_url": None,
            }

        # 클립 생성
        clips: list[ImageClip] = []
        for idx, img_path in enumerate(local_images):
            is_first = idx == 0
            is_last = idx == len(local_images) - 1
            clip = _create_clip_with_transition(
                img_path,
                duration_per_image,
                transition,
                transition_duration,
                resolution,
                is_first,
                is_last,
            )
            clips.append(clip)

        # 영상 결합
        if transition == "fade":
            # 페이드 전환: 클립 오버랩
            final_clip = concatenate_videoclips(
                clips, method="compose", padding=-transition_duration
            )
        else:
            final_clip = concatenate_videoclips(clips, method="compose")

        # 영상 저장
        video_filename = f"{timeline_id}_{uuid.uuid4().hex[:8]}.mp4"
        video_path = VIDEOS_DIR / video_filename

        final_clip.write_videofile(
            str(video_path),
            fps=fps,
            codec="libx264",
            audio=False,
            logger=None,
        )

        # 리소스 정리
        final_clip.close()
        for clip in clips:
            clip.close()

        # 임시 파일 삭제
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        video_url = f"/media/storyboard/videos/{video_filename}"

        return {
            "success": True,
            "video_url": video_url,
            "duration": len(local_images) * duration_per_image,
            "image_count": len(local_images),
        }

    except Exception as e:
        # 임시 파일 삭제
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "success": False,
            "error": str(e),
            "video_url": None,
        }
