"""영상 생성 서비스 - moviepy를 사용한 이미지 → 영상 변환"""
import asyncio
import ipaddress
import logging
import shutil
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import requests
from moviepy import (  # type: ignore[import-untyped]
    ImageClip,
    concatenate_videoclips,
    vfx,
)

logger = logging.getLogger(__name__)

# 전환 효과 타입
TransitionType = Literal["fade", "slide", "zoom", "none"]


def _check_ffmpeg_available() -> bool:
    """ffmpeg가 PATH에 있는지 확인"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# 모듈 로드 시 ffmpeg 확인
if not _check_ffmpeg_available():
    logger.error(
        "ffmpeg가 설치되어 있지 않거나 PATH에 없습니다. "
        "영상 생성 기능을 사용하려면 ffmpeg를 설치해주세요.\n"
        "  - Ubuntu/Debian: sudo apt-get install ffmpeg\n"
        "  - macOS: brew install ffmpeg\n"
        "  - Windows: https://ffmpeg.org/download.html 에서 다운로드 후 PATH에 추가"
    )

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


def _is_private_ip(ip_str: str) -> bool:
    """IP 주소가 사설/루프백/링크로컬인지 확인"""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        )
    except ValueError:
        return True  # 파싱 실패 시 차단


def _validate_url_for_ssrf(url: str) -> bool:
    """URL이 SSRF 공격에 안전한지 검증"""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            return False

        # DNS 확인하여 실제 IP 검증
        addr_infos = socket.getaddrinfo(hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
        for addr_info in addr_infos:
            ip_str = str(addr_info[4][0])
            if _is_private_ip(ip_str):
                return False

        return True
    except (socket.gaierror, socket.herror, ValueError):
        return False


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
            # 로컬 파일인 경우 - 경로 순회 공격 방지
            import shutil
            relative_path = url.replace("/media/", "")
            source_path = (MEDIA_DIR / relative_path).resolve()
            media_dir_resolved = MEDIA_DIR.resolve()

            # source_path가 MEDIA_DIR 내부에 있는지 확인 (Path.relative_to 사용)
            try:
                source_path.relative_to(media_dir_resolved)
            except ValueError:
                # source_path가 media_dir 외부에 있음 (경로 순회 시도)
                logger.warning(f"경로 순회 시도 감지: {url}")
                return False

            if source_path.exists() and source_path.is_file():
                shutil.copy(source_path, local_path)
                return True
        elif url.startswith("http://") or url.startswith("https://"):
            # 원격 URL인 경우 - SSRF 방지
            if not _validate_url_for_ssrf(url):
                return False

            # 스트리밍 다운로드로 메모리 효율화
            with requests.get(url, timeout=30, stream=True) as response:
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        return False
    except requests.RequestException as e:
        logger.error(f"이미지 다운로드 실패 (url={url}): {e}")
        return False
    except OSError as e:
        logger.error(f"파일 작업 실패 (url={url}): {e}")
        return False
    except Exception as e:
        logger.error(f"예기치 않은 오류 (url={url}): {e}", exc_info=True)
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


def _generate_video_sync(
    timeline_id: str,
    image_urls: list[str],
    duration_per_image: float,
    transition: TransitionType,
    transition_duration: float,
    resolution: tuple[int, int],
    fps: int,
) -> dict[str, Any]:
    """
    영상 생성 동기 함수 (스레드에서 실행)

    모든 블로킹 I/O 및 CPU 작업을 수행합니다.
    """
    _ensure_dirs()

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
            shutil.rmtree(temp_dir, ignore_errors=True)
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
            # 페이드 전환 검증: transition_duration이 duration_per_image보다 작아야 함
            if transition_duration >= duration_per_image:
                for clip in clips:
                    clip.close()
                shutil.rmtree(temp_dir, ignore_errors=True)
                return {
                    "success": False,
                    "error": f"전환 시간({transition_duration}초)이 이미지 표시 시간({duration_per_image}초)보다 작아야 합니다",
                    "video_url": None,
                }
            # 페이드 전환: 클립 오버랩
            final_clip = concatenate_videoclips(
                clips, method="compose", padding=-transition_duration
            )
        else:
            final_clip = concatenate_videoclips(clips, method="compose")

        # 실제 영상 길이 저장 (close 전에)
        actual_duration = final_clip.duration

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
        shutil.rmtree(temp_dir, ignore_errors=True)

        video_url = f"/media/storyboard/videos/{video_filename}"

        return {
            "success": True,
            "video_url": video_url,
            "duration": actual_duration,
            "image_count": len(local_images),
        }

    except Exception as e:
        # 임시 파일 삭제
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "success": False,
            "error": str(e),
            "video_url": None,
        }


async def generate_video(
    timeline_id: str,
    image_urls: list[str],
    duration_per_image: float = DEFAULT_DURATION_PER_IMAGE,
    transition: TransitionType = "fade",
    transition_duration: float = DEFAULT_TRANSITION_DURATION,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    fps: int = DEFAULT_FPS,
) -> dict[str, Any]:
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
    if len(image_urls) < 2:
        return {
            "success": False,
            "error": "최소 2개 이상의 이미지가 필요합니다",
            "video_url": None,
        }

    # 블로킹 작업을 스레드에서 실행하여 이벤트 루프 차단 방지
    return await asyncio.to_thread(
        _generate_video_sync,
        timeline_id,
        image_urls,
        duration_per_image,
        transition,
        transition_duration,
        resolution,
        fps,
    )
