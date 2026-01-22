"""STT (Speech-to-Text) 서비스 - OpenAI Whisper API 사용"""
import tempfile
from pathlib import Path
from typing import BinaryIO

from openai import OpenAI

from app.core.config import settings

# 지원되는 오디오 포맷
SUPPORTED_AUDIO_FORMATS = {"wav", "mp3", "webm", "m4a", "ogg", "flac"}


async def transcribe_audio(
    audio_file: BinaryIO,
    filename: str,
    language: str = "ko",
) -> str:
    """
    음성 파일을 텍스트로 변환

    Args:
        audio_file: 오디오 파일 객체
        filename: 원본 파일명
        language: 언어 코드 (기본값: ko)

    Returns:
        변환된 텍스트
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # 파일 확장자 확인
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"지원하지 않는 오디오 포맷입니다: {extension}")

    # 임시 파일로 저장 후 Whisper API 호출
    with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=True) as temp_file:
        content = audio_file.read()
        temp_file.write(content)
        temp_file.flush()

        with open(temp_file.name, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
                response_format="text",
            )

    return response
