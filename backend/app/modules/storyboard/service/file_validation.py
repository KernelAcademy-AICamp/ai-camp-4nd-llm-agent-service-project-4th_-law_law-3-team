"""파일 업로드 보안 검증 게이트 (SEC-01~06)"""

import asyncio
import hashlib
import io
from dataclasses import dataclass, field

from fastapi import HTTPException, UploadFile

from ..schema.models import EvidenceType

# 파일 크기 제한 (바이트)
MAX_FILE_SIZES: dict[str, int] = {
    "audio": 25 * 1024 * 1024,     # 25MB
    "image": 20 * 1024 * 1024,     # 20MB
    "document": 10 * 1024 * 1024,  # 10MB
    "text": 5 * 1024 * 1024,       # 5MB
}
MAX_FILES_PER_BATCH = 10
MAX_FILES_PER_DAY = 50  # SEC-04: 세션별 일일 파일 업로드 제한

# 매직 넘버 → MIME 타입 매핑
MAGIC_NUMBERS: dict[bytes, str] = {
    b'\xff\xd8\xff': "image/jpeg",
    b'\x89PNG': "image/png",
    b'GIF8': "image/gif",
    b'RIFF': "audio/wav",       # RIFF....WAVE
    b'\xff\xfb': "audio/mpeg",  # MP3
    b'\xff\xf3': "audio/mpeg",
    b'ID3\x03': "audio/mpeg",   # MP3 with ID3 tag
    b'%PDF': "application/pdf",
    b'PK\x03\x04': "application/zip",  # DOCX, XLSX 등
}

# M4A는 offset 4에 'ftyp' 시그니처
_M4A_MAGIC_OFFSET = 4
_M4A_MAGIC = b'ftyp'

# MIME → 카테고리 매핑
MIME_TO_CATEGORY: dict[str, str] = {
    "image/jpeg": "image",
    "image/png": "image",
    "image/gif": "image",
    "image/webp": "image",
    "audio/wav": "audio",
    "audio/mpeg": "audio",
    "audio/mp4": "audio",
    "audio/webm": "audio",
    "application/pdf": "document",
    "application/zip": "document",
    "text/plain": "text",
}

# MIME → EvidenceType 매핑
MIME_TO_EVIDENCE_TYPE: dict[str, EvidenceType] = {
    "image/jpeg": EvidenceType.PHOTO,
    "image/png": EvidenceType.MESSENGER_SCREENSHOT,
    "image/gif": EvidenceType.MESSENGER_SCREENSHOT,
    "image/webp": EvidenceType.MESSENGER_SCREENSHOT,
    "audio/wav": EvidenceType.VOICE_RECORDING,
    "audio/mpeg": EvidenceType.VOICE_RECORDING,
    "audio/mp4": EvidenceType.VOICE_RECORDING,
    "audio/webm": EvidenceType.VOICE_RECORDING,
    "application/pdf": EvidenceType.DOCUMENT,
    "application/zip": EvidenceType.DOCUMENT,
    "text/plain": EvidenceType.KAKAO_TXT,
}

# 허용 확장자 → 허용 MIME 목록
ALLOWED_EXTENSIONS: dict[str, list[str]] = {
    ".jpg": ["image/jpeg"],
    ".jpeg": ["image/jpeg"],
    ".png": ["image/png"],
    ".gif": ["image/gif"],
    ".webp": ["image/webp"],
    ".wav": ["audio/wav"],
    ".mp3": ["audio/mpeg"],
    ".m4a": ["audio/mp4"],
    ".webm": ["audio/webm"],
    ".pdf": ["application/pdf"],
    ".docx": ["application/zip"],
    ".xlsx": ["application/zip"],
    ".txt": ["text/plain"],
}


@dataclass
class ValidatedFile:
    """검증 완료된 파일"""
    original_filename: str
    content: bytes
    mime_type: str
    file_size: int
    file_hash: str
    evidence_type: EvidenceType
    is_duplicate: bool = False
    seen_hashes: set[str] = field(default_factory=set, repr=False)


class FileValidationGate:
    """
    파일 업로드 보안 검증 게이트

    SEC-02: 매직 넘버 + 확장자 검증
    SEC-04: 크기/개수 제한 + 일일 할당량
    SEC-05: EXIF 메타데이터 제거
    SEC-06: SHA-256 해시 기반 중복 감지
    """

    def __init__(self) -> None:
        # SEC-04: 세션별 일일 파일 카운터 {session_id: (date, count)}
        self._daily_counters: dict[str, tuple[str, int]] = {}

    def _check_daily_quota(self, session_id: str, file_count: int) -> None:
        """SEC-04: 일일 파일 업로드 할당량 검사."""
        if not session_id:
            return  # 세션 ID 없으면 할당량 검사 스킵 (MVP)

        from datetime import date

        today = date.today().isoformat()
        stored = self._daily_counters.get(session_id)

        if stored is None or stored[0] != today:
            current_count = 0
        else:
            current_count = stored[1]

        if current_count + file_count > MAX_FILES_PER_DAY:
            remaining = max(0, MAX_FILES_PER_DAY - current_count)
            raise HTTPException(
                status_code=429,
                detail=(
                    f"일일 파일 업로드 한도를 초과했습니다 "
                    f"(오늘 {current_count}개 사용, 잔여 {remaining}개, 요청 {file_count}개)"
                ),
            )

        self._daily_counters[session_id] = (today, current_count + file_count)

    async def validate_and_prepare(
        self,
        files: list[UploadFile],
        session_id: str = "",
    ) -> list[ValidatedFile]:
        """
        파일 목록 검증 후 ValidatedFile 목록 반환.

        1. 파일 개수 제한 확인 (MAX_FILES_PER_BATCH)
        2. 각 파일별:
           a. 매직 넘버로 실제 파일 유형 판별
           b. 확장자 ↔ 매직 넘버 일치 검증
           c. 파일 크기 제한 확인
           d. SHA-256 해시 계산 (중복 감지)
           e. 이미지인 경우 EXIF 메타데이터 스트리핑
        3. ValidatedFile 목록 반환 (또는 HTTPException 발생)
        """
        if len(files) > MAX_FILES_PER_BATCH:
            raise HTTPException(
                status_code=400,
                detail=f"파일 개수가 제한을 초과했습니다 (최대 {MAX_FILES_PER_BATCH}개)",
            )

        # SEC-04: 일일 할당량 검사
        self._check_daily_quota(session_id, len(files))

        seen_hashes: set[str] = set()
        seen_metadata: set[str] = set()  # SEC-12: 파일명+크기 기반 근사 중복 감지
        validated: list[ValidatedFile] = []

        for upload_file in files:
            content = await upload_file.read()
            filename = upload_file.filename or "unknown"

            # 매직 넘버 기반 MIME 타입 감지
            mime_type = self._check_magic_number(content, filename)

            # 확장자 일치 검증
            self._validate_extension(filename, mime_type)

            # 카테고리 및 크기 제한 확인
            category = MIME_TO_CATEGORY.get(mime_type, "document")
            max_size = MAX_FILE_SIZES.get(category, MAX_FILE_SIZES["document"])
            if len(content) > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"'{filename}' 파일 크기가 제한을 초과했습니다 (최대 {max_size // (1024 * 1024)}MB)",
                )

            # SHA-256 해시 계산 (중복 감지)
            file_hash = self._compute_hash(content)
            # SEC-12: 해시 일치 또는 파일명+크기 동일 시 중복 플래그
            metadata_key = self._compute_metadata_key(filename, len(content))
            is_duplicate = file_hash in seen_hashes or metadata_key in seen_metadata
            seen_hashes.add(file_hash)
            seen_metadata.add(metadata_key)

            # 이미지 EXIF 제거 (Pillow 동기 I/O → 별도 스레드)
            if category == "image":
                content = await asyncio.to_thread(self._strip_exif, content)

            evidence_type = MIME_TO_EVIDENCE_TYPE.get(mime_type, EvidenceType.OTHER)

            validated.append(
                ValidatedFile(
                    original_filename=filename,
                    content=content,
                    mime_type=mime_type,
                    file_size=len(content),
                    file_hash=file_hash,
                    evidence_type=evidence_type,
                    is_duplicate=is_duplicate,
                )
            )

        return validated

    def _check_magic_number(self, header: bytes, filename: str) -> str:
        """매직 넘버 → MIME 타입 감지. 텍스트 파일은 디코딩으로 판별."""
        # M4A: offset 4에 'ftyp' 확인
        if len(header) >= 8 and header[_M4A_MAGIC_OFFSET:_M4A_MAGIC_OFFSET + 4] == _M4A_MAGIC:
            return "audio/mp4"

        for magic, mime in MAGIC_NUMBERS.items():
            if header[:len(magic)] == magic:
                return mime

        # 텍스트 파일 판별 (UTF-8 디코딩 시도)
        try:
            header[:512].decode("utf-8")
            return "text/plain"
        except (UnicodeDecodeError, ValueError):
            pass

        # 확장자 기반 폴백
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        allowed = ALLOWED_EXTENSIONS.get(ext)
        if allowed:
            return allowed[0]

        raise HTTPException(
            status_code=415,
            detail=f"지원하지 않는 파일 형식입니다: '{filename}'",
        )

    def _validate_extension(self, filename: str, detected_mime: str) -> None:
        """확장자와 감지된 MIME 타입 일치 여부 검증."""
        if "." not in filename:
            return
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        allowed_mimes = ALLOWED_EXTENSIONS.get(ext)
        if allowed_mimes is None:
            raise HTTPException(
                status_code=415,
                detail=f"허용되지 않는 확장자입니다: '{ext}'",
            )
        if detected_mime not in allowed_mimes:
            raise HTTPException(
                status_code=422,
                detail=f"파일 내용이 확장자와 일치하지 않습니다: '{filename}' (감지된 유형: {detected_mime})",
            )

    def _strip_exif(self, image_bytes: bytes) -> bytes:
        """Pillow로 EXIF 메타데이터 제거. 실패 시 HTTPException 발생 (메타데이터 유출 방지)."""
        from PIL import Image, UnidentifiedImageError

        try:
            img = Image.open(io.BytesIO(image_bytes))
            output = io.BytesIO()
            # EXIF 없이 저장 (info 딕셔너리 제거)
            img_format = img.format or "JPEG"
            img.save(output, format=img_format)
            return output.getvalue()
        except UnidentifiedImageError as exc:
            raise HTTPException(
                status_code=422,
                detail="이미지 파일을 처리할 수 없습니다. 지원하지 않는 형식이거나 손상된 파일입니다.",
            ) from exc
        except Exception as exc:
            # 메타데이터 제거 실패 시 원본을 허용하지 않음 (SEC-05)
            raise HTTPException(
                status_code=422,
                detail="이미지 메타데이터 제거에 실패했습니다. 파일을 다시 확인해주세요.",
            ) from exc

    def _compute_hash(self, file_bytes: bytes) -> str:
        """SHA-256 해시 계산."""
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    def _compute_metadata_key(filename: str, file_size: int) -> str:
        """SEC-12: 파일명 + 크기 기반 메타데이터 키 (근사 중복 감지)."""
        return f"{filename.lower().strip()}:{file_size}"
