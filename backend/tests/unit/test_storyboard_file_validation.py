"""파일 검증 게이트 단위 테스트"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import hashlib

import pytest

from app.modules.storyboard.service.file_validation import (
    ALLOWED_EXTENSIONS,
    MAGIC_NUMBERS,
    MIME_TO_CATEGORY,
    MIME_TO_EVIDENCE_TYPE,
    FileValidationGate,
)


class TestComputeMetadataKey:
    """SEC-12: _compute_metadata_key 단위 테스트"""

    def test_basic(self) -> None:
        key = FileValidationGate._compute_metadata_key("test.txt", 1024)
        assert key == "test.txt:1024"

    def test_case_insensitive(self) -> None:
        key1 = FileValidationGate._compute_metadata_key("Test.TXT", 100)
        key2 = FileValidationGate._compute_metadata_key("test.txt", 100)
        assert key1 == key2

    def test_strip_whitespace(self) -> None:
        key1 = FileValidationGate._compute_metadata_key("  test.txt  ", 100)
        key2 = FileValidationGate._compute_metadata_key("test.txt", 100)
        assert key1 == key2

    def test_different_size_different_key(self) -> None:
        key1 = FileValidationGate._compute_metadata_key("test.txt", 100)
        key2 = FileValidationGate._compute_metadata_key("test.txt", 200)
        assert key1 != key2

    def test_different_name_different_key(self) -> None:
        key1 = FileValidationGate._compute_metadata_key("a.txt", 100)
        key2 = FileValidationGate._compute_metadata_key("b.txt", 100)
        assert key1 != key2


class TestComputeHash:
    """SHA-256 해시 계산 테스트"""

    def test_deterministic(self) -> None:
        gate = FileValidationGate()
        content = b"hello world"
        hash1 = gate._compute_hash(content)
        hash2 = gate._compute_hash(content)
        assert hash1 == hash2

    def test_matches_hashlib(self) -> None:
        gate = FileValidationGate()
        content = b"test content for hashing"
        expected = hashlib.sha256(content).hexdigest()
        assert gate._compute_hash(content) == expected

    def test_different_content_different_hash(self) -> None:
        gate = FileValidationGate()
        assert gate._compute_hash(b"aaa") != gate._compute_hash(b"bbb")


class TestCheckMagicNumber:
    """매직 넘버 감지 테스트"""

    def test_jpeg(self) -> None:
        gate = FileValidationGate()
        header = b"\xff\xd8\xff" + b"\x00" * 100
        assert gate._check_magic_number(header, "photo.jpg") == "image/jpeg"

    def test_png(self) -> None:
        gate = FileValidationGate()
        header = b"\x89PNG" + b"\x00" * 100
        assert gate._check_magic_number(header, "img.png") == "image/png"

    def test_pdf(self) -> None:
        gate = FileValidationGate()
        header = b"%PDF-1.4" + b"\x00" * 100
        assert gate._check_magic_number(header, "doc.pdf") == "application/pdf"

    def test_text_utf8(self) -> None:
        gate = FileValidationGate()
        header = "카카오톡 대화 내용입니다".encode("utf-8")
        assert gate._check_magic_number(header, "chat.txt") == "text/plain"

    def test_unsupported_format_raises(self) -> None:
        gate = FileValidationGate()
        header = bytes(range(256))  # 임의 바이너리
        with pytest.raises(Exception):
            gate._check_magic_number(header, "unknown.xyz")


class TestValidateExtension:
    """확장자 검증 테스트"""

    def test_matching_extension(self) -> None:
        gate = FileValidationGate()
        # 일치하면 에러 없음
        gate._validate_extension("photo.jpg", "image/jpeg")

    def test_mismatched_extension(self) -> None:
        gate = FileValidationGate()
        with pytest.raises(Exception):
            gate._validate_extension("photo.jpg", "image/png")

    def test_unknown_extension(self) -> None:
        gate = FileValidationGate()
        with pytest.raises(Exception):
            gate._validate_extension("file.exe", "application/octet-stream")

    def test_no_extension(self) -> None:
        gate = FileValidationGate()
        # 확장자 없으면 검증 스킵
        gate._validate_extension("noext", "image/jpeg")


class TestDailyQuota:
    """SEC-04: 일일 업로드 할당량 테스트"""

    def test_within_quota(self) -> None:
        gate = FileValidationGate()
        # 50개 이하면 에러 없음
        gate._check_daily_quota("session-1", 10)

    def test_exceed_quota(self) -> None:
        gate = FileValidationGate()
        # 먼저 40개 사용
        gate._check_daily_quota("session-1", 40)
        # 추가 20개 → 초과
        with pytest.raises(Exception):
            gate._check_daily_quota("session-1", 20)

    def test_no_session_id_skips(self) -> None:
        gate = FileValidationGate()
        # 세션 ID 없으면 할당량 검사 스킵
        gate._check_daily_quota("", 100)


class TestConstants:
    """상수 정합성 테스트"""

    def test_allowed_extensions_not_empty(self) -> None:
        assert len(ALLOWED_EXTENSIONS) > 0

    def test_all_categories_have_size_limit(self) -> None:
        from app.modules.storyboard.service.file_validation import MAX_FILE_SIZES

        for mime, category in MIME_TO_CATEGORY.items():
            assert category in MAX_FILE_SIZES, f"{mime} → {category} 크기 제한 없음"

    def test_all_mimes_have_evidence_type(self) -> None:
        for mime in MIME_TO_CATEGORY:
            assert mime in MIME_TO_EVIDENCE_TYPE, f"{mime} evidence_type 매핑 없음"

    def test_magic_numbers_not_empty(self) -> None:
        assert len(MAGIC_NUMBERS) > 0
