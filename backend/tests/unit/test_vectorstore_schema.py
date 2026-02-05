"""
schema_v2.py 단위 테스트

대상:
- create_law_chunk(): 법령 청크 생성 헬퍼
- create_precedent_chunk(): 판례 청크 생성 헬퍼
- LegalChunk: Pydantic 검증 모델
- 스키마 상수: LEGAL_CHUNKS_SCHEMA, COLUMN_GROUPS
"""

import numpy as np
import pytest

from app.tools.vectorstore.schema_v2 import (
    ALL_COLUMNS,
    COMMON_COLUMNS,
    LAW_COLUMNS,
    LEGAL_CHUNKS_SCHEMA,
    PRECEDENT_COLUMNS,
    VECTOR_DIM,
    LegalChunk,
    create_law_chunk,
    create_precedent_chunk,
)


@pytest.fixture
def _sample_vector() -> list[float]:
    """테스트용 1024차원 벡터"""
    rng = np.random.RandomState(42)
    vec = rng.randn(VECTOR_DIM).astype(np.float32)
    return vec.tolist()


# ============================================================================
# create_law_chunk 테스트 (FR-01, FR-04)
# ============================================================================


class TestCreateLawChunk:
    """법령 청크 생성 헬퍼 테스트"""

    def test_create_law_chunk_required_fields(self, _sample_vector: list[float]) -> None:
        """법령 청크 생성 시 공통 필드가 올바르게 매핑되는지 확인"""
        result = create_law_chunk(
            source_id="010719",
            chunk_index=0,
            title="민법",
            content="[법령] 민법 제750조: 불법행위 내용",
            vector=_sample_vector,
            enforcement_date="2023-08-08",
            department="법무부",
        )

        assert result["id"] == "010719_0"
        assert result["source_id"] == "010719"
        assert result["data_type"] == "법령"
        assert result["title"] == "민법"
        assert result["content"] == "[법령] 민법 제750조: 불법행위 내용"
        assert result["date"] == "2023-08-08"
        assert result["source_name"] == "법무부"
        assert result["chunk_index"] == 0
        assert result["total_chunks"] == 1
        assert len(result["vector"]) == VECTOR_DIM

    def test_create_law_chunk_law_specific_fields(
        self, _sample_vector: list[float]
    ) -> None:
        """법령 전용 필드(law_type, article_no 등)가 매핑되는지 확인"""
        result = create_law_chunk(
            source_id="010719",
            chunk_index=0,
            title="민법",
            content="[법령] 민법 제750조",
            vector=_sample_vector,
            enforcement_date="2023-08-08",
            department="법무부",
            law_type="법률",
            article_no="제750조",
            promulgation_date="20230808",
            promulgation_no="19592",
        )

        assert result["law_type"] == "법률"
        assert result["article_no"] == "제750조"
        assert result["promulgation_date"] == "20230808"
        assert result["promulgation_no"] == "19592"

    def test_create_law_chunk_precedent_fields_null(
        self, _sample_vector: list[float]
    ) -> None:
        """법령 청크에서 판례 전용 필드가 모두 None인지 확인"""
        result = create_law_chunk(
            source_id="010719",
            chunk_index=0,
            title="민법",
            content="내용",
            vector=_sample_vector,
            enforcement_date="2023-08-08",
            department="법무부",
        )

        for col in PRECEDENT_COLUMNS:
            assert result[col] is None, f"법령 청크의 {col} 필드가 None이 아닙니다"


# ============================================================================
# create_precedent_chunk 테스트 (FR-02, FR-04)
# ============================================================================


class TestCreatePrecedentChunk:
    """판례 청크 생성 헬퍼 테스트"""

    def test_create_precedent_chunk_required_fields(
        self, _sample_vector: list[float]
    ) -> None:
        """판례 청크 생성 시 공통 필드가 올바르게 매핑되는지 확인"""
        result = create_precedent_chunk(
            source_id="76396",
            chunk_index=0,
            title="손해배상(기)",
            content="[판례] 불법행위로 인한 손해배상",
            vector=_sample_vector,
            decision_date="2023-05-15",
            court_name="대법원",
        )

        assert result["id"] == "76396_0"
        assert result["source_id"] == "76396"
        assert result["data_type"] == "판례"
        assert result["title"] == "손해배상(기)"
        assert result["date"] == "2023-05-15"
        assert result["source_name"] == "대법원"

    def test_create_precedent_chunk_precedent_specific_fields(
        self, _sample_vector: list[float]
    ) -> None:
        """판례 전용 필드(case_number, case_type 등)가 매핑되는지 확인"""
        result = create_precedent_chunk(
            source_id="76396",
            chunk_index=0,
            title="손해배상(기)",
            content="[판례] 내용",
            vector=_sample_vector,
            decision_date="2023-05-15",
            court_name="대법원",
            case_number="2023다12345",
            case_type="민사",
            judgment_type="판결",
            judgment_status="확정",
            reference_provisions="민법 제750조",
            reference_cases="2022다99999",
        )

        assert result["case_number"] == "2023다12345"
        assert result["case_type"] == "민사"
        assert result["judgment_type"] == "판결"
        assert result["judgment_status"] == "확정"
        assert result["reference_provisions"] == "민법 제750조"
        assert result["reference_cases"] == "2022다99999"

    def test_create_precedent_chunk_law_fields_null(
        self, _sample_vector: list[float]
    ) -> None:
        """판례 청크에서 법령 전용 필드가 모두 None인지 확인"""
        result = create_precedent_chunk(
            source_id="76396",
            chunk_index=0,
            title="손해배상(기)",
            content="내용",
            vector=_sample_vector,
            decision_date="2023-05-15",
            court_name="대법원",
        )

        for col in LAW_COLUMNS:
            assert result[col] is None, f"판례 청크의 {col} 필드가 None이 아닙니다"


# ============================================================================
# LegalChunk Pydantic 모델 테스트 (FR-03, FR-04)
# ============================================================================


class TestLegalChunkValidation:
    """LegalChunk Pydantic 모델 유효성 검증 테스트"""

    def test_legal_chunk_validation_valid_law(
        self, _sample_vector: list[float]
    ) -> None:
        """유효한 법령 LegalChunk가 검증을 통과하는지 확인"""
        chunk = LegalChunk(
            id="010719_0",
            source_id="010719",
            data_type="법령",
            title="민법",
            content="민법 제750조",
            vector=_sample_vector,
            date="2023-08-08",
            source_name="법무부",
            law_type="법률",
        )
        assert chunk.validate_by_type() is True

    def test_legal_chunk_validation_valid_precedent(
        self, _sample_vector: list[float]
    ) -> None:
        """유효한 판례 LegalChunk가 검증을 통과하는지 확인"""
        chunk = LegalChunk(
            id="76396_0",
            source_id="76396",
            data_type="판례",
            title="손해배상(기)",
            content="판결 내용",
            vector=_sample_vector,
            date="2023-05-15",
            source_name="대법원",
            case_number="2023다12345",
            case_type="민사",
        )
        assert chunk.validate_by_type() is True

    def test_legal_chunk_validation_law_with_precedent_fields(
        self, _sample_vector: list[float]
    ) -> None:
        """법령 데이터에 판례 필드가 설정되면 ValueError 발생"""
        chunk = LegalChunk(
            id="010719_0",
            source_id="010719",
            data_type="법령",
            title="민법",
            content="내용",
            vector=_sample_vector,
            date="2023-08-08",
            source_name="법무부",
            case_number="2023다12345",  # 판례 필드 설정
        )
        with pytest.raises(ValueError, match="판례 필드"):
            chunk.validate_by_type()

    def test_legal_chunk_validation_precedent_with_law_fields(
        self, _sample_vector: list[float]
    ) -> None:
        """판례 데이터에 법령 필드가 설정되면 ValueError 발생"""
        chunk = LegalChunk(
            id="76396_0",
            source_id="76396",
            data_type="판례",
            title="손해배상",
            content="내용",
            vector=_sample_vector,
            date="2023-05-15",
            source_name="대법원",
            law_type="법률",  # 법령 필드 설정
        )
        with pytest.raises(ValueError, match="법령 필드"):
            chunk.validate_by_type()


# ============================================================================
# ID 형식 테스트 (FR-18)
# ============================================================================


class TestIdFormat:
    """청크 ID 형식 규칙 테스트"""

    def test_id_format_law(self, _sample_vector: list[float]) -> None:
        """법령 ID 형식이 {source_id}_{chunk_index}인지 확인"""
        result = create_law_chunk(
            source_id="010719",
            chunk_index=3,
            title="민법",
            content="내용",
            vector=_sample_vector,
            enforcement_date="2023-08-08",
            department="법무부",
        )
        assert result["id"] == "010719_3"

    def test_id_format_precedent(self, _sample_vector: list[float]) -> None:
        """판례 ID 형식이 {source_id}_{chunk_index}인지 확인"""
        result = create_precedent_chunk(
            source_id="76396",
            chunk_index=5,
            title="손해배상",
            content="내용",
            vector=_sample_vector,
            decision_date="2023-05-15",
            court_name="대법원",
        )
        assert result["id"] == "76396_5"


# ============================================================================
# 스키마 상수 테스트
# ============================================================================


class TestSchemaConstants:
    """스키마 상수 및 컬럼 그룹 검증"""

    def test_schema_column_count(self) -> None:
        """LEGAL_CHUNKS_SCHEMA가 정확히 20개 컬럼인지 확인"""
        assert len(LEGAL_CHUNKS_SCHEMA) == 20

    def test_vector_dimension(self) -> None:
        """VECTOR_DIM이 1024인지 확인"""
        assert VECTOR_DIM == 1024

    def test_column_groups(self) -> None:
        """컬럼 그룹 개수가 올바른지 확인"""
        assert len(COMMON_COLUMNS) == 10
        assert len(LAW_COLUMNS) == 4
        assert len(PRECEDENT_COLUMNS) == 6
        assert len(ALL_COLUMNS) == 20
        assert ALL_COLUMNS == COMMON_COLUMNS + LAW_COLUMNS + PRECEDENT_COLUMNS
