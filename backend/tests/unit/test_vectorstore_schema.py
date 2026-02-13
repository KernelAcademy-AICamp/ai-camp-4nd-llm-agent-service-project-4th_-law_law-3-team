"""
schema_v2.py 단위 테스트

대상:
- create_chunk(): 범용 청크 생성 헬퍼
- create_law_chunk(): 법령 청크 생성 (하위 호환 래퍼)
- create_precedent_chunk(): 판례 청크 생성 (하위 호환 래퍼)
- LegalChunk: Pydantic 검증 모델
- 스키마 상수: LEGAL_CHUNKS_SCHEMA, COLUMN_GROUPS
"""

import numpy as np
import pytest

from app.tools.vectorstore.schema_v2 import (
    ALL_COLUMNS,
    COMMON_COLUMNS,
    LEGAL_CHUNKS_SCHEMA,
    VECTOR_DIM,
    LegalChunk,
    create_chunk,
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
# create_chunk 범용 테스트
# ============================================================================


class TestCreateChunk:
    """범용 청크 생성 헬퍼 테스트"""

    def test_create_chunk_all_fields(self, _sample_vector: list[float]) -> None:
        """모든 필드를 지정하여 청크 생성"""
        result = create_chunk(
            data_type="헌법재판",
            source_id="177507",
            title="112신고 결과 메세지 미전송 위헌확인",
            content="공권력 불행사에 대한 헌법소원 요약",
            vector=_sample_vector,
            source_name="헌법재판소",
            date="20220920",
            chunk_index=0,
            total_chunks=1,
        )

        assert result["id"] == "177507_0"
        assert result["source_id"] == "177507"
        assert result["data_type"] == "헌법재판"
        assert result["title"] == "112신고 결과 메세지 미전송 위헌확인"
        assert result["source_name"] == "헌법재판소"
        assert result["date"] == "20220920"
        assert result["chunk_index"] == 0
        assert result["total_chunks"] == 1
        assert len(result["vector"]) == VECTOR_DIM

    def test_create_chunk_minimal(self, _sample_vector: list[float]) -> None:
        """최소 필수 필드만으로 청크 생성"""
        result = create_chunk(
            data_type="행정심판",
            source_id="12345",
            title="제목",
            content="내용",
            vector=_sample_vector,
        )

        assert result["data_type"] == "행정심판"
        assert result["source_name"] == ""
        assert result["date"] is None
        assert result["chunk_index"] == 0
        assert result["total_chunks"] == 1

    def test_create_chunk_has_exactly_10_keys(
        self, _sample_vector: list[float]
    ) -> None:
        """청크 dict가 정확히 10개 키를 가지는지 확인"""
        result = create_chunk(
            data_type="판례",
            source_id="1",
            title="t",
            content="c",
            vector=_sample_vector,
        )
        assert len(result) == 10


# ============================================================================
# create_law_chunk 하위 호환 래퍼 테스트
# ============================================================================


class TestCreateLawChunk:
    """법령 청크 생성 래퍼 테스트"""

    def test_create_law_chunk_required_fields(
        self, _sample_vector: list[float]
    ) -> None:
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

    def test_create_law_chunk_ignores_extra_kwargs(
        self, _sample_vector: list[float]
    ) -> None:
        """제거된 법령 전용 필드가 kwargs로 전달되어도 에러 없이 무시"""
        result = create_law_chunk(
            source_id="010719",
            chunk_index=0,
            title="민법",
            content="내용",
            vector=_sample_vector,
            enforcement_date="2023-08-08",
            department="법무부",
            law_type="법률",
            article_no="제750조",
            promulgation_date="20230808",
            promulgation_no="19592",
        )

        assert result["data_type"] == "법령"
        assert len(result) == 10


# ============================================================================
# create_precedent_chunk 하위 호환 래퍼 테스트
# ============================================================================


class TestCreatePrecedentChunk:
    """판례 청크 생성 래퍼 테스트"""

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

    def test_create_precedent_chunk_ignores_extra_kwargs(
        self, _sample_vector: list[float]
    ) -> None:
        """제거된 판례 전용 필드가 kwargs로 전달되어도 에러 없이 무시"""
        result = create_precedent_chunk(
            source_id="76396",
            chunk_index=0,
            title="손해배상(기)",
            content="내용",
            vector=_sample_vector,
            decision_date="2023-05-15",
            court_name="대법원",
            case_number="2023다12345",
            case_type="민사",
            judgment_type="판결",
            reference_provisions="민법 제750조",
        )

        assert result["data_type"] == "판례"
        assert len(result) == 10


# ============================================================================
# LegalChunk Pydantic 모델 테스트
# ============================================================================


class TestLegalChunkValidation:
    """LegalChunk Pydantic 모델 유효성 검증 테스트"""

    def test_legal_chunk_valid_law(self, _sample_vector: list[float]) -> None:
        """유효한 법령 LegalChunk 생성"""
        chunk = LegalChunk(
            id="010719_0",
            source_id="010719",
            data_type="법령",
            title="민법",
            content="민법 제750조",
            vector=_sample_vector,
            source_name="법무부",
            date="2023-08-08",
        )
        assert chunk.data_type == "법령"
        assert chunk.date == "2023-08-08"

    def test_legal_chunk_valid_precedent(
        self, _sample_vector: list[float]
    ) -> None:
        """유효한 판례 LegalChunk 생성"""
        chunk = LegalChunk(
            id="76396_0",
            source_id="76396",
            data_type="판례",
            title="손해배상(기)",
            content="판결 내용",
            vector=_sample_vector,
            source_name="대법원",
            date="2023-05-15",
        )
        assert chunk.data_type == "판례"

    def test_legal_chunk_no_date(self, _sample_vector: list[float]) -> None:
        """date 없는 타입 (예: 헌법재판) LegalChunk 생성"""
        chunk = LegalChunk(
            id="177507_0",
            source_id="177507",
            data_type="헌법재판",
            title="위헌확인",
            content="내용",
            vector=_sample_vector,
        )
        assert chunk.date is None
        assert chunk.source_name == ""

    def test_legal_chunk_to_dict(self, _sample_vector: list[float]) -> None:
        """to_dict()가 모든 필드를 포함하는지 확인"""
        chunk = LegalChunk(
            id="1_0",
            source_id="1",
            data_type="판례",
            title="t",
            content="c",
            vector=_sample_vector,
        )
        d = chunk.to_dict()
        assert set(d.keys()) == {
            "id", "source_id", "data_type", "title", "content",
            "vector", "source_name", "chunk_index", "total_chunks", "date",
        }


# ============================================================================
# ID 형식 테스트
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

    def test_id_format_generic(self, _sample_vector: list[float]) -> None:
        """범용 create_chunk ID 형식 확인"""
        result = create_chunk(
            data_type="헌법재판",
            source_id="177507",
            title="t",
            content="c",
            vector=_sample_vector,
            chunk_index=2,
        )
        assert result["id"] == "177507_2"


# ============================================================================
# 스키마 상수 테스트
# ============================================================================


class TestSchemaConstants:
    """스키마 상수 및 컬럼 그룹 검증"""

    def test_schema_column_count(self) -> None:
        """LEGAL_CHUNKS_SCHEMA가 정확히 10개 컬럼인지 확인"""
        assert len(LEGAL_CHUNKS_SCHEMA) == 10

    def test_vector_dimension(self) -> None:
        """VECTOR_DIM이 1024인지 확인"""
        assert VECTOR_DIM == 1024

    def test_column_groups(self) -> None:
        """컬럼 그룹 개수가 올바른지 확인"""
        assert len(COMMON_COLUMNS) == 9
        assert len(ALL_COLUMNS) == 10
        assert ALL_COLUMNS == COMMON_COLUMNS + ["date"]
