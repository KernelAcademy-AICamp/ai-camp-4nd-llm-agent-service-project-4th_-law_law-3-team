"""
LanceDB 통합 테스트

대상:
- E2E 벡터 검색 흐름 (추가 → 검색 → 필터 → 삭제)
- FTS 인덱스 + MeCab 사전 토크나이징 (requires_fts)
- 하이브리드 검색 RRF (requires_fts)

벡터 검색 E2E: 6 tests
FTS E2E: 6 tests (requires_mecab + requires_fts)
하이브리드 검색: 3 tests (requires_mecab + requires_fts)
"""

from typing import Callable

import pytest

from app.tools.vectorstore.base import SearchResult
from app.tools.vectorstore.lancedb import LanceDBStore

# ============================================================================
# Section A: 벡터 검색 E2E (6 tests) — FR-17
# ============================================================================


class TestVectorSearchE2E:
    """벡터 검색 End-to-End 테스트"""

    def test_e2e_add_and_search_law(
        self,
        lancedb_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """법령 추가 → 동일 벡터 검색 → 최상위 결과 확인"""
        vec = make_random_vector(0)
        lancedb_store.add_law_documents(
            source_ids=["E2E_LAW_001"],
            chunk_indices=[0],
            embeddings=[vec],
            titles=["테스트 법령"],
            contents=["[법령] 테스트용 법령 내용"],
            enforcement_dates=["2024-01-01"],
            departments=["법무부"],
        )

        result = lancedb_store.search(query_embedding=vec, n_results=1)
        assert isinstance(result, SearchResult)
        assert len(result.ids[0]) >= 1
        assert result.ids[0][0] == "E2E_LAW_001_0"

    def test_e2e_add_and_search_precedent(
        self,
        lancedb_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """판례 추가 → 동일 벡터 검색 → 판례 결과 확인"""
        vec = make_random_vector(50)
        lancedb_store.add_precedent_documents(
            source_ids=["E2E_PREC_001"],
            chunk_indices=[0],
            embeddings=[vec],
            titles=["테스트 판례"],
            contents=["[판례] 테스트용 판례 내용"],
            decision_dates=["2024-01-15"],
            court_names=["대법원"],
        )

        result = lancedb_store.search(query_embedding=vec, n_results=1)
        assert result.ids[0][0] == "E2E_PREC_001_0"
        assert result.metadatas[0][0]["data_type"] == "판례"

    def test_e2e_mixed_search_filter_by_type(
        self,
        lancedb_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """법령+판례 추가 → search_by_type('법령') → 법령만 반환"""
        # 법령 추가
        lancedb_store.add_law_documents(
            source_ids=["MIXED_LAW"],
            chunk_indices=[0],
            embeddings=[make_random_vector(0)],
            titles=["혼합테스트 법령"],
            contents=["[법령] 혼합 테스트"],
            enforcement_dates=["2024-01-01"],
            departments=["법무부"],
        )
        # 판례 추가
        lancedb_store.add_precedent_documents(
            source_ids=["MIXED_PREC"],
            chunk_indices=[0],
            embeddings=[make_random_vector(1)],
            titles=["혼합테스트 판례"],
            contents=["[판례] 혼합 테스트"],
            decision_dates=["2024-01-15"],
            court_names=["대법원"],
        )

        result = lancedb_store.search_by_type(
            query_embedding=make_random_vector(42),
            data_type="법령",
            n_results=10,
        )
        for meta in result.metadatas[0]:
            assert meta["data_type"] == "법령"

    def test_e2e_source_id_extraction_flow(
        self,
        lancedb_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """검색 → metadatas에서 source_id 추출 → 문자열 확인"""
        vec = make_random_vector(7)
        lancedb_store.add_law_documents(
            source_ids=["SRC_EXTRACT_001"],
            chunk_indices=[0],
            embeddings=[vec],
            titles=["소스ID 추출 테스트"],
            contents=["[법령] source_id 추출 테스트"],
            enforcement_dates=["2024-02-01"],
            departments=["법무부"],
        )

        result = lancedb_store.search(query_embedding=vec, n_results=1)
        source_id = result.metadatas[0][0]["source_id"]
        assert isinstance(source_id, str)
        assert source_id == "SRC_EXTRACT_001"

    def test_e2e_multi_chunk_document(
        self,
        lancedb_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """3청크 문서 추가 → get_by_source_id → 3청크 반환, chunk_index 순서"""
        lancedb_store.add_law_documents(
            source_ids=["MULTI_001", "MULTI_001", "MULTI_001"],
            chunk_indices=[0, 1, 2],
            embeddings=[make_random_vector(i) for i in range(3)],
            titles=["멀티청크"] * 3,
            contents=[f"[법령] 청크 {i}" for i in range(3)],
            enforcement_dates=["2024-03-01"] * 3,
            departments=["법무부"] * 3,
            total_chunks_list=[3, 3, 3],
        )

        result = lancedb_store.get_by_source_id("MULTI_001")
        assert len(result["ids"]) == 3

        # chunk_index 순서 확인
        indices = [m["chunk_index"] for m in result["metadatas"]]
        assert indices == sorted(indices)

    def test_e2e_add_search_delete_lifecycle(
        self,
        lancedb_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """추가 → 검색 확인 → 삭제 → 재검색 → 결과 없음"""
        vec = make_random_vector(99)
        lancedb_store.add_precedent_documents(
            source_ids=["LIFECYCLE_001"],
            chunk_indices=[0],
            embeddings=[vec],
            titles=["라이프사이클 테스트"],
            contents=["[판례] 라이프사이클 테스트 내용"],
            decision_dates=["2024-04-01"],
            court_names=["서울중앙지방법원"],
        )

        # 존재 확인
        result = lancedb_store.get_by_source_id("LIFECYCLE_001")
        assert len(result["ids"]) == 1

        # 삭제
        lancedb_store.delete_by_source_id("LIFECYCLE_001")

        # 삭제 확인
        result = lancedb_store.get_by_source_id("LIFECYCLE_001")
        assert result["ids"] == []


# ============================================================================
# Section B: FTS E2E (6 tests) — FR-23, FR-24, FR-25, FR-28
# ============================================================================


@pytest.mark.requires_mecab
@pytest.mark.requires_fts
class TestFTSIntegration:
    """FTS 인덱스 + MeCab 사전 토크나이징 통합 테스트

    FTS 인프라(content_tokenized 컬럼, FTS 인덱스 생성,
    FTS 검색 메서드)가 구현된 후 활성화됩니다.
    """

    def test_fts_index_creation(
        self,
        lancedb_store: LanceDBStore,
        mecab_tokenizer: object,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """content_tokenized 컬럼으로 FTS 인덱스 생성 성공"""
        pytest.skip("FTS 인프라 미구현 — content_tokenized 컬럼 추가 후 활성화")

    def test_fts_keyword_search(
        self,
        lancedb_store: LanceDBStore,
        mecab_tokenizer: object,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """MeCab 토크나이징 쿼리 → FTS 검색 → 결과 존재"""
        pytest.skip("FTS 인프라 미구현 — FTS 검색 메서드 추가 후 활성화")

    def test_fts_search_accuracy(
        self,
        lancedb_store: LanceDBStore,
        mecab_tokenizer: object,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """'손해배상' 검색 → 손해배상 관련 문서가 상위"""
        pytest.skip("FTS 인프라 미구현 — FTS 검색 메서드 추가 후 활성화")

    def test_fts_match_query(
        self,
        lancedb_store: LanceDBStore,
        mecab_tokenizer: object,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """단일 키워드 '손해' → 매칭 결과 확인"""
        pytest.skip("FTS 인프라 미구현 — FTS 검색 메서드 추가 후 활성화")

    def test_fts_phrase_query(
        self,
        lancedb_store: LanceDBStore,
        mecab_tokenizer: object,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """구문 '손해 배상' → 해당 토큰 연속 포함 문서"""
        pytest.skip("FTS 인프라 미구현 — FTS 검색 메서드 추가 후 활성화")

    def test_fts_boolean_query(
        self,
        lancedb_store: LanceDBStore,
        mecab_tokenizer: object,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """'손해' AND '배상' → 두 토큰 모두 포함 문서"""
        pytest.skip("FTS 인프라 미구현 — FTS 검색 메서드 추가 후 활성화")


# ============================================================================
# Section C: 하이브리드 검색 E2E (3 tests) — FR-26
# ============================================================================


@pytest.mark.requires_mecab
@pytest.mark.requires_fts
class TestHybridSearch:
    """벡터 + FTS 하이브리드 검색 (RRF) 통합 테스트

    하이브리드 검색(벡터 + FTS 결합)이 구현된 후 활성화됩니다.
    RRF (Reciprocal Rank Fusion) 알고리즘으로 결과를 결합합니다.
    """

    def test_hybrid_search_rrf(
        self,
        lancedb_store: LanceDBStore,
        mecab_tokenizer: object,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """벡터 Top-K + FTS Top-K → RRF 결합 → 최종 순위"""
        pytest.skip("하이브리드 검색 미구현 — RRF 메서드 추가 후 활성화")

    def test_hybrid_search_vector_only_fallback(
        self,
        lancedb_store: LanceDBStore,
        mecab_tokenizer: object,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """FTS 결과 없는 쿼리 → 벡터 결과만 반환"""
        pytest.skip("하이브리드 검색 미구현 — RRF 메서드 추가 후 활성화")

    def test_hybrid_search_fts_boost(
        self,
        lancedb_store: LanceDBStore,
        mecab_tokenizer: object,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """정확한 키워드 쿼리 → FTS 결과가 순위에 기여"""
        pytest.skip("하이브리드 검색 미구현 — RRF 메서드 추가 후 활성화")
