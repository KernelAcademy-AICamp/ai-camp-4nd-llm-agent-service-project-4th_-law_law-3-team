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
from app.tools.vectorstore.mecab_tokenizer import MeCabTokenizer

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

# FTS 테스트 전용 픽스처: content_tokenized가 포함된 데이터
@pytest.fixture
def fts_store(
    lancedb_store: LanceDBStore,
    make_random_vector: Callable[[int], list[float]],
    mecab_tokenizer: MeCabTokenizer,
) -> LanceDBStore:
    """
    MeCab 토크나이징된 content_tokenized가 포함된 LanceDBStore

    법령 3건 + 판례 3건, 각 1청크 = 총 6레코드
    FTS 인덱스 생성 완료 상태
    """
    # 법령 데이터
    law_contents = [
        "[법령] 민법 제750조: 고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다.",
        "[법령] 상법 제1조: 상사에 관하여 본법에 규정이 없으면 상관습법에 의한다.",
        "[법령] 도로교통법 제1조: 도로에서 일어나는 교통상의 모든 위험과 장해를 방지한다.",
    ]
    law_tokenized = [mecab_tokenizer.tokenize(c) for c in law_contents]

    lancedb_store.add_law_documents(
        source_ids=["FTS_LAW_001", "FTS_LAW_002", "FTS_LAW_003"],
        chunk_indices=[0, 0, 0],
        embeddings=[make_random_vector(i) for i in range(3)],
        titles=["민법", "상법", "도로교통법"],
        contents=law_contents,
        enforcement_dates=["2023-08-08", "2023-01-01", "2024-01-01"],
        departments=["법무부", "법무부", "경찰청"],
        content_tokenized_list=law_tokenized,
    )

    # 판례 데이터
    prec_contents = [
        "[판례] 불법행위로 인한 손해배상의 범위에 관한 판결. 민법 제750조에 의한 손해배상 책임.",
        "[판례] 교통사고처리특례법위반 사건 - 도로교통법 위반으로 인한 교통사고.",
        "[판례] 상법상 상행위에 해당하는 매매계약의 해제와 원상회복 의무.",
    ]
    prec_tokenized = [mecab_tokenizer.tokenize(c) for c in prec_contents]

    lancedb_store.add_precedent_documents(
        source_ids=["FTS_PREC_001", "FTS_PREC_002", "FTS_PREC_003"],
        chunk_indices=[0, 0, 0],
        embeddings=[make_random_vector(i + 100) for i in range(3)],
        titles=["손해배상(기)", "교통사고처리특례법위반", "매매대금반환"],
        contents=prec_contents,
        decision_dates=["2023-05-15", "2023-07-20", "2024-01-10"],
        court_names=["대법원", "서울중앙지방법원", "서울고등법원"],
        content_tokenized_list=prec_tokenized,
    )

    # FTS 인덱스 생성
    lancedb_store.create_fts_index("content_tokenized")

    return lancedb_store


@pytest.mark.requires_mecab
@pytest.mark.requires_fts
class TestFTSIntegration:
    """FTS 인덱스 + MeCab 사전 토크나이징 통합 테스트"""

    def test_fts_index_creation(
        self,
        fts_store: LanceDBStore,
    ) -> None:
        """content_tokenized 컬럼으로 FTS 인덱스 생성 성공"""
        # fts_store 픽스처에서 이미 인덱스 생성됨
        # 재생성해도 에러 없음 (replace=True)
        fts_store.create_fts_index("content_tokenized")
        assert fts_store.count() == 6

    def test_fts_keyword_search(
        self,
        fts_store: LanceDBStore,
    ) -> None:
        """MeCab 토크나이징 쿼리 → FTS 검색 → 결과 존재"""
        result = fts_store.search_fts("손해배상", n_results=5)
        assert isinstance(result, SearchResult)
        assert len(result.ids[0]) > 0

    def test_fts_search_accuracy(
        self,
        fts_store: LanceDBStore,
    ) -> None:
        """'손해배상' 검색 → 손해배상 관련 문서가 상위"""
        result = fts_store.search_fts("손해배상", n_results=5)
        assert len(result.ids[0]) > 0

        # 손해배상 관련 문서(FTS_LAW_001 또는 FTS_PREC_001)가 포함되어야 함
        all_ids = result.ids[0]
        damage_related = {"FTS_LAW_001_0", "FTS_PREC_001_0"}
        found = damage_related.intersection(set(all_ids))
        assert len(found) > 0, f"손해배상 관련 문서가 없음: {all_ids}"

    def test_fts_match_query(
        self,
        fts_store: LanceDBStore,
    ) -> None:
        """단일 키워드 '도로' → 도로교통법 관련 문서 매칭"""
        result = fts_store.search_fts("도로교통법", n_results=5)
        assert len(result.ids[0]) > 0

        all_ids = result.ids[0]
        traffic_related = {"FTS_LAW_003_0", "FTS_PREC_002_0"}
        found = traffic_related.intersection(set(all_ids))
        assert len(found) > 0, f"도로교통법 관련 문서가 없음: {all_ids}"

    def test_fts_phrase_query(
        self,
        fts_store: LanceDBStore,
    ) -> None:
        """'불법행위 손해배상' → 관련 문서 검색"""
        result = fts_store.search_fts("불법행위 손해배상", n_results=5)
        assert len(result.ids[0]) > 0

        # 불법행위+손해배상 모두 포함하는 문서가 상위에 있어야 함
        all_ids = result.ids[0]
        expected = {"FTS_LAW_001_0", "FTS_PREC_001_0"}
        found = expected.intersection(set(all_ids))
        assert len(found) > 0, f"불법행위/손해배상 관련 문서가 없음: {all_ids}"

    def test_fts_boolean_query(
        self,
        fts_store: LanceDBStore,
    ) -> None:
        """'상법 매매' → 상법+매매 관련 문서 검색"""
        result = fts_store.search_fts("상법 매매", n_results=5)
        assert len(result.ids[0]) > 0

        all_ids = result.ids[0]
        commercial_related = {"FTS_LAW_002_0", "FTS_PREC_003_0"}
        found = commercial_related.intersection(set(all_ids))
        assert len(found) > 0, f"상법/매매 관련 문서가 없음: {all_ids}"


# ============================================================================
# Section C: 하이브리드 검색 E2E (3 tests) — FR-26
# ============================================================================


@pytest.mark.requires_mecab
@pytest.mark.requires_fts
class TestHybridSearch:
    """벡터 + FTS 하이브리드 검색 (RRF) 통합 테스트"""

    def test_hybrid_search_rrf(
        self,
        fts_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """벡터 Top-K + FTS Top-K → RRF 결합 → 최종 순위"""
        result = fts_store.hybrid_search(
            query_embedding=make_random_vector(0),
            query_text="손해배상",
            n_results=5,
        )
        assert isinstance(result, SearchResult)
        assert len(result.ids[0]) > 0
        assert len(result.distances[0]) == len(result.ids[0])

    def test_hybrid_search_vector_only_fallback(
        self,
        fts_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """FTS 결과 없는 쿼리 → 벡터 결과만 반환"""
        # 매우 특수한 검색어로 FTS 결과가 없도록 함
        result = fts_store.hybrid_search(
            query_embedding=make_random_vector(42),
            query_text="xyznonexistent123",
            n_results=5,
        )
        assert isinstance(result, SearchResult)
        # 벡터 결과는 존재해야 함
        assert len(result.ids[0]) > 0

    def test_hybrid_search_fts_boost(
        self,
        fts_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """정확한 키워드 쿼리 → FTS 결과가 순위에 기여"""
        # "손해배상"과 관련 없는 벡터(seed=50)로 검색하되 텍스트로는 "손해배상" 검색
        result = fts_store.hybrid_search(
            query_embedding=make_random_vector(50),
            query_text="손해배상",
            n_results=5,
        )
        assert len(result.ids[0]) > 0

        # FTS 기여: 손해배상 관련 문서가 결과에 포함되어야 함
        all_ids = result.ids[0]
        damage_related = {"FTS_LAW_001_0", "FTS_PREC_001_0"}
        found = damage_related.intersection(set(all_ids))
        assert len(found) > 0, (
            f"FTS가 순위에 기여하지 않음: 손해배상 관련 문서 미포함. 결과: {all_ids}"
        )
