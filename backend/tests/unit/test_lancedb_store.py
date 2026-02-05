"""
LanceDBStore 단위 테스트

대상:
- LanceDBStore.__init__(): 초기화
- add_law_documents(), add_precedent_documents(): 문서 추가
- search(), search_by_type(): 벡터 검색
- get_by_source_id(), delete_by_source_id(): CRUD
- count(), count_by_type(): 카운트
- reset(): 초기화
- _escape_sql(), _build_filter_conditions(): 내부 헬퍼
- get_vector_store(): 팩토리 함수

모든 테스트는 tmp_path 기반 실제 LanceDB를 사용.
"""

from typing import Callable

from app.tools.vectorstore.base import SearchResult
from app.tools.vectorstore.lancedb import LanceDBStore

# ============================================================================
# 초기화 테스트 (FR-05)
# ============================================================================


class TestStoreInitialization:
    """LanceDBStore 초기화 테스트"""

    def test_store_initialization_default(self, lancedb_store: LanceDBStore) -> None:
        """기본 설정으로 초기화 시 DB 연결 및 테이블명 확인"""
        assert lancedb_store.db is not None
        assert lancedb_store.table_name == "test_legal_chunks"

    def test_store_initialization_custom_collection(
        self, tmp_path, monkeypatch
    ) -> None:
        """커스텀 collection_name으로 초기화"""
        from app.core.config import settings

        lancedb_dir = tmp_path / "custom_lancedb"
        lancedb_dir.mkdir()
        monkeypatch.setattr(settings, "LANCEDB_URI", str(lancedb_dir))
        monkeypatch.setattr(settings, "LANCEDB_TABLE_NAME", "default_table")

        store = LanceDBStore(collection_name="custom_table")
        assert store.table_name == "custom_table"
        store.reset()


# ============================================================================
# 문서 추가 테스트 (FR-06, FR-07)
# ============================================================================


class TestAddDocuments:
    """문서 추가 테스트"""

    def test_add_law_documents(
        self,
        lancedb_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """법령 문서 추가 후 카운트 검증"""
        lancedb_store.add_law_documents(
            source_ids=["010719", "010719"],
            chunk_indices=[0, 1],
            embeddings=[make_random_vector(0), make_random_vector(1)],
            titles=["민법", "민법"],
            contents=["내용1", "내용2"],
            enforcement_dates=["2023-08-08", "2023-08-08"],
            departments=["법무부", "법무부"],
            total_chunks_list=[2, 2],
        )
        assert lancedb_store.count() == 2

    def test_add_precedent_documents(
        self,
        lancedb_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """판례 문서 추가 후 카운트 검증"""
        lancedb_store.add_precedent_documents(
            source_ids=["76396", "76396"],
            chunk_indices=[0, 1],
            embeddings=[make_random_vector(10), make_random_vector(11)],
            titles=["손해배상", "손해배상"],
            contents=["내용1", "내용2"],
            decision_dates=["2023-05-15", "2023-05-15"],
            court_names=["대법원", "대법원"],
            total_chunks_list=[2, 2],
        )
        assert lancedb_store.count() == 2

    def test_add_law_documents_metadata(
        self,
        lancedb_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """추가된 법령 문서의 메타데이터 필드 검증"""
        lancedb_store.add_law_documents(
            source_ids=["010719"],
            chunk_indices=[0],
            embeddings=[make_random_vector(0)],
            titles=["민법"],
            contents=["내용"],
            enforcement_dates=["2023-08-08"],
            departments=["법무부"],
            law_types=["법률"],
        )
        result = lancedb_store.get_by_ids(["010719_0"])
        assert len(result["ids"]) == 1
        assert result["metadatas"][0]["data_type"] == "법령"
        assert result["metadatas"][0]["title"] == "민법"

    def test_add_precedent_documents_metadata(
        self,
        lancedb_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """추가된 판례 문서의 메타데이터 필드 검증"""
        lancedb_store.add_precedent_documents(
            source_ids=["76396"],
            chunk_indices=[0],
            embeddings=[make_random_vector(10)],
            titles=["손해배상(기)"],
            contents=["내용"],
            decision_dates=["2023-05-15"],
            court_names=["대법원"],
            case_numbers=["2023다12345"],
            case_types=["민사"],
        )
        result = lancedb_store.get_by_ids(["76396_0"])
        assert len(result["ids"]) == 1
        assert result["metadatas"][0]["data_type"] == "판례"
        assert result["metadatas"][0]["case_number"] == "2023다12345"


# ============================================================================
# 검색 테스트 (FR-08, FR-09, FR-19)
# ============================================================================


class TestSearch:
    """벡터 검색 테스트"""

    def test_search_returns_search_result(
        self,
        populated_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """search()가 SearchResult 인스턴스를 반환하는지 확인"""
        result = populated_store.search(
            query_embedding=make_random_vector(0),
            n_results=3,
        )
        assert isinstance(result, SearchResult)
        assert result.ids is not None
        assert result.distances is not None

    def test_search_vector_similarity(
        self,
        populated_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """삽입된 벡터와 동일한 벡터로 검색 시 해당 문서가 최상위"""
        # seed=0으로 삽입된 첫 번째 법령 문서와 동일한 벡터
        query_vec = make_random_vector(0)
        result = populated_store.search(query_embedding=query_vec, n_results=1)

        assert len(result.ids[0]) >= 1
        assert result.ids[0][0] == "010719_0"

    def test_search_n_results(
        self,
        populated_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """n_results 파라미터가 반환 결과 수를 제한하는지 확인"""
        result = populated_store.search(
            query_embedding=make_random_vector(42),
            n_results=3,
        )
        assert len(result.ids[0]) <= 3

    def test_search_by_type_law(
        self,
        populated_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """data_type='법령' 필터 검색 시 법령만 반환"""
        result = populated_store.search_by_type(
            query_embedding=make_random_vector(42),
            data_type="법령",
            n_results=10,
        )
        for meta in result.metadatas[0]:
            assert meta["data_type"] == "법령"

    def test_search_by_type_precedent(
        self,
        populated_store: LanceDBStore,
        make_random_vector: Callable[[int], list[float]],
    ) -> None:
        """data_type='판례' 필터 검색 시 판례만 반환"""
        result = populated_store.search_by_type(
            query_embedding=make_random_vector(42),
            data_type="판례",
            n_results=10,
        )
        for meta in result.metadatas[0]:
            assert meta["data_type"] == "판례"


# ============================================================================
# 카운트 테스트 (FR-10)
# ============================================================================


class TestCount:
    """문서 수 카운트 테스트"""

    def test_count_total(self, populated_store: LanceDBStore) -> None:
        """전체 문서 수 카운트 (법령 6 + 판례 6 = 12)"""
        assert populated_store.count() == 12

    def test_count_by_type(self, populated_store: LanceDBStore) -> None:
        """유형별 문서 수 카운트"""
        assert populated_store.count_by_type("법령") == 6
        assert populated_store.count_by_type("판례") == 6


# ============================================================================
# source_id 기반 조회/삭제 테스트 (FR-11, FR-12)
# ============================================================================


class TestSourceIdOperations:
    """source_id 기반 CRUD 테스트"""

    def test_get_by_source_id(self, populated_store: LanceDBStore) -> None:
        """source_id로 해당 문서의 모든 청크 조회"""
        result = populated_store.get_by_source_id("010719")
        assert len(result["ids"]) == 2
        # chunk_index 정렬 확인
        metadatas = result["metadatas"]
        assert metadatas[0]["chunk_index"] <= metadatas[1]["chunk_index"]

    def test_get_by_source_id_not_found(
        self, populated_store: LanceDBStore
    ) -> None:
        """존재하지 않는 source_id 조회 시 빈 결과"""
        result = populated_store.get_by_source_id("nonexistent")
        assert result["ids"] == []

    def test_delete_by_source_id(self, populated_store: LanceDBStore) -> None:
        """source_id로 청크 삭제 후 카운트 감소 확인"""
        initial_count = populated_store.count()
        populated_store.delete_by_source_id("010719")
        assert populated_store.count() == initial_count - 2
        result = populated_store.get_by_source_id("010719")
        assert result["ids"] == []


# ============================================================================
# 내부 헬퍼 테스트 (FR-13, FR-14)
# ============================================================================


class TestInternalHelpers:
    """내부 헬퍼 함수 테스트"""

    def test_escape_sql(self, lancedb_store: LanceDBStore) -> None:
        """SQL 특수문자 이스케이프 처리"""
        assert lancedb_store._escape_sql("it's") == "it''s"
        assert lancedb_store._escape_sql("normal") == "normal"
        assert lancedb_store._escape_sql("a'b'c") == "a''b''c"

    def test_build_filter_conditions_string(
        self, lancedb_store: LanceDBStore
    ) -> None:
        """문자열 필터 조건 생성"""
        conditions = lancedb_store._build_filter_conditions(
            {"data_type": "법령"}
        )
        assert len(conditions) == 1
        assert conditions[0] == "data_type = '법령'"

    def test_build_filter_conditions_list(
        self, lancedb_store: LanceDBStore
    ) -> None:
        """리스트 필터 조건 (IN 절) 생성"""
        conditions = lancedb_store._build_filter_conditions(
            {"data_type": ["법령", "판례"]}
        )
        assert len(conditions) == 1
        assert "IN" in conditions[0]
        assert "'법령'" in conditions[0]
        assert "'판례'" in conditions[0]


# ============================================================================
# reset 테스트 (FR-15)
# ============================================================================


class TestReset:
    """테이블 초기화 테스트"""

    def test_reset(self, populated_store: LanceDBStore) -> None:
        """reset() 후 count()가 0인지 확인"""
        assert populated_store.count() > 0
        populated_store.reset()
        assert populated_store.count() == 0


# ============================================================================
# 팩토리 함수 테스트 (FR-16)
# ============================================================================


class TestFactory:
    """get_vector_store() 팩토리 함수 테스트"""

    def test_get_vector_store_returns_lancedb(
        self, tmp_path, monkeypatch
    ) -> None:
        """VECTOR_DB=lancedb일 때 LanceDBStore 인스턴스 반환"""
        from app.core.config import settings

        lancedb_dir = tmp_path / "factory_lancedb"
        lancedb_dir.mkdir()
        monkeypatch.setattr(settings, "LANCEDB_URI", str(lancedb_dir))
        monkeypatch.setattr(settings, "LANCEDB_TABLE_NAME", "factory_test")
        monkeypatch.setattr(settings, "VECTOR_DB", "lancedb")

        from app.tools.vectorstore import get_vector_store

        store = get_vector_store()
        assert isinstance(store, LanceDBStore)
        store.reset()


# ============================================================================
# SearchResult dict-like 접근 테스트 (FR-19)
# ============================================================================


class TestSearchResult:
    """SearchResult 데이터클래스 동작 테스트"""

    def test_search_result_dict_access(self) -> None:
        """SearchResult의 dict-like 접근 (get, __getitem__)"""
        result = SearchResult(
            ids=[["id1", "id2"]],
            distances=[[0.1, 0.2]],
            documents=[["doc1", "doc2"]],
            metadatas=[[{"key": "val1"}, {"key": "val2"}]],
        )
        assert result["ids"] == [["id1", "id2"]]
        assert result.get("distances") == [[0.1, 0.2]]
        assert result.get("nonexistent", "default") == "default"
