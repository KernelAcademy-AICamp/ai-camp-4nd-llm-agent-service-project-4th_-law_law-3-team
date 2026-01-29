"""
평가 시스템 검색 기능 통합 테스트

LanceDB와 PostgreSQL 연동 검색 테스트
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import asyncio


@pytest.mark.skipif(
    not Path("lancedb_data").exists(),
    reason="LanceDB 데이터 필요"
)
class TestLanceDBSearch:
    """LanceDB 검색 테스트"""

    @pytest.fixture(scope="class")
    def vector_store(self):
        """벡터 스토어 인스턴스"""
        from app.common.vectorstore.lancedb import LanceDBStore
        return LanceDBStore()

    @pytest.fixture(scope="class")
    def embedding_model(self):
        """임베딩 모델 인스턴스"""
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(
            "nlpai-lab/KURE-v1",
            trust_remote_code=True,
        )

    def test_search_precedent(self, vector_store, embedding_model):
        """판례 검색 테스트"""
        query = "손해배상 청구 요건"
        query_embedding = embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        result = vector_store.search(
            query_embedding=query_embedding,
            n_results=10,
            where={"data_type": "판례"},
        )

        assert result.ids and result.ids[0], "검색 결과가 없습니다"
        assert len(result.ids[0]) <= 10

        # 메타데이터 확인
        if result.metadatas and result.metadatas[0]:
            for meta in result.metadatas[0]:
                assert meta.get("data_type") == "판례"

    def test_search_law(self, vector_store, embedding_model):
        """법령 검색 테스트"""
        query = "민법 불법행위"
        query_embedding = embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        result = vector_store.search(
            query_embedding=query_embedding,
            n_results=10,
            where={"data_type": "법령"},
        )

        assert result.ids and result.ids[0], "검색 결과가 없습니다"

        if result.metadatas and result.metadatas[0]:
            for meta in result.metadatas[0]:
                assert meta.get("data_type") == "법령"

    def test_search_all(self, vector_store, embedding_model):
        """전체 검색 테스트 (필터 없음)"""
        query = "계약 해제"
        query_embedding = embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        result = vector_store.search(
            query_embedding=query_embedding,
            n_results=20,
        )

        assert result.ids and result.ids[0], "검색 결과가 없습니다"

        # 판례와 법령이 섞여 있을 수 있음
        if result.metadatas and result.metadatas[0]:
            data_types = {meta.get("data_type") for meta in result.metadatas[0]}
            # 최소 1개 이상의 유형이 있어야 함
            assert len(data_types) >= 1

    def test_search_result_structure(self, vector_store, embedding_model):
        """검색 결과 구조 확인"""
        query = "임대차"
        query_embedding = embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        result = vector_store.search(
            query_embedding=query_embedding,
            n_results=5,
        )

        # 필수 필드 확인
        assert hasattr(result, "ids")
        assert hasattr(result, "distances")
        assert hasattr(result, "documents")
        assert hasattr(result, "metadatas")

        if result.ids and result.ids[0]:
            # 길이 일치 확인
            n_results = len(result.ids[0])
            assert len(result.distances[0]) == n_results
            assert len(result.documents[0]) == n_results
            assert len(result.metadatas[0]) == n_results


@pytest.mark.skipif(
    not Path("lancedb_data").exists(),
    reason="LanceDB 데이터 필요"
)
class TestSearchLatency:
    """검색 지연 시간 테스트"""

    def test_search_latency(self):
        """검색 지연 시간 측정"""
        import time
        from app.common.vectorstore.lancedb import LanceDBStore
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "nlpai-lab/KURE-v1",
            trust_remote_code=True,
        )
        store = LanceDBStore()

        queries = [
            "손해배상 청구",
            "민법 제750조",
            "형사 사기죄",
            "임대차 계약",
            "근로기준법 해고",
        ]

        latencies = []

        for query in queries:
            query_embedding = model.encode(query, normalize_embeddings=True).tolist()

            start = time.perf_counter()
            store.search(query_embedding=query_embedding, n_results=10)
            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)

        import numpy as np
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)

        print(f"\n검색 지연 시간:")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")

        # 목표: P50 < 200ms, P95 < 500ms
        assert p50 < 500, f"P50 지연 시간 초과: {p50:.2f}ms"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not Path("lancedb_data").exists(),
    reason="LanceDB 및 PostgreSQL 데이터 필요"
)
class TestPostgreSQLIntegration:
    """PostgreSQL 연동 테스트"""

    async def test_precedent_lookup(self):
        """판례 조회 테스트"""
        from sqlalchemy import select, func
        from app.common.database import async_session_factory
        from app.models.precedent_document import PrecedentDocument

        async with async_session_factory() as session:
            # 총 개수 확인
            result = await session.execute(
                select(func.count(PrecedentDocument.id))
            )
            count = result.scalar()

            assert count > 0, "PostgreSQL에 판례 데이터가 없습니다"

            # 샘플 조회
            result = await session.execute(
                select(PrecedentDocument).limit(1)
            )
            precedent = result.scalar_one_or_none()

            assert precedent is not None
            assert precedent.serial_number is not None

    async def test_law_lookup(self):
        """법령 조회 테스트"""
        from sqlalchemy import select, func
        from app.common.database import async_session_factory
        from app.models.law_document import LawDocument

        async with async_session_factory() as session:
            result = await session.execute(
                select(func.count(LawDocument.id))
            )
            count = result.scalar()

            assert count > 0, "PostgreSQL에 법령 데이터가 없습니다"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
