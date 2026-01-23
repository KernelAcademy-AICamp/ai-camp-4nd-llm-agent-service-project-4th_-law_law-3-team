"""
LanceDB 벡터 저장소 구현체

디스크 기반 벡터 DB로 메모리 효율적인 대용량 데이터 처리 지원
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

import lancedb
import pyarrow as pa

from app.core.config import settings
from app.common.vectorstore.base import VectorStoreBase, SearchResult
from app.common.vectorstore.schema import create_legal_chunk_schema


class LanceDBStore(VectorStoreBase):
    """
    LanceDB 기반 벡터 저장소 구현체

    디스크 기반이므로 메모리 효율적이며, 원문 텍스트 저장에 부담이 없습니다.

    Usage:
        store = LanceDBStore()

        # 문서 추가
        store.add_documents(
            ids=["doc_1", "doc_2"],
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            metadatas=[{"doc_type": "precedent"}, {"doc_type": "precedent"}],
            documents=["텍스트1", "텍스트2"],  # LanceDB는 텍스트 저장 권장
        )

        # 유사 문서 검색
        results = store.search(query_embedding=[0.1, 0.2, ...], n_results=5)
    """

    def __init__(self, collection_name: Optional[str] = None):
        # 데이터 디렉토리 생성
        db_path = Path(settings.LANCEDB_URI)
        db_path.mkdir(parents=True, exist_ok=True)

        # DB 연결
        self.db = lancedb.connect(str(db_path))
        self.table_name = collection_name or settings.LANCEDB_TABLE_NAME
        self._table: Optional[lancedb.table.Table] = None
        self._vector_dim: Optional[int] = None

        # 기존 테이블이 있으면 열기
        if self.table_name in self.db.table_names():
            self._table = self.db.open_table(self.table_name)

    @property
    def table(self) -> Optional[lancedb.table.Table]:
        """테이블 접근자 (lazy initialization)"""
        return self._table

    def _ensure_table(self, vector_dim: int) -> lancedb.table.Table:
        """
        테이블 존재 확인 및 생성

        Args:
            vector_dim: 임베딩 벡터 차원

        Returns:
            LanceDB 테이블
        """
        if self._table is not None:
            return self._table

        # 새 테이블 생성
        schema = create_legal_chunk_schema(vector_dim)
        self._table = self.db.create_table(
            self.table_name,
            schema=schema,
        )
        self._vector_dim = vector_dim
        return self._table

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """
        문서 추가 (임베딩 포함)

        Args:
            ids: 문서 ID 목록
            embeddings: 임베딩 벡터 목록
            metadatas: 메타데이터 목록 (필터링용)
            documents: 문서 텍스트 목록 (LanceDB에서는 저장 권장)
        """
        if not ids or not embeddings:
            return

        # 벡터 차원 확인 및 테이블 초기화
        vector_dim = len(embeddings[0])
        table = self._ensure_table(vector_dim)

        # 데이터 변환
        data = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if metadatas else {}
            text = documents[i] if documents else ""

            record = {
                "vector": embeddings[i],
                "id": doc_id,
                "doc_id": int(meta.get("doc_id", 0)),
                "text": text,
                "source": meta.get("source", "unknown"),
                "doc_type": meta.get("doc_type", "unknown"),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "case_number": meta.get("case_number", "") or "",
                "court_name": meta.get("court_name", "") or "",
                "decision_date": str(meta.get("decision_date", "") or ""),
                "chunk_start": int(meta.get("chunk_start", 0)),
                "chunk_end": int(meta.get("chunk_end", 0)),
            }
            data.append(record)

        # LanceDB에 배치 추가
        if data:
            table.add(data)

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> SearchResult:
        """
        유사 문서 검색

        Args:
            query_embedding: 검색 쿼리 임베딩 벡터
            n_results: 반환할 결과 수
            where: 필터 조건 (예: {"doc_type": "precedent"})
            include: 결과에 포함할 필드 (호환성용, LanceDB는 모든 필드 반환)

        Returns:
            SearchResult 객체
        """
        if self._table is None:
            return SearchResult(ids=[[]], distances=[[]], metadatas=[[]], documents=[[]])

        # 쿼리 빌더 시작
        query = self._table.search(query_embedding).limit(n_results)

        # 필터 적용 (SQL 스타일 문자열로 변환)
        if where:
            filter_conditions = []
            for key, value in where.items():
                if isinstance(value, str):
                    # SQL 인젝션 방지를 위해 작은따옴표 이스케이프
                    escaped_value = value.replace("'", "''")
                    filter_conditions.append(f"{key} = '{escaped_value}'")
                elif isinstance(value, (int, float)):
                    filter_conditions.append(f"{key} = {value}")

            if filter_conditions:
                query = query.where(" AND ".join(filter_conditions))

        # 검색 실행 (Pandas DataFrame으로 반환)
        df = query.to_pandas()

        if df.empty:
            return SearchResult(ids=[[]], distances=[[]], metadatas=[[]], documents=[[]])

        # 결과 변환 (2중 리스트 구조 유지 - Base 인터페이스 호환)
        ids_list = df["id"].tolist()
        distances_list = df["_distance"].tolist()
        documents_list = df["text"].tolist()

        # 메타데이터 추출
        metadata_columns = [
            "doc_id", "source", "doc_type", "chunk_index",
            "case_number", "court_name", "decision_date",
            "chunk_start", "chunk_end"
        ]
        metadatas_list = df[metadata_columns].to_dict(orient="records")

        return SearchResult(
            ids=[ids_list],
            distances=[distances_list],
            documents=[documents_list],
            metadatas=[metadatas_list],
        )

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        ID로 문서 조회

        Args:
            ids: 조회할 문서 ID 목록

        Returns:
            조회 결과 딕셔너리
        """
        if self._table is None or not ids:
            return {"ids": [], "documents": [], "metadatas": []}

        # SQL IN 절 생성
        formatted_ids = ", ".join([f"'{id.replace(chr(39), chr(39)+chr(39))}'" for id in ids])
        df = self._table.search().where(f"id IN ({formatted_ids})").limit(len(ids)).to_pandas()

        if df.empty:
            return {"ids": [], "documents": [], "metadatas": []}

        # 메타데이터 추출
        metadata_columns = [
            "doc_id", "source", "doc_type", "chunk_index",
            "case_number", "court_name", "decision_date",
            "chunk_start", "chunk_end"
        ]

        return {
            "ids": df["id"].tolist(),
            "documents": df["text"].tolist(),
            "metadatas": df[metadata_columns].to_dict(orient="records"),
        }

    def delete_by_ids(self, ids: List[str]) -> None:
        """
        ID로 문서 삭제

        Args:
            ids: 삭제할 문서 ID 목록
        """
        if self._table is None or not ids:
            return

        # SQL IN 절 생성 (SQL 인젝션 방지)
        formatted_ids = ", ".join([f"'{id.replace(chr(39), chr(39)+chr(39))}'" for id in ids])
        self._table.delete(f"id IN ({formatted_ids})")

    def count(self) -> int:
        """컬렉션 문서 수"""
        if self._table is None:
            return 0
        return len(self._table)

    def reset(self) -> None:
        """컬렉션 초기화 (주의: 모든 데이터 삭제)"""
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
        self._table = None
        self._vector_dim = None

    # ChromaDB 호환성을 위한 collection 속성
    @property
    def collection(self) -> "LanceDBCollectionAdapter":
        """ChromaDB API 호환성을 위한 어댑터"""
        return LanceDBCollectionAdapter(self)


class LanceDBCollectionAdapter:
    """
    ChromaDB collection API 호환성을 위한 어댑터

    create_embeddings.py의 기존 코드와 호환성 유지용
    """

    def __init__(self, store: LanceDBStore):
        self._store = store

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """ChromaDB의 collection.get() 호환"""
        if self._store._table is None:
            return {"ids": [], "metadatas": [], "documents": []}

        # 전체 조회 또는 필터 조회
        if ids:
            return self._store.get_by_ids(ids)

        # where 조건으로 조회
        query = self._store._table.search()

        if where:
            filter_conditions = []
            for key, value in where.items():
                if isinstance(value, str):
                    escaped_value = value.replace("'", "''")
                    filter_conditions.append(f"{key} = '{escaped_value}'")
                elif isinstance(value, (int, float)):
                    filter_conditions.append(f"{key} = {value}")

            if filter_conditions:
                query = query.where(" AND ".join(filter_conditions))

        # 대용량 조회를 위해 큰 limit 설정
        df = query.limit(1000000).to_pandas()

        if df.empty:
            return {"ids": [], "metadatas": [], "documents": []}

        metadata_columns = [
            "doc_id", "source", "doc_type", "chunk_index",
            "case_number", "court_name", "decision_date",
            "chunk_start", "chunk_end"
        ]

        result: Dict[str, Any] = {"ids": df["id"].tolist()}

        if include is None or "metadatas" in include:
            result["metadatas"] = df[metadata_columns].to_dict(orient="records")

        if include is None or "documents" in include:
            result["documents"] = df["text"].tolist()

        return result
