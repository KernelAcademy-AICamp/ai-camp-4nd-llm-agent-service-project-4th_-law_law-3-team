"""
LanceDB 벡터 저장소 구현체 (v2)

스키마 v2 기반:
- 단일 테이블 + NULL 허용 방식
- 법령과 판례를 data_type 컬럼으로 구분
- 해당하지 않는 필드는 NULL
디스크 기반 벡터 DB로 메모리 효율적인 대용량 데이터 처리 지원
FTS는 PostgreSQL fts_index 테이블에서 수행 (keyword_search.py 참조)
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb

from app.core.config import settings
from app.tools.vectorstore.base import SearchResult, VectorStoreBase
from app.tools.vectorstore.schema_v2 import (
    LEGAL_CHUNKS_SCHEMA,
    create_chunk,
)

logger = logging.getLogger(__name__)

# Per-thread MeCabTokenizer 캐시 (MeCab Tagger는 thread-safe하지 않음)
_thread_local = threading.local()


def _load_decomposition_map() -> dict[str, list[str]]:
    """decomposition_map.json 로드 (프로세스 내 1회)"""
    cached = getattr(_load_decomposition_map, "_cache", None)
    if cached is not None:
        return cached

    import json

    path = Path(settings.MECAB_USERDIC_PATH).parent / "decomposition_map.json"
    if not path.exists():
        _load_decomposition_map._cache = {}  # type: ignore[attr-defined]
        return {}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    _load_decomposition_map._cache = data  # type: ignore[attr-defined]
    return data


def _get_thread_tokenizer() -> "MeCabTokenizer":  # noqa: F821
    """스레드별 MeCabTokenizer 인스턴스 반환 (캐싱)

    MeCab Tagger는 thread-safe하지 않으므로 threading.local()로
    스레드별 독립 인스턴스를 유지한다.
    """
    tokenizer = getattr(_thread_local, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer

    from app.tools.vectorstore.mecab_tokenizer import MeCabTokenizer

    _thread_local.tokenizer = MeCabTokenizer(
        userdic_path=str(Path(settings.MECAB_USERDIC_PATH)),
        decomposition_map=_load_decomposition_map(),
    )
    return _thread_local.tokenizer


class LanceDBStore(VectorStoreBase):
    """
    LanceDB 기반 벡터 저장소 구현체 (v2)

    스키마 v2 특징:
    - 단일 테이블에 법령/판례 통합 저장
    - data_type 컬럼으로 문서 유형 구분 ("법령" | "판례")
    - 해당하지 않는 필드는 NULL

    Usage:
        store = LanceDBStore()

        # 법령 문서 추가
        store.add_law_documents(
            source_ids=["010719"],
            embeddings=[[0.1, 0.2, ...]],
            titles=["민법"],
            contents=["[법령] 민법 제750조: ..."],
            enforcement_dates=["2023-08-08"],
            departments=["법무부"],
        )

        # 유사 문서 검색
        results = store.search(
            query_embedding=[0.1, 0.2, ...],
            n_results=5,
            where={"data_type": "판례"},
        )
    """

    def __init__(self, collection_name: Optional[str] = None):
        # 데이터 디렉토리 생성
        db_path = Path(settings.LANCEDB_URI)
        db_path.mkdir(parents=True, exist_ok=True)

        # DB 연결
        self.db = lancedb.connect(str(db_path))
        self.table_name = collection_name or settings.LANCEDB_TABLE_NAME
        self._table: Optional[lancedb.table.Table] = None

        # 기존 테이블이 있으면 열기
        if self.table_name in self.db.table_names():
            self._table = self.db.open_table(self.table_name)

    @property
    def table(self) -> Optional[lancedb.table.Table]:
        """테이블 접근자 (lazy initialization)"""
        return self._table

    def _ensure_table(self) -> lancedb.table.Table:
        """
        테이블 존재 확인 및 생성

        Returns:
            LanceDB 테이블
        """
        if self._table is not None:
            return self._table

        # 새 테이블 생성 (스키마 v2 사용)
        self._table = self.db.create_table(
            self.table_name,
            schema=LEGAL_CHUNKS_SCHEMA,
        )
        return self._table

    def add_law_documents(
        self,
        source_ids: List[str],
        chunk_indices: List[int],
        embeddings: List[List[float]],
        titles: List[str],
        contents: List[str],
        enforcement_dates: List[str],
        departments: List[str],
        total_chunks_list: Optional[List[int]] = None,
        **_kwargs: Any,
    ) -> None:
        """법령 문서 배치 추가"""
        if not source_ids:
            return

        table = self._ensure_table()
        data = []

        for i in range(len(source_ids)):
            chunk = create_chunk(
                data_type="법령",
                source_id=source_ids[i],
                title=titles[i],
                content=contents[i],
                vector=embeddings[i],
                source_name=departments[i],
                date=enforcement_dates[i] or None,
                chunk_index=chunk_indices[i],
                total_chunks=total_chunks_list[i] if total_chunks_list else 1,
            )
            data.append(chunk)

        if data:
            table.add(data)

    def add_precedent_documents(
        self,
        source_ids: List[str],
        chunk_indices: List[int],
        embeddings: List[List[float]],
        titles: List[str],
        contents: List[str],
        decision_dates: List[str],
        court_names: List[str],
        total_chunks_list: Optional[List[int]] = None,
        **_kwargs: Any,
    ) -> None:
        """판례 문서 배치 추가"""
        if not source_ids:
            return

        table = self._ensure_table()
        data = []

        for i in range(len(source_ids)):
            chunk = create_chunk(
                data_type="판례",
                source_id=source_ids[i],
                title=titles[i],
                content=contents[i],
                vector=embeddings[i],
                source_name=court_names[i],
                date=decision_dates[i] or None,
                chunk_index=chunk_indices[i],
                total_chunks=total_chunks_list[i] if total_chunks_list else 1,
            )
            data.append(chunk)

        if data:
            table.add(data)

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """문서 추가 (하위 호환성 유지용)"""
        if not ids or not embeddings:
            return

        table = self._ensure_table()
        data = []

        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if metadatas else {}
            text = documents[i] if documents else ""
            data_type = meta.get("data_type", meta.get("doc_type", "판례"))

            # ID 파싱 (source_id_chunkIndex 형식)
            parts = doc_id.rsplit("_", 1)
            source_id = parts[0] if len(parts) > 1 else doc_id
            chunk_index = int(parts[1]) if len(parts) > 1 else 0

            record = create_chunk(
                data_type=data_type,
                source_id=source_id,
                title=meta.get("title", ""),
                content=text,
                vector=embeddings[i],
                source_name=meta.get("source_name", ""),
                date=meta.get("date") or None,
                chunk_index=chunk_index,
                total_chunks=int(meta.get("total_chunks", 1)),
            )

            # id 덮어쓰기 (원래 전달된 id 사용)
            record["id"] = doc_id
            data.append(record)

        if data:
            table.add(data)

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> SearchResult:
        """유사 문서 검색"""
        if self._table is None:
            return SearchResult(ids=[[]], distances=[[]], metadatas=[[]], documents=[[]])

        # 쿼리 빌더
        query = self._table.search(query_embedding).limit(n_results)

        # 필터 적용 (SQL 스타일)
        if where:
            filter_conditions = self._build_filter_conditions(where)
            if filter_conditions:
                query = query.where(" AND ".join(filter_conditions))

        # 검색 실행
        df = query.to_pandas()

        if df.empty:
            return SearchResult(ids=[[]], distances=[[]], metadatas=[[]], documents=[[]])

        # 결과 변환
        ids_list = df["id"].tolist()
        distances_list = df["_distance"].tolist()
        documents_list = df["content"].tolist()

        # 메타데이터 추출 (data_type에 따라 해당 컬럼만)
        metadatas_list = self._extract_metadatas(df)

        return SearchResult(
            ids=[ids_list],
            distances=[distances_list],
            documents=[documents_list],
            metadatas=[metadatas_list],
        )

    def search_by_type(
        self,
        query_embedding: List[float],
        data_type: str,
        n_results: int = 10,
        additional_filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """문서 유형별 검색 (최적화)"""
        where = {"data_type": data_type}
        if additional_filters:
            where.update(additional_filters)
        return self.search(query_embedding, n_results, where)

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """ID로 문서 조회"""
        if self._table is None or not ids:
            return {"ids": [], "documents": [], "metadatas": []}

        # SQL IN 절 생성 (SQL 인젝션 방지)
        formatted_ids = ", ".join([f"'{self._escape_sql(id_)}'" for id_ in ids])
        df = self._table.search().where(f"id IN ({formatted_ids})").limit(len(ids)).to_pandas()

        if df.empty:
            return {"ids": [], "documents": [], "metadatas": []}

        return {
            "ids": df["id"].tolist(),
            "documents": df["content"].tolist(),
            "metadatas": self._extract_metadatas(df),
        }

    def get_by_source_id(self, source_id: str) -> Dict[str, Any]:
        """원본 문서 ID로 모든 청크 조회"""
        if self._table is None:
            return {"ids": [], "documents": [], "metadatas": []}

        escaped_id = self._escape_sql(source_id)
        df = self._table.search().where(f"source_id = '{escaped_id}'").limit(1000).to_pandas()

        if df.empty:
            return {"ids": [], "documents": [], "metadatas": []}

        # chunk_index로 정렬
        df = df.sort_values("chunk_index")

        return {
            "ids": df["id"].tolist(),
            "documents": df["content"].tolist(),
            "metadatas": self._extract_metadatas(df),
        }

    def delete_by_ids(self, ids: List[str]) -> None:
        """ID로 문서 삭제"""
        if self._table is None or not ids:
            return

        formatted_ids = ", ".join([f"'{self._escape_sql(id_)}'" for id_ in ids])
        self._table.delete(f"id IN ({formatted_ids})")

    def delete_by_source_id(self, source_id: str) -> None:
        """원본 문서 ID로 모든 청크 삭제"""
        if self._table is None:
            return

        escaped_id = self._escape_sql(source_id)
        self._table.delete(f"source_id = '{escaped_id}'")

    def count(self) -> int:
        """테이블 레코드 수"""
        if self._table is None:
            return 0
        return len(self._table)

    def count_by_type(self, data_type: str) -> int:
        """문서 유형별 레코드 수 (컬럼 최소 선택으로 메모리 절약)"""
        if self._table is None:
            return 0

        escaped_type = self._escape_sql(data_type)

        try:
            # select(["id"])로 벡터 컬럼 제외하여 메모리 절약
            df = (
                self._table.search()
                .where(f"data_type = '{escaped_type}'")
                .select(["id"])
                .limit(1_000_000)
                .to_pandas()
            )
            return len(df)
        except RuntimeError:
            try:
                df = (
                    self._table.search()
                    .where(f"doc_type = '{escaped_type}'")
                    .select(["id"])
                    .limit(1_000_000)
                    .to_pandas()
                )
                return len(df)
            except RuntimeError:
                return 0

    def reset(self) -> None:
        """테이블 초기화 (모든 데이터 삭제)"""
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
        self._table = None

    def _escape_sql(self, value: str) -> str:
        """SQL 인젝션 방지를 위한 이스케이프"""
        return value.replace("'", "''")

    def _build_filter_conditions(self, where: Dict[str, Any]) -> List[str]:
        """WHERE 조건 빌드"""
        conditions = []
        for key, value in where.items():
            if isinstance(value, str):
                escaped_value = self._escape_sql(value)
                conditions.append(f"{key} = '{escaped_value}'")
            elif isinstance(value, (int, float)):
                conditions.append(f"{key} = {value}")
            elif isinstance(value, list):
                if all(isinstance(v, str) for v in value):
                    formatted = ", ".join([f"'{self._escape_sql(v)}'" for v in value])
                else:
                    formatted = ", ".join([str(v) for v in value])
                conditions.append(f"{key} IN ({formatted})")
        return conditions

    def _extract_metadatas(self, df: Any) -> List[Dict[str, Any]]:
        """DataFrame에서 메타데이터 추출 (schema_v2 10컬럼 기준)"""
        metadatas = []
        for _, row in df.iterrows():
            meta: Dict[str, Any] = {
                "source_id": row.get("source_id"),
                "data_type": row.get("data_type"),
                "title": row.get("title"),
                "date": row.get("date"),
                "source_name": row.get("source_name"),
                "chunk_index": row.get("chunk_index"),
                "total_chunks": row.get("total_chunks"),
            }
            metadatas.append(meta)
        return metadatas

    # ChromaDB 호환성을 위한 collection 속성
    @property
    def collection(self) -> "LanceDBCollectionAdapter":
        """ChromaDB API 호환성을 위한 어댑터"""
        return LanceDBCollectionAdapter(self)


class LanceDBCollectionAdapter:
    """ChromaDB collection API 호환성을 위한 어댑터"""

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

        if ids:
            return self._store.get_by_ids(ids)

        # where 조건으로 조회
        query = self._store._table.search()

        if where:
            conditions = self._store._build_filter_conditions(where)
            if conditions:
                query = query.where(" AND ".join(conditions))

        df = query.limit(1000000).to_pandas()

        if df.empty:
            return {"ids": [], "metadatas": [], "documents": []}

        result: Dict[str, Any] = {"ids": df["id"].tolist()}

        if include is None or "metadatas" in include:
            result["metadatas"] = self._store._extract_metadatas(df)

        if include is None or "documents" in include:
            result["documents"] = df["content"].tolist()

        return result
