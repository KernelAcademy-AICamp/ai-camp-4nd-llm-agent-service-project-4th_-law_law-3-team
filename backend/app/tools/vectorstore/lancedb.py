"""
LanceDB 벡터 저장소 구현체 (v2)

스키마 v2 기반:
- 단일 테이블 + NULL 허용 방식
- 법령과 판례를 data_type 컬럼으로 구분
- 해당하지 않는 필드는 NULL

디스크 기반 벡터 DB로 메모리 효율적인 대용량 데이터 처리 지원
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

import lancedb
import pyarrow as pa

from app.core.config import settings
from app.tools.vectorstore.base import VectorStoreBase, SearchResult
from app.tools.vectorstore.schema_v2 import (
    LEGAL_CHUNKS_SCHEMA,
    TABLE_NAME,
    COMMON_COLUMNS,
    LAW_COLUMNS,
    PRECEDENT_COLUMNS,
    create_law_chunk,
    create_precedent_chunk,
)


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
        promulgation_dates: Optional[List[str]] = None,
        promulgation_nos: Optional[List[str]] = None,
        law_types: Optional[List[str]] = None,
        article_nos: Optional[List[str]] = None,
    ) -> None:
        """법령 문서 배치 추가"""
        if not source_ids:
            return

        table = self._ensure_table()
        data = []

        for i in range(len(source_ids)):
            chunk = create_law_chunk(
                source_id=source_ids[i],
                chunk_index=chunk_indices[i],
                title=titles[i],
                content=contents[i],
                vector=embeddings[i],
                enforcement_date=enforcement_dates[i],
                department=departments[i],
                total_chunks=total_chunks_list[i] if total_chunks_list else 1,
                promulgation_date=promulgation_dates[i] if promulgation_dates else None,
                promulgation_no=promulgation_nos[i] if promulgation_nos else None,
                law_type=law_types[i] if law_types else None,
                article_no=article_nos[i] if article_nos else None,
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
        case_numbers: Optional[List[str]] = None,
        case_types: Optional[List[str]] = None,
        judgment_types: Optional[List[str]] = None,
        judgment_statuses: Optional[List[str]] = None,
        reference_provisions_list: Optional[List[str]] = None,
        reference_cases_list: Optional[List[str]] = None,
    ) -> None:
        """판례 문서 배치 추가"""
        if not source_ids:
            return

        table = self._ensure_table()
        data = []

        for i in range(len(source_ids)):
            chunk = create_precedent_chunk(
                source_id=source_ids[i],
                chunk_index=chunk_indices[i],
                title=titles[i],
                content=contents[i],
                vector=embeddings[i],
                decision_date=decision_dates[i],
                court_name=court_names[i],
                total_chunks=total_chunks_list[i] if total_chunks_list else 1,
                case_number=case_numbers[i] if case_numbers else None,
                case_type=case_types[i] if case_types else None,
                judgment_type=judgment_types[i] if judgment_types else None,
                judgment_status=judgment_statuses[i] if judgment_statuses else None,
                reference_provisions=reference_provisions_list[i] if reference_provisions_list else None,
                reference_cases=reference_cases_list[i] if reference_cases_list else None,
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

            if data_type == "법령":
                record = create_law_chunk(
                    source_id=source_id,
                    chunk_index=chunk_index,
                    title=meta.get("title", ""),
                    content=text,
                    vector=embeddings[i],
                    enforcement_date=meta.get("date", meta.get("enforcement_date", "")),
                    department=meta.get("source_name", meta.get("department", "")),
                    total_chunks=int(meta.get("total_chunks", 1)),
                    promulgation_date=meta.get("promulgation_date"),
                    promulgation_no=meta.get("promulgation_no"),
                    law_type=meta.get("law_type"),
                    article_no=meta.get("article_no"),
                )
            else:
                record = create_precedent_chunk(
                    source_id=source_id,
                    chunk_index=chunk_index,
                    title=meta.get("title", meta.get("case_name", "")),
                    content=text,
                    vector=embeddings[i],
                    decision_date=meta.get("date", meta.get("decision_date", "")),
                    court_name=meta.get("source_name", meta.get("court_name", "")),
                    total_chunks=int(meta.get("total_chunks", 1)),
                    case_number=meta.get("case_number"),
                    case_type=meta.get("case_type"),
                    judgment_type=meta.get("judgment_type"),
                    judgment_status=meta.get("judgment_status"),
                    reference_provisions=meta.get("reference_provisions"),
                    reference_cases=meta.get("reference_cases"),
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
        """문서 유형별 레코드 수"""
        if self._table is None:
            return 0

        escaped_type = self._escape_sql(data_type)

        try:
            df = self._table.search().where(f"data_type = '{escaped_type}'").limit(1000000).to_pandas()
            return len(df)
        except RuntimeError:
            try:
                df = self._table.search().where(f"doc_type = '{escaped_type}'").limit(1000000).to_pandas()
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

    def _extract_metadatas(self, df) -> List[Dict[str, Any]]:
        """DataFrame에서 메타데이터 추출"""
        metadatas = []
        for _, row in df.iterrows():
            meta = {
                "source_id": row.get("source_id"),
                "data_type": row.get("data_type"),
                "title": row.get("title"),
                "date": row.get("date"),
                "source_name": row.get("source_name"),
                "chunk_index": row.get("chunk_index"),
                "total_chunks": row.get("total_chunks"),
            }

            # data_type에 따라 해당 필드만 추가
            if row.get("data_type") == "법령":
                meta.update({
                    "promulgation_date": row.get("promulgation_date"),
                    "promulgation_no": row.get("promulgation_no"),
                    "law_type": row.get("law_type"),
                    "article_no": row.get("article_no"),
                })
            elif row.get("data_type") == "판례":
                meta.update({
                    "case_number": row.get("case_number"),
                    "case_type": row.get("case_type"),
                    "judgment_type": row.get("judgment_type"),
                    "judgment_status": row.get("judgment_status"),
                    "reference_provisions": row.get("reference_provisions"),
                    "reference_cases": row.get("reference_cases"),
                })

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
