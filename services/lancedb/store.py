"""
LanceDB 저장소 래퍼 (마이크로서비스용)

backend/app/tools/vectorstore/lancedb.py 핵심 검색 로직 추출.
app 패키지 의존 없이 독립 동작.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb

from tokenizer import LegalTermDictionary, MeCabTokenizer

logger = logging.getLogger(__name__)

# Per-thread MeCabTokenizer 캐시
_thread_local = threading.local()


def _get_thread_tokenizer(
    legal_dict: Optional[LegalTermDictionary],
    userdic_path: Optional[str],
) -> MeCabTokenizer:
    """스레드별 MeCabTokenizer 인스턴스 반환 (캐싱)"""
    tokenizer = getattr(_thread_local, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer

    _thread_local.tokenizer = MeCabTokenizer(
        legal_dict=legal_dict,
        userdic_path=userdic_path,
    )
    return _thread_local.tokenizer


class LanceDBServiceStore:
    """마이크로서비스용 LanceDB 저장소"""

    def __init__(
        self,
        db_uri: Optional[str] = None,
        table_name: Optional[str] = None,
        legal_dict: Optional[LegalTermDictionary] = None,
        userdic_path: Optional[str] = None,
    ) -> None:
        uri = db_uri or os.environ.get("LANCEDB_URI", "./lancedb_data")
        self.table_name = table_name or os.environ.get("LANCEDB_TABLE_NAME", "legal_chunks")
        self._legal_dict = legal_dict
        self._userdic_path = userdic_path

        db_path = Path(uri)
        db_path.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(db_path))
        self._table: Optional[lancedb.table.Table] = None

        if self.table_name in self.db.table_names():
            self._table = self.db.open_table(self.table_name)

    @property
    def is_ready(self) -> bool:
        return self._table is not None

    def count(self) -> int:
        if self._table is None:
            return 0
        return len(self._table)

    def count_by_type(self, data_type: str) -> int:
        if self._table is None:
            return 0

        escaped_type = self._escape_sql(data_type)
        try:
            df = (
                self._table.search()
                .where(f"data_type = '{escaped_type}'")
                .select(["id"])
                .limit(1_000_000)
                .to_pandas()
            )
            return len(df)
        except RuntimeError:
            return 0

    # =========================================================================
    # 벡터 검색
    # =========================================================================

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """벡터 유사 검색"""
        empty = {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

        if self._table is None:
            return empty

        query = self._table.search(query_embedding).limit(n_results)

        if where:
            conditions = self._build_filter_conditions(where)
            if conditions:
                query = query.where(" AND ".join(conditions))

        df = query.to_pandas()

        if df.empty:
            return empty

        return {
            "ids": [df["id"].tolist()],
            "distances": [df["_distance"].tolist()],
            "documents": [df["content"].tolist()],
            "metadatas": [self._extract_metadatas(df)],
        }

    # =========================================================================
    # FTS 검색
    # =========================================================================

    def search_fts(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """MeCab 사전 토크나이징 기반 FTS 검색"""
        empty = {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

        if self._table is None:
            return empty

        tokenizer = _get_thread_tokenizer(self._legal_dict, self._userdic_path)
        tokenized_query = tokenizer.tokenize_query(query)

        if not tokenized_query.strip():
            return empty

        fts_query = self._table.search(tokenized_query, query_type="fts").limit(n_results)

        if where:
            conditions = self._build_filter_conditions(where)
            if conditions:
                fts_query = fts_query.where(" AND ".join(conditions))

        try:
            df = fts_query.to_pandas()
        except Exception as e:
            logger.warning("FTS 검색 실패: %s", e)
            return empty

        if df.empty:
            return empty

        scores = df["_score"].tolist()
        distances = [1.0 / (1.0 + s) for s in scores]

        return {
            "ids": [df["id"].tolist()],
            "distances": [distances],
            "documents": [df["content"].tolist()],
            "metadatas": [self._extract_metadatas(df)],
        }

    # =========================================================================
    # 하이브리드 검색 (RRF)
    # =========================================================================

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        rrf_k: int = 60,
    ) -> Dict[str, Any]:
        """벡터 + FTS 하이브리드 검색 (RRF 결합)"""
        fetch_k = n_results * 2
        vector_result = self.search(query_embedding, n_results=fetch_k, where=where)
        fts_result = self.search_fts(query_text, n_results=fetch_k, where=where)

        vector_ids = vector_result["ids"][0] if vector_result["ids"][0] else []
        fts_ids = fts_result["ids"][0] if fts_result["ids"][0] else []

        if not fts_ids:
            return self.search(query_embedding, n_results=n_results, where=where)

        rrf_ranked_ids = self._reciprocal_rank_fusion(vector_ids, fts_ids, k=rrf_k)
        final_ids = rrf_ranked_ids[:n_results]

        if not final_ids:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

        doc_result = self.get_by_ids(final_ids)

        # RRF 랭킹 순서로 재정렬
        id_to_idx: Dict[str, int] = {
            doc_id: i for i, doc_id in enumerate(doc_result["ids"])
        }
        ordered_indices = [
            id_to_idx[doc_id] for doc_id in final_ids if doc_id in id_to_idx
        ]
        ordered_ids = [doc_result["ids"][i] for i in ordered_indices]
        ordered_docs = [doc_result["documents"][i] for i in ordered_indices] if doc_result.get("documents") else []
        ordered_metas = [doc_result["metadatas"][i] for i in ordered_indices] if doc_result.get("metadatas") else []

        rrf_distances = [i / max(len(ordered_ids), 1) for i in range(len(ordered_ids))]

        return {
            "ids": [ordered_ids],
            "distances": [rrf_distances],
            "documents": [ordered_docs],
            "metadatas": [ordered_metas],
        }

    # =========================================================================
    # 문서 조회
    # =========================================================================

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """ID로 문서 조회"""
        if self._table is None or not ids:
            return {"ids": [], "documents": [], "metadatas": []}

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
        """source_id로 모든 청크 조회"""
        if self._table is None:
            return {"ids": [], "documents": [], "metadatas": []}

        escaped_id = self._escape_sql(source_id)
        df = self._table.search().where(f"source_id = '{escaped_id}'").limit(1000).to_pandas()

        if df.empty:
            return {"ids": [], "documents": [], "metadatas": []}

        df = df.sort_values("chunk_index")

        return {
            "ids": df["id"].tolist(),
            "documents": df["content"].tolist(),
            "metadatas": self._extract_metadatas(df),
        }

    # =========================================================================
    # 인덱스
    # =========================================================================

    def create_vector_index(self, index_type: str = "IVF_FLAT") -> bool:
        """벡터 인덱스 생성"""
        if self._table is None:
            logger.warning("벡터 인덱스 생성 스킵: 테이블 없음")
            return False

        row_count = len(self._table)
        if row_count == 0:
            logger.warning("벡터 인덱스 생성 스킵: 테이블 비어있음")
            return False

        try:
            existing = self._table.list_indices()
            for idx in existing:
                idx_columns = idx.get("columns", []) if isinstance(idx, dict) else getattr(idx, "columns", [])
                if "vector" in idx_columns:
                    logger.info("벡터 인덱스 이미 존재, 스킵")
                    return False
        except Exception:
            pass

        num_partitions = max(16, int(row_count**0.5))
        logger.info(
            "벡터 인덱스 생성: type=%s, partitions=%d, rows=%d",
            index_type, num_partitions, row_count,
        )

        self._table.create_index(
            metric="cosine",
            index_type=index_type,
            num_partitions=num_partitions,
            vector_column_name="vector",
            replace=True,
        )
        logger.info("벡터 인덱스 생성 완료")
        return True

    def create_fts_index(self, field: str = "content_tokenized") -> None:
        """FTS 인덱스 생성"""
        if self._table is None:
            logger.warning("FTS 인덱스 생성 스킵: 테이블 없음")
            return
        self._table.create_fts_index(field, replace=True)
        logger.info("FTS 인덱스 생성 완료: %s", field)

    # =========================================================================
    # 내부 유틸
    # =========================================================================

    @staticmethod
    def _escape_sql(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _build_filter_conditions(where: Dict[str, Any]) -> List[str]:
        conditions: List[str] = []
        for key, value in where.items():
            if isinstance(value, str):
                escaped = value.replace("'", "''")
                conditions.append(f"{key} = '{escaped}'")
            elif isinstance(value, (int, float)):
                conditions.append(f"{key} = {value}")
            elif isinstance(value, list):
                if all(isinstance(v, str) for v in value):
                    formatted = ", ".join([f"'{v.replace(chr(39), chr(39)*2)}'" for v in value])
                else:
                    formatted = ", ".join([str(v) for v in value])
                conditions.append(f"{key} IN ({formatted})")
        return conditions

    @staticmethod
    def _extract_metadatas(df: Any) -> List[Dict[str, Any]]:
        """DataFrame에서 메타데이터 추출"""
        metadatas: List[Dict[str, Any]] = []
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

            ct = row.get("content_tokenized")
            if ct is not None:
                meta["content_tokenized"] = ct

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

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_ids: List[str],
        fts_ids: List[str],
        k: int = 60,
    ) -> List[str]:
        """RRF 알고리즘"""
        scores: Dict[str, float] = {}
        for rank, doc_id in enumerate(vector_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, doc_id in enumerate(fts_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
