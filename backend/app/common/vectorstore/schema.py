"""
LanceDB 스키마 정의

PyArrow 스키마를 사용하여 동적 벡터 차원을 지원합니다.
임베딩 모델에 따라 벡터 차원이 달라질 수 있으므로 런타임에 스키마를 생성합니다.
"""

import pyarrow as pa


def create_legal_chunk_schema(vector_dim: int) -> pa.Schema:
    """
    법률 문서 청크용 PyArrow 스키마 생성

    Args:
        vector_dim: 임베딩 벡터 차원 (모델에 따라 다름)
            - OpenAI text-embedding-3-small: 1536
            - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: 384

    Returns:
        PyArrow 스키마
    """
    return pa.schema([
        # 벡터 데이터
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),

        # 식별자
        pa.field("id", pa.utf8()),
        pa.field("doc_id", pa.int64()),

        # 텍스트 데이터 (LanceDB는 디스크 기반이라 원문 저장에 부담 없음)
        pa.field("text", pa.utf8()),

        # 메타데이터 (필터링용)
        pa.field("source", pa.utf8()),
        pa.field("doc_type", pa.utf8()),
        pa.field("chunk_index", pa.int32()),
        pa.field("case_number", pa.utf8()),
        pa.field("court_name", pa.utf8()),
        pa.field("decision_date", pa.utf8()),

        # 구조 정보 (정밀 검색 시 활용)
        pa.field("chunk_start", pa.int32()),
        pa.field("chunk_end", pa.int32()),
    ])
