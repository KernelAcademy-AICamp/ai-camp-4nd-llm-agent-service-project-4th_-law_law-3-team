"""
판례/법령 인제스트 파이프라인

config-driven 구조로 다양한 데이터 타입을 지원하는 통합 인제스트 시스템.

모듈 구성:
- config: IngestConfig dataclass (데이터 타입별 설정)
- types/: 데이터 타입별 설정 (precedent, law 등)
- vector_writer: LanceDB 벡터 저장 + ANN 인덱스
- db_writer: PostgreSQL 적재 + BM25 search_text 동시 생성
- search_text_rebuilder: search_text 독립 재빌드 (토크나이저 변경 시)
- cli: 통합 CLI 엔트리포인트
"""
