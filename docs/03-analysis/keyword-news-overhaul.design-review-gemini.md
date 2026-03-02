# Gemini CLI 설계 리뷰 보고서 — 키워드/뉴스 수집 시스템 전면 재설계

> **작성자**: Gemini CLI (시니어 소프트웨어 아키텍트)
> **대상**: `docs/02-design/features/keyword-news-overhaul.design.md` v1.0
> **작성일**: 2026-03-01

---

## 1. 종합 평가

본 설계는 기존의 단순 수집 구조를 넘어 **멀티 소스 통합(RRF), 5차원 스코어링, 서킷 브레이커** 등 엔터프라이즈급 안정성과 정교함을 갖춘 훌륭한 설계. 특히 `Provider Abstraction`을 통한 확장성 고려와 `CircuitBreaker` 도입은 외부 API 의존도가 높은 시스템에서 매우 적절한 선택.

**결론**: 즉시 구현 착수 가능. 아래 보완사항만 추가 검토 필요.

---

## 2. [보완] 추가/수정이 필요한 항목

### 2.1 CircuitBreaker 상태 관리의 영속성

현재 설계된 CircuitBreaker가 메모리 기반(in-memory)이라면, 멀티 워커(Gunicorn/Uvicorn workers) 환경에서 각 프로세스마다 상태가 달라질 수 있음. **Redis 기반의 공유 상태 관리**를 고려하거나, 최소한 프로세스 간 동기화 전략이 필요.

### 2.2 데이터 모델 정규화 (Dataclass vs Pydantic)

- `RawTrendItem`(Dataclass)과 `NewsArticle`(Pydantic)이 혼용됨
- 내부 로직(Tools)에서는 가벼운 Dataclass를 쓰더라도, 최종 서비스 레이어에서는 Pydantic으로 통일하여 **Validation 및 자동 문서화(Swagger)** 혜택을 온전히 누리는 것이 좋음
- `merged_sources: list[TrendSource]` 필드 추가 시, DB 저장(SQLAlchemy JSON type)과의 매핑 로직을 `NewsArticle` 모델에 명시 필요

### 2.3 병렬 처리 제어 (Concurrency Limit)

6개 이상의 소스를 병렬 수집(Step 2)할 때, 외부 API의 Rate Limit에 걸리지 않도록 `asyncio.Semaphore`를 통한 **동시 요청 수 제한** 필요.

---

## 3. [대안] 더 나은 접근 방식 제안

### 3.1 2단계 중복 제거(Dedup) 최적화

- 모든 기사의 임베딩을 비교하는 것은 O(N^2) 비용 발생
- **제안**: 1단계로 **SimHash 또는 MinHash**를 사용하여 텍스트가 거의 유사한 것들을 먼저 쳐내고, 남은 후보군에 대해서만 2단계 임베딩 비교를 수행하는 하이브리드 방식 추천

### 3.2 Weighted RRF 가중치의 동적 관리

- `SOURCE_AUTHORITY_WEIGHTS`가 하드코딩되어 있음
- **제안**: 소스별 '응답 성공률'이나 '최근 7일간의 클릭률(CTR)' 등을 반영하여 가중치를 주기적으로 미세 조정할 수 있는 **Configuration 서비스**로 분리하는 것이 확장성에 유리

### 3.3 Legal Relevance 산출 로직 강화

- 현재 가중 결합(kw 0.2 + RAG 0.4 + LLM 0.4) 방식에서, RAG 검색 시 단순 키워드가 아닌 **Legal Taxonomy(법률 분류 체계)**를 Prompt에 주입하여 LLM이 분류하게 하는 것이 정확도가 훨씬 높음

---

## 4. [확인] 설계가 적절한 항목

| 항목 | 평가 |
|------|------|
| Provider Abstraction (3계층) | 매우 뛰어남. 특정 API 차단 시 즉시 대응 가능 |
| Graceful Degradation | 견고함. `sources_failed` 반환하며 수집 지속 |
| Frontend-Backend 계약 | snake_case 유지 및 인터페이스 동기화 전략 명확 |
| Early Signal/Convergence | 법률 도메인 차별화 인사이트 제공 핵심 포인트 |

---

## 5. [보안 및 성능]

- **SSRF 방어**: `community_domains` 화이트리스트 필터링 도입 높이 평가
- **Embedding 모델 로드**: `KURE-v1` 모델이 무거우므로, API 서버 기동 시가 아닌 별도의 **Inference 전용 컨테이너/Worker** 또는 지연 로드(lazy load) 고려 필요

---

## Agent Team 반영 결정

| # | Gemini 제안 | 반영 여부 | 사유 |
|---|------------|----------|------|
| 1 | CircuitBreaker Redis 공유 | **Phase 3 연기** | 현재 단일 워커 운영, 스케일아웃 시 도입 |
| 2 | Dataclass/Pydantic 정규화 | **채택** | 서비스 레이어 경계에서 Pydantic 변환 명시 |
| 3 | asyncio.Semaphore 동시성 제한 | **채택** | `MAX_CONCURRENT_SOURCES = 4` 설정 추가 |
| 4 | SimHash/MinHash 사전 필터링 | **Phase 3 연기** | 현재 최대 60건 수준, N^2 비용 허용 범위 |
| 5 | RRF 가중치 동적 관리 | **Phase 3 연기** | 초기 고정값 운영 후 데이터 축적 후 도입 |
| 6 | Legal Taxonomy Prompt 주입 | **채택** | _calculate_legal_v2에 카테고리별 법률 분류 체계 주입 |
| 7 | Embedding 지연 로드 | **채택** | 기존 패턴(lazy singleton) 준수하여 첫 호출 시 로드 |
