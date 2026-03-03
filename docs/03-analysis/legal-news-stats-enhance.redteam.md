# Red Team 검증 보고서: 법률 뉴스 통계 대시보드 개선

> **Feature**: `legal-news-stats-enhance`
> **검증일**: 2026-02-28
> **검증 도구**: Gemini CLI (gemini-3-flash-preview)

---

## Red Team 검증 보고서

### 1. 취약점 (Critical/High/Medium/Low)

- **[Medium] SQL Injection 및 부적절한 파라미터 처리 (Injection):**
  - **내용:** `days` 쿼리 파라미터를 SQL `INTERVAL` 연산에 직접 결합할 경우 SQL Injection 위험이 있습니다. 비록 `ge=1, le=90`으로 검증하지만, 문자열 포맷팅(f-string) 등을 사용해 쿼리를 구성하는 습관은 지양해야 합니다.
  - **대응:** SQLAlchemy의 `text()` 또는 바인드 파라미터(`:days`)를 사용하여 안전하게 쿼리를 구성해야 합니다.

- **[High] 서비스 거부 공격 (DoS) 및 자원 고갈:**
  - **내용:** `GET /api/legal-news/stats/rag-contribution` API는 대규모 테이블(`law_documents`, `news_articles`)에 대해 실시간 `COUNT(*)`를 수행합니다. 데이터가 수백만 건 이상으로 증가할 경우, 해당 API 호출만으로 DB CPU 점유율을 100%로 끌어올려 전체 서비스를 마비시키는 DoS 공격 포인트가 될 수 있습니다.
  - **대응:** 통계 데이터에 대한 서버 사이드 캐싱(Redis 등)을 도입하고, 호출 주기를 제한(Rate Limiting)해야 합니다.

- **[Low] 정보 노출 (Information Disclosure):**
  - **내용:** RAG 인덱싱 현황(`is_indexed` 건수 등)을 외부에 노출함으로써, 시스템의 내부 처리 속도나 RAG 파이프라인의 병목 지점을 공격자가 유추할 수 있습니다.
  - **대응:** 일반 사용자에게는 구체적인 숫자보다 백분율(%) 위주의 정보를 제공하는 것을 권장합니다.

### 2. 아키텍처 개선 제안

- **모듈 간 강결합(Tight Coupling) 문제:**
  - **현황:** `legal_news` 모듈의 `service.py`에서 타 모듈의 `LawDocument` 모델을 직접 import하여 참조하도록 설계되었습니다.
  - **제안:** `app.core.stats` 등 공통 통계 인터페이스를 구축하거나, `LawDocument` 서비스에서 제공하는 API/내부 메서드를 통해 데이터를 가져오는 느슨한 결합 방식으로 변경해야 합니다.

- **프론트엔드 연산 부하 분산:**
  - **현황:** FR-01에서 프론트엔드에서 기존 데이터를 집계하여 계산한다고 명시되어 있습니다.
  - **제안:** `useMemo`를 통해 불필요한 재연산을 방지해야 합니다.

### 3. 고급 기능 추가 제안

- **상세 보기 드릴다운 (Drill-down):** 도넛 차트의 특정 카테고리 클릭 시 해당 카테고리로 필터링된 뉴스 목록으로 이동
- **트렌드 변화율(WoW/MoM) 표시:** "지난주 대비 수집량 15% 증가"와 같은 트렌드 변화 지표
- **RAG 검색 성능 지표 추가:** 실제 검색 결과에 포함된 횟수나 점수 기여도(Quality) 지표

### 4. 성능 최적화 제안

- **PostgreSQL 통계 전용 인덱스:** `(published_at, category)` 조합의 Composite Index 검토
- **Materialized View 또는 Summary Table:** RAG 기여도 데이터에 대한 1시간 단위 갱신 캐시

### 5. 운영 안정성 제안

- **Deadman's Snitch (수집 감시):** 수집 건수 0건 시 알림
- **로깅 강화:** 비정상적 요청 모니터링
- **에러 핸들링:** RAG 통계 조회 실패 시 Graceful Degradation

---

**총평:** 본 기획은 기능적으로 우수하나, 서비스 규모가 커질 경우 발생할 DB 부하와 모듈 간 의존성 관리가 핵심 리스크입니다.
