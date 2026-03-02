# 설계 리뷰 (External Consultant): 법률 뉴스 통계 대시보드 개선

> **Feature**: `legal-news-stats-enhance`
> **검증일**: 2026-02-28
> **검증 도구**: Codex CLI (gpt-5.3-codex, read-only)

---

## Codex CLI 설계 리뷰 보고서

> **참고**: Codex CLI가 stdin 설계 문서 대신 저장소의 `legal-news-pipeline.design.md`를 리뷰함.
> 동일 모듈(`legal_news`)에 대한 피드백이므로 본 Feature 관련 항목만 선별 분석.

### 1. 아키텍처 검토
- **[Critical]** `/{article_id}` 동적 경로가 다른 정적 경로보다 먼저 선언되면 라우팅 충돌 위험
- **[High]** 다이어그램 `GET /search` vs 실제 `POST /search` 메서드 불일치

### 2. API 계약 검토
- **[High]** Backend `source: str` vs Frontend `'lawtimes' | 'naver'` 타입 비대칭
- **[High]** `published_at` 타입 `datetime | None` vs `str | None` 혼재
- **[Medium]** `extra="ignore"` 사용으로 계약 깨짐 늦은 발견 위험

### 3. 성능 검토
- **[High]** 목록 조회 `count`가 `order_by` 포함 서브쿼리 그대로 카운트 → 불필요 비용
- **[Medium]** 검색 API 타임아웃/서킷브레이커 미명시

### 4. 개선 제안
1. API 계약 단일화: Backend `Enum` + Frontend 타입 생성 (OpenAPI 기반)
2. 라우팅/문서 정합성 수정
3. 타입 안전성 강화: Pydantic `extra="forbid"` / zod 도입

---

## PM 분석: 본 Feature(`legal-news-stats-enhance`)에 대한 반영 판단

> Codex CLI는 본 Feature 설계 문서가 아닌 상위 파이프라인 설계 문서를 리뷰함.
> 동일 모듈에 대한 기존 코드 품질 피드백으로, 본 Feature 구현에 직접 관련된 항목만 선별.

| # | 제안 | 채택 | 사유 |
|---|------|------|------|
| 1 | `/{article_id}` 라우팅 충돌 방지 | **이미 반영** | 설계 문서에 `/stats/rag-contribution`을 `/{article_id}` 앞에 배치 명시 |
| 2 | `source: str` vs 프론트엔드 타입 비대칭 | **미반영** | 본 Feature 스코프 외 (기존 코드 이슈). 별도 리팩토링 대상 |
| 3 | `published_at` 타입 혼재 | **미반영** | 본 Feature는 통계 집계만 다루며 `published_at` 직접 노출 안함 |
| 4 | `extra="ignore"` → `extra="forbid"` | **미반영** | 기존 모듈 전체 패턴 변경 필요. 본 Feature에서 신규 스키마는 `extra` 미설정 (기본 ignore) |
| 5 | count 쿼리 order_by 분리 | **미반영** | 본 Feature의 count는 `func.count()` 직접 사용 (order_by 없음) |
| 6 | 검색 API 타임아웃 | **미반영** | 본 Feature 스코프 외 |

**총평**: Codex CLI가 대상 문서를 잘못 선택했으나, 동일 모듈에 대한 유용한 피드백 제공. 본 Feature 설계에는 이미 핵심 사항(라우팅 순서)이 반영되어 있어 추가 수정 불필요.
