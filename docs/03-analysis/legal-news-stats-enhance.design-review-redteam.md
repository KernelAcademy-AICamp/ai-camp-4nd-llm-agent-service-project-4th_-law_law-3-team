# 설계 리뷰 (Red Team): 법률 뉴스 통계 대시보드 개선

> **Feature**: `legal-news-stats-enhance`
> **검증일**: 2026-02-28
> **검증 도구**: Gemini CLI (설계 리뷰)

---

## Red Team 설계 리뷰 보고서

### 1. 아키텍처 및 상세 설계 검토

#### [보완] RAG 지표의 정합성 및 LanceDB 연동
- `news_indexed_count`를 PostgreSQL `is_indexed` 플래그에 의존 → 실제 LanceDB 벡터 인덱스 건수와의 정합성 확인 지표(`sync_status`) 추가 제안
- `LawDocument`에 대해서도 인덱싱 완료 여부 체크 → `law_indexed_count` 추가 제안

#### [대안] 대규모 데이터 대응을 위한 성능 최적화
- 데이터 수백만 건 시 매 요청마다 `func.count()` DB 부하 우려
- 캐시 테이블(Materialized View), Redis 캐싱 + 백그라운드 태스크 주기적 업데이트 제안

#### [확인] 보안 및 안정성
- `timedelta` + SQLAlchemy ORM 사용 → SQL Injection 안전 (적절)
- `/{article_id}` 경로 마지막 배치 → 경로 충돌 방지 (적절)

### 2. 프론트엔드 개선 제안

#### [보완] 차트 시각화 및 UX
- 카테고리 10개 이상 시 상위 N개 외 '기타' 그룹핑 로직 제안
- `RagContributionCard` 모바일 뷰 반응형 레이아웃 조정 필요

#### [대안] 데이터 페칭 전략
- React Query (TanStack Query) 도입 고려 → `useNewsStats` 내부 로직 단순화

### 3. 보안 고려사항
- `/stats/*` 엔드포인트 Rate Limiting 적용 제안 (`slowapi` 등)

---

## PM 분석: 설계 리뷰 피드백 반영 판단

| # | 제안 | 채택 | 사유 |
|---|------|------|------|
| 1 | LanceDB 실측 카운트 교차 검증 | **미반영** | LanceDB 카운트 조회는 별도 의존성 필요, 본 Feature 스코프 초과. `is_indexed` 필드로 충분. 향후 개선 시 검토 |
| 2 | 대규모 데이터 캐싱 (Redis/Materialized View) | **미반영** | 현재 데이터량 소규모 (법령 ~400건, 뉴스 ~200건). 성능 이슈 발생 시 후속 작업으로 |
| 3 | 카테고리 '기타' 그룹핑 | **반영** | 기존 service.py에 이미 7개 카테고리 CASE WHEN 분류 시 미매칭은 '기타'로 처리됨. 프론트엔드 추가 그룹핑 불필요 |
| 4 | 모바일 반응형 레이아웃 | **참고** | Tailwind 유틸리티로 기본 반응형 보장, 추가 미디어 쿼리는 필요 시 적용 |
| 5 | React Query 도입 | **미반영** | 프로젝트 전체에 React Query 미사용 중. 본 Feature만 도입하면 패턴 불일치. 전체 도입은 별도 결정 필요 |
| 6 | Rate Limiting | **미반영** | 본 Feature 스코프 외. 전체 API Rate Limiting은 별도 인프라 작업 |

**총평**: 설계가 견고하며 즉시 구현 착수 가능. 핵심 피드백(LanceDB 정합성, 캐싱)은 데이터 규모 증가 시 후속 과제로 기록.
