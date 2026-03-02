# Red Team 코드 리뷰 보고서 — 법률 뉴스 통계 그래프

## 리뷰 대상
- Backend: schema, service, router (3파일)
- Frontend: types, services, hook, charts, page layout (7파일)

## 1. 취약점 (Critical/High/Medium/Low)

### Critical: 서비스 거부 공격(DoS) 가능성 (In-memory Data Processing)
- `get_news_category_stats`가 **모든 뉴스 데이터**를 메모리로 로드하여 Python에서 분류
- 데이터 수만 건 이상 시 OOM 또는 CPU 점유율 급증 위험

### High: 검색 인젝션 및 부하 (Hybrid Search)
- `search_news_service`에서 쿼리를 LanceDB/리랭커로 직접 전달 (기존 코드, 이번 변경과 무관)

### Medium: SQL 성능 저하 (Function on Column)
- `func.date(NewsArticle.published_at)` 사용 시 인덱스 무효화 → Full Table Scan 가능

### Low: 인증/Rate Limiting 부재
- 공개 서비스 시 무단 크롤링 위험 (기존 전체 API에 해당)

## 2. 버그 및 논리적 오류

- 카테고리 분류 대소문자 처리 불일치 (한국어이므로 실질적 영향 미미)
- 타임존 처리 혼선 (`kst` vs DB 타임존 설정)

## 3. 성능 최적화 제안

- SQL Aggregation(CASE WHEN) 활용하여 DB단 분류
- `published_at::date` 함수형 인덱스 생성
- 통계 캐싱 (5~10분 단위)

## 4. 코드 품질 개선 제안

- 카테고리 규칙 설정 파일 분리
- 프론트엔드 예기치 못한 소스값 Fallback

## 5. 에러 핸들링 개선 제안

- 검색 서비스 외부 모듈 예외 처리 (기존 코드)
- 프론트엔드 에러 메시지 세분화 + 재시도 버튼

## Agent Team 평가

| 항목 | 채택 여부 | 사유 |
|------|----------|------|
| Critical: 전체 로드 DoS | ✅ 채택 | `get_news_category_stats`에 SQL CASE WHEN + GROUP BY 적용 |
| Medium: 함수형 인덱스 | ⏭️ 보류 | 현재 데이터량 소규모, 추후 성능 이슈 시 적용 |
| 타임존 혼선 | ✅ 채택 | `collected_at` 대신 `published_at` 기준 통일 확인 |
| 카테고리 규칙 외부화 | ⏭️ 보류 | 현재 7개 규칙으로 충분, 추후 관리자 기능 추가 시 적용 |
| 에러 메시지 세분화 | ⏭️ 보류 | 현재 MVP 단계, 추후 UX 개선 시 적용 |
