# 외부 컨설팅 코드 리뷰 보고서 — 법률 뉴스 통계 그래프

## 리뷰어: Codex CLI (gpt-5.3-codex, read-only)

### 1. 아키텍처 평가
- **High**: 카테고리 통계가 전체 기사를 메모리 로드 후 Python 분류 → DB 집계 전환 필요
- **Medium**: 카테고리 규칙 하드코딩 (OCP 위배)
- **Medium**: search_news_service의 db 미사용 + 외부 의존 결합 (기존 코드)

### 2. 타입 안전성 및 API 계약
- **High**: 백엔드 source가 자유 문자열 vs 프론트 'lawtimes'|'naver' 유니온 불일치
- **High**: 막대 차트에서 lawtimes 아닌 값을 전부 naver로 집계 → 미지 소스 오표시
- **Medium**: published_at 타입 혼재 (datetime vs string)

### 3. 성능 및 확장성
- **High**: 전체 데이터 로딩 + Python 분류 → 병목
- **Medium**: 기간 변경 시 카테고리 통계 불필요 재조회
- **Medium**: 빈 날짜 0건 보간 미적용 → 시계열 왜곡

### 4. 모범 사례 대비 격차
- **High**: 관련 테스트 미존재
- **Medium**: 수동 useEffect 기반 페칭 (React Query/SWR 미사용)
- **Low**: 접근성 속성 부족

### 5. 우선 개선 권고사항
- P0: API source 타입 강제, 카테고리 SQL 집계 전환
- P1: 미지 소스 별도 버킷, 날짜 0건 보간
- P1: 테스트 추가
- P2: React Query 도입

## Agent Team 채택 결정

| 항목 | 우선순위 | 채택 | 사유 |
|------|---------|------|------|
| 카테고리 SQL 집계 전환 | P0 | ✅ | 양쪽 리뷰어 모두 Critical/High 지적 |
| 막대차트 미지 소스 처리 | P1 | ✅ | 데이터 정합성 향상 |
| 기간 변경 시 카테고리 재조회 방지 | P1 | ✅ | 불필요한 네트워크 비용 절감 |
| 날짜 0건 보간 | P1 | ✅ | 시계열 가독성 향상 |
| 테스트 추가 | P1 | ⏭️ 보류 | 현재 세션 범위 밖, 별도 작업 |
| React Query 도입 | P2 | ⏭️ 보류 | 기존 프로젝트 패턴 유지 |
| 접근성 속성 | P2 | ⏭️ 보류 | 추후 UX 개선 시 적용 |
