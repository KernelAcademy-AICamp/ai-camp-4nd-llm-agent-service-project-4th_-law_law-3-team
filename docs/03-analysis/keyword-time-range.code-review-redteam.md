# Red Team 코드 리뷰 보고서 - Keyword Time Range Selection

## 검증 도구
- Gemini CLI (Red Team)

---

## 발견 사항 (5건)

### 1. [Critical] 캐시 키 불일치 Bug → **즉시 수정 완료**
- `collect_keywords`에서 캐시 키가 `{user_id}:{time_range}:keywords`로 변경되었으나
- `search_keyword_news`에서는 기존 `{user_id}:keywords`로 조회하여 항상 KeywordNotFoundError 발생
- **수정**: `search_keyword_news`에서 모든 TimeRange 값의 캐시를 순회하여 keyword_id 탐색

### 2. [High] 하드코딩된 TEMP_USER_ID → **보류 (기존 이슈)**
- 모든 사용자가 동일 ID 사용. 인증 시스템 구현 시 교체 예정 (TODO 주석 존재)

### 3. [Medium] 인메모리 캐시 분산 환경 불일치 → **보류 (기존 이슈)**
- Redis 전환은 장기 과제로 분류

### 4. [Medium] 뉴스 검색 기간 "7d" 하드코딩 → **보류 (기존 이슈)**
- `search_news_for_keyword`의 time_range는 뉴스 검색 전용. 키워드 수집 시간 범위와는 독립적 기능

### 5. [Medium] SSRF 잔존 위험 → **보류 (기존 이슈)**
- 화이트리스트 필터링 적용 중. config 레벨 관리

---

## 채택 요약

| # | 심각도 | 판정 | 사유 |
|---|--------|------|------|
| 1 | Critical | **즉시 수정** | 캐시 키 불일치로 뉴스 검색 실패 |
| 2 | High | 보류 | 기존 이슈, 인증 시스템 미구현 |
| 3 | Medium | 보류 | 장기 과제 (Redis) |
| 4 | Medium | 보류 | 기존 이슈, 독립 기능 |
| 5 | Medium | 보류 | 기존 이슈, 화이트리스트 적용 중 |

---

## 변경 이력
- 2026-02-27: Red Team 코드 리뷰 수행 (Gemini CLI)
