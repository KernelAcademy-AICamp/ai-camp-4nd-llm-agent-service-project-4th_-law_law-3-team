# Red Team 검증 보고서: 법률신문 크롤러 v2.0→v2.1

> 검증 도구: Gemini CLI (Red Team)
> 검증 일시: 2026-02-27
> 대상: `backend/app/tools/news_pipeline/sources/lawtimes_source.py`

## 1. 취약점 (Critical/High/Medium/Low)

### [Medium] SSRF 방어 TOCTOU 위험
- `validate_url()`과 실제 `httpx.get()` 사이 DNS Rebinding 가능성
- **팀 판단**: 불채택 - 고정 도메인(lawtimes.co.kr)만 사용, SSRF 가드 존재

### [Low] 하드코딩 브라우저 헤더
- 고정 UA가 블랙리스트 등록 시 일괄 차단
- **팀 판단**: 불채택 - 현재 잘 동작 중, fake-useragent 과도한 복잡성

### [Low] robots.txt SSRF 보호 부재
- `_check_robots_txt()` 내 `validate_url` 미호출
- **팀 판단**: 수용 - BASE_URL 상수 고정이므로 실질 위험 낮음

## 2. 크롤링 안정성

- HTML 구조 의존성: ND소프트 CMS 전용 셀렉터 사용 → 개편 시 즉시 감지 필요
- **페이지네이션 종료 조건**: `_MAX_PAGES_PER_SECTION = 20` 추가하여 해결 (v2.1)
- 예외 처리: `SourceFetchError` 래핑 적절

## 3. 성능 최적화

- **순차적 본문 수집**: `asyncio.gather` + `Semaphore` 제안
- **팀 판단**: 향후 과제 - rate_limit 2초가 주 병목, 서버 부하 고려

## 4. 윤리적 크롤링

- robots.txt 준수: RobotFileParser 사용 (우수)
- **UA 일관성**: v2.1에서 수정 - 검사/실제 동일 UA 사용
- Rate Limit: 2.0초 적용 (양호)

## 5. 코드 품질

- `dateparser` 도입 제안 → 불채택 (현재 4개 포맷 충분)
- 로그 상세화 제안 → 기존 로그로 충분 (URL 포함)

## 적용 결과

| 지적사항 | 적용 | 비고 |
|---------|------|------|
| 페이지네이션 상한 | **적용** | `_MAX_PAGES_PER_SECTION = 20` |
| robots.txt UA 일관성 | **적용** | `_ROBOT_UA = _BROWSER_HEADERS["User-Agent"]` |
| SSRF TOCTOU | 불채택 | 고정 도메인만 사용 |
| asyncio.gather | 향후 과제 | rate_limit이 주 병목 |
