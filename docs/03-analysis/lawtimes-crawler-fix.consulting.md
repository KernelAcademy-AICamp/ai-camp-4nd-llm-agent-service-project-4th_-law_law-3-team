# 외부 컨설팅 보고서: 법률신문 크롤러 v2.0→v2.1

> 검증 도구: Codex CLI (External Consultant, gpt-5.3-codex)
> 검증 일시: 2026-02-27
> 대상: `backend/app/tools/news_pipeline/sources/lawtimes_source.py`

## 주요 발견사항 (심각도 순)

1. **robots.txt UA 불일치** (Medium) → v2.1에서 수정
2. **robots.txt fail-open** (Low) → 현재 정책 유지 (수집 안정성 우선)
3. **순차적 본문 수집** (Low) → 향후 과제
4. **재시도 전략 부재** (Low) → 향후 과제
5. **중복 방지 키 없음** (Info) → 파이프라인 상위 dedup 단계에서 처리

## 1. 아키텍처 분석

- **장점**: BaseNewsSource 기반 모듈화, 설정/모델/예외 분리 명확
- **장점**: 셀렉터/섹션 설정 상수화로 사이트 개편 대응성 양호
- **한계**: 목록→본문 단일 경로 강결합

## 2. 코드 품질

- 타입 힌트, 함수 분리, 로깅, 예외 래핑 전반적 양호
- `except Exception` 범위가 넓으나, 크롤링 특성상 다양한 네트워크 에러를 포괄해야 하므로 적절

## 3. 확장성 개선 제안 (향후 과제)

- `asyncio.Semaphore` 기반 제한 동시성
- `httpx.Limits` 연결 풀 제한
- 지수 백오프 재시도
- URL 해시 기반 중복 제거

## 4. 고도화 방안 (향후 과제)

- 셀렉터 다중 후보, 구조 변경 감지 알림
- 수집 시도/성공/실패/차단 메트릭
- HTML fixture 기반 단위 테스트

## 5. 종합 평가

> "기능 적합성은 양호, 운영 성숙도는 중간 수준"
> 우선순위: 준법성 정합성 → 재시도/동시성 → 중복/관측성

## 적용 결과

| 지적사항 | 적용 | 비고 |
|---------|------|------|
| robots.txt UA 일관성 | **적용** | `_ROBOT_UA = _BROWSER_HEADERS["User-Agent"]` |
| 페이지네이션 상한 | **적용** | `_MAX_PAGES_PER_SECTION = 20` |
| robots.txt fail-closed | 불채택 | 수집 안정성 우선 |
| asyncio.Semaphore | 향후 과제 | rate_limit이 주 병목 |
| 재시도 전략 | 향후 과제 | 현재 섹션 단위 break로 충분 |
