# Keyword Time Range Selection - 완료 보고서

> **완료 일자**: 2026-02-27
> **팀 구성**: Agent Team 7명 + Gemini CLI (Red Team) + Codex CLI (External Consultant)
> **대상**: 콘텐츠 마케팅 → 키워드 탐색 시간 범위 선택 기능

---

## 1. 요구사항 및 결과 요약

### 1.1 요구사항

키워드 탐색 시 수집 기간을 사용자가 선택할 수 있는 기능:
- 48시간 이내 (기본값)
- 일주일 이내
- 2주일 이내
- 한달 이내

### 1.2 결과

| 항목 | 기대 | 결과 |
|------|------|------|
| 시간 범위 드롭다운 UI | 4개 옵션 | ✅ 4개 옵션 (힌트 텍스트 포함) |
| Backend time_range 파라미터 | API 쿼리 파라미터 | ✅ `?time_range=7d` |
| 무효값 검증 | 422 에러 | ✅ FastAPI Enum 자동 검증 |
| 캐시 분리 | 시간 범위별 독립 캐시 | ✅ `{user_id}:{time_range}:keywords` |
| localStorage 유지 | 재방문 시 선택값 복원 | ✅ `keyword-time-range` 키 |
| E2E 전체 흐름 | 선택 → 수집 → 뉴스 검색 | ✅ 검증 완료 |

---

## 2. PDCA 실행 이력

### Phase 1: Plan (기획)

| 단계 | 내용 | 결과 |
|------|------|------|
| Agent Team 분석 | 7명이 현황 분석 및 변경 계획 수립 | 7개 파일, 4개 레이어 변경 식별 |
| Red Team 검증 | Gemini CLI → DoS 위험, 캐시 파편화, UX 개선 제안 | 힌트 텍스트 + localStorage 채택 |
| Consultant 검증 | Codex CLI → 파라미터 일관성, 캐시 키 정교화, Redis 전환 제안 | P0/P1 부분 채택, P2/P3 보류 |
| 계획 확정 | 8개 파일 변경, 외부 피드백 3건 채택 | `keyword-time-range.plan.md` |

### Phase 2: Do (구현)

| 파일 | 변경 내용 |
|------|----------|
| `backend/.../schema/__init__.py` | `TimeRange.DAYS_14 = "14d"` 추가 |
| `backend/.../router/__init__.py` | `time_range: TimeRange` 쿼리 파라미터 추가 |
| `backend/.../content_marketing_service.py` | 캐시 키 정교화, 하드코딩 제거, 뉴스 검색 캐시 순회 |
| `backend/.../collector.py` | `collect_community_keywords(time_range=...)` 파라미터 추가 |
| `frontend/.../types/index.ts` | `TimeRange` union에 `'14d'` 추가 |
| `frontend/.../services/index.ts` | `streamKeywordCollect(timeRange)` 파라미터 추가 |
| `frontend/.../hooks/useKeywordFlow.ts` | `timeRange` 상태 + localStorage 영속화 |
| `frontend/.../components/KeywordCollector.tsx` | 시간 범위 드롭다운 UI (힌트 텍스트 포함) |

### Phase 3: Check (검증)

#### 정적 검증

| 도구 | 결과 |
|------|------|
| `ruff check` | ✅ All checks passed |
| `mypy` | ✅ Success: no issues found |
| `npm run build` | ✅ Compiled successfully |

#### 코드 리뷰 (3중 검증)

| 검증자 | 발견 사항 | 조치 |
|--------|----------|------|
| Agent Team | 보안 스캔 SAFE | 통과 |
| **Red Team (Gemini CLI)** | **Critical: 캐시 키 불일치** — `search_keyword_news`가 기존 키 사용 | **즉시 수정** |
| Red Team | High: 하드코딩 TEMP_USER_ID | 보류 (기존 이슈) |
| Red Team | Medium: 인메모리 캐시, 뉴스 7d 하드코딩, SSRF | 보류 (기존 이슈) |
| External Consultant (Codex CLI) | 아키텍처/타입/확장성 양호 | **승인 (Approved)** |

#### 캐시 키 불일치 버그 수정

**문제**: `collect_keywords`/`collect_keywords_stream` → 캐시 키 `f"{user_id}:{time_range}:keywords"` 사용
`search_keyword_news` → 기존 `f"{user_id}:keywords"`로 조회 → 항상 KeywordNotFoundError

**수정**: `search_keyword_news`에서 모든 `TimeRange` enum 값을 순회하며 keyword_id 탐색

```python
for tr in TimeRangeEnum:
    cache_key = f"{user_id}:{tr.value}:keywords"
    cached = _keyword_cache.get(cache_key)
    # ... TTL 검증 후 keyword_id 매칭
```

#### E2E 런타임 검증

| 테스트 | 결과 |
|--------|------|
| API `?time_range=7d` → SSE 스트리밍 | ✅ 정상 시작 |
| API `?time_range=invalid` → 422 에러 | ✅ Enum 검증 작동 |
| UI 드롭다운 4개 옵션 표시 | ✅ 힌트 텍스트 포함 |
| UI "일주일 이내" 선택 → 키워드 수집 | ✅ 4개 키워드 수집 완료 |
| 키워드 → 뉴스 검색 (캐시 키 수정 검증) | ✅ 10건 뉴스 정상 로드 |

### Phase 4: Act (개선)

| 항목 | 상태 | 비고 |
|------|------|------|
| Red Team Critical 버그 수정 | ✅ 완료 | 캐시 순회 방식으로 해결 |
| Re-verification (ruff + mypy) | ✅ 통과 | 수정 후 재검증 완료 |

---

## 3. 보고서 목록

| 보고서 | 경로 |
|--------|------|
| 기획서 | `docs/01-plan/features/keyword-time-range.plan.md` |
| Red Team 기획 검증 | `docs/03-analysis/keyword-time-range.redteam.md` |
| Consultant 기획 검증 | `docs/03-analysis/keyword-time-range.consulting.md` |
| Red Team 코드 리뷰 | `docs/03-analysis/keyword-time-range.code-review-redteam.md` |
| Consultant 코드 리뷰 | `docs/03-analysis/keyword-time-range.code-review-consulting.md` |
| **완료 보고서** | `docs/04-report/keyword-time-range.report.md` |

---

## 4. 향후 과제 (보류 항목)

| # | 항목 | 출처 | 우선순위 |
|---|------|------|---------|
| 1 | Redis 캐시 전환 (분산 환경 대응) | Red Team + Consultant | 장기 |
| 2 | 인증 시스템 구현 (TEMP_USER_ID 교체) | Red Team | 중기 |
| 3 | 30d 범위에 대한 Rate Limiting 강화 | Red Team | 중기 |
| 4 | 캐시 TTL을 time_range별 차등 적용 | Consultant | 단기 |
| 5 | `search_news_for_keyword` time_range 파라미터화 | Consultant | 단기 |

---

## 변경 이력
- 2026-02-27: PDCA 완료 보고서 작성
