# Keyword Time Range Selection Plan

## 1. 개요

### 1.1 요구사항
키워드 탐색 시 시간 범위를 사용자가 선택할 수 있는 기능 추가:
- **48시간 이내** (기본값)
- **일주일 이내**
- **2주일 이내**
- **한달 이내**

### 1.2 현재 상태 분석

**핵심 문제**: 시간 범위가 모든 레이어에서 하드코딩됨

| 레이어 | 현황 | 문제 |
|--------|------|------|
| 프론트엔드 UI | 시간 범위 선택 UI 없음 | 사용자 선택 불가 |
| 프론트엔드 API 호출 | `max_keywords`만 전송 | `time_range` 미전송 |
| 백엔드 엔드포인트 | `max_keywords`만 수신 | `time_range` 미수신 |
| 백엔드 스키마 | `TimeRange` enum 정의됨 (48h, 7d, 30d) | 14d 누락 |
| 서비스 레이어 | `request.time_range` 무시 | 하드코딩 `"48h"` 사용 |
| Collector | `time_range` 파라미터 없음 | 하드코딩 `"48h"` 사용 |
| 캐시 키 | `f"{user_id}:keywords"` | 시간 범위별 캐시 미분리 |

### 1.3 영향 범위

- **Frontend**: `KeywordCollector.tsx`, `useKeywordFlow.ts`, `types/index.ts`
- **Backend**: `schema/__init__.py`, `router/__init__.py`, `content_marketing_service.py`, `collector.py`

---

## 2. 설계

### 2.1 TimeRange Enum 확장

```python
# 현재 (3개)
class TimeRange(str, Enum):
    HOURS_48 = "48h"
    DAYS_7 = "7d"
    DAYS_30 = "30d"

# 변경 (4개) - 14d 추가
class TimeRange(str, Enum):
    HOURS_48 = "48h"
    DAYS_7 = "7d"
    DAYS_14 = "14d"   # NEW
    DAYS_30 = "30d"
```

### 2.2 Frontend UI 설계

`KeywordCollector.tsx`에 시간 범위 선택 드롭다운 추가:

```
┌─────────────────────────────────────────────────────┐
│  키워드 탐색                                         │
│  커뮤니티 트렌드에서 법률 콘텐츠 키워드를 발견하세요     │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │ 48시간 이내 ▼ │  │  키워드 수집  │                  │
│  └──────────────┘  └──────────────┘                  │
│                                                      │
│  기간 옵션:                                          │
│  - 48시간 이내 (기본)                                │
│  - 일주일 이내                                       │
│  - 2주일 이내                                        │
│  - 한달 이내                                         │
└─────────────────────────────────────────────────────┘
```

### 2.3 데이터 흐름 (변경 후)

```
Frontend:
  KeywordCollector.tsx
    └─ timeRange 상태 (useState)
    └─ <select> 시간 범위 선택 UI
    └─ handleCollect(timeRange)
        └─ streamKeywordCollect(maxKeywords, timeRange)
            └─ /api/content-marketing/keywords/collect/stream
                ?max_keywords=10&time_range=7d

Backend:
  router/__init__.py
    └─ collect_keywords_stream_endpoint(max_keywords, time_range)
        └─ KeywordCollectRequest(time_range=time_range, ...)

  content_marketing_service.py
    └─ collect_keywords_stream(request, user_id)
        ├─ 캐시 키: f"{user_id}:{request.time_range.value}:keywords"
        ├─ SourceConfig(time_range=request.time_range.value)
        └─ _collector.collect_community_keywords(time_range=...)

  collector.py
    └─ collect_community_keywords(time_range="7d", ...)
        └─ SourceConfig(time_range=time_range)
```

### 2.4 캐시 전략

시간 범위별 캐시 분리:
- **캐시 키**: `f"{user_id}:{time_range}:keywords"` (시간 범위 포함)
- **캐시 TTL**: 기존 `KEYWORD_COLLECT_CACHE_TTL` 유지
- 시간 범위가 변경되면 별도 캐시로 관리 (기존 캐시는 유지)

---

## 3. 변경 파일 목록

| # | 파일 | 변경 내용 | 위험도 |
|---|------|----------|--------|
| 1 | `backend/app/modules/content_marketing/schema/__init__.py` | `TimeRange` enum에 `DAYS_14` 추가 | 저 |
| 2 | `backend/app/modules/content_marketing/router/__init__.py` | 엔드포인트에 `time_range` 쿼리 파라미터 추가 | 저 |
| 3 | `backend/app/services/service_function/content_marketing_service.py` | 하드코딩 제거, `request.time_range` 사용, 캐시 키에 시간 범위 포함 | 중 |
| 4 | `backend/app/tools/trend/collector.py` | `collect_community_keywords()`에 `time_range` 파라미터 추가 | 저 |
| 5 | `frontend/src/features/content-marketing/types/index.ts` | `TimeRange` 타입에 `'14d'` 추가 | 저 |
| 6 | `frontend/src/features/content-marketing/components/KeywordCollector.tsx` | 시간 범위 선택 드롭다운 UI 추가 | 중 |
| 7 | `frontend/src/features/content-marketing/hooks/useKeywordFlow.ts` | `handleCollect(timeRange)` 파라미터 추가, API 호출에 전달 | 저 |

---

## 4. 구현 순서

1. **Backend Schema**: `TimeRange.DAYS_14` 추가
2. **Backend Collector**: `collect_community_keywords(time_range=...)` 파라미터 추가
3. **Backend Service**: 하드코딩 제거 + 캐시 키 변경
4. **Backend Router**: 엔드포인트에 `time_range` 쿼리 파라미터 추가
5. **Frontend Types**: `TimeRange` 타입 업데이트
6. **Frontend Hook**: `handleCollect` 시간 범위 전달
7. **Frontend UI**: 드롭다운 컴포넌트 추가

---

## 5. 엣지 케이스

- 기존 캐시: 시간 범위 없는 기존 캐시 키(`user:keywords`)는 자연 만료로 정리
- 잘못된 time_range 값: FastAPI 자동 검증 (Enum 타입이므로 유효하지 않은 값은 422 에러)
- 긴 시간 범위(30d)의 API 응답 시간: Tavily/Naver API 자체 타임아웃으로 제한
- 초기화 버튼: 시간 범위 변경 시 기존 결과 초기화 여부 → 변경 시 자동 초기화

---

## 6. 테스트 계획

1. 프론트엔드에서 각 시간 범위 선택 후 키워드 수집 → 결과 확인
2. 시간 범위 변경 시 캐시가 분리되는지 확인 (같은 범위 재요청 시 캐시 히트)
3. curl로 API 직접 호출하여 `time_range` 파라미터 정상 전달 확인

---

## 7. 외부 검증 피드백 통합

### 7.1 Red Team (Gemini CLI) 피드백

| # | 지적 | 판정 | 사유 |
|---|------|------|------|
| 1 | 30d 범위 반복 요청 시 API 비용 폭증/DoS 위험 | **인지** | 기존 Rate Limiter로 대응, 향후 범위별 차등 제한 검토 |
| 2 | 캐시 파편화 (user × time_range) | **인지** | 캐시 키에 time_range 포함으로 분리 관리 |
| 3 | UX: 시간 범위별 예상 소요시간 안내 | **채택** | 드롭다운 옵션에 힌트 텍스트 추가 |
| 4 | UX: localStorage로 마지막 선택값 유지 | **채택** | 사용자 편의성 향상 |
| 5 | 전역 트렌드 캐시, 비동기 워커, Pre-fetching | **보류** | 현 단계 과도한 복잡성 |

### 7.2 External Consultant (Codex CLI) 피드백

| # | 지적 | 판정 | 사유 |
|---|------|------|------|
| P0 | 파라미터 전달 일관성 + 캐시 키 정교화 | **채택** | 이번 구현의 핵심 |
| P1 | 관측성 지표 + UX 미세개선 | **부분 채택** | UX 힌트만 반영 |
| P2 | Redis/작업큐 전환 | **보류** | 장기 과제 |
| P3 | A/B 테스트/개인화 | **보류** | 장기 과제 |

### 7.3 반영 항목 요약

1. **드롭다운 UX 힌트**: 각 시간 범위 옵션에 속도/범위 힌트 표시 (예: "빠름", "폭넓음")
2. **localStorage 선택값 유지**: 마지막 선택한 시간 범위를 localStorage에 저장하여 재방문 시 복원
3. **캐시 키 정교화**: `f"{user_id}:{time_range}:keywords"` 형식으로 시간 범위별 캐시 분리

---

## 변경 이력
- 2026-02-27: Agent Team 분석 기반 초안 작성
- 2026-02-27: Red Team + External Consultant 피드백 통합 (Section 7)
