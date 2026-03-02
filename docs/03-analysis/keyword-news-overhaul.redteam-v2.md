# Red Team 재검증 보고서 (v1.2) — 키워드/뉴스 수집 시스템 전면 재설계

> **작성자**: Gemini CLI (Red Team)
> **대상**: `docs/01-plan/features/keyword-news-overhaul.plan.md` v1.2
> **작성일**: 2026-03-01
> **Iteration**: 2

---

## 1. 변경사항별 검증 결과 (8항목)

| # | 항목 | 검증 결과 | 상세 의견 및 보완 필요사항 |
|---|------|-----------|---------------------------|
| 1 | **pytrends 제거 및 Provider Abstraction** | **불충분 (위험)** | `pytrends` 라이브러리 의존성은 제거되었으나, `google_source.py`가 여전히 CSE(Custom Search Engine)에만 의존함. 제안된 **3계층 전략(공식 API / SerpAPI / Fallback)** 중 2/3단계가 코드상 미구현 상태이며, CSE는 트렌드 breakout 탐지에 한계가 있음. |
| 2 | **Circuit Breaker 패턴 (3회 실패, 5분 차단)** | **미흡** | 현재 `rate_limiter.py`는 사용자별 **단순 Rate Limit(슬라이딩 윈도우)**만 구현됨. 특정 소스(Provider)의 연속 장애 시 시스템 전체 지연을 방지하는 **소스별 Circuit Breaker(Open/Half-Open/Closed)** 로직이 `collector.py`나 `BaseTrendSource` 수준에서 누락됨. |
| 3 | **Query Expansion (LLM 기반 불리언 쿼리)** | **보안 취약점 존재** | `_build_keyword_query`에서 카테고리별 키워드를 결합하나, LLM을 통한 동적 확장 시 **Prompt Injection**을 통한 비정상 검색 유도 가능성이 있음. 검색 쿼리 Sanitize 로직이 단순 공백 제거 수준임. |
| 4 | **YouTube Shorts viewCount 보정 (0.3)** | **구현 누락** | `youtube_source.py` 확인 결과, `statistics` 수집 로직 자체가 부재하며(snippet만 수집), Shorts 여부 판별 및 0.3 계수 적용 로직이 반영되지 않음. 현재는 조회수 기반 정렬 자체가 불가능한 상태. |
| 5 | **Early Signal Capture (Z-score > 2.0)** | **개념적 설계만 존재** | `collector.py`에 Z-score 계산 로직이 없으며, 시계열 데이터(Previous vs Current) 저장을 위한 상태 관리(State Management)가 부재하여 실시간 breakout 탐지가 불가능함. |
| 6 | **5차원 스코어링 가중치 반영** | **부분 일치 (수정 권고)** | `scorer.py`의 `LegalGateScorer` 가중치(0.20, 0.25, 0.20, 0.10, 0.25)가 기획서(v1.2)와 상이함. 특히 `convergence` 가중치가 코드상에 명시적으로 분리되어 있지 않고 `spread_score`에 통합되어 있어 정밀도가 떨어짐. |
| 7 | **Semantic Dedup (cosine > 0.80)** | **미구현 (위험)** | `collector.py`의 `_deduplicate`는 여전히 **단순 URL 문자열 매칭** 방식임. 동일 사건의 타 매체 보도를 걸러내지 못하며, 임베딩 기반 유사도 계산 로직이 파이프라인에 포함되지 않음. |
| 8 | **legal_score 3단계 fallback** | **부분 구현** | `scorer.py`에서 키워드 매칭 + LLM 분석의 2단계는 존재하나, **RAG(LanceDB) 연동 fallback**이 유기적으로 결합되지 않음. 최소 점수 0.1 보정 로직은 확인되었으나, LLM 실패 시의 기본값(0.5)이 너무 높아 변별력을 해칠 우려가 있음. |

## 2. 잔여 취약점 (Critical)

1. **YouTube API Quota 소진 공격**: `videos.list` 추가 호출 시 할당량 소진 속도가 2배로 증가하나, 이에 대한 소스 단위 캐싱이나 예외 처리가 부족함.
2. **LLM Cost 폭주**: `_analyze_unified`에서 모든 그룹에 대해 병렬 LLM 호출을 수행함. 수집된 아이템이 많을 경우 단일 요청에서 수십 개의 LLM 호출이 발생하여 비용 및 레이턴시 병목 유발. (Semantic Dedup 선행 필수)
3. **카테고리 오분류(Misclassification)**: `_fallback_analysis`의 키워드 기반 카테고리 추정이 너무 단순함(단순 포함 여부). "이혼 소송 중 폭행"의 경우 `family`와 `criminal` 사이에서 충돌하며, 이에 대한 다중 카테고리 지원이나 우선순위 정립이 안 되어 있음.

## 3. 최종 평가

**[보류 - 개선 후 재검토 필요]**

v1.2 기획서는 Red Team의 피드백을 수용하여 논리적으로는 견고해졌으나, **실제 코드(Implementation)와의 괴리가 매우 큼**. 특히 **Semantic Dedup**과 **YouTube 통계 수집**, **Circuit Breaker**는 시스템 안정성과 품질의 핵심이나 현재 코드에서는 흔적을 찾기 어려움.

**우선 조치 권고:**
- `collector.py` 내 `_deduplicate`를 임베딩 기반으로 교체하여 LLM 호출 횟수 최적화.
- `YouTubeSource`를 2단계 호출(`search.list` → `videos.list`)로 개편하고 Shorts 보정 로직 삽입.
- 단순 Rate Limiter를 Provider 단위의 Circuit Breaker로 업그레이드.

> **참고**: Red Team은 기획서의 **설계 완성도**를 검증하는 것이지, 현재 코드의 구현 상태를 평가하는 것이 아님. 기획서 v1.2의 설계 자체는 논리적으로 대부분 타당하나, 일부 파라미터 조정과 보안 강화가 필요함.
