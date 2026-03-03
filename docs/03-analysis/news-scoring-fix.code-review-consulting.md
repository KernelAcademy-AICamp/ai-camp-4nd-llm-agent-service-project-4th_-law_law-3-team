# External Consultant 코드 리뷰 보고서 - News Article Scoring Fix

## 검증 대상
- `backend/app/tools/trend/article_scorer.py`
- `backend/app/tools/trend/collector.py`
- `backend/app/services/service_function/content_marketing_service.py`

## 검증 도구
- Codex CLI (External Consultant, gpt-5.3-codex)

---

## 발견 사항 (6건)

### 1. [High] 정규화 후 빈 키워드 가드 누락 → **즉시 채택, 수정 완료**
- `_calculate_relevance()`에서 HTML 스트리핑 후 빈 문자열이 되면 `"" in title`이 True
- **수정**: `if not keyword_clean: return 0.0` 가드 추가

### 2. [Medium] legal_score 기사별 변별력 부족 → **보류**
- 매칭/비매칭 법령을 모두 related_laws에 포함하여 기사 간 차이 적음
- 사유: 설계 단계에서 인정한 한계. "키워드 컨텍스트 기반 기본 점수" 접근 유지
- 향후: 매칭 법령만 related_laws, 나머지는 후보군으로 분리 검토

### 3. [Medium] Collector private 멤버 강결합 → **보류 (기존 이슈)**
- 서비스 레이어가 `_collector._community_source` 등 private에 직접 접근
- 이번 변경이 아닌 기존 아키텍처 이슈. 별도 리팩토링 이슈로 추적

### 4. [Low-Medium] 캐시 sources_used 하드코딩 → **보류 (기존 이슈)**
- `["tavily", "naver"]` 고정값이 실제 소스와 불일치 가능

### 5. [Low-Medium] RAG 메타데이터 키 계약 불명확 → **보류 (기존 이슈)**
- `metadata["case_name"]`으로 법령명 매핑. `law_name` 우선 fallback 권장

### 6. [Low] 테스트 갭 → **별도 이슈**
- `score_articles`, `_enrich_articles` 회귀 테스트 필요

---

## 채택 요약

| # | 심각도 | 판정 | 사유 |
|---|--------|------|------|
| 1 | High | **즉시 채택** | 오탐 버그 수정 |
| 2 | Medium | 보류 | 설계 인정 한계, v3 로드맵 |
| 3 | Medium | 보류 | 기존 이슈 |
| 4 | Low-Med | 보류 | 기존 이슈 |
| 5 | Low-Med | 보류 | 기존 이슈 |
| 6 | Low | 별도 이슈 | 테스트 작성 |

---

## 변경 이력
- 2026-02-27: External Consultant 코드 리뷰 수행 (Codex CLI)
