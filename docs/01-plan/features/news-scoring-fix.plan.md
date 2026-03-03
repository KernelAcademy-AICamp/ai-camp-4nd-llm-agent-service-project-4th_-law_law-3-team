# News Article Scoring Fix Plan

## 1. 개요

### 1.1 문제 정의
키워드 뉴스 검색 결과의 3개 점수 항목 중 `recency_score`만 정상 작동하고, `relevance_score`와 `legal_score`가 항상 0점으로 산출됨.

### 1.2 영향 범위
- **Frontend**: `KeywordNewsList.tsx` - ScoreBar 3개 중 2개가 항상 0%
- **Backend**: `article_scorer.py`, `collector.py`, `content_marketing_service.py`
- **사용자 경험**: 뉴스 기사 랭킹이 최신성으로만 결정되어 법적 관련도 높은 기사가 매몰됨

---

## 2. 근본 원인 분석 (Root Cause Analysis)

### 2.1 legal_score 항상 0점 (Critical)

**데이터 흐름 추적:**
```
collector.py:search_news_for_keyword()
  → RawTrendItem 수집
  → NewsArticle 변환 (line 367-375)  ← 문제 지점
    → related_laws=[] (기본값)
    → legal_issue_label=None (기본값)
  → score_articles() 호출
    → _calculate_legal(None, []) → 항상 0.0
```

**원인**: `RawTrendItem → NewsArticle` 변환 시 `related_laws`와 `legal_issue_label` 필드를 채우지 않음. 빈 리스트와 None이 그대로 `_calculate_legal()`에 전달되어 0점 반환.

**추가 원인**: `content_marketing_service.py:search_keyword_news()`에서 `_search_related_laws()` 결과는 `KeywordNewsResponse.related_laws` (응답 루트 레벨)에만 저장되고, 개별 `NewsArticle.related_laws`에는 반영되지 않음.

### 2.2 relevance_score 0점 또는 비정상 (High)

**현재 로직 (`article_scorer.py:_calculate_relevance()`):**
```python
title_match = 1.0 if keyword_lower in title_lower else 0.0
snippet_freq = snippet_lower.count(keyword_lower) / 5.0
```

**원인**: 정확한 부분 문자열 매칭(exact substring matching)을 사용하여:
1. 한국어 복합 키워드가 제목/snippet에 정확히 포함되지 않으면 0점
2. 예: 키워드 "부동산 사기" → 제목 "부동산사기" (공백 없음) → 매칭 실패
3. 예: 키워드 "이혼 재산분할" → 제목 "이혼 시 재산 분할" → 매칭 실패
4. 뉴스 API가 `<b>` 등 HTML 태그를 포함한 snippet 반환 → 키워드 매칭 방해

---

## 3. 해결 방안

### 3.1 legal_score 수정 (핵심)

**접근 방식**: 점수 계산 전에 RAG enrichment 결과를 개별 기사에 분배

**Option A (채택 - 경량)**: `search_keyword_news()`에서 RAG 결과를 각 `NewsArticle`에 분배
- `_search_related_laws()` 결과를 응답 루트뿐만 아니라 각 기사의 `related_laws` 필드에도 할당
- 키워드 기반 법률 검색이므로 모든 기사에 동일한 관련 법률 적용 (합리적)
- `legal_issue_label`은 RAG 결과의 `issue_label`에서 파생

**Option B (미채택 - 무거움)**: 기사별 개별 RAG 검색
- 각 기사 제목+snippet으로 개별 벡터 검색
- 정확도 높지만 10건 × RAG 검색 = 지연 시간 증가
- 현재 단계에서는 과도한 복잡성

**구현 계획**:
1. `content_marketing_service.py:search_keyword_news()` 수정
   - RAG 결과(`RelatedLawBrief.law_name`)를 각 `NewsArticle.related_laws`에 할당
   - RAG 결과(`RelatedLawBrief.issue_label`)을 첫 번째 항목으로 `legal_issue_label` 설정
2. 점수 재계산을 위해 `score_articles()` 호출 시점을 RAG enrichment 이후로 이동
   - 현재: `collector.py`에서 점수 계산 → RAG enrichment 결과 반영 불가
   - 변경: `content_marketing_service.py`에서 RAG 결과 반영 후 재계산

### 3.2 relevance_score 개선

**접근 방식**: 공백 정규화 + HTML 태그 제거 + 토큰 기반 매칭 추가

**구현 계획**:
1. `article_scorer.py:_calculate_relevance()` 수정
   - HTML 태그 제거 (`<b>`, `<em>` 등)
   - 공백 정규화 (연속 공백 → 단일 공백)
   - 공백 제거 매칭 추가 (한국어 복합어 대응)
   - 토큰 단위 매칭: 키워드를 공백으로 분리 → 각 토큰이 제목/snippet에 포함되는 비율 계산
2. 가중치 조정
   - 기존: title_match(0.6) + snippet_freq(0.4)
   - 변경: title_match(0.4) + title_token_match(0.2) + snippet_freq(0.2) + snippet_token_match(0.2)

### 3.3 데이터 흐름 재설계

**현재 흐름 (문제)**:
```
collector.search_news_for_keyword()
  → NewsArticle 생성 (related_laws=[], legal_issue_label=None)
  → score_articles() ← legal_score 항상 0

content_marketing_service.search_keyword_news()
  → news_task (위 함수) + law_task (_search_related_laws) 병렬
  → related_laws를 KeywordNewsResponse 루트에만 할당
```

**변경 후 흐름**:
```
collector.search_news_for_keyword()
  → NewsArticle 생성 (점수 미계산)
  → 기사 리스트 반환 (점수 계산 제거)

content_marketing_service.search_keyword_news()
  → news_task + law_task 병렬 실행
  → RAG 결과를 각 NewsArticle에 분배
  → score_articles() 호출 (enriched 기사에 대해 점수 계산)
  → KeywordNewsResponse 구성
```

---

## 4. 변경 파일 목록

| # | 파일 | 변경 내용 | 위험도 |
|---|------|----------|--------|
| 1 | `backend/app/tools/trend/article_scorer.py` | `_calculate_relevance()` 개선: HTML 제거, 공백 정규화, 토큰 매칭 | 중 |
| 2 | `backend/app/tools/trend/collector.py` | `search_news_for_keyword()`에서 `score_articles()` 호출 제거 | 저 |
| 3 | `backend/app/services/service_function/content_marketing_service.py` | RAG 결과 기사 분배 + 점수 재계산 로직 추가 | 고 |
| 4 | (선택) `backend/app/modules/content_marketing/schema/__init__.py` | 스키마 변경 없음 (기존 필드 활용) | - |

---

## 5. 테스트 계획

### 5.1 수동 검증
1. 프론트엔드에서 키워드 수집 → 뉴스 검색 → ScoreBar 3개 모두 0 이상 확인
2. curl로 API 직접 호출하여 JSON 응답의 score 값 확인

### 5.2 엣지 케이스
- RAG enrichment 실패 시: legal_score=0 유지 (graceful degradation)
- 빈 키워드: relevance_score=0 유지
- HTML 태그 포함 snippet: 태그 제거 후 정상 매칭
- 공백 포함/미포함 키워드: 정규화 후 매칭

---

## 6. 롤백 계획
- `collector.py`의 `score_articles()` 호출을 원래 위치로 복원
- `content_marketing_service.py`의 enrichment 로직 제거
- `article_scorer.py`의 `_calculate_relevance()` 원복

---

## 7. 외부 검증 피드백 반영 사항

### 7.1 Red Team (Gemini CLI) 피드백 - 채택/보류

| # | 피드백 | 판정 | 사유 |
|---|--------|------|------|
| 1 | 랭킹 동질성 문제 (기사 간 legal_score 변별력) | **부분 채택** | 기사별 title/snippet과 법령명 간 단순 키워드 겹침 비율로 가산점 추가. Jaccard 유사도는 현 단계에서 과도 |
| 2 | Rate Limiter Redis 전환 | **보류** | 현재 단일 워커 운영. 멀티 워커 전환 시 별도 이슈로 처리 |
| 3 | 서비스 결합도 완화 | **채택** | `_enrich_articles()` 헬퍼 함수로 enrichment 로직 분리 |
| 4 | Semantic Relevance (임베딩 유사도) | **보류** | LLM 호출 없는 경량 스코어링 원칙 유지. 향후 v3 로드맵 |
| 5 | RAG 결과 별도 캐싱 | **채택** | 키워드별 법률 RAG 결과를 TTL 캐시로 관리 |
| 6 | 서킷 브레이커 | **보류** | 현 단계에서 graceful degradation으로 충분 |

### 7.2 External Consultant (Codex CLI) 피드백 - 채택/보류

| # | 피드백 | 판정 | 사유 |
|---|--------|------|------|
| 1 | 기사 단위 Enrichment 계약 확정 | **채택** | article-level fan-out 필수화 |
| 2 | 파이프라인 4단계 분리 | **부분 채택** | 현 구조 내에서 enrich/score 단계 분리 적용 |
| 3 | Legal score 변별력 (기사별 confidence) | **채택** | 기사 title과 법령명의 키워드 겹침 비율로 기사별 법률 점수 차등화 |
| 4 | Relevance exact+token+phrase proximity | **부분 채택** | exact + token 매칭 적용. phrase proximity는 보류 |
| 5 | 회귀 방지 가드레일 | **채택** | 점수 산출 후 전체 0점 경고 로깅 추가 |
| 6 | UX 정렬 옵션 다변화 | **보류** | 프론트엔드 별도 이슈로 추적 |

### 7.3 수정된 구현 계획 요약

1. **collector.py**: `score_articles()` 호출 제거, raw 기사 리스트만 반환
2. **content_marketing_service.py**:
   - `_enrich_articles()` 헬퍼 함수 신설
   - RAG 결과를 기사별로 분배 (법령명 키워드 겹침 비율로 기사별 차등화)
   - `legal_issue_label` 설정 (RAG issue_label 기반)
   - enrichment 후 `score_articles()` 호출
   - RAG 결과 TTL 캐시 추가
3. **article_scorer.py**:
   - HTML 태그 제거 + 공백 정규화
   - 공백 제거 매칭 + 토큰 매칭 추가
   - 가중치: exact(0.3) + no_space(0.2) + token(0.3) + snippet_freq(0.2)
   - 전체 0점 경고 로깅

---

## 변경 이력
- 2026-02-27: 초안 작성 (Agent Team 분석 기반)
- 2026-02-27: Red Team + External Consultant 피드백 반영 (v2)
