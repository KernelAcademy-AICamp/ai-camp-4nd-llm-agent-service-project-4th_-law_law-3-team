# 콘텐츠 마케팅 검색 소스 미작동 분석 보고서

## 1. 문제 상황

키워드 검색과 트렌드 검색 시 **Tavily, 네이버, Perplexity만 작동**하고,
YouTube API, Newsdata.io, NewsAPI.org, Google 뉴스(CSE)는 API 키를 설정했음에도 검색에 활용되지 않음.

## 2. 코드 분석 결과

### 2.1 아키텍처 개요

```
[키워드 수집 Flow]
  collect_community_keywords() → Tavily + Naver만 사용 (하드코딩)

[뉴스 검색 Flow]
  search_news_for_keyword() → 7개 소스 전체 사용 (API 키 기반 자동 활성화)

[트렌드 대시보드 Flow]
  collect() → Stage 1: Tavily, Stage 3: 6개 뉴스 소스 전체
```

### 2.2 근본 원인 (4가지)

#### 원인 1: 키워드 수집에서 4개 소스 미사용 (설계 제한)

`collector.py:collect_community_keywords()` (line 243-302):
- **Tavily + Naver만** 병렬 수집하도록 하드코딩
- YouTube, Google CSE, NewsData.io, NewsAPI.org는 이 플로우에서 완전히 제외
- 설계 의도: "커뮤니티 인기글 수집"이므로 뉴스 소스 불필요하다는 판단
- **문제**: 키워드 수집의 다양성과 품질이 제한됨

```python
# collector.py line 272-285 (현재 코드)
tasks = [self._community_source.safe_fetch(...)]  # Tavily만
if naver_source:
    tasks.append(naver_source.safe_fetch(...))      # Naver만
# YouTube, Google, NewsData, NewsAPI → 아예 호출하지 않음
```

#### 원인 2: safe_fetch() 무음 실패 (Graceful Degradation 과도)

`sources/__init__.py:safe_fetch()` (line 60-70):
- 모든 예외를 잡아서 빈 리스트 반환
- 로그 수준이 `WARNING`이라 운영 환경에서 보이지 않을 수 있음
- 사용자에게 어떤 소스가 실패했는지 전혀 노출되지 않음

```python
async def safe_fetch(self, query, config):
    try:
        return await self.fetch(query, config)
    except Exception:
        logger.warning("소스 %s 수집 실패, 건너뜀", self.name.value)
        return []  # ← 실패 원인 불명
```

#### 원인 3: sources_used 하드코딩 (캐시 히트 시)

`content_marketing_service.py` line 310, 369:
- 캐시 히트 시 `sources_used=["tavily", "naver"]` 하드코딩
- 실제 사용된 소스와 무관하게 항상 이 두 개만 표시

```python
# line 310 (캐시 히트 시)
sources_used=["tavily", "naver"],  # ← 하드코딩!
```

#### 원인 4: 뉴스 검색에서도 4개 소스가 결과를 못 반환할 가능성

`search_news_for_keyword()` line 377-381:
- `sources_used`는 **실제 결과를 반환한 소스만** 포함
- 4개 소스가 에러/타임아웃/빈 결과이면 `sources_used`에 표시되지 않음
- `safe_fetch()`가 에러를 삼키므로 원인 파악 불가

### 2.3 소스별 잠재적 실패 원인

| 소스 | 잠재적 실패 원인 |
|------|----------------|
| YouTube API | 일일 할당량 초과 (10,000 units/day), 한국어 검색어 인코딩 이슈 |
| Google CSE | 일일 100건 제한 (무료), CSE ID 미설정 또는 설정 오류 |
| NewsData.io | 무료 플랜 200건/day, `timeframe` 48h 제한, 한국어 결과 부족 |
| NewsAPI.org | 무료 플랜에서 `/v2/everything` 접근 제한 (100건/day), 1개월 이전 데이터 접근 불가 |

### 2.4 프론트엔드 UI 이슈

- **소스 선택 UI 없음**: 사용자가 어떤 소스를 사용할지 선택 불가
- **소스 상태 표시 없음**: 어떤 소스가 활성/비활성/실패인지 표시 불가
- `sourcesUsed`는 읽기 전용 텍스트로만 표시 (결과가 있는 소스만)

## 3. 수정 방향 (초안)

### 3.1 키워드 수집 플로우 확장
- `collect_community_keywords()`에 뉴스 소스도 병렬 수집 추가
- 또는 별도 `collect_keywords_with_news()` 메서드 신설

### 3.2 에러 가시성 개선
- `safe_fetch()` 반환을 `(items, error_info)` 튜플로 변경
- 실패한 소스 정보를 API 응답에 포함 (`sources_failed` 필드)

### 3.3 sources_used 정확성 보장
- 캐시 히트 시에도 원래 `sources_used` 보존
- 하드코딩 제거

### 3.4 프론트엔드 소스 상태 표시
- 소스별 활성/비활성/실패 뱃지 UI 추가
- 소스 선택 토글 (선택적)

## 4. 에이전트 팀 검토 필요 사항

- [ ] 키워드 수집에 뉴스 소스를 포함하는 것이 적절한가?
- [ ] safe_fetch() 에러 처리 개선 방안
- [ ] sources_used 정합성 보장 전략
- [ ] 프론트엔드 소스 상태 UI 설계
- [ ] API 키 검증 및 소스별 health check 메커니즘
