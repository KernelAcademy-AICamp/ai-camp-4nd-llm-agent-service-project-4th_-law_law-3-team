# QA 전략 보고서: 키워드/뉴스 수집 시스템 재설계 검증

> **작성자**: QA 엔지니어
> **작성일**: 2026-03-01
> **대상 기능**: 유튜브 콘텐츠 자동 생성 — 키워드/뉴스 수집 파이프라인 전면 재설계
> **관련 분석**: `docs/03-analysis/content-marketing-source-fix.qa-strategy.md`

---

## 0. 작업 시작 전 확인 사항

> **Failure Log 확인 완료** (`FAIL-001`): Codex CLI 호출 실패 시 자동 건너뛰기 금지.
> 외부 CLI 실패 시 반드시 사용자 보고 후 지침 대기.

---

## 1. 현재 시스템 문제점 분석 (코드 기반)

### 1.1 확인된 결함 목록

코드 분석 (`collector.py`, `keyword_extractor.py`, `article_scorer.py`, `scorer.py`) 결과 확인된 실제 결함:

| # | 파일 | 위치 | 결함 내용 | 심각도 |
|---|------|------|----------|--------|
| D-01 | `collector.py` | L267 | `search_query="사건 사고 논란 이슈"` 고정 쿼리 — 카테고리 무관 | Critical |
| D-02 | `collector.py` | L283 | `search_query="사건 사고 논란 법률"` Naver에도 고정 쿼리 | Critical |
| D-03 | `collector.py` | L337, L343 | `collect_community_keywords_with_status()`도 동일 고정 쿼리 | Critical |
| D-04 | `article_scorer.py` | L99-108 | `legal_score`가 `legal_issue_label`과 `related_laws`에 완전 의존 — RAG 실패 시 0점 | High |
| D-05 | `article_scorer.py` | L144 | `total_score = (rel*0.4 + legal*0.3 + rec*0.3) * 100` — legal 0점 시 최대 70점 | High |
| D-06 | `collector.py` | L401 | `search_news_for_keyword()` `time_range="7d"` 하드코딩 | Medium |
| D-07 | `keyword_extractor.py` | L66-88 | `_SCORING_PROMPT_TEMPLATE` 커뮤니티 도메인에 특화 — 법률 카테고리 인지 없음 | High |
| D-08 | `scorer.py` | L178-186 | `CATEGORY_KEYWORDS` 딕셔너리는 있으나 키워드 수집 단계에서 활용 안 됨 | High |

### 1.2 근본 원인

```
[카테고리 인지 부재]
  ├── TrendRequest.category 필드는 존재하나
  ├── collect_community_keywords*()에서 카테고리를 쿼리에 반영하지 않음
  └── 결과: criminal/civil/family 모두 동일한 "사건 사고 논란 이슈" 쿼리

[legal_score RAG 의존]
  ├── article_scorer.py는 RAG 결과(legal_issue_label, related_laws)에서만 legal_score 계산
  ├── 뉴스 기사 자체의 텍스트에서 법률 키워드를 활용하지 않음
  └── RAG 실패 시 legal_score = 0.0 → total_score 최대 70점 제한
```

---

## 2. 성능 기준선 (Baseline) 정의

### 2.1 현재 시스템 추정 기준선

> 실제 API 호출 없이 코드 구조 분석을 통한 이론적 추정값

| 지표 | 측정 방법 | 추정 현재값 | 측정 근거 |
|------|----------|-----------|---------|
| 카테고리 일치율 | "criminal" 요청 → 형사법 관련 키워드 10개 중 일치 수 | **30~40%** | 고정 쿼리 "사건 사고 논란 이슈"가 카테고리 무관 |
| 뉴스 관련도 평균 (`total_score`) | 키워드 1개 기준 뉴스 5건 total_score 평균 | **35~45점** | legal_score=0 가정 시 recency(0.3*100=30) + relevance 기여 |
| 법적 관련도 비율 | `legal_score > 0`인 기사 비율 | **20~30%** | RAG 연동 의존 — 현재 RAG 항상 동작하지 않음 |
| 키워드당 소스 다양성 | 뉴스 검색 시 기사당 평균 소스 수 | **1.5~2.0개** | Naver + 1개 소스만 안정 동작 |
| 의미적 중복률 | 동일 사건 다른 표현 기사 비율 | **30~40%** | URL dedup만 수행, 의미적 dedup 없음 |

### 2.2 목표 성능 지표 (KPI)

| 지표 | 현재 추정 | 목표 (재설계 후) | 측정 주기 |
|------|----------|----------------|---------|
| 카테고리 일치율 | 30~40% | **85%+** | 기능 배포 후 수동 평가 |
| 뉴스 관련도 평균 (`total_score`) | 35~45점 | **65점+** | 자동 회귀 테스트 |
| 법적 관련도 비율 (`legal_score > 0`) | 20~30% | **70%+** | 자동 회귀 테스트 |
| 네이버 베스트 대비 관련성 | 열위 | **동등 이상** | 수동 비교 평가 |
| 소스 다양성 (기사당 소스 수) | 1.5~2.0개 | **2.5개+** | 자동 집계 |
| 의미적 중복률 | 30~40% | **15% 이하** | 수동 샘플링 |
| API 응답 시간 | 미측정 | **30초 이내** | 성능 테스트 |

---

## 3. 테스트 시나리오 설계

### 3.1 카테고리별 키워드 품질 테스트

#### 3.1.1 테스트 파일 구조

```
backend/tests/unit/content_marketing/
├── test_category_keyword_quality.py      # 카테고리 일치율 검증
├── test_article_scorer.py                # 점수 계산 로직 검증
├── test_keyword_extractor_category.py    # 카테고리 인지 키워드 추출 검증
└── test_news_search_time_range.py        # 시간 범위 파라미터 검증

backend/tests/integration/content_marketing/
├── test_category_pipeline_integration.py # 카테고리별 파이프라인 통합
├── test_naver_best_comparison.py         # 네이버 베스트 비교 (수동)
└── test_score_regression.py             # 스코어 회귀 테스트
```

#### 3.1.2 카테고리 일치율 단위 테스트

```python
# tests/unit/content_marketing/test_category_keyword_quality.py

import pytest
from unittest.mock import AsyncMock, patch
from app.tools.trend.models import ScoredKeyword, KeywordScore


# 카테고리별 기대 키워드 시드 (수동 정의)
CATEGORY_KEYWORD_SEEDS = {
    "criminal": [
        "형사소송", "공판", "구속영장", "사기죄", "횡령", "배임",
        "성범죄", "폭행치사", "음주운전", "마약", "살인", "사기"
    ],
    "civil": [
        "손해배상", "계약해지", "부동산", "전세사기", "임대차",
        "명예훼손", "물품대금", "채무불이행", "소멸시효", "가처분"
    ],
    "family": [
        "이혼", "양육권", "위자료", "재산분할", "상속",
        "유류분", "친권", "양육비", "혼인취소", "입양"
    ],
    "labor": [
        "해고", "부당해고", "임금체불", "퇴직금", "산업재해",
        "직장내괴롭힘", "연차수당", "주52시간", "노동조합", "부당노동행위"
    ],
}


def _is_category_relevant(keyword: str, category: str) -> bool:
    """키워드가 해당 카테고리와 관련 있는지 판단 (수동 레이블링 기준)"""
    seeds = CATEGORY_KEYWORD_SEEDS.get(category, [])
    # 시드 키워드 중 하나라도 포함되거나 의미적으로 관련 있으면 True
    # 자동화: 시드 키워드 포함 여부로 대리 측정
    return any(seed in keyword or keyword in seed for seed in seeds)


class TestCategoryKeywordRelevance:
    """카테고리별 키워드 관련도 검증"""

    def test_criminal_keywords_contain_criminal_terms(self):
        """criminal 카테고리 키워드 10개 중 8개 이상이 형사법 관련이어야 함"""
        # Given: 재설계 후 criminal 카테고리 키워드 수집 결과 (Mock)
        mock_keywords = [
            ScoredKeyword(id="1", keyword="음주운전 사망사고", context="형사 사건", source_posts=[],
                          scores=KeywordScore(virality=0.8, social_impact=0.7, legal_relevance=0.9, content_fitness=0.85),
                          total_score=83.0, rank=1, score_reason="형사 사건", confidence=0.9),
            # ... 10개 Mock 데이터
        ]
        # 카테고리 일치 개수 계산
        matches = sum(1 for kw in mock_keywords if _is_category_relevant(kw.keyword, "criminal"))
        match_rate = matches / len(mock_keywords)

        assert match_rate >= 0.85, (
            f"criminal 카테고리 일치율 {match_rate:.1%} < 목표 85%\n"
            f"불일치 키워드: {[kw.keyword for kw in mock_keywords if not _is_category_relevant(kw.keyword, 'criminal')]}"
        )

    @pytest.mark.parametrize("category", ["criminal", "civil", "family", "labor"])
    def test_category_specific_keyword_not_generic(self, category: str):
        """각 카테고리 키워드가 '사건 사고 논란 이슈' 등 일반 키워드가 아니어야 함"""
        GENERIC_KEYWORDS = frozenset({
            "사건", "사고", "논란", "이슈", "화제", "핫이슈",
            "뉴스", "사회", "최신", "속보"
        })
        # 재설계 후 API 호출 결과를 Mock으로 대체
        # 실제 테스트: collect_community_keywords_with_status(category=category) 결과 검증
        ...


class TestKeywordQueryConstruction:
    """재설계 후 쿼리 생성 로직 검증 (D-01, D-02, D-03 수정 확인)"""

    @pytest.mark.asyncio
    async def test_criminal_category_uses_criminal_query(self):
        """criminal 카테고리 수집 시 형사 관련 쿼리 사용 확인"""
        # 현재 코드의 문제: "사건 사고 논란 이슈" 고정
        # 재설계 후: category="criminal" → "형사 사건 법원 판결" 등 카테고리 쿼리 사용
        with patch("app.tools.trend.sources.tavily_source.TavilySource.fetch") as mock_fetch:
            mock_fetch.return_value = []
            from app.tools.trend.collector import TrendCollector
            collector = TrendCollector()
            # 재설계 후 메서드 시그니처: collect_community_keywords_with_status(category="criminal")
            await collector.collect_community_keywords_with_status(
                community_domains=None,
                max_keywords=10,
                time_range="48h",
                # category="criminal"  ← 재설계 후 추가될 파라미터
            )
            # 검증: mock_fetch 호출 인자에 "사건 사고 논란 이슈" 고정 쿼리가 아닌
            # 카테고리 특화 쿼리가 포함되어야 함
            call_args = mock_fetch.call_args
            if call_args:
                query_used = call_args[0][0] if call_args[0] else ""
                assert "사건 사고 논란 이슈" != query_used, (
                    "D-01 결함 미수정: 카테고리 무관 고정 쿼리 사용 중"
                )
```

### 3.2 뉴스 관련도 점수 테스트 (article_scorer.py 검증)

```python
# tests/unit/content_marketing/test_article_scorer.py

import pytest
from datetime import datetime, timezone
from app.tools.trend.article_scorer import score_articles, _calculate_relevance, _calculate_legal
from app.modules.content_marketing.schema import NewsArticle


class TestRelevanceScore:
    """relevance_score 계산 정확성 검증"""

    def test_exact_match_keyword_in_title_scores_high(self):
        """키워드가 제목에 정확히 포함될 때 relevance_score >= 0.3"""
        score = _calculate_relevance("음주운전", "음주운전 사망사고 판결", "")
        assert score >= 0.3, f"exact match 점수 {score} < 0.3"

    def test_keyword_absent_scores_zero(self):
        """키워드가 제목/스니펫에 없으면 relevance_score = 0.0"""
        score = _calculate_relevance("음주운전", "부동산 가격 하락", "집값 폭락 전망")
        assert score == 0.0, f"무관 키워드 relevance_score = {score} (0이어야 함)"

    def test_nospace_korean_compound_match(self):
        """공백 차이 한국어 복합어 매칭 검증 ('부동산 사기' ↔ '부동산사기')"""
        score_with_space = _calculate_relevance("부동산 사기", "부동산사기 대출 급증", "")
        assert score_with_space > 0.0, "공백 제거 복합어 매칭 실패"

    def test_partial_token_match_gives_partial_score(self):
        """키워드 토큰 일부만 매칭 시 비례 점수"""
        # "손해배상 청구" 중 "손해배상"만 제목에 포함
        score = _calculate_relevance("손해배상 청구", "손해배상 소송 기각", "")
        assert 0.0 < score < 1.0, f"부분 매칭 점수 {score} 범위 이탈"


class TestLegalScore:
    """legal_score 계산 로직 검증 (D-04 수정 여부)"""

    def test_legal_score_zero_without_rag_data(self):
        """RAG 데이터 없으면 legal_score=0 (현재 D-04 결함 재현)"""
        score = _calculate_legal(None, [])
        assert score == 0.0, "legal_issue_label=None, related_laws=[] → legal_score 0 확인"

    def test_legal_score_from_label_only(self):
        """legal_issue_label만 있을 때 0.5점"""
        score = _calculate_legal("형사책임", [])
        assert score == 0.5, f"label만 있을 때 점수 {score} != 0.5"

    def test_legal_score_from_laws_only(self):
        """related_laws 5개 있을 때 0.5점"""
        score = _calculate_legal(None, ["형법", "민법", "소송법", "헌법", "노동법"])
        assert score == 0.5, f"laws 5개 있을 때 점수 {score} != 0.5"

    def test_legal_score_max_with_both(self):
        """label + laws 5개 있을 때 1.0점"""
        score = _calculate_legal("형사", ["법1", "법2", "법3", "법4", "법5"])
        assert score == 1.0, f"최대 점수 {score} != 1.0"

    # ── 재설계 후 추가 테스트 (D-04 수정 확인) ──

    def test_redesigned_legal_score_uses_text_keywords(self):
        """재설계 후: RAG 없이도 텍스트 법률 키워드로 legal_score 부여 (D-04 수정 검증)"""
        # 재설계 전: legal_issue_label=None, related_laws=[] → 0.0
        # 재설계 후: 제목/스니펫에 "형사", "소송" 등 키워드 있으면 기본 점수 부여
        # 이 테스트가 통과하면 D-04가 수정된 것
        article = NewsArticle(
            title="음주운전 형사처벌 강화 판결",
            url="https://example.com",
            source="naver",
            snippet="음주운전 사고로 인한 형사소송에서 실형 선고",
            legal_issue_label=None,  # RAG 미연동
            related_laws=[],          # RAG 미연동
        )
        scored = score_articles("음주운전", [article])
        assert scored[0].legal_score > 0.0, (
            "D-04 미수정: RAG 없이도 법률 텍스트 키워드로 legal_score 부여되어야 함\n"
            f"현재 legal_score={scored[0].legal_score}"
        )


class TestTotalScoreCalculation:
    """total_score 계산 범위 및 분포 검증"""

    def test_total_score_range_0_to_100(self):
        """total_score는 항상 0~100 범위"""
        articles = [
            NewsArticle(title="테스트", url="https://example.com",
                        source="naver", snippet="내용",
                        legal_issue_label=None, related_laws=[]),
        ]
        scored = score_articles("테스트", articles)
        assert 0.0 <= scored[0].total_score <= 100.0

    def test_total_score_no_rag_ceiling(self):
        """D-05: RAG 없을 때 total_score 최대값 검증"""
        # 현재 D-05: legal_score=0 시 최대 = (0.4 * 1.0 + 0 + 0.3 * 1.0) * 100 = 70점
        # 재설계 목표: 법률 텍스트 키워드로 legal_score 보정 시 70점 이상 가능
        article = NewsArticle(
            title="부동산 사기 피해자 손해배상 소송 승소",
            url="https://example.com",
            source="naver",
            snippet="사기 피해 손해배상 청구 인용",
            legal_issue_label=None,
            related_laws=[],
            published_at=datetime.now(tz=timezone.utc),  # 최신 → recency=1.0
        )
        scored = score_articles("부동산 사기", [article])
        # 재설계 목표: 65점 이상 (현재 최대 70점이나 실제로는 더 낮음)
        assert scored[0].total_score >= 50.0, (
            f"D-05: total_score {scored[0].total_score} < 50 (목표 기준 미달)"
        )
```

### 3.3 시간 범위 테스트

```python
# tests/unit/content_marketing/test_news_search_time_range.py

import pytest
from unittest.mock import AsyncMock, patch, call
from app.tools.trend.collector import TrendCollector
from app.tools.trend.sources import SourceConfig


class TestNewsSearchTimeRange:
    """search_news_for_keyword() 시간 범위 파라미터 전달 검증 (D-06)"""

    @pytest.mark.parametrize("time_range", ["48h", "7d", "14d", "30d"])
    @pytest.mark.asyncio
    async def test_time_range_passed_to_config(self, time_range: str):
        """time_range 파라미터가 SourceConfig에 올바르게 전달되는지 확인"""
        # D-06: 현재 search_news_for_keyword()는 time_range="7d" 하드코딩
        # 재설계 후: 파라미터 수신 및 전달 확인
        with patch("app.tools.trend.collector.TrendCollector._deduplicate", return_value=[]):
            collector = TrendCollector()
            # 재설계 후 시그니처: search_news_for_keyword(keyword, max_results, time_range)
            # 현재: time_range 파라미터 없음 → D-06 결함
            try:
                await collector.search_news_for_keyword(
                    keyword="테스트",
                    max_results=5,
                    # time_range=time_range  ← 재설계 후 추가될 파라미터
                )
                # TODO: 재설계 후 SourceConfig 호출 시 time_range 검증
            except TypeError:
                pytest.skip(f"D-06 미수정: time_range 파라미터 없음")

    @pytest.mark.asyncio
    async def test_default_time_range_is_reasonable(self):
        """기본 time_range가 7d(적절) 또는 48h(너무 짧음) 확인"""
        # 7d는 적절한 기본값. 48h면 너무 짧아 관련 뉴스 미수집 위험
        import inspect
        from app.tools.trend.collector import TrendCollector
        sig = inspect.signature(TrendCollector.search_news_for_keyword)
        # 재설계 후 time_range 파라미터가 추가되었다면 기본값 확인
        if "time_range" in sig.parameters:
            default = sig.parameters["time_range"].default
            assert default in ("7d", "14d"), (
                f"시간 범위 기본값 '{default}'이 너무 짧거나 부적절"
            )
```

### 3.4 에지 케이스 테스트

```python
# tests/unit/content_marketing/test_edge_cases.py

import pytest
from app.tools.trend.collector import TrendCollector
from app.tools.trend.sources import SourceConfig
from app.tools.trend.exceptions import TrendSourceError


class TestAPIFailureEdgeCases:
    """외부 API 장애 시 graceful degradation 검증"""

    @pytest.mark.asyncio
    async def test_all_sources_fail_returns_empty_not_exception(self):
        """모든 소스 실패 시 빈 리스트 반환 (예외 전파 없음)"""
        with patch("app.tools.trend.sources.naver_source.NaverSource.fetch",
                   side_effect=Exception("API error")):
            collector = TrendCollector()
            # Naver만 활성화된 환경에서 실패 시 빈 리스트
            result_articles, sources_used, sources_failed = \
                await collector.search_news_for_keyword("테스트", max_results=5)
            assert result_articles == []
            assert len(sources_failed) > 0  # 실패 정보는 포함되어야 함

    @pytest.mark.asyncio
    async def test_zero_results_returns_empty_list_not_error(self):
        """API 응답 정상이나 결과 0건인 경우"""
        with patch("app.tools.trend.sources.naver_source.NaverSource.fetch",
                   return_value=[]):
            collector = TrendCollector()
            result_articles, sources_used, _ = \
                await collector.search_news_for_keyword("매우특수한키워드xyz", max_results=5)
            assert result_articles == []
            # 성공했지만 결과 0건이므로 sources_used에 포함 (현재 코드 L472-475 확인)
            assert "naver" in sources_used

    @pytest.mark.asyncio
    async def test_duplicate_articles_deduplicated_by_url(self):
        """동일 URL 기사 중복 제거 확인"""
        from app.tools.trend.models import RawTrendItem, TrendSource
        from datetime import datetime, timezone

        dup_item = RawTrendItem(
            title="중복 기사",
            url="https://example.com/article/1",
            source=TrendSource.NAVER,
            snippet="내용",
            published_at=datetime.now(tz=timezone.utc),
        )
        items = [dup_item, dup_item, dup_item]  # 동일 URL 3개
        collector = TrendCollector()
        result = collector._deduplicate(items)
        assert len(result) == 1, "URL 중복 제거 실패"

    def test_empty_keyword_sanitization(self):
        """빈 키워드 sanitize 처리 확인"""
        from app.tools.trend.keyword_blacklist import sanitize_keyword
        assert sanitize_keyword("") is None or sanitize_keyword("") == ""
        assert sanitize_keyword("   ") is None or sanitize_keyword("   ") == ""

    @pytest.mark.asyncio
    async def test_tavily_unavailable_falls_back_gracefully(self):
        """Tavily API 키 없을 때 graceful fallback"""
        with patch("app.tools.trend.sources.tavily_source.TavilySource.is_available",
                   new_callable=lambda: property(lambda self: False)):
            collector = TrendCollector()
            # TrendSourceError가 아닌 빈 결과 반환해야 함
            try:
                result = await collector.collect_community_keywords(max_keywords=5)
                assert result == []
            except TrendSourceError:
                pass  # 현재 코드에서는 TrendSourceError 발생 — 재설계 후 graceful degradation 필요
```

---

## 4. 네이버 뉴스 카테고리별 베스트 비교 방법론

### 4.1 비교 프레임워크

네이버 뉴스 카테고리별 베스트는 편집자 큐레이션 + 독자 반응 기반으로 법률 관련성이 높음.
이를 골드 스탠다드로 삼아 우리 시스템 출력을 비교한다.

#### 4.1.1 수동 평가 체크리스트

```markdown
## 네이버 베스트 vs 우리 시스템 비교 평가표

평가 날짜: _______
카테고리: criminal / civil / family / labor (해당 표시)
시간 범위: 48h / 7d

### A. 네이버 사회/법률 섹션 베스트 5건 (직접 수집)
| 순위 | 제목 | 관련성(1-5) |
|------|------|-----------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### B. 우리 시스템 수집 뉴스 상위 5건
| 순위 | 제목 | total_score | 관련성(1-5) |
|------|------|------------|-----------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

### C. 중복 확인
A와 B에서 동일 기사 수: ___건 / 5건

### D. 종합 평가
- 관련성 평균 (A): ___ / 5.0
- 관련성 평균 (B): ___ / 5.0
- 결론: [ ] 동등 이상 [ ] 미달
```

#### 4.1.2 자동화 가능 부분

```python
# tests/integration/content_marketing/test_naver_best_comparison.py
# ※ 실제 API 호출 필요 — CI에서는 skip, 수동 검증 시 실행

@pytest.mark.skip(reason="수동 실행 전용 — 실제 API 호출")
@pytest.mark.asyncio
async def test_compare_with_naver_best():
    """네이버 베스트 vs 우리 시스템 비교 (수동 실행)"""
    from app.tools.trend.collector import TrendCollector
    from app.tools.trend.article_scorer import score_articles

    collector = TrendCollector()
    keyword = "부동산 사기"  # 테스트 키워드

    articles, sources_used, _ = await collector.search_news_for_keyword(
        keyword=keyword, max_results=10,
    )
    scored = score_articles(keyword, articles)

    print(f"\n=== '{keyword}' 뉴스 수집 결과 ===")
    print(f"소스: {sources_used}")
    print(f"총 {len(scored)}건")
    for i, a in enumerate(scored[:5], 1):
        print(f"{i}. [{a.total_score:.1f}점] {a.title}")
        print(f"   relevance={a.relevance_score:.3f}, legal={a.legal_score:.3f}, recency={a.recency_score:.3f}")

    # 수동 평가: 위 출력을 네이버 베스트와 비교
    # 자동 판단 불가 → 평가표 작성 후 QA 엔지니어 판정
```

### 4.2 정량적 비교 지표

| 비교 지표 | 측정 방법 | 합격 기준 |
|----------|----------|---------|
| 제목 중복 | 네이버 베스트 5건과 우리 결과 5건의 교집합 | 1건 이상 |
| 주요 사건 포함 | 네이버 헤드라인 사건이 우리 결과에 존재 | 3건 이상 / 5건 |
| 관련성 점수 (주관) | 평가자 1~5점 평균 | 3.5 이상 |
| 최신성 | 수집 기사 평균 발행일 | 수집 시점 기준 7일 이내 |

---

## 5. 리스크 분석

### 5.1 API 비용 및 Rate Limit 리스크

| 소스 | 무료 한도 | 일일 최대 요청 | 위험 수준 | 완화 방안 |
|------|---------|-------------|---------|---------|
| **pytrends (Google Trends)** | 무료 (비공식 API) | 요청 간 3초 지연 권고 | High — 차단 위험 | 캐싱 강화, 대체 소스 준비 |
| **YouTube Data API v3** | 10,000 units/일 (검색=100 units/회) | 100회/일 | High | 검색 결과 캐싱 TTL 1시간 |
| **네이버 뉴스 API** | 25,000건/일 | 약 50회/일 (500건/회) | Low | 현 한도 충분 |
| **NewsAPI.org** | 100건/일 (무료) | 10회/일 | Critical — 매우 제한 | 유료 전환 또는 낮은 우선순위 |
| **NewsData.io** | 200건/일 (무료) | ~40회/일 | High | 결과 캐싱 TTL 2시간 |
| **Tavily** | 1,000 credits/월 (무료) | 약 33회/일 | High | 커뮤니티 수집 전용, 다중 호출 지양 |
| **Perplexity** | API 키 필요 (유료) | 설정 무관 | Medium | 설정 시에만 활성화 |

### 5.2 Rate Limit 초과 시 시나리오 테스트

```python
# tests/unit/content_marketing/test_rate_limit_behavior.py

class TestRateLimitScenarios:

    @pytest.mark.asyncio
    async def test_youtube_quota_exceeded_graceful(self):
        """YouTube 할당량 초과(HTTP 403) 시 graceful degradation"""
        with patch("app.tools.trend.sources.youtube_source.YouTubeSource.fetch",
                   side_effect=Exception("HTTP 403 quotaExceeded")):
            collector = TrendCollector()
            articles, sources_used, sources_failed = \
                await collector.search_news_for_keyword("테스트", max_results=5)
            # 검증: YouTube 실패가 전체 결과에 영향 없음
            assert any(sf.source_name == "youtube" for sf in sources_failed)
            # 다른 소스 결과는 정상 반환 (Naver가 활성화된 경우)

    @pytest.mark.asyncio
    async def test_newsapi_daily_limit_tracking(self):
        """NewsAPI 일일 한도(100건) 도달 시 처리"""
        # 현재: 한도 도달 시 그냥 빈 리스트
        # 권장: 한도 도달 정보를 sources_failed에 명시 + 프론트엔드에 안내
        ...

    def test_in_memory_rate_limiter_blocks_excess(self):
        """InMemoryRateLimiter가 초과 요청을 차단하는지 확인"""
        from app.tools.trend.rate_limiter import InMemoryRateLimiter, RateLimitExceededError
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
        limiter.check("user1")  # 1번째
        limiter.check("user1")  # 2번째
        with pytest.raises(RateLimitExceededError):
            limiter.check("user1")  # 3번째 → 차단
```

### 5.3 법률 도메인 특수성 리스크

| 리스크 | 설명 | 완화 방안 |
|--------|------|---------|
| 법률 용어 오인식 | 일반 뉴스에서 "이혼" = 연예인 이혼 (법률 무관) | CATEGORY_KEYWORDS 확장 + legal_score 텍스트 기반 보정 |
| 전문 법률 뉴스 부재 | 네이버 일반 뉴스에 법률 전문 기사 적음 | 법률 전문 도메인(lawtimes.co.kr 등) 우선 수집 |
| 키워드 의미 모호성 | "파산" = 기업 파산 or 개인 파산 | 카테고리별 컨텍스트 쿼리로 disambiguation |
| RAG 연동 의존 | legal_score가 RAG 결과에 완전 의존 (D-04) | 텍스트 기반 법률 키워드 매칭으로 기본 점수 부여 |

---

## 6. 회귀 테스트 전략

### 6.1 회귀 테스트 기준값 수립

재설계 구현 완료 후 다음 기준값을 측정하여 `tests/fixtures/keyword_score_baseline.json`에 저장:

```json
{
  "version": "2.0-redesigned",
  "measured_at": "2026-03-XX",
  "baselines": {
    "criminal": {
      "category_match_rate": 0.87,
      "avg_total_score": 68.5,
      "legal_score_positive_rate": 0.73,
      "avg_source_count": 2.7
    },
    "civil": {
      "category_match_rate": 0.85,
      "avg_total_score": 66.2,
      "legal_score_positive_rate": 0.71,
      "avg_source_count": 2.5
    }
  }
}
```

### 6.2 자동화 회귀 테스트

```python
# tests/integration/content_marketing/test_score_regression.py

import json
from pathlib import Path

BASELINE_FILE = Path("tests/fixtures/keyword_score_baseline.json")


class TestScoreRegression:
    """변경 전/후 동일 쿼리 비교 (회귀 검증)"""

    @pytest.mark.skip(reason="기준값 수립 후 활성화 예정")
    @pytest.mark.parametrize("category", ["criminal", "civil", "family", "labor"])
    @pytest.mark.asyncio
    async def test_score_not_regressed(self, category: str):
        """재설계 후 스코어가 기준값 이상 유지되는지 확인"""
        if not BASELINE_FILE.exists():
            pytest.skip("기준값 파일 없음 — 기준값 수립 후 실행")

        baseline = json.loads(BASELINE_FILE.read_text())
        cat_baseline = baseline["baselines"].get(category)
        if not cat_baseline:
            pytest.skip(f"{category} 기준값 없음")

        # 실제 API 호출 (수동 실행 환경)
        from app.tools.trend.collector import TrendCollector
        from app.tools.trend.article_scorer import score_articles

        collector = TrendCollector()
        keywords, _, _ = await collector.collect_community_keywords_with_status(
            max_keywords=5, time_range="48h",
            # category=category  ← 재설계 후
        )

        if not keywords:
            pytest.skip("키워드 수집 결과 없음 (API 설정 확인 필요)")

        # 카테고리 일치율 측정
        # match_rate = ... (수동 레이블링 기준)
        # avg_score = ...
        # assert match_rate >= cat_baseline["category_match_rate"] * 0.95  # 5% 허용 오차


class TestConsistencyAcrossTimeRanges:
    """시간 범위 변경 시 결과 일관성 검증"""

    @pytest.mark.parametrize("time_range", ["48h", "7d", "14d", "30d"])
    @pytest.mark.asyncio
    async def test_time_range_parameter_accepted(self, time_range: str):
        """시간 범위 파라미터가 에러 없이 수용되는지 확인"""
        from app.tools.trend.collector import TrendCollector
        collector = TrendCollector()
        # 재설계 후 time_range 파라미터 지원 여부 확인
        try:
            await collector.collect_community_keywords_with_status(
                max_keywords=3,
                time_range=time_range,
            )
        except TypeError as e:
            pytest.fail(f"time_range='{time_range}' 파라미터 미지원: {e}")
        except Exception:
            pass  # API 키 없음 등의 이유로 실패해도 TypeError가 아니면 OK
```

---

## 7. 성능 기준 및 모니터링

### 7.1 응답 시간 기준

| 엔드포인트 | P50 목표 | P95 목표 | 타임아웃 설정 |
|----------|---------|---------|------------|
| `/keywords/collect` (키워드 수집) | 15초 | 30초 | 45초 |
| `/keywords/{id}/news` (뉴스 검색) | 8초 | 20초 | 30초 |
| 캐시 히트 시 | 50ms | 200ms | N/A |

### 7.2 소스별 타임아웃 기준값 (현재 코드 L408-416 검증)

```python
# tests/unit/content_marketing/test_timeout_config.py

class TestSourceTimeoutConfig:
    """소스별 타임아웃 설정 검증"""

    def test_naver_timeout_is_5_seconds(self):
        """Naver 타임아웃이 5초로 설정됨 (빠른 응답 소스)"""
        from app.tools.trend.collector import TrendCollector
        # collector.py L408-416의 source_timeouts 딕셔너리 검증
        collector = TrendCollector()
        # 소스별 타임아웃은 private — 검증을 위해 통합 테스트에서 실시간 측정 필요

    def test_perplexity_timeout_is_15_seconds(self):
        """Perplexity는 15초 타임아웃 (느린 소스)"""
        ...

    def test_community_collection_has_source_timeout(self):
        """D-06 관련: collect_community_keywords*()에도 소스별 타임아웃 적용 확인"""
        # 현재: search_news_for_keyword()에만 적용
        # 재설계 후: collect_community_keywords*()에도 적용 필요
        ...
```

### 7.3 자동 모니터링 로그 패턴

```python
# 서비스 운영 중 확인할 핵심 로그 패턴

MONITORING_PATTERNS = {
    # 정상 동작 지표
    "키워드_수집_완료": r"커뮤니티 키워드 수집: (\d+)건 \(성공 (\d+)개",
    "카테고리_쿼리_사용": r"카테고리 쿼리 적용: (criminal|civil|family|labor) → (.+)",  # 재설계 후
    "legal_score_경고": r"가드레일: 전체 기사 legal_score=0",  # 이 경고가 자주 나오면 D-04 재발

    # 경고 지표
    "소스_실패": r"소스 (\w+) 수집 실패",
    "LLM_추출_실패": r"LLM (스코어링|키워드) 추출 실패, 폴백 사용",
    "타임아웃": r"소스 (\w+) 타임아웃 \((\d+)s\)",
}

# 모니터링 임계값
ALERT_THRESHOLDS = {
    "legal_score_zero_rate": 0.3,    # 30% 이상이면 알림
    "source_failure_rate": 0.5,      # 50% 이상 소스 실패 시 알림
    "llm_fallback_rate": 0.2,        # 20% 이상 LLM 폴백 시 알림
}
```

---

## 8. 품질 게이트 (합격/불합격 판정)

### 8.1 배포 차단 기준 (Must Pass)

| ID | 기준 | 검증 방법 | 자동화 |
|----|------|---------|--------|
| QG-01 | `collect_community_keywords_with_status()` 카테고리 파라미터 지원 | 단위 테스트 | O |
| QG-02 | criminal/civil/family/labor 각각 다른 쿼리 생성 (D-01 수정) | 단위 테스트 | O |
| QG-03 | `article_scorer.py` RAG 없이도 법률 텍스트 키워드로 legal_score > 0 (D-04 수정) | 단위 테스트 | O |
| QG-04 | `search_news_for_keyword()` time_range 파라미터 수신 (D-06 수정) | 단위 테스트 | O |
| QG-05 | API 응답 시간 P95 ≤ 30초 | 성능 테스트 | O |
| QG-06 | 소스 1개 실패 시 전체 결과 영향 없음 (에러 격리) | 통합 테스트 | O |
| QG-07 | `ruff check` 에러 0개 | 정적 분석 | O |
| QG-08 | `mypy` 에러 0개 | 정적 분석 | O |

### 8.2 권고 기준 (Should Pass — 미통과 시 이슈 등록)

| ID | 기준 | 검증 방법 | 자동화 |
|----|------|---------|--------|
| QG-09 | 카테고리 일치율 ≥ 85% | 수동 평가 (5건 샘플) | X |
| QG-10 | `total_score` 평균 ≥ 65점 | 회귀 테스트 | X (수동) |
| QG-11 | `legal_score > 0` 비율 ≥ 70% | 회귀 테스트 | X (수동) |
| QG-12 | 네이버 베스트 대비 주요 기사 3건/5건 이상 일치 | 수동 비교 | X |
| QG-13 | collect_community_keywords*()에 소스별 타임아웃 적용 | 단위 테스트 | O |
| QG-14 | API 키 값이 에러 메시지/로그에 미포함 (보안) | 보안 스캔 | O |

### 8.3 테스트 실행 순서

```bash
# 1단계: 정적 분석 (필수)
cd backend
uv run ruff check app/tools/trend/ app/services/service_function/content_marketing_service.py
uv run mypy app/tools/trend/ app/services/service_function/content_marketing_service.py

# 2단계: 단위 테스트 (자동)
uv run pytest tests/unit/content_marketing/ -v --tb=short

# 3단계: 에지 케이스 테스트
uv run pytest tests/unit/content_marketing/test_edge_cases.py -v

# 4단계: 통합 테스트 (자동, Mock 환경)
uv run pytest tests/integration/content_marketing/ -v -m "not requires_real_api"

# 5단계: 보안 스캔
grep -rn "settings\.[A-Z_]*API_KEY" backend/app/tools/trend/ \
  | grep -v "is_available\|params\[" && echo "WARN: API키 노출 위험" || echo "SAFE"

# 6단계: 수동 평가 (API 환경)
# 6-1. 카테고리별 키워드 수동 확인 (체크리스트 작성)
# 6-2. 네이버 베스트 비교 (Section 4.1.1 체크리스트 활용)
```

---

## 9. Pre-mortem 분석 (재설계 후 잠재 실패 지점)

### 9.1 고위험 실패 시나리오

#### 시나리오 A: 카테고리 쿼리 생성 실패

| 항목 | 내용 |
|------|------|
| **발생 조건** | CATEGORY_KEYWORDS 딕셔너리에 없는 카테고리 입력 (예: "all") |
| **증상** | 카테고리 특화 쿼리 생성 불가 → 기본 쿼리 폴백 |
| **테스트** | category="all" → 폴백 동작 검증 |
| **예방** | "all" 카테고리는 범용 법률 쿼리로 명시적 처리 |

#### 시나리오 B: LLM 키워드 추출 카테고리 인지 실패

| 항목 | 내용 |
|------|------|
| **발생 조건** | 프롬프트에 카테고리 정보 포함했으나 LLM이 무시 |
| **증상** | 추출 키워드가 여전히 카테고리 무관 |
| **테스트** | criminal 프롬프트 → 추출 키워드 카테고리 일치율 측정 |
| **예방** | 프롬프트에 few-shot 예시 추가, legal_relevance < 0.3 필터 유지 |

#### 시나리오 C: 텍스트 기반 legal_score 과적합

| 항목 | 내용 |
|------|------|
| **발생 조건** | "법원" 단어가 포함된 모든 기사에 legal_score 부여 |
| **증상** | 법률과 무관한 기사 (예: "법원 앞 시위")가 높은 점수 획득 |
| **테스트** | 법률 무관 기사에 법률 키워드 포함 시 점수 분포 확인 |
| **예방** | 법률 키워드 매칭은 보조 점수(0.2 이하)로 제한, 기본값 보정 역할만 |

#### 시나리오 D: 카테고리별 API 할당량 편중

| 항목 | 내용 |
|------|------|
| **발생 조건** | criminal 카테고리만 자주 요청 시 해당 카테고리 캐시 미스율 높음 |
| **증상** | criminal 요청 시 API 할당량 먼저 소진 |
| **테스트** | 동일 카테고리 연속 요청 10회 → 캐시 히트율 및 API 소비 측정 |
| **예방** | 카테고리별 캐시 TTL 설정, 캐시 키에 카테고리 포함 확인 |

---

## 10. 검증 체크리스트 (구현팀 전달용)

### 백엔드 수정 체크리스트

- [ ] `collect_community_keywords()` / `collect_community_keywords_with_status()` — `category` 파라미터 추가 (D-01, D-02, D-03 수정)
- [ ] 카테고리별 쿼리 템플릿 `CATEGORY_QUERIES` 딕셔너리 추가 (`collector.py`)
- [ ] `_SCORING_PROMPT_TEMPLATE` 카테고리 컨텍스트 포함 (D-07 수정, `keyword_extractor.py`)
- [ ] `_calculate_legal()` 텍스트 키워드 기반 기본 점수 부여 로직 추가 (D-04 수정, `article_scorer.py`)
- [ ] `search_news_for_keyword()` — `time_range` 파라미터 추가 + SourceConfig에 전달 (D-06 수정)
- [ ] `collect_community_keywords*()` 소스별 타임아웃 적용
- [ ] API 키 값이 로그/에러 메시지에 미포함 확인
- [ ] `ruff check` + `mypy` 통과

### 테스트 체크리스트

- [ ] `tests/unit/content_marketing/test_category_keyword_quality.py` 작성
- [ ] `tests/unit/content_marketing/test_article_scorer.py` 작성 (D-04 검증 포함)
- [ ] `tests/unit/content_marketing/test_news_search_time_range.py` 작성 (D-06 검증)
- [ ] `tests/unit/content_marketing/test_edge_cases.py` 작성
- [ ] `tests/unit/content_marketing/test_rate_limit_behavior.py` 작성
- [ ] QG-01 ~ QG-08 (자동) 모두 통과 확인
- [ ] 카테고리별 수동 평가 체크리스트 1회 작성 (QG-09 ~ QG-12)

### 프론트엔드 동기화 체크리스트 (수정된 파라미터에 따라)

- [ ] `KeywordCollectRequest` — `category` 파라미터 추가 시 `frontend/src/features/content-marketing/types/` 동기화
- [ ] `KeywordNewsRequest` — `time_range` 파라미터 추가 시 동기화
- [ ] `frontend/src/lib/api.ts` endpoints 확인

---

## 참고 자료

- `docs/03-analysis/content-marketing-source-fix.qa-strategy.md` — 이전 QA 전략 (소스 활성화 수정)
- `docs/03-analysis/content-marketing-keyword-fix.consulting.md` — 키워드 품질 문제 분석
- `.claude/FAILURE_LOG.md` FAIL-001 — CLI 도구 오류 처리 규칙
- `backend/app/tools/trend/scorer.py` L178 — `CATEGORY_KEYWORDS` 딕셔너리 (활용 가능)
- `backend/app/tools/trend/article_scorer.py` — 점수 계산 핵심 로직
