"""뉴스 기사 점수 계산 모듈

v1 (score_articles): 3차원 경량 스코어링
- relevance_score: exact(0.3) + no_space(0.2) + token(0.3) + snippet_freq(0.2)
- legal_score: legal_issue_label 유무(0.5) + related_laws 개수(0.5)
- recency_score: exp(-ln2 * days/7) 지수 감쇠 (7일 반감기)
- total_score: (rel*0.4 + legal*0.3 + rec*0.3) * 100

v2 (score_articles_v2): 5차원 스코어링
- relevance 0.25, legal 0.25, recency 0.15, engagement 0.20, convergence 0.15
- engagement: log10 정규화 + 댓글 부스트
- convergence: 키워드 수렴 점수 패스스루
- score_breakdown 설명 가능성 필드 포함
"""

import logging
import math
import re
from datetime import datetime, timezone

from app.modules.content_marketing.schema import NewsArticle

logger = logging.getLogger(__name__)

# 가중치 상수
_RELEVANCE_WEIGHT: float = 0.4
_LEGAL_WEIGHT: float = 0.3
_RECENCY_WEIGHT: float = 0.3

# 지수 감쇠 반감기 (일)
_HALF_LIFE_DAYS: float = 7.0
_DECAY_LAMBDA: float = math.log(2) / _HALF_LIFE_DAYS

# legal_score 관련 법률 개수 정규화 기준
_MAX_RELATED_LAWS: int = 5

# HTML 태그 제거 패턴
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """HTML 태그 제거 + 공백 정규화"""
    cleaned = _HTML_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _calculate_relevance(keyword: str, title: str, snippet: str) -> float:
    """키워드 관련도 점수 (0~1)

    4가지 신호를 결합:
    - exact match: 원본 키워드가 텍스트에 그대로 포함
    - no-space match: 공백 제거 후 한국어 복합어 매칭
    - token match: 키워드를 토큰으로 분리하여 포함 비율
    - snippet frequency: snippet 내 키워드 출현 빈도
    """
    if not keyword:
        return 0.0

    keyword_clean = _strip_html(keyword).lower()
    if not keyword_clean:
        return 0.0

    title_clean = _strip_html(title).lower()
    snippet_clean = _strip_html(snippet).lower()

    # 공백 제거 버전 (한국어 복합어 대응: "부동산 사기" ↔ "부동산사기")
    keyword_nospace = keyword_clean.replace(" ", "")
    title_nospace = title_clean.replace(" ", "")
    snippet_nospace = snippet_clean.replace(" ", "")

    # 1. Exact match — 원본 키워드가 제목에 포함 (0.3 가중치)
    exact_title = 1.0 if keyword_clean in title_clean else 0.0

    # 2. No-space match — 공백 제거 후 매칭 (0.2 가중치)
    nospace_title = 1.0 if keyword_nospace in title_nospace else 0.0
    nospace_snippet = 1.0 if keyword_nospace in snippet_nospace else 0.0
    nospace_score = max(nospace_title, nospace_snippet * 0.6)

    # 3. Token match — 키워드 토큰별 포함 비율 (0.3 가중치)
    tokens = [t for t in keyword_clean.split() if t]
    if tokens:
        title_token_hits = sum(1 for t in tokens if t in title_clean)
        snippet_token_hits = sum(1 for t in tokens if t in snippet_clean)
        title_token_ratio = title_token_hits / len(tokens)
        snippet_token_ratio = snippet_token_hits / len(tokens)
        token_score = title_token_ratio * 0.6 + snippet_token_ratio * 0.4
    else:
        token_score = 0.0

    # 4. Snippet frequency — snippet 내 키워드 출현 빈도 (0.2 가중치)
    if snippet_clean and keyword_clean:
        count = snippet_clean.count(keyword_clean)
        snippet_freq = min(count / 5.0, 1.0)
    else:
        snippet_freq = 0.0

    return (
        exact_title * 0.3
        + nospace_score * 0.2
        + token_score * 0.3
        + snippet_freq * 0.2
    )


def _calculate_legal(legal_issue_label: str | None, related_laws: list[str]) -> float:
    """법적 관련도 점수 (0~1) — v1 호환"""
    # legal_issue_label 유무 (0.5 가중치)
    label_score = 1.0 if legal_issue_label else 0.0

    # related_laws 개수 (0.5 가중치, 최대 5개 기준 정규화)
    laws_count = len(related_laws)
    laws_score = min(laws_count / _MAX_RELATED_LAWS, 1.0)

    return label_score * 0.5 + laws_score * 0.5


# ── v2: 카테고리 연동 법적 점수 (설계 §2.5) ──

# 카테고리별 Legal Taxonomy 키워드 (Gemini 리뷰 반영)
_CATEGORY_LEGAL_TAXONOMY: dict[str, list[str]] = {
    "criminal": ["형사", "형법", "처벌", "벌금", "구속", "기소", "공소", "무죄", "유죄", "징역", "집행유예", "수사"],
    "civil": ["민사", "민법", "손해배상", "계약", "채권", "채무", "불법행위", "소유권", "부동산", "대여금"],
    "labor": ["노동", "근로기준법", "해고", "임금", "산재", "근로자", "사용자", "퇴직금", "직장내 괴롭힘"],
    "family": ["가사", "이혼", "양육", "친권", "상속", "재산분할", "가사소송", "면접교섭"],
    "administrative": ["행정", "행정소송", "처분", "인허가", "과태료", "행정심판", "취소소송"],
    "corporate": ["회사법", "상법", "M&A", "주주", "이사", "공정거래", "금융", "증권", "지배구조"],
    "ip": ["특허", "상표", "저작권", "영업비밀", "디자인권", "지식재산", "침해", "실용신안"],
}


def _calculate_legal_v2(
    title: str,
    snippet: str,
    legal_issue_label: str | None,
    related_laws: list[str],
    category: str,
) -> float:
    """카테고리 연동 법적 관련도 점수 (0~1) — v2

    3-way 가중 결합 (설계 §2.5):
    - kw_score (0.2): 카테고리 Legal Taxonomy 키워드 매칭
    - rag_score (0.4): RAG 기반 legal_issue_label + related_laws
    - taxonomy_score (0.4): 카테고리 특화 분류 정확도
    """
    text_lower = f"{title} {snippet}".lower()

    # 1. kw_score: 카테고리 Taxonomy 키워드 매칭 (0.2)
    taxonomy_keywords = _CATEGORY_LEGAL_TAXONOMY.get(category, [])
    if taxonomy_keywords:
        hits = sum(1 for kw in taxonomy_keywords if kw in text_lower)
        kw_score = min(hits / max(len(taxonomy_keywords) * 0.3, 1.0), 1.0)
    else:
        # 'all' 카테고리 — 아무 법률 키워드라도 포함되면 점수 부여
        all_keywords = [kw for keywords in _CATEGORY_LEGAL_TAXONOMY.values() for kw in keywords]
        hits = sum(1 for kw in all_keywords if kw in text_lower)
        kw_score = min(hits / 5.0, 1.0)

    # 2. rag_score: RAG enrichment 결과 (0.4)
    label_score = 1.0 if legal_issue_label else 0.0
    laws_count = len(related_laws)
    laws_score = min(laws_count / _MAX_RELATED_LAWS, 1.0)
    rag_score = label_score * 0.6 + laws_score * 0.4

    # 3. taxonomy_score: 카테고리-기사 정합성 (0.4)
    if category == "all" or not taxonomy_keywords:
        # 카테고리 미지정 시 RAG 점수를 그대로 활용
        taxonomy_score = rag_score
    else:
        # legal_issue_label이 카테고리 Taxonomy에 매칭되는지 검증
        label_match = 0.0
        if legal_issue_label:
            label_lower = legal_issue_label.lower()
            label_match = 1.0 if any(kw in label_lower for kw in taxonomy_keywords) else 0.3

        # related_laws 중 카테고리 관련 법령 비율
        law_match = 0.0
        if related_laws:
            matching_laws = sum(
                1 for law in related_laws
                if any(kw in law.lower() for kw in taxonomy_keywords)
            )
            law_match = min(matching_laws / max(laws_count, 1), 1.0)

        taxonomy_score = label_match * 0.5 + law_match * 0.5

    return kw_score * 0.2 + rag_score * 0.4 + taxonomy_score * 0.4


def _calculate_recency(published_at: datetime | None) -> float:
    """최신성 점수 (0~1), 지수 감쇠"""
    if published_at is None:
        return 0.0

    now = datetime.now(tz=timezone.utc)
    # timezone-aware 변환
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    days_ago = (now - published_at).total_seconds() / 86400.0
    if days_ago < 0:
        return 1.0

    return math.exp(-_DECAY_LAMBDA * days_ago)


def score_articles(keyword: str, articles: list[NewsArticle]) -> list[NewsArticle]:
    """뉴스 기사에 점수를 계산하고 total_score 내림차순으로 정렬 반환

    Args:
        keyword: 검색 키워드
        articles: 점수 미계산 뉴스 기사 리스트

    Returns:
        점수가 채워진 NewsArticle 리스트 (total_score 내림차순)
    """
    scored: list[NewsArticle] = []

    for article in articles:
        relevance = _calculate_relevance(keyword, article.title, article.snippet)
        legal = _calculate_legal(article.legal_issue_label, article.related_laws)
        recency = _calculate_recency(article.published_at)
        total = (relevance * _RELEVANCE_WEIGHT + legal * _LEGAL_WEIGHT + recency * _RECENCY_WEIGHT) * 100

        scored.append(article.model_copy(update={
            "relevance_score": round(relevance, 4),
            "legal_score": round(legal, 4),
            "recency_score": round(recency, 4),
            "total_score": round(total, 2),
        }))

    scored.sort(key=lambda a: a.total_score, reverse=True)

    # 가드레일: 전체 기사의 relevance/legal이 모두 0이면 경고
    if scored:
        all_relevance_zero = all(a.relevance_score == 0.0 for a in scored)
        all_legal_zero = all(a.legal_score == 0.0 for a in scored)
        if all_relevance_zero:
            logger.warning(
                "가드레일: 전체 기사 relevance_score=0 (keyword='%s', count=%d)",
                keyword[:30], len(scored),
            )
        if all_legal_zero:
            logger.warning(
                "가드레일: 전체 기사 legal_score=0 (keyword='%s', count=%d)",
                keyword[:30], len(scored),
            )

    return scored


# ── v2: 5차원 스코어링 ──

_V2_RELEVANCE_WEIGHT: float = 0.25
_V2_LEGAL_WEIGHT: float = 0.25
_V2_RECENCY_WEIGHT: float = 0.15
_V2_ENGAGEMENT_WEIGHT: float = 0.20
_V2_CONVERGENCE_WEIGHT: float = 0.15

# engagement 정규화 기준값
_VIEW_COUNT_CAP: int = 100_000
_COMMENT_COUNT_CAP: int = 500
_COMMENT_BOOST_THRESHOLD: int = 50


def _calculate_engagement(
    view_count: int | None,
    comment_count: int | None,
) -> float:
    """참여도 점수 (0~1), log10 정규화 + 댓글 부스트

    - view: log10(view+1) / log10(CAP+1) capped at 1.0 (가중 0.6)
    - comment: log10(comment+1) / log10(CAP+1) capped at 1.0 (가중 0.4)
    - 댓글 50개 이상이면 0.1 부스트 (최대 1.0)
    """
    views = max(view_count or 0, 0)
    comments = max(comment_count or 0, 0)

    view_score = min(math.log10(views + 1) / math.log10(_VIEW_COUNT_CAP + 1), 1.0)
    comment_score = min(math.log10(comments + 1) / math.log10(_COMMENT_COUNT_CAP + 1), 1.0)

    base = view_score * 0.6 + comment_score * 0.4

    # 댓글 부스트
    if comments >= _COMMENT_BOOST_THRESHOLD:
        base = min(base + 0.1, 1.0)

    return base


def score_articles_v2(
    keyword: str,
    articles: list[NewsArticle],
    keyword_convergence_score: float = 0.0,
    category: str = "all",
) -> list[NewsArticle]:
    """5차원 스코어링으로 기사 점수 계산 및 정렬

    Args:
        keyword: 검색 키워드
        articles: 점수 미계산 뉴스 기사 리스트
        keyword_convergence_score: 키워드의 수렴 점수 (0~1)
        category: 법률 카테고리 (legal_v2 점수에 사용)

    Returns:
        5차원 점수가 채워진 NewsArticle 리스트 (total_score 내림차순)
    """
    scored: list[NewsArticle] = []

    for article in articles:
        relevance = _calculate_relevance(keyword, article.title, article.snippet)
        legal = _calculate_legal_v2(
            title=article.title,
            snippet=article.snippet,
            legal_issue_label=article.legal_issue_label,
            related_laws=article.related_laws,
            category=category,
        )
        recency = _calculate_recency(article.published_at)
        engagement = _calculate_engagement(article.view_count, article.comment_count)
        convergence = keyword_convergence_score

        total = (
            relevance * _V2_RELEVANCE_WEIGHT
            + legal * _V2_LEGAL_WEIGHT
            + recency * _V2_RECENCY_WEIGHT
            + engagement * _V2_ENGAGEMENT_WEIGHT
            + convergence * _V2_CONVERGENCE_WEIGHT
        ) * 100

        breakdown = {
            "relevance": round(relevance, 4),
            "legal": round(legal, 4),
            "recency": round(recency, 4),
            "engagement": round(engagement, 4),
            "convergence": round(convergence, 4),
        }

        scored.append(article.model_copy(update={
            "relevance_score": round(relevance, 4),
            "legal_score": round(legal, 4),
            "recency_score": round(recency, 4),
            "engagement_score": round(engagement, 4),
            "convergence_score": round(convergence, 4),
            "total_score": round(total, 2),
            "score_breakdown": breakdown,
        }))

    scored.sort(key=lambda a: a.total_score, reverse=True)

    # 가드레일
    if scored:
        all_relevance_zero = all(a.relevance_score == 0.0 for a in scored)
        all_legal_zero = all(a.legal_score == 0.0 for a in scored)
        if all_relevance_zero:
            logger.warning(
                "가드레일(v2): 전체 기사 relevance_score=0 (keyword='%s', count=%d)",
                keyword[:30], len(scored),
            )
        if all_legal_zero:
            logger.warning(
                "가드레일(v2): 전체 기사 legal_score=0 (keyword='%s', count=%d)",
                keyword[:30], len(scored),
            )

    return scored
