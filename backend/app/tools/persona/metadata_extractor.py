"""MetadataExtractor — 대화 이력 → PII 없는 메타데이터 추출

Design 문서 §5.1 기반.
Red Team [심각 3] 반영: 원문 텍스트를 LLM에 전달하지 않고 메타데이터만 추출.
"""

import logging
from collections import Counter

from app.tools.persona.models import ChatMetadata

logger = logging.getLogger(__name__)

# 카테고리별 키워드 매핑
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "family": ["이혼", "양육권", "위자료", "재산분할", "상속", "유류분"],
    "criminal": ["사기", "횡령", "배임", "폭행", "협박", "명예훼손"],
    "labor": ["해고", "부당해고", "임금", "퇴직금", "산재"],
    "civil": ["손해배상", "채무불이행", "계약해제", "부동산", "전세"],
    "ip": ["특허", "상표", "저작권"],
    "corporate": ["회생", "파산", "조세"],
}


class MetadataExtractor:
    """대화 이력 → 메타데이터 추출 (원문 텍스트를 LLM에 전달하지 않음)"""

    LEGAL_KEYWORD_DICT: frozenset[str] = frozenset([
        "이혼", "양육권", "위자료", "재산분할", "상속", "유류분",
        "사기", "횡령", "배임", "폭행", "협박", "명예훼손",
        "해고", "부당해고", "임금", "퇴직금", "산재",
        "손해배상", "채무불이행", "계약해제", "부동산", "전세",
        "특허", "상표", "저작권", "회생", "파산", "조세",
    ])

    def extract(
        self,
        messages: list[dict[str, str]],
        top_k: int = 20,
    ) -> ChatMetadata:
        """대화 이력에서 PII 없는 메타데이터만 추출"""
        agent_types: Counter[str] = Counter(
            m.get("agent_type", "unknown") for m in messages
        )

        keyword_counter: Counter[str] = Counter()
        total_length = 0
        for msg in messages:
            content = msg.get("content", "")
            total_length += len(content)
            for keyword in self.LEGAL_KEYWORD_DICT:
                count = content.count(keyword)
                if count > 0:
                    keyword_counter[keyword] += count

        top_keywords = keyword_counter.most_common(top_k)
        category_map = self._estimate_categories(keyword_counter)

        return ChatMetadata(
            total_conversations=len(messages),
            agent_type_distribution=dict(agent_types),
            top_legal_keywords=top_keywords,
            category_distribution=category_map,
            avg_message_length=total_length / max(len(messages), 1),
            date_range_days=30,
        )

    def _estimate_categories(
        self,
        keyword_counter: Counter[str],
    ) -> dict[str, float]:
        """키워드 빈도 → 법률 카테고리 분포 추정"""
        scores: dict[str, int] = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            scores[category] = sum(
                keyword_counter.get(k, 0) for k in keywords
            )
        total = max(sum(scores.values()), 1)
        return {
            cat: round(score / total, 2)
            for cat, score in scores.items()
            if score > 0
        }
