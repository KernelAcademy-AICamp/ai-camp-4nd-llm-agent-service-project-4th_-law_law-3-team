"""
변호사 서비스

위치 및 전문분야 추출, 변호사 검색 지원
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 전문분야 키워드 매핑
SPECIALTY_KEYWORDS: Dict[str, List[str]] = {
    "민사": ["민사", "계약", "채권", "채무", "손해배상", "임대차", "전세", "월세"],
    "형사": ["형사", "범죄", "고소", "고발", "구속", "기소", "재판"],
    "가사": ["이혼", "양육권", "상속", "유언", "재산분할", "가사"],
    "부동산": ["부동산", "토지", "건물", "등기", "분양", "재개발"],
    "기업": ["회사", "법인", "기업", "M&A", "합병", "인수"],
    "노동": ["노동", "근로", "해고", "임금", "퇴직금", "산재"],
    "행정": ["행정", "허가", "인허가", "소송", "취소"],
    "지적재산권": ["특허", "상표", "저작권", "지식재산", "IP"],
    "세무": ["세금", "세무", "조세", "탈세", "국세"],
    "의료": ["의료", "병원", "의사", "의료사고", "의료분쟁"],
}

# 지역 패턴 (시/도, 구/군/시)
REGION_PATTERNS = [
    r"(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)",
    r"(강남|서초|송파|마포|영등포|종로|중구|용산|성동|광진|동대문|중랑|성북|강북|도봉|노원|"
    r"은평|서대문|양천|구로|금천|동작|관악|강서|강동|잠실|판교|분당|일산|수원|성남)",
]


class LawyerService:
    """변호사 서비스 클래스"""

    def extract_location(self, message: str) -> Optional[Dict[str, Any]]:
        """
        메시지에서 위치 정보 추출

        Args:
            message: 사용자 메시지

        Returns:
            {"region": "지역명", "sub_region": "세부지역"} 또는 None
        """
        location: Dict[str, Any] = {}

        # 시/도 추출
        for pattern in REGION_PATTERNS:
            match = re.search(pattern, message)
            if match:
                region = match.group(1)
                if region in ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
                              "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]:
                    location["region"] = region
                else:
                    location["sub_region"] = region

        if location:
            return location
        return None

    def extract_specialty(self, message: str) -> Optional[str]:
        """
        메시지에서 전문분야 추출

        Args:
            message: 사용자 메시지

        Returns:
            전문분야명 또는 None
        """
        message_lower = message.lower()

        for specialty, keywords in SPECIALTY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return specialty

        return None

    def extract_requirements(self, message: str) -> Dict[str, Any]:
        """
        메시지에서 변호사 검색 요구사항 추출

        Args:
            message: 사용자 메시지

        Returns:
            {"location": {...}, "specialty": "...", "keywords": [...]}
        """
        return {
            "location": self.extract_location(message),
            "specialty": self.extract_specialty(message),
            "keywords": self._extract_keywords(message),
        }

    def _extract_keywords(self, message: str) -> List[str]:
        """메시지에서 검색 키워드 추출"""
        keywords = []

        # 법률 용어 키워드 추출
        legal_terms = [
            "손해배상", "계약위반", "사기", "횡령", "배임",
            "이혼", "상속", "유언", "임대차", "전세",
            "해고", "퇴직금", "산재", "의료사고",
        ]

        for term in legal_terms:
            if term in message:
                keywords.append(term)

        return keywords


_lawyer_service: Optional[LawyerService] = None


def get_lawyer_service() -> LawyerService:
    """LawyerService 싱글톤 인스턴스 반환"""
    global _lawyer_service
    if _lawyer_service is None:
        _lawyer_service = LawyerService()
    return _lawyer_service
