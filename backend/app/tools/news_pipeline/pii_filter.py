"""개인정보(PII) 마스킹 필터"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# PII 패턴 (한국 법률 뉴스 특화)
_PHONE_PATTERN = re.compile(
    r"0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}",
)
_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
)
_ADDRESS_PATTERN = re.compile(
    r"(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)"
    r"(?:특별시|광역시|특별자치시|도|특별자치도)?\s+"
    r"[가-힣]+(?:시|군|구)\s+"
    r"[가-힣]+(?:읍|면|동|로|길)\s*"
    r"[\d\-가-힣]*",
)
_RESIDENT_ID_PATTERN = re.compile(
    r"\d{6}[-\s]?[1-4]\d{6}",
)

# 마스킹 대체 문자열
MASK_MAP: dict[str, str] = {
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "address": "[ADDRESS]",
    "resident_id": "[RESIDENT_ID]",
}


class PIIFilter:
    """PII 마스킹 필터

    정규식 기반으로 전화번호, 이메일, 주소, 주민번호를 마스킹.
    법률 뉴스 특성상 판사/검사/변호사 실명은 공인으로 분류하여 유지.
    """

    def mask(self, text: str) -> tuple[str, int]:
        """텍스트 내 PII를 마스킹하고 (마스킹된 텍스트, 마스킹 건수) 반환"""
        count = 0

        # 주민번호 (가장 먼저 — 가장 민감)
        text, n = _RESIDENT_ID_PATTERN.subn(MASK_MAP["resident_id"], text)
        count += n

        # 전화번호
        text, n = _PHONE_PATTERN.subn(MASK_MAP["phone"], text)
        count += n

        # 이메일
        text, n = _EMAIL_PATTERN.subn(MASK_MAP["email"], text)
        count += n

        # 주소
        text, n = _ADDRESS_PATTERN.subn(MASK_MAP["address"], text)
        count += n

        if count > 0:
            logger.info("PII 마스킹 %d건 적용", count)

        return text, count
