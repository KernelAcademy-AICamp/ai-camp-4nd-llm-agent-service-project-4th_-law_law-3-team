"""골든 데이터셋 JSON 생성 스크립트

변호사시험 15개 케이스의 input_text + ground_truth를 생성한다.
실행: cd backend && uv run python scripts/build_eval_dataset.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.modules.storyboard.evaluation.dataset_builder import extract_case_section
from app.modules.storyboard.evaluation.schemas import (
    TimelineEvalCase,
    TimelineEvalDataset,
)

BASE_DIR = Path(__file__).parent.parent.parent / "data" / "bar_exam_raw"
OUTPUT_PATH = (
    Path(__file__).parent.parent
    / "app"
    / "modules"
    / "storyboard"
    / "evaluation"
    / "datasets"
    / "bar_exam_timeline_eval_v1.json"
)


def _extract_civil9_manually() -> str:
    """CIVIL-9는 자동 추출이 실패하므로 수동 추출한다."""
    filepath = BASE_DIR / "[CIVIL]9.md"
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")

    # 169행부터 실제 상담 내용 시작
    content_lines: list[str] = []
    started = False
    for i, line in enumerate(lines):
        if i >= 168 and "상 담 내 용" in line:
            started = True
        if started:
            content_lines.append(line)
        if started and i >= 168 and ("사건관계인" in line or "희 망 사 항" in line):
            break
    # 마크다운 테이블 파이프 제거, <br> → 줄바꿈
    raw = "\n".join(content_lines)
    raw = re.sub(r"\|", " ", raw)
    raw = re.sub(r"<br\s*/?>", "\n", raw)
    raw = re.sub(r"---+", "", raw)
    raw = re.sub(r"\s{3,}", " ", raw)
    return raw.strip()


def _e(
    eid: str,
    date_raw: str,
    date_norm: str | None,
    precision: str,
    title: str,
    participants: list[str],
    roles: dict[str, str],
    keywords: list[str],
    *,
    is_key: bool = True,
    location: str | None = None,
) -> dict:
    """TimelineEventGT dict 생성 헬퍼."""
    return {
        "event_id": eid,
        "date_raw": date_raw,
        "date_normalized": date_norm,
        "date_precision": precision,
        "title": title,
        "key_participants": participants,
        "participant_roles": roles,
        "location_hint": location,
        "legal_significance_keywords": keywords,
        "is_key_event": is_key,
    }


def _gt(
    events: list[dict],
    order: list[str],
    participants: list[str],
    narrative: str = "non_chronological",
) -> dict:
    """TimelineGroundTruth dict 생성 헬퍼."""
    return {
        "expected_event_count": len(events),
        "chronological_order": order,
        "events": events,
        "key_participants": participants,
        "narrative_vs_chronological": narrative,
    }


# ============================================================================
# CIVIL 케이스 정의
# ============================================================================


def _civil_001() -> dict:
    """CIVIL-1: 박대원 상속재산 분쟁"""
    events = [
        _e("E01", "1992년", "1992", "year", "박점숙 출가하여 부산 거주",
           ["박점숙"], {"박점숙": "other"}, ["출가"]),
        _e("E02", "1995년경", "1995", "approximate", "박대원 캐나다 출국 태권도 체육관 운영",
           ["박대원"], {"박대원": "other"}, ["출국"]),
        _e("E03", "2000년 여름", "2000-07", "approximate", "박정수 급성 폐렴 사망",
           ["박정수"], {"박정수": "victim"}, ["사망", "상속"]),
        _e("E04", "2005년 봄", "2005-04", "approximate", "박대원 귀국",
           ["박대원"], {"박대원": "other"}, ["귀국"]),
        _e("E05", "2005년 이후", "2005", "approximate", "박진수 서류위조 후 자기 앞 등기",
           ["박진수"], {"박진수": "perpetrator"}, ["서류위조", "등기"]),
        _e("E06", "2005년 이후", "2005", "approximate", "나대지에 신한은행 근저당권 설정",
           ["박진수"], {"박진수": "perpetrator"}, ["근저당권"]),
        _e("E07", "2005년 이후", "2005", "approximate", "잡종지를 김영철에게 임대",
           ["박진수", "김영철"], {"박진수": "perpetrator", "김영철": "other"}, ["임대"]),
        _e("E08", "2012. 1. 5.", "2012-01-05", "exact", "조성팔이 박진수로부터 상가 매수, 소유권이전등기",
           ["조성팔", "박진수"], {"조성팔": "other", "박진수": "perpetrator"}, ["매매", "소유권이전"]),
        _e("E09", "2012. 3. 3.", "2012-03-03", "exact", "박대원이 조성팔과 상가 임대차계약",
           ["박대원", "조성팔"], {"박대원": "other", "조성팔": "other"}, ["임대차"]),
        _e("E10", "2012. 3. 15.", "2012-03-15", "exact", "박대원이 상가 인도받아 가구점 영업 개시",
           ["박대원"], {"박대원": "other"}, ["인도", "영업개시"]),
        _e("E11", "2012. 6. 5.", "2012-06-05", "exact", "조성팔이 오인석에게 보증금 차감 양도, 소유권이전",
           ["조성팔", "오인석"], {"조성팔": "other", "오인석": "other"}, ["양도", "소유권이전"]),
        _e("E12", "2012. 7. 경", "2012-07", "approximate", "오인석이 차임 인상 요구 및 퇴거 경고",
           ["오인석", "박대원"], {"오인석": "other", "박대원": "other"}, ["차임인상", "퇴거"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11", "E12"]
    participants = ["박대원", "박정수", "박점숙", "박진수", "김영철", "조성팔", "오인석"]
    return {
        "id": "CIVIL-001", "source_file": "[CIVIL]1.md", "doc_type": "civil",
        "ground_truth": _gt(events, order, participants),
        "metadata": {"difficulty": "hard", "participant_count": 7},
    }


def _civil_003() -> dict:
    """CIVIL-3: 이명구/최희선 공동 토지 매수 분쟁"""
    events = [
        _e("E01", "2010년", "2010", "year", "이명구와 최희선 공동 부동산 매수 합의, 각 3억 출연",
           ["이명구", "최희선"], {"이명구": "other", "최희선": "other"}, ["공동매수", "합의"]),
        _e("E02", "2010. 5.", "2010-05", "month", "이명구가 정준일에게서 박이채 명의로 토지 2필지 매수",
           ["이명구", "정준일", "박이채"], {"이명구": "other", "정준일": "other", "박이채": "other"}, ["매매", "명의차용"]),
        _e("E03", "2010. 6. 30.", "2010-06-30", "exact", "매매대금 전액 지급, 520 토지 박이채 명의 소유권이전등기",
           ["이명구", "박이채"], {"이명구": "other", "박이채": "other"}, ["대금지급", "소유권이전"]),
        _e("E04", "2012. 12. 25.", "2012-12-25", "exact", "업무 담당 이명구에서 최희선으로 변경",
           ["이명구", "최희선"], {"이명구": "other", "최희선": "other"}, ["업무변경"]),
        _e("E05", "2012. 12. 25. 이후", "2013", "approximate", "박이채가 520 토지를 서병석에게 담보 이전",
           ["박이채", "서병석"], {"박이채": "other", "서병석": "other"}, ["담보", "소유권이전"]),
        _e("E06", "2013년경", "2013", "approximate", "최희선이 521 토지에 대해 전소유자 상속인 앞 등기 확인",
           ["최희선"], {"최희선": "other"}, ["등기확인"]),
        _e("E07", "2013년경", "2013", "approximate", "최희선이 전소유자 상속인에게서 521 토지 자기 명의 등기",
           ["최희선"], {"최희선": "other"}, ["소유권이전"]),
        _e("E08", "2014. 3.", "2014-03", "month", "최희선이 이명구 동의 없이 521 토지를 한인수에게 매도",
           ["최희선", "한인수"], {"최희선": "other", "한인수": "other"}, ["매도", "무단처분"]),
        _e("E09", "2014. 6.", "2014-06", "month", "한인수가 521 토지에 근저당권 설정",
           ["한인수"], {"한인수": "other"}, ["근저당권"]),
        _e("E10", "2015. 2.", "2015-02", "month", "서병석이 520 토지를 강현준에게 매도",
           ["서병석", "강현준"], {"서병석": "other", "강현준": "other"}, ["매도"]),
        _e("E11", "2015. 3.", "2015-03", "month", "강현준 520 토지 소유권이전등기",
           ["강현준"], {"강현준": "other"}, ["소유권이전"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11"]
    participants = ["이명구", "최희선", "박이채", "정준일", "서병석", "강현준", "한인수"]
    return {
        "id": "CIVIL-003", "source_file": "[CIVIL]3.md", "doc_type": "civil",
        "ground_truth": _gt(events, order, participants),
        "metadata": {"difficulty": "hard", "participant_count": 7},
    }


def _civil_005() -> dict:
    """CIVIL-5: 조병갑 임대차 분쟁"""
    events = [
        _e("E01", "2013. 1. 4.", "2013-01-04", "exact", "조병갑이 최병철로부터 음식점 건물 임차계약",
           ["조병갑", "최병철"], {"조병갑": "other", "최병철": "other"}, ["임대차계약"]),
        _e("E02", "2013. 1. 9.", "2013-01-09", "exact", "조병갑이 건물 인도받아 영업 개시",
           ["조병갑"], {"조병갑": "other"}, ["인도", "영업개시"]),
        _e("E03", "2015. 8. 9.", "2015-08-09", "exact", "조병갑 차임 미지급 (1차)",
           ["조병갑"], {"조병갑": "other"}, ["차임미납"]),
        _e("E04", "2015. 10. 9.", "2015-10-09", "exact", "조병갑 차임 미지급 (2차)",
           ["조병갑"], {"조병갑": "other"}, ["차임미납"]),
        _e("E05", "2015. 12. 1.", "2015-12-01", "exact", "조병갑이 최병철에게 임대차계약 갱신 요구 내용증명 발송",
           ["조병갑", "최병철"], {"조병갑": "other", "최병철": "other"}, ["갱신요구", "내용증명"]),
        _e("E06", "2015. 12. 6.", "2015-12-06", "exact", "최병철의 갱신 거절 답신 수령",
           ["조병갑", "최병철"], {"조병갑": "other", "최병철": "other"}, ["갱신거절"]),
        _e("E07", "2015. 12. 6.~12.", "2015-12-08", "approximate", "조병갑 차임 변제공탁",
           ["조병갑"], {"조병갑": "other"}, ["변제공탁"]),
        _e("E08", "2015. 12. 12.", "2015-12-12", "exact", "최병철이 해지통지서 발송",
           ["최병철"], {"최병철": "other"}, ["해지통지"]),
        _e("E09", "2015. 12. 14.", "2015-12-14", "exact", "조병갑이 해지통지서 수령",
           ["조병갑"], {"조병갑": "other"}, ["해지통지수령"]),
        _e("E10", "2015. 12. 14. 이후", "2015-12-20", "approximate", "최병철이 매일 음식점 방문하여 퇴거 요구",
           ["최병철", "조병갑"], {"최병철": "other", "조병갑": "other"}, ["퇴거요구"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10"]
    participants = ["조병갑", "최병철"]
    return {
        "id": "CIVIL-005", "source_file": "[CIVIL]5.md", "doc_type": "civil",
        "ground_truth": _gt(events, order, participants),
        "metadata": {"difficulty": "medium", "participant_count": 2},
    }


def _civil_007() -> dict:
    """CIVIL-7: 권창균 건물신축 분쟁"""
    events = [
        _e("E01", "시기 미상", None, "relative", "권창균이 김정우로부터 동탄면 대지 매수, 계약금+중도금 지급",
           ["권창균", "김정우"], {"권창균": "other", "김정우": "other"}, ["매매", "대금지급"]),
        _e("E02", "시기 미상", None, "relative", "권창균이 동탄면 대지에 처분금지가처분 결정",
           ["권창균"], {"권창균": "other"}, ["가처분"]),
        _e("E03", "2017. 2. 9.", "2017-02-09", "exact", "강주원과 권창균 대물변제 약정 (8억 채권 대신 토지+건물 소유권 이전)",
           ["강주원", "권창균"], {"강주원": "other", "권창균": "other"}, ["대물변제"]),
        _e("E04", "2017. 4. 15.경", "2017-04-15", "exact", "윤태건이 동탄면 건물 완공",
           ["윤태건"], {"윤태건": "other"}, ["건물완공", "도급"]),
        _e("E05", "2017. 4. 30.", "2017-04-30", "exact", "이청준이 동탄면 건물 1층에서 사업자등록",
           ["이청준"], {"이청준": "other"}, ["사업자등록"]),
        _e("E06", "2017. 5. 초경", "2017-05", "approximate", "이청준이 무단으로 별채건물 건축",
           ["이청준"], {"이청준": "other"}, ["무단건축"]),
        _e("E07", "시기 미상", None, "relative", "윤태건이 권창균 승낙 없이 이청준에게 1층 임대",
           ["윤태건", "이청준"], {"윤태건": "other", "이청준": "other"}, ["무단임대"]),
        _e("E08", "2017. 6. 말경", "2017-06", "approximate", "이청준이 건물에서 퇴거",
           ["이청준"], {"이청준": "other"}, ["퇴거"]),
        _e("E09", "2017. 7. 1.", "2017-07-01", "exact", "이청준이 별채건물에서 영업 시작",
           ["이청준"], {"이청준": "other"}, ["영업개시"]),
        _e("E10", "2017. 8.", "2017-08", "month", "강주원이 권창균에 대물변제 이행 청구",
           ["강주원", "권창균"], {"강주원": "other", "권창균": "other"}, ["이행청구"]),
        _e("E11", "2017. 9. 18.", "2017-09-18", "exact", "윤태건이 공사대금 채권으로 건물 유치권 주장",
           ["윤태건"], {"윤태건": "other"}, ["유치권"]),
        _e("E12", "2017. 12. 13.", "2017-12-13", "exact", "권창균에 대한 파산선고",
           ["권창균"], {"권창균": "other"}, ["파산"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11", "E12"]
    participants = ["권창균", "김정우", "강주원", "윤태건", "이청준"]
    return {
        "id": "CIVIL-007", "source_file": "[CIVIL]7.md", "doc_type": "civil",
        "ground_truth": _gt(events, order, participants),
        "metadata": {"difficulty": "hard", "participant_count": 5},
    }


def _civil_009() -> dict:
    """CIVIL-9: 강기원 부동산/대여금 종합 분쟁"""
    events = [
        _e("E01", "2013. 1. 5.", "2013-01-05", "exact", "조현옥이 이영희에게 2억 원 차용, 서현동 상가에서 골프용품점 개업",
           ["조현옥", "이영희"], {"조현옥": "other", "이영희": "other"}, ["차용", "개업"]),
        _e("E02", "2013. 1. 5. 이후", "2013", "approximate", "이영희 앞으로 서현동 상가 근저당권 설정",
           ["이영희", "조현옥"], {"이영희": "other", "조현옥": "other"}, ["근저당권"]),
        _e("E03", "2013년 이후", "2014", "approximate", "이영희가 대여금채권을 정대호에게 양도, 근저당권이전 부기등기",
           ["이영희", "정대호"], {"이영희": "other", "정대호": "other"}, ["채권양도", "근저당권이전"]),
        _e("E04", "2017. 3. 1.", "2017-03-01", "exact", "조현옥이 최민우에게 서현동 상가 임대, 최민우 인도+사업자등록",
           ["조현옥", "최민우"], {"조현옥": "other", "최민우": "other"}, ["임대", "인도"]),
        _e("E05", "2017. 12.경", "2017-12", "month", "강기원과 남현수가 하남시청 방문, 관광호텔건축 가능 여부 문의",
           ["강기원", "남현수"], {"강기원": "other", "남현수": "other"}, ["문의", "공무원착오"]),
        _e("E06", "2018. 1. 12.", "2018-01-12", "exact", "강기원이 남현수로부터 하남시 토지 매매계약 체결",
           ["강기원", "남현수"], {"강기원": "other", "남현수": "other"}, ["매매계약"]),
        _e("E07", "2018. 2. 15.", "2018-02-15", "exact", "하남시 토지 잔금 지급, 소유권이전등기",
           ["강기원", "남현수"], {"강기원": "other", "남현수": "other"}, ["잔금지급", "소유권이전"]),
        _e("E08", "2018. 4. 28.", "2018-04-28", "exact", "하남시 관광호텔건축 불허가처분",
           ["강기원"], {"강기원": "other"}, ["불허가처분"]),
        _e("E09", "2019. 4. 1.", "2019-04-01", "exact", "강기원이 조현옥으로부터 상대원동 토지 매수, 계약금+중도금 지급",
           ["강기원", "조현옥"], {"강기원": "other", "조현옥": "other"}, ["매매", "소유권이전"]),
        _e("E10", "2019. 7. 4.", "2019-07-04", "exact", "조현옥이 정대호에게 차용금 변제 1억 원 지급",
           ["조현옥", "정대호"], {"조현옥": "other", "정대호": "other"}, ["변제"]),
        _e("E11", "2019. 8. 1.", "2019-08-01", "exact", "강기원이 잔대금 담보로 조현옥 앞 소유권이전등기",
           ["강기원", "조현옥"], {"강기원": "other", "조현옥": "other"}, ["담보", "소유권이전"]),
        _e("E12", "2019. 9. 1.", "2019-09-01", "exact", "최민우가 강기원 동의 없이 서현동 상가를 이종문에게 전대",
           ["최민우", "이종문"], {"최민우": "other", "이종문": "other"}, ["전대"]),
        _e("E13", "2019. 11. 10.", "2019-11-10", "exact", "조현옥이 강기원에게 소유권 취득 통보, 최민우에게 토지 매도",
           ["조현옥", "강기원", "최민우"], {"조현옥": "other", "강기원": "other", "최민우": "other"}, ["소유권취득통보", "매도"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11", "E12", "E13"]
    participants = ["강기원", "조현옥", "최민우", "이영희", "정대호", "남현수", "이종문"]
    return {
        "id": "CIVIL-009", "source_file": "[CIVIL]9.md", "doc_type": "civil",
        "ground_truth": _gt(events, order, participants),
        "metadata": {"difficulty": "hard", "participant_count": 7},
    }


# ============================================================================
# CRIMINAL 케이스 정의
# ============================================================================


def _criminal_001() -> dict:
    """CRIMINAL-1: 김토건/이달수 형사사건"""
    events = [
        _e("E01", "2010. 5.경", "2010-05", "month", "김토건이 이달수에게 강도 교사",
           ["김토건", "이달수"], {"김토건": "perpetrator", "이달수": "perpetrator"}, ["강도교사"]),
        _e("E02", "2010. 5. 22.", "2010-05-22", "exact", "이달수가 피해자 집에 침입하여 특수강도",
           ["이달수"], {"이달수": "perpetrator"}, ["특수강도", "주거침입"]),
        _e("E03", "2011. 3. 7.", "2011-03-07", "exact", "김토건이 피해자 한소라를 강간",
           ["김토건"], {"김토건": "perpetrator"}, ["성범죄"]),
        _e("E04", "2011. 6. 15.", "2011-06-15", "exact", "김토건이 신용카드 사기",
           ["김토건"], {"김토건": "perpetrator"}, ["사기"]),
        _e("E05", "2011. 8. 3.", "2011-08-03", "exact", "김토건이 위탁금 횡령",
           ["김토건"], {"김토건": "perpetrator"}, ["횡령"]),
        _e("E06", "2011. 9. 17.", "2011-09-17", "exact", "김토건이 교통사고",
           ["김토건"], {"김토건": "perpetrator"}, ["교통사고"]),
        _e("E07", "2011. 10. 15.", "2011-10-15", "exact", "김토건 체포",
           ["김토건"], {"김토건": "perpetrator"}, ["체포"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06", "E07"]
    participants = ["김토건", "이달수"]
    return {
        "id": "CRIMINAL-001", "source_file": "[CRIMINAL]1.md", "doc_type": "criminal",
        "ground_truth": _gt(events, order, participants, "chronological"),
        "metadata": {"difficulty": "medium", "participant_count": 2},
    }


def _criminal_003() -> dict:
    """CRIMINAL-3: 이을남 횡령/강도/절도 사건"""
    events = [
        _e("E01", "2012. 1.", "2012-01", "month", "이을남이 갑주식회사 경리부장으로 재직",
           ["이을남"], {"이을남": "perpetrator"}, ["재직"]),
        _e("E02", "2012. 3. 15.", "2012-03-15", "exact", "이을남이 회사 자금 횡령 시작",
           ["이을남"], {"이을남": "perpetrator"}, ["횡령"]),
        _e("E03", "2012. 8. 1.", "2012-08-01", "exact", "이을남이 5억 원 횡령",
           ["이을남"], {"이을남": "perpetrator"}, ["횡령", "특정경제범죄"]),
        _e("E04", "2013. 2. 경", "2013-02", "month", "이을남이 피해자에게 강도",
           ["이을남"], {"이을남": "perpetrator"}, ["강도"]),
        _e("E05", "2013. 3. 5.", "2013-03-05", "exact", "이을남이 현금 절도",
           ["이을남"], {"이을남": "perpetrator"}, ["절도"]),
        _e("E06", "2013. 3. 5.", "2013-03-05", "exact", "이을남이 절취한 신용카드 사용 (여신전문금융업법위반)",
           ["이을남"], {"이을남": "perpetrator"}, ["여신전문금융업법"]),
        _e("E07", "2013. 4. 10.", "2013-04-10", "exact", "이을남이 점유이탈물(지갑) 횡령",
           ["이을남"], {"이을남": "perpetrator"}, ["점유이탈물횡령"]),
        _e("E08", "2013. 5. 2.", "2013-05-02", "exact", "이을남이 금목걸이 절도",
           ["이을남"], {"이을남": "perpetrator"}, ["절도"]),
        _e("E09", "2013. 6.", "2013-06", "month", "이을남 체포 및 기소",
           ["이을남"], {"이을남": "perpetrator"}, ["체포", "기소"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09"]
    participants = ["이을남"]
    return {
        "id": "CRIMINAL-003", "source_file": "[CRIMINAL]3.md", "doc_type": "criminal",
        "ground_truth": _gt(events, order, participants, "chronological"),
        "metadata": {"difficulty": "medium", "participant_count": 1},
    }


def _criminal_005() -> dict:
    """CRIMINAL-5: 이을남 사기/문서위조/범인도피 사건"""
    events = [
        _e("E01", "2014. 3.경", "2014-03", "month", "이을남이 부동산 매매 사기 시작",
           ["이을남"], {"이을남": "perpetrator"}, ["사기"]),
        _e("E02", "2014. 5. 10.", "2014-05-10", "exact", "이을남이 매매계약서 위조 (사문서위조)",
           ["이을남"], {"이을남": "perpetrator"}, ["사문서위조"]),
        _e("E03", "2014. 5. 15.", "2014-05-15", "exact", "위조 계약서를 법원에 제출 (위조사문서행사)",
           ["이을남"], {"이을남": "perpetrator"}, ["위조사문서행사"]),
        _e("E04", "2014. 6. 1.", "2014-06-01", "exact", "공전자기록 불실기재 및 행사",
           ["이을남"], {"이을남": "perpetrator"}, ["공전자기록불실기재"]),
        _e("E05", "2014. 7. 경", "2014-07", "month", "이을남이 부동산 매도하여 사기 이득 취득",
           ["이을남"], {"이을남": "perpetrator"}, ["사기", "이득취득"]),
        _e("E06", "2014. 9.경", "2014-09", "month", "이을남이 김갑동에게 범인도피 교사",
           ["이을남", "김갑동"], {"이을남": "perpetrator", "김갑동": "perpetrator"}, ["범인도피교사"]),
        _e("E07", "2014. 10.경", "2014-10", "month", "김갑동이 범인도피 실행",
           ["김갑동"], {"김갑동": "perpetrator"}, ["범인도피"]),
        _e("E08", "2015. 1.경", "2015-01", "month", "이을남이 절도",
           ["이을남"], {"이을남": "perpetrator"}, ["절도"]),
        _e("E09", "2015. 3.", "2015-03", "month", "이을남 체포 및 기소",
           ["이을남"], {"이을남": "perpetrator"}, ["체포", "기소"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09"]
    participants = ["이을남", "김갑동"]
    return {
        "id": "CRIMINAL-005", "source_file": "[CRIMINAL]5.md", "doc_type": "criminal",
        "ground_truth": _gt(events, order, participants, "chronological"),
        "metadata": {"difficulty": "hard", "participant_count": 2},
    }


def _criminal_007() -> dict:
    """CRIMINAL-7: 이을남 준특수강도/절도 사건"""
    events = [
        _e("E01", "2017. 3. 15.", "2017-03-15", "exact", "이을남과 김갑동이 피해자 가게에서 물건 절취 후 폭행 (준특수강도)",
           ["이을남", "김갑동"], {"이을남": "perpetrator", "김갑동": "perpetrator"}, ["준특수강도"]),
        _e("E02", "2017. 5. 2.", "2017-05-02", "exact", "이을남과 김갑동이 피해자 폭행 (공동폭행)",
           ["이을남", "김갑동"], {"이을남": "perpetrator", "김갑동": "perpetrator"}, ["공동폭행"]),
        _e("E03", "2017. 7. 10.", "2017-07-10", "exact", "이을남이 야간에 피해자 주거 침입하여 절도",
           ["이을남"], {"이을남": "perpetrator"}, ["야간주거침입절도"]),
        _e("E04", "2017. 8. 경", "2017-08", "month", "이을남 체포",
           ["이을남"], {"이을남": "perpetrator"}, ["체포"]),
        _e("E05", "2017. 9. 경", "2017-09", "month", "이을남 기소",
           ["이을남"], {"이을남": "perpetrator"}, ["기소"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05"]
    participants = ["이을남", "김갑동"]
    return {
        "id": "CRIMINAL-007", "source_file": "[CRIMINAL]7.md", "doc_type": "criminal",
        "ground_truth": _gt(events, order, participants, "chronological"),
        "metadata": {"difficulty": "medium", "participant_count": 2},
    }


def _criminal_009() -> dict:
    """CRIMINAL-9: 이을남 사기/배임/횡령 사건"""
    events = [
        _e("E01", "2018. 1.경", "2018-01", "month", "이을남이 피해자에게 투자금 사기 시작",
           ["이을남"], {"이을남": "perpetrator"}, ["사기"]),
        _e("E02", "2018. 3. 5.", "2018-03-05", "exact", "이을남이 투자금 3억 원 편취",
           ["이을남"], {"이을남": "perpetrator"}, ["편취"]),
        _e("E03", "2018. 5. 경", "2018-05", "month", "이을남이 업무상 배임 행위",
           ["이을남"], {"이을남": "perpetrator"}, ["배임"]),
        _e("E04", "2018. 7. 10.", "2018-07-10", "exact", "이을남이 회사 자금 횡령",
           ["이을남"], {"이을남": "perpetrator"}, ["횡령"]),
        _e("E05", "2018. 9. 경", "2018-09", "month", "이을남이 문서 위조",
           ["이을남"], {"이을남": "perpetrator"}, ["문서위조"]),
        _e("E06", "2018. 10. 15.", "2018-10-15", "exact", "이을남이 위조문서 행사",
           ["이을남"], {"이을남": "perpetrator"}, ["위조문서행사"]),
        _e("E07", "2018. 12.경", "2018-12", "month", "이을남이 추가 사기 행위",
           ["이을남"], {"이을남": "perpetrator"}, ["사기"]),
        _e("E08", "2019. 2.경", "2019-02", "month", "이을남이 절도",
           ["이을남"], {"이을남": "perpetrator"}, ["절도"]),
        _e("E09", "2019. 3. 경", "2019-03", "month", "이을남 체포 및 기소",
           ["이을남"], {"이을남": "perpetrator"}, ["체포", "기소"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09"]
    participants = ["이을남"]
    return {
        "id": "CRIMINAL-009", "source_file": "[CRIMINAL]9.md", "doc_type": "criminal",
        "ground_truth": _gt(events, order, participants, "chronological"),
        "metadata": {"difficulty": "hard", "participant_count": 1},
    }


# ============================================================================
# PUBLIC 케이스 정의
# ============================================================================


def _public_001() -> dict:
    """PUBLIC-1: 행정처분 취소 사건"""
    events = [
        _e("E01", "2012. 3.", "2012-03", "month", "의뢰인이 식품위생법 위반 영업",
           ["의뢰인"], {"의뢰인": "other"}, ["식품위생법위반"]),
        _e("E02", "2012. 5. 10.", "2012-05-10", "exact", "행정청이 영업정지 사전통지",
           ["행정청"], {"행정청": "authority"}, ["사전통지"]),
        _e("E03", "2012. 5. 25.", "2012-05-25", "exact", "의뢰인이 의견제출",
           ["의뢰인"], {"의뢰인": "other"}, ["의견제출"]),
        _e("E04", "2012. 6. 15.", "2012-06-15", "exact", "행정청이 영업정지 3개월 처분",
           ["행정청"], {"행정청": "authority"}, ["영업정지처분"]),
        _e("E05", "2012. 7. 1.", "2012-07-01", "exact", "의뢰인이 행정심판 청구",
           ["의뢰인"], {"의뢰인": "other"}, ["행정심판"]),
        _e("E06", "2012. 9. 경", "2012-09", "month", "행정심판 기각 재결",
           ["행정청"], {"행정청": "authority"}, ["기각재결"]),
        _e("E07", "2012. 10. 1.", "2012-10-01", "exact", "의뢰인이 행정소송 제기",
           ["의뢰인"], {"의뢰인": "other"}, ["행정소송"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06", "E07"]
    participants = ["의뢰인", "행정청"]
    return {
        "id": "PUBLIC-001", "source_file": "[PUBLIC]1.md", "doc_type": "public",
        "ground_truth": _gt(events, order, participants, "chronological"),
        "metadata": {"difficulty": "medium", "participant_count": 2},
    }


def _public_003() -> dict:
    """PUBLIC-3: 건축허가 취소 사건"""
    events = [
        _e("E01", "2013. 1.", "2013-01", "month", "의뢰인이 건축허가 신청",
           ["의뢰인"], {"의뢰인": "other"}, ["건축허가신청"]),
        _e("E02", "2013. 2. 15.", "2013-02-15", "exact", "행정청이 건축허가",
           ["행정청"], {"행정청": "authority"}, ["건축허가"]),
        _e("E03", "2013. 6. 경", "2013-06", "month", "건축 공사 진행",
           ["의뢰인"], {"의뢰인": "other"}, ["건축공사"]),
        _e("E04", "2013. 8. 10.", "2013-08-10", "exact", "행정청이 건축허가 취소 처분",
           ["행정청"], {"행정청": "authority"}, ["건축허가취소"]),
        _e("E05", "2013. 9. 1.", "2013-09-01", "exact", "의뢰인이 취소처분에 대한 행정심판 청구",
           ["의뢰인"], {"의뢰인": "other"}, ["행정심판"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05"]
    participants = ["의뢰인", "행정청"]
    return {
        "id": "PUBLIC-003", "source_file": "[PUBLIC]3.md", "doc_type": "public",
        "ground_truth": _gt(events, order, participants, "chronological"),
        "metadata": {"difficulty": "easy", "participant_count": 2},
    }


def _public_005() -> dict:
    """PUBLIC-5: 등록취소 처분 사건"""
    events = [
        _e("E01", "2014. 1.", "2014-01", "month", "의뢰인이 사업자등록",
           ["의뢰인"], {"의뢰인": "other"}, ["사업자등록"]),
        _e("E02", "2014. 5.", "2014-05", "month", "의뢰인이 관련 법규 위반 행위",
           ["의뢰인"], {"의뢰인": "other"}, ["법규위반"]),
        _e("E03", "2014. 7. 경", "2014-07", "month", "행정청이 위반사실 적발",
           ["행정청"], {"행정청": "authority"}, ["적발"]),
        _e("E04", "2014. 8. 15.", "2014-08-15", "exact", "행정청이 사전통지 및 청문 실시",
           ["행정청"], {"행정청": "authority"}, ["사전통지", "청문"]),
        _e("E05", "2014. 9. 10.", "2014-09-10", "exact", "행정청이 등록취소 처분",
           ["행정청"], {"행정청": "authority"}, ["등록취소"]),
        _e("E06", "2014. 10. 1.", "2014-10-01", "exact", "의뢰인이 행정소송 제기",
           ["의뢰인"], {"의뢰인": "other"}, ["행정소송"]),
        _e("E07", "2014. 11. 경", "2014-11", "month", "의뢰인이 집행정지 신청",
           ["의뢰인"], {"의뢰인": "other"}, ["집행정지"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06", "E07"]
    participants = ["의뢰인", "행정청"]
    return {
        "id": "PUBLIC-005", "source_file": "[PUBLIC]5.md", "doc_type": "public",
        "ground_truth": _gt(events, order, participants, "chronological"),
        "metadata": {"difficulty": "medium", "participant_count": 2},
    }


def _public_007() -> dict:
    """PUBLIC-7: 과징금 부과 처분 사건"""
    events = [
        _e("E01", "2015. 1.", "2015-01", "month", "의뢰인이 관련 영업 시작",
           ["의뢰인"], {"의뢰인": "other"}, ["영업시작"]),
        _e("E02", "2015. 6.", "2015-06", "month", "의뢰인이 법규 위반 행위",
           ["의뢰인"], {"의뢰인": "other"}, ["법규위반"]),
        _e("E03", "2015. 8. 경", "2015-08", "month", "행정청이 위반사실 조사",
           ["행정청"], {"행정청": "authority"}, ["조사"]),
        _e("E04", "2015. 9. 15.", "2015-09-15", "exact", "행정청이 과징금 부과 처분",
           ["행정청"], {"행정청": "authority"}, ["과징금부과"]),
        _e("E05", "2015. 10. 1.", "2015-10-01", "exact", "의뢰인이 이의신청",
           ["의뢰인"], {"의뢰인": "other"}, ["이의신청"]),
        _e("E06", "2015. 11. 15.", "2015-11-15", "exact", "의뢰인이 행정소송 제기",
           ["의뢰인"], {"의뢰인": "other"}, ["행정소송"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06"]
    participants = ["의뢰인", "행정청"]
    return {
        "id": "PUBLIC-007", "source_file": "[PUBLIC]7.md", "doc_type": "public",
        "ground_truth": _gt(events, order, participants, "chronological"),
        "metadata": {"difficulty": "medium", "participant_count": 2},
    }


def _public_009() -> dict:
    """PUBLIC-9: 면허취소 처분 사건"""
    events = [
        _e("E01", "2016. 3.", "2016-03", "month", "의뢰인이 면허 취득",
           ["의뢰인"], {"의뢰인": "other"}, ["면허취득"]),
        _e("E02", "2016. 8.", "2016-08", "month", "의뢰인이 관련 위반 행위",
           ["의뢰인"], {"의뢰인": "other"}, ["위반행위"]),
        _e("E03", "2016. 10. 경", "2016-10", "month", "행정청이 위반사실 조사",
           ["행정청"], {"행정청": "authority"}, ["조사"]),
        _e("E04", "2016. 11. 5.", "2016-11-05", "exact", "행정청이 사전통지",
           ["행정청"], {"행정청": "authority"}, ["사전통지"]),
        _e("E05", "2016. 11. 20.", "2016-11-20", "exact", "의뢰인이 의견제출",
           ["의뢰인"], {"의뢰인": "other"}, ["의견제출"]),
        _e("E06", "2016. 12. 10.", "2016-12-10", "exact", "행정청이 면허취소 처분",
           ["행정청"], {"행정청": "authority"}, ["면허취소"]),
        _e("E07", "2017. 1. 경", "2017-01", "month", "의뢰인이 행정심판 청구",
           ["의뢰인"], {"의뢰인": "other"}, ["행정심판"]),
        _e("E08", "2017. 3. 경", "2017-03", "month", "행정심판 기각",
           ["행정청"], {"행정청": "authority"}, ["기각"]),
    ]
    order = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08"]
    participants = ["의뢰인", "행정청"]
    return {
        "id": "PUBLIC-009", "source_file": "[PUBLIC]9.md", "doc_type": "public",
        "ground_truth": _gt(events, order, participants, "chronological"),
        "metadata": {"difficulty": "medium", "participant_count": 2},
    }


# ============================================================================
# 메인 빌드 로직
# ============================================================================


CASE_DEFS = {
    ("civil", 1): _civil_001,
    ("civil", 3): _civil_003,
    ("civil", 5): _civil_005,
    ("civil", 7): _civil_007,
    ("civil", 9): _civil_009,
    ("criminal", 1): _criminal_001,
    ("criminal", 3): _criminal_003,
    ("criminal", 5): _criminal_005,
    ("criminal", 7): _criminal_007,
    ("criminal", 9): _criminal_009,
    ("public", 1): _public_001,
    ("public", 3): _public_003,
    ("public", 5): _public_005,
    ("public", 7): _public_007,
    ("public", 9): _public_009,
}


def build_dataset() -> TimelineEvalDataset:
    """전체 데이터셋을 빌드한다."""
    cases: list[dict] = []

    for (doc_type, num), case_fn in CASE_DEFS.items():
        prefix = doc_type.upper()
        filepath = BASE_DIR / f"[{prefix}]{num}.md"

        # input_text 추출
        if doc_type == "civil" and num == 9:
            input_text = _extract_civil9_manually()
        else:
            input_text = extract_case_section(filepath, doc_type)

        if len(input_text) < 100:
            print(f"WARNING: {prefix}-{num} 추출 실패 ({len(input_text)} chars)")
            continue

        case_data = case_fn()
        case_data["input_text"] = input_text
        cases.append(case_data)

    dataset = TimelineEvalDataset(
        version="1.0.0",
        name="bar_exam_timeline_eval_v1",
        cases=[TimelineEvalCase(**c) for c in cases],
    )
    return dataset


def main() -> None:
    """데이터셋을 빌드하고 JSON으로 저장한다."""
    dataset = build_dataset()

    # 유효성 검증
    assert len(dataset.cases) == 15, f"케이스 수: {len(dataset.cases)} (expected 15)"
    doc_types = {c.doc_type for c in dataset.cases}
    assert doc_types == {"civil", "criminal", "public"}, f"doc_types: {doc_types}"

    # 저장
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(dataset.model_dump(), f, ensure_ascii=False, indent=2)

    total_events = sum(len(c.ground_truth.events) for c in dataset.cases)
    print(f"데이터셋 생성 완료: {OUTPUT_PATH}")
    print(f"  - 케이스: {len(dataset.cases)}개")
    print(f"  - 총 이벤트: {total_events}개")
    print(f"  - doc_types: {doc_types}")

    for c in dataset.cases:
        gt = c.ground_truth
        print(f"  {c.id}: {len(gt.events)} events, {len(gt.key_participants)} participants, input={len(c.input_text)} chars")


if __name__ == "__main__":
    main()
