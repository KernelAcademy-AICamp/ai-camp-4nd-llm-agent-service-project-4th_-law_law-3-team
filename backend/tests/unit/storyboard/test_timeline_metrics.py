"""스토리보드 타임라인 메트릭 단위 테스트"""

import pytest

from app.modules.storyboard.evaluation.metrics import (
    chronological_order_score,
    composite_score,
    event_count_accuracy,
    evaluate_case,
    key_event_coverage,
    match_events,
    participant_recall,
    role_accuracy,
)
from app.modules.storyboard.evaluation.schemas import DatePrecision, TimelineEventGT


# --- 헬퍼 ---


def _make_gt_event(
    event_id: str,
    title: str,
    date_normalized: str | None = "2012-01-05",
    date_raw: str = "2012. 1. 5.",
    participants: list[str] | None = None,
    roles: dict[str, str] | None = None,
    is_key: bool = True,
) -> TimelineEventGT:
    return TimelineEventGT(
        event_id=event_id,
        date_raw=date_raw,
        date_normalized=date_normalized,
        date_precision=DatePrecision.EXACT,
        title=title,
        key_participants=participants or [],
        participant_roles=roles or {},
        is_key_event=is_key,
    )


def _make_pred_event(title: str, date: str = "2012-01-05") -> dict[str, str]:
    return {"date": date, "title": title}


# === event_count_accuracy ===


class TestEventCountAccuracy:
    def test_perfect_match(self) -> None:
        assert event_count_accuracy(10, 10) == 1.0

    def test_partial_match(self) -> None:
        result = event_count_accuracy(7, 10)
        assert abs(result - 0.7) < 1e-6

    def test_over_prediction(self) -> None:
        result = event_count_accuracy(15, 10)
        assert abs(result - 0.5) < 1e-6

    def test_zero_gt(self) -> None:
        assert event_count_accuracy(0, 0) == 1.0
        assert event_count_accuracy(5, 0) == 0.0

    def test_clipping_at_zero(self) -> None:
        """예측이 GT의 2배 이상이면 0으로 클리핑"""
        result = event_count_accuracy(30, 10)
        assert result == 0.0


# === chronological_order_score (Kendall's Tau) ===


class TestChronologicalOrderScore:
    def test_perfect_order(self) -> None:
        gt = ["E01", "E02", "E03", "E04"]
        pred = ["E01", "E02", "E03", "E04"]
        assert chronological_order_score(pred, gt) == 1.0

    def test_reverse_order(self) -> None:
        gt = ["E01", "E02", "E03", "E04"]
        pred = ["E04", "E03", "E02", "E01"]
        assert chronological_order_score(pred, gt) == 0.0

    def test_partial_disorder(self) -> None:
        gt = ["E01", "E02", "E03", "E04"]
        pred = ["E01", "E03", "E02", "E04"]  # 1쌍 역전
        result = chronological_order_score(pred, gt)
        # tau = (5-1)/6 = 2/3, score = (2/3 + 1)/2 ≈ 0.833
        assert 0.8 < result < 0.9

    def test_single_element(self) -> None:
        assert chronological_order_score(["E01"], ["E01"]) == 1.0

    def test_empty(self) -> None:
        assert chronological_order_score([], ["E01", "E02"]) == 0.0

    def test_partial_common(self) -> None:
        """예측에 GT의 일부만 있는 경우"""
        gt = ["E01", "E02", "E03", "E04"]
        pred = ["E01", "E03"]  # E02, E04 빠짐 — 순서는 맞음
        assert chronological_order_score(pred, gt) == 1.0

    def test_no_common(self) -> None:
        gt = ["E01", "E02"]
        pred = ["E99", "E98"]
        assert chronological_order_score(pred, gt) == 0.0


# === participant_recall ===


class TestParticipantRecall:
    def test_perfect_recall(self) -> None:
        gt = ["박대원", "박진수", "김영철"]
        pred = ["박대원", "박진수", "김영철"]
        assert participant_recall(pred, gt) == 1.0

    def test_partial_recall(self) -> None:
        gt = ["박대원", "박진수", "김영철"]
        pred = ["박대원"]
        result = participant_recall(pred, gt)
        assert abs(result - 1 / 3) < 1e-6

    def test_substring_match(self) -> None:
        """부분 문자열 매칭 지원"""
        gt = ["박대원"]
        pred = ["의뢰인 박대원"]
        assert participant_recall(pred, gt) == 1.0

    def test_empty_gt(self) -> None:
        assert participant_recall(["박대원"], []) == 1.0

    def test_empty_pred(self) -> None:
        assert participant_recall([], ["박대원"]) == 0.0


# === role_accuracy ===


class TestRoleAccuracy:
    def test_perfect_accuracy(self) -> None:
        gt = {"박대원": "victim", "박진수": "perpetrator"}
        pred = {"박대원": "victim", "박진수": "perpetrator"}
        assert role_accuracy(pred, gt) == 1.0

    def test_partial_accuracy(self) -> None:
        gt = {"박대원": "victim", "박진수": "perpetrator"}
        pred = {"박대원": "victim", "박진수": "other"}
        assert role_accuracy(pred, gt) == 0.5

    def test_no_matching_participants(self) -> None:
        gt = {"박대원": "victim"}
        pred = {"김영철": "other"}
        assert role_accuracy(pred, gt) == 0.0

    def test_empty_gt(self) -> None:
        assert role_accuracy({"박대원": "victim"}, {}) == 1.0

    def test_substring_match_roles(self) -> None:
        gt = {"박대원": "victim"}
        pred = {"의뢰인 박대원": "victim"}
        assert role_accuracy(pred, gt) == 1.0


# === match_events ===


class TestMatchEvents:
    def test_exact_match(self) -> None:
        gt = [_make_gt_event("E01", "서류 위조 등기", "2001-03-05")]
        pred = [_make_pred_event("서류 위조 등기 이전", "2001-03-05")]
        pairs = match_events(gt, pred)
        assert len(pairs) == 1
        assert pairs[0][1] is not None

    def test_partial_keyword_match(self) -> None:
        gt = [_make_gt_event("E01", "박진수 서류 위조", "2001-03-05")]
        pred = [_make_pred_event("박진수 문서 위조", "2001-03-05")]
        pairs = match_events(gt, pred)
        # "박진수"와 "위조" 2개 키워드 일치 + 날짜 일치
        assert pairs[0][1] is not None

    def test_no_match(self) -> None:
        gt = [_make_gt_event("E01", "계약 체결", "2012-01-05")]
        pred = [_make_pred_event("형사 고소", "2007-09-01")]
        pairs = match_events(gt, pred)
        assert pairs[0][1] is None

    def test_empty_pred(self) -> None:
        gt = [_make_gt_event("E01", "계약 체결")]
        pairs = match_events(gt, [])
        assert len(pairs) == 1
        assert pairs[0][1] is None

    def test_one_to_one_matching(self) -> None:
        """하나의 예측이 여러 GT에 중복 매칭되지 않아야 함"""
        gt = [
            _make_gt_event("E01", "서류 위조 접수", "2001-03-05"),
            _make_gt_event("E02", "서류 위조 판결", "2001-03-05"),
        ]
        pred = [_make_pred_event("서류 위조 접수 처리", "2001-03-05")]
        pairs = match_events(gt, pred)
        matched = [p for _, p in pairs if p is not None]
        assert len(matched) == 1  # 한 예측은 한 GT에만 매칭


# === key_event_coverage ===


class TestKeyEventCoverage:
    def test_all_key_events_matched(self) -> None:
        gt1 = _make_gt_event("E01", "계약 체결", is_key=True)
        gt2 = _make_gt_event("E02", "부가 설명", is_key=False)
        pairs = [(gt1, {"date": "2012-01-05", "title": "계약"}), (gt2, None)]
        assert key_event_coverage(pairs) == 1.0

    def test_no_key_events_matched(self) -> None:
        gt1 = _make_gt_event("E01", "계약 체결", is_key=True)
        pairs = [(gt1, None)]
        assert key_event_coverage(pairs) == 0.0

    def test_partial_coverage(self) -> None:
        gt1 = _make_gt_event("E01", "이벤트1", is_key=True)
        gt2 = _make_gt_event("E02", "이벤트2", is_key=True)
        pairs = [
            (gt1, {"date": "2012-01", "title": "이벤트1"}),
            (gt2, None),
        ]
        assert key_event_coverage(pairs) == 0.5

    def test_no_key_events_in_gt(self) -> None:
        gt1 = _make_gt_event("E01", "부가", is_key=False)
        pairs = [(gt1, None)]
        assert key_event_coverage(pairs) == 1.0


# === composite_score ===


class TestCompositeScore:
    def test_perfect_score(self) -> None:
        assert abs(composite_score(1.0, 1.0, 1.0, 1.0, 1.0) - 1.0) < 1e-9

    def test_zero_score(self) -> None:
        assert composite_score(0.0, 0.0, 0.0, 0.0, 0.0) == 0.0

    def test_weighted_sum(self) -> None:
        # 순서 0.25 + 핵심이벤트 0.25 + 이벤트수 0.20 + 참여자 0.20 + 역할 0.10
        result = composite_score(
            event_count_acc=0.5,
            chrono_order=0.8,
            part_recall=0.6,
            role_acc=1.0,
            key_coverage=0.7,
        )
        expected = 0.25 * 0.8 + 0.25 * 0.7 + 0.20 * 0.5 + 0.20 * 0.6 + 0.10 * 1.0
        assert abs(result - expected) < 1e-6


# === evaluate_case (통합) ===


class TestEvaluateCase:
    def test_basic_evaluation(self) -> None:
        gt_events = [
            _make_gt_event("E01", "서류 위조 등기", "2001-03-05"),
            _make_gt_event("E02", "형사 고소 제기", "2007-09-01"),
        ]
        gt_order = ["E01", "E02"]
        gt_participants = ["박대원", "박진수"]
        gt_roles = {"박대원": "victim", "박진수": "perpetrator"}

        pred_events = [
            _make_pred_event("서류 위조 등기 이전", "2001-03-05"),
            _make_pred_event("형사 고소 제기", "2007-09-01"),
        ]
        pred_participants = ["박대원", "박진수"]
        pred_roles = {"박대원": "victim", "박진수": "perpetrator"}

        result = evaluate_case(
            gt_events=gt_events,
            gt_order=gt_order,
            gt_participants=gt_participants,
            pred_events=pred_events,
            pred_participants=pred_participants,
            pred_roles=pred_roles,
            gt_roles=gt_roles,
            case_id="TEST-001",
        )

        assert result.case_id == "TEST-001"
        assert result.event_count_accuracy == 1.0
        assert result.participant_recall == 1.0
        assert result.role_accuracy == 1.0
        assert result.composite_score > 0.8
