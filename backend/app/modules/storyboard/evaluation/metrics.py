"""스토리보드 타임라인 추출 평가 메트릭

기존 backend/evaluation/metrics/retrieval.py 스타일을 따름.
각 함수는 0.0~1.0 범위의 float를 반환한다.
"""

from __future__ import annotations

from .schemas import TimelineCaseResult, TimelineEventGT

# --- 이벤트 매칭 ---


def _normalize_for_matching(text: str) -> str:
    """매칭용 텍스트 정규화 (공백·마침표 제거, 소문자)"""
    return text.replace(" ", "").replace(".", "").replace(",", "").lower()


def _extract_keywords(title: str) -> set[str]:
    """제목에서 키워드 추출 (2자 이상 단어)"""
    words = title.replace(",", " ").replace(".", " ").split()
    return {w for w in words if len(w) >= 2}


def match_events(
    gt_events: list[TimelineEventGT],
    pred_events: list[dict[str, str]],
) -> list[tuple[TimelineEventGT, dict[str, str] | None]]:
    """GT 이벤트와 예측 이벤트를 매칭한다.

    매칭 로직:
    1. 날짜(정규화된 값)가 동일하고 제목 키워드 2개 이상 일치 → 매칭
    2. 날짜만 일치하고 제목 키워드 1개 이상 일치 → 약한 매칭 (후순위)
    3. 날짜 없이 제목 키워드 3개 이상 일치 → 대안 매칭

    Args:
        gt_events: 정답 이벤트 목록
        pred_events: 예측 이벤트 목록 (각 dict에 "date", "title" 키 필요)

    Returns:
        (GT 이벤트, 매칭된 예측 이벤트 또는 None) 쌍 목록
    """
    used_pred_indices: set[int] = set()
    result: list[tuple[TimelineEventGT, dict[str, str] | None]] = []

    for gt in gt_events:
        best_match: dict[str, str] | None = None
        best_match_idx: int = -1
        best_score: int = 0

        gt_date_norm = _normalize_for_matching(gt.date_normalized or "")
        gt_keywords = _extract_keywords(gt.title)

        for idx, pred in enumerate(pred_events):
            if idx in used_pred_indices:
                continue

            pred_date = _normalize_for_matching(pred.get("date", ""))
            pred_keywords = _extract_keywords(pred.get("title", ""))

            keyword_overlap = len(gt_keywords & pred_keywords)
            date_match = (
                gt_date_norm != ""
                and pred_date != ""
                and (gt_date_norm in pred_date or pred_date in gt_date_norm)
            )

            score = 0
            if date_match and keyword_overlap >= 2:
                score = 100 + keyword_overlap
            elif date_match and keyword_overlap >= 1:
                score = 50 + keyword_overlap
            elif keyword_overlap >= 3:
                score = 10 + keyword_overlap

            if score > best_score:
                best_score = score
                best_match = pred
                best_match_idx = idx

        if best_match is not None:
            used_pred_indices.add(best_match_idx)
        result.append((gt, best_match))

    return result


# --- 개별 메트릭 ---


def event_count_accuracy(predicted_count: int, gt_count: int) -> float:
    """이벤트 수 정확도를 계산한다.

    1 - |predicted - gt| / gt. 0 미만이면 0으로 클리핑.

    Args:
        predicted_count: 예측된 이벤트 수
        gt_count: 정답 이벤트 수

    Returns:
        0.0~1.0 범위의 정확도
    """
    if gt_count == 0:
        return 1.0 if predicted_count == 0 else 0.0
    return max(0.0, 1.0 - abs(predicted_count - gt_count) / gt_count)


def chronological_order_score(
    predicted_order: list[str],
    gt_order: list[str],
) -> float:
    """Kendall's Tau로 시간 순서 점수를 계산한다.

    GT 순서에서 나타나는 예측 이벤트만 비교한다.
    scipy 없이 직접 구현.

    Args:
        predicted_order: 예측된 이벤트 ID 순서 목록
        gt_order: 정답 시간순 이벤트 ID 목록

    Returns:
        0.0~1.0 범위의 순서 점수 (1.0 = 완벽히 일치)
    """
    # GT에 있는 이벤트만 필터링
    common = [eid for eid in predicted_order if eid in gt_order]
    if len(common) <= 1:
        return 1.0 if len(common) == len(gt_order) else 0.0

    # GT 순서에서의 인덱스 매핑
    gt_index = {eid: i for i, eid in enumerate(gt_order)}
    ranks = [gt_index[eid] for eid in common]

    # Kendall's Tau 계산: concordant - discordant 쌍
    n = len(ranks)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            if ranks[i] < ranks[j]:
                concordant += 1
            elif ranks[i] > ranks[j]:
                discordant += 1
            # 동일하면 무시

    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 1.0

    # tau를 0~1 범위로 변환: (tau + 1) / 2
    tau = (concordant - discordant) / total_pairs
    return (tau + 1.0) / 2.0


def participant_recall(
    predicted_participants: list[str],
    gt_participants: list[str],
) -> float:
    """참여자 재현율을 계산한다.

    GT 참여자 중 예측에서 찾을 수 있는 비율. 부분 문자열 매칭 지원.

    Args:
        predicted_participants: 예측된 참여자 이름 목록
        gt_participants: 정답 참여자 이름 목록

    Returns:
        0.0~1.0 범위의 재현율
    """
    if not gt_participants:
        return 1.0

    pred_normalized = [_normalize_for_matching(p) for p in predicted_participants]
    matched = 0

    for gt_name in gt_participants:
        gt_norm = _normalize_for_matching(gt_name)
        for pred_norm in pred_normalized:
            if gt_norm in pred_norm or pred_norm in gt_norm:
                matched += 1
                break

    return matched / len(gt_participants)


def role_accuracy(
    predicted_roles: dict[str, str],
    gt_roles: dict[str, str],
) -> float:
    """매칭된 참여자 중 역할 일치 비율을 계산한다.

    Args:
        predicted_roles: 예측된 참여자-역할 매핑
        gt_roles: 정답 참여자-역할 매핑

    Returns:
        0.0~1.0 범위의 역할 정확도
    """
    if not gt_roles:
        return 1.0

    matched_count = 0
    total_compared = 0

    gt_norm_roles = {
        _normalize_for_matching(name): role for name, role in gt_roles.items()
    }
    pred_norm_roles = {
        _normalize_for_matching(name): role for name, role in predicted_roles.items()
    }

    for gt_name_norm, gt_role in gt_norm_roles.items():
        # 부분 문자열 매칭으로 예측 참여자 찾기
        for pred_name_norm, pred_role in pred_norm_roles.items():
            if gt_name_norm in pred_name_norm or pred_name_norm in gt_name_norm:
                total_compared += 1
                if gt_role == pred_role:
                    matched_count += 1
                break

    if total_compared == 0:
        return 0.0

    return matched_count / total_compared


def key_event_coverage(
    matched_pairs: list[tuple[TimelineEventGT, dict[str, str] | None]],
) -> float:
    """핵심 이벤트 포함율을 계산한다.

    GT에서 is_key_event=True인 이벤트 중 매칭된 비율.

    Args:
        matched_pairs: match_events()의 결과

    Returns:
        0.0~1.0 범위의 포함율
    """
    key_events = [(gt, pred) for gt, pred in matched_pairs if gt.is_key_event]
    if not key_events:
        return 1.0

    matched = sum(1 for _, pred in key_events if pred is not None)
    return matched / len(key_events)


# --- 종합 점수 ---

# 가중치
_WEIGHTS = {
    "chronological_order": 0.25,
    "key_event_coverage": 0.25,
    "event_count_accuracy": 0.20,
    "participant_recall": 0.20,
    "role_accuracy": 0.10,
}


def composite_score(
    event_count_acc: float,
    chrono_order: float,
    part_recall: float,
    role_acc: float,
    key_coverage: float,
) -> float:
    """종합 점수를 계산한다.

    가중합: 순서 0.25 + 핵심이벤트 0.25 + 이벤트수 0.20 + 참여자 0.20 + 역할 0.10

    Args:
        event_count_acc: 이벤트 수 정확도
        chrono_order: 시간 순서 점수
        part_recall: 참여자 재현율
        role_acc: 역할 정확도
        key_coverage: 핵심 이벤트 포함율

    Returns:
        0.0~1.0 범위의 종합 점수
    """
    return (
        _WEIGHTS["chronological_order"] * chrono_order
        + _WEIGHTS["key_event_coverage"] * key_coverage
        + _WEIGHTS["event_count_accuracy"] * event_count_acc
        + _WEIGHTS["participant_recall"] * part_recall
        + _WEIGHTS["role_accuracy"] * role_acc
    )


def evaluate_case(
    gt_events: list[TimelineEventGT],
    gt_order: list[str],
    gt_participants: list[str],
    pred_events: list[dict[str, str]],
    pred_participants: list[str],
    pred_roles: dict[str, str],
    gt_roles: dict[str, str],
    case_id: str = "",
) -> TimelineCaseResult:
    """단일 케이스를 평가하여 결과를 반환한다.

    Args:
        gt_events: 정답 이벤트 목록
        gt_order: 정답 시간순 이벤트 ID 목록
        gt_participants: 정답 참여자 목록
        pred_events: 예측 이벤트 목록 (각 dict에 date, title 키)
        pred_participants: 예측 참여자 목록
        pred_roles: 예측 참여자-역할 매핑
        gt_roles: 정답 참여자-역할 매핑
        case_id: 케이스 ID

    Returns:
        TimelineCaseResult
    """
    matched_pairs = match_events(gt_events, pred_events)

    # 매칭된 이벤트 ID 순서 추출 (예측 순서 기준)
    pred_matched_ids: list[str] = []
    for gt_evt, pred in matched_pairs:
        if pred is not None:
            pred_matched_ids.append(gt_evt.event_id)

    evt_count_acc = event_count_accuracy(len(pred_events), len(gt_events))
    chrono_score = chronological_order_score(pred_matched_ids, gt_order)
    part_recall = participant_recall(pred_participants, gt_participants)
    role_acc = role_accuracy(pred_roles, gt_roles)
    key_coverage = key_event_coverage(matched_pairs)
    comp = composite_score(evt_count_acc, chrono_score, part_recall, role_acc, key_coverage)

    return TimelineCaseResult(
        case_id=case_id,
        event_count_accuracy=round(evt_count_acc, 4),
        chronological_order_score=round(chrono_score, 4),
        participant_recall=round(part_recall, 4),
        role_accuracy=round(role_acc, 4),
        key_event_coverage=round(key_coverage, 4),
        composite_score=round(comp, 4),
    )
