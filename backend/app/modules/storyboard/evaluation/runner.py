"""스토리보드 타임라인 추출 평가 실행기

데이터셋을 로드하여 현재 추출 로직으로 실행하고 메트릭을 계산한다.
CLI에서 직접 실행 가능:
    uv run python -m app.modules.storyboard.evaluation.runner \
        --dataset app/modules/storyboard/evaluation/datasets/bar_exam_timeline_eval_v1.json \
        --prompt-version v1_baseline
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import evaluate_case
from .schemas import (
    TimelineEvalCase,
    TimelineEvalDataset,
    TimelineEvalResult,
)

logger = logging.getLogger(__name__)

# 평가 결과 저장 디렉토리
_EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


def _load_dataset(dataset_path: str | Path) -> TimelineEvalDataset:
    """데이터셋 JSON 파일을 로드한다."""
    path = Path(dataset_path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return TimelineEvalDataset(**data)


def _extract_pred_events(
    timeline_items: list[Any],
) -> list[dict[str, str]]:
    """TimelineItem 목록에서 메트릭 비교용 dict 목록을 추출한다."""
    events: list[dict[str, str]] = []
    for item in timeline_items:
        events.append({
            "date": item.date,
            "title": item.title,
        })
    return events


def _extract_pred_participants(timeline_items: list[Any]) -> list[str]:
    """TimelineItem 목록에서 참여자 이름 목록을 추출한다."""
    participants: list[str] = []
    seen: set[str] = set()
    for item in timeline_items:
        for p in item.participants_detailed:
            if p.name not in seen:
                participants.append(p.name)
                seen.add(p.name)
    return participants


def _extract_pred_roles(timeline_items: list[Any]) -> dict[str, str]:
    """TimelineItem 목록에서 참여자-역할 매핑을 추출한다."""
    roles: dict[str, str] = {}
    for item in timeline_items:
        for p in item.participants_detailed:
            if p.name not in roles:
                roles[p.name] = p.role.value
    return roles


async def _evaluate_single_case(
    case: TimelineEvalCase,
) -> dict[str, Any]:
    """단일 케이스를 평가한다."""
    from ..service import extract_timeline_from_text

    logger.info("평가 중: %s (%s)", case.id, case.doc_type)

    response = await extract_timeline_from_text(case.input_text)

    if not response.success or not response.timeline:
        logger.warning("추출 실패: %s", case.id)
        return {
            "case_id": case.id,
            "doc_type": case.doc_type,
            "success": False,
            "result": None,
        }

    gt = case.ground_truth
    gt_roles: dict[str, str] = {}
    for evt in gt.events:
        gt_roles.update(evt.participant_roles)

    pred_events = _extract_pred_events(response.timeline)
    pred_participants = _extract_pred_participants(response.timeline)
    pred_roles = _extract_pred_roles(response.timeline)

    case_result = evaluate_case(
        gt_events=gt.events,
        gt_order=gt.chronological_order,
        gt_participants=gt.key_participants,
        pred_events=pred_events,
        pred_participants=pred_participants,
        pred_roles=pred_roles,
        gt_roles=gt_roles,
        case_id=case.id,
    )

    return {
        "case_id": case.id,
        "doc_type": case.doc_type,
        "success": True,
        "result": case_result,
        "pred_event_count": len(response.timeline),
        "gt_event_count": gt.expected_event_count,
    }


async def run_evaluation(
    dataset_path: str | Path,
    prompt_version: str = "v1_baseline",
    model: str = "gpt-4o-mini",
) -> TimelineEvalResult:
    """데이터셋 전체를 현재 추출 로직으로 실행하고 메트릭을 계산한다.

    Args:
        dataset_path: 데이터셋 JSON 파일 경로
        prompt_version: 프롬프트 버전 태그
        model: 사용 모델명

    Returns:
        TimelineEvalResult 평가 결과
    """
    dataset = _load_dataset(dataset_path)
    logger.info(
        "데이터셋 로드: %s (%d 케이스)", dataset.name, len(dataset.cases)
    )

    # 각 케이스 순차 실행 (API rate limit 고려)
    raw_results: list[dict[str, Any]] = []
    for case in dataset.cases:
        result = await _evaluate_single_case(case)
        raw_results.append(result)

    # 결과 집계
    from .schemas import TimelineCaseResult

    case_results: list[TimelineCaseResult] = []
    by_type: dict[str, list[TimelineCaseResult]] = defaultdict(list)

    for raw in raw_results:
        if raw["success"] and raw["result"] is not None:
            cr: TimelineCaseResult = raw["result"]
            case_results.append(cr)
            by_type[raw["doc_type"]].append(cr)
        else:
            # 실패 케이스: 0점 처리
            zero_result = TimelineCaseResult(
                case_id=raw["case_id"],
                event_count_accuracy=0.0,
                chronological_order_score=0.0,
                participant_recall=0.0,
                role_accuracy=0.0,
                key_event_coverage=0.0,
                composite_score=0.0,
            )
            case_results.append(zero_result)
            by_type[raw["doc_type"]].append(zero_result)

    # doc_type별 평균
    aggregate_by_doc_type: dict[str, dict[str, float]] = {}
    for doc_type, results in by_type.items():
        if not results:
            continue
        n = len(results)
        aggregate_by_doc_type[doc_type] = {
            "event_count_accuracy": sum(r.event_count_accuracy for r in results) / n,
            "chronological_order_score": sum(
                r.chronological_order_score for r in results
            )
            / n,
            "participant_recall": sum(r.participant_recall for r in results) / n,
            "role_accuracy": sum(r.role_accuracy for r in results) / n,
            "key_event_coverage": sum(r.key_event_coverage for r in results) / n,
            "composite_score": sum(r.composite_score for r in results) / n,
        }

    # 전체 평균
    n_total = len(case_results)
    overall: dict[str, float] = {}
    if n_total > 0:
        overall = {
            "event_count_accuracy": sum(
                r.event_count_accuracy for r in case_results
            )
            / n_total,
            "chronological_order_score": sum(
                r.chronological_order_score for r in case_results
            )
            / n_total,
            "participant_recall": sum(
                r.participant_recall for r in case_results
            )
            / n_total,
            "role_accuracy": sum(r.role_accuracy for r in case_results) / n_total,
            "key_event_coverage": sum(
                r.key_event_coverage for r in case_results
            )
            / n_total,
            "composite_score": sum(r.composite_score for r in case_results)
            / n_total,
        }

    eval_result = TimelineEvalResult(
        dataset_name=dataset.name,
        model=model,
        prompt_version=prompt_version,
        case_results=case_results,
        aggregate_by_doc_type=aggregate_by_doc_type,
        overall=overall,
    )

    # 결과 저장
    _save_result(eval_result, prompt_version)

    return eval_result


def _save_result(result: TimelineEvalResult, prompt_version: str) -> Path:
    """평가 결과를 JSON 파일로 저장한다."""
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{prompt_version}_{timestamp}.json"
    filepath = _EXPERIMENTS_DIR / filename

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    logger.info("결과 저장: %s", filepath)
    return filepath


# --- CLI 진입점 ---


def main() -> None:
    """CLI에서 평가를 실행한다."""
    import argparse

    parser = argparse.ArgumentParser(
        description="스토리보드 타임라인 추출 평가 실행기"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="데이터셋 JSON 파일 경로",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1_baseline",
        help="프롬프트 버전 태그 (기본: v1_baseline)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="사용 모델명 (기본: gpt-4o-mini)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    result = asyncio.run(
        run_evaluation(
            dataset_path=args.dataset,
            prompt_version=args.prompt_version,
            model=args.model,
        )
    )

    # 요약 출력
    print("\n" + "=" * 60)
    print(f"평가 완료: {result.dataset_name}")
    print(f"모델: {result.model} | 프롬프트: {result.prompt_version}")
    print("=" * 60)

    print("\n📊 문서 유형별 결과:")
    for doc_type, metrics in result.aggregate_by_doc_type.items():
        print(f"\n  [{doc_type}]")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")

    print("\n📈 전체 평균:")
    for metric_name, value in result.overall.items():
        print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
