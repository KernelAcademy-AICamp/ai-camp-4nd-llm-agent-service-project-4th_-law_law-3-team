"""변호사시험 기록형 타임라인 추출 통합 테스트

골든 데이터셋을 사용하여 extract_timeline_from_text()의 품질을 평가한다.
OpenAI API 키가 필요하므로 requires_openai 마커를 사용한다.

실행:
    cd backend
    uv run pytest tests/integration/storyboard/test_bar_exam_extraction.py -v -m slow
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# 데이터셋 경로
_DATASET_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "app"
    / "modules"
    / "storyboard"
    / "evaluation"
    / "datasets"
    / "bar_exam_timeline_eval_v1.json"
)


def _skip_if_no_openai() -> None:
    """OpenAI API 키가 없으면 테스트 건너뛰기"""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY 환경변수 필요")


def _skip_if_no_dataset() -> None:
    """골든 데이터셋 파일이 없으면 건너뛰기"""
    if not _DATASET_PATH.exists():
        pytest.skip(f"데이터셋 없음: {_DATASET_PATH}")


# ============================================================================
# 데이터셋 검증 테스트 (API 불필요)
# ============================================================================


class TestDatasetValidation:
    """골든 데이터셋 스키마 유효성 검증"""

    def test_dataset_file_exists(self) -> None:
        """데이터셋 JSON 파일이 존재하는지 확인"""
        _skip_if_no_dataset()
        assert _DATASET_PATH.exists()

    def test_dataset_loads_as_valid_json(self) -> None:
        """JSON 파싱 가능한지 확인"""
        _skip_if_no_dataset()
        with _DATASET_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "cases" in data

    def test_dataset_matches_schema(self) -> None:
        """데이터셋이 TimelineEvalDataset 스키마를 만족하는지 확인"""
        _skip_if_no_dataset()
        from app.modules.storyboard.evaluation.schemas import (
            TimelineEvalDataset,
        )

        with _DATASET_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        dataset = TimelineEvalDataset(**data)
        assert len(dataset.cases) >= 1
        assert dataset.version
        assert dataset.name

    def test_dataset_has_all_doc_types(self) -> None:
        """민사/형사/공법 3개 유형이 모두 포함되어 있는지 확인"""
        _skip_if_no_dataset()
        from app.modules.storyboard.evaluation.schemas import (
            TimelineEvalDataset,
        )

        with _DATASET_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        dataset = TimelineEvalDataset(**data)
        doc_types = {c.doc_type for c in dataset.cases}
        assert "civil" in doc_types
        assert "criminal" in doc_types
        assert "public" in doc_types

    def test_each_case_has_ground_truth(self) -> None:
        """각 케이스에 ground_truth가 있는지 확인"""
        _skip_if_no_dataset()
        from app.modules.storyboard.evaluation.schemas import (
            TimelineEvalDataset,
        )

        with _DATASET_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        dataset = TimelineEvalDataset(**data)
        for case in dataset.cases:
            gt = case.ground_truth
            assert gt.expected_event_count > 0, f"{case.id}: expected_event_count == 0"
            assert len(gt.events) > 0, f"{case.id}: events 비어있음"
            assert len(gt.chronological_order) > 0, f"{case.id}: chronological_order 비어있음"
            assert len(gt.key_participants) > 0, f"{case.id}: key_participants 비어있음"

    def test_each_case_has_nonempty_input(self) -> None:
        """각 케이스의 input_text가 비어있지 않은지 확인"""
        _skip_if_no_dataset()
        from app.modules.storyboard.evaluation.schemas import (
            TimelineEvalDataset,
        )

        with _DATASET_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        dataset = TimelineEvalDataset(**data)
        for case in dataset.cases:
            assert len(case.input_text) > 100, (
                f"{case.id}: input_text 너무 짧음 ({len(case.input_text)}자)"
            )


# ============================================================================
# 문서 유형 감지 테스트 (API 불필요)
# ============================================================================


class TestDocTypeDetection:
    """데이터셋 케이스의 문서 유형 감지 정확도 검증"""

    def test_doc_type_detection_accuracy(self) -> None:
        """데이터셋의 각 케이스에서 doc_type이 올바르게 감지되는지 확인"""
        _skip_if_no_dataset()
        from app.modules.storyboard.evaluation.schemas import (
            TimelineEvalDataset,
        )
        from app.modules.storyboard.service.doc_type_detector import (
            detect_document_type,
        )

        with _DATASET_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        dataset = TimelineEvalDataset(**data)

        correct = 0
        total = 0
        mismatches: list[str] = []

        for case in dataset.cases:
            detected = detect_document_type(case.input_text)
            total += 1
            if detected == case.doc_type:
                correct += 1
            else:
                mismatches.append(
                    f"{case.id}: expected={case.doc_type}, detected={detected}"
                )

        accuracy = correct / total if total > 0 else 0
        # 80% 이상 정확도 기대 (general 폴백 허용)
        assert accuracy >= 0.8, (
            f"문서 유형 감지 정확도 {accuracy:.2%} < 80%\n불일치: {mismatches}"
        )


# ============================================================================
# 타임라인 추출 평가 테스트 (OpenAI API 필요)
# ============================================================================


@pytest.mark.slow
@pytest.mark.requires_openai
class TestBarExamExtraction:
    """변호사시험 기록형 타임라인 추출 통합 테스트"""

    @pytest.mark.asyncio
    async def test_single_case_extraction(self) -> None:
        """단일 케이스 추출 성공 확인"""
        _skip_if_no_openai()
        _skip_if_no_dataset()
        from app.modules.storyboard.evaluation.schemas import (
            TimelineEvalDataset,
        )
        from app.modules.storyboard.service import extract_timeline_from_text

        with _DATASET_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        dataset = TimelineEvalDataset(**data)

        # 첫 번째 케이스로 추출 테스트
        case = dataset.cases[0]
        response = await extract_timeline_from_text(case.input_text)

        assert response.success, f"추출 실패: {case.id}"
        assert response.timeline is not None
        assert len(response.timeline) > 0, f"타임라인 비어있음: {case.id}"

    @pytest.mark.asyncio
    async def test_extraction_returns_dates(self) -> None:
        """추출 결과에 date 필드가 포함되는지 확인"""
        _skip_if_no_openai()
        _skip_if_no_dataset()
        from app.modules.storyboard.evaluation.schemas import (
            TimelineEvalDataset,
        )
        from app.modules.storyboard.service import extract_timeline_from_text

        with _DATASET_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        dataset = TimelineEvalDataset(**data)
        case = dataset.cases[0]

        response = await extract_timeline_from_text(case.input_text)
        assert response.success

        for item in response.timeline:
            assert item.date, f"date 비어있음: {item.title}"

    @pytest.mark.asyncio
    async def test_extraction_returns_participants(self) -> None:
        """추출 결과에 participants_detailed가 포함되는지 확인"""
        _skip_if_no_openai()
        _skip_if_no_dataset()
        from app.modules.storyboard.evaluation.schemas import (
            TimelineEvalDataset,
        )
        from app.modules.storyboard.service import extract_timeline_from_text

        with _DATASET_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        dataset = TimelineEvalDataset(**data)
        case = dataset.cases[0]

        response = await extract_timeline_from_text(case.input_text)
        assert response.success

        has_participants = any(
            len(item.participants_detailed) > 0 for item in response.timeline
        )
        assert has_participants, "참여자 정보가 하나도 없음"


@pytest.mark.slow
@pytest.mark.requires_openai
class TestFullEvaluation:
    """전체 평가 파이프라인 통합 테스트"""

    @pytest.mark.asyncio
    async def test_run_evaluation_completes(self) -> None:
        """전체 평가 실행이 완료되는지 확인"""
        _skip_if_no_openai()
        _skip_if_no_dataset()
        from app.modules.storyboard.evaluation.runner import run_evaluation

        result = await run_evaluation(
            dataset_path=_DATASET_PATH,
            prompt_version="v1_baseline",
            model="gpt-4o-mini",
        )

        assert result.dataset_name
        assert len(result.case_results) > 0
        assert result.overall, "전체 평균 메트릭 비어있음"

    @pytest.mark.asyncio
    async def test_evaluation_composite_scores(self) -> None:
        """평가 결과의 composite_score가 유효 범위인지 확인"""
        _skip_if_no_openai()
        _skip_if_no_dataset()
        from app.modules.storyboard.evaluation.runner import run_evaluation

        result = await run_evaluation(
            dataset_path=_DATASET_PATH,
            prompt_version="v1_baseline",
            model="gpt-4o-mini",
        )

        for cr in result.case_results:
            assert 0.0 <= cr.composite_score <= 1.0, (
                f"{cr.case_id}: composite_score {cr.composite_score} 범위 초과"
            )

    @pytest.mark.asyncio
    async def test_evaluation_has_doc_type_breakdown(self) -> None:
        """평가 결과에 doc_type별 집계가 포함되는지 확인"""
        _skip_if_no_openai()
        _skip_if_no_dataset()
        from app.modules.storyboard.evaluation.runner import run_evaluation

        result = await run_evaluation(
            dataset_path=_DATASET_PATH,
            prompt_version="v1_baseline",
            model="gpt-4o-mini",
        )

        assert len(result.aggregate_by_doc_type) > 0, "doc_type별 집계 비어있음"
        for doc_type, metrics in result.aggregate_by_doc_type.items():
            assert "composite_score" in metrics, (
                f"{doc_type}: composite_score 없음"
            )

    @pytest.mark.asyncio
    async def test_evaluation_saves_result(self, tmp_path: Path) -> None:
        """평가 결과가 experiments/ 디렉토리에 저장되는지 확인"""
        _skip_if_no_openai()
        _skip_if_no_dataset()
        import app.modules.storyboard.evaluation.runner as runner_module

        # experiments 디렉토리를 임시 경로로 변경
        original_dir = runner_module._EXPERIMENTS_DIR
        runner_module._EXPERIMENTS_DIR = tmp_path

        try:
            result = await runner_module.run_evaluation(
                dataset_path=_DATASET_PATH,
                prompt_version="test_save",
                model="gpt-4o-mini",
            )

            saved_files = list(tmp_path.glob("test_save_*.json"))
            assert len(saved_files) == 1, f"저장된 파일 수: {len(saved_files)}"

            with saved_files[0].open(encoding="utf-8") as f:
                saved_data = json.load(f)
            assert saved_data["dataset_name"] == result.dataset_name
        finally:
            runner_module._EXPERIMENTS_DIR = original_dir
