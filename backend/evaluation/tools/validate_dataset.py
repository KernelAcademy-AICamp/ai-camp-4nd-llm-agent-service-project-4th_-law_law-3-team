"""
데이터셋 검증 도구

평가 데이터셋의 유효성을 검사:
1. 스키마 검증
2. Ground Truth 문서 존재 여부 확인
3. 데이터 분포 분석
"""

import json
from pathlib import Path
from typing import Optional

from evaluation.schemas import EvalDataset
from evaluation.tools.dataset_builder import DatasetBuilder
from evaluation.config import eval_settings, QUERY_TYPE_DISTRIBUTION


class DatasetValidator:
    """
    데이터셋 검증기

    Usage:
        validator = DatasetValidator()
        report = await validator.validate("evaluation/datasets/eval_dataset_v1.json")
        print(report)
    """

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    async def validate(
        self,
        dataset_path: str,
        check_documents: bool = True,
    ) -> dict:
        """
        데이터셋 검증

        Args:
            dataset_path: 데이터셋 경로
            check_documents: Ground Truth 문서 존재 여부 확인

        Returns:
            검증 결과 딕셔너리
        """
        self.errors = []
        self.warnings = []
        self.info = []

        path = Path(dataset_path)
        if not path.is_absolute():
            path = eval_settings.datasets_dir / dataset_path

        if not path.exists():
            self.errors.append(f"데이터셋 파일이 존재하지 않습니다: {path}")
            return self._build_report()

        try:
            builder = DatasetBuilder.load(path)
            dataset = builder.dataset
        except Exception as e:
            self.errors.append(f"데이터셋 로드 실패: {str(e)}")
            return self._build_report()

        self._validate_schema(dataset)
        self._validate_ids(dataset)
        self._validate_content(dataset)
        self._analyze_distribution(dataset)

        if check_documents:
            await self._check_documents_exist(dataset)

        return self._build_report()

    def _validate_schema(self, dataset: EvalDataset) -> None:
        """스키마 검증"""
        if not dataset.name:
            self.errors.append("데이터셋 이름이 비어있습니다")

        if not dataset.queries:
            self.warnings.append("데이터셋에 쿼리가 없습니다")

        self.info.append(f"총 쿼리 수: {len(dataset.queries)}")

    def _validate_ids(self, dataset: EvalDataset) -> None:
        """ID 유효성 검증"""
        ids = [q.id for q in dataset.queries]

        if len(ids) != len(set(ids)):
            duplicates = [id for id in ids if ids.count(id) > 1]
            self.errors.append(f"중복된 쿼리 ID: {set(duplicates)}")

        for query in dataset.queries:
            if not query.id.startswith("Q-"):
                self.warnings.append(f"비표준 쿼리 ID 형식: {query.id}")

    def _validate_content(self, dataset: EvalDataset) -> None:
        """내용 검증"""
        for query in dataset.queries:
            if not query.question.strip():
                self.errors.append(f"{query.id}: 질문이 비어있습니다")

            if len(query.question) < 10:
                self.warnings.append(f"{query.id}: 질문이 너무 짧습니다 ({len(query.question)}자)")

            if len(query.question) > 500:
                self.warnings.append(f"{query.id}: 질문이 너무 깁니다 ({len(query.question)}자)")

            if not query.ground_truth.source_documents:
                self.errors.append(f"{query.id}: Ground Truth 문서가 없습니다")

            if not query.ground_truth.key_points:
                self.warnings.append(f"{query.id}: Key Points가 없습니다")

    def _analyze_distribution(self, dataset: EvalDataset) -> None:
        """분포 분석"""
        if not dataset.queries:
            return

        by_category: dict[str, int] = {}
        by_type: dict[str, int] = {}
        by_difficulty: dict[str, int] = {}

        for query in dataset.queries:
            cat = query.metadata.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            qtype = query.metadata.query_type.value
            by_type[qtype] = by_type.get(qtype, 0) + 1

            diff = query.metadata.difficulty.value
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

        total = len(dataset.queries)

        self.info.append(f"카테고리 분포: {by_category}")
        self.info.append(f"쿼리 유형 분포: {by_type}")
        self.info.append(f"난이도 분포: {by_difficulty}")

        for qtype, target_ratio in QUERY_TYPE_DISTRIBUTION.items():
            actual_count = by_type.get(qtype, 0)
            actual_ratio = actual_count / total if total > 0 else 0
            target_count = int(total * target_ratio)

            if actual_count < target_count * 0.5:
                self.warnings.append(
                    f"'{qtype}' 쿼리 부족: {actual_count}개 (목표: {target_count}개, {target_ratio*100:.0f}%)"
                )

    async def _check_documents_exist(self, dataset: EvalDataset) -> None:
        """Ground Truth 문서 존재 여부 확인"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from sqlalchemy import select
        from app.core.database import async_session_factory
        from app.models.precedent_document import PrecedentDocument
        from app.models.law_document import LawDocument

        precedent_ids = set()
        law_ids = set()

        for query in dataset.queries:
            for doc in query.ground_truth.source_documents:
                if doc.doc_type.value == "precedent":
                    precedent_ids.add(doc.doc_id)
                else:
                    law_ids.add(doc.doc_id)

        async with async_session_factory() as session:
            if precedent_ids:
                result = await session.execute(
                    select(PrecedentDocument.serial_number)
                )
                existing_precedents = {r[0] for r in result}
                missing = precedent_ids - existing_precedents
                if missing:
                    self.errors.append(f"존재하지 않는 판례: {missing}")

            if law_ids:
                result = await session.execute(
                    select(LawDocument.law_id)
                )
                existing_laws = {r[0] for r in result}
                missing = law_ids - existing_laws
                if missing:
                    self.errors.append(f"존재하지 않는 법령: {missing}")

    def _build_report(self) -> dict:
        """검증 리포트 생성"""
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "summary": {
                "error_count": len(self.errors),
                "warning_count": len(self.warnings),
            },
        }


async def main():
    """CLI 실행"""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="데이터셋 검증")
    parser.add_argument(
        "dataset",
        type=str,
        help="검증할 데이터셋 경로",
    )
    parser.add_argument(
        "--skip-doc-check",
        action="store_true",
        help="문서 존재 여부 검사 생략",
    )

    args = parser.parse_args()

    validator = DatasetValidator()
    report = await validator.validate(
        args.dataset,
        check_documents=not args.skip_doc_check,
    )

    print("\n=== 데이터셋 검증 결과 ===\n")

    if report["valid"]:
        print("✅ 유효한 데이터셋입니다.\n")
    else:
        print("❌ 데이터셋에 오류가 있습니다.\n")

    if report["errors"]:
        print("🔴 오류:")
        for error in report["errors"]:
            print(f"  - {error}")
        print()

    if report["warnings"]:
        print("🟡 경고:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
        print()

    if report["info"]:
        print("ℹ️ 정보:")
        for info in report["info"]:
            print(f"  - {info}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
