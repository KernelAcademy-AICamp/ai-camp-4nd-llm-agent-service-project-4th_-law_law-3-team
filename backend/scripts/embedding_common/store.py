"""
LanceDB 테이블 생성/연결 유틸리티

임베딩 스크립트에서 LanceDB 테이블 접근 시 공통으로 사용합니다.
"""

from pathlib import Path
from typing import Any, Optional

from scripts.embedding_common.config import DEFAULT_CONFIG
from scripts.embedding_common.schema import LEGAL_CHUNKS_SCHEMA


class EmbeddingStore:
    """임베딩 스크립트용 LanceDB 저장소"""

    def __init__(
        self,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> None:
        import lancedb

        db_path = db_path or str(DEFAULT_CONFIG["LANCEDB_URI"])
        self.db_path = db_path
        self.table_name = table_name or str(DEFAULT_CONFIG["LANCEDB_TABLE_NAME"])

        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self._table: Any = None

        if self.table_name in self.db.table_names():
            try:
                self._table = self.db.open_table(self.table_name)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    "테이블 '%s' 열기 실패 (손상 가능): %s", self.table_name, e
                )
                self._table = None

    @property
    def table(self) -> Any:
        return self._table

    def ensure_table(self) -> Any:
        """테이블이 없으면 생성"""
        if self._table is None:
            self._table = self.db.create_table(
                self.table_name,
                schema=LEGAL_CHUNKS_SCHEMA,
            )
        return self._table

    def reset(self) -> None:
        """테이블 삭제 후 재생성"""
        if self.table_name in self.db.table_names():
            try:
                self.db.drop_table(self.table_name)
            except Exception:
                import logging
                import shutil

                logging.getLogger(__name__).warning(
                    "drop_table 실패, lance 디렉토리 직접 삭제"
                )
                lance_dir = Path(self.db_path) / f"{self.table_name}.lance"
                if lance_dir.exists():
                    shutil.rmtree(lance_dir)
        self._table = None

    def count(self) -> int:
        """총 레코드 수"""
        if self._table is None:
            return 0
        return len(self._table)

    def count_by_type(self, data_type: str) -> int:
        """유형별 레코드 수"""
        if self._table is None:
            return 0
        try:
            result = (
                self._table.search()
                .where(f"data_type = '{data_type}'")
                .select(["id"])
                .limit(1_000_000)
                .to_arrow()
            )
            return result.num_rows
        except (ValueError, KeyError, AttributeError) as e:
            print(f"[WARN] count_by_type failed: {e}")
            return 0

    def get_existing_source_ids(self, data_type: str) -> set[str]:
        """특정 유형의 기존 source_id 목록 조회"""
        if self._table is None:
            return set()
        try:
            result = (
                self._table.search()
                .where(f"data_type = '{data_type}'")
                .select(["source_id"])
                .limit(1_000_000)
                .to_pandas()
            )
            return set(result["source_id"].unique())
        except Exception:
            return set()

    def add_batch(self, data: list[dict[str, Any]]) -> None:
        """배치 데이터 추가"""
        if not data:
            return
        table = self.ensure_table()
        table.add(data)
