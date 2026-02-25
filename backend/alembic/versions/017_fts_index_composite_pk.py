"""fts_index 복합 PK (source_id, data_type)

source_id 단독 PK → (source_id, data_type) 복합 PK로 변경.
타입 간 serial_number 중복 시 덮어쓰기 방지 (58,394건 충돌 해소).

Revision ID: 017
Revises: f61f09ed1c72
Create Date: 2026-02-25

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "017"
down_revision: Union[str, Sequence[str], None] = "f61f09ed1c72"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """source_id 단독 PK → (source_id, data_type) 복합 PK"""
    # 기존 데이터 전체 삭제 (PK 변경 후 전체 FTS 재빌드 필요)
    op.execute("TRUNCATE TABLE fts_index")

    # 기존 PK 삭제
    op.drop_constraint("fts_index_pkey", "fts_index", type_="primary")

    # 복합 PK 생성
    op.create_primary_key("fts_index_pkey", "fts_index", ["source_id", "data_type"])


def downgrade() -> None:
    """(source_id, data_type) 복합 PK → source_id 단독 PK"""
    op.execute("TRUNCATE TABLE fts_index")
    op.drop_constraint("fts_index_pkey", "fts_index", type_="primary")
    op.create_primary_key("fts_index_pkey", "fts_index", ["source_id"])
