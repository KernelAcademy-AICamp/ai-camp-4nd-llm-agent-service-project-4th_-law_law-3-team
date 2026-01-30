"""add lancedb tables (law_documents, precedent_documents)

Revision ID: 003
Revises: 002
Create Date: 2026-01-28

LanceDB 전용 PostgreSQL 테이블 생성
- law_documents: 법령 원본 데이터 (data/law_cleaned.json)
- precedent_documents: 판례 원본 데이터 (data/precedents_cleaned.json)

기존 legal_documents 테이블은 ChromaDB 호환을 위해 유지
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. law_documents 테이블 생성 (법령 원본)
    op.create_table(
        'law_documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('law_id', sa.String(length=50), nullable=False,
                  comment='법령 ID (원본 law_id)'),
        sa.Column('law_name', sa.String(length=500), nullable=False,
                  comment='법령명 (민법, 형법 등)'),
        sa.Column('law_type', sa.String(length=50), nullable=True,
                  comment='법령 유형 (법률/시행령/시행규칙)'),
        sa.Column('ministry', sa.String(length=200), nullable=True,
                  comment='소관부처'),
        sa.Column('promulgation_date', sa.String(length=20), nullable=True,
                  comment='공포일자 (YYYYMMDD 형식)'),
        sa.Column('promulgation_no', sa.String(length=50), nullable=True,
                  comment='공포번호'),
        sa.Column('enforcement_date', sa.Date(), nullable=True,
                  comment='시행일'),
        sa.Column('content', sa.Text(), nullable=True,
                  comment='조문 전체 텍스트'),
        sa.Column('supplementary', sa.Text(), nullable=True,
                  comment='부칙'),
        sa.Column('raw_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False,
                  comment='원본 JSON 데이터 전체'),
        sa.Column('created_at', sa.DateTime(), nullable=True,
                  comment='레코드 생성일시'),
        sa.Column('updated_at', sa.DateTime(), nullable=True,
                  comment='레코드 수정일시'),
        sa.PrimaryKeyConstraint('id')
    )

    # law_documents 인덱스 생성
    op.create_index('idx_law_docs_law_id', 'law_documents', ['law_id'], unique=True)
    op.create_index('idx_law_docs_name', 'law_documents', ['law_name'], unique=False)
    op.create_index('idx_law_docs_type', 'law_documents', ['law_type'], unique=False)
    op.create_index('idx_law_docs_ministry', 'law_documents', ['ministry'], unique=False)
    op.create_index('idx_law_docs_enforcement', 'law_documents', ['enforcement_date'], unique=False)

    # 2. precedent_documents 테이블 생성 (판례 원본)
    op.create_table(
        'precedent_documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('serial_number', sa.String(length=100), nullable=False,
                  comment='판례정보일련번호'),
        sa.Column('case_name', sa.Text(), nullable=True,
                  comment='사건명'),
        sa.Column('case_number', sa.String(length=100), nullable=True,
                  comment='사건번호 (예: 84나3990)'),
        sa.Column('decision_date', sa.Date(), nullable=True,
                  comment='선고일자'),
        sa.Column('court_name', sa.String(length=100), nullable=True,
                  comment='법원명 (대법원, 서울고법 등)'),
        sa.Column('case_type', sa.String(length=50), nullable=True,
                  comment='사건종류명 (민사/형사/행정)'),
        sa.Column('judgment_type', sa.String(length=100), nullable=True,
                  comment='판결유형 (예: 제11민사부판결)'),
        sa.Column('summary', sa.Text(), nullable=True,
                  comment='판시사항'),
        sa.Column('reasoning', sa.Text(), nullable=True,
                  comment='판결요지'),
        sa.Column('ruling', sa.Text(), nullable=True,
                  comment='주문'),
        sa.Column('claim', sa.Text(), nullable=True,
                  comment='청구취지'),
        sa.Column('full_reason', sa.Text(), nullable=True,
                  comment='이유 (전체)'),
        sa.Column('full_text', sa.Text(), nullable=True,
                  comment='판례내용 (전문)'),
        sa.Column('reference_provisions', sa.Text(), nullable=True,
                  comment='참조조문 (예: 민법 제750조, 제756조)'),
        sa.Column('reference_cases', sa.Text(), nullable=True,
                  comment='참조판례'),
        sa.Column('raw_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False,
                  comment='원본 JSON 데이터 전체'),
        sa.Column('created_at', sa.DateTime(), nullable=True,
                  comment='레코드 생성일시'),
        sa.Column('updated_at', sa.DateTime(), nullable=True,
                  comment='레코드 수정일시'),
        sa.PrimaryKeyConstraint('id')
    )

    # precedent_documents 인덱스 생성
    op.create_index('idx_precedent_docs_serial', 'precedent_documents', ['serial_number'], unique=True)
    op.create_index('idx_precedent_docs_case_number', 'precedent_documents', ['case_number'], unique=False)
    op.create_index('idx_precedent_docs_court', 'precedent_documents', ['court_name'], unique=False)
    op.create_index('idx_precedent_docs_case_type', 'precedent_documents', ['case_type'], unique=False)
    op.create_index('idx_precedent_docs_date', 'precedent_documents', ['decision_date'], unique=False)


def downgrade() -> None:
    # precedent_documents 인덱스 및 테이블 삭제
    op.drop_index('idx_precedent_docs_date', table_name='precedent_documents')
    op.drop_index('idx_precedent_docs_case_type', table_name='precedent_documents')
    op.drop_index('idx_precedent_docs_court', table_name='precedent_documents')
    op.drop_index('idx_precedent_docs_case_number', table_name='precedent_documents')
    op.drop_index('idx_precedent_docs_serial', table_name='precedent_documents')
    op.drop_table('precedent_documents')

    # law_documents 인덱스 및 테이블 삭제
    op.drop_index('idx_law_docs_enforcement', table_name='law_documents')
    op.drop_index('idx_law_docs_ministry', table_name='law_documents')
    op.drop_index('idx_law_docs_type', table_name='law_documents')
    op.drop_index('idx_law_docs_name', table_name='law_documents')
    op.drop_index('idx_law_docs_law_id', table_name='law_documents')
    op.drop_table('law_documents')
