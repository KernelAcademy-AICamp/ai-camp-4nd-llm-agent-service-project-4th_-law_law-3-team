"""create legal_documents table

Revision ID: 001
Revises:
Create Date: 2026-01-19

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # legal_documents 테이블 생성
    op.create_table(
        'legal_documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('doc_type', sa.String(length=20), nullable=False,
                  comment='문서 유형: precedent, constitutional, administration, legislation'),
        sa.Column('serial_number', sa.String(length=50), nullable=False,
                  comment='원본 일련번호 (판례정보일련번호 등)'),
        sa.Column('case_name', sa.Text(), nullable=True,
                  comment='사건명/안건명'),
        sa.Column('case_number', sa.Text(), nullable=True,
                  comment='사건번호/안건번호'),
        sa.Column('decision_date', sa.Date(), nullable=True,
                  comment='선고일/의결일/종국일'),
        sa.Column('court_name', sa.Text(), nullable=True,
                  comment='법원명/재결청/해석기관'),
        sa.Column('court_type', sa.Text(), nullable=True,
                  comment='법원종류/재결례유형'),
        sa.Column('case_type', sa.Text(), nullable=True,
                  comment='사건종류 (민사/형사/헌마 등)'),
        sa.Column('summary', sa.Text(), nullable=True,
                  comment='판시사항/결정요지/질의요지/주문'),
        sa.Column('reasoning', sa.Text(), nullable=True,
                  comment='판결요지/이유/회답'),
        sa.Column('full_text', sa.Text(), nullable=True,
                  comment='판례내용/전문'),
        sa.Column('claim', sa.Text(), nullable=True,
                  comment='청구취지 (행정심판례)'),
        sa.Column('reference_articles', sa.Text(), nullable=True,
                  comment='참조조문'),
        sa.Column('reference_cases', sa.Text(), nullable=True,
                  comment='참조판례'),
        sa.Column('raw_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False,
                  comment='원본 데이터 전체 (JSON)'),
        sa.Column('created_at', sa.DateTime(), nullable=True,
                  comment='레코드 생성일시'),
        sa.Column('updated_at', sa.DateTime(), nullable=True,
                  comment='레코드 수정일시'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('doc_type', 'serial_number', name='uq_legal_docs_type_serial')
    )

    # 인덱스 생성
    op.create_index('idx_legal_docs_type', 'legal_documents', ['doc_type'], unique=False)
    op.create_index('idx_legal_docs_case_number', 'legal_documents', ['case_number'], unique=False)
    op.create_index('idx_legal_docs_decision_date', 'legal_documents', ['decision_date'], unique=False)
    op.create_index('idx_legal_docs_court', 'legal_documents', ['court_name'], unique=False)
    op.create_index('idx_legal_docs_case_type', 'legal_documents', ['case_type'], unique=False)
    op.create_index('idx_legal_docs_search', 'legal_documents',
                    ['doc_type', 'case_type', 'court_name', 'decision_date'], unique=False)

    # GIN 인덱스 (JSONB)
    op.create_index('idx_legal_docs_raw', 'legal_documents', ['raw_data'],
                    unique=False, postgresql_using='gin')


def downgrade() -> None:
    # 인덱스 삭제
    op.drop_index('idx_legal_docs_raw', table_name='legal_documents')
    op.drop_index('idx_legal_docs_search', table_name='legal_documents')
    op.drop_index('idx_legal_docs_case_type', table_name='legal_documents')
    op.drop_index('idx_legal_docs_court', table_name='legal_documents')
    op.drop_index('idx_legal_docs_decision_date', table_name='legal_documents')
    op.drop_index('idx_legal_docs_case_number', table_name='legal_documents')
    op.drop_index('idx_legal_docs_type', table_name='legal_documents')

    # 테이블 삭제
    op.drop_table('legal_documents')
