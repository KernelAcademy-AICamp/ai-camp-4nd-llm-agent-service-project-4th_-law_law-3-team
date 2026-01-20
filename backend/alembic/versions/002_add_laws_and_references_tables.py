"""add laws and references tables

Revision ID: 002
Revises: 001
Create Date: 2026-01-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. legal_documents 테이블에 source 컬럼 추가
    op.add_column(
        'legal_documents',
        sa.Column('source', sa.String(length=50), nullable=True,
                  comment='데이터 출처 (precedents, ftc, nhrck 등)')
    )

    # 기존 데이터에 기본값 설정
    op.execute("UPDATE legal_documents SET source = doc_type WHERE source IS NULL")

    # NOT NULL 제약 조건 추가
    op.alter_column('legal_documents', 'source', nullable=False, server_default='')

    # serial_number 컬럼 길이 확장 (50 -> 100)
    op.alter_column(
        'legal_documents', 'serial_number',
        existing_type=sa.String(length=50),
        type_=sa.String(length=100),
        existing_nullable=False
    )

    # 기존 unique constraint 삭제 및 새로운 constraint 생성
    op.drop_constraint('uq_legal_docs_type_serial', 'legal_documents', type_='unique')
    op.create_unique_constraint(
        'uq_legal_docs_type_serial_source',
        'legal_documents',
        ['doc_type', 'serial_number', 'source']
    )

    # source 인덱스 추가
    op.create_index('idx_legal_docs_source', 'legal_documents', ['source'], unique=False)

    # 2. laws 테이블 생성
    op.create_table(
        'laws',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('law_id', sa.String(length=20), nullable=False,
                  comment='법령 ID'),
        sa.Column('law_name', sa.Text(), nullable=False,
                  comment='법령명 (민법, 형법 등)'),
        sa.Column('law_type', sa.String(length=20), nullable=True,
                  comment='법률, 대통령령, 부령 등'),
        sa.Column('ministry', sa.Text(), nullable=True,
                  comment='소관부처'),
        sa.Column('promulgation_date', sa.Date(), nullable=True,
                  comment='공포일'),
        sa.Column('promulgation_no', sa.String(length=20), nullable=True,
                  comment='공포번호'),
        sa.Column('enforcement_date', sa.Date(), nullable=True,
                  comment='시행일'),
        sa.Column('content', sa.Text(), nullable=True,
                  comment='전체 조문 내용'),
        sa.Column('supplementary', sa.Text(), nullable=True,
                  comment='부칙'),
        sa.Column('raw_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False,
                  comment='원본 데이터 전체 (JSON)'),
        sa.Column('created_at', sa.DateTime(), nullable=True,
                  comment='레코드 생성일시'),
        sa.Column('updated_at', sa.DateTime(), nullable=True,
                  comment='레코드 수정일시'),
        sa.PrimaryKeyConstraint('id')
    )

    # laws 인덱스 생성
    op.create_index('idx_laws_law_id', 'laws', ['law_id'], unique=True)
    op.create_index('idx_laws_law_name', 'laws', ['law_name'], unique=False)
    op.create_index('idx_laws_law_type', 'laws', ['law_type'], unique=False)
    op.create_index('idx_laws_ministry', 'laws', ['ministry'], unique=False)
    op.create_index('idx_laws_enforcement', 'laws', ['enforcement_date'], unique=False)

    # 3. legal_references 테이블 생성
    op.create_table(
        'legal_references',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('ref_type', sa.String(length=20), nullable=False,
                  comment='참조 유형: treaty, admin_rule, law_term'),
        sa.Column('serial_number', sa.String(length=100), nullable=False,
                  comment='일련번호'),
        sa.Column('title', sa.Text(), nullable=True,
                  comment='명칭 (조약명/규칙명/용어명)'),
        sa.Column('content', sa.Text(), nullable=True,
                  comment='내용/정의'),
        sa.Column('organization', sa.Text(), nullable=True,
                  comment='소관기관/체결국가'),
        sa.Column('category', sa.Text(), nullable=True,
                  comment='분류/종류'),
        sa.Column('effective_date', sa.Date(), nullable=True,
                  comment='발효일/시행일'),
        sa.Column('raw_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False,
                  comment='원본 데이터 전체 (JSON)'),
        sa.Column('created_at', sa.DateTime(), nullable=True,
                  comment='레코드 생성일시'),
        sa.Column('updated_at', sa.DateTime(), nullable=True,
                  comment='레코드 수정일시'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('ref_type', 'serial_number', name='uq_legal_refs_type_serial')
    )

    # legal_references 인덱스 생성
    op.create_index('idx_legal_refs_type', 'legal_references', ['ref_type'], unique=False)
    op.create_index('idx_legal_refs_title', 'legal_references', ['title'], unique=False)
    op.create_index('idx_legal_refs_org', 'legal_references', ['organization'], unique=False)


def downgrade() -> None:
    # legal_references 인덱스 및 테이블 삭제
    op.drop_index('idx_legal_refs_org', table_name='legal_references')
    op.drop_index('idx_legal_refs_title', table_name='legal_references')
    op.drop_index('idx_legal_refs_type', table_name='legal_references')
    op.drop_table('legal_references')

    # laws 인덱스 및 테이블 삭제
    op.drop_index('idx_laws_enforcement', table_name='laws')
    op.drop_index('idx_laws_ministry', table_name='laws')
    op.drop_index('idx_laws_law_type', table_name='laws')
    op.drop_index('idx_laws_law_name', table_name='laws')
    op.drop_index('idx_laws_law_id', table_name='laws')
    op.drop_table('laws')

    # legal_documents 변경 롤백
    op.drop_index('idx_legal_docs_source', table_name='legal_documents')
    op.drop_constraint('uq_legal_docs_type_serial_source', 'legal_documents', type_='unique')
    op.create_unique_constraint(
        'uq_legal_docs_type_serial',
        'legal_documents',
        ['doc_type', 'serial_number']
    )

    # serial_number 컬럼 길이 복원
    op.alter_column(
        'legal_documents', 'serial_number',
        existing_type=sa.String(length=100),
        type_=sa.String(length=50),
        existing_nullable=False
    )

    op.drop_column('legal_documents', 'source')
