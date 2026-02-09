"""add legal terms table

Revision ID: 006
Revises: 005
Create Date: 2026-02-06

법률 용어 사전 PostgreSQL 마이그레이션
- legal_terms 테이블: MeCab 토크나이저 법률 복합명사 보강용
- UNIQUE 제약조건: term
- 인덱스: term, source_code, is_korean_only, priority, term_length
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '006'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'legal_terms',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('term', sa.String(length=200), nullable=False,
                  comment='법령용어명 한글'),
        sa.Column('term_hanja', sa.String(length=200), nullable=True,
                  comment='법령용어명 한자'),
        sa.Column('definition', sa.Text(), nullable=True,
                  comment='법령용어 정의'),
        sa.Column('source', sa.Text(), nullable=True,
                  comment='출처 법령명'),
        sa.Column('source_code', sa.String(length=20), nullable=True,
                  comment='사전유형 (법령정의사전, 법령한영사전 등)'),
        sa.Column('serial_number', sa.String(length=20), nullable=True,
                  comment='법령용어 일련번호'),
        sa.Column('term_length', sa.Integer(), nullable=False,
                  comment='용어 글자 수 (필터링용)'),
        sa.Column('is_korean_only', sa.Boolean(), nullable=False,
                  comment='한글 전용 여부'),
        sa.Column('priority', sa.Integer(), nullable=False,
                  comment='토크나이저 로드 우선순위 (높을수록 우선)'),
        sa.Column('created_at', sa.DateTime(), nullable=True,
                  comment='레코드 생성일시'),
        sa.Column('updated_at', sa.DateTime(), nullable=True,
                  comment='레코드 수정일시'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('term', name='uq_legal_terms_term'),
        comment='법률 용어 사전 (MeCab 토크나이저 보강용)',
    )

    # 단일 컬럼 인덱스
    op.create_index('idx_legal_terms_term', 'legal_terms',
                    ['term'], unique=False)
    op.create_index('idx_legal_terms_source_code', 'legal_terms',
                    ['source_code'], unique=False)
    op.create_index('idx_legal_terms_is_korean', 'legal_terms',
                    ['is_korean_only'], unique=False)
    op.create_index('idx_legal_terms_priority', 'legal_terms',
                    ['priority'], unique=False)
    op.create_index('idx_legal_terms_length', 'legal_terms',
                    ['term_length'], unique=False)


def downgrade() -> None:
    # 인덱스 삭제 (역순)
    op.drop_index('idx_legal_terms_length', table_name='legal_terms')
    op.drop_index('idx_legal_terms_priority', table_name='legal_terms')
    op.drop_index('idx_legal_terms_is_korean', table_name='legal_terms')
    op.drop_index('idx_legal_terms_source_code', table_name='legal_terms')
    op.drop_index('idx_legal_terms_term', table_name='legal_terms')

    # 테이블 삭제
    op.drop_table('legal_terms')
