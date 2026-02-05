"""add trial statistics table

Revision ID: 005
Revises: 004
Create Date: 2026-02-05

재판 통계 데이터 PostgreSQL 마이그레이션
- trial_statistics 테이블: 법원별/카테고리별/연도별 사건 처리 건수
- UNIQUE 제약조건: (category, court_name, year)
- 단일/복합 인덱스: 카테고리, 법원, 연도, 상위법원 조회 최적화
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'trial_statistics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False,
                  comment='사건 카테고리 (민사_본안_단독, 형사_공판 등)'),
        sa.Column('court_name', sa.String(length=100), nullable=False,
                  comment='법원명 (서울중앙지방법원, 고양지원 등)'),
        sa.Column('court_type', sa.String(length=20), nullable=False,
                  comment='법원 유형 (main: 본원, branch: 지원)'),
        sa.Column('parent_court', sa.String(length=100), nullable=True,
                  comment='지원의 상위 본원명 (본원은 NULL)'),
        sa.Column('year', sa.Integer(), nullable=False,
                  comment='연도 (2015~2024)'),
        sa.Column('case_count', sa.Integer(), nullable=False,
                  comment='사건 처리 건수'),
        sa.Column('created_at', sa.DateTime(), nullable=True,
                  comment='레코드 생성일시'),
        sa.Column('updated_at', sa.DateTime(), nullable=True,
                  comment='레코드 수정일시'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('category', 'court_name', 'year',
                            name='uq_trial_stats_cat_court_year'),
        comment='재판 통계 (법원별/카테고리별/연도별 사건 처리 건수)',
    )

    # 단일 컬럼 인덱스
    op.create_index('idx_trial_stats_category', 'trial_statistics',
                    ['category'], unique=False)
    op.create_index('idx_trial_stats_court', 'trial_statistics',
                    ['court_name'], unique=False)
    op.create_index('idx_trial_stats_year', 'trial_statistics',
                    ['year'], unique=False)
    op.create_index('idx_trial_stats_parent', 'trial_statistics',
                    ['parent_court'], unique=False)

    # 복합 인덱스
    op.create_index('idx_trial_stats_cat_year', 'trial_statistics',
                    ['category', 'year'], unique=False)


def downgrade() -> None:
    # 인덱스 삭제 (역순)
    op.drop_index('idx_trial_stats_cat_year', table_name='trial_statistics')
    op.drop_index('idx_trial_stats_parent', table_name='trial_statistics')
    op.drop_index('idx_trial_stats_year', table_name='trial_statistics')
    op.drop_index('idx_trial_stats_court', table_name='trial_statistics')
    op.drop_index('idx_trial_stats_category', table_name='trial_statistics')

    # 테이블 삭제
    op.drop_table('trial_statistics')
