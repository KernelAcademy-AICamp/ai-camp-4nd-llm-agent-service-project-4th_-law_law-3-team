"""add lawyers table

Revision ID: 004
Revises: 003
Create Date: 2026-02-05

변호사 데이터 PostgreSQL 마이그레이션
- lawyers 테이블: data/lawyers_with_coords.json (17,326건)
- GIN 인덱스: specialties ARRAY 검색 최적화
- 좌표 복합 인덱스: 바운딩 박스 검색 최적화
- 비정규화 지역 필드: 통계 GROUP BY 최적화
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'lawyers',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('detail_id', sa.String(length=100), nullable=True,
                  comment='대한변협 상세 ID'),
        sa.Column('name', sa.String(length=100), nullable=False,
                  comment='변호사 이름'),
        sa.Column('status', sa.String(length=20), nullable=True,
                  comment='활동 상태'),
        sa.Column('birth_year', sa.String(length=10), nullable=True,
                  comment='출생연도'),
        sa.Column('photo_url', sa.Text(), nullable=True,
                  comment='프로필 사진 URL'),
        sa.Column('office_name', sa.String(length=500), nullable=True,
                  comment='소속 사무소명'),
        sa.Column('address', sa.Text(), nullable=True,
                  comment='사무소 주소'),
        sa.Column('phone', sa.String(length=50), nullable=True,
                  comment='전화번호'),
        sa.Column('fax', sa.String(length=50), nullable=True,
                  comment='팩스번호'),
        sa.Column('email', sa.String(length=200), nullable=True,
                  comment='이메일'),
        sa.Column('birthdate', sa.String(length=20), nullable=True,
                  comment='생년월일'),
        sa.Column('local_bar', sa.String(length=100), nullable=True,
                  comment='소속 지방변호사회'),
        sa.Column('qualification', sa.Text(), nullable=True,
                  comment='자격 정보'),
        sa.Column('klaw_url', sa.Text(), nullable=True,
                  comment='대한변협 프로필 URL'),
        sa.Column('latitude', sa.Float(), nullable=True,
                  comment='위도'),
        sa.Column('longitude', sa.Float(), nullable=True,
                  comment='경도'),
        sa.Column('specialties', postgresql.ARRAY(sa.Text()),
                  server_default='{}', nullable=False,
                  comment='전문분야 목록'),
        sa.Column('province', sa.String(length=20), nullable=True,
                  comment='시/도'),
        sa.Column('district', sa.String(length=50), nullable=True,
                  comment='시/군/구'),
        sa.Column('region', sa.String(length=50), nullable=True,
                  comment='시도 시군구 결합'),
        sa.Column('created_at', sa.DateTime(), nullable=True,
                  comment='레코드 생성일시'),
        sa.Column('updated_at', sa.DateTime(), nullable=True,
                  comment='레코드 수정일시'),
        sa.PrimaryKeyConstraint('id'),
        comment='변호사 정보 (data/lawyers_with_coords.json 마이그레이션)',
    )

    # 단일 컬럼 인덱스
    op.create_index('idx_lawyers_detail_id', 'lawyers', ['detail_id'], unique=True)
    op.create_index('idx_lawyers_name', 'lawyers', ['name'], unique=False)
    op.create_index('idx_lawyers_status', 'lawyers', ['status'], unique=False)
    op.create_index('idx_lawyers_office_name', 'lawyers', ['office_name'], unique=False)
    op.create_index('idx_lawyers_province', 'lawyers', ['province'], unique=False)
    op.create_index('idx_lawyers_district', 'lawyers', ['district'], unique=False)

    # 복합/특수 인덱스
    op.create_index('idx_lawyers_coords', 'lawyers', ['latitude', 'longitude'], unique=False)
    op.create_index('idx_lawyers_region', 'lawyers', ['region'], unique=False)
    op.create_index(
        'idx_lawyers_specialties', 'lawyers', ['specialties'],
        unique=False, postgresql_using='gin',
    )


def downgrade() -> None:
    # 인덱스 삭제 (역순)
    op.drop_index('idx_lawyers_specialties', table_name='lawyers')
    op.drop_index('idx_lawyers_region', table_name='lawyers')
    op.drop_index('idx_lawyers_coords', table_name='lawyers')
    op.drop_index('idx_lawyers_district', table_name='lawyers')
    op.drop_index('idx_lawyers_province', table_name='lawyers')
    op.drop_index('idx_lawyers_office_name', table_name='lawyers')
    op.drop_index('idx_lawyers_status', table_name='lawyers')
    op.drop_index('idx_lawyers_name', table_name='lawyers')
    op.drop_index('idx_lawyers_detail_id', table_name='lawyers')

    # 테이블 삭제
    op.drop_table('lawyers')
