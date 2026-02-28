"""add graph tables (Neo4j → PostgreSQL)

Revision ID: 018
Revises: 017
Create Date: 2026-02-27

Neo4j 그래프 데이터를 PostgreSQL로 마이그레이션하기 위한 테이블:
- law_documents에 abbreviation, citation_count 컬럼 추가
- statute_aliases: 비공식 약칭
- statute_hierarchy: 법령 계급 (시행령→법률)
- statute_relations: 법령 관련 관계
- case_statute_citations: 판례→법령 인용
- case_case_citations: 판례→판례 인용
- pg_trgm GIN 인덱스: 법령 퍼지 검색
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers
revision: str = "018"
down_revision: Union[str, Sequence[str], None] = "017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0. pg_trgm 확장 (법령 퍼지 검색용)
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # 1. law_documents에 abbreviation, citation_count 컬럼 추가
    op.add_column(
        "law_documents",
        sa.Column(
            "abbreviation",
            sa.String(200),
            nullable=True,
            comment="법령 약칭 (공식)",
        ),
    )
    op.add_column(
        "law_documents",
        sa.Column(
            "citation_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="판례 인용 횟수",
        ),
    )
    op.create_index("idx_law_docs_abbreviation", "law_documents", ["abbreviation"])

    # 2. statute_aliases 테이블
    op.create_table(
        "statute_aliases",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "law_doc_id",
            sa.Integer(),
            sa.ForeignKey("law_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("alias_name", sa.String(200), nullable=False),
        sa.Column("category", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_sa_alias_name", "statute_aliases", ["alias_name"], unique=True)
    op.create_index("idx_sa_law_doc_id", "statute_aliases", ["law_doc_id"])

    # 3. statute_hierarchy 테이블
    op.create_table(
        "statute_hierarchy",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "child_id",
            sa.Integer(),
            sa.ForeignKey("law_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_id",
            sa.Integer(),
            sa.ForeignKey("law_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("child_id", "parent_id", name="uq_statute_hierarchy"),
    )
    op.create_index("idx_sh_child", "statute_hierarchy", ["child_id"])
    op.create_index("idx_sh_parent", "statute_hierarchy", ["parent_id"])

    # 4. statute_relations 테이블
    op.create_table(
        "statute_relations",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "law_doc_id_1",
            sa.Integer(),
            sa.ForeignKey("law_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "law_doc_id_2",
            sa.Integer(),
            sa.ForeignKey("law_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "law_doc_id_1 < law_doc_id_2",
            name="ck_statute_relations_order",
        ),
        sa.UniqueConstraint("law_doc_id_1", "law_doc_id_2", name="uq_statute_relations"),
    )
    op.create_index("idx_sr_id_1", "statute_relations", ["law_doc_id_1"])
    op.create_index("idx_sr_id_2", "statute_relations", ["law_doc_id_2"])

    # 5. case_statute_citations 테이블
    op.create_table(
        "case_statute_citations",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "case_doc_id",
            sa.Integer(),
            sa.ForeignKey("precedent_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "law_doc_id",
            sa.Integer(),
            sa.ForeignKey("law_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("case_doc_id", "law_doc_id", name="uq_case_statute_citation"),
    )
    op.create_index("idx_csc_case", "case_statute_citations", ["case_doc_id"])
    op.create_index("idx_csc_statute", "case_statute_citations", ["law_doc_id"])

    # 6. case_case_citations 테이블
    op.create_table(
        "case_case_citations",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "citing_case_id",
            sa.Integer(),
            sa.ForeignKey("precedent_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "cited_case_id",
            sa.Integer(),
            sa.ForeignKey("precedent_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("citing_case_id", "cited_case_id", name="uq_case_case_citation"),
    )
    op.create_index("idx_ccc_citing", "case_case_citations", ["citing_case_id"])
    op.create_index("idx_ccc_cited", "case_case_citations", ["cited_case_id"])

    # 7. Trigram GIN 인덱스 (법령 퍼지 검색)
    op.execute(
        "CREATE INDEX idx_law_docs_name_trgm ON law_documents "
        "USING gin (law_name gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX idx_law_docs_abbr_trgm ON law_documents "
        "USING gin (abbreviation gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX idx_sa_alias_trgm ON statute_aliases "
        "USING gin (alias_name gin_trgm_ops)"
    )


def downgrade() -> None:
    # Trigram 인덱스 삭제
    op.execute("DROP INDEX IF EXISTS idx_sa_alias_trgm")
    op.execute("DROP INDEX IF EXISTS idx_law_docs_abbr_trgm")
    op.execute("DROP INDEX IF EXISTS idx_law_docs_name_trgm")

    # 테이블 삭제 (역순)
    op.drop_table("case_case_citations")
    op.drop_table("case_statute_citations")
    op.drop_table("statute_relations")
    op.drop_table("statute_hierarchy")
    op.drop_table("statute_aliases")

    # law_documents 컬럼 삭제
    op.drop_index("idx_law_docs_abbreviation", table_name="law_documents")
    op.drop_column("law_documents", "citation_count")
    op.drop_column("law_documents", "abbreviation")
