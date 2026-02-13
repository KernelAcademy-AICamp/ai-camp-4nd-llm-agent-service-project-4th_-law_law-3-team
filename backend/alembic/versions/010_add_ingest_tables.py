"""add ingest tables: 8개 신규 인제스트 테이블 생성

Revision ID: 010
Revises: 009
Create Date: 2026-02-13

신규 테이블:
- admin_rule_documents (행정규칙)
- constitutional_documents (헌법재판소 결정례)
- administration_documents (행정심판례)
- legislation_documents (법령해석례)
- treaty_documents (조약)
- decisions_committee_documents (위원회 결정례)
- interpretation_ministry_documents (부처 유권해석)
- special_admin_appeal_documents (특별행정심판 재결례)
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. admin_rule_documents (행정규칙)
    op.create_table(
        "admin_rule_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("admin_rule_id", sa.String(100), nullable=False, comment="행정규칙 ID"),
        sa.Column("serial_number", sa.String(100), nullable=True, comment="행정규칙 일련번호"),
        sa.Column("admin_rule_name", sa.Text(), nullable=True, comment="행정규칙명"),
        sa.Column("admin_rule_type", sa.String(100), nullable=True, comment="행정규칙종류"),
        sa.Column("ministry", sa.String(200), nullable=True, comment="소관부처명"),
        sa.Column("content", sa.Text(), nullable=True, comment="조문내용"),
        sa.Column("supplementary", sa.Text(), nullable=True, comment="부칙내용"),
        sa.Column("ai_summary", sa.Text(), nullable=True, comment="AI 생성 요약"),
        sa.Column("created_at", sa.DateTime(), nullable=True, comment="레코드 생성일시"),
        sa.Column("updated_at", sa.DateTime(), nullable=True, comment="레코드 수정일시"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("admin_rule_id"),
    )
    op.create_index("ix_admin_rule_documents_admin_rule_id", "admin_rule_documents", ["admin_rule_id"])
    op.create_index("ix_admin_rule_documents_ministry", "admin_rule_documents", ["ministry"])

    # 2. constitutional_documents (헌법재판소 결정례)
    op.create_table(
        "constitutional_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("serial_number", sa.String(100), nullable=False, comment="헌재결정례 일련번호"),
        sa.Column("case_number", sa.String(200), nullable=True, comment="사건번호"),
        sa.Column("case_name", sa.Text(), nullable=True, comment="사건명"),
        sa.Column("case_type_name", sa.String(100), nullable=True, comment="사건종류명"),
        sa.Column("case_type_code", sa.String(20), nullable=True, comment="사건종류코드"),
        sa.Column("decision_date", sa.Date(), nullable=True, comment="종국일자"),
        sa.Column("court_division_code", sa.String(20), nullable=True, comment="재판부구분코드"),
        sa.Column("summary", sa.Text(), nullable=True, comment="결정요지"),
        sa.Column("reasoning", sa.Text(), nullable=True, comment="판시사항"),
        sa.Column("ruling", sa.Text(), nullable=True, comment="주문"),
        sa.Column("full_text", sa.Text(), nullable=True, comment="전문"),
        sa.Column("reason", sa.Text(), nullable=True, comment="이유"),
        sa.Column("reference_provisions", sa.Text(), nullable=True, comment="심판대상조문"),
        sa.Column("reference_statutes", sa.Text(), nullable=True, comment="참조조문"),
        sa.Column("reference_cases", sa.Text(), nullable=True, comment="참조판례"),
        sa.Column("ai_summary", sa.Text(), nullable=True, comment="AI 생성 요약"),
        sa.Column("created_at", sa.DateTime(), nullable=True, comment="레코드 생성일시"),
        sa.Column("updated_at", sa.DateTime(), nullable=True, comment="레코드 수정일시"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("serial_number"),
    )
    op.create_index("ix_constitutional_documents_serial_number", "constitutional_documents", ["serial_number"])
    op.create_index("ix_constitutional_documents_case_number", "constitutional_documents", ["case_number"])
    op.create_index("ix_constitutional_documents_decision_date", "constitutional_documents", ["decision_date"])
    op.create_index("ix_constitutional_documents_case_type_name", "constitutional_documents", ["case_type_name"])

    # 3. administration_documents (행정심판례)
    op.create_table(
        "administration_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("serial_number", sa.String(100), nullable=False, comment="행정심판례 일련번호"),
        sa.Column("case_name", sa.Text(), nullable=True, comment="사건명"),
        sa.Column("case_number", sa.String(200), nullable=True, comment="사건번호"),
        sa.Column("decision_date", sa.Date(), nullable=True, comment="의결일자"),
        sa.Column("disposition_date", sa.Date(), nullable=True, comment="처분일자"),
        sa.Column("disposition_agency", sa.String(200), nullable=True, comment="처분청"),
        sa.Column("adjudication_agency", sa.String(200), nullable=True, comment="재결청"),
        sa.Column("case_type_name", sa.String(100), nullable=True, comment="재결례유형명"),
        sa.Column("case_type_code", sa.String(20), nullable=True, comment="재결례유형코드"),
        sa.Column("ruling", sa.Text(), nullable=True, comment="주문"),
        sa.Column("claim", sa.Text(), nullable=True, comment="청구취지"),
        sa.Column("reason", sa.Text(), nullable=True, comment="이유"),
        sa.Column("adjudication_summary", sa.Text(), nullable=True, comment="재결요지"),
        sa.Column("ai_summary", sa.Text(), nullable=True, comment="AI 생성 요약"),
        sa.Column("created_at", sa.DateTime(), nullable=True, comment="레코드 생성일시"),
        sa.Column("updated_at", sa.DateTime(), nullable=True, comment="레코드 수정일시"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("serial_number"),
    )
    op.create_index("ix_administration_documents_serial_number", "administration_documents", ["serial_number"])
    op.create_index("ix_administration_documents_case_number", "administration_documents", ["case_number"])
    op.create_index("ix_administration_documents_decision_date", "administration_documents", ["decision_date"])

    # 4. legislation_documents (법령해석례)
    op.create_table(
        "legislation_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("serial_number", sa.String(100), nullable=False, comment="법령해석례 일련번호"),
        sa.Column("case_name", sa.Text(), nullable=True, comment="안건명"),
        sa.Column("case_number", sa.String(200), nullable=True, comment="안건번호"),
        sa.Column("interpretation_date", sa.Date(), nullable=True, comment="해석일자"),
        sa.Column("registration_date", sa.String(50), nullable=True, comment="등록일시"),
        sa.Column("interpretation_agency_code", sa.String(20), nullable=True, comment="해석기관코드"),
        sa.Column("interpretation_agency_name", sa.String(200), nullable=True, comment="해석기관명"),
        sa.Column("inquiry_agency_code", sa.String(20), nullable=True, comment="질의기관코드"),
        sa.Column("inquiry_agency_name", sa.String(200), nullable=True, comment="질의기관명"),
        sa.Column("management_agency_code", sa.String(20), nullable=True, comment="관리기관코드"),
        sa.Column("inquiry", sa.Text(), nullable=True, comment="질의요지"),
        sa.Column("answer", sa.Text(), nullable=True, comment="회답"),
        sa.Column("reason", sa.Text(), nullable=True, comment="이유"),
        sa.Column("ai_summary", sa.Text(), nullable=True, comment="AI 생성 요약"),
        sa.Column("created_at", sa.DateTime(), nullable=True, comment="레코드 생성일시"),
        sa.Column("updated_at", sa.DateTime(), nullable=True, comment="레코드 수정일시"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("serial_number"),
    )
    op.create_index("ix_legislation_documents_serial_number", "legislation_documents", ["serial_number"])
    op.create_index("ix_legislation_documents_interpretation_date", "legislation_documents", ["interpretation_date"])

    # 5. treaty_documents (조약)
    op.create_table(
        "treaty_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("serial_number", sa.String(100), nullable=False, comment="조약 일련번호"),
        sa.Column("treaty_number", sa.String(50), nullable=True, comment="조약번호"),
        sa.Column("treaty_name_kr", sa.Text(), nullable=True, comment="조약명 (한글)"),
        sa.Column("treaty_name_en", sa.Text(), nullable=True, comment="조약명 (영문)"),
        sa.Column("treaty_type_code", sa.String(20), nullable=True, comment="조약구분코드"),
        sa.Column("country_code", sa.String(20), nullable=True, comment="국가코드"),
        sa.Column("country_name", sa.String(200), nullable=True, comment="국가명"),
        sa.Column("bilateral_type_code", sa.String(20), nullable=True, comment="양자다자구분코드"),
        sa.Column("bilateral_type_name", sa.String(50), nullable=True, comment="양자다자구분명"),
        sa.Column("signing_date", sa.String(50), nullable=True, comment="서명일자"),
        sa.Column("signing_place", sa.String(200), nullable=True, comment="서명장소"),
        sa.Column("effective_date", sa.String(50), nullable=True, comment="발효일자"),
        sa.Column("proclamation_number", sa.String(100), nullable=True, comment="공포번호"),
        sa.Column("proclamation_date", sa.String(50), nullable=True, comment="공포일자"),
        sa.Column("parliament_consent", sa.String(10), nullable=True, comment="국회비준동의여부"),
        sa.Column("parliament_consent_date", sa.String(50), nullable=True, comment="국회동의일자"),
        sa.Column("content", sa.Text(), nullable=True, comment="조약내용"),
        sa.Column("note", sa.Text(), nullable=True, comment="비고"),
        sa.Column("ai_summary", sa.Text(), nullable=True, comment="AI 생성 요약"),
        sa.Column("created_at", sa.DateTime(), nullable=True, comment="레코드 생성일시"),
        sa.Column("updated_at", sa.DateTime(), nullable=True, comment="레코드 수정일시"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("serial_number"),
    )
    op.create_index("ix_treaty_documents_serial_number", "treaty_documents", ["serial_number"])
    op.create_index("ix_treaty_documents_treaty_number", "treaty_documents", ["treaty_number"])

    # 6. decisions_committee_documents (위원회 결정례)
    op.create_table(
        "decisions_committee_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("serial_number", sa.String(100), nullable=False, comment="결정문 일련번호"),
        sa.Column("ai_summary", sa.Text(), nullable=True, comment="AI 생성 요약 (결정문요약)"),
        sa.Column("committee_name", sa.String(200), nullable=True, comment="위원회명"),
        sa.Column("case_name", sa.Text(), nullable=True, comment="안건명 (사건명/안건명/제목 통합)"),
        sa.Column("decision_date", sa.String(50), nullable=True, comment="의결일자 (의결일자/의결일/결정일자/등록일 통합)"),
        sa.Column("case_number", sa.String(200), nullable=True, comment="사건번호"),
        sa.Column("decision_number", sa.String(200), nullable=True, comment="결정번호/의결번호/의안번호 통합"),
        sa.Column("ruling", sa.Text(), nullable=True, comment="주문"),
        sa.Column("reason", sa.Text(), nullable=True, comment="이유"),
        sa.Column("decision_summary", sa.Text(), nullable=True, comment="결정요지/판단요지/판정요지/판정사항 통합"),
        sa.Column("claim", sa.Text(), nullable=True, comment="청구취지"),
        sa.Column("appendix", sa.Text(), nullable=True, comment="별지"),
        sa.Column("action_reason", sa.Text(), nullable=True, comment="조치이유 (금융위/증권선물위)"),
        sa.Column("action_content", sa.Text(), nullable=True, comment="조치내용 (금융위/증권선물위)"),
        sa.Column("full_text", sa.Text(), nullable=True, comment="의결문/결정례전문/내용 통합"),
        sa.Column("created_at", sa.DateTime(), nullable=True, comment="레코드 생성일시"),
        sa.Column("updated_at", sa.DateTime(), nullable=True, comment="레코드 수정일시"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("serial_number"),
    )
    op.create_index("ix_decisions_committee_documents_serial_number", "decisions_committee_documents", ["serial_number"])
    op.create_index("ix_decisions_committee_documents_committee_name", "decisions_committee_documents", ["committee_name"])
    op.create_index("ix_decisions_committee_documents_decision_date", "decisions_committee_documents", ["decision_date"])

    # 7. interpretation_ministry_documents (부처 유권해석)
    op.create_table(
        "interpretation_ministry_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("serial_number", sa.String(100), nullable=False, comment="법령해석 일련번호"),
        sa.Column("case_name", sa.Text(), nullable=True, comment="안건명"),
        sa.Column("case_number", sa.String(200), nullable=True, comment="안건번호 (일부 부처만 보유)"),
        sa.Column("interpretation_date", sa.Date(), nullable=True, comment="해석일자"),
        sa.Column("inquiry", sa.Text(), nullable=True, comment="질의요지"),
        sa.Column("related_law", sa.Text(), nullable=True, comment="관련법령"),
        sa.Column("answer", sa.Text(), nullable=True, comment="회답"),
        sa.Column("reason", sa.Text(), nullable=True, comment="이유 (일부 부처만 보유)"),
        sa.Column("ministry_name", sa.String(200), nullable=True, comment="부처명"),
        sa.Column("ai_summary", sa.Text(), nullable=True, comment="AI 생성 요약"),
        sa.Column("created_at", sa.DateTime(), nullable=True, comment="레코드 생성일시"),
        sa.Column("updated_at", sa.DateTime(), nullable=True, comment="레코드 수정일시"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("serial_number"),
    )
    op.create_index("ix_interpretation_ministry_documents_serial_number", "interpretation_ministry_documents", ["serial_number"])
    op.create_index("ix_interpretation_ministry_documents_interpretation_date", "interpretation_ministry_documents", ["interpretation_date"])
    op.create_index("ix_interpretation_ministry_documents_ministry_name", "interpretation_ministry_documents", ["ministry_name"])

    # 8. special_admin_appeal_documents (특별행정심판 재결례)
    op.create_table(
        "special_admin_appeal_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("serial_number", sa.String(100), nullable=False, comment="특별행정심판 재결례 일련번호"),
        sa.Column("case_name", sa.Text(), nullable=True, comment="사건명"),
        sa.Column("case_number", sa.String(200), nullable=True, comment="재결번호"),
        sa.Column("decision_date", sa.Date(), nullable=True, comment="의결일자"),
        sa.Column("adjudication_agency", sa.String(200), nullable=True, comment="재결청"),
        sa.Column("case_type", sa.String(100), nullable=True, comment="재결례유형명"),
        sa.Column("ruling", sa.Text(), nullable=True, comment="주문"),
        sa.Column("claim", sa.Text(), nullable=True, comment="청구취지"),
        sa.Column("reason", sa.Text(), nullable=True, comment="이유"),
        # 조세심판원 특화
        sa.Column("adjudication_summary", sa.Text(), nullable=True, comment="재결요지 (조세심판원)"),
        sa.Column("related_rulings", sa.Text(), nullable=True, comment="참조결정 (조세심판원)"),
        sa.Column("following_rulings", sa.Text(), nullable=True, comment="따른결정 (조세심판원)"),
        sa.Column("tax_category", sa.String(200), nullable=True, comment="세목 (조세심판원)"),
        sa.Column("related_law", sa.Text(), nullable=True, comment="관련법령 (조세심판원)"),
        # 해양안전심판원 특화
        sa.Column("vessel_type", sa.String(200), nullable=True, comment="선박유형 (해양안전심판원)"),
        sa.Column("accident_type", sa.String(100), nullable=True, comment="사고유형 (해양안전심판원)"),
        sa.Column("tribunal_location", sa.String(100), nullable=True, comment="해심위치 (해양안전심판원)"),
        sa.Column("related_persons", sa.Text(), nullable=True, comment="해양사고관련자 (해양안전심판원)"),
        sa.Column("appendix", sa.Text(), nullable=True, comment="별지 (해양안전심판원)"),
        sa.Column("retrial_notice", sa.Text(), nullable=True, comment="재심청구안내 (해양안전심판원)"),
        sa.Column("ai_summary", sa.Text(), nullable=True, comment="AI 생성 요약"),
        sa.Column("created_at", sa.DateTime(), nullable=True, comment="레코드 생성일시"),
        sa.Column("updated_at", sa.DateTime(), nullable=True, comment="레코드 수정일시"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("serial_number"),
    )
    op.create_index("ix_special_admin_appeal_documents_serial_number", "special_admin_appeal_documents", ["serial_number"])
    op.create_index("ix_special_admin_appeal_documents_case_number", "special_admin_appeal_documents", ["case_number"])
    op.create_index("ix_special_admin_appeal_documents_decision_date", "special_admin_appeal_documents", ["decision_date"])
    op.create_index("ix_special_admin_appeal_documents_adjudication_agency", "special_admin_appeal_documents", ["adjudication_agency"])
    op.create_index("ix_special_admin_appeal_documents_case_type", "special_admin_appeal_documents", ["case_type"])


def downgrade() -> None:
    op.drop_table("special_admin_appeal_documents")
    op.drop_table("interpretation_ministry_documents")
    op.drop_table("decisions_committee_documents")
    op.drop_table("treaty_documents")
    op.drop_table("legislation_documents")
    op.drop_table("administration_documents")
    op.drop_table("constitutional_documents")
    op.drop_table("admin_rule_documents")
