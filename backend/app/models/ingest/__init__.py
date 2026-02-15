"""
인제스트 전용 ORM 모델

data/ 데이터를 PostgreSQL에 저장하기 위한 테이블 정의.
적재 로직은 scripts/ingest/types/ 에 위치.
"""

from app.models.ingest.admin_rule_document import AdminRuleDocument
from app.models.ingest.administration_document import AdministrationDocument
from app.models.ingest.constitutional_document import ConstitutionalDocument
from app.models.ingest.dec_civil_rights_document import DecCivilRightsDocument
from app.models.ingest.dec_employment_document import DecEmploymentDocument
from app.models.ingest.dec_environment_document import DecEnvironmentDocument
from app.models.ingest.dec_fair_trade_document import DecFairTradeDocument
from app.models.ingest.dec_financial_document import DecFinancialDocument
from app.models.ingest.dec_human_rights_document import DecHumanRightsDocument
from app.models.ingest.dec_industrial_document import DecIndustrialDocument
from app.models.ingest.dec_labor_document import DecLaborDocument
from app.models.ingest.dec_privacy_document import DecPrivacyDocument
from app.models.ingest.dec_securities_document import DecSecuritiesDocument
from app.models.ingest.interpretation_ministry_document import (
    InterpretationMinistryDocument,
)
from app.models.ingest.legislation_document import LegislationDocument
from app.models.ingest.special_admin_appeal_document import (
    SpecialAdminAppealDocument,
)
from app.models.ingest.treaty_document import TreatyDocument

__all__ = [
    "AdminRuleDocument",
    "ConstitutionalDocument",
    "AdministrationDocument",
    "DecCivilRightsDocument",
    "DecEnvironmentDocument",
    "DecFinancialDocument",
    "DecPrivacyDocument",
    "DecSecuritiesDocument",
    "DecEmploymentDocument",
    "DecFairTradeDocument",
    "DecHumanRightsDocument",
    "DecIndustrialDocument",
    "DecLaborDocument",
    "InterpretationMinistryDocument",
    "LegislationDocument",
    "SpecialAdminAppealDocument",
    "TreatyDocument",
]
