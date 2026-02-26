"""
SQLAlchemy 모델 정의
"""

from app.models.fts_index import FtsIndex
from app.models.ingest import (
    AdministrationDocument,
    AdminRuleDocument,
    ConstitutionalDocument,
    DecCivilRightsDocument,
    DecEmploymentDocument,
    DecEnvironmentDocument,
    DecFairTradeDocument,
    DecFinancialDocument,
    DecHumanRightsDocument,
    DecIndustrialDocument,
    DecLaborDocument,
    DecMediaDocument,
    DecPrivacyDocument,
    DecSecuritiesDocument,
    InterpretationMinistryDocument,
    LegislationDocument,
    LocalOrdinanceDocument,
    SpecialAdminAppealDocument,
    TreatyDocument,
)
from app.models.law import Law
from app.models.law_document import LawDocument
from app.models.lawyer import Lawyer
from app.models.legal_document import COMMITTEE_SOURCES, DocType, LegalDocument
from app.models.legal_reference import LegalReference, RefType
from app.models.legal_term import LegalTerm
from app.models.precedent_document import PrecedentDocument
from app.models.trial_statistics import TrialStatistics

__all__ = [
    "FtsIndex",
    "LegalDocument",
    "DocType",
    "COMMITTEE_SOURCES",
    "Law",
    "Lawyer",
    "LegalReference",
    "LegalTerm",
    "RefType",
    "LawDocument",
    "PrecedentDocument",
    "TrialStatistics",
    # ingest 모델
    "AdminRuleDocument",
    "ConstitutionalDocument",
    "AdministrationDocument",
    "LegislationDocument",
    "TreatyDocument",
    "DecCivilRightsDocument",
    "DecEmploymentDocument",
    "DecEnvironmentDocument",
    "DecFairTradeDocument",
    "DecFinancialDocument",
    "DecHumanRightsDocument",
    "DecIndustrialDocument",
    "DecLaborDocument",
    "DecMediaDocument",
    "DecPrivacyDocument",
    "DecSecuritiesDocument",
    "InterpretationMinistryDocument",
    "LocalOrdinanceDocument",
    "SpecialAdminAppealDocument",
]
