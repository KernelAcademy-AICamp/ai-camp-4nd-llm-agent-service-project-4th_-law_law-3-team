"""
SQLAlchemy 모델 정의
"""

from app.models.case_case_citation import CaseCaseCitation
from app.models.case_statute_citation import CaseStatuteCitation
from app.models.chat_conversation import ChatConversation, ChatMessage
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
from app.models.law_article import LawArticle
from app.models.law_document import LawDocument
from app.models.lawyer import Lawyer
from app.models.lawyer_persona import (
    LawyerPersonaFeedbackModel,
    LawyerPersonaModel,
)
from app.models.legal_term import LegalTerm
from app.models.news_article import NewsArticle
from app.models.news_article_dlq import NewsArticleDLQ
from app.models.precedent_document import PrecedentDocument
from app.models.statute_alias import StatuteAlias
from app.models.statute_hierarchy import StatuteHierarchy
from app.models.statute_relation import StatuteRelation
from app.models.trial_statistics import TrialStatistics
from app.models.workspace_case import (
    IdentityLink,
    WorkspaceActivityLog,
    WorkspaceCase,
    WorkspaceCaseTimelineItem,
)

__all__ = [
    "ChatConversation",
    "ChatMessage",
    "FtsIndex",
    "IdentityLink",
    "Lawyer",
    "LegalTerm",
    "LawArticle",
    "LawDocument",
    "PrecedentDocument",
    "TrialStatistics",
    # 콘텐츠 마케팅 v2.0
    "LawyerPersonaModel",
    "LawyerPersonaFeedbackModel",
    # 그래프 모델 (PostgreSQL)
    "StatuteHierarchy",
    "StatuteAlias",
    "StatuteRelation",
    "CaseStatuteCitation",
    "CaseCaseCitation",
    # 뉴스 파이프라인
    "NewsArticle",
    "NewsArticleDLQ",
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
    # 워크스페이스 MVP
    "WorkspaceActivityLog",
    "WorkspaceCase",
    "WorkspaceCaseTimelineItem",
]
