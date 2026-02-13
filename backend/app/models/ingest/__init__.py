"""
인제스트 전용 ORM 모델

data/ingest_source/ 데이터를 PostgreSQL에 저장하기 위한 테이블 정의.
적재 로직은 scripts/ingest/types/ 에 위치.
"""

from app.models.ingest.admin_rule_document import AdminRuleDocument
from app.models.ingest.administration_document import AdministrationDocument
from app.models.ingest.constitutional_document import ConstitutionalDocument
from app.models.ingest.decisions_committee_document import (
    DecisionsCommitteeDocument,
)
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
    "LegislationDocument",
    "TreatyDocument",
    "DecisionsCommitteeDocument",
    "InterpretationMinistryDocument",
    "SpecialAdminAppealDocument",
]
