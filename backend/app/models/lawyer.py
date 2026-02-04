"""
변호사 ORM 모델

lawyers 테이블: data/lawyers_with_coords.json 데이터를 PostgreSQL에 저장
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY

from app.core.database import Base


class Lawyer(Base):
    """변호사 정보 테이블"""

    __tablename__ = "lawyers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    detail_id = Column(
        String(100), unique=True, nullable=True, index=True,
        comment="대한변협 상세 ID",
    )
    name = Column(String(100), nullable=False, index=True, comment="변호사 이름")
    status = Column(String(20), nullable=True, index=True, comment="활동 상태")
    birth_year = Column(String(10), nullable=True, comment="출생연도")
    photo_url = Column(Text, nullable=True, comment="프로필 사진 URL")
    office_name = Column(
        String(500), nullable=True, index=True, comment="소속 사무소명",
    )
    address = Column(Text, nullable=True, comment="사무소 주소")
    phone = Column(String(50), nullable=True, comment="전화번호")
    fax = Column(String(50), nullable=True, comment="팩스번호")
    email = Column(String(200), nullable=True, comment="이메일")
    birthdate = Column(String(20), nullable=True, comment="생년월일")
    local_bar = Column(String(100), nullable=True, comment="소속 지방변호사회")
    qualification = Column(Text, nullable=True, comment="자격 정보")
    klaw_url = Column(Text, nullable=True, comment="대한변협 프로필 URL")

    # 좌표
    latitude = Column(Float, nullable=True, comment="위도")
    longitude = Column(Float, nullable=True, comment="경도")

    # 전문분야 (PostgreSQL ARRAY + GIN 인덱스)
    specialties: Any = Column(
        ARRAY(Text), server_default="{}", nullable=False,
        comment="전문분야 목록",
    )

    # 비정규화 지역 필드 (통계 쿼리 최적화)
    province = Column(String(20), nullable=True, index=True, comment="시/도")
    district = Column(String(50), nullable=True, index=True, comment="시/군/구")
    region = Column(String(50), nullable=True, index=True, comment="시도 시군구 결합")

    # 타임스탬프
    created_at = Column(
        DateTime, default=datetime.utcnow, comment="레코드 생성일시",
    )
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        comment="레코드 수정일시",
    )

    __table_args__ = (
        # 좌표 복합 인덱스 (바운딩 박스 검색 최적화)
        Index("idx_lawyers_coords", "latitude", "longitude"),
        # 전문분야 GIN 인덱스 (@>, && 연산 최적화)
        Index(
            "idx_lawyers_specialties", "specialties",
            postgresql_using="gin",
        ),
        # 지역 인덱스 (GROUP BY 통계 쿼리 최적화)
        Index("idx_lawyers_region", "region"),
        {"comment": "변호사 정보 (data/lawyers_with_coords.json 마이그레이션)"},
    )

    def __repr__(self) -> str:
        return f"<Lawyer(id={self.id}, name='{self.name}', region='{self.region}')>"
