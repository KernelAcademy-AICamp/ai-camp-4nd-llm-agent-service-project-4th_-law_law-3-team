"""
변호사 찾기 에이전트

위치 기반 변호사 추천 기능
"""

import re
from typing import Any

from app.common.agent_base import ActionType, AgentResponse, BaseAgent, ChatAction
from app.modules.lawyer_finder.service import find_nearby_lawyers, search_lawyers


# 지역명 → 좌표 매핑 (주요 지역)
DISTRICT_COORDS: dict[str, dict[str, float]] = {
    "강남": {"latitude": 37.4979, "longitude": 127.0276},
    "강남구": {"latitude": 37.4979, "longitude": 127.0276},
    "서초": {"latitude": 37.4837, "longitude": 127.0324},
    "서초구": {"latitude": 37.4837, "longitude": 127.0324},
    "종로": {"latitude": 37.5735, "longitude": 126.9790},
    "종로구": {"latitude": 37.5735, "longitude": 126.9790},
    "중구": {"latitude": 37.5640, "longitude": 126.9975},
    "마포": {"latitude": 37.5538, "longitude": 126.9096},
    "마포구": {"latitude": 37.5538, "longitude": 126.9096},
    "영등포": {"latitude": 37.5264, "longitude": 126.8963},
    "영등포구": {"latitude": 37.5264, "longitude": 126.8963},
    "송파": {"latitude": 37.5048, "longitude": 127.1144},
    "송파구": {"latitude": 37.5048, "longitude": 127.1144},
    "부산": {"latitude": 35.1796, "longitude": 129.0756},
    "대구": {"latitude": 35.8714, "longitude": 128.6014},
    "인천": {"latitude": 37.4563, "longitude": 126.7052},
    "광주": {"latitude": 35.1595, "longitude": 126.8526},
    "대전": {"latitude": 36.3504, "longitude": 127.3845},
    "울산": {"latitude": 35.5384, "longitude": 129.3114},
    "수원": {"latitude": 37.2636, "longitude": 127.0286},
    "성남": {"latitude": 37.4200, "longitude": 127.1267},
    "고양": {"latitude": 37.6584, "longitude": 126.8320},
    "용인": {"latitude": 37.2411, "longitude": 127.1776},
}


def extract_district(message: str) -> str | None:
    """메시지에서 지역명 추출"""
    for district in DISTRICT_COORDS.keys():
        if district in message:
            return district
    return None


def format_lawyer_list(lawyers: list[dict], limit: int = 5) -> str:
    """변호사 목록을 읽기 좋은 텍스트로 변환"""
    if not lawyers:
        return "검색 결과가 없습니다."

    lines = []
    for i, lawyer in enumerate(lawyers[:limit], 1):
        name = lawyer.get("name", "이름 없음")
        office = lawyer.get("office_name") or "사무소 정보 없음"
        address = lawyer.get("address") or ""
        distance = lawyer.get("distance")

        line = f"**{i}. {name}**"
        if office != "사무소 정보 없음":
            line += f"\n   - 사무소: {office}"
        if address:
            line += f"\n   - 주소: {address}"
        if distance is not None:
            line += f"\n   - 거리: {distance}km"

        lines.append(line)

    return "\n\n".join(lines)


class LawyerFinderAgent(BaseAgent):
    """변호사 찾기 에이전트"""

    @property
    def name(self) -> str:
        return "lawyer_finder"

    @property
    def description(self) -> str:
        return "위치 기반 변호사 추천"

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResponse:
        """변호사 검색 처리"""
        session_data = session_data or {}

        # 1. 메시지에서 지역명 추출 시도
        district = extract_district(message)

        # 2. 위치 결정 (우선순위: 메시지 지역명 > 사용자 위치 > 위치 요청)
        latitude = None
        longitude = None
        location_source = None

        if district and district in DISTRICT_COORDS:
            coords = DISTRICT_COORDS[district]
            latitude = coords["latitude"]
            longitude = coords["longitude"]
            location_source = district
        elif user_location:
            latitude = user_location.get("latitude")
            longitude = user_location.get("longitude")
            location_source = "현재 위치"

        # 3. 위치가 없으면 위치 요청
        if latitude is None or longitude is None:
            return AgentResponse(
                message="변호사를 찾으려면 위치 정보가 필요합니다.\n\n"
                "아래 버튼을 눌러 현재 위치를 공유하거나, "
                "특정 지역명을 알려주세요. (예: '강남구 변호사 추천해줘')",
                sources=[],
                actions=[
                    ChatAction(
                        type=ActionType.REQUEST_LOCATION,
                        label="현재 위치 공유",
                        action="request_location",
                    ),
                ],
                session_data={"active_agent": self.name, "awaiting_location": True},
            )

        # 4. 변호사 검색
        lawyers = find_nearby_lawyers(
            latitude=latitude,
            longitude=longitude,
            radius_m=5000,  # 5km 반경
            limit=10,
        )

        if not lawyers:
            # 범위 확대해서 재검색
            lawyers = find_nearby_lawyers(
                latitude=latitude,
                longitude=longitude,
                radius_m=10000,  # 10km 반경
                limit=10,
            )

        # 5. 응답 생성
        if lawyers:
            lawyer_text = format_lawyer_list(lawyers, limit=5)
            response = f"**{location_source}** 주변 변호사를 찾았습니다:\n\n{lawyer_text}"

            if len(lawyers) > 5:
                response += f"\n\n(총 {len(lawyers)}명 중 5명 표시)"

            actions = [
                ChatAction(
                    type=ActionType.LINK,
                    label="지도에서 보기",
                    url="/lawyer-finder",
                ),
                ChatAction(
                    type=ActionType.BUTTON,
                    label="다른 지역 검색",
                    action="reset_search",
                ),
            ]
        else:
            response = (
                f"{location_source} 주변에서 변호사를 찾지 못했습니다.\n"
                "다른 지역을 검색해보시거나, 범위를 넓혀서 검색해드릴까요?"
            )
            actions = [
                ChatAction(
                    type=ActionType.BUTTON,
                    label="범위 넓혀 검색",
                    action="expand_search",
                ),
            ]

        return AgentResponse(
            message=response,
            sources=[],
            actions=actions,
            session_data={
                "active_agent": self.name,
                "last_search_location": location_source,
                "last_latitude": latitude,
                "last_longitude": longitude,
            },
        )

    def can_handle(self, message: str) -> bool:
        """변호사 찾기 관련 키워드 확인"""
        keywords = ["변호사 찾", "변호사 추천", "근처 변호사", "주변 변호사"]
        return any(kw in message for kw in keywords)
