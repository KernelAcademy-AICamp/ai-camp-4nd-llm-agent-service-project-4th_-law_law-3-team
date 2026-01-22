"""
변호사 찾기 에이전트

위치 기반 변호사 추천 기능 - 변호사 찾기 페이지로 네비게이션
"""

from typing import Any

from app.common.agent_base import ActionType, AgentResponse, BaseAgent, ChatAction
from app.modules.lawyer_finder.service import find_nearby_lawyers, search_lawyers

# 검색 반경 (미터)
DEFAULT_SEARCH_RADIUS = 3000   # 3km
EXPANDED_SEARCH_RADIUS = 10000  # 10km


# 지역명 → 좌표 매핑 (서울 25개 구 + 주요 도시)
DISTRICT_COORDS: dict[str, dict[str, float]] = {
    # 서울 25개 구
    "강남": {"latitude": 37.4979, "longitude": 127.0276},
    "강남구": {"latitude": 37.4979, "longitude": 127.0276},
    "강동": {"latitude": 37.5301, "longitude": 127.1238},
    "강동구": {"latitude": 37.5301, "longitude": 127.1238},
    "강북": {"latitude": 37.6396, "longitude": 127.0257},
    "강북구": {"latitude": 37.6396, "longitude": 127.0257},
    "강서": {"latitude": 37.5509, "longitude": 126.8495},
    "강서구": {"latitude": 37.5509, "longitude": 126.8495},
    "관악": {"latitude": 37.4784, "longitude": 126.9516},
    "관악구": {"latitude": 37.4784, "longitude": 126.9516},
    "광진": {"latitude": 37.5385, "longitude": 127.0823},
    "광진구": {"latitude": 37.5385, "longitude": 127.0823},
    "구로": {"latitude": 37.4954, "longitude": 126.8874},
    "구로구": {"latitude": 37.4954, "longitude": 126.8874},
    "금천": {"latitude": 37.4519, "longitude": 126.9020},
    "금천구": {"latitude": 37.4519, "longitude": 126.9020},
    "노원": {"latitude": 37.6542, "longitude": 127.0568},
    "노원구": {"latitude": 37.6542, "longitude": 127.0568},
    "도봉": {"latitude": 37.6688, "longitude": 127.0471},
    "도봉구": {"latitude": 37.6688, "longitude": 127.0471},
    "동대문": {"latitude": 37.5744, "longitude": 127.0400},
    "동대문구": {"latitude": 37.5744, "longitude": 127.0400},
    "동작": {"latitude": 37.5124, "longitude": 126.9393},
    "동작구": {"latitude": 37.5124, "longitude": 126.9393},
    "마포": {"latitude": 37.5538, "longitude": 126.9096},
    "마포구": {"latitude": 37.5538, "longitude": 126.9096},
    "서대문": {"latitude": 37.5791, "longitude": 126.9368},
    "서대문구": {"latitude": 37.5791, "longitude": 126.9368},
    "서초": {"latitude": 37.4837, "longitude": 127.0324},
    "서초구": {"latitude": 37.4837, "longitude": 127.0324},
    "성동": {"latitude": 37.5633, "longitude": 127.0371},
    "성동구": {"latitude": 37.5633, "longitude": 127.0371},
    "성북": {"latitude": 37.5894, "longitude": 127.0167},
    "성북구": {"latitude": 37.5894, "longitude": 127.0167},
    "송파": {"latitude": 37.5048, "longitude": 127.1144},
    "송파구": {"latitude": 37.5048, "longitude": 127.1144},
    "양천": {"latitude": 37.5270, "longitude": 126.8561},
    "양천구": {"latitude": 37.5270, "longitude": 126.8561},
    "영등포": {"latitude": 37.5264, "longitude": 126.8963},
    "영등포구": {"latitude": 37.5264, "longitude": 126.8963},
    "용산": {"latitude": 37.5311, "longitude": 126.9810},
    "용산구": {"latitude": 37.5311, "longitude": 126.9810},
    "은평": {"latitude": 37.6027, "longitude": 126.9291},
    "은평구": {"latitude": 37.6027, "longitude": 126.9291},
    "종로": {"latitude": 37.5735, "longitude": 126.9790},
    "종로구": {"latitude": 37.5735, "longitude": 126.9790},
    "중구": {"latitude": 37.5640, "longitude": 126.9975},
    "중랑": {"latitude": 37.6063, "longitude": 127.0925},
    "중랑구": {"latitude": 37.6063, "longitude": 127.0925},
    # 주요 도시
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
    "분당": {"latitude": 37.3595, "longitude": 127.1086},
    "일산": {"latitude": 37.6761, "longitude": 126.7727},
    "판교": {"latitude": 37.3947, "longitude": 127.1119},
}

# 전문분야 키워드 → 카테고리 ID 매핑 (일반 키워드)
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "civil-family": [
        "민사", "가사", "양육권", "재산분할", "유언",
        "채권", "계약", "가족", "친권", "면접교섭"
    ],
    "criminal": [
        "고소", "고발", "폭행", "사기", "횡령", "배임", "성범죄",
        "음주운전", "마약", "절도", "살인", "상해", "협박", "명예훼손"
    ],
    "real-estate": [
        "건축", "임대차", "전세", "분양", "토지", "아파트", "주택", "상가"
    ],
    "labor": [
        "산업재해", "해고", "퇴직금", "임금", "근로계약",
        "노동조합", "부당해고", "직장 내 괴롭힘", "성희롱"
    ],
    "corporate": [
        "기업", "상사", "회사", "법인", "M&A", "주주",
        "이사회", "정관", "스타트업", "벤처"
    ],
    "finance": [
        "자본시장", "투자", "대출", "채무",
        "파산", "회생", "개인회생", "신용"
    ],
    "tax": [
        "세금", "세무", "부가세", "소득세", "법인세",
        "상속세", "증여세", "세무조사"
    ],
    "public": [
        "행정", "공정거래", "공공", "인허가", "규제", "정부",
        "공무원", "지방자치", "환경"
    ],
    "ip": [
        "지식재산", "상표", "디자인", "영업비밀",
        "기술이전", "라이선스"
    ],
    "it-media": [
        "정보통신", "미디어", "콘텐츠", "게임", "인터넷",
        "개인정보", "데이터", "AI", "플랫폼"
    ],
    "medical": [
        "바이오", "의료사고", "의료분쟁", "제약",
        "병원", "헬스케어"
    ],
    "international": [
        "국제", "해외", "무역", "수출입", "외국인", "비자", "이민",
        "국제거래"
    ],
}

# 정확한 전문분야 목록 (DB의 specialties 필드와 매칭되는 값)
# 키워드 → (실제 전문분야명, 카테고리 ID) 매핑
SPECIALTY_KEYWORDS: dict[str, tuple[str, str]] = {
    # civil-family (민사·가사)
    "민사": ("민사법", "civil-family"),
    "민사법": ("민사법", "civil-family"),
    "손해배상": ("손해배상", "civil-family"),
    "민사집행": ("민사집행", "civil-family"),
    "가사": ("가사법", "civil-family"),
    "가사법": ("가사법", "civil-family"),
    "이혼": ("이혼", "civil-family"),
    "상속": ("상속", "civil-family"),
    "성년후견": ("성년후견", "civil-family"),
    "소년": ("소년법", "civil-family"),
    "소년법": ("소년법", "civil-family"),
    # criminal (형사)
    "형사": ("형사법", "criminal"),
    "형사법": ("형사법", "criminal"),
    "군형법": ("군형법", "criminal"),
    "군사": ("군형법", "criminal"),
    # real-estate (부동산·건설)
    "부동산": ("부동산", "real-estate"),
    "건설": ("건설", "real-estate"),
    "임대차": ("임대차관련법", "real-estate"),
    "임대차관련법": ("임대차관련법", "real-estate"),
    "재개발": ("재개발·재건축", "real-estate"),
    "재건축": ("재개발·재건축", "real-estate"),
    "수용": ("수용 및 보상", "real-estate"),
    "보상": ("수용 및 보상", "real-estate"),
    "등기": ("등기·경매", "real-estate"),
    "경매": ("등기·경매", "real-estate"),
    # labor (노동·산재)
    "노동": ("노동법", "labor"),
    "노동법": ("노동법", "labor"),
    "산재": ("산재", "labor"),
    # corporate (기업·상사)
    "회사법": ("회사법", "corporate"),
    "상사법": ("상사법", "corporate"),
    "인수합병": ("인수합병", "corporate"),
    "영업비밀": ("영업비밀", "corporate"),
    "채권추심": ("채권추심", "corporate"),
    # finance (금융·자본시장)
    "금융": ("금융", "finance"),
    "증권": ("증권", "finance"),
    "보험": ("보험", "finance"),
    "도산": ("도산", "finance"),
    # tax (조세·관세)
    "조세": ("조세법", "tax"),
    "조세법": ("조세법", "tax"),
    "관세": ("관세", "tax"),
    # public (공정·행정·공공)
    "공정거래": ("공정거래", "public"),
    "국가계약": ("국가계약", "public"),
    "행정법": ("행정법", "public"),
    # ip (지식재산)
    "특허": ("특허", "ip"),
    "저작권": ("저작권", "ip"),
    # it-media (IT·미디어·콘텐츠)
    "IT": ("IT", "it-media"),
    "언론": ("언론·방송통신", "it-media"),
    "방송": ("언론·방송통신", "it-media"),
    "방송통신": ("언론·방송통신", "it-media"),
    "엔터": ("엔터테인먼트", "it-media"),
    "엔터테인먼트": ("엔터테인먼트", "it-media"),
    "스포츠": ("스포츠", "it-media"),
    # medical (의료·바이오·식품)
    "의료": ("의료", "medical"),
    "식품": ("식품·의약", "medical"),
    "의약": ("식품·의약", "medical"),
    # international (국제·해외)
    "국제관계": ("국제관계법", "international"),
    "국제관계법": ("국제관계법", "international"),
    "국제중재": ("국제중재", "international"),
    "중재": ("중재", "international"),
    "해외투자": ("해외투자", "international"),
    "해상": ("해상", "international"),
    "이민": ("이주 및 비자", "international"),
    "비자": ("이주 및 비자", "international"),
}

# 카테고리 ID → 한글명 매핑
CATEGORY_NAMES: dict[str, str] = {
    "civil-family": "민사·가사",
    "criminal": "형사",
    "real-estate": "부동산·건설",
    "labor": "노동·산재",
    "corporate": "기업·상사",
    "finance": "금융·자본시장",
    "tax": "조세·관세",
    "public": "공정·행정·공공",
    "ip": "지식재산(IP)",
    "it-media": "IT·미디어·콘텐츠",
    "medical": "의료·바이오·식품",
    "international": "국제·해외",
}


def extract_district(message: str) -> tuple[str | None, dict[str, float] | None]:
    """
    Extract a district name and its coordinates from a user message.
    
    Matches known district keys against the message using longest-name-first priority. If a match is found, returns a normalized district name (appends "구" for Seoul districts when appropriate; preserves major city/region names) and the coordinate dict from DISTRICT_COORDS. If no district is found, returns (None, None).
    
    Returns:
        tuple[str | None, dict[str, float] | None]: (district_name, coords) where `district_name` is a normalized name or `None`, and `coords` is a dict with latitude/longitude from DISTRICT_COORDS or `None`.
    """
    # 긴 이름부터 매칭 (강남구 vs 강남)
    sorted_districts = sorted(DISTRICT_COORDS.keys(), key=len, reverse=True)
    for district in sorted_districts:
        if district in message:
            coords = DISTRICT_COORDS[district]
            # 구 이름으로 정규화 (강남 → 강남구)
            district_name = district if district.endswith("구") else f"{district}구"
            # 서울 외 지역은 그대로
            if district in ["부산", "대구", "인천", "광주", "대전", "울산",
                           "수원", "성남", "고양", "용인", "분당", "일산", "판교"]:
                district_name = district
            return district_name, coords
    return None, None


def extract_specialty(message: str) -> tuple[str | None, str | None, str | None]:
    """
    Extract a legal specialty and its category from a user's message.
    
    Matches specific specialty keywords first (longer keywords prioritized) and falls back to category-level keywords if no precise specialty is found.
    
    Returns:
        (specialty, category_id, category_name)
        - specialty (str | None): Exact specialty name that matches DB values (e.g., "이혼"), or None if only a category matched.
        - category_id (str | None): Category identifier (e.g., "civil-family") when a specialty or category keyword is found, otherwise None.
        - category_name (str | None): Human-readable Korean category name corresponding to category_id, or None if no match was found.
    """
    # 1. 정확한 전문분야 키워드 먼저 확인
    # 긴 키워드부터 매칭 (예: "국제중재" vs "중재")
    sorted_keywords = sorted(SPECIALTY_KEYWORDS.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if keyword in message:
            specialty, category_id = SPECIALTY_KEYWORDS[keyword]
            category_name = CATEGORY_NAMES.get(category_id)
            return specialty, category_id, category_name

    # 2. 정확한 전문분야를 못 찾으면 일반 카테고리 키워드로 시도
    for category_id, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in message:
                return None, category_id, CATEGORY_NAMES[category_id]

    return None, None, None


def extract_category(message: str) -> tuple[str | None, str | None]:
    """
    Extract a category identifier and its human-readable name from the given message.
    
    Parameters:
        message (str): User-provided text to analyze for category keywords.
    
    Returns:
        tuple[str | None, str | None]: (category_id, category_name) where each element is the matched category identifier and display name, or `None` if no category was found.
    """
    _, category_id, category_name = extract_specialty(message)
    return category_id, category_name


class LawyerFinderAgent(BaseAgent):
    """변호사 찾기 에이전트 - 변호사 찾기 페이지로 네비게이션"""

    @property
    def name(self) -> str:
        """
        Agent identifier for the lawyer finder agent.
        
        Returns:
            str: The fixed agent name "lawyer_finder".
        """
        return "lawyer_finder"

    @property
    def description(self) -> str:
        """
        Provide the agent's short descriptive label for UI display.
        
        Returns:
            description (str): The agent description string "위치 기반 변호사 추천 - 지도에서 변호사 찾기".
        """
        return "위치 기반 변호사 추천 - 지도에서 변호사 찾기"

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResponse:
        """
        Handle a lawyer search request and navigate the user to the map-based lawyer finder.
        
        The function extracts district and specialty information from the incoming message, restores any pending
        category/specialty from session data, and determines the search location from district coordinates or the
        provided user_location. If no location is available, it prompts the user to share their location and returns
        an AgentResponse that requests location while preserving pending search criteria in session_data. If a
        location is available, it performs a tiered search by category (3 km radius, then 10 km radius, then a global
        category-wide search), constructs navigation parameters for the map view, and returns an AgentResponse that
        navigates to "/lawyer-finder" with appropriate filters and session_data updated with the last search context.
        
        Parameters:
            message (str): The user's input message to parse for district and specialty keywords.
            history (list[dict[str, str]] | None): Optional conversation history; not required for search logic.
            session_data (dict[str, Any] | None): Session state used to restore pending_category or pending_specialty
                when awaiting location, and updated in the returned AgentResponse (keys like awaiting_location,
                pending_category, pending_specialty, last_search_location, last_category, search_mode, etc.).
            user_location (dict[str, float] | None): Optional user-provided location with keys
                "latitude" and "longitude" (float).
        
        Returns:
            AgentResponse: Response containing a message, zero or more ChatAction entries (either a REQUEST_LOCATION
            action when prompting for location, or a NAVIGATE action to "/lawyer-finder" plus a reset BUTTON), and
            updated session_data reflecting the agent's state and last search results.
        """
        session_data = session_data or {}

        # 1. 메시지에서 지역명 추출
        district_name, district_coords = extract_district(message)

        # 2. 메시지에서 전문분야 추출 (specialty: 정확한 전문분야, category: 대분류)
        specialty, category_id, category_name = extract_specialty(message)

        # 세션에서 pending 값 복원 (위치 요청 후 위치 공유 시)
        if not category_id and session_data.get("pending_category"):
            category_id = session_data.get("pending_category")
            category_name = CATEGORY_NAMES.get(category_id)
        if not specialty and session_data.get("pending_specialty"):
            specialty = session_data.get("pending_specialty")

        # 표시용 전문분야명 (specialty가 있으면 specialty, 없으면 category_name)
        display_specialty = specialty or category_name

        # 3. 위치 결정 (우선순위: 메시지 지역명 > 사용자 위치)
        latitude = None
        longitude = None
        location_source = None

        if district_coords:
            latitude = district_coords["latitude"]
            longitude = district_coords["longitude"]
            location_source = district_name
        elif user_location:
            latitude = user_location.get("latitude")
            longitude = user_location.get("longitude")
            location_source = "현재 위치"

        # 4. 위치도 없고 전문분야만 있는 경우 - 위치 요청
        if latitude is None and longitude is None:
            # 전문분야가 있으면 해당 정보 포함
            if display_specialty:
                msg = (
                    f"**{display_specialty}** 전문 변호사를 찾아드릴게요.\n\n"
                    "위치를 알려주시면 주변 변호사를 지도에서 보여드릴 수 있어요.\n"
                    "아래 버튼을 눌러 현재 위치를 공유하거나, "
                    "지역명을 알려주세요. (예: '강남구')"
                )
            else:
                msg = (
                    "변호사를 찾아드릴게요.\n\n"
                    "어떤 지역에서 찾으시나요?\n"
                    "현재 위치를 공유하거나 지역명을 알려주세요. (예: '서초구 이혼 변호사')"
                )

            return AgentResponse(
                message=msg,
                sources=[],
                actions=[
                    ChatAction(
                        type=ActionType.REQUEST_LOCATION,
                        label="현재 위치 공유",
                        action="request_location",
                    ),
                ],
                session_data={
                    "active_agent": self.name,
                    "awaiting_location": True,
                    "pending_category": category_id,
                    "pending_specialty": specialty,
                },
            )

        # 5. 3단계 반경 확장 검색 (항상 category_id로 검색)
        # 1단계: 3km
        search_results = find_nearby_lawyers(
            latitude=latitude,
            longitude=longitude,
            radius_m=DEFAULT_SEARCH_RADIUS,
            category=category_id,
        )
        actual_radius = DEFAULT_SEARCH_RADIUS
        search_mode = "nearby"  # nearby | expanded | all

        # 2단계: 10km (결과 없을 때)
        if not search_results:
            search_results = find_nearby_lawyers(
                latitude=latitude,
                longitude=longitude,
                radius_m=EXPANDED_SEARCH_RADIUS,
                category=category_id,
            )
            actual_radius = EXPANDED_SEARCH_RADIUS
            search_mode = "expanded"

        # 3단계: 전체 검색 (10km에도 없을 때)
        if not search_results:
            search_results = search_lawyers(category=category_id)
            search_mode = "all"

        # 6. 네비게이션 파라미터 구성 (검색 모드에 따라)
        nav_params: dict[str, Any] = {}

        if search_mode in ("nearby", "expanded"):
            nav_params["lat"] = latitude
            nav_params["lng"] = longitude
            nav_params["radius"] = actual_radius
        elif search_mode == "all":
            # 전체 검색 모드: 위치 없이 카테고리로만 검색
            nav_params["searchAll"] = "true"

        if category_id:
            nav_params["category"] = category_id

        if district_name and district_name.endswith("구"):
            nav_params["sigungu"] = district_name

        # 7. 응답 메시지 생성 (검색 모드에 따라)
        result_count = len(search_results)

        if search_mode == "nearby":
            if category_name:
                msg = (
                    f"**{location_source}** 주변 3km 내에 "
                    f"**{category_name}** 분야 변호사 **{result_count}명**을 찾았습니다!"
                )
            else:
                msg = (
                    f"**{location_source}** 주변 3km 내에 "
                    f"변호사 **{result_count}명**을 찾았습니다!"
                )
        elif search_mode == "expanded":
            if category_name:
                msg = (
                    f"**{location_source}** 주변 3km 내에는 "
                    f"**{category_name}** 분야 변호사가 없어서 "
                    f"10km까지 확장하여 검색했습니다.\n\n"
                    f"**{category_name}** 분야 변호사 **{result_count}명**을 찾았습니다!"
                )
            else:
                msg = (
                    f"**{location_source}** 주변 3km 내에는 변호사가 없어서 "
                    f"10km까지 확장하여 검색했습니다.\n\n"
                    f"변호사 **{result_count}명**을 찾았습니다!"
                )
        else:  # search_mode == "all"
            if result_count > 0:
                if category_name:
                    msg = (
                        f"**{location_source}** 주변 10km 내에도 "
                        f"**{category_name}** 분야 변호사가 없어서 "
                        f"전체 지역에서 검색했습니다.\n\n"
                        f"**{category_name}** 분야 변호사 **{result_count}명**을 찾았습니다!"
                    )
                else:
                    msg = (
                        f"**{location_source}** 주변 10km 내에도 변호사가 없어서 "
                        f"전체 지역에서 검색했습니다.\n\n"
                        f"변호사 **{result_count}명**을 찾았습니다!"
                    )
            else:
                if category_name:
                    msg = (
                        f"**{category_name}** 분야 변호사를 찾지 못했습니다.\n\n"
                        "다른 전문분야로 검색해보시겠어요?"
                    )
                else:
                    msg = "변호사를 찾지 못했습니다.\n\n다른 조건으로 검색해보시겠어요?"

        msg += "\n\n🗺️ 지도로 이동합니다..."

        # 8. 항상 지도로 이동 (NAVIGATE 액션)
        return AgentResponse(
            message=msg,
            sources=[],
            actions=[
                ChatAction(
                    type=ActionType.NAVIGATE,
                    label="지도에서 변호사 찾기",
                    url="/lawyer-finder",
                    params=nav_params,
                ),
                ChatAction(
                    type=ActionType.BUTTON,
                    label="다른 조건으로 검색",
                    action="reset_search",
                ),
            ],
            session_data={
                "active_agent": self.name,
                "last_search_location": location_source,
                "last_category": category_id,
                "last_latitude": latitude,
                "last_longitude": longitude,
                "search_result_count": result_count,
                "search_mode": search_mode,
            },
        )

    def can_handle(self, message: str) -> bool:
        """
        Determine whether the input message requests finding or recommending a lawyer.
        
        Returns:
            `true` if the message contains any lawyer-finding or recommendation keywords, `false` otherwise.
        """
        keywords = [
            "변호사 찾", "변호사를 찾", "변호사 추천", "변호사를 추천",
            "근처 변호사", "주변 변호사", "변호사 소개", "변호사를 소개",
            "변호사 알려", "변호사를 알려", "변호사 검색", "변호사를 검색"
        ]
        return any(kw in message for kw in keywords)