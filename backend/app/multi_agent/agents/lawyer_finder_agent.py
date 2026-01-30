"""
변호사 찾기 에이전트

위치 기반 변호사 추천 기능 - 변호사 찾기 페이지로 네비게이션
"""

from typing import Any

from app.multi_agent.agents.base_chat import ActionType, BaseChatAgent, ChatAction
from app.multi_agent.schemas.plan import AgentResult
from app.modules.lawyer_finder.service import find_nearby_lawyers, search_lawyers

# 검색 반경 (미터)
DEFAULT_SEARCH_RADIUS = 3000  # 3km
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

# 전문분야 키워드 → 카테고리 ID 매핑
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

# 정확한 전문분야 키워드 → (실제 전문분야명, 카테고리 ID) 매핑
SPECIALTY_KEYWORDS: dict[str, tuple[str, str]] = {
    # civil-family
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
    # criminal
    "형사": ("형사법", "criminal"),
    "형사법": ("형사법", "criminal"),
    "군형법": ("군형법", "criminal"),
    "군사": ("군형법", "criminal"),
    # real-estate
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
    # labor
    "노동": ("노동법", "labor"),
    "노동법": ("노동법", "labor"),
    "산재": ("산재", "labor"),
    # corporate
    "회사법": ("회사법", "corporate"),
    "상사법": ("상사법", "corporate"),
    "인수합병": ("인수합병", "corporate"),
    "영업비밀": ("영업비밀", "corporate"),
    "채권추심": ("채권추심", "corporate"),
    # finance
    "금융": ("금융", "finance"),
    "증권": ("증권", "finance"),
    "보험": ("보험", "finance"),
    "도산": ("도산", "finance"),
    # tax
    "조세": ("조세법", "tax"),
    "조세법": ("조세법", "tax"),
    "관세": ("관세", "tax"),
    # public
    "공정거래": ("공정거래", "public"),
    "국가계약": ("국가계약", "public"),
    "행정법": ("행정법", "public"),
    # ip
    "특허": ("특허", "ip"),
    "저작권": ("저작권", "ip"),
    # it-media
    "IT": ("IT", "it-media"),
    "언론": ("언론·방송통신", "it-media"),
    "방송": ("언론·방송통신", "it-media"),
    "방송통신": ("언론·방송통신", "it-media"),
    "엔터": ("엔터테인먼트", "it-media"),
    "엔터테인먼트": ("엔터테인먼트", "it-media"),
    "스포츠": ("스포츠", "it-media"),
    # medical
    "의료": ("의료", "medical"),
    "식품": ("식품·의약", "medical"),
    "의약": ("식품·의약", "medical"),
    # international
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
    """메시지에서 지역명과 좌표 추출"""
    sorted_districts = sorted(DISTRICT_COORDS.keys(), key=len, reverse=True)
    for district in sorted_districts:
        if district in message:
            coords = DISTRICT_COORDS[district]
            district_name = district if district.endswith("구") else f"{district}구"
            if district in [
                "부산", "대구", "인천", "광주", "대전", "울산",
                "수원", "성남", "고양", "용인", "분당", "일산", "판교"
            ]:
                district_name = district
            return district_name, coords
    return None, None


def extract_specialty(message: str) -> tuple[str | None, str | None, str | None]:
    """메시지에서 전문분야 추출"""
    sorted_keywords = sorted(SPECIALTY_KEYWORDS.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if keyword in message:
            specialty, category_id = SPECIALTY_KEYWORDS[keyword]
            category_name = CATEGORY_NAMES.get(category_id)
            return specialty, category_id, category_name

    for category_id, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in message:
                return None, category_id, CATEGORY_NAMES[category_id]

    return None, None, None


class LawyerFinderAgent(BaseChatAgent):
    """변호사 찾기 에이전트"""

    @property
    def name(self) -> str:
        return "lawyer_finder"

    @property
    def description(self) -> str:
        return "위치 기반 변호사 추천 - 지도에서 변호사 찾기"

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        """변호사 검색 처리"""
        session_data = session_data or {}

        # 1. 지역명 추출
        district_name, district_coords = extract_district(message)

        # 2. 전문분야 추출
        specialty, category_id, category_name = extract_specialty(message)

        # 세션에서 pending 값 복원
        if not category_id and session_data.get("pending_category"):
            category_id = str(session_data.get("pending_category"))
            category_name = CATEGORY_NAMES.get(category_id)
        if not specialty and session_data.get("pending_specialty"):
            specialty = session_data.get("pending_specialty")

        display_specialty = specialty or category_name

        # 3. 위치 결정
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

        # 4. 위치 없으면 요청
        if latitude is None and longitude is None:
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

            return AgentResult(
                message=msg,
                sources=[],
                actions=[
                    ChatAction(
                        type=ActionType.REQUEST_LOCATION,
                        label="현재 위치 공유",
                        action="request_location",
                    ).model_dump(),
                ],
                session_data={
                    "active_agent": self.name,
                    "awaiting_location": True,
                    "pending_category": category_id,
                    "pending_specialty": specialty,
                },
                agent_used=self.name,
            )

        # 5. 3단계 반경 확장 검색
        assert latitude is not None and longitude is not None

        search_results = find_nearby_lawyers(
            latitude=latitude,
            longitude=longitude,
            radius_m=DEFAULT_SEARCH_RADIUS,
            category=category_id,
        )
        actual_radius = DEFAULT_SEARCH_RADIUS
        search_mode = "nearby"

        if not search_results:
            search_results = find_nearby_lawyers(
                latitude=latitude,
                longitude=longitude,
                radius_m=EXPANDED_SEARCH_RADIUS,
                category=category_id,
            )
            actual_radius = EXPANDED_SEARCH_RADIUS
            search_mode = "expanded"

        if not search_results:
            search_results = search_lawyers(category=category_id)
            search_mode = "all"

        # 6. 네비게이션 파라미터 구성
        nav_params: dict[str, Any] = {}

        if search_mode in ("nearby", "expanded"):
            nav_params["lat"] = latitude
            nav_params["lng"] = longitude
            nav_params["radius"] = actual_radius
        elif search_mode == "all":
            nav_params["searchAll"] = "true"

        if category_id:
            nav_params["category"] = category_id

        if district_name and district_name.endswith("구"):
            nav_params["sigungu"] = district_name

        # 7. 응답 메시지 생성
        result_count = len(search_results)
        msg = self._build_response_message(
            search_mode=search_mode,
            location_source=location_source,
            category_name=category_name,
            result_count=result_count,
        )
        msg += "\n\n🗺️ 지도로 이동합니다..."

        # 8. 결과 반환
        return AgentResult(
            message=msg,
            sources=[],
            actions=[
                ChatAction(
                    type=ActionType.NAVIGATE,
                    label="지도에서 변호사 찾기",
                    url="/lawyer-finder",
                    params=nav_params,
                ).model_dump(),
                ChatAction(
                    type=ActionType.BUTTON,
                    label="다른 조건으로 검색",
                    action="reset_search",
                ).model_dump(),
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
            agent_used=self.name,
        )

    def _build_response_message(
        self,
        search_mode: str,
        location_source: str | None,
        category_name: str | None,
        result_count: int,
    ) -> str:
        """응답 메시지 생성"""
        if search_mode == "nearby":
            if category_name:
                return (
                    f"**{location_source}** 주변 3km 내에 "
                    f"**{category_name}** 분야 변호사 **{result_count}명**을 찾았습니다!"
                )
            return (
                f"**{location_source}** 주변 3km 내에 "
                f"변호사 **{result_count}명**을 찾았습니다!"
            )

        if search_mode == "expanded":
            if category_name:
                return (
                    f"**{location_source}** 주변 3km 내에는 "
                    f"**{category_name}** 분야 변호사가 없어서 "
                    f"10km까지 확장하여 검색했습니다.\n\n"
                    f"**{category_name}** 분야 변호사 **{result_count}명**을 찾았습니다!"
                )
            return (
                f"**{location_source}** 주변 3km 내에는 변호사가 없어서 "
                f"10km까지 확장하여 검색했습니다.\n\n"
                f"변호사 **{result_count}명**을 찾았습니다!"
            )

        # search_mode == "all"
        if result_count > 0:
            if category_name:
                return (
                    f"**{location_source}** 주변 10km 내에도 "
                    f"**{category_name}** 분야 변호사가 없어서 "
                    f"전체 지역에서 검색했습니다.\n\n"
                    f"**{category_name}** 분야 변호사 **{result_count}명**을 찾았습니다!"
                )
            return (
                f"**{location_source}** 주변 10km 내에도 변호사가 없어서 "
                f"전체 지역에서 검색했습니다.\n\n"
                f"변호사 **{result_count}명**을 찾았습니다!"
            )

        if category_name:
            return (
                f"**{category_name}** 분야 변호사를 찾지 못했습니다.\n\n"
                "다른 전문분야로 검색해보시겠어요?"
            )
        return "변호사를 찾지 못했습니다.\n\n다른 조건으로 검색해보시겠어요?"

    def can_handle(self, message: str) -> bool:
        """변호사 찾기 관련 키워드 확인"""
        keywords = [
            "변호사 찾", "변호사를 찾", "변호사 추천", "변호사를 추천",
            "근처 변호사", "주변 변호사", "변호사 소개", "변호사를 소개",
            "변호사 알려", "변호사를 알려", "변호사 검색", "변호사를 검색"
        ]
        return any(kw in message for kw in keywords)
