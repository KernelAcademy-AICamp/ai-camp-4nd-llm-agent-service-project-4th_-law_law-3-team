"""
변호사 시장 분석 어드바이저 에이전트

신입 변호사 대상 시장 진입 전략 추천 + 통계 조회.
2계층 의도분석 (추천형 / 조회형) + 다중 데이터 조합 + LLM 인사이트 생성.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from app.modules.lawyer_stats.schema import StatsIntent
from app.multi_agent.agents.base_chat import (
    ActionType,
    BaseChatAgent,
    ChatAction,
    normalize_chunk_content,
)
from app.multi_agent.schemas.plan import AgentResult
from app.tools.llm import get_chat_model

logger = logging.getLogger(__name__)

# =============================================================================
# 프롬프트
# =============================================================================
_INTENT_SYSTEM_PROMPT = """\
당신은 변호사 통계 질문 분류기입니다.
사용자 메시지를 분석하여 아래 JSON 형식으로만 응답하세요. 설명 없이 JSON만 출력합니다.

## 질문 유형 (query_type)
추천형:
- recommend_market: 어디에 개업할지 종합 추천 (지역+분야)
- recommend_specialty: 특정 지역에서 어떤 분야가 좋을지
- recommend_region: 특정 분야로 어디 지역이 좋을지

조회형:
- overview: 전체 현황
- region: 지역별 변호사 수
- density: 인구 대비 밀도
- specialty: 전문분야별 분포
- cross: 지역×전문분야 교차분석
- demand: 사건 수요/부담지수
- prediction: 향후 예측

## 지역명 약칭 규칙
서울, 부산, 대구, 인천, 광주, 대전, 울산, 세종, 경기, 강원, 충북, 충남, 전북, 전남, 경북, 경남, 제주
(구/군 단위가 아닌 시/도 단위만 가능)

## 수요 카테고리
민사, 형사, 가사, 행정, 소년보호, 가정보호

## 중요: 지역 추출 규칙
- 사용자가 특정 지역을 언급하면 **반드시** regions 배열에 시/도 단위로 포함
- "수원", "성남" → regions: ["경기"]
- "부산 해운대" → regions: ["부산"]
- "서울 강남" → regions: ["서울"]
- "경기도 어디가 좋을까" → regions: ["경기"]
- 지역 언급 없으면 regions: []

## JSON 형식
{
  "query_type": "recommend_market",
  "regions": [],
  "province": null,
  "specialty_interest": null,
  "prediction_year": null,
  "demand_category": null,
  "demand_year": null
}

<example>
사용자: 서울에서 개업하려면 어떤 분야가 좋을까요?
응답: {"query_type": "recommend_specialty", "regions": ["서울"], "province": "서울", "specialty_interest": null, "prediction_year": null, "demand_category": null, "demand_year": null}
</example>

<example>
사용자: 전국 변호사 현황 알려줘
응답: {"query_type": "overview", "regions": [], "province": null, "specialty_interest": null, "prediction_year": null, "demand_category": null, "demand_year": null}
</example>

<example>
사용자: 형사 사건 수요가 많은 지역은?
응답: {"query_type": "demand", "regions": [], "province": null, "specialty_interest": null, "prediction_year": null, "demand_category": "형사", "demand_year": null}
</example>"""

_RECOMMENDATION_SYSTEM_PROMPT = """\
당신은 신입 변호사를 위한 시장 분석 어드바이저입니다.
통계 데이터를 바탕으로 개업 전략을 추천하세요.

## 분석 관점
- **경쟁 강도**: 인구 대비 변호사 밀도가 낮은 = 경쟁 적음
- **수요 기회**: 사건 부담지수(사건수/변호사수)가 높은 = 수요 많음
- **분야 블루오션**: 전문 변호사 비율이 낮은데 수요는 있는 분야

## 응답 형식
1. 핵심 추천 (지역 or 분야) — 수치 근거와 함께
2. 경쟁/수요 비교 분석
3. 주의사항

## 주의
- 시/도 단위 데이터입니다 (구 단위 분석 불가)
- 추세/예측은 참고용이며 확정적 조언이 아님을 명시
- 답변은 한국어로 작성"""

_QUERY_SYSTEM_PROMPT = """\
변호사 통계 데이터를 바탕으로 사용자 질문에 간결하게 답변하세요.
핵심 수치를 먼저 제시하고, 상위/하위 비교를 포함하세요.
답변은 한국어로 작성합니다."""


# =============================================================================
# 에이전트
# =============================================================================
class LawyerStatsAgent(BaseChatAgent):
    """변호사 시장 분석 어드바이저"""

    @property
    def name(self) -> str:
        return "lawyer_stats"

    @property
    def description(self) -> str:
        return "변호사 시장 분석 및 개업 전략 추천"

    @property
    def supports_streaming(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # process (비스트리밍 - 기본 구현)
    # ------------------------------------------------------------------
    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        """메시지 처리 (비스트리밍)"""
        intent = await self._parse_intent(message, history)
        stats_data = await self._fetch_stats(intent)
        response = await self._generate_response(message, intent, stats_data, history)
        stats_filter = self._build_filter(intent)

        return AgentResult(
            message=response,
            sources=[],
            actions=[
                ChatAction(
                    type=ActionType.NAVIGATE,
                    label="통계 대시보드 보기",
                    url="/lawyer-stats",
                ).model_dump(),
            ],
            session_data={
                "active_agent": self.name,
                "stats_filter": stats_filter,
            },
            agent_used=self.name,
        )

    # ------------------------------------------------------------------
    # process_stream (스트리밍)
    # ------------------------------------------------------------------
    async def process_stream(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """스트리밍 메시지 처리"""
        intent = await self._parse_intent(message, history)
        stats_data = await self._fetch_stats(intent)
        stats_filter = self._build_filter(intent)

        model = get_chat_model()
        messages = self._build_llm_messages(message, intent, stats_data, history)

        async for chunk in model.astream(messages):
            if chunk.content:
                text = normalize_chunk_content(chunk.content)
                if text:
                    yield ("token", {"content": text})

        yield ("sources", {"sources": []})
        yield ("metadata", {
            "agent_used": self.name,
            "actions": [
                ChatAction(
                    type=ActionType.NAVIGATE,
                    label="통계 대시보드 보기",
                    url="/lawyer-stats",
                ).model_dump(),
            ],
            "session_data": {
                "active_agent": self.name,
                "stats_filter": stats_filter,
            },
        })
        yield ("done", {})

    # ------------------------------------------------------------------
    # 의도 분석
    # ------------------------------------------------------------------
    async def _parse_intent(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> StatsIntent:
        """LLM으로 사용자 메시지 의도 분석"""
        model = get_chat_model()

        messages: list[tuple[str, str]] = [("system", _INTENT_SYSTEM_PROMPT)]
        if history:
            for h in history[-4:]:
                messages.append((h.get("role", "user"), h.get("content", "")))
        messages.append(("user", message))

        try:
            response = await model.ainvoke(messages)
            content = response.content if isinstance(response.content, str) else str(response.content)

            # JSON 파싱 (```json ... ``` 블록 처리)
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                json_lines = [
                    line for line in lines
                    if not line.strip().startswith("```")
                ]
                cleaned = "\n".join(json_lines)

            parsed = json.loads(cleaned)
            return StatsIntent(**parsed)
        except (json.JSONDecodeError, ValueError, KeyError):
            logger.warning("의도 분석 실패, 기본값(overview) 사용: %s", message[:100])
            return StatsIntent(query_type="overview")

    # ------------------------------------------------------------------
    # 데이터 조회
    # ------------------------------------------------------------------
    async def _fetch_stats(self, intent: StatsIntent) -> dict[str, Any]:
        """의도에 따라 단일/다중 데이터 조회"""
        if intent.query_type.startswith("recommend"):
            return await self._fetch_recommendation_data(intent)
        return await self._fetch_query_data(intent)

    async def _fetch_recommendation_data(
        self, intent: StatsIntent
    ) -> dict[str, Any]:
        """추천형: 다중 서비스 함수 동시 호출"""
        from app.core.config import settings
        from app.services.service_function.lawyer_stats_service import (
            calculate_by_specialty,
            calculate_cross_analysis,
            calculate_cross_analysis_by_province,
            calculate_density_by_region,
            calculate_overview,
        )

        result: dict[str, Any] = {}

        # 동기 서비스 함수 (JSON 모드) 를 asyncio.to_thread로 병렬 실행
        if not settings.USE_DB_LAWYERS:
            density_task = asyncio.to_thread(calculate_density_by_region)
            specialty_task = asyncio.to_thread(calculate_by_specialty)
            overview_task = asyncio.to_thread(calculate_overview)

            if intent.province:
                cross_task = asyncio.to_thread(
                    calculate_cross_analysis_by_province, intent.province
                )
            else:
                cross_task = asyncio.to_thread(calculate_cross_analysis)

            density, specialty, overview, cross = await asyncio.gather(
                density_task, specialty_task, overview_task, cross_task
            )
            result["density"] = density
            result["specialty"] = specialty
            result["overview"] = overview
            result["cross"] = cross
        else:
            # DB 모드
            from app.core.database import async_session_factory
            from app.services.service_function.lawyer_stats_db_service import (
                calculate_by_specialty_db,
                calculate_cross_analysis_by_province_db,
                calculate_cross_analysis_db,
                calculate_density_by_region_db,
                calculate_overview_db,
            )

            async with async_session_factory() as db:
                density_task = calculate_density_by_region_db(db)
                specialty_task = calculate_by_specialty_db(db)
                overview_task = calculate_overview_db(db)

                if intent.province:
                    cross_task = calculate_cross_analysis_by_province_db(
                        db, intent.province
                    )
                else:
                    cross_task = calculate_cross_analysis_db(db)

                density, specialty, overview, cross = await asyncio.gather(
                    density_task, specialty_task, overview_task, cross_task
                )
                result["density"] = density
                result["specialty"] = specialty
                result["overview"] = overview
                result["cross"] = cross

        # 수요 데이터 (항상 DB 필요)
        try:
            from app.core.database import async_session_factory
            from app.services.service_function.case_demand_service import (
                calculate_demand_by_region,
            )

            category = intent.demand_category or "민사"
            year = intent.demand_year or 2024

            async with async_session_factory() as db:
                demand = await calculate_demand_by_region(db, category, year)
                result["demand"] = demand
        except (ValueError, RuntimeError):
            logger.warning("수요 데이터 조회 실패, 수요 데이터 없이 진행")
            result["demand"] = None

        return result

    async def _fetch_query_data(self, intent: StatsIntent) -> dict[str, Any]:
        """조회형: 단일 서비스 함수 호출"""
        from app.core.config import settings
        from app.services.service_function.lawyer_stats_service import (
            calculate_by_region,
            calculate_by_specialty,
            calculate_cross_analysis,
            calculate_cross_analysis_by_province,
            calculate_density_by_region,
            calculate_overview,
        )

        query_type = intent.query_type

        if not settings.USE_DB_LAWYERS:
            if query_type == "overview":
                return {"overview": calculate_overview()}
            elif query_type == "region":
                return {"region": calculate_by_region()}
            elif query_type in ("density", "prediction"):
                year = intent.prediction_year or "current"
                include_change = query_type == "prediction"
                return {"density": calculate_density_by_region(year, include_change)}
            elif query_type == "specialty":
                return {"specialty": calculate_by_specialty()}
            elif query_type == "cross":
                if intent.province:
                    return {"cross": calculate_cross_analysis_by_province(intent.province)}
                return {"cross": calculate_cross_analysis()}
            elif query_type == "demand":
                try:
                    from app.core.database import async_session_factory
                    from app.services.service_function.case_demand_service import (
                        calculate_demand_by_region,
                    )

                    category = intent.demand_category or "민사"
                    year = intent.demand_year or 2024
                    async with async_session_factory() as db:
                        return {"demand": await calculate_demand_by_region(db, category, year)}
                except (ValueError, RuntimeError):
                    logger.warning("수요 데이터 조회 실패")
                    return {"demand": None}
        else:
            # DB 모드
            from app.core.database import async_session_factory
            from app.services.service_function.lawyer_stats_db_service import (
                calculate_by_region_db,
                calculate_by_specialty_db,
                calculate_cross_analysis_by_province_db,
                calculate_cross_analysis_db,
                calculate_density_by_region_db,
                calculate_overview_db,
            )

            async with async_session_factory() as db:
                if query_type == "overview":
                    return {"overview": await calculate_overview_db(db)}
                elif query_type == "region":
                    return {"region": await calculate_by_region_db(db)}
                elif query_type in ("density", "prediction"):
                    year = intent.prediction_year or "current"
                    include_change = query_type == "prediction"
                    return {"density": await calculate_density_by_region_db(db, year, include_change)}
                elif query_type == "specialty":
                    return {"specialty": await calculate_by_specialty_db(db)}
                elif query_type == "cross":
                    if intent.province:
                        return {"cross": await calculate_cross_analysis_by_province_db(db, intent.province)}
                    return {"cross": await calculate_cross_analysis_db(db)}
                elif query_type == "demand":
                    from app.services.service_function.case_demand_service import (
                        calculate_demand_by_region,
                    )

                    category = intent.demand_category or "민사"
                    year = intent.demand_year or 2024
                    return {"demand": await calculate_demand_by_region(db, category, year)}

        return {"overview": calculate_overview()}

    # ------------------------------------------------------------------
    # LLM 응답 생성
    # ------------------------------------------------------------------
    def _build_llm_messages(
        self,
        message: str,
        intent: StatsIntent,
        stats_data: dict[str, Any],
        history: list[dict[str, str]] | None = None,
    ) -> list[tuple[str, str]]:
        """LLM 메시지 구성"""
        is_recommendation = intent.query_type.startswith("recommend")
        system_prompt = (
            _RECOMMENDATION_SYSTEM_PROMPT if is_recommendation
            else _QUERY_SYSTEM_PROMPT
        )

        messages: list[tuple[str, str]] = [("system", system_prompt)]

        if history:
            for h in history[-4:]:
                messages.append((h.get("role", "user"), h.get("content", "")))

        # 데이터를 컨텍스트로 직렬화 (크기 제한)
        data_context = self._serialize_stats_data(stats_data)

        user_message = f"## 데이터\n{data_context}\n\n## 사용자 질문\n{message}"
        messages.append(("user", user_message))

        return messages

    async def _generate_response(
        self,
        message: str,
        intent: StatsIntent,
        stats_data: dict[str, Any],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """LLM 응답 생성 (비스트리밍)"""
        model = get_chat_model()
        messages = self._build_llm_messages(message, intent, stats_data, history)

        response = await model.ainvoke(messages)
        content = response.content
        return content if isinstance(content, str) else str(content)

    def _serialize_stats_data(self, stats_data: dict[str, Any]) -> str:
        """통계 데이터를 LLM 컨텍스트용 문자열로 직렬화 (크기 제한)"""
        parts: list[str] = []

        if "overview" in stats_data and stats_data["overview"]:
            parts.append(f"### 전체 현황\n{json.dumps(stats_data['overview'], ensure_ascii=False, indent=2)}")

        if "density" in stats_data and stats_data["density"]:
            density = stats_data["density"]
            # 리스트인 경우 상위 10개만
            if isinstance(density, list):
                density = density[:10]
            parts.append(f"### 밀도 (상위 10)\n{json.dumps(density, ensure_ascii=False, indent=2)}")

        if "specialty" in stats_data and stats_data["specialty"]:
            specialty = stats_data["specialty"]
            if isinstance(specialty, list):
                # 세부 전문분야는 상위 3개만
                for item in specialty:
                    if isinstance(item, dict) and "specialties" in item:
                        item["specialties"] = item["specialties"][:3]
            parts.append(f"### 전문분야\n{json.dumps(specialty, ensure_ascii=False, indent=2)}")

        if "cross" in stats_data and stats_data["cross"]:
            cross = stats_data["cross"]
            if isinstance(cross, dict):
                # 교차분석 데이터: 상위 지역만
                cross_data = cross.get("data", [])
                regions = cross.get("regions", [])[:5]
                filtered_data = [
                    c for c in cross_data
                    if c.get("region") in regions
                ]
                cross = {"data": filtered_data, "regions": regions, "categories": cross.get("categories", [])}
            parts.append(f"### 교차분석\n{json.dumps(cross, ensure_ascii=False, indent=2)}")

        if "demand" in stats_data and stats_data["demand"]:
            demand = stats_data["demand"]
            if isinstance(demand, dict):
                demand_data = demand.get("data", [])[:10]
                demand = {
                    "data": demand_data,
                    "category": demand.get("category", ""),
                    "year": demand.get("year", 2024),
                }
            parts.append(f"### 수요\n{json.dumps(demand, ensure_ascii=False, indent=2)}")

        if "region" in stats_data and stats_data["region"]:
            region = stats_data["region"]
            if isinstance(region, list):
                region = region[:10]
            parts.append(f"### 지역별\n{json.dumps(region, ensure_ascii=False, indent=2)}")

        return "\n\n".join(parts) if parts else "데이터 없음"

    # ------------------------------------------------------------------
    # 대시보드 필터 구성
    # ------------------------------------------------------------------
    def _build_filter(self, intent: StatsIntent) -> dict[str, Any]:
        """질문 의도에 맞는 대시보드 필터 구성

        항상 완전한 필터를 반환하여, 이전 대시보드 상태가 잔류하지 않도록 한다.
        """
        query_type = intent.query_type

        # 지역: province > regions[0] > null(전체)
        province = intent.province
        if not province and intent.regions:
            province = intent.regions[0]

        stats_filter: dict[str, Any] = {
            "selectedProvince": province,
        }

        if query_type in ("recommend_market", "recommend_region"):
            stats_filter["indicatorGroup"] = "supply"
            stats_filter["viewMode"] = "density"
            stats_filter["activeTab"] = "region"
        elif query_type == "recommend_specialty":
            stats_filter["indicatorGroup"] = "supply"
            stats_filter["activeTab"] = "cross"
        elif query_type == "overview":
            stats_filter["indicatorGroup"] = "supply"
            stats_filter["viewMode"] = "count"
            stats_filter["activeTab"] = "region"
        elif query_type == "region":
            stats_filter["indicatorGroup"] = "supply"
            stats_filter["viewMode"] = "count"
            stats_filter["activeTab"] = "region"
        elif query_type == "density":
            stats_filter["indicatorGroup"] = "supply"
            stats_filter["viewMode"] = "density"
            stats_filter["activeTab"] = "region"
        elif query_type == "specialty":
            stats_filter["indicatorGroup"] = "supply"
            stats_filter["activeTab"] = "region"
        elif query_type == "cross":
            stats_filter["indicatorGroup"] = "supply"
            stats_filter["activeTab"] = "cross"
        elif query_type == "demand":
            stats_filter["indicatorGroup"] = "demand"
            stats_filter["viewMode"] = "case_count"
            stats_filter["activeTab"] = "region"
            if intent.demand_category:
                stats_filter["demandCategory"] = intent.demand_category
            if intent.demand_year:
                stats_filter["demandYear"] = intent.demand_year
        elif query_type == "prediction":
            stats_filter["indicatorGroup"] = "supply"
            stats_filter["viewMode"] = "prediction"
            stats_filter["activeTab"] = "region"
            if intent.prediction_year:
                stats_filter["predictionYear"] = intent.prediction_year

        return stats_filter

    def can_handle(self, message: str) -> bool:
        """변호사 통계/시장 관련 키워드 확인"""
        keywords = [
            "변호사 통계", "변호사 현황", "변호사 분포", "지역별 변호사",
            "개업", "시장 분석", "밀도", "전문분야", "수요",
        ]
        return any(kw in message for kw in keywords)
