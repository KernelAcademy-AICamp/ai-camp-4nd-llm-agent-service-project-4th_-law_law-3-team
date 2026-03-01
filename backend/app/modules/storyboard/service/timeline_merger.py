"""타임라인 증분 병합 서비스 (Phase 3)

3단계 병합 알고리즘:
  Step 1: 날짜 범위 기반 후보 쌍 탐색
  Step 2: LLM 유사도 중복 감지 (최대 5쌍)
  Step 3: 병합 실행 (중복→evidence_ids 추가, 신규→추가, 충돌→MergeConflict 기록)
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings

from ..schema.models import EvidenceFile, TimelineItem
from ..schema.responses import MergeConflict, MergeReport, MergeTimelineResponse

logger = logging.getLogger(__name__)

# SEC-08: 병합 충돌 감사 로그 전용 logger
audit_logger = logging.getLogger("storyboard.merge_audit")

# LLM 중복 감지 최대 후보 쌍 수 (비용 절감)
MAX_DUPLICATE_CANDIDATES = 5

# 날짜 유사도: ±N일 이내를 "겹친다"로 간주
DATE_OVERLAP_DAYS = 30

# 중복 판단 LLM 임계 점수 (0~1, 이 이상이면 중복)
DUPLICATE_THRESHOLD = 0.7

_DUPLICATE_SYSTEM_PROMPT = """당신은 법률 사건 타임라인 분석 전문가입니다.
두 타임라인 이벤트 쌍이 동일한 사건을 설명하는지 판단합니다.

각 쌍에 대해 다음 기준으로 판단하세요:
1. 날짜/시간이 동일하거나 매우 근접한가?
2. 등장인물이 겹치는가?
3. 사건 내용이 실질적으로 동일한가? (표현 방식이 달라도 됨)

반드시 다음 JSON 형식으로만 응답하세요:
{
  "results": [
    {
      "pair_index": 0,
      "is_duplicate": true,
      "score": 0.95,
      "reason": "날짜와 등장인물, 사건 내용이 일치함"
    }
  ]
}

score: 0.0~1.0 (1.0 = 완전히 동일한 사건)
is_duplicate: score >= 0.7 이면 true"""


@dataclass
class _DuplicateResult:
    """LLM 중복 판단 결과"""
    pair_index: int
    is_duplicate: bool
    score: float
    existing_item: TimelineItem
    new_item: TimelineItem


def _parse_date(date_str: str) -> datetime | None:
    """날짜 문자열에서 datetime 추출. YYYY-MM-DD 또는 YYYY.MM.DD 등 지원."""
    cleaned = date_str.strip()
    if not cleaned or "미상" in cleaned:
        return None
    # YYYY-MM-DD, YYYY.MM.DD, YYYY/MM/DD 패턴 추출
    match = re.search(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", cleaned)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    # YYYY-MM 패턴 (일자 없음)
    match = re.search(r"(\d{4})[-./](\d{1,2})", cleaned)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), 15)
        except ValueError:
            return None
    return None


def _dates_overlap(date_a: str, date_b: str) -> bool:
    """두 날짜가 ±DATE_OVERLAP_DAYS일 이내인지 확인."""
    dt_a = _parse_date(date_a)
    dt_b = _parse_date(date_b)
    if dt_a is None or dt_b is None:
        return False
    return abs(dt_a - dt_b) <= timedelta(days=DATE_OVERLAP_DAYS)


class TimelineMerger:
    """
    기존 타임라인 + 신규 항목 증분 병합

    사용법:
        merger = TimelineMerger()
        response = await merger.merge(
            existing_items=[...],
            new_items=[...],
            existing_evidence=[...],
            new_evidence=[...],
        )
    """

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    @staticmethod
    def _parse_llm_json(raw: str) -> dict[str, Any]:
        """다단계 JSON 파싱 fallback: 코드블록 → 직접 파싱 → 정규식 추출."""
        candidates: list[str] = []
        # 1) ```json ... ``` 코드블록
        if "```json" in raw:
            parts = raw.split("```json")
            for part in parts[1:]:
                if "```" in part:
                    candidates.append(part.split("```")[0])
        elif "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 3:
                candidates.append(parts[1])
        # 2) 원본 그대로
        candidates.append(raw)
        # 3) { ... } 정규식 추출
        brace_match = re.search(r"\{[\s\S]*\}", raw)
        if brace_match:
            candidates.append(brace_match.group(0))

        for candidate in candidates:
            try:
                return json.loads(candidate.strip())  # type: ignore[no-any-return]
            except (json.JSONDecodeError, ValueError):
                continue
        raise ValueError(f"LLM JSON 파싱 실패: {raw[:200]}")

    async def merge(
        self,
        existing_items: list[TimelineItem],
        new_items: list[TimelineItem],
        existing_evidence: list[EvidenceFile],
        new_evidence: list[EvidenceFile],
    ) -> MergeTimelineResponse:
        """
        3단계 증분 병합 실행.

        Returns:
            MergeTimelineResponse (병합된 항목 + 증거 + 보고서)
        """
        if not new_items:
            return MergeTimelineResponse(
                success=True,
                merged_items=list(existing_items),
                merged_evidence=list(existing_evidence) + list(new_evidence),
                merge_report=MergeReport(
                    new_items_added=0,
                    duplicates_detected=0,
                    items_updated=0,
                ),
            )

        # Step 1: 날짜 범위 기반 후보 쌍 탐색
        candidates = self._find_date_overlap_candidates(existing_items, new_items)

        # Step 2: LLM 중복 감지 (후보가 있을 때만)
        duplicate_results: list[_DuplicateResult] = []
        if candidates:
            try:
                duplicate_results = await self._check_duplicates_with_llm(candidates)
            except Exception as exc:
                logger.warning("LLM 중복 감지 실패, 날짜 기반으로만 처리: %s", exc)

        # Step 3: 병합 실행
        return self._apply_merge(
            existing_items=existing_items,
            new_items=new_items,
            existing_evidence=existing_evidence,
            new_evidence=new_evidence,
            duplicate_results=duplicate_results,
        )

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------

    def _find_date_overlap_candidates(
        self,
        existing: list[TimelineItem],
        new_items: list[TimelineItem],
    ) -> list[tuple[TimelineItem, TimelineItem]]:
        """
        날짜가 겹치는 (기존, 신규) 쌍을 최대 MAX_DUPLICATE_CANDIDATES 개 반환.
        제목 앞 5글자도 비교하여 완전 무관한 쌍은 제외.
        """
        candidates: list[tuple[TimelineItem, TimelineItem]] = []
        for new_item in new_items:
            for existing_item in existing:
                if not _dates_overlap(existing_item.date, new_item.date):
                    continue
                # 제목 접두어 빠른 필터 (완전히 다른 사건 제거)
                title_a = existing_item.title[:5].strip()
                title_b = new_item.title[:5].strip()
                if title_a and title_b and title_a != title_b:
                    # 날짜 겹치지만 제목이 전혀 다른 경우 — 후보 포함 (LLM이 최종 판단)
                    pass
                candidates.append((existing_item, new_item))
                if len(candidates) >= MAX_DUPLICATE_CANDIDATES:
                    return candidates
        return candidates

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------

    async def _check_duplicates_with_llm(
        self,
        candidates: list[tuple[TimelineItem, TimelineItem]],
    ) -> list[_DuplicateResult]:
        """LLM에게 후보 쌍의 중복 여부를 일괄 판단 요청."""
        pairs_payload: list[dict[str, Any]] = []
        for idx, (existing_item, new_item) in enumerate(candidates):
            pairs_payload.append({
                "pair_index": idx,
                "existing": {
                    "date": existing_item.date,
                    "title": existing_item.title,
                    "description": existing_item.description,
                    "participants": existing_item.participants[:3],
                },
                "new": {
                    "date": new_item.date,
                    "title": new_item.title,
                    "description": new_item.description,
                    "participants": new_item.participants[:3],
                },
            })

        user_message = (
            f"다음 {len(pairs_payload)}개 쌍의 중복 여부를 판단해주세요:\n\n"
            + json.dumps(pairs_payload, ensure_ascii=False, indent=2)
        )

        client = self._get_client()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _DUPLICATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        raw = response.choices[0].message.content or ""
        # 다단계 JSON 파싱 fallback
        data = self._parse_llm_json(raw)
        llm_results: list[dict[str, Any]] = data.get("results", [])

        results: list[_DuplicateResult] = []
        for llm_result in llm_results:
            pair_idx = int(llm_result.get("pair_index", -1))
            if pair_idx < 0 or pair_idx >= len(candidates):
                continue
            existing_item, new_item = candidates[pair_idx]
            results.append(
                _DuplicateResult(
                    pair_index=pair_idx,
                    is_duplicate=bool(llm_result.get("is_duplicate", False)),
                    score=float(llm_result.get("score", 0.0)),
                    existing_item=existing_item,
                    new_item=new_item,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Step 3
    # ------------------------------------------------------------------

    def _apply_merge(
        self,
        existing_items: list[TimelineItem],
        new_items: list[TimelineItem],
        existing_evidence: list[EvidenceFile],
        new_evidence: list[EvidenceFile],
        duplicate_results: list[_DuplicateResult],
    ) -> MergeTimelineResponse:
        """
        중복 결과를 바탕으로 병합 수행.

        - 중복 확인된 신규 항목: 기존 항목의 evidence_ids에 추가 (흡수)
        - 충돌(날짜 겹침 but 내용 다름): MergeConflict 기록 후 양쪽 유지
        - 나머지 신규 항목: 그대로 추가
        """
        from ..service import _date_sort_key

        # 중복으로 판단된 신규 항목 ID 집합
        absorbed_new_ids: set[str] = set()
        updated_count = 0
        conflicts: list[MergeConflict] = []

        # 기존 항목 복사본 (변경 가능)
        existing_map: dict[str, TimelineItem] = {item.id: item for item in existing_items}

        for dup in duplicate_results:
            if not dup.is_duplicate:
                # 충돌: 날짜는 겹치지만 내용이 다름 → MergeConflict 기록
                if dup.score >= 0.3:
                    conflicts.append(
                        MergeConflict(
                            existing_item_id=dup.existing_item.id,
                            new_item_id=dup.new_item.id,
                            conflict_type="content_contradiction",
                            description=(
                                f"날짜 겹침({dup.existing_item.date}) but 내용 불일치 "
                                f"(유사도 {dup.score:.2f})"
                            ),
                        )
                    )
                continue

            # 중복 처리: 신규 항목의 evidence_ids를 기존 항목에 흡수
            absorbed_new_ids.add(dup.new_item.id)
            target = existing_map.get(dup.existing_item.id)
            if target is None:
                continue

            # evidence_ids 병합 (중복 없이)
            existing_ev_set = set(target.evidence_ids)
            for ev_id in dup.new_item.evidence_ids:
                if ev_id not in existing_ev_set:
                    target.evidence_ids.append(ev_id)
                    existing_ev_set.add(ev_id)
            updated_count += 1

        # 신규 항목 중 흡수되지 않은 것만 추가
        added_items: list[TimelineItem] = [
            item for item in new_items if item.id not in absorbed_new_ids
        ]

        # 최종 병합 목록 구성
        all_items: list[TimelineItem] = list(existing_map.values()) + added_items
        all_items.sort(key=lambda item: _date_sort_key(item.date))

        # order / scene_number 재할당
        for idx, item in enumerate(all_items):
            item.order = idx
            item.scene_number = idx + 1

        # 증거 파일 병합 (기존 + 신규 모두 유지)
        merged_evidence = list(existing_evidence) + list(new_evidence)

        report = MergeReport(
            new_items_added=len(added_items),
            duplicates_detected=len(absorbed_new_ids),
            items_updated=updated_count,
            conflicts=conflicts,
        )

        # SEC-08: 병합 충돌 감사 로그
        if conflicts:
            audit_logger.info(
                "병합 충돌 감지: %d건 | 중복흡수: %d건 | 신규추가: %d건",
                len(conflicts),
                len(absorbed_new_ids),
                len(added_items),
            )
            for conflict in conflicts:
                audit_logger.info(
                    "  충돌 상세: type=%s, existing=%s, new=%s, desc=%s",
                    conflict.conflict_type,
                    conflict.existing_item_id,
                    conflict.new_item_id,
                    conflict.description,
                )

        return MergeTimelineResponse(
            success=True,
            merged_items=all_items,
            merged_evidence=merged_evidence,
            merge_report=report,
        )
