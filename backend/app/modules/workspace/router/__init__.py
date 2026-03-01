"""
워크스페이스 라우터 — 사건 CRUD + 타임라인 API
"""

import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.core.database import async_session_factory
from app.modules.workspace.schema import (
    CaseCreateRequest,
    CaseUpdateRequest,
    TimelineItemUpdateRequest,
    TimelineRebuildRequest,
)
from app.services.workspace.activity_logger import ActivityLogger
from app.services.workspace.timeline_engine import TimelineEngine
from app.services.workspace.workspace_case_service import WorkspaceCaseService

router = APIRouter()


# ── 사건 CRUD ──


@router.post("/cases", status_code=201)
async def create_case(
    request: Request,
    body: CaseCreateRequest,
) -> dict[str, Any]:
    """사건 생성"""
    session_token: str = request.state.session_token

    async with async_session_factory() as db:
        case = await WorkspaceCaseService.create_case(
            db,
            session_token,
            body.case_name,
            body.case_type,
            body.conversation_ids,
        )
        await db.commit()

        return {
            "id": str(case.id),
            "case_name": case.case_name,
            "case_type": case.case_type,
            "status": case.status,
            "tagged_items": case.tagged_items,
            "created_at": case.created_at.isoformat() if case.created_at else None,
        }


@router.get("/cases")
async def list_cases(
    request: Request,
    status: str = Query("active"),
    search: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
) -> dict[str, Any]:
    """사건 목록 조회"""
    session_token: str = request.state.session_token

    async with async_session_factory() as db:
        items, total = await WorkspaceCaseService.list_cases(
            db, session_token, status, search, page, page_size
        )

    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/cases/{case_id}")
async def get_case(
    request: Request,
    case_id: str,
) -> dict[str, Any]:
    """사건 상세 조회"""
    session_token: str = request.state.session_token

    async with async_session_factory() as db:
        result = await WorkspaceCaseService.get_case_detail(
            db, uuid.UUID(case_id), session_token
        )
        if not result:
            raise HTTPException(status_code=404, detail="사건을 찾을 수 없습니다.")
        return result


@router.patch("/cases/{case_id}")
async def update_case(
    request: Request,
    case_id: str,
    body: CaseUpdateRequest,
) -> dict[str, Any]:
    """사건 수정"""
    session_token: str = request.state.session_token

    async with async_session_factory() as db:
        result = await WorkspaceCaseService.update_case(
            db, uuid.UUID(case_id), session_token, body.model_dump(exclude_none=True)
        )
        if not result:
            raise HTTPException(status_code=404, detail="사건을 찾을 수 없습니다.")
        await db.commit()
        return result


@router.delete("/cases/{case_id}", status_code=204)
async def delete_case(
    request: Request,
    case_id: str,
) -> None:
    """사건 삭제"""
    session_token: str = request.state.session_token

    async with async_session_factory() as db:
        deleted = await WorkspaceCaseService.delete_case(
            db, uuid.UUID(case_id), session_token
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="사건을 찾을 수 없습니다.")
        await db.commit()


# ── 타임라인 API ──


@router.get("/cases/{case_id}/timeline")
async def get_timeline(
    request: Request,
    case_id: str,
) -> dict[str, Any]:
    """타임라인 조회"""
    session_token: str = request.state.session_token

    async with async_session_factory() as db:
        detail = await WorkspaceCaseService.get_case_detail(
            db, uuid.UUID(case_id), session_token
        )
        if not detail:
            raise HTTPException(status_code=404, detail="사건을 찾을 수 없습니다.")

        return {
            "items": detail.get("timeline", []),
            "total": len(detail.get("timeline", [])),
        }


@router.post("/cases/{case_id}/timeline/rebuild")
async def rebuild_timeline(
    request: Request,
    case_id: str,
    body: TimelineRebuildRequest,
) -> dict[str, Any]:
    """타임라인 재생성"""
    session_token: str = request.state.session_token

    async with async_session_factory() as db:
        detail = await WorkspaceCaseService.get_case_detail(
            db, uuid.UUID(case_id), session_token
        )
        if not detail:
            raise HTTPException(status_code=404, detail="사건을 찾을 수 없습니다.")

        # 태그 수집: 사건 태그 + 연결된 대화 태그
        all_tags: list[dict[str, Any]] = detail.get("tagged_items", []) or []

        if body.include_conversations:
            from sqlalchemy import select

            from app.models.chat_conversation import ChatConversation

            conv_result = await db.execute(
                select(ChatConversation.tagged_items).where(
                    ChatConversation.case_id == uuid.UUID(case_id),
                )
            )
            for row in conv_result.scalars().all():
                if row:
                    all_tags.extend(row)

        items = await TimelineEngine.rebuild(
            db, uuid.UUID(case_id), all_tags, body.include_manual
        )

        await ActivityLogger.log(
            db, session_token, "rebuild_timeline",
            case_id=uuid.UUID(case_id),
            detail={"tag_count": len(all_tags), "item_count": len(items)},
        )
        await db.commit()

        return {"items": items, "total": len(items)}


@router.patch("/cases/{case_id}/timeline/items/{item_id}")
async def update_timeline_item(
    request: Request,
    case_id: str,
    item_id: str,
    body: TimelineItemUpdateRequest,
) -> dict[str, Any]:
    """타임라인 항목 수정 (source_type → manual)"""
    session_token: str = request.state.session_token

    async with async_session_factory() as db:
        # 소유권 확인
        detail = await WorkspaceCaseService.get_case_detail(
            db, uuid.UUID(case_id), session_token
        )
        if not detail:
            raise HTTPException(status_code=404, detail="사건을 찾을 수 없습니다.")

        from sqlalchemy import select, update

        from app.models.workspace_case import WorkspaceCaseTimelineItem

        result = await db.execute(
            select(WorkspaceCaseTimelineItem).where(
                WorkspaceCaseTimelineItem.id == uuid.UUID(item_id),
                WorkspaceCaseTimelineItem.case_id == uuid.UUID(case_id),
            )
        )
        item = result.scalar_one_or_none()
        if not item:
            raise HTTPException(status_code=404, detail="항목을 찾을 수 없습니다.")

        update_values: dict[str, Any] = {"source_type": "manual"}
        if body.title is not None:
            update_values["title"] = body.title
        if body.description is not None:
            update_values["description"] = body.description
        if body.date_text is not None:
            update_values["date_text"] = body.date_text
        if body.category is not None:
            update_values["category"] = body.category

        from sqlalchemy import func

        update_values["updated_at"] = func.now()
        await db.execute(
            update(WorkspaceCaseTimelineItem)
            .where(WorkspaceCaseTimelineItem.id == uuid.UUID(item_id))
            .values(**update_values)
        )
        await db.commit()

        return {
            "id": item_id,
            "title": body.title or item.title,
            "description": body.description or item.description,
            "date_text": body.date_text or item.date_text,
            "category": body.category or item.category,
            "source_type": "manual",
        }


# ── 사건 내보내기 ──


@router.get("/cases/{case_id}/export")
async def export_case(
    request: Request,
    case_id: str,
    format: str = Query("json", pattern="^(json|txt)$"),
    include: str = Query("all", pattern="^(all|timeline|conversations)$"),
) -> StreamingResponse:
    """사건 내보내기"""
    session_token: str = request.state.session_token

    async with async_session_factory() as db:
        detail = await WorkspaceCaseService.get_case_detail(
            db, uuid.UUID(case_id), session_token
        )
        if not detail:
            raise HTTPException(status_code=404, detail="사건을 찾을 수 없습니다.")

        case_name_safe = (detail["case_name"] or "사건")[:50].replace("/", "_")

        if format == "json":
            export_data: dict[str, Any] = {
                "case_name": detail["case_name"],
                "case_type": detail["case_type"],
                "status": detail["status"],
                "tagged_items": detail["tagged_items"],
            }
            if include in ("all", "timeline"):
                export_data["timeline"] = detail["timeline"]
            if include in ("all", "conversations"):
                export_data["conversations"] = detail["conversations"]

            content = json.dumps(export_data, ensure_ascii=False, indent=2)
            media_type = "application/json"
            filename = f"{case_name_safe}.json"
        else:
            lines: list[str] = [f"# {detail['case_name']}\n"]
            if detail["case_type"]:
                lines.append(f"사건 유형: {detail['case_type']}\n")
            lines.append("")

            if include in ("all", "timeline") and detail["timeline"]:
                lines.append("## 타임라인\n")
                for t in detail["timeline"]:
                    date_str = t.get("date_text", "")
                    lines.append(f"- [{date_str}] {t['title']}")
                    if t.get("description"):
                        lines.append(f"  {t['description']}")
                lines.append("")

            content = "\n".join(lines)
            media_type = "text/plain; charset=utf-8"
            filename = f"{case_name_safe}.txt"

    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=\"{filename}\"",
        },
    )
