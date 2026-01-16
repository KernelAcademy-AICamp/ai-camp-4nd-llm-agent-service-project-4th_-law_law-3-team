"""
스토리보드 모듈 - 타임라인 기반 스토리보드 생성
이미지 생성 AI를 활용하여 사건 내용을 시각화
"""
from fastapi import APIRouter
from typing import List

router = APIRouter()


@router.post("/generate")
async def generate_storyboard(
    title: str,
    events: List[dict],
):
    """타임라인 기반 스토리보드 생성"""
    return {
        "title": title,
        "storyboard_id": "generated_id",
        "panels": [],
    }


@router.get("/{storyboard_id}")
async def get_storyboard(storyboard_id: str):
    """스토리보드 조회"""
    return {"storyboard_id": storyboard_id}


@router.post("/{storyboard_id}/regenerate-panel")
async def regenerate_panel(storyboard_id: str, panel_index: int, new_prompt: str):
    """특정 패널 이미지 재생성"""
    return {
        "storyboard_id": storyboard_id,
        "panel_index": panel_index,
        "new_image_url": "",
    }
