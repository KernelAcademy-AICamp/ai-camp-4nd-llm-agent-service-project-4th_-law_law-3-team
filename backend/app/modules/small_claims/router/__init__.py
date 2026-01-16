"""
소액 소송 에이전트 모듈 - 나홀로 소송 지원
중고거래 사기, 떼인 알바비, 층간소음 등 소액 사건 처리 지원
"""
from fastapi import APIRouter, UploadFile, File
from typing import List

router = APIRouter()


@router.post("/interview/start")
async def start_interview(case_type: str):
    """자연어 인터뷰 시작"""
    return {
        "session_id": "interview_session_id",
        "case_type": case_type,
        "next_question": "어떤 상황인지 자세히 말씀해 주세요.",
    }


@router.post("/interview/{session_id}/answer")
async def submit_answer(session_id: str, answer: str):
    """인터뷰 답변 제출"""
    return {
        "session_id": session_id,
        "next_question": "상대방의 연락처나 계좌 정보를 알고 계신가요?",
        "collected_info": {},
        "is_complete": False,
    }


@router.post("/documents/generate")
async def generate_documents(
    session_id: str,
    document_types: List[str],
):
    """법률 서류 자동 생성 (내용증명, 지급명령신청서, 소액심판청구서)"""
    return {
        "session_id": session_id,
        "documents": [],
    }


@router.post("/evidence/upload")
async def upload_evidence(
    session_id: str,
    files: List[UploadFile] = File(...),
    evidence_type: str = "chat_log",
):
    """증거 자료 업로드"""
    return {
        "session_id": session_id,
        "uploaded_files": [],
    }


@router.post("/evidence/{session_id}/organize")
async def organize_evidence(session_id: str):
    """증거 자료 타임라인 정리 및 PDF 변환"""
    return {
        "session_id": session_id,
        "organized_pdf_url": "",
        "timeline": [],
    }


@router.get("/guide/{case_type}")
async def get_lawsuit_guide(case_type: str):
    """소송 절차 가이드 조회"""
    return {
        "case_type": case_type,
        "steps": [
            {"step": 1, "title": "내용증명 발송", "description": "..."},
            {"step": 2, "title": "지급명령 신청", "description": "..."},
            {"step": 3, "title": "소액심판 청구", "description": "..."},
        ],
    }
