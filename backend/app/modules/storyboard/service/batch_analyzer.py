"""다중 파일 병렬 분석 오케스트레이터 (Phase 2)"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from ..schema.models import EvidenceFile, EvidenceType, TimelineItem
from ..schema.responses import BatchAnalysisResult
from . import extract_timeline_from_text
from .file_validation import ValidatedFile
from .job_manager import JobManager, JobStatus
from .kakao_parser import KakaoTalkParser

logger = logging.getLogger(__name__)

# 동시 분석 제한 (서버 부하 방지)
MAX_CONCURRENT_ANALYSIS = 3

# 슬라이딩 윈도우: 카카오톡 대용량 분할 (NFR-06)
KAKAO_CHUNK_SIZE = 12_000
KAKAO_CHUNK_OVERLAP = 500


class BatchAnalyzer:
    """
    다중 파일 → 통합 타임라인 생성

    파이프라인:
    1. FileValidationGate → ValidatedFile[]
    2. EvidenceType별 분석기 라우팅
    3. asyncio.gather(Semaphore) 병렬 실행
    4. 결과 병합 + topic 자동 분류
    5. SSE 진행 상태 보고
    """

    def __init__(self, job_manager: JobManager) -> None:
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSIS)
        self._job_manager = job_manager
        self._kakao_parser = KakaoTalkParser()

    async def analyze_batch(
        self,
        job_id: str,
        validated_files: list[ValidatedFile],
        context: str = "",
    ) -> BatchAnalysisResult:
        """
        메인 배치 분석 흐름:
        1. 파일별 분석 태스크 생성
        2. Semaphore 제한 하에 병렬 실행
        3. 파일별 진행 SSE 이벤트 발행
        4. 전체 결과 병합 → topic 분류
        """
        total = len(validated_files)
        await self._job_manager.update_progress(
            job_id,
            status=JobStatus.PROCESSING,
            current_step=0,
            message=f"배치 분석 시작 ({total}개 파일)...",
        )

        tasks = [
            self._analyze_single(job_id, idx, vfile, total, context)
            for idx, vfile in enumerate(validated_files)
        ]
        results: list[list[TimelineItem] | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        return self._merge_results(results, validated_files)

    async def _analyze_single(
        self,
        job_id: str,
        index: int,
        vfile: ValidatedFile,
        total: int,
        context: str,
    ) -> list[TimelineItem]:
        """
        단일 파일 분석 (Semaphore 적용).

        EvidenceType별 라우팅:
        - KAKAO_TXT → KakaoTalkParser → 슬라이딩 윈도우 → extract_timeline
        - VOICE_RECORDING → transcribe_audio → extract_timeline
        - MESSENGER_SCREENSHOT / PHOTO → analyze_image
        - DOCUMENT → 텍스트 추출 → extract_timeline
        - TEXT_INPUT → extract_timeline (직접)
        """
        async with self._semaphore:
            await self._job_manager.update_progress(
                job_id,
                message=f"분석 중: {vfile.original_filename} ({index + 1}/{total})",
            )
            try:
                items = await self._route_by_evidence_type(vfile, context)
            except Exception as exc:
                logger.error(
                    "파일 분석 실패 (file=%s): %s",
                    vfile.original_filename,
                    exc,
                    exc_info=True,
                )
                items = []

            await self._job_manager.update_progress(
                job_id,
                current_step=index + 1,
                message=f"완료: {vfile.original_filename} ({index + 1}/{total})",
            )
            return items

    async def _route_by_evidence_type(
        self,
        vfile: ValidatedFile,
        context: str,
    ) -> list[TimelineItem]:
        """EvidenceType에 따라 적절한 분석기로 라우팅."""
        if vfile.evidence_type == EvidenceType.KAKAO_TXT:
            return await self._analyze_kakao(vfile, context)

        if vfile.evidence_type == EvidenceType.VOICE_RECORDING:
            return await self._analyze_audio(vfile, context)

        if vfile.evidence_type in (
            EvidenceType.MESSENGER_SCREENSHOT,
            EvidenceType.PHOTO,
        ):
            is_messenger = vfile.evidence_type == EvidenceType.MESSENGER_SCREENSHOT
            return await self._analyze_image(vfile, context, is_messenger=is_messenger)

        if vfile.evidence_type == EvidenceType.DOCUMENT:
            return await self._analyze_document(vfile, context)

        # TEXT_INPUT / OTHER → 텍스트로 직접 처리
        text = vfile.content.decode("utf-8", errors="replace")
        if context:
            text = f"[컨텍스트: {context}]\n\n{text}"
        result = await extract_timeline_from_text(text)
        return result.timeline if result.success else []

    async def _analyze_kakao(
        self,
        vfile: ValidatedFile,
        context: str,
    ) -> list[TimelineItem]:
        """카카오톡 .txt 파일 → 슬라이딩 윈도우 분석."""
        text = vfile.content.decode("utf-8", errors="replace")
        messages = self._kakao_parser.parse(text)

        if not messages:
            # 파싱 실패 시 원본 텍스트로 직접 추출
            result = await extract_timeline_from_text(text)
            return result.timeline if result.success else []

        chunks = self._kakao_parser.to_chunks(messages)
        all_items: list[TimelineItem] = []

        for chunk in chunks:
            input_text = chunk
            if context:
                input_text = f"[컨텍스트: {context}]\n\n{chunk}"
            result = await extract_timeline_from_text(input_text)
            if result.success:
                all_items.extend(result.timeline)

        return all_items

    async def _analyze_audio(
        self,
        vfile: ValidatedFile,
        context: str,
    ) -> list[TimelineItem]:
        """음성 파일 → STT → 타임라인 추출."""
        try:
            import io

            from .stt import transcribe_audio

            audio_io = io.BytesIO(vfile.content)
            text = await transcribe_audio(
                audio_file=audio_io,
                filename=vfile.original_filename,
                language="ko",
            )
            if not text:
                return []
            if context:
                text = f"[컨텍스트: {context}]\n\n{text}"
            result = await extract_timeline_from_text(text)
            return result.timeline if result.success else []
        except Exception as exc:
            logger.warning("음성 분석 실패 (file=%s): %s", vfile.original_filename, exc)
            return []

    async def _analyze_image(
        self,
        vfile: ValidatedFile,
        context: str,
        is_messenger: bool = False,
    ) -> list[TimelineItem]:
        """이미지 → Vision API → 타임라인 추출."""
        try:
            import io

            from .vision import analyze_image

            image_io = io.BytesIO(vfile.content)
            result_dict = await analyze_image(
                image_file=image_io,
                filename=vfile.original_filename,
                additional_context=context,
                is_messenger=is_messenger,
            )
            if not result_dict.get("success"):
                return []
            raw_items: list[Any] = result_dict.get("timeline", [])
            items: list[TimelineItem] = []
            for raw in raw_items:
                try:
                    items.append(TimelineItem(**raw))
                except Exception:
                    pass
            return items
        except Exception as exc:
            logger.warning("이미지 분석 실패 (file=%s): %s", vfile.original_filename, exc)
            return []

    async def _analyze_document(
        self,
        vfile: ValidatedFile,
        context: str,
    ) -> list[TimelineItem]:
        """문서(PDF/DOCX 등) → 텍스트 추출 → 타임라인 추출."""
        text = self._extract_text_from_document(vfile)
        if not text:
            return []
        if context:
            text = f"[컨텍스트: {context}]\n\n{text}"
        result = await extract_timeline_from_text(text)
        return result.timeline if result.success else []

    def _extract_text_from_document(self, vfile: ValidatedFile) -> str:
        """문서에서 텍스트 추출. PDF는 pypdf, DOCX는 python-docx 사용."""
        mime = vfile.mime_type
        content = vfile.content

        if mime == "application/pdf":
            try:
                import io

                import pypdf  # type: ignore[import-not-found]

                reader = pypdf.PdfReader(io.BytesIO(content))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n".join(pages)
            except Exception as exc:
                logger.warning("PDF 텍스트 추출 실패: %s", exc)
                return ""

        if mime == "application/zip":
            # DOCX는 ZIP 기반
            try:
                import io
                import zipfile

                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    if "word/document.xml" in zf.namelist():
                        import re

                        xml_content = zf.read("word/document.xml").decode("utf-8", errors="replace")
                        # XML 태그 제거
                        text = re.sub(r"<[^>]+>", " ", xml_content)
                        return " ".join(text.split())
            except Exception as exc:
                logger.warning("DOCX 텍스트 추출 실패: %s", exc)
                return ""

        # 기타: UTF-8 디코딩 시도
        return content.decode("utf-8", errors="replace")

    def _merge_results(
        self,
        results: list[list[TimelineItem] | BaseException],
        files: list[ValidatedFile],
    ) -> BatchAnalysisResult:
        """
        결과 병합:
        1. 날짜순 정렬 (기존 _date_sort_key 재활용)
        2. EvidenceFile 메타데이터 생성
        3. evidence_ids 연결
        4. topic 수집
        """
        from ..service import _date_sort_key

        all_items: list[TimelineItem] = []
        evidence_files: list[EvidenceFile] = []
        failed_files: list[str] = []
        success_count = 0

        now_iso = datetime.now(timezone.utc).isoformat()

        for idx, result in enumerate(results):
            vfile = files[idx]
            if isinstance(result, BaseException):
                logger.error("파일 분석 예외 (file=%s): %s", vfile.original_filename, result)
                failed_files.append(vfile.original_filename)
                continue

            # EvidenceFile 생성
            evidence_id = str(uuid.uuid4())
            extracted_ids = [item.id for item in result]

            # 추출된 타임라인 항목에 evidence_ids 연결
            for item in result:
                if evidence_id not in item.evidence_ids:
                    item.evidence_ids.append(evidence_id)

            evidence_file = EvidenceFile(
                evidence_id=evidence_id,
                evidence_type=vfile.evidence_type,
                filename=vfile.original_filename,
                uploaded_at=now_iso,
                file_size_kb=max(1, vfile.file_size // 1024),
                file_hash=vfile.file_hash,
                session_id="",  # 라우터에서 세션 ID 주입
                extracted_timeline_ids=extracted_ids,
                source_description=None,
            )
            evidence_files.append(evidence_file)
            all_items.extend(result)
            success_count += 1

        # 날짜순 정렬
        all_items.sort(key=lambda item: _date_sort_key(item.date))

        # order, scene_number 재할당
        for idx, item in enumerate(all_items):
            item.order = idx
            item.scene_number = idx + 1

        # 전체 topic 수집 (중복 제거, 순서 유지)
        seen_topics: set[str] = set()
        topics: list[str] = []
        for item in all_items:
            if item.topic and item.topic not in seen_topics:
                seen_topics.add(item.topic)
                topics.append(item.topic)

        return BatchAnalysisResult(
            timeline_items=all_items,
            evidence_files=evidence_files,
            topics=topics,
            total_files=len(files),
            success_count=success_count,
            failed_files=failed_files,
        )
