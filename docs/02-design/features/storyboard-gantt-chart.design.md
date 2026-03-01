# Storyboard Gantt Chart (사건 타임라인 간트차트) Design Document

> **Summary**: 법률 사건 타임라인을 vis-timeline 간트차트로 시각화 + 다중 증거 파일 업로드/병합의 상세 설계
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Version**: 0.2.0
> **Author**: Claude (Team Lead)
> **Date**: 2026-03-01
> **Status**: Draft (v0.2 — Red Team + External Consultant 설계 리뷰 통합)
> **Planning Doc**: [storyboard-gantt-chart.plan.md](../../01-plan/features/storyboard-gantt-chart.plan.md)

### Pipeline References

| Phase | Document | Status |
|-------|----------|--------|
| Phase 1 | [Plan Document](../../01-plan/features/storyboard-gantt-chart.plan.md) | v0.2 |
| Phase 2 | Design (본 문서) | v0.2 |
| Phase 3 | Red Team Report (Plan) | [redteam.md](../../03-analysis/storyboard-gantt-chart.redteam.md) |
| Phase 4 | Consulting Report (Plan) | [consulting.md](../../03-analysis/storyboard-gantt-chart.consulting.md) |
| Phase 5 | Red Team Report (Design) | [design-redteam.md](../../03-analysis/storyboard-gantt-chart.design-redteam.md) |
| Phase 6 | Consulting Report (Design) | [design-consulting.md](../../03-analysis/storyboard-gantt-chart.design-consulting.md) |

---

## 1. Overview

### 1.1 Design Goals

1. **기존 아키텍처와의 일관성**: storyboard 모듈의 기존 패턴(router → service → schema)을 그대로 유지하며 확장한다.
2. **하위 호환성 보장**: 기존 `TimelineItem` 필드와 카드 리스트 뷰를 100% 유지하면서 간트차트 뷰를 추가한다.
3. **증거-타임라인 구조적 연결**: 텍스트 목록(`evidence_items: list[str]`)에서 파일 참조 기반(`evidence_ids: list[str]` + `EvidenceFile`)으로 전환한다.
4. **보안 우선 설계**: Red Team 피드백 기반 SEC-01~06을 설계 단계에서 반영한다 (프롬프트 인젝션 방어, 파일 검증 게이트).

### 1.2 Design Principles

- **Single Responsibility**: 간트차트 렌더링(`GanttChartView`), 증거 업로드(`EvidenceUploadPanel`), 상세 패널(`GanttDetailPanel`)을 각각 독립 컴포넌트로 분리
- **Open/Closed**: `EvidenceType` enum 확장으로 새 증거 유형 추가 가능, vis-timeline 커스터마이징은 CSS + template 함수로 대응
- **Dependency Inversion**: 파일 분석 로직은 `EvidenceAnalyzer` 프로토콜로 추상화하여 파서(kakao, vision, stt) 교체 가능
- **기존 패턴 준수**: `ModuleRegistry`, `job_manager` SSE, `next/dynamic({ ssr: false })` 등 기존 인프라 100% 재활용

---

## 2. Architecture

### 2.1 Component Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Next.js App (/storyboard)                             │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    StoryboardPage (page.tsx)                         │ │
│  │                                                                     │ │
│  │  ┌──────────────────┐  ┌──────────────────────────────────────────┐│ │
│  │  │  MultiInputPanel  │  │  TimelineToolbar                        ││ │
│  │  │  (기존 유지)       │  │  + ViewToggle (카드 ↔ 간트)             ││ │
│  │  └──────────────────┘  └──────────────────────────────────────────┘│ │
│  │                                                                     │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  EvidenceUploadPanel (신규)                                   │  │ │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │  │ │
│  │  │  │ 📱 카카오 │ │ 🎤 녹음  │ │ 📄 문서  │ │ 📷 사진  │ + Add │  │ │
│  │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │  │ │
│  │  │  [분석 중... ██████░░░░ 60%] SSE Progress                   │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                                                                     │ │
│  │  ┌────────────────────────┐  ┌──────────────────────────────────┐  │ │
│  │  │ (A) TimelineView       │  │ (B) GanttChartView (신규)        │  │ │
│  │  │     (기존 카드 뷰)      │  │     vis-timeline 기반             │  │ │
│  │  │     viewMode='card'    │  │     viewMode='gantt'             │  │ │
│  │  └────────────────────────┘  │  ┌──────────────────────────┐    │  │ │
│  │          (ViewToggle로 전환)  │  │ GanttDetailPanel (신규)  │    │  │ │
│  │                              │  │ 항목 클릭 시 사이드 패널  │    │  │ │
│  │                              │  └──────────────────────────┘    │  │ │
│  │                              └──────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│                         API Service (fetch + SSE)                        │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────┼─────────────────────────────────────────────┐
│                     FastAPI Backend                                       │
│                            │                                             │
│  ┌─────────────────────────┴─────────────────────────────────────────┐   │
│  │              /api/storyboard/* (Router)                            │   │
│  │                                                                   │   │
│  │  기존 엔드포인트:                                                  │   │
│  │  POST /extract, /validate, /transcribe, /analyze-image            │   │
│  │  POST /generate-image, /generate-images-batch, /generate-video    │   │
│  │  GET  /jobs/{id}, /jobs/{id}/status (SSE)                         │   │
│  │                                                                   │   │
│  │  신규 엔드포인트:                                                  │   │
│  │  POST /analyze-batch        ← 다중 파일 일괄 분석                  │   │
│  │  POST /merge                ← 증분 타임라인 병합                   │   │
│  │  GET  /evidence/{id}        ← 증거 파일 원본 조회                  │   │
│  └────────────────────────────┬──────────────────────────────────────┘   │
│                               │                                          │
│  ┌────────────────────────────┴──────────────────────────────────────┐   │
│  │              Service Layer                                         │   │
│  │                                                                   │   │
│  │  기존 재활용:                                                      │   │
│  │  ├─ extract_timeline_from_text() ← topic/date_start/date_end 확장 │   │
│  │  ├─ transcribe_audio() (stt.py)                                   │   │
│  │  ├─ analyze_image() (vision.py) ← messenger 특화 프롬프트 추가     │   │
│  │  ├─ detect_document_type() (doc_type_detector.py) ← KAKAO 추가    │   │
│  │  └─ job_manager (SSE) ← 배치 분석 확장                            │   │
│  │                                                                   │   │
│  │  신규:                                                             │   │
│  │  ├─ FileValidationGate      ← SEC-01~06 보안 게이트               │   │
│  │  ├─ kakao_parser.py         ← 카카오톡 .txt 파싱                   │   │
│  │  ├─ batch_analyzer.py       ← 다중 파일 병렬 분석 오케스트레이터    │   │
│  │  └─ timeline_merger.py      ← 증분 병합 알고리즘                   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

#### 시퀀스 A: 다중 파일 업로드 → 간트차트 렌더링

```
사용자                Frontend                    Backend
  │                     │                           │
  │  파일 N개 선택       │                           │
  │────────────────────>│                           │
  │                     │  POST /analyze-batch      │
  │                     │  (multipart/form-data)    │
  │                     │─────────────────────────->│
  │                     │                           │── FileValidationGate
  │                     │                           │   ├─ magic number 검증
  │                     │                           │   ├─ 크기/개수 제한
  │                     │                           │   ├─ SHA-256 중복 체크
  │                     │                           │   └─ EXIF 스트리핑
  │                     │                           │
  │                     │  202 { job_id }           │── BatchAnalyzeJob 생성
  │                     │<─────────────────────────│
  │                     │                           │
  │                     │  GET /jobs/{id}/status     │── asyncio.gather(Semaphore=3)
  │                     │  (SSE stream)             │   ├─ .txt → kakao_parser
  │  진행률 표시         │<─ event: progress ─────── │   ├─ 음성 → Whisper STT
  │<────────────────────│<─ event: progress ─────── │   ├─ 이미지 → Vision
  │                     │<─ event: progress ─────── │   └─ 문서 → 텍스트 추출
  │                     │                           │
  │                     │<─ event: complete ──────── │── 결과 병합 + topic 분류
  │                     │  { timeline_items,        │
  │                     │    evidence_files }        │
  │                     │                           │
  │  간트차트 렌더링     │                           │
  │<────────────────────│                           │
  │  (vis-timeline)     │                           │
```

#### 시퀀스 B: 증분 병합

```
사용자                Frontend                    Backend
  │                     │                           │
  │  추가 파일 업로드    │                           │
  │────────────────────>│                           │
  │                     │  POST /merge              │
  │                     │  { existing_timeline,     │
  │                     │    new_files }             │
  │                     │─────────────────────────->│
  │                     │                           │── 1. 신규 파일 분석
  │                     │                           │── 2. 날짜 범위 매칭
  │                     │                           │── 3. 유사도 기반 중복 감지
  │                     │                           │── 4. 병합 실행
  │                     │  { merged_timeline,       │
  │                     │    merge_report }          │
  │                     │<─────────────────────────│
  │  간트차트 업데이트   │                           │
  │<────────────────────│                           │
```

---

## 3. Backend Design

### 3.1 Schema 변경 (`schema/models.py`)

#### 3.1.1 신규 Enum: EvidenceType

```python
class EvidenceType(str, Enum):
    """증거 파일 유형"""
    KAKAO_TXT = "kakao_txt"
    MESSENGER_SCREENSHOT = "messenger_screenshot"
    VOICE_RECORDING = "voice_recording"
    DOCUMENT = "document"
    PHOTO = "photo"
    TEXT_INPUT = "text_input"
    OTHER = "other"
```

**위치**: `schema/models.py`의 기존 Enum 블록 뒤에 추가

#### 3.1.2 신규 모델: EvidenceFile

```python
class EvidenceFile(BaseModel):
    """증거 파일 메타데이터"""
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="UUID v4")
    evidence_type: EvidenceType = Field(..., description="증거 유형")
    filename: str = Field(..., description="원본 파일명")
    uploaded_at: str = Field(..., description="업로드 시각 (ISO 8601)")
    file_size_kb: int = Field(..., description="파일 크기 (KB)")
    file_hash: str | None = Field(None, description="SHA-256 해시 (중복 감지용)")
    session_id: str = Field(..., description="소유자 세션 ID (IDOR 방어, Red Team 반영)")
    extracted_timeline_ids: list[str] = Field(default_factory=list, description="추출된 타임라인 항목 ID")
    tags: list[str] = Field(default_factory=list, description="사용자 정의 태그")
    source_description: str | None = Field(None, description="증거 설명")
```

#### 3.1.3 TimelineItem 확장 (하위 호환)

기존 필드를 **모두 유지**하고 아래 필드를 추가:

```python
class TimelineItem(BaseModel):
    # ... 기존 필드 전부 유지 ...

    # ── 간트차트 전용 신규 필드 ──
    topic: str | None = Field(None, description="사건 주제 (AI 자동 분류: 폭행, 협박, 금전 등)")
    date_start: str | None = Field(None, description="기간 시작일 (YYYY-MM-DD)")
    date_end: str | None = Field(None, description="기간 종료일 (YYYY-MM-DD)")
    evidence_ids: list[str] = Field(default_factory=list, description="연결된 증거 파일 ID (N:M)")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="추출 신뢰도 (0.0~1.0)")
```

**하위 호환 전략:**
- `topic`, `date_start`, `date_end`가 `None`이면 기존 카드 뷰 동작 그대로
- `evidence_ids`는 기본값 `[]`이므로 기존 데이터에 영향 없음
- `evidence_items: list[str]`은 그대로 유지 (텍스트 기반 레거시 지원)

#### 3.1.4 TimelineData 확장

```python
class TimelineData(BaseModel):
    # ... 기존 필드 유지 ...
    evidence_files: list[EvidenceFile] = Field(default_factory=list, description="증거 파일 목록")
    topics: list[str] = Field(default_factory=list, description="전체 주제 목록 (간트차트 Y축)")
```

### 3.2 Schema 변경 (`schema/responses.py`)

#### 3.2.1 신규 요청/응답 모델

```python
class AnalyzeBatchRequest(BaseModel):
    """다중 파일 배치 분석 요청 (multipart로 수신하므로 Form 파라미터)"""
    # 실제로는 UploadFile[] + Form 으로 수신, 이 모델은 문서화용
    pass


class AnalyzeBatchResponse(BaseModel):
    """배치 분석 응답 (job 생성)"""
    success: bool
    job_id: str | None = None
    error: str | None = None


class MergeTimelineRequest(BaseModel):
    """증분 병합 요청"""
    existing_items: list[TimelineItem] = Field(..., description="기존 타임라인 항목")
    existing_evidence: list[EvidenceFile] = Field(default_factory=list, description="기존 증거 파일")
    # 신규 파일은 multipart로 수신


class MergeTimelineResponse(BaseModel):
    """증분 병합 응답"""
    success: bool
    merged_items: list[TimelineItem] = Field(default_factory=list)
    merged_evidence: list[EvidenceFile] = Field(default_factory=list)
    merge_report: MergeReport | None = None
    error: str | None = None


class MergeReport(BaseModel):
    """병합 결과 보고"""
    new_items_added: int = 0
    duplicates_detected: int = 0
    items_updated: int = 0
    conflicts: list[MergeConflict] = Field(default_factory=list)


class MergeConflict(BaseModel):
    """병합 충돌 정보"""
    existing_item_id: str
    new_item_id: str
    conflict_type: str  # "date_overlap", "content_contradiction"
    description: str


class BatchJobProgress(BaseModel):
    """배치 분석 진행 상태 (SSE event data 확장)"""
    job_id: str
    status: str
    progress: int  # 0~100
    current_file: str | None = None
    current_file_index: int = 0
    total_files: int = 0
    message: str = ""
    # 완료 시 결과
    result: BatchAnalysisResult | None = None
    error: str | None = None


class BatchAnalysisResult(BaseModel):
    """배치 분석 최종 결과"""
    timeline_items: list[TimelineItem]
    evidence_files: list[EvidenceFile]
    topics: list[str]
    summary: str | None = None
    total_files: int
    success_count: int
    failed_files: list[str] = Field(default_factory=list)
```

### 3.3 Service 설계

#### 3.3.1 FileValidationGate (`service/file_validation.py`) — SEC-01~06

```python
"""파일 업로드 보안 검증 게이트 (SEC-01~06)"""

# 파일 크기 제한 상수
MAX_FILE_SIZES: dict[str, int] = {
    "audio": 25 * 1024 * 1024,     # 25MB
    "image": 20 * 1024 * 1024,     # 20MB
    "document": 10 * 1024 * 1024,  # 10MB
    "text": 5 * 1024 * 1024,       # 5MB
}
MAX_FILES_PER_BATCH = 10
MAX_FILES_PER_DAY = 50  # 사용자별 일일 할당량


class FileValidationGate:
    """
    SEC-02: 매직 넘버 + 확장자 검증
    SEC-04: 크기/개수 제한
    SEC-05: EXIF 메타데이터 제거
    SEC-06: SHA-256 해시 기반 중복 감지
    """

    async def validate_and_prepare(
        self,
        files: list[UploadFile],
    ) -> list[ValidatedFile]:
        """
        1. 파일 개수 제한 확인 (MAX_FILES_PER_BATCH)
        2. 각 파일별:
           a. 매직 넘버로 실제 파일 유형 판별
           b. 확장자 ↔ 매직 넘버 일치 검증
           c. 파일 크기 제한 확인
           d. SHA-256 해시 계산 (중복 감지)
           e. 이미지인 경우 EXIF 메타데이터 스트리핑
        3. ValidatedFile 목록 반환 (또는 HTTPException 발생)
        """
        ...

    def _check_magic_number(self, header: bytes, filename: str) -> str:
        """매직 넘버 → MIME 타입 매핑"""
        ...

    def _strip_exif(self, image_bytes: bytes) -> bytes:
        """Pillow로 EXIF 메타데이터 제거"""
        ...

    def _compute_hash(self, file_bytes: bytes) -> str:
        """SHA-256 해시 계산"""
        ...


# 매직 넘버 → MIME 매핑 상수
MAGIC_NUMBERS: dict[bytes, str] = {
    b'\xff\xd8\xff': 'image/jpeg',
    b'\x89PNG': 'image/png',
    b'GIF8': 'image/gif',
    b'RIFF': 'audio/wav',       # RIFF....WAVE
    b'\xff\xfb': 'audio/mpeg',  # MP3
    b'\xff\xf3': 'audio/mpeg',
    b'ID3': 'audio/mpeg',       # MP3 with ID3 tag
    b'ftyp': 'audio/mp4',       # M4A (offset 4)
    b'%PDF': 'application/pdf',
    b'PK': 'application/zip',   # DOCX, XLSX 등
}


@dataclass
class ValidatedFile:
    """검증 완료된 파일"""
    original_filename: str
    content: bytes
    mime_type: str
    file_size: int
    file_hash: str
    evidence_type: EvidenceType
    is_duplicate: bool = False
```

#### 3.3.2 KakaoTalkParser (`service/kakao_parser.py`)

```python
"""카카오톡 대화 내보내기 .txt 파일 파서 (FR-02)"""

import re
from dataclasses import dataclass


@dataclass
class KakaoMessage:
    """파싱된 카카오톡 메시지"""
    sender: str
    datetime_str: str       # "YYYY-MM-DD HH:MM"
    date: str               # "YYYY-MM-DD"
    time: str               # "HH:MM"
    content: str


# 카카오톡 내보내기 형식 패턴 (다중 형식 지원)
KAKAO_PATTERNS: list[re.Pattern] = [
    # 형식 1: "2024년 1월 15일 오후 3:42, 홍길동 : 내용"
    re.compile(
        r'^(\d{4})년 (\d{1,2})월 (\d{1,2})일 (오전|오후) (\d{1,2}):(\d{2}),\s*(.+?)\s*:\s*(.+)$'
    ),
    # 형식 2: "[홍길동] [오후 3:42] 내용" (날짜 헤더 별도)
    re.compile(
        r'^\[(.+?)\]\s*\[(오전|오후)\s*(\d{1,2}):(\d{2})\]\s*(.+)$'
    ),
]

# 날짜 헤더 패턴
DATE_HEADER_PATTERN = re.compile(
    r'^-+\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*\w+요일\s*-+$'
)


class KakaoTalkParser:
    """
    카카오톡 .txt 파일 → KakaoMessage 리스트

    지원 형식:
    1. PC 카카오톡 내보내기 (날짜 + 시간 한 줄)
    2. 모바일 카카오톡 내보내기 (날짜 헤더 + [발신자] [시간] 형식)
    """

    def parse(self, text: str) -> list[KakaoMessage]:
        """텍스트 → 메시지 리스트 (시간순 정렬)"""
        ...

    def to_timeline_text(self, messages: list[KakaoMessage]) -> str:
        """
        메시지 리스트 → LLM 입력용 텍스트
        (슬라이딩 윈도우 대상: NFR-06)
        """
        ...
```

#### 3.3.3 BatchAnalyzer (`service/batch_analyzer.py`)

```python
"""다중 파일 병렬 분석 오케스트레이터"""

import asyncio

# 동시 분석 제한 (서버 부하 방지)
MAX_CONCURRENT_ANALYSIS = 3

# 슬라이딩 윈도우: 카카오톡 대용량 분할 (NFR-06)
KAKAO_CHUNK_SIZE = 12_000  # 문자 수 기준 (~3,000 토큰)
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

    def __init__(self, job_manager: JobManager):
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSIS)
        self._job_manager = job_manager

    async def analyze_batch(
        self,
        job_id: str,
        validated_files: list[ValidatedFile],
    ) -> BatchAnalysisResult:
        """
        메인 배치 분석 흐름:
        1. 파일별 분석 태스크 생성
        2. Semaphore 제한 하에 병렬 실행
        3. 파일별 진행 SSE 이벤트 발행
        4. 전체 결과 병합 → topic 분류
        """
        tasks = []
        for i, vfile in enumerate(validated_files):
            tasks.append(self._analyze_single(job_id, i, vfile))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._merge_results(results, validated_files)

    async def _analyze_single(
        self, job_id: str, index: int, vfile: ValidatedFile,
    ) -> list[TimelineItem]:
        """
        단일 파일 분석 (Semaphore 적용)

        EvidenceType별 라우팅:
        - KAKAO_TXT → KakaoTalkParser → 슬라이딩 윈도우 → extract_timeline
        - VOICE_RECORDING → transcribe_audio → extract_timeline
        - MESSENGER_SCREENSHOT → analyze_image (messenger 프롬프트)
        - PHOTO → analyze_image (일반 프롬프트)
        - DOCUMENT → 텍스트 추출 → extract_timeline
        - TEXT_INPUT → extract_timeline (직접)
        """
        async with self._semaphore:
            # SEC-01: LLM 호출 시 System/User Role 분리
            # 사용자 데이터는 반드시 User 메시지에만 포함
            ...

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
        4. topic 자동 분류 (전체 항목 기준)
        """
        ...
```

#### 3.3.4 TopicClassifier — 사건 주제 분류 전략

```python
"""사건 주제 자동 분류 (FR-08)"""

# 방법: extract_timeline_from_text 프롬프트에 topic 필드 추가
# → LLM이 추출 시점에 함께 분류 (별도 API 호출 불필요)

# 추출 프롬프트에 추가할 지시:
TOPIC_EXTRACTION_INSTRUCTION = """
## topic (사건 주제) 분류 지시
각 이벤트의 `topic`에 해당 사건의 법적 주제를 한국어로 작성.
- 예시: "폭행", "협박", "금전 갈취", "스토킹", "명예훼손", "업무 배제", "인사 불이익"
- 하나의 이벤트에 하나의 topic만 배정 (가장 핵심적인 것)
- 전체 타임라인에서 topic이 3~8개 범위가 되도록 적절히 그룹핑
- 너무 세분화하지 말 것 (예: "1차 폭행", "2차 폭행" → "폭행"으로 통합)
"""

# date_start, date_end 추출 지시:
DATE_RANGE_INSTRUCTION = """
## date_start, date_end (기간) 지시
- 단발성 이벤트: date_start = date_end = date
- 지속적 사건: date_start = 시작일, date_end = 종료일
- 종료일 불명: date_end = null (간트차트에서 현재까지 표시)
"""
```

#### 3.3.5 TimelineMerger (`service/timeline_merger.py`)

```python
"""증분 타임라인 병합 (FR-07)"""


class TimelineMerger:
    """
    3단계 병합 알고리즘:

    Step 1: 날짜 범위 매칭
      - 기존 항목과 신규 항목의 date/date_start/date_end 비교
      - 날짜 범위 겹침이 있는 항목쌍을 후보로 선정

    Step 2: LLM 유사도 기반 중복 감지
      - 후보쌍의 title + description을 LLM에 전달
      - "같은 사건인가?" 판단 (yes/no + confidence)
      - 후보가 5개 이하일 때만 LLM 호출 (비용 제어)

    Step 3: 병합 실행
      - 중복: 기존 항목에 evidence_ids 추가, description 보강
      - 신규: 타임라인에 추가, 날짜순 재정렬
      - 충돌: MergeConflict 보고 (동일 시간대 상반 내용)
    """

    async def merge(
        self,
        existing_items: list[TimelineItem],
        new_items: list[TimelineItem],
        existing_evidence: list[EvidenceFile],
        new_evidence: list[EvidenceFile],
    ) -> MergeTimelineResponse:
        ...

    def _find_date_overlap_candidates(
        self,
        existing: list[TimelineItem],
        new: list[TimelineItem],
    ) -> list[tuple[TimelineItem, TimelineItem]]:
        """날짜 범위 겹침 후보쌍"""
        ...

    async def _check_duplicates_with_llm(
        self,
        candidates: list[tuple[TimelineItem, TimelineItem]],
    ) -> list[DuplicateResult]:
        """LLM 기반 중복 판단 (max 5쌍)"""
        ...
```

### 3.4 API Endpoint 설계

#### 3.4.1 POST `/api/storyboard/analyze-batch`

```python
@router.post("/analyze-batch", response_model=AnalyzeBatchResponse)
async def analyze_batch_endpoint(
    files: list[UploadFile] = File(..., description="증거 파일 (최대 10개)"),
    context: str = Form(default="", description="추가 컨텍스트"),
) -> AnalyzeBatchResponse:
    """
    다중 증거 파일 일괄 분석 → 통합 타임라인 생성

    1. FileValidationGate 통과
    2. BatchAnalyzeJob 생성 (job_id 반환)
    3. 백그라운드에서 asyncio.create_task 실행
    4. SSE /jobs/{id}/status 로 진행 상태 확인

    에러 코드:
    - 400: 파일 없음 / 개수 초과
    - 413: 파일 크기 초과
    - 415: 지원하지 않는 파일 형식
    - 422: 파일 검증 실패 (매직 넘버 불일치)
    """
```

#### 3.4.2 POST `/api/storyboard/merge`

```python
@router.post("/merge", response_model=MergeTimelineResponse)
async def merge_timeline_endpoint(
    existing_timeline: str = Form(..., description="기존 타임라인 JSON"),
    files: list[UploadFile] = File(default=[], description="추가 증거 파일"),
    text: str = Form(default="", description="추가 텍스트 입력"),
) -> MergeTimelineResponse:
    """
    기존 타임라인에 새 증거/텍스트 병합

    1. 기존 타임라인 JSON 파싱
    2. 신규 입력 분석 (파일 + 텍스트)
    3. TimelineMerger로 3단계 병합
    4. 병합 결과 + 보고서 반환
    """
```

#### 3.4.3 GET `/api/storyboard/evidence/{evidence_id}`

```python
@router.get("/evidence/{evidence_id}")
async def get_evidence_file(evidence_id: str) -> FileResponse:
    """
    증거 파일 원본 조회

    SEC-03: UUID v4 + 세션 기반 소유권 검증
    - evidence_id는 UUID v4 형식 필수
    - (현재는 세션 기반, 추후 인증 체계 도입 시 확장)
    """
```

### 3.5 extract_timeline_from_text 프롬프트 확장

기존 `EXTRACTION_SYSTEM_PROMPT`에 아래 내용을 **추가** (기존 프롬프트는 수정하지 않음):

```python
# service/__init__.py에 추가할 프롬프트 확장

GANTT_CHART_FIELDS_INSTRUCTION = """
## 간트차트 추가 필드 (신규)

각 이벤트에 아래 필드를 추가로 추출하세요:

- **topic**: 사건의 법적 주제 (한국어, 예: "폭행", "협박", "금전 갈취")
  - 전체 타임라인에서 3~8개 범위로 그룹핑
  - 하나의 이벤트에 하나의 topic만

- **date_start**: 사건 시작일 (YYYY-MM-DD 또는 YYYY-MM 또는 YYYY)
  - 단발성이면 date와 동일
  - 지속적이면 시작일

- **date_end**: 사건 종료일 (같은 형식)
  - 단발성이면 date_start와 동일
  - 현재 진행 중이면 null

- **confidence**: 추출 신뢰도 (0.0~1.0)
  - 1.0: 날짜/내용이 원문에 명확히 기재
  - 0.7~0.9: 문맥에서 추론
  - 0.5 미만: 추정이 많음
"""
```

### 3.6 doc_type_detector 확장

```python
# service/doc_type_detector.py에 추가

# 카카오톡 내보내기 파일 감지 패턴
KAKAOTALK_INDICATORS = [
    "카카오톡 대화",
    "님과 카카오톡 대화",
    "저장한 날짜",
    re.compile(r'\d{4}년 \d{1,2}월 \d{1,2}일 (오전|오후)'),
    re.compile(r'\[.+\] \[(오전|오후) \d{1,2}:\d{2}\]'),
]

def detect_document_type(text: str) -> str:
    """
    기존 civil/criminal/public 분류에 추가:
    - "kakaotalk": 카카오톡 대화 내보내기
    """
    # 카카오톡 패턴 우선 검사
    for indicator in KAKAOTALK_INDICATORS:
        if isinstance(indicator, str) and indicator in text:
            return "kakaotalk"
        if isinstance(indicator, re.Pattern) and indicator.search(text):
            return "kakaotalk"

    # 기존 분류 로직 유지
    ...
```

---

## 4. Frontend Design

### 4.1 신규 의존성

```json
{
  "dependencies": {
    "vis-timeline": "^7.7.3",
    "@datou/react-vis-timeline": "^1.2.0",
    "vis-data": "^7.1.9"
  }
}
```

**SSR 제약 해결**: `next/dynamic({ ssr: false })` — 기존 mock-trial의 Phaser.js와 동일 패턴

### 4.2 컴포넌트 설계

#### 4.2.1 GanttChartView (`components/GanttChartView.tsx`)

```typescript
// next/dynamic으로 SSR 제외
const GanttChartView = dynamic(() => import('./GanttChartViewInner'), {
  ssr: false,
  loading: () => <GanttChartSkeleton />,
})
```

**GanttChartViewInner 핵심 로직:**

```typescript
interface GanttChartViewProps {
  items: TimelineItem[]
  evidenceFiles: EvidenceFile[]
  onItemSelect: (item: TimelineItem) => void
  onEvidenceClick: (evidenceId: string) => void
}

// vis-timeline 데이터 변환
function timelineItemsToVisItems(items: TimelineItem[]): DataSet<VisTimelineItem> {
  return new DataSet(items.map(item => ({
    id: item.id,
    group: item.topic ?? 'uncategorized',
    content: item.title,
    start: item.date_start ?? item.date,
    end: item.date_end ?? item.date_start ?? item.date,
    type: (item.date_start && item.date_end && item.date_start !== item.date_end)
      ? 'range' : 'point',
    className: `gantt-item confidence-${getConfidenceLevel(item.confidence)}`,
    title: item.descriptionShort ?? item.description,  // 호버 툴팁
  })))
}

// Y축 그룹 = topics
function topicsToVisGroups(items: TimelineItem[]): DataSet<VisTimelineGroup> {
  const topics = [...new Set(items.map(i => i.topic).filter(Boolean))]
  return new DataSet([
    ...topics.map(topic => ({ id: topic, content: topic })),
    // 증거 마커 행 (별도 그룹)
    { id: 'evidence-marker', content: '증거', className: 'evidence-group' },
  ])
}

// 증거 마커 = EvidenceFile → 포인트 아이콘
function evidenceToVisMarkers(
  files: EvidenceFile[],
  items: TimelineItem[],
): DataSet<VisTimelineItem> {
  // evidence_id → 연결된 TimelineItem의 날짜 매핑
  // 유형별 아이콘: 📱 카카오, 🎤 녹음, 📷 사진, 📄 문서
  ...
}
```

**vis-timeline 옵션:**

```typescript
const options: TimelineOptions = {
  orientation: { axis: 'top', item: 'top' },
  stack: true,
  showCurrentTime: false,
  zoomMin: 1000 * 60 * 60 * 24 * 7,     // 최소 1주
  zoomMax: 1000 * 60 * 60 * 24 * 365 * 5, // 최대 5년
  tooltip: { followMouse: true, overflowMethod: 'cap' },
  groupOrder: 'content',  // 알파벳순 (topic)
  margin: { item: 10, axis: 40 },
  locale: 'ko',
}
```

#### 4.2.2 EvidenceUploadPanel (`components/EvidenceUploadPanel.tsx`)

```typescript
interface EvidenceUploadPanelProps {
  onUploadComplete: (result: BatchAnalysisResult) => void
  existingEvidence: EvidenceFile[]
}

// 핵심 기능:
// 1. 드래그앤드롭 + 파일 선택 (최대 10개)
// 2. 파일별 유형 자동 감지 + 아이콘 표시
// 3. SSE로 파일별 분석 진행률 표시
// 4. 완료 후 onUploadComplete 콜백

// 파일 유형 아이콘 매핑
const EVIDENCE_TYPE_ICONS: Record<EvidenceType, { icon: string; label: string }> = {
  kakao_txt: { icon: '📱', label: '카카오톡' },
  messenger_screenshot: { icon: '💬', label: '메신저 캡처' },
  voice_recording: { icon: '🎤', label: '음성 녹음' },
  document: { icon: '📄', label: '문서' },
  photo: { icon: '📷', label: '사진' },
  text_input: { icon: '✏️', label: '텍스트' },
  other: { icon: '📎', label: '기타' },
}
```

#### 4.2.3 GanttDetailPanel (`components/GanttDetailPanel.tsx`)

```typescript
interface GanttDetailPanelProps {
  selectedItem: TimelineItem | null
  evidenceFiles: EvidenceFile[]
  onClose: () => void
  onEvidenceClick: (evidenceId: string) => void
}

// 표시 내용:
// 1. 이벤트 기본 정보 (날짜, 제목, 상세 설명)
// 2. 참여자 정보 (역할별 색상 코딩 — 기존 PARTICIPANT_ROLE_CONFIG 재활용)
// 3. 연결된 증거 목록 (evidence_ids → EvidenceFile 매핑)
//    - 각 증거 클릭 시 원본 조회 (GET /evidence/{id})
// 4. 법적 의미 (legalSignificance)
// 5. 신뢰도 배지 (FR-13: confidence 기반)
// 6. 핵심 대사 (keyDialogue)
```

#### 4.2.4 ViewToggle (TimelineToolbar 확장)

```typescript
// 기존 TimelineToolbar.tsx에 추가
type ViewMode = 'card' | 'gantt'

// 토글 버튼: 카드 뷰 ↔ 간트차트 뷰
// 간트차트 뷰 시 이미지 생성 관련 버튼 숨김
```

### 4.3 타입 변경 (`types/index.ts`)

```typescript
// 신규 타입 추가

export type EvidenceType =
  | 'kakao_txt'
  | 'messenger_screenshot'
  | 'voice_recording'
  | 'document'
  | 'photo'
  | 'text_input'
  | 'other'

export interface EvidenceFile {
  evidence_id: string
  evidence_type: EvidenceType
  filename: string
  uploaded_at: string
  file_size_kb: number
  file_hash?: string
  extracted_timeline_ids: string[]
  tags: string[]
  source_description?: string
}

// TimelineItem 확장 (기존 필드 유지 + 추가)
export interface TimelineItem {
  // ... 기존 필드 전부 유지 ...

  // 간트차트 전용
  topic?: string
  dateStart?: string   // camelCase (프론트엔드)
  dateEnd?: string
  evidenceIds?: string[]
  confidence?: number
}

// TimelineData 확장
export interface TimelineData {
  // ... 기존 필드 유지 ...
  evidence_files?: EvidenceFile[]
  topics?: string[]
}

// 배치 분석 응답
export interface BatchAnalysisResult {
  timeline_items: TimelineItem[]
  evidence_files: EvidenceFile[]
  topics: string[]
  summary?: string
  total_files: number
  success_count: number
  failed_files: string[]
}

// 병합 응답
export interface MergeTimelineResponse {
  success: boolean
  merged_items: TimelineItem[]
  merged_evidence: EvidenceFile[]
  merge_report?: MergeReport
  error?: string
}

export interface MergeReport {
  new_items_added: number
  duplicates_detected: number
  items_updated: number
  conflicts: MergeConflict[]
}

export interface MergeConflict {
  existing_item_id: string
  new_item_id: string
  conflict_type: 'date_overlap' | 'content_contradiction'
  description: string
}
```

### 4.4 Hook 설계

#### useGanttChart (`hooks/useGanttChart.ts`)

```typescript
/**
 * 간트차트 상태 관리 훅
 *
 * 책임:
 * 1. TimelineItem[] → vis-timeline DataSet 변환
 * 2. 줌/패닝 상태 관리
 * 3. 선택된 항목 상태
 * 4. 필터링 (topic별, 날짜 범위)
 */
export function useGanttChart(items: TimelineItem[], evidenceFiles: EvidenceFile[]) {
  const [selectedItem, setSelectedItem] = useState<TimelineItem | null>(null)
  const [visibleTopics, setVisibleTopics] = useState<Set<string>>(new Set())

  const visItems = useMemo(() => timelineItemsToVisItems(items), [items])
  const visGroups = useMemo(() => topicsToVisGroups(items), [items])
  const visMarkers = useMemo(
    () => evidenceToVisMarkers(evidenceFiles, items),
    [evidenceFiles, items],
  )

  return { visItems, visGroups, visMarkers, selectedItem, setSelectedItem, visibleTopics, setVisibleTopics }
}
```

#### useEvidenceUpload (`hooks/useEvidenceUpload.ts`)

```typescript
/**
 * 증거 파일 업로드 + SSE 진행 상태 관리 훅
 *
 * 책임:
 * 1. 파일 업로드 (POST /analyze-batch)
 * 2. SSE 진행 상태 수신
 * 3. 완료 시 결과 반환
 * 4. 에러 핸들링
 */
export function useEvidenceUpload() {
  const [isUploading, setIsUploading] = useState(false)
  const [progress, setProgress] = useState<BatchJobProgress | null>(null)
  const [error, setError] = useState<string | null>(null)

  const uploadFiles = useCallback(async (files: File[], context?: string) => {
    // 1. FormData 구성
    // 2. POST /api/storyboard/analyze-batch
    // 3. SSE /api/storyboard/jobs/{job_id}/status 구독
    // 4. 완료 시 BatchAnalysisResult 반환
  }, [])

  return { uploadFiles, isUploading, progress, error }
}
```

### 4.5 스타일링

```css
/* vis-timeline 커스텀 스타일 (Tailwind + CSS 변수) */

/* 간트차트 기간 막대 */
.gantt-item.vis-item.vis-range {
  @apply rounded-md border-0;
  background-color: var(--gantt-bar-color, hsl(220, 70%, 50%));
  opacity: 0.85;
}

/* 신뢰도별 투명도 */
.confidence-high { opacity: 1.0; }
.confidence-medium { opacity: 0.7; }
.confidence-low { opacity: 0.5; border-style: dashed; }

/* 증거 마커 */
.evidence-marker .vis-item-content {
  font-size: 1.2rem;
  cursor: pointer;
}

/* 증거 그룹 행 */
.evidence-group {
  background-color: hsl(0, 0%, 95%);
  border-top: 2px solid hsl(0, 0%, 80%);
}

/* 다크 모드 대응 */
.dark .gantt-item.vis-item.vis-range {
  background-color: var(--gantt-bar-color-dark, hsl(220, 60%, 40%));
}
```

---

## 5. Implementation Order (구현 순서)

### Phase 1: 간트차트 기본 뷰 (MVP)

| 순서 | 작업 | 파일 | 의존성 |
|------|------|------|--------|
| 1-1 | `vis-timeline` + wrapper 설치 | `frontend/package.json` | 없음 |
| 1-2 | `TimelineItem`에 `topic`, `date_start`, `date_end`, `confidence` 추가 | `backend/.../schema/models.py` | 없음 |
| 1-3 | 프론트엔드 타입 동기화 | `frontend/.../types/index.ts` | 1-2 |
| 1-4 | 추출 프롬프트에 `topic`/`date_start`/`date_end`/`confidence` 지시 추가 | `backend/.../service/__init__.py` | 1-2 |
| 1-5 | `GanttChartView` 컴포넌트 구현 | `frontend/.../components/GanttChartView.tsx` | 1-1, 1-3 |
| 1-6 | `useGanttChart` 훅 구현 | `frontend/.../hooks/useGanttChart.ts` | 1-5 |
| 1-7 | `GanttDetailPanel` 컴포넌트 구현 | `frontend/.../components/GanttDetailPanel.tsx` | 1-5 |
| 1-8 | `TimelineToolbar`에 ViewToggle 추가 | `frontend/.../components/TimelineToolbar.tsx` | 1-5 |
| 1-9 | vis-timeline CSS 커스터마이징 | `frontend/src/styles/gantt.css` | 1-5 |

### Phase 2: 다중 증거 업로드 + 연결

| 순서 | 작업 | 파일 | 의존성 |
|------|------|------|--------|
| 2-1 | `EvidenceFile`, `EvidenceType` 스키마 추가 | `backend/.../schema/models.py` | Phase 1 |
| 2-2 | 신규 요청/응답 스키마 추가 | `backend/.../schema/responses.py` | 2-1 |
| 2-3 | `FileValidationGate` 구현 | `backend/.../service/file_validation.py` | 없음 |
| 2-4 | `KakaoTalkParser` 구현 | `backend/.../service/kakao_parser.py` | 없음 |
| 2-5 | `BatchAnalyzer` 구현 | `backend/.../service/batch_analyzer.py` | 2-3, 2-4 |
| 2-6 | `POST /analyze-batch` 엔드포인트 | `backend/.../router/__init__.py` | 2-2, 2-5 |
| 2-7 | `GET /evidence/{id}` 엔드포인트 | `backend/.../router/__init__.py` | 2-1 |
| 2-8 | `doc_type_detector` KAKAO 확장 | `backend/.../service/doc_type_detector.py` | 2-4 |
| 2-9 | `job_manager` 배치 분석 확장 | `backend/.../service/job_manager.py` | 2-5 |
| 2-10 | 프론트엔드 타입 동기화 | `frontend/.../types/index.ts` | 2-2 |
| 2-11 | `useEvidenceUpload` 훅 구현 | `frontend/.../hooks/useEvidenceUpload.ts` | 2-10 |
| 2-12 | `EvidenceUploadPanel` 컴포넌트 구현 | `frontend/.../components/EvidenceUploadPanel.tsx` | 2-11 |
| 2-13 | 증거 마커 렌더링 확장 | `frontend/.../components/GanttChartView.tsx` | 2-10 |

### Phase 3: 증분 병합 + 고급 기능

| 순서 | 작업 | 파일 | 의존성 |
|------|------|------|--------|
| 3-1 | `TimelineMerger` 구현 | `backend/.../service/timeline_merger.py` | Phase 2 |
| 3-2 | `POST /merge` 엔드포인트 | `backend/.../router/__init__.py` | 3-1 |
| 3-3 | 메신저 Vision 특화 프롬프트 | `backend/.../service/vision.py` | 없음 |
| 3-4 | 슬라이딩 윈도우 분할 (NFR-06) | `backend/.../service/batch_analyzer.py` | 없음 |
| 3-5 | 병합 UI (추가 파일 업로드) | `frontend/.../components/EvidenceUploadPanel.tsx` | 3-2 |
| 3-6 | 충돌 표시 UI (FR-15) | `frontend/.../components/GanttChartView.tsx` | 3-2 |

---

## 6. Testing Strategy

### 6.1 Backend 단위 테스트

| 테스트 대상 | 파일 | 주요 테스트 케이스 |
|------------|------|-------------------|
| KakaoTalkParser | `tests/unit/test_kakao_parser.py` | PC/모바일 형식, 멀티라인 메시지, 빈 파일, 유니코드 |
| FileValidationGate | `tests/unit/test_file_validation.py` | 매직 넘버 일치/불일치, 크기 초과, EXIF 제거, 해시 중복 |
| TopicClassifier | `tests/unit/test_topic_classifier.py` | topic 추출 정확도 (가상 데이터) |
| TimelineMerger | `tests/unit/test_timeline_merger.py` | 중복 감지, 날짜 겹침, 병합 보고서 |

### 6.2 Backend 통합 테스트

| 테스트 대상 | 파일 | 주요 테스트 케이스 |
|------------|------|-------------------|
| analyze-batch API | `tests/integration/test_storyboard_batch.py` | 다중 파일 업로드, SSE 진행률, 에러 응답 |
| merge API | `tests/integration/test_storyboard_merge.py` | 증분 병합, 충돌 감지 |

### 6.3 FE-BE 계약 테스트 (Consultant 반영)

| 테스트 대상 | 파일 | 주요 테스트 케이스 |
|------------|------|-------------------|
| API 스키마 계약 | `tests/contract/test_storyboard_schema.py` | Backend Pydantic 스키마 → Frontend TypeScript 타입 필드 일치 검증 |
| 에러 응답 구조 | `tests/contract/test_error_responses.py` | 4xx/5xx 응답의 `code`, `message`, `details` 필드 존재 확인 |

### 6.4 보안/프롬프트 가드 테스트 (Red Team + Consultant 반영)

| 테스트 대상 | 파일 | 주요 테스트 케이스 |
|------------|------|-------------------|
| Prompt Injection Guard | `tests/unit/test_prompt_guard.py` | 악의적 프롬프트 주입 시 추출 결과 필터링 동작 |
| Zip Slip 방어 | `tests/unit/test_zip_slip.py` | 상위 디렉토리 참조 포함 압축 파일 차단 |
| Token Limit Guard | `tests/unit/test_token_limit.py` | 50,000자 초과 청크의 사전 차단 동작 |

### 6.5 프론트엔드 (수동 검증)

| 검증 항목 | 기준 |
|----------|------|
| vis-timeline 렌더링 | 100개 항목에서 60fps |
| 뷰 전환 | 카드 ↔ 간트 토글 시 데이터 유지 |
| 반응형 | 모바일에서 조회 가능 (편집 불가) |
| SSE 진행률 | 파일별 진행률 정확히 표시 |
| 접근성 | 토글/상태 버튼에 `aria-label`, `aria-expanded` 포함, 키보드 포커스 흐름 정상 동작 |

### 6.6 CI 게이트 (Consultant 반영, 로드맵)

| 게이트 | 기준 | 우선순위 |
|--------|------|---------|
| Coverage Threshold | Backend 단위 테스트 80%+ | Phase 2 |
| LLM Eval Regression | 추출 F1 ≥ 0.75 유지 | Phase 3 |
| Accessibility Smoke | axe-core 자동 스캔 Critical 0건 | Phase 3 |

---

## 7. Security Design (Red Team + Consultant 반영)

### 7.1 핵심 보안 대응 (기존)

| SEC ID | 위협 | 설계 대응 | 구현 위치 |
|--------|------|----------|----------|
| SEC-01 | 프롬프트 인젝션 | System Role에 추출 지시, User Role에 사용자 데이터 분리. Rule-based 결과 필터링 (날짜 형식 검증, 신뢰도 임계값) | `batch_analyzer.py`, `service/__init__.py` |
| SEC-02 | 악성 파일 업로드 | 매직 넘버 검증, 확장자↔MIME 일치, 실행 권한 제거, 격리 경로(`/tmp/storyboard/`) | `file_validation.py` |
| SEC-03 | IDOR | `evidence_id`는 UUID v4, `session_id` 필드로 소유권 명시 연결, Router 레벨 검증 강제 | `router/__init__.py`, `EvidenceFile.session_id` |
| SEC-04 | DoS | 파일당 크기 제한, 배치 10개, 일일 50개, 처리 타임아웃 60초/파일 | `file_validation.py`, `batch_analyzer.py` |
| SEC-05 | EXIF 유출 | Pillow로 EXIF 자동 스트리핑 | `file_validation.py` |
| SEC-06 | 중복 업로드 | SHA-256 해시 비교 → 중복 시 기존 EvidenceFile 재사용 | `file_validation.py` |

### 7.2 설계 리뷰 추가 보안 항목 (v0.2 신규)

| SEC ID | 등급 | 위협 | 설계 대응 | 출처 |
|--------|------|------|----------|------|
| SEC-07 | High | SSE Connection Leak | `JobManager`에 하트비트(30초) 추가, 응답 없는 구독자 60초 후 강제 제거 Cleanup 루틴 | Red Team |
| SEC-08 | High | Timeline Merge Conflict Bypass | 병합 전 Diff/Review UI 단계 강제, 중요 필드 변경 시 감사 로그 기록 | Red Team |
| SEC-09 | Medium | Zip Slip / Path Traversal | DOCX 등 PK 매직넘버 파일 처리 시 `zipfile` 개별 경로 검증 추출, 상위 디렉토리 참조 차단 | Red Team |
| SEC-10 | Medium | LLM Token Explosion | 슬라이딩 윈도우 분할 전 청크당 최대 길이(50,000자) + tiktoken 토큰 수 사전 검사 | Red Team |
| SEC-11 | Medium | 증거 삭제 시 참조 무결성 | `EvidenceFile` 삭제 시 `TimelineItem.evidence_ids`에서 해당 ID 제거 (Nullify 전략) | Red Team |
| SEC-12 | Low | 유사 파일 중복 | SHA-256 외 메타데이터 보완 감지 (파일 크기 + 이름 + 생성일 조합) | Red Team |

---

## 8. Architecture Improvement Notes (Red Team + Consultant 반영)

> 아래 항목은 현재 설계의 범위를 넘지 않되, 구현 시 고려해야 할 아키텍처 개선 방향이다.

### 8.1 Backend

| 항목 | 현재 설계 | 개선 방향 | 우선순위 |
|------|----------|----------|---------|
| JobManager 영속성 | 인메모리 `_jobs: dict` | Redis 또는 DB 기반 상태 관리 (서버 재시작 시 작업 소실 방지) | Phase 2 이후 |
| Worker 분리 | `asyncio.create_task` | CPU/IO 집약 작업(Whisper STT 등)은 `ProcessPoolExecutor` 또는 Celery 워커 분리 | 트래픽 증가 시 |
| BatchAnalyzer DI | 파서 직접 참조 | Registry 패턴으로 분석기 동적 로드 (새 증거 유형 추가 시 OCP 준수) | Phase 3 |
| API 응답 코드 | `200 OK` (analyze-batch) | `201 Created` + Location 헤더에 job_id 경로 포함 | Phase 1 |
| 에러 구조 표준화 | `detail` 중심 HTTPException | `{code, message, details[], request_id}` 공통 스키마 | Phase 1 |

### 8.2 Frontend

| 항목 | 현재 설계 | 개선 방향 | 우선순위 |
|------|----------|----------|---------|
| 동적 줌 레벨 | `zoomMin`/`zoomMax` 고정 | 데이터 시간 범위에 따라 초기 줌 레벨 동적 설정 (수년 vs 수분) | Phase 2 |
| 상태 동기화 | `useTimelineState` + `useGanttChart` 분리 | 병합 후 DataSet 업데이트 전략 명확화 (전체 교체 vs 부분 업데이트) | Phase 2 |
| 대용량 렌더링 | `stack: true` | 500개+ 이벤트 시 Virtual Scrolling 또는 그룹 클러스터링 | Phase 3 |
| 접근성 | 미정의 | 토글/상태 버튼에 `aria-label`, `aria-expanded`, `aria-controls` + 키보드 포커스 흐름 | Phase 2 |

### 8.3 Participant 모델 하위 호환성 (Red Team 반영)

기존 `participants: list[str]`과 신규 `participants_detailed: list[Participant]`가 공존하므로, API 시리얼라이저에 **자동 승격(Promotion) 로직**을 구현한다:

```python
# 응답 시: participants가 str 배열이면 Participant(name=str, role="unknown")으로 승격
if isinstance(item.participants[0], str):
    item.participants_detailed = [
        Participant(name=p, role="unknown") for p in item.participants
    ]
```

---

## 9. API-Frontend Contract

**규칙**: Backend는 snake_case, Frontend는 camelCase. API 응답은 snake_case로 전달되며, 프론트엔드에서 camelCase로 매핑하여 사용한다 (기존 `types/index.ts` 패턴 준수).

| Backend (snake_case) | Frontend (camelCase) | 비고 |
|---------------------|---------------------|------|
| `date_start` | `dateStart` | 간트차트 시작 시점 |
| `date_end` | `dateEnd` | 간트차트 종료 시점 |
| `evidence_ids` | `evidenceIds` | TimelineItem 내 증거 참조 |
| `evidence_id` | `evidenceId` | EvidenceFile 고유 식별자 |
| `evidence_type` | `evidenceType` | 증거 유형 (kakao_talk 등) |
| `file_size_kb` | `fileSizeKb` | 파일 크기 |
| `session_id` | `sessionId` | 소유자 세션 (IDOR 방어) |
| `content_hash` | `contentHash` | SHA-256 파일 해시 |

> **참고**: 기존 `TimelineItem` 필드(`imageUrl`, `descriptionShort` 등)는 이미 camelCase이며 변경하지 않는다. 신규 필드도 동일 패턴을 따른다.

---

## Change History

| Version | Date | Changes |
|---------|------|---------|
| v0.1 | 2026-03-01 | 초안 작성 (Plan v0.2 기반 상세 설계) |
| v0.2 | 2026-03-01 | Red Team + External Consultant 설계 리뷰 통합: 보안 항목 6개 추가 (SEC-07~12), API-Frontend Contract 모순 해결, Testing Strategy 강화 (계약 테스트, 프롬프트 가드, CI 게이트), Architecture Improvement Notes 추가 (JobManager 영속성, 워커 분리, 동적 줌, 접근성, Participant 승격), EvidenceFile에 session_id 추가 |
