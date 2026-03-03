# 스토리보드 증거 파일 업로드 기능 조사 보고서

> **Summary**: 다중 증거 파일 업로드, 문자메시지 캡처본 파싱, 증거-타임라인 연결 구조, 증분 병합 알고리즘에 대한 기술 조사
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Author**: researcher-evidence (Claude Sonnet 4.6)
> **Date**: 2026-03-01
> **Status**: Research Complete

---

## 1. 현재 구현 상태

### 1.1 백엔드 (`backend/app/modules/storyboard/`)

| 엔드포인트 | 입력 | 처리 | 제한 |
|-----------|------|------|------|
| `POST /extract` | 텍스트 | GPT-4o-mini 타임라인 추출 (단일/2단계 장문) | 단일 텍스트만 |
| `POST /transcribe` | 음성 1개 | Whisper STT | 단일 파일만, 25MB 제한 |
| `POST /analyze-image` | 이미지 1개 | Gemini Vision 타임라인 추출 | 단일 파일만, 20MB 제한 |
| `POST /generate-images-batch` | 타임라인 항목 목록 | 이미지 배치 생성 (SSE 진행 상태) | 이미지 생성 전용 |

**핵심 갭:**
- 다중 파일 동시 업로드 엔드포인트 없음
- `TimelineItem.evidence_items: list[str]` — 문자열 목록만, 파일 객체 참조 구조 없음
- 증분 병합 API 없음 (새 증거 추가 시 기존 타임라인과 병합 불가)

### 1.2 프론트엔드 (`frontend/src/features/storyboard/`)

| 컴포넌트 | 현황 | 제한 |
|---------|------|------|
| `MultiInputPanel.tsx` | 탭 방식 (텍스트/음성/이미지) | 한 번에 한 개 파일만 처리 |
| `TimelineItem` 타입 | `evidenceItems?: string[]` | 파일 객체 참조 없음 |
| `storyboardService` | 각 모드별 단일 API 호출 | 병렬 처리 없음 |

### 1.3 재활용 가능한 기존 코드

| 파일 | 재활용 포인트 |
|-----|------------|
| `service/job_manager.py` | SSE 기반 진행 상태 관리 → 다중 파일 배치에 확장 |
| `service/__init__.py` | `_date_sort_key()`, `_parse_timeline_response()` → 병합 알고리즘 재활용 |
| `service/vision.py` | `analyze_image()` → 문자 캡처본 특화 프롬프트 파라미터 추가 |
| `service/doc_type_detector.py` | `KAKAOTALK_TXT` 타입 추가 |

---

## 2. 문자메시지 캡처본 파싱 접근법

### 2.1 카카오톡 .txt 내보내기 파싱 (1순위 권장)

카카오톡은 "대화 내보내기" 기능으로 `.txt` 파일 생성:

```
저장한 날짜 : 2024-01-20 15:30:00
---------------
2024년 1월 15일 화요일
---------------
[홍길동] [오후 2:30] 계약서 확인했어요
[이순신] [오후 2:31] 네, 서명하겠습니다
[홍길동] [오후 2:35] 내일 사무실로 오세요
```

**구현 방안:**
- `doc_type_detector.py`에 `KAKAOTALK_TXT` 타입 추가
- 정규식으로 `[발신자] [시간] 메시지` 패턴 파싱
- 날짜 구분선(`----`)으로 날짜 컨텍스트 추적
- 추출된 대화를 기존 `extract_timeline_from_text()` 파이프라인에 전달

**장점:** OCR 없이 100% 정확도, 발신자/시간 완벽 구분
**참고:** `kakaotalk_msg_preprocessor` 오픈소스 라이브러리

### 2.2 스크린샷 OCR + Vision LLM (2순위)

현재 `vision.py`의 Gemini Vision API로 대응 가능하나 특화 프롬프트 필요:

**말풍선 위치 기반 발신자 구분 프롬프트 추가 필요:**
```
이미지에서 채팅 대화를 추출하세요:
- 오른쪽 말풍선: 나(사용자) 발언
- 왼쪽 말풍선: 상대방 발언
- 타임스탬프는 말풍선 아래/옆 작은 텍스트
- 날짜 구분선은 가운데 회색 텍스트
```

**알려진 제약:**
- 타임스탬프 OCR 오인식률 높음 (작은 폰트, 흐린 색상)
- 이모지, 이미지 썸네일 포함 시 혼란 가능
- Fallback: GPT-4o 재시도 또는 사용자 수동 보정

**추천 구현 순서:** `.txt` 파싱 → 스크린샷 OCR → 통합 처리

---

## 3. 다중 파일 → 통합 타임라인 UX 패턴

### 3.1 업계 표준 (ChronoVault, OpenText eDiscovery, DISCO Timelines)

1. **파일별 개별 분석 → 자동 병합** 패턴이 표준
   - 각 파일 독립 분석 → 타임라인 항목 생성 → 날짜 기준 정렬/병합
   - 100+ 시간 → 10분 이내 단축이 업계 벤치마크

2. **파일별 진행 상태 표시**
   - 기존 `job_manager.py`의 SSE 구조를 2계층으로 확장:
     - `batch_job_id`: 전체 배치 진행률
     - `file_job_id`: 파일별 개별 진행률

3. **충돌 처리 전략**
   - 날짜±1일 범위 중복 → LLM 유사도 판단 → 자동/수동 병합

4. **GOV.UK MOJ Design System 멀티파일 패턴 (영국 법무부 기준)**
   - 드래그앤드롭 + 클릭 선택 병행
   - 업로드 완료 파일 목록 즉시 표시 + 제거 버튼
   - 파일 유형/크기 유효성 피드백 인라인

---

## 4. 증거 메타데이터 스키마 설계

### 4.1 백엔드 추가 스키마 (`schema/models.py` 확장)

```python
class EvidenceType(str, Enum):
    KAKAO_TXT = "kakao_txt"               # 카카오톡 .txt 내보내기
    KAKAO_SCREENSHOT = "kakao_screenshot"  # 카카오톡 스크린샷
    SMS_SCREENSHOT = "sms_screenshot"      # SMS/문자 스크린샷
    VOICE_RECORDING = "voice_recording"    # 음성 녹음
    DOCUMENT = "document"                  # 계약서, 진술서 등
    PHOTO = "photo"                        # 현장 사진
    OTHER = "other"


class EvidenceFile(BaseModel):
    evidence_id: str                          # UUID
    evidence_type: EvidenceType
    filename: str
    uploaded_at: str                          # ISO datetime
    file_size_kb: int
    extracted_timeline_ids: list[str]         # 이 증거에서 추출된 타임라인 항목 ID 목록
    tags: list[str]                           # 사용자 정의 태그
    source_description: str | None = None    # 증거 설명 (선택)


# TimelineItem 확장 (기존 필드 유지)
class TimelineItem(BaseModel):
    # ... 기존 필드 유지 ...
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="연결된 증거 파일 ID 목록 (N:M)"
    )
    # evidence_items: list[str] 하위 호환용 유지
```

### 4.2 프론트엔드 타입 확장 (`types/index.ts`)

```typescript
export type EvidenceType =
  | 'kakao_txt'
  | 'kakao_screenshot'
  | 'sms_screenshot'
  | 'voice_recording'
  | 'document'
  | 'photo'
  | 'other'

export interface EvidenceFile {
  evidence_id: string
  evidence_type: EvidenceType
  filename: string
  uploaded_at: string
  file_size_kb: number
  extracted_timeline_ids: string[]
  tags: string[]
  source_description?: string
}

// TimelineItem 확장
export interface TimelineItem {
  // ... 기존 필드 유지 ...
  evidenceIds?: string[]  // 연결된 증거 파일 ID 목록
}
```

### 4.3 N:M 관계 구조

```
EvidenceFile (1) ──── extracted_timeline_ids ────► TimelineItem (N)
TimelineItem (1) ──── evidence_ids ──────────────► EvidenceFile (N)
```

---

## 5. 증분 타임라인 병합 알고리즘

### 5.1 3단계 병합 알고리즘

```
1단계: 날짜 범위 기반 후보군 추출
   입력: 신규 타임라인 항목 목록 + 기존 타임라인
   처리: 신규 항목의 date ± 3일 범위에 있는 기존 항목 = 후보군
   특수: date가 "날짜 미상"이면 title 키워드 유사도로만 비교

2단계: 내용 유사도 판단 (LLM 활용)
   후보군이 있으면 GPT-4o-mini에 판단 요청:
   - SAME   → 동일 사건, 병합 실행
   - RELATED → 관련 사건, 상호 참조만 추가
   - DIFFERENT → 독립 사건, 신규 항목으로 추가

3단계: 병합 실행
   SAME:
     - 기존 항목의 evidence_ids에 신규 증거 ID 추가
     - description_detailed는 더 긴 것 우선
     - participants_detailed는 합집합 (name 기준 중복 제거)
   RELATED:
     - 각 항목 독립 유지
     - evidence_items에 상호 참조 텍스트 추가
   DIFFERENT:
     - 신규 항목 추가
     - 전체 타임라인 _date_sort_key() 재정렬
```

### 5.2 비용 최적화

| 신규 항목 수 | 처리 방식 |
|------------|---------|
| 1-5개 | 전체 LLM 판단 |
| 6-20개 | 날짜 범위 매칭 → 충돌 후보만 LLM 처리 |
| 21개 이상 | 날짜 범위 매칭 → 상위 5개 충돌만 LLM 처리 + 나머지 자동 추가 |

### 5.3 신규 API 엔드포인트 제안

```
POST /api/storyboard/merge
Request:
  existing_timeline: list[TimelineItem]  # 기존 타임라인
  new_items: list[TimelineItem]          # 신규 추출 항목
  evidence_id: str                        # 신규 항목의 증거 파일 ID

Response:
  merged_timeline: list[TimelineItem]    # 병합된 전체 타임라인
  merge_report: MergeReport              # 병합 결과 요약
    same_count: int      # 병합된 항목 수
    related_count: int   # 연결된 항목 수
    new_count: int       # 신규 추가 항목 수
    conflicts: list[ConflictItem]  # 사용자 검토 필요 항목
```

---

## 6. 구현 우선순위 로드맵

| 우선순위 | 기능 | 구현 복잡도 | 예상 효과 | 재활용 가능 코드 |
|---------|------|------------|---------|--------------|
| P1 | 카카오톡 .txt 파싱 (`doc_type_detector.py` 확장) | 낮음 | 높음 | `doc_type_detector.py`, `extract_timeline_from_text()` |
| P1 | `EvidenceFile` 스키마 정의 | 낮음 | 높음 | 신규 |
| P2 | 다중 이미지 일괄 업로드 엔드포인트 | 중간 | 높음 | `job_manager.py` SSE |
| P2 | 문자 캡처본 특화 Vision 프롬프트 | 낮음 | 중간 | `vision.py` |
| P3 | 파일별 진행 상태 UI (SSE 2계층) | 중간 | 중간 | `job_manager.py` |
| P3 | 증분 병합 API (`POST /merge`) | 높음 | 높음 | `_date_sort_key()`, `_parse_timeline_response()` |
| P4 | 증거 태깅 UI 컴포넌트 | 낮음 | 중간 | 신규 |
| P4 | 타임라인-증거 연결 뷰 | 중간 | 중간 | `TimelineCard.tsx` |

---

## 7. 관련 외부 레퍼런스

| 레퍼런스 | 적용 포인트 |
|---------|-----------|
| [ChronoVault (NexLaw)](https://www.nexlaw.ai/products/chronovault/) | 증거-타임라인 연결 UX, 자동 병합 패턴 |
| [OpenText eDiscovery Chronology](https://businessprocessincubator.com/content/opentexttm-ediscovery-ce-25-2-introducing-ediscovery-chronology-for-enhanced-evidence-organization) | 이벤트 메타데이터 구조, 감사 추적 |
| [DISCO Timelines](https://csdisco.com/blog/fact-based-legal-timelines) | 사실 기반 타임라인 구성, 문서-사실 연결 |
| [MOJ Design System 멀티파일 업로드](https://design-patterns.service.justice.gov.uk/components/multi-file-upload/) | 법무 서비스용 멀티파일 UI 패턴 |
| [kakaotalk_msg_preprocessor](https://github.com/uoneway/kakaotalk_msg_preprocessor) | 카카오톡 .txt 파싱 참조 구현 |
| [Google Cloud Vision DOCUMENT_TEXT_DETECTION](https://docs.cloud.google.com/vision/docs/ocr) | 밀집 텍스트 OCR (스크린샷 타임스탬프) |
