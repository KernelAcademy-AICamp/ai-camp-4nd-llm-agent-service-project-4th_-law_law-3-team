# Storyboard Gantt Chart (사건 타임라인 간트차트) Planning Document

> **Summary**: 법률 사건의 타임라인을 간트차트 형식으로 시각화 — 다중 증거 파일 업로드, 주제별 기간 막대, 증거 마커를 통해 사용자가 본인 사건의 흐름을 직관적으로 확인·이해할 수 있는 기능
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Author**: Claude (Team Lead)
> **Date**: 2026-03-01
> **Status**: Draft (v0.2 — Red Team + External Consultant 피드백 통합)

---

## 1. Overview

### 1.1 Purpose

사용자가 본인의 법률 사건 관련 증거(음성 녹음, 문자메시지 캡처본, 사진, 문서 등)를 업로드하면, AI가 자동으로 타임라인을 추출하고 **간트차트 형식**으로 시각화하여 사건의 흐름을 한눈에 파악할 수 있도록 한다.

### 1.2 Background

**현재 구현 상태:**
- 텍스트/음성/이미지 입력 → AI 타임라인 추출 (GPT-4o-mini, Whisper, Gemini Vision)
- 수직 카드 리스트 형태의 타임라인 뷰 (`TimelineView.tsx`)
- 스토리보드 이미지 생성 (Gemini 2.0 Flash) + 영상 생성 (moviepy)
- 단일 파일만 처리 가능, 다중 파일 업로드 미지원
- 증거-타임라인 항목 간 구조적 연결 없음 (`evidence_items: list[str]`은 텍스트 목록만)

**핵심 문제:**
현재 시각화는 "영화 스토리보드" 방향으로 설계되어 있으나, 실제 사용자 니즈는 **"본인 사건의 타임라인을 증거 기반으로 확인·이해"** 하는 것이다. 이미지 생성보다 **간트차트 형식의 데이터 시각화**가 우선이다.

### 1.3 비전

```
┌──────────────────────────────────────────────────────────────────┐
│  📂 증거 업로드 영역                                              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                           │
│  │ 📱   │ │ 🎤   │ │ 📄   │ │ 📷   │  + 파일 추가              │
│  │카카오 │ │녹음  │ │계약서│ │사진  │                            │
│  │톡.txt│ │파일  │ │.pdf  │ │.jpg  │                            │
│  └──────┘ └──────┘ └──────┘ └──────┘                           │
│                                                                  │
│  📊 간트차트 타임라인                                             │
│  시간축 →     1월      2월      3월      4월      5월            │
│  ────────────────────────────────────────────────────            │
│  폭행        ████                                                │
│  협박               ██████████                                   │
│  금전 갈취          ███████                                      │
│  스토킹                    ████████████████                      │
│  ────────────────────────────────────────────────────            │
│  증거 마커                                                       │
│  📱 문자     ▲  ▲        ▲              ▲                       │
│  🎤 녹음              ▲                                          │
│  📷 사진     ▲                    ▲                              │
│  📄 문서                                        ▲                │
│  ────────────────────────────────────────────────────            │
│                                                                  │
│  📋 상세 패널 (클릭 시)                                           │
│  ┌──────────────────────────────────────────────────┐           │
│  │ 2024.02.15 - 폭행 사건                            │           │
│  │ 장소: 사무실 / 가해자: B과장                       │           │
│  │ 연결된 증거: 카카오톡.txt (라인 23-45), 사진.jpg   │           │
│  │ 법적 의미: 형법 제260조 폭행죄 구성요건 해당       │           │
│  └──────────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

### 1.4 핵심 사용자 시나리오

**시나리오 1: 직장 내 괴롭힘 사건**
```
1. 사용자가 /storyboard 페이지 접속
2. 카카오톡 대화 내보내기 .txt 파일 업로드
3. 음성 녹음 파일 2개 추가 업로드
4. 문자메시지 캡처본(스크린샷) 3장 추가
5. AI가 각 증거를 분석하여 자동 타임라인 생성
6. 간트차트에 "폭언", "업무 배제", "인사 불이익" 등 주제별 기간 막대 표시
7. 각 증거가 해당 시점에 마커로 표시
8. 사용자가 마커 클릭 → 원본 증거 확인 가능
9. 새 증거 추가 시 기존 타임라인에 자동 병합
10. JSON 내보내기로 변호사 상담 시 활용
```

**시나리오 2: 사건 순서가 뒤섞인 입력**
```
1. 사용자가 사건 내용을 시간순이 아닌 순서로 텍스트 입력
2. 추가로 과거 날짜의 증거 파일 업로드
3. AI가 모든 입력을 분석하여 시간순 자동 정렬
4. 간트차트에서 전체 사건 흐름을 시간축 기준으로 파악
5. 빠진 기간이나 증거 공백을 시각적으로 인지 가능
```

### 1.5 Related Documents

| 문서 | 설명 |
|------|------|
| `backend/app/modules/storyboard/` | 현재 스토리보드 백엔드 모듈 |
| `frontend/src/features/storyboard/` | 현재 스토리보드 프론트엔드 |
| `docs/03-analysis/storyboard-gantt-chart.redteam.md` | Red Team 검증 보고서 (Gemini CLI) |
| `docs/03-analysis/storyboard-gantt-chart.consulting.md` | 외부 컨설팅 보고서 (Codex CLI) |

---

## 2. Requirements Analysis (요구사항 분석)

### 2.1 기능 요구사항

| ID | 요구사항 | 우선순위 | 설명 |
|----|---------|---------|------|
| FR-01 | 다중 파일 업로드 | **Critical** | 음성, 이미지, 문서, 텍스트 파일을 한 번에 여러 개 업로드 |
| FR-02 | 카카오톡 .txt 파싱 | **Critical** | 카카오톡 대화 내보내기 파일에서 발신자/시간/내용 자동 추출 |
| FR-03 | 문자 캡처본 Vision 분석 | **High** | 메신저 스크린샷에서 대화 내용, 발신자, 시간 추출 |
| FR-04 | 간트차트 시각화 | **Critical** | X축(시간) + Y축(주제별 행) + 기간 막대 + 증거 마커 |
| FR-05 | 증거-타임라인 연결 | **Critical** | 증거 파일과 타임라인 항목 간 N:M 매핑 |
| FR-06 | 증거 마커 표시 | **High** | 간트차트 위에 증거 유형별(문자/녹음/사진/문서) 마커 표시 |
| FR-07 | 증분 타임라인 병합 | **High** | 새 증거 추가 시 기존 타임라인과 자동 병합 |
| FR-08 | 사건 주제 자동 분류 | **High** | AI가 타임라인 항목을 주제별(폭행/협박/금전 등)로 자동 그룹핑 |
| FR-09 | 상세 패널 | **Medium** | 간트차트 항목 클릭 시 상세 정보 + 연결된 증거 표시 |
| FR-10 | 줌/패닝 인터랙션 | **Medium** | 시간축 줌인/줌아웃, 좌우 패닝 |
| FR-11 | 뷰 전환 | **Medium** | 간트차트 뷰 ↔ 기존 카드 리스트 뷰 토글 |
| FR-12 | JSON 내보내기/가져오기 | **Low** | 기존 기능 유지 + 증거 메타데이터 포함 |
| FR-13 | 근거 신뢰도 배지 | **Medium** | 이벤트별 근거 개수·신뢰도 시각 표시 (Consultant 제안) |
| FR-14 | 타임라인-원문 양방향 탐색 | **Medium** | 이벤트 클릭 → 원문 하이라이트 / 원문 선택 → 이벤트 후보 생성 (Consultant 제안) |
| FR-15 | 충돌/누락 인사이트 | **Low** | 동일 시간대 상반 내용 경고 마커, 증거 공백 구간 알림 (Red Team + Consultant 공통 제안) |

### 2.2 보안 요구사항 (Red Team 피드백 반영)

| ID | 요구사항 | 우선순위 | 설명 |
|----|---------|---------|------|
| SEC-01 | 프롬프트 인젝션 방어 | **Critical** | 증거 파일 내 악의적 텍스트가 LLM 추출 로직을 오작동시키지 않도록 System Role 분리 + Rule-based 필터링 |
| SEC-02 | 파일 업로드 검증 | **Critical** | 매직 넘버(Magic Number) 체크, 확장자 검증, 실행 권한 제거, 격리 저장 경로 |
| SEC-03 | 증거 접근 제어 | **High** | evidence_id는 UUID v4, 모든 요청에 세션 기반 소유권 검증 (IDOR 방지) |
| SEC-04 | DoS 방지 | **High** | 파일당 크기 제한 + 동시 업로드 수 제한 + 처리 타임아웃 + 사용자별 일일 할당량 |
| SEC-05 | EXIF 메타데이터 제거 | **Medium** | 이미지 업로드 시 GPS/기기정보 등 불필요한 메타데이터 자동 스트리핑 |
| SEC-06 | 파일 해시 기반 멱등성 | **Medium** | SHA-256 해시로 동일 파일 중복 업로드 감지 → 중복 타임라인 생성 방지 |

### 2.3 비기능 요구사항

| ID | 요구사항 | 기준 |
|----|---------|------|
| NFR-01 | 파일 업로드 크기 | 음성 25MB, 이미지 20MB, 문서 10MB (기존 제한 유지) |
| NFR-02 | 동시 파일 처리 | 최대 10개 파일 병렬 분석 |
| NFR-03 | 타임라인 항목 수 | 100개 이상에서도 간트차트 렌더링 성능 유지 |
| NFR-04 | 반응형 | 모바일은 조회 전용 경량화, 편집은 데스크톱 중심 |
| NFR-05 | 접근성 | 키보드 탐색, 스크린리더 지원 |
| NFR-06 | LLM 컨텍스트 관리 | 대용량 카카오톡 대화는 슬라이딩 윈도우(월/주 단위) 분할 분석 후 병합 |

---

## 3. Gap Analysis (현재 vs 목표)

### 3.1 시각화 방식

| 항목 | 현재 | 목표 | Gap 심각도 |
|------|------|------|-----------|
| 타임라인 뷰 | 수직 카드 리스트 | 간트차트 (시간축 + 주제별 행) | **Critical** |
| 시간축 | 날짜별 그룹 헤더 | 연속적 X축 (줌/패닝) | **Critical** |
| 주제별 행 | 없음 | Y축에 사건 카테고리별 행 | **Critical** |
| 기간 표현 | 점(이벤트)만 | 막대(기간) + 점(이벤트) 혼합 | **High** |
| 증거 마커 | 없음 | 유형별 아이콘 마커 | **High** |
| 인터랙션 | 클릭 → 카드 선택 | 줌/패닝/호버 툴팁/클릭 상세 | **Medium** |

### 3.2 입력 방식

| 항목 | 현재 | 목표 | Gap 심각도 |
|------|------|------|-----------|
| 파일 업로드 | 단일 파일 | 다중 파일 동시 업로드 | **High** |
| 카카오톡 .txt | 미지원 | 자동 파싱 (발신자/시간/내용) | **Critical** |
| 문자 캡처본 | 일반 이미지 분석 | 메신저 특화 Vision 프롬프트 | **Medium** |
| 증분 추가 | 전체 교체 | 기존 타임라인에 병합 | **High** |

### 3.3 데이터 구조

| 항목 | 현재 | 목표 | Gap 심각도 |
|------|------|------|-----------|
| 증거 연결 | `evidence_items: list[str]` (텍스트) | `evidence_ids: list[str]` (파일 참조 N:M) | **Critical** |
| 증거 메타데이터 | 없음 | `EvidenceFile` 스키마 (유형/태그/파일정보) | **High** |
| 사건 주제 분류 | 없음 | `topic` 필드 (AI 자동 분류) | **High** |
| 사건 기간 | `date` (단일) | `date_start` + `date_end` (기간) | **High** |

---

## 4. Technical Stack (기술 스택 선정)

### 4.1 간트차트 라이브러리 — vis-timeline (1순위 추천)

| 평가 항목 | vis-timeline | D3.js 직접 구현 |
|----------|-------------|----------------|
| Weekly Downloads | 151,500 (압도적 1위) | N/A |
| Y축 그룹 (주제별 행) | 내장 지원 | 직접 구현 |
| 기간 막대 + 포인트 마커 | 내장 (range + point 아이템) | 직접 구현 |
| 줌/패닝 | 내장 | 직접 구현 |
| 호버 툴팁 | 내장 | 직접 구현 |
| 초기 개발 공수 | 1~2일 | 1~2주 |
| 커스터마이징 | 중간 (CSS + template) | 무제한 |
| 번들 크기 | ~500KB (gzip) | 최소 |
| SSR | 불가 → `next/dynamic` | 가능 |
| 라이선스 | MIT / Apache-2.0 | MIT |

**결정: vis-timeline + `@datou/react-vis-timeline` React wrapper**

**이유:**
1. 요구사항의 핵심(Y축 그룹 + 기간 bar + 포인트 마커)이 vis-timeline 데이터 모델에 정확히 대응
2. 줌/패닝/클릭/호버 이벤트 모두 내장되어 개발 공수 대폭 절감
3. 주간 15만 다운로드로 충분히 검증된 라이브러리
4. SSR 제약은 `next/dynamic({ ssr: false })`로 해결 (기존 Phaser.js도 동일 패턴 사용 중)

**대안(D3.js 직접 구현)은 MVP 이후 법률 특화 UX가 필요할 때 고려.**

### 4.2 파일 처리

| 파일 유형 | 처리 방식 | 기존 코드 재활용 |
|----------|----------|----------------|
| 카카오톡 .txt | 정규식 파싱 (신규 doc_type) | `doc_type_detector.py` 확장 |
| 음성 녹음 | Whisper STT → 텍스트 추출 | `stt.py` 재활용 |
| 메신저 스크린샷 | Gemini Vision (특화 프롬프트) | `vision.py` 확장 |
| 일반 이미지/사진 | Gemini Vision | `vision.py` 재활용 |
| 문서 (PDF 등) | 텍스트 추출 → 타임라인 추출 | 신규 구현 필요 |

### 4.3 다중 파일 처리 아키텍처

```
사용자: 파일 N개 업로드
    ↓
프론트엔드: POST /api/storyboard/analyze-batch
    ↓
백엔드: 파일 검증 게이트
    ├─ 매직 넘버 검증 (확장자 위조 탐지)
    ├─ 파일 크기/개수 제한 확인
    ├─ SHA-256 해시 → 중복 업로드 감지
    └─ EXIF 메타데이터 스트리핑 (이미지)
    ↓
백엔드: BatchAnalyzeJob 생성
    ├─ 파일별 유형 감지 (EvidenceType)
    ├─ 파일별 개별 분석 (asyncio.gather, Semaphore=3)
    │   ├─ .txt → 카카오톡 파서 or 텍스트 추출
    │   ├─ 음성 → Whisper STT → 텍스트 추출
    │   ├─ 이미지 → Gemini Vision → 타임라인
    │   └─ 문서 → 텍스트 추출 → 타임라인
    ├─ LLM 입력 시 System/User Role 엄격 분리 (인젝션 방어)
    ├─ 결과 병합 (날짜 정렬 + 중복 감지)
    └─ EvidenceFile 메타데이터 + 타임라인 항목 반환
    ↓
프론트엔드: SSE로 파일별 진행 상태 수신
    ↓
간트차트 렌더링
```

---

## 5. Data Model Changes (데이터 모델 변경)

### 5.1 Backend 스키마 추가/변경

```python
# 신규: 증거 유형 Enum
class EvidenceType(str, Enum):
    KAKAO_TXT = "kakao_txt"              # 카카오톡 .txt 내보내기
    MESSENGER_SCREENSHOT = "messenger_screenshot"  # 메신저 스크린샷
    VOICE_RECORDING = "voice_recording"  # 음성 녹음
    DOCUMENT = "document"                # 문서 (PDF, 계약서 등)
    PHOTO = "photo"                      # 사진
    TEXT_INPUT = "text_input"            # 직접 텍스트 입력
    OTHER = "other"

# 신규: 증거 파일 모델
class EvidenceFile(BaseModel):
    evidence_id: str                     # UUID
    evidence_type: EvidenceType
    filename: str
    uploaded_at: str                     # ISO datetime
    file_size_kb: int
    extracted_timeline_ids: list[str]    # 추출된 타임라인 항목 ID 목록
    tags: list[str]                      # 사용자 정의 태그
    source_description: str | None       # 증거 설명

# 변경: TimelineItem 확장
class TimelineItem(BaseModel):
    # 기존 필드 유지...

    # 신규 필드
    topic: str | None                    # 사건 주제 (AI 자동 분류: 폭행, 협박, 금전 등)
    date_start: str | None              # 기간 시작일 (간트차트 막대 시작)
    date_end: str | None                # 기간 종료일 (간트차트 막대 끝)
    evidence_ids: list[str] = []        # 연결된 증거 파일 ID (N:M)
    # evidence_items: list[str] 하위 호환 유지

# 신규: 타임라인 전체 데이터 확장
class TimelineData(BaseModel):
    # 기존 필드 유지...
    evidence_files: list[EvidenceFile] = []  # 증거 파일 목록
```

### 5.2 간트차트 데이터 매핑

```typescript
// vis-timeline 데이터 모델 매핑
// Y축 그룹 = 사건 주제 (topic)
const groups = [
  { id: 'assault', content: '폭행' },
  { id: 'threat', content: '협박' },
  { id: 'extortion', content: '금전 갈취' },
]

// 기간 막대 = TimelineItem (date_start ~ date_end)
const items = [
  { id: 'item-1', group: 'assault', start: '2024-01-15', end: '2024-02-20',
    content: '직장 내 폭행', type: 'range' },
]

// 증거 마커 = EvidenceFile (날짜 포인트)
const markers = [
  { id: 'ev-1', group: 'evidence-kakao', start: '2024-01-20',
    content: '📱', type: 'point', className: 'evidence-marker' },
]
```

---

## 6. Implementation Scope (구현 범위)

### 6.1 Phase 1: 간트차트 기본 뷰 (MVP)

**목표:** 기존 타임라인 데이터를 간트차트로 시각화

| 작업 | 파일 | 설명 |
|------|------|------|
| vis-timeline 설치 | `frontend/package.json` | `vis-timeline`, `@datou/react-vis-timeline` |
| 간트차트 컴포넌트 | `frontend/src/features/storyboard/components/GanttChartView.tsx` | 신규 |
| 뷰 전환 토글 | `TimelineToolbar.tsx` 수정 | 카드뷰 ↔ 간트차트 토글 |
| 주제 자동 분류 | `backend/.../service/__init__.py` 확장 | 추출 프롬프트에 `topic` 필드 추가 |
| 기간 지원 | `schema/models.py` 확장 | `date_start`, `date_end`, `topic` 필드 |
| 상세 패널 | `GanttDetailPanel.tsx` | 항목 클릭 시 사이드 패널 |

### 6.2 Phase 2: 다중 증거 업로드 + 연결

**목표:** 여러 증거 파일 업로드 → 자동 타임라인 추출 → 증거 연결

| 작업 | 파일 | 설명 |
|------|------|------|
| 증거 스키마 | `schema/models.py` | `EvidenceFile`, `EvidenceType` 추가 |
| 다중 파일 업로드 UI | `EvidenceUploadPanel.tsx` | 드래그앤드롭 + 파일 목록 |
| 배치 분석 API | `router/__init__.py` | `POST /analyze-batch` 엔드포인트 |
| 카카오톡 파서 | `service/kakao_parser.py` | .txt 정규식 파싱 |
| 증거 마커 렌더링 | `GanttChartView.tsx` 확장 | 유형별 아이콘 마커 |
| SSE 진행 상태 | `job_manager.py` 확장 | 파일별 sub_job 진행률 |

### 6.3 Phase 3: 증분 병합 + 고급 기능

**목표:** 새 증거 추가 시 기존 타임라인과 병합, 메신저 스크린샷 특화 분석

| 작업 | 파일 | 설명 |
|------|------|------|
| 증분 병합 API | `POST /merge` | 날짜 범위 매칭 + LLM 중복 판단 |
| 메신저 Vision 프롬프트 | `vision.py` 확장 | 발신/수신 구분, 시간 추출 특화 |
| 증거 태깅 UI | `EvidenceTagEditor.tsx` | 유형/날짜/관련 사건 태깅 |
| 증거 공백 시각화 | `GanttChartView.tsx` | 증거 없는 기간 하이라이트 |
| 테스트 데이터셋 | `evaluation/datasets/` | 가상 데이터 + 순서 뒤섞기 테스트 |

---

## 7. API Changes (API 변경 사항)

### 7.1 신규 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/storyboard/analyze-batch` | 다중 파일 일괄 분석 → 통합 타임라인 |
| POST | `/api/storyboard/merge` | 기존 타임라인에 새 항목 병합 |
| GET | `/api/storyboard/evidence/{id}` | 증거 파일 원본 조회 |

### 7.2 변경 엔드포인트

| 메서드 | 경로 | 변경 사항 |
|--------|------|----------|
| POST | `/api/storyboard/extract` | 응답에 `topic`, `date_start`, `date_end` 필드 추가 |
| POST | `/api/storyboard/analyze-image` | 메신저 특화 프롬프트 옵션 추가 |

### 7.3 프론트엔드 4곳 동기화 확인

| 위치 | 확인 사항 |
|------|----------|
| `frontend/src/lib/modules.ts` | `storyboard` 모듈 `enabled: true` 유지 |
| `frontend/src/lib/api.ts` | `storyboard` endpoints 변경 없음 (모듈 내부 서비스에서 직접 호출) |
| `frontend/next.config.js` | `/api/storyboard/**` 프록시 규칙 유지 |
| `backend/app/modules/storyboard/router/__init__.py` | 신규 엔드포인트 추가 |

---

## 8. Milestones (마일스톤)

| Phase | 목표 | 주요 산출물 | 의존성 |
|-------|------|-----------|--------|
| **Phase 1** | 간트차트 기본 뷰 | `GanttChartView.tsx`, `topic`/`date_start`/`date_end` 필드 | 없음 |
| **Phase 2** | 다중 증거 업로드 | `EvidenceUploadPanel.tsx`, `/analyze-batch`, `kakao_parser.py` | Phase 1 |
| **Phase 3** | 증분 병합 + 고급 | `/merge`, 메신저 Vision 특화, 테스트 데이터셋 | Phase 2 |

---

## 9. Risk Analysis (리스크 분석)

| 리스크 | 확률 | 영향 | 완화 방안 |
|--------|------|------|----------|
| vis-timeline SSR 미지원 | 확실 | 낮음 | `next/dynamic({ ssr: false })` — 기존 Phaser.js와 동일 패턴 |
| vis-timeline 번들 크기 (~500KB) | 확실 | 중간 | 코드 스플리팅 + lazy loading으로 초기 로드 영향 최소화 |
| 카카오톡 .txt 형식 변경 | 낮음 | 높음 | 정규식 패턴을 설정 파일로 분리하여 업데이트 용이하게 |
| 메신저 스크린샷 OCR 정확도 | 중간 | 중간 | Gemini Vision + 재시도 fallback, 사용자 수동 수정 UI 제공 |
| LLM 기반 중복 감지 비용 | 중간 | 낮음 | 날짜 범위 1차 필터 → LLM은 후보군만 처리 (5개 이하) |
| 사건 주제 자동 분류 정확도 | 중간 | 중간 | AI 분류 후 사용자가 드래그로 주제 변경 가능하게 |
| vis-timeline React wrapper 유지보수 | 낮음 | 중간 | 필요시 useRef + useEffect로 직접 통합 가능 (vanilla API 직접 사용) |
| **악성 파일 업로드 (RCE)** | 낮음 | **치명** | 매직 넘버 검증 + 격리 저장 경로 + 실행 권한 제거 (Red Team) |
| **LLM 프롬프트 인젝션** | 중간 | **높음** | System/User Role 엄격 분리 + Rule-based 결과 필터링 (Red Team) |
| **대용량 파일 DoS** | 중간 | 높음 | 파일 크기/개수 제한 + 처리 타임아웃 + 사용자별 Quota (Red Team) |
| **AI API 비용 폭증** | 중간 | 높음 | 파일당/사용자당 토큰 비용 추적 + 비정상 폭증 시 서킷 브레이커 (Red Team) |
| **LLM 컨텍스트 한도 초과** | 높음 | 중간 | 슬라이딩 윈도우(월/주 단위) 분할 분석 + 요약본 병합 전략 (Red Team) |

---

## 10. Out of Scope (범위 외)

- 스토리보드 이미지 생성 기능 (기존 유지, 간트차트와 독립)
- 영상 생성 기능 (기존 유지)
- 실시간 협업 (다중 사용자 동시 편집) — Consultant 제안이나 현 단계에서는 범위 외
- 서버 영구 저장 (현재 세션 기반 유지, DB 저장은 추후)
- PDF 보고서 자동 생성
- PII 자동 비식별화 — Red Team/Consultant 공통 제안이나 별도 프로젝트로 분리
- 비동기 워커 (Celery/RabbitMQ) — Red Team 제안이나 현 규모에서는 asyncio.gather로 충분, 트래픽 증가 시 도입
- 증거 무결성 검증 (Hash Chaining) — Red Team 제안, 법적 효력 요건이 확정된 후 도입
- AI 거버넌스 프레임워크 (감사로그, HITL 승인체계) — Consultant 제안, 플랫폼 전체 과제로 분리

---

## 11. Success Metrics (성공 지표)

| 지표 | 목표 | 비고 |
|------|------|------|
| 간트차트 렌더링 성능 | 100개 항목에서 60fps 유지 | |
| 카카오톡 .txt 파싱 정확도 | 발신자/시간/내용 분리 95% 이상 | |
| 증분 병합 정확도 | 중복 감지 정확도 90% 이상 | |
| 사건 주제 분류 정확도 | 80% 이상 (사용자 수정 허용) | |
| 다중 파일 처리 시간 | 5개 파일 기준 30초 이내 | |
| 날짜 정규화 정확도 | 95% 이상 | Consultant 제안 KPI |
| 증거-이벤트 링크 정확도 | 90% 이상 | Consultant 제안 KPI |
| 악성 파일 차단율 | 위조 확장자 100% 탐지 | Red Team 보안 KPI |

---

## Appendix A: 기존 코드 재활용 매핑

| 기존 코드 | 재활용 방식 |
|----------|-----------|
| `vision.py` → `analyze_image()` | 메신저 특화 프롬프트 파라미터 추가 |
| `stt.py` → `transcribe_audio()` | 그대로 재활용 |
| `job_manager.py` → SSE 구조 | 다중 파일 배치 처리에 확장 |
| `service/__init__.py` → `_date_sort_key()` | 병합 알고리즘에 재활용 |
| `doc_type_detector.py` | `KAKAOTALK_TXT` 타입 추가 |
| `TimelineView.tsx` | 기존 카드 뷰로 유지 (토글 전환) |
| `useTimelineState.ts` | 증거 상태 관리 훅 확장 |
| `image_generation.py` | 기존 유지 (간트차트와 독립) |

---

---

## 12. 외부 검증 요약 (3중 검증 결과)

### 12.1 Red Team (Gemini CLI) — 보안/아키텍처 관점

**채택 항목:**
- SEC-01~06: 보안 요구사항으로 반영 (프롬프트 인젝션, 파일 검증, IDOR, DoS, EXIF, 멱등성)
- 파일 처리 아키텍처에 보안 게이트 추가
- LLM Context Chunking → NFR-06으로 반영
- AI 비용 모니터링 → 리스크 분석에 반영

**보류/범위 외 판정:**
- 비동기 워커 (Celery/RabbitMQ): 현 규모에서 asyncio로 충분, 추후 도입
- 파일 처리 샌드박스: 별도 컨테이너 격리는 인프라 과제로 분리
- 증거 무결성 Hash Chaining: 법적 효력 요건 확정 후
- PII 비식별화: 플랫폼 전체 과제

### 12.2 External Consultant (Codex CLI) — UX/ML/비즈니스 관점

**채택 항목:**
- FR-13 근거 신뢰도 배지, FR-14 양방향 탐색, FR-15 충돌/누락 인사이트
- 날짜 정규화 정확도 / 증거-이벤트 링크 정확도 KPI 추가
- 모바일 조회 전용 경량화 → NFR-04 수정 반영

**보류/범위 외 판정:**
- 협업 UX (코멘트, 버전 비교, 승인 워크플로우): 별도 프로젝트
- AI 거버넌스 프레임워크: 플랫폼 전체 과제
- Continuous Evals / Trace Grading: 기존 RAG 평가 인프라 활용, 별도 계획
- 비즈니스 전략 (과금 모델 등): 사업 기획 영역, 기술 기획서 범위 외

---

## Change History

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v0.1 | 2026-03-01 | 초안 작성 (Agent Team + 연구 에이전트 조사 결과 통합) |
| v0.2 | 2026-03-01 | Red Team (Gemini) + External Consultant (Codex) 피드백 통합: 보안 요구사항 6건, UX 요구사항 3건, 성공 지표 3건 추가, 리스크 5건 추가, Out of Scope 명확화 |
