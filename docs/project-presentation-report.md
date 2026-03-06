# 법률 서비스 플랫폼 — 프로젝트 최종 발표 보고서

> **프로젝트명**: AI 기반 법률 서비스 통합 플랫폼
> **기술 스택**: Next.js 14 + FastAPI + LangGraph + PostgreSQL 17 + LanceDB
> **작성일**: 2026-03-04

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [서비스 기능 소개 (12개 모듈)](#2-서비스-기능-소개)
3. [기술 아키텍처](#3-기술-아키텍처)
4. [AI/ML 시스템](#4-aiml-시스템)
5. [데이터 인프라](#5-데이터-인프라)
6. [핵심 설계 결정 및 차별점](#6-핵심-설계-결정-및-차별점)
7. [프로젝트 규모 요약](#7-프로젝트-규모-요약)

---

## 1. 프로젝트 개요

### 1-1. 프로젝트 목표

**"법률 정보의 민주화"** — 일반인과 변호사 모두에게 AI 기반의 법률 서비스를 제공하여, 법률 정보 접근 장벽을 낮추고 법률 전문가의 업무 생산성을 향상시키는 통합 플랫폼.

### 1-2. 핵심 가치

| 가치 | 설명 |
|------|------|
| **접근성** | 법률 용어를 몰라도 자연어로 판례·법령 검색 가능 |
| **정확성** | 58만+ 법률 문서 기반 RAG로 근거 있는 답변 제공 |
| **실용성** | 소액소송 서류 자동 생성, 사건 타임라인 시각화 등 즉시 활용 |
| **생산성** | 변호사 대상 콘텐츠 마케팅·사건 워크스페이스 제공 |

### 1-3. 역할 기반 서비스 구조

```
┌──────────────────────────────────────────────────┐
│                  통합 플랫폼 홈                   │
│            (역할 선택: 일반인 / 변호사)            │
├─────────────────────┬────────────────────────────┤
│    일반인 (User)     │      변호사 (Lawyer)        │
├─────────────────────┼────────────────────────────┤
│ · 주변 변호사 찾기   │ · 변호사 통계 대시보드      │
│ · 소액소송 도우미    │ · 콘텐츠 마케팅             │
│                     │ · 로스쿨 학습               │
│                     │ · 법령 체계도               │
│                     │ · 사건 워크스페이스          │
├─────────────────────┴────────────────────────────┤
│              공통 서비스                          │
│  · 판례 검색  · 법령 검색  · 스토리보드           │
│  · 모의 법정  · 법률 뉴스  · 통합 AI 채팅        │
└──────────────────────────────────────────────────┘
```

---

## 2. 서비스 기능 소개

### 2-1. 주변 변호사 찾기

> GPS 위치 기반으로 주변 변호사를 지도에서 검색하고, 전문분야·지역·이름으로 상세 필터링

| 항목 | 내용 |
|------|------|
| **대상** | 일반인 |
| **핵심 기술** | 카카오맵 JavaScript SDK, GPS Geolocation API, 바운딩 박스 검색 |
| **데이터** | 변호사 17,326명 (좌표·전문분야·사무소 정보) |

**주요 기능**
- 현재 위치 기반 반경 검색 (기본 5km, 최대 5,000명)
- 줌 레벨에 따라 마커/클러스터 자동 전환 (동 단위 클러스터링)
- 12대 전문분야 분류 체계 (형사, 민사/가족, 부동산, 노동 등)
- AI 채팅에서 "변호사 찾아줘" → 자동 지도 이동 + 반경 확장 액션

**API 엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/lawyer-finder/nearby` | 반경 내 변호사 검색 |
| GET | `/api/lawyer-finder/clusters` | 줌 레벨별 클러스터 데이터 |
| GET | `/api/lawyer-finder/search` | 이름/사무소/지역/전문분야 검색 |
| GET | `/api/lawyer-finder/categories` | 12대 전문분야 분류 |
| GET | `/api/lawyer-finder/{id}` | 변호사 상세 정보 |

---

### 2-2. 변호사 통계 대시보드

> 지역별·전문분야별 변호사 분포를 Choropleth 지도, 히트맵, 인구 예측 차트로 시각화

| 항목 | 내용 |
|------|------|
| **대상** | 변호사 |
| **핵심 기술** | react-simple-maps (TopoJSON), recharts, 인구추계 데이터 |
| **데이터** | 변호사 17,326명 + 인구추계 (2025~2040) + 재판통계 (2015~2024) |

**5가지 지도 뷰 모드**

| 뷰 모드 | 시각화 | 색상 |
|---------|--------|------|
| 변호사 수 | 지역별 변호사 절대 수 | 빨강 그라데이션 |
| 인구 대비 밀도 | 10만명당 변호사 수 | 에메랄드 그라데이션 |
| 인구 예측 | 2030/2035/2040 추계 반영 | 보라 그라데이션 |
| 사건 수요 | 법원별 사건 접수 현황 | 앰버 그라데이션 |
| 부담 지수 | 변호사당 사건 부담 | 로즈 그라데이션 |

**주요 기능**
- 시/도 선택 시 자동 줌/센터 이동 (17개 시도 프리셋)
- 지역 × 전문분야 교차 분석 히트맵
- 법원 위치 마커 + 사건 수 로그 스케일 표시
- 법원 클릭 → 관할 지역 하이라이트

**API 엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/lawyer-stats/overview` | 전체 현황 요약 |
| GET | `/api/lawyer-stats/density-by-region` | 인구 대비 밀도 (연도별) |
| GET | `/api/lawyer-stats/cross-analysis` | 지역 × 전문분야 교차 분석 |
| GET | `/api/lawyer-stats/demand-by-region` | 법원별 사건 수요 |

---

### 2-3. 판례 검색

> RAG 기반 하이브리드 검색으로 92,000+ 판례에서 유사 판례를 찾고, AI에게 질문

| 항목 | 내용 |
|------|------|
| **대상** | 공통 |
| **핵심 기술** | LanceDB 벡터 검색 + BM25 키워드 검색 + RRF 병합 + Cross-encoder 리랭킹 |
| **데이터** | 판례 92,055건 벡터 임베딩 + PostgreSQL 원문 |

**주요 기능**
- 자연어 질문 → 쿼리 리라이팅 → 하이브리드 검색 → 리랭킹 → AI 답변
- 판례 상세 뷰 (주문, 판결요지, 전문)
- 특정 판례에 AI 질문 (그래프 컨텍스트 자동 보강: 인용 법령 + 유사 판례)
- 법원 필터 (대법원, 고등법원, 지방법원 등)
- 채팅 응답 내 판례번호 자동 인식 → 클릭 가능 버튼 변환

**API 엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/case-precedent/search` | 유사도 기반 법률 문서 검색 |
| GET | `/api/case-precedent/precedents` | 판례 키워드 검색 |
| GET | `/api/case-precedent/precedents/{id}` | 판례 상세 (LanceDB + PostgreSQL) |
| POST | `/api/case-precedent/precedents/{id}/ask` | 특정 판례에 AI 질문 |

---

### 2-4. 법령 검색 + 법령 체계도

> RAG 기반 법령 검색 + D3.js Force-Directed 그래프로 법령 간 계급 관계 시각화

| 항목 | 내용 |
|------|------|
| **대상** | 공통 (체계도: 변호사) |
| **핵심 기술** | react-force-graph-2d (D3 Force), PostgreSQL Recursive CTE |
| **데이터** | 법령 5,548건 + 법령 계급/인용 그래프 관계 |

**주요 기능**
- 법령명/약칭 퍼지 검색 (pg_trgm similarity + ILIKE 폴백)
- Force-Directed 그래프: 법령 유형별 색상·아이콘 (헌법, 법률, 대통령령, 총리령 등)
- 상위법/하위법/관련법 계층 탐색 (Recursive CTE, 최대 depth=5, 200노드 상한)
- 법령 노드 클릭 → 하위 법령 지연 로딩

**API 엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/case-precedent/statutes/search` | 법령 검색 |
| GET | `/api/case-precedent/statutes/hierarchy/{id}` | 법령 계층 (상위/하위/관련) |
| GET | `/api/case-precedent/statutes/graph` | Force-directed 그래프 데이터 |
| GET | `/api/case-precedent/statutes/{id}/children` | 하위 법령 지연 로딩 |

---

### 2-5. 사건 스토리보드

> 텍스트·음성·이미지 등 멀티모달 입력으로 사건 타임라인을 자동 추출하고, 간트차트·웹툰·영상으로 시각화

| 항목 | 내용 |
|------|------|
| **대상** | 공통 |
| **핵심 기술** | vis-timeline (간트차트), Whisper STT, Gemini Vision, Gemini 2.0 Flash (이미지/영상 생성), moviepy |
| **멀티모달** | 텍스트, 음성 (STT), 이미지 (Vision AI), 파일 (증거) |

**주요 기능**
- 텍스트 → LLM 타임라인 자동 추출 (날짜·주체·행위·법적의미)
- 음성 녹음 → Whisper STT → 타임라인 추출
- 이미지/증거 → Gemini Vision 분석 → 타임라인 추출
- 다중 증거 일괄 업로드 + SSE 실시간 진행률
- 3단계 병합: 날짜 후보 → LLM 중복 감지 → 적용 (기존 타임라인에 증분 병합)
- 간트차트 뷰 (vis-timeline) + 카드 뷰 전환
- 타임라인 → 웹툰 스토리보드 이미지 생성 (Gemini 2.0 Flash)
- 이미지 → 영상 생성 (moviepy, TransitionType 선택)
- 신뢰도별 시각적 구분 (opacity 차등)

**API 엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/storyboard/extract` | 텍스트 → 타임라인 추출 |
| POST | `/api/storyboard/transcribe` | 음성 → STT (Whisper, 25MB) |
| POST | `/api/storyboard/analyze-image` | 이미지 → 타임라인 추출 |
| POST | `/api/storyboard/analyze-batch` | 다중 증거 일괄 분석 |
| POST | `/api/storyboard/merge` | 기존 타임라인에 증거 병합 |
| POST | `/api/storyboard/generate-images-batch` | 이미지 일괄 생성 (비동기) |
| POST | `/api/storyboard/generate-video` | 이미지 → 영상 생성 |
| GET | `/api/storyboard/jobs/{job_id}/status` | SSE 진행 상태 |

---

### 2-6. 소액소송 도우미

> 4단계 위저드 인터뷰로 사건 정보를 수집하고, AI가 법률 서류(내용증명·지급명령·소액심판청구서)를 자동 생성

| 항목 | 내용 |
|------|------|
| **대상** | 일반인 |
| **핵심 기술** | LangGraph interrupt 패턴 (멀티턴), LLM 서류 생성, PDF/DOCX 출력 |
| **분쟁 유형** | 물품대금, 사기, 보증금, 용역대금, 임금체불 (5종) |

**4단계 위저드 흐름**

```
Step 1: 분쟁 유형 선택
    ↓
Step 2: 사건 정보 입력 (자연어 인터뷰 5단계: 분쟁유형→피고→금액→날짜→경위)
    ↓
Step 3: 증거 첨부 (파일 업로드 + 타임라인 정리)
    ↓
Step 4: AI 서류 생성 (내용증명 / 지급명령 / 소액심판청구서) → PDF + DOCX
```

**주요 기능**
- AI 채팅과 위저드 UI 양방향 동기화 (`useSmallClaimsSync`)
- 분쟁 유형별 관련 판례 RAG 검색
- 증거 체크리스트 자동 제공
- 소송 절차 가이드 (5개 유형별)
- 법원 비용 계산기

**API 엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/small-claims/interview/start` | 인터뷰 세션 시작 |
| POST | `/api/small-claims/interview/{id}/answer` | 답변 → 다음 질문/완료 |
| POST | `/api/small-claims/generate-document` | AI 법률 서류 생성 (PDF+DOCX) |
| GET | `/api/small-claims/guide/{case_type}` | 소송 절차 가이드 |
| GET | `/api/small-claims/related-cases/{type}` | RAG 관련 판례 |

---

### 2-7. 모의 법정

> Phaser.js 픽셀아트 게임 엔진으로 구현된 법정 시뮬레이션. 4역할 AI 에이전트가 실제 재판 절차를 수행

| 항목 | 내용 |
|------|------|
| **대상** | 공통 |
| **핵심 기술** | Phaser.js 3.90 (Canvas), LPC 스프라이트 시스템, LangGraph 서브그래프 |
| **재판 유형** | 형사 (모두진술→증거조사→구형→최후변론), 민사 (소장→답변→심리→판결) |

**주요 기능**
- 픽셀아트 법정 씬 (대법원 로비 → 법정 내부, 타일맵/배경이미지/Graphics 3단계 폴백)
- 6종 LPC 캐릭터 스프라이트 (832x1344, 64px 프레임, walk/speak/react/idle 애니메이션)
- 8가지 감정 표현 (neutral, angry, thinking, sad, confident, stern, recording, judging)
- 도트 스프라이트 감정 아이콘 (8x8 도트, PIXEL_SIZE=3)
- 대화 속도 제어 (1x/2x/4x/즉시) + Space 키 진행 + 스킵
- 말풍선 NineSlice + 타이핑 효과 + 페이지 분할
- BGM/SFX (로비, 법정, 의사봉, 타이핑, 이의제기, 단계전환)
- 데모 시나리오 2개 (형사 사기, 민사 손해배상) — 프로덕션에서도 사용 가능
- 판례/법령 RAG 검색으로 증거 보강

**페이지 흐름**: setup → briefing (데모 시나리오) → trial → verdict

**API 엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/mock-trial/case-types` | 사건 유형 목록 |
| GET | `/api/mock-trial/roles/{type}` | 역할 목록 |
| POST | `/api/mock-trial/search-evidence` | 판례/법령 검색 |
| GET | `/api/mock-trial/stage-info/{type}` | 절차별 단계 정보 |

---

### 2-8. 로스쿨 학습

> RAG 기반 법학 튜터. AI가 판례를 분석하고 학습 가이드·퀴즈를 제공

| 항목 | 내용 |
|------|------|
| **대상** | 변호사 (로스쿨 학생) |
| **핵심 기술** | RAG 검색 + LLM 스트리밍 (LawStudyAgent) |

**주요 기능**
- AI 채팅으로 법학 개념 질문 → RAG 기반 판례/법령 검색 + 해설
- 주제별/난이도별 퀴즈 생성
- 판례 요약 조회

---

### 2-9. 콘텐츠 마케팅

> 변호사 대상 유튜브/SNS 콘텐츠 자동 생성 — 트렌드 분석 → 키워드 수집 → 대본 생성 → 웹툰 자동 생성

| 항목 | 내용 |
|------|------|
| **대상** | 변호사 |
| **핵심 기술** | 7종 외부 API 병렬 수집, 5차원 트렌드 스코어링, Gemini Pro 이미지 생성, SSE 스트리밍 |
| **외부 API** | Tavily, Naver, Google CSE, NewsData.io, NewsAPI, Google Trends, YouTube |

**서비스 흐름**

```
Step 1: 페르소나 설정
    │   AI 분석 (대화 이력 30건+, 신뢰도 0.6+) 또는 수동 온보딩
    ↓
Step 2: 트렌드 분석
    │   7종 소스 병렬 수집 → 5차원 스코어링 → Legal Gate 필터 (0.3 미만 제외)
    ↓
Step 3: 키워드 탐색
    │   커뮤니티 트렌드 키워드 수집 (SSE) → 관련 뉴스 검색
    ↓
Step 4: 유튜브 대본 생성
    │   페르소나 + 트렌드/키워드 → LLM 대본 SSE 스트리밍
    ↓
Step 5: 웹툰 스토리보드 생성
    │   대본 → Gemini Pro Image → 최대 14패널 WebP
    │   개별 패널 재생성 가능
    ↓
Step 6: 피드백 (rating + feedback_type)
```

**5차원 트렌드 스코어링**

| 차원 | 가중치 | 설명 |
|------|--------|------|
| 언급 빈도 | 0.25 | 수집된 소스에서의 등장 횟수 |
| 법률 연관도 | 0.60 | 법률 키워드·주제와의 관련성 |
| 논란도 | 0.20 | 찬반 의견 분포 |
| 확산도 | 0.10 | 소셜 미디어 전파 속도 |
| 채널 적합도 | 0.25 | 유튜브/블로그 콘텐츠 적합성 |

**API 엔드포인트 (17개)**

| 카테고리 | 주요 엔드포인트 |
|---------|---------------|
| 페르소나 | `POST /persona/analyze`, `POST /persona/onboarding`, `GET /persona/current` |
| 트렌드 | `POST /trends`, `GET /trends/{id}` |
| 대본 | `POST /script/generate` (SSE), `POST /script/metadata` |
| 키워드 | `GET /keywords/collect/stream` (SSE), `POST /keywords/{id}/news` |
| 웹툰 | `POST /script/webtoon`, `GET /script/webtoon/{id}/stream` (SSE) |

---

### 2-10. 사건 워크스페이스

> 사건별 대화·태그·타임라인을 통합 관리하는 워크스페이스

| 항목 | 내용 |
|------|------|
| **대상** | 변호사 |
| **핵심 기술** | HttpOnly 쿠키 세션 기반 소유권, LLM 태그 추출, 구조화 요약 |
| **연동** | 통합 채팅 대화 자동 연결, 스토리보드 타임라인 연동 |

**주요 기능**
- 사건 CRUD (생성, 조회, 수정, 삭제)
- 4탭 상세 뷰: 요약 / 태그 / 타임라인 / 대화
- 대화에서 자동 태그 추출 (6가지 태그 타입: 인물, 날짜, 금액, 법령, 쟁점, 증거)
- 5턴마다 자동 분류 + 10턴마다 구조화 요약
- 태그 기반 타임라인 AI 재생성
- JSON/TXT 형식 내보내기

**API 엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/workspace/cases` | 사건 생성 (대화 연결) |
| GET | `/api/workspace/cases` | 사건 목록 (상태/검색/페이지네이션) |
| GET | `/api/workspace/cases/{id}` | 사건 상세 (태그+타임라인+대화) |
| POST | `/api/workspace/cases/{id}/timeline/rebuild` | 타임라인 AI 재생성 |
| GET | `/api/workspace/cases/{id}/export` | 사건 내보내기 |

---

### 2-11. 법률 뉴스

> 법률 뉴스 자동 수집 파이프라인 + 하이브리드 검색 + 통계 대시보드

| 항목 | 내용 |
|------|------|
| **대상** | 공통 |
| **핵심 기술** | 크롤링 파이프라인 (로타임즈 + 네이버), Upstage Solar 요약, SimHash 중복 제거 |
| **검색** | LanceDB 벡터 + BM25 키워드 + 리랭커 |

**주요 기능**
- 뉴스 소스 2종: 법률신문(로타임즈), 네이버 법률 뉴스
- AI 자동 요약 (Upstage Solar)
- 규칙 기반 카테고리 분류 (판결큐레이션, 법조계인사, 법령동향, 형사검찰, 소송재판)
- SimHash 기반 유사 기사 중복 제거 (임계값 3)
- 하이브리드 검색 (벡터 + BM25 + 리랭커)
- 일별 수집 통계 바 차트 + 카테고리 도넛 차트
- RAG 기여도 카드 (법령 DB vs 뉴스 소스 비율)
- Dead Letter Queue (DLQ) — 처리 실패 기사 재처리

**API 엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/legal-news/list` | 뉴스 목록 (소스/날짜 필터) |
| POST | `/api/legal-news/search` | 하이브리드 검색 |
| GET | `/api/legal-news/stats/daily` | 일별 수집 통계 |
| GET | `/api/legal-news/stats/category` | 카테고리 분포 |
| GET | `/api/legal-news/stats/rag-contribution` | RAG 기여도 분석 |

---

### 2-12. 통합 AI 채팅

> 모든 모듈을 하나의 채팅 인터페이스로 통합. SSE 스트리밍 + 에이전트 자동 라우팅 + 페이지 자동 이동

| 항목 | 내용 |
|------|------|
| **대상** | 공통 |
| **핵심 기술** | LangGraph StateGraph, SSE 스트리밍, requestAnimationFrame 최적화 |
| **에이전트** | 9개 전문 에이전트 + 1개 일반 폴백 |

**핵심 기능**

| 기능 | 설명 |
|------|------|
| SSE 스트리밍 | 토큰 단위 실시간 응답 (rAF 배치 업데이트) |
| 에이전트 자동 라우팅 | 페이지 URL → 에이전트 자동 감지, 키워드 기반 의도 분류 |
| 페이지 자동 이동 | AI 응답의 NAVIGATE 액션 → 해당 모듈 페이지로 자동 이동 |
| 세션 유지 | 활성 에이전트 고정 + 탈출 키워드 감지 ("그만", "종료", "처음으로") |
| 대화 영속화 | PostgreSQL 체크포인터 + 대화 이력 저장 |
| 판례번호 인식 | 정규식 패턴 자동 감지 → 클릭 가능 버튼 변환 |
| 모드 전환 | split (우측 50% 패널) / floating (380x600px 말풍선) |

**에이전트 라우팅 맵**

```
페이지 경로               →  에이전트           →  자동 이동 페이지
/lawyer-finder            →  lawyer_finder      →  /lawyer-finder
/case-precedent?agent=... →  case_search        →  /case-precedent?agent=case_search
/case-precedent?agent=... →  law_search         →  /case-precedent?agent=law_search
/storyboard               →  storyboard         →  /storyboard
/small-claims             →  small_claims        →  /small-claims
/lawyer-stats             →  lawyer_stats        →  /lawyer-stats
/law-study                →  law_study           →  /law-study
(채팅 전용)               →  mock_trial          →  /mock-trial
(채팅 전용)               →  content_marketing   →  /content-marketing
(채팅 전용)               →  workspace           →  /workspace
(폴백)                    →  general             →  -
```

**SSE 이벤트 프로토콜**

| 이벤트 | 페이로드 | 설명 |
|--------|---------|------|
| `token` | `{ content }` | 텍스트 토큰 스트리밍 |
| `sources` | `[{ source_id, title, ... }]` | 참조 문서 목록 |
| `metadata` | `{ agent_used, actions, session_data }` | 메타데이터 |
| `done` | `{ conversation_id }` | 응답 완료 |
| `error` | `{ message }` | 에러 발생 |

**API 엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/chat/stream` | SSE 스트리밍 채팅 |
| POST | `/api/chat` | 일반 채팅 (JSON) |
| GET | `/api/chat/agents` | 사용 가능 에이전트 목록 |

---

## 3. 기술 아키텍처

### 3-1. 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            클라이언트 (브라우저)                             │
│  Next.js 14 (App Router) + TypeScript + Tailwind CSS + Zustand + React Query │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌─────────┐ ┌────────────────┐    │
│  │ KakaoMap  │ │ Choropleth│ │ Phaser.js │ │vis-timeline│ │react-force-graph│ │
│  │(지도 SDK) │ │(TopoJSON)│ │(픽셀아트) │ │(간트차트)│ │ (법령 그래프)  │    │
│  └──────────┘ └──────────┘ └───────────┘ └─────────┘ └────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                     Next.js API Routes (SSE 프록시)                         │
│        /api/chat/stream, /api/storyboard/*, /api/content-marketing/*        │
│                    + next.config.js rewrites (일반 API)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                       FastAPI 서버 (:8000)                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  미들웨어: SessionMiddleware (HttpOnly 쿠키) → CORS → RateLimiter  │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  ModuleRegistry (자동 등록)     │    통합 채팅 API (/api/chat)      │    │
│  │  ┌─────────────────────────┐   │    ┌───────────────────────────┐  │    │
│  │  │ 11개 기능 모듈          │   │    │ LangGraph StateGraph       │  │    │
│  │  │ /api/lawyer-finder     │   │    │  ┌─────────────────────┐  │  │    │
│  │  │ /api/lawyer-stats      │   │    │  │    router_node       │  │  │    │
│  │  │ /api/case-precedent    │   │    │  │  (RulesRouter 의도)  │  │  │    │
│  │  │ /api/storyboard        │   │    │  └────────┬────────────┘  │  │    │
│  │  │ /api/small-claims      │   │    │    ┌──────┴───────┐       │  │    │
│  │  │ /api/mock-trial        │   │    │    ▼              ▼       │  │    │
│  │  │ /api/law-study         │   │    │  9개 에이전트   simple_chat│  │    │
│  │  │ /api/content-marketing │   │    │  + 3개 서브그래프 (폴백)  │  │    │
│  │  │ /api/workspace         │   │    │                           │  │    │
│  │  │ /api/legal-news        │   │    │  AsyncPostgresSaver       │  │    │
│  │  │ /api/multi-agent       │   │    │  (체크포인터 영속화)       │  │    │
│  │  └─────────────────────────┘   │    └───────────────────────────┘  │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                      RAG 파이프라인                                 │    │
│  │  쿼리 리라이팅 → 하이브리드 검색 → RRF 병합 → 리랭킹 → 원문 조회  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
├──────────────┬──────────────────┬───────────────────┬───────────────────────┤
│              │                  │                   │                       │
│  PostgreSQL 17                  │    LanceDB         │    외부 API           │
│  ┌──────────────────┐          │  ┌───────────────┐ │  ┌─────────────────┐ │
│  │ 법률 데이터       │          │  │ legal_chunks  │ │  │ OpenAI / Claude │ │
│  │ (판례 92K, 법령 5.5K)│       │  │ (656,532 벡터)│ │  │ Gemini / Solar  │ │
│  │ 변호사 17,326     │          │  │               │ │  │ 카카오맵 / Whisper││
│  │ 법률용어 72,700   │          │  │ local_ordinance│ │  │ Tavily / Naver  │ │
│  │ FTS 425,209       │          │  │ (240만 벡터)  │ │  │ NewsData / NewsAPI││
│  │ 인제스트 21타입    │          │  └───────────────┘ │  └─────────────────┘ │
│  │ 워크스페이스 6테이블│          │                   │                       │
│  │ 그래프 관계 5테이블│          │    MeCab userdic   │                       │
│  │ pg_textsearch BM25│          │  ┌───────────────┐ │                       │
│  └──────────────────┘          │  │ 37,366+ 법률  │ │                       │
│                                │  │ 복합명사 사전  │ │                       │
│                                │  └───────────────┘ │                       │
└────────────────────────────────┴───────────────────┴───────────────────────┘
```

### 3-2. 프론트엔드 기술 스택

| 항목 | 기술 | 용도 |
|------|------|------|
| **프레임워크** | Next.js 14.2 (App Router) | SSR + CSR 하이브리드, API Routes SSE 프록시 |
| **언어** | TypeScript 5.3 | 타입 안전성 |
| **스타일링** | Tailwind CSS 3.4 | Apple HIG 디자인 시스템 |
| **상태 관리** | Zustand 4.4 | 전역 상태 (UI, 채팅) |
| **서버 상태** | React Query 5.17 | API 캐싱 (staleTime: 5분) |
| **HTTP** | Axios 1.6 | API 클라이언트 (timeout: 180초) |
| **지도** | 카카오맵 SDK | 변호사 위치 + 클러스터 |
| **지도** | react-simple-maps 3.0 | Choropleth 통계 지도 |
| **차트** | recharts 3.7 | 바, 도넛, 히트맵 차트 |
| **타임라인** | vis-timeline 7.7 | 간트차트 |
| **그래프** | react-force-graph-2d 1.29 | 법령 관계 그래프 |
| **게임** | Phaser.js 3.90 | 모의 법정 픽셀아트 |
| **애니메이션** | framer-motion 12.28 | 랜딩 페이지 |
| **마크다운** | react-markdown 10.1 | AI 응답 렌더링 |
| **아이콘** | lucide-react 0.312 | UI 아이콘 세트 |

### 3-3. 백엔드 기술 스택

| 항목 | 기술 | 용도 |
|------|------|------|
| **프레임워크** | FastAPI (Python, async) | REST API + SSE |
| **ORM** | SQLAlchemy (asyncpg) | 비동기 DB 접근 |
| **마이그레이션** | Alembic | 24개 버전 관리 |
| **AI 오케스트레이션** | LangGraph | 멀티에이전트 StateGraph |
| **벡터 DB** | LanceDB | 임베디드 벡터 검색 |
| **임베딩** | KURE-v1 (1024차원) | 한국어 법률 도메인 |
| **리랭킹** | BGE-reranker-v2-m3-ko | Cross-encoder |
| **ONNX** | ONNX Runtime | INT8/FP16 추론 최적화 |
| **형태소** | MeCab + userdic | BM25 명사 추출 |
| **BM25** | pg_textsearch + BMW | Block-Max WAND 최적화 |
| **Rate Limit** | slowapi | 분당 30회 일반 / 10회 AI |
| **멀티 LLM** | OpenAI, Claude, Gemini, Solar | 용도별 LLM 선택 |
| **세션** | HttpOnly 쿠키 | 30일 유효 세션 |

### 3-4. 모듈 자동 등록 시스템

```
backend/app/modules/
├── case_precedent/    → /api/case-precedent
├── lawyer_finder/     → /api/lawyer-finder
├── lawyer_stats/      → /api/lawyer-stats
├── small_claims/      → /api/small-claims
├── storyboard/        → /api/storyboard
├── mock_trial/        → /api/mock-trial
├── law_study/         → /api/law-study
├── content_marketing/ → /api/content-marketing
├── workspace/         → /api/workspace
├── legal_news/        → /api/legal-news
└── (자동 스캔)

ModuleRegistry.register_all_modules()
  1. modules/ 폴더 스캔
  2. router/__init__.py 존재 여부 확인
  3. ENABLED_MODULES 환경변수로 필터링
  4. importlib 동적 import
  5. snake_case → /api/kebab-case 자동 변환
  6. app.include_router() 등록
```

---

## 4. AI/ML 시스템

### 4-1. LangGraph 멀티에이전트 아키텍처

```
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  router_node │  ← RulesRouter (키워드 패턴 + 신뢰도 점수)
                    │              │  ← 세션 유지 / 세션 전환 판단
                    └──────┬───────┘
           ┌───────┬───────┼───────┬───────┬───────┐
           ▼       ▼       ▼       ▼       ▼       ▼
    ┌───────────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
    │legal_search││lawyer ││small ││story ││mock  ││ ...  │
    │   _node   ││finder ││claims││board ││trial ││      │
    │  (RAG)    ││_node  ││_sub  ││_sub  ││_sub  ││      │
    └─────┬─────┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘
          └─────────┴───────┴───────┴───────┴───────┘
                              ▼
                       ┌──────────────┐
                       │     END      │
                       └──────────────┘
```

**9개 전문 에이전트 + 1개 폴백**

| 에이전트 | RAG | 스트리밍 | 핵심 기능 |
|---------|-----|---------|----------|
| LegalSearchAgent | O | O | Focus+Supplementary 병렬 검색, 사용자/변호사별 프롬프트 |
| LawyerFinderAgent | X | X | 위치 기반 반경 검색, 클러스터링 |
| SmallClaimsAgent | O | X | 소액소송 단계별 가이드 (서브그래프, interrupt) |
| StoryboardAgent | X | O | 사건 타임라인 생성, 태그 수집 |
| LawyerStatsAgent | X | O | 변호사 통계 안내 |
| LawStudyAgent | O | O | 법학 학습 문제/해설 생성 |
| MockTrialAgent | X | O | 모의 법정 시뮬레이션, 감정 상태 |
| ContentMarketingAgent | X | X | 트렌드 분석 + 대본 생성 |
| WorkspaceAgent | X | O | 사건 CRUD, 태그 자동 수집 |
| SimpleChatAgent | X | O | 일반 LLM 채팅 (폴백) |

**라우팅 3단계 우선순위**:
1. 세션 유지 — `active_agent` 존재 시 현재 에이전트 유지 (단, 신뢰도 0.9+ 명시적 의도 시 전환)
2. 키워드 매칭 — `INTENT_PATTERNS` (키워드, 신뢰도) 튜플 매칭
3. 기본 폴백 — `general` (신뢰도 0.3)

### 4-2. RAG 파이프라인 5단계

```
사용자 쿼리: "임대차 보증금을 돌려받지 못하면 어떻게 해야 하나요?"
    │
    ▼
━━━ Step 1: 쿼리 리라이팅 ━━━
    │  LLM 기반 법률 용어 변환
    │  "임대차 보증금 반환 명도 퇴거 주택임대차보호법"
    │  (일상 표현→법률 용어, 범용 표현 제거, 적용 법률명 추가)
    ▼
━━━ Step 2: 하이브리드 검색 ━━━
    │  ┌─ 벡터 검색 (LanceDB) ──── KURE-v1 1024차원 cosine ──┐
    │  │  656,532건 + 240만건 벡터                              │
    │  │                                                        │ asyncio.gather
    │  └─ BM25 검색 (PostgreSQL) ── MeCab 명사 + pg_textsearch ─┘
    │     425,209건 FTS 인덱스, BMW 최적화
    ▼
━━━ Step 3: RRF 병합 ━━━
    │  Reciprocal Rank Fusion (k=60)
    │  score(d) = Σ 1/(k + rank_i)
    │  양쪽 등장 문서: search_source = "both"
    ▼
━━━ Step 4: Cross-encoder 리랭킹 ━━━
    │  dragonkue/bge-reranker-v2-m3-ko (Sigmoid 0~1)
    │  적응형 Truncation: head(3000자) + tail(1000자)
    │  최소 점수 임계값: 0.01
    │  ONNX INT8 최적화 (3.52x 속도 향상)
    ▼
━━━ Step 5: 원문 배치 조회 + 포맷팅 ━━━
    │  top-k만 PostgreSQL 원문 조회 (data_type별 병렬)
    │  법령: law_articles 조문 단위 교체
    │  LLM 스트리밍 응답 생성
    ▼
최종 응답 (SSE 스트리밍)
```

**검색 모드 프리셋**

| 프리셋 | 용도 | 검색 수 | 리랭킹 top-k |
|--------|------|---------|-------------|
| `legal_search_precedent` | 판례 검색 | 10 (주) + 2 (보충) | top-5 |
| `legal_search_law` | 법령 검색 | 10 (주) + 2 (보충) | top-5 |
| `legal_search_all` | 전체 검색 | 20 | top-7 |
| `small_claims` | 소액소송 | 10 (판례) | top-3 |
| `quick_search` | 빠른 검색 | 5 | 없음 |

### 4-3. 임베딩 시스템

| 항목 | 내용 |
|------|------|
| **모델** | `nlpai-lab/KURE-v1` (한국어 법률 도메인 최적화) |
| **차원** | 1024차원 |
| **모델 크기** | 2.3GB |
| **정규화** | `normalize_embeddings=True` (cosine 유사도) |
| **캐시** | LRU 1,024개 (~4MB), Thundering Herd 방지 (inflight Future 공유) |
| **스레드풀** | 전용 ThreadPoolExecutor (max_workers=4) |

**ONNX 최적화 변형**

| Variant | 정밀도 | Cosine 유사도 | 속도 |
|---------|--------|--------------|------|
| `ort-opt` | FP32 | 1.000 (무손실) | 기준 |
| `ort-opt-qdq` | INT8 | 0.999 | +23% |
| `onnx-fp16` | FP16 | 1.000 | 중간 |

### 4-4. 리랭킹 시스템

| 항목 | 내용 |
|------|------|
| **모델** | `dragonkue/bge-reranker-v2-m3-ko` (CrossEncoder) |
| **모델 크기** | 2.1GB |
| **활성화 함수** | Sigmoid (0~1 출력) |
| **최소 점수** | 0.01 |
| **배치 크기** | 32 |
| **적응형 Truncation** | head 3,000자 + tail 1,000자 (최대 4,000자) |

**ONNX 리랭커 최적화**

| Variant | Pearson 상관 | 속도 향상 |
|---------|-------------|----------|
| `ort-opt` | FP32 무손실 | 기준 |
| `ort-opt-qdq` | 0.9999 | **3.52x** |

### 4-5. ONNX 품질 게이트

앱 시작 시 자동 실행되는 ONNX 모델 품질 검증 시스템.

| 검증 대상 | 메트릭 | 임계값 |
|----------|--------|--------|
| 임베딩 | cosine similarity 평균 | ≥ 0.995 |
| 리랭커 | Pearson 상관계수 | ≥ 0.990 |

- 16개 법률 도메인 테스트 쿼리로 PyTorch vs ONNX 비교
- 품질 미달 시 자동 PyTorch 폴백 (`ONNX_QUALITY_GATE_FALLBACK=true`)
- 검증 후 PyTorch 모델 즉시 해제 → ~2.3GB 메모리 절약

### 4-6. BM25 전문검색

| 항목 | 내용 |
|------|------|
| **엔진** | pg_textsearch v0.5.1 (Timescale, PostgreSQL 확장) |
| **최적화** | BMW (Block-Max WAND) — `ORDER BY <@> ASC LIMIT n` 패턴 |
| **인덱스 건수** | 425,209건 (21개 타입 통합) |
| **토크나이저** | MeCab (법률 userdic 37,366+ 엔트리) |
| **명사 필터** | NNG + NNP (일반/고유명사), 2자 이상 |
| **법률 용어 보강** | 72,700건 frozenset → 복합명사 추가 인식 |

### 4-7. 그래프 검색

PostgreSQL Recursive CTE로 법령 계급·판례 인용 관계를 탐색.

**그래프 테이블**

| 테이블 | 관계 | 설명 |
|--------|------|------|
| `statute_hierarchy` | child → parent | 법령 계급 (시행령→법률) |
| `statute_relations` | RELATED_TO | 법령 관련 관계 (양방향) |
| `case_statute_citations` | 판례 → 법령 | 판례가 인용한 법령 |
| `case_case_citations` | 판례 → 판례 | 판례 간 인용 |

**안전 상수**: 최대 depth=5, 최대 200노드, 쿼리 타임아웃 5초

### 4-8. RAG 평가 시스템

Gradio UI (포트 7860)로 검색 품질을 자동 평가하는 시스템.

**평가 메트릭 및 목표**

| 메트릭 | 공식 | 목표 |
|--------|------|------|
| Recall@10 | `|retrieved ∩ relevant| / |relevant|` | ≥ 0.80 |
| MRR | `1/rank(첫 정답)` 평균 | ≥ 0.70 |
| Hit Rate@10 | 상위 10에 정답 포함 비율 | ≥ 0.90 |
| NDCG@10 | `DCG/IDCG` (순위 가중) | ≥ 0.75 |

- Solar Pro LLM 기반 자동 질문 생성
- 6가지 쿼리 유형 (단순조회, 개념검색, 비교검색, 참조추적, 시간검색, 복합검색)
- 성능 목표: P50 < 200ms, P95 < 500ms

---

## 5. 데이터 인프라

### 5-1. PostgreSQL 17

| 항목 | 수치 |
|------|------|
| Alembic 마이그레이션 | 24개 |
| ORM 모델 파일 | 38개 |
| 테이블 수 | ~45개 (인제스트 19 + 핵심 도메인 26) |
| 확장 | pg_textsearch, uuid-ossp, pg_trgm |
| 커스텀 빌드 | PostgreSQL 17 Alpine + pg_textsearch 소스 빌드 |

**주요 테이블 관계**

```
law_documents ←── statute_hierarchy ──→ law_documents   (법령 계급)
     ↑                                      ↑
case_statute_citations              statute_aliases     (약칭)
     |
precedent_documents ←── case_case_citations ──→ precedent_documents (인용)

workspace_cases ──→ timeline_items ──→ activity_logs    (워크스페이스)
     |
chat_conversations ──→ chat_messages                    (대화)

fts_index: PK(source_id, data_type) — 21개 타입 통합 BM25 인덱스
```

### 5-2. LanceDB 벡터 DB

| 테이블 | 벡터 수 | 임베딩 모델 | 인덱스 |
|--------|---------|------------|--------|
| `legal_chunks` | 656,532건 | KURE-v1 (1024차원) | IVF_FLAT (nprobes=30) |
| `local_ordinance_chunks` | ~240만건 | KURE-v1 (1024차원) | IVF_FLAT |
| **합계** | **~316만건** | | |

### 5-3. MeCab 법률 사전

| 항목 | 수치 |
|------|------|
| 원본 법률 용어 | 72,700건 (PostgreSQL `legal_terms` 테이블) |
| MeCab userdic 엔트리 | 37,366+ (한글 전용 + 혼합 + 괄호 변형) |
| 수동 보강 | `manual_terms.json` (VV+ETN 동사 활용형, 복합어 분리 방지) |
| 출력 | `legal_terms.dic` (컴파일된 바이너리) + 분해맵 JSON |

### 5-4. 21개 타입 통합 인제스트 파이프라인

```
data/ingest_source/*.json (48개 파일, ~4.9GB)
        │
        ▼
   ┌─── db 적재 ───┐─── fts 인덱스 ───┐─── vector 임베딩 ───┐
   │ PostgreSQL     │ pg_textsearch    │ LanceDB             │
   │ ORM 저장       │ MeCab 명사 추출  │ KURE-v1 1024차원    │
   │ 1,000건 배치   │ BM25 인덱스 빌드 │ 청크 단위 임베딩    │
   │ 멱등성(UPSERT) │ 425,209건        │ 656,532건 + 240만건 │
   └───────────────┘└─────────────────┘└────────────────────┘
```

**21개 데이터 타입**

| # | 타입 | 건수 | # | 타입 | 건수 |
|---|------|------|---|------|------|
| 1 | 판례 | 92,055 | 12 | 위원회(인권) | 3,721 |
| 2 | 법령 | 5,548 | 13 | 위원회(개인정보) | 1,448 |
| 3 | 행정규칙 | 17,332 | 14 | 위원회(고용) | 118 |
| 4 | 헌재결정례 | 31,718 | 15 | 위원회(금융) | 662 |
| 5 | 행정심판례 | 34,254 | 16 | 위원회(산업) | 782 |
| 6 | 법령해석례 | 8,597 | 17 | 위원회(환경) | 358 |
| 7 | 부처유권해석 | 37,455 | 18 | 위원회(증권) | 636 |
| 8 | 특별행정심판 | 149,073 | 19 | 위원회(시민권) | 635 |
| 9 | 조약 | 3,589 | 20 | 위원회(공정거래) | 7,728 |
| 10 | 자치법규 | 160,276 | 21 | 위원회(언론) | 811 |
| 11 | 위원회(노동) | 40,714 | | **합계** | **~597,000건** |

### 5-5. Docker 인프라

**개발 환경 (docker-compose.yml)**

| 컨테이너 | 이미지 | 포트 |
|---------|--------|------|
| `law-platform-db` | 커스텀 (PG 17 + pg_textsearch) | 5432 |
| `lancedb-service` | 커스텀 (선택적) | 8100 |

**프로덕션 환경 (docker-compose.prod.yml)**

| 컨테이너 | 역할 |
|---------|------|
| `law-platform-db-prod` | PostgreSQL 17 + pg_textsearch |
| `law-platform-backend` | FastAPI 백엔드 |
| `law-platform-nginx` | Nginx 리버스 프록시 (80/443) |
| `law-platform-certbot` | Let's Encrypt 인증서 자동 갱신 |

### 5-6. Feature Flag 시스템

| Flag | 기본값 | 설명 |
|------|--------|------|
| `USE_DB_LAWYERS` | `False` | 변호사: PostgreSQL / JSON 전환 |
| `USE_HYBRID_SEARCH` | `True` | 벡터 + BM25 하이브리드 |
| `USE_BM25_SEARCH` | `True` | BM25 키워드 검색 |
| `USE_LOCAL_EMBEDDING` | `True` | 로컬 KURE-v1 / OpenAI 전환 |
| `USE_ONNX_EMBEDDING` | `False` | ONNX 임베딩 최적화 |
| `USE_ONNX_RERANKER` | `False` | ONNX 리랭커 최적화 |
| `ONNX_QUALITY_GATE_ENABLED` | `True` | 품질 게이트 자동 검증 |
| `ONNX_QUALITY_GATE_FALLBACK` | `True` | 품질 미달 시 PyTorch 폴백 |
| `USE_LEGAL_TERM_DICT` | `False` | MeCab 법률 복합명사 보강 |
| `NEWS_PIPELINE_ENABLED` | `True` | 뉴스 파이프라인 |
| `ENABLED_MODULES` | `[]` | 모듈 선택적 활성화 |

---

## 6. 핵심 설계 결정 및 차별점

### 6-1. Feature Flag 기반 이중 데이터 소스 패턴

```
                     ┌─ USE_DB_LAWYERS=true  ─→ PostgreSQL (17,326건)
LawyerFinderService ─┤
                     └─ USE_DB_LAWYERS=false ─→ JSON 파일 (롤백 보장)
```

- JSON 파일 서비스를 삭제하지 않고 유지 → 즉시 롤백 가능
- 함수명 규칙: JSON 함수명 + `_db` 접미사
- 점진적 마이그레이션 가능 (모듈별 독립 전환)

### 6-2. ONNX 품질 게이트 자동 폴백

```
앱 시작 → ONNX 모델 로드 → 16개 테스트 쿼리 검증
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
             cosine ≥ 0.995                  cosine < 0.995
             Pearson ≥ 0.990                 Pearson < 0.990
                    │                               │
                    ▼                               ▼
            ONNX 세션 활성화              ONNX 비활성화 → PyTorch 폴백
            PyTorch 메모리 해제                  (~2.3GB 절약 안됨)
            (~2.3GB 절약)
```

- 하드웨어/환경 차이에도 품질 보장
- 배포 후 수동 검증 불필요

### 6-3. asyncio.gather 병렬화 전략

RAG 파이프라인의 I/O 바운드 작업을 병렬 실행:

```python
# Focus + Supplementary 병렬 검색
focus_docs, sup_docs = await asyncio.gather(
    _search_and_deduplicate(query, focus_config, precomputed),
    _search_and_deduplicate(query, sup_config, precomputed),
)

# 벡터 + BM25 병렬 검색
vector_results, keyword_results = await asyncio.gather(
    asyncio.to_thread(_search_vector_ids, ...),
    asyncio.to_thread(search_by_keyword, ...),
)
```

- 임베딩 사전 계산 (1회) → Focus/Supplementary 공유
- 리랭킹 + 원문 조회도 병렬 실행

### 6-4. HttpOnly 쿠키 세션 기반 인증

```
요청 → SessionMiddleware → session_token 쿠키 확인
  ├─ 있으면 → request.state.session_token 주입
  └─ 없으면 → secrets.token_hex(32) 생성 → Set-Cookie (httponly, samesite=lax)
```

- JWT 대비 XSS 공격에 안전 (JavaScript 접근 불가)
- 30일 만료, `secure=not DEBUG`
- 모든 워크스페이스 데이터가 session_token으로 소유권 격리

### 6-5. LangGraph interrupt 패턴 (멀티턴 대화)

소액소송 서브그래프에서 사용:

```
SmallClaimsSubgraph
  ├─ interview_node → interrupt() → 사용자 응답 대기
  ├─ (사용자 응답) → interview_node → interrupt() → ...
  ├─ (5단계 완료) → document_generation_node
  └─ END
```

- `AsyncPostgresSaver` 체크포인터로 상태 영속화
- 세션 중단 후에도 이어서 진행 가능

### 6-6. requestAnimationFrame 스트리밍 최적화

```typescript
// 기존: 토큰마다 setState → 렌더링 병목
onToken: (token) => setMessages(prev => [...prev, token])  // ❌

// 최적화: rAF로 프레임당 1회 배치 업데이트
onToken: (token) => {
  bufferRef.current += token
  if (!rafRef.current) {
    rafRef.current = requestAnimationFrame(() => {
      setStreamingText(bufferRef.current)  // ✅ 프레임당 1회
      rafRef.current = null
    })
  }
}
```

- 고빈도 토큰 스트리밍에서 렌더링 부하 대폭 절감
- `memo(MessageBubble)`: 완료 메시지 재렌더 방지

### 6-7. SSE 프록시 아키텍처

```
Next.js rewrites (일반 API) → FastAPI
Next.js API Routes (SSE) → FastAPI (버퍼링 우회)
```

- `next.config.js` rewrites는 SSE를 버퍼링하므로, 4개 SSE 경로는 API Route로 별도 처리
- 프록시 타임아웃: 120초

---

## 7. 프로젝트 규모 요약

### 7-1. 코드 규모

| 영역 | 파일 수 | 코드 줄 수 |
|------|---------|-----------|
| Backend Python (app/) | 256개 | 46,040줄 |
| Backend Python (scripts/) | 107개 | 36,003줄 |
| Backend Python (기타) | 105개 | 23,657줄 |
| Frontend TypeScript (src/) | 224개 | 36,305줄 |
| **합계** | **692개** | **~142,000줄** |

### 7-2. 데이터 규모

| 항목 | 수치 |
|------|------|
| 원본 법률 문서 | ~597,000건 (21개 타입) |
| 벡터 임베딩 (LanceDB) | ~316만건 (legal 65.6만 + 자치법규 240만) |
| BM25 전문검색 인덱스 | 425,209건 |
| 변호사 데이터 | 17,326명 |
| 법률 용어 사전 | 72,700건 |
| MeCab userdic | 37,366+ 엔트리 |
| 인제스트 소스 파일 | 48개, ~4.9GB |

### 7-3. AI 모델 규모

| 모델 | 크기 | 용도 |
|------|------|------|
| KURE-v1 | 2.3GB | 한국어 법률 임베딩 (1024차원) |
| BGE-reranker-v2-m3-ko | 2.1GB | Cross-encoder 리랭킹 |
| ONNX 변형 | 추가 ~4GB | INT8/FP16 최적화 버전 |
| **합계** | **~8.4GB** | |

### 7-4. 아키텍처 규모

| 항목 | 수치 |
|------|------|
| 프론트엔드 모듈 | 12개 |
| 백엔드 API 모듈 | 11개 |
| API 엔드포인트 (합계) | 91개 |
| LangGraph 에이전트 | 9개 + 1개 폴백 |
| LangGraph 서브그래프 | 3개 (소액소송, 모의법정, 스토리보드) |
| Alembic 마이그레이션 | 24개 |
| PostgreSQL 테이블 | ~45개 |
| 인제스트 데이터 타입 | 21개 |
| Feature Flags | 15개 |

### 7-5. 의존성

| 영역 | 패키지 수 | 주요 패키지 |
|------|----------|------------|
| Backend (프로덕션) | ~57개 | FastAPI, SQLAlchemy, LangGraph, LanceDB, sentence-transformers, ONNX Runtime, MeCab |
| Frontend (프로덕션) | 27개 | Next.js, React, Tailwind, Zustand, React Query, Phaser.js, D3, vis-timeline |
| Frontend (개발) | 11개 | TypeScript, ESLint, PostCSS |

### 7-6. 주요 외부 API 연동

| API | 용도 |
|-----|------|
| OpenAI (GPT) | LLM 응답 생성, 타임라인 추출 |
| Anthropic Claude | LLM 대안 |
| Google Gemini | Vision AI, 이미지 생성, LLM |
| Upstage Solar | 뉴스 요약, 쿼리 리라이팅, RAG 질문 생성 |
| 카카오맵 SDK | 변호사 위치 지도 |
| OpenAI Whisper | 음성→텍스트 STT |
| Tavily / Naver / Google CSE | 트렌드 뉴스 수집 |
| NewsData.io / NewsAPI | 뉴스 데이터 수집 |

---

> **본 보고서는 프로젝트의 전체 아키텍처, 12개 서비스 모듈, AI/ML 시스템, 데이터 인프라를 종합적으로 기술합니다.**
> **총 ~142,000줄의 코드, ~597,000건의 법률 문서, ~316만건의 벡터 임베딩으로 구성된 AI 기반 법률 서비스 통합 플랫폼입니다.**
