# Legal News Pipeline (법률 뉴스 수집/요약 파이프라인) Planning Document

> **Summary**: 법률신문 + 네이버 법 관련 기사를 매일 자동 수집하여 본문 정제, AI 요약, 메타데이터 구조화, RAG 인덱싱용 청킹 데이터를 생성하는 파이프라인. 기존 법제처 RAG를 변경하지 않고 "보조 코퍼스"로 추가.
>
> **Project**: law-3 (Legal President / 법률 대통령)
> **Author**: Claude (PDCA Agent Team)
> **Date**: 2026-02-26
> **Status**: Draft (v0.3 — 3중 검증 완료)
> **LLM**: Upstage Solar-Pro2
> **Scheduler**: Local cron/systemd

---

## 1. Overview

### 1.1 Purpose

기존 법제처 기반 RAG(법령/판례 원문)에 **"법률 뉴스/기사 요약"** 코퍼스를 추가하여, 최신 법률 이슈·판결 동향·입법 변화를 반영할 수 있는 보조 검색 데이터셋을 구축한다. 이 코퍼스는 "기사 요약(보조 근거)"임이 항상 명시된다.

### 1.2 Background

- 기존 RAG는 법제처 법령/판례 원문 기반으로, 정적 데이터이므로 최신 법률 동향 반영이 어려움
- 법률신문, 네이버 법 관련 기사는 최신 판결/입법/사회 이슈를 빠르게 반영
- 뉴스 코퍼스를 별도 테이블/인덱스로 추가하면 기존 RAG를 수정하지 않고 검색 범위를 확장 가능
- content_marketing 모듈에 이미 Naver 뉴스 검색, Rate Limiter, HTML 정제 등 재사용 가능한 인프라가 존재

### 1.3 핵심 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                    법률 뉴스 수집/요약 파이프라인                        │
│                                                                      │
│  ┌─── Stage 1: 수집 (Crawl) ────────┐                               │
│  │                                    │                               │
│  │  [법률신문]      [네이버 뉴스]      │                               │
│  │  RSS/HTML 크롤   키워드 검색 API    │                               │
│  │       │               │            │                               │
│  │       └───────┬───────┘            │                               │
│  │               ▼                    │                               │
│  │  ┌──────────────────────┐         │                               │
│  │  │  중복 제거 (3단계)     │         │                               │
│  │  │  URL → 제목+일자 → 해시│         │                               │
│  │  └────────┬─────────────┘         │                               │
│  └───────────┼────────────────────────┘                               │
│              ▼                                                        │
│  ┌─── Stage 2: 정제 (Clean) ────────┐                               │
│  │                                    │                               │
│  │  HTML 제거, 광고/네비 필터          │                               │
│  │  본문 정규화, 길이 검증 (≥500자)     │                               │
│  │               │                    │                               │
│  └───────────────┼────────────────────┘                               │
│                  ▼                                                    │
│  ┌─── Stage 2.5: PII 필터 ──────────┐                               │
│  │                                    │                               │
│  │  인명/전화번호/주소 패턴 마스킹      │                               │
│  │  (정규식 기반 + 키워드 필터)         │                               │
│  └────────────────┼───────────────────┘                               │
│                   ▼                                                   │
│  ┌─── Stage 3: 요약 (Summarize) ────┐                               │
│  │                                    │                               │
│  │  Upstage Solar-Pro2                │                               │
│  │  ┌──────────────────┐             │                               │
│  │  │ 구조화 요약 생성   │             │                               │
│  │  │ - 한줄 요지        │             │                               │
│  │  │ - 주요 쟁점 3~5개  │             │                               │
│  │  │ - 언급 법령/판례   │             │                               │
│  │  │ - 실무적 시사점    │             │                               │
│  │  └────────┬─────────┘             │                               │
│  └───────────┼────────────────────────┘                               │
│              ▼                                                        │
│  ┌─── Stage 4: 저장 (Store) ────────┐                               │
│  │                                    │                               │
│  │  PostgreSQL: news_articles 테이블   │                               │
│  │  (메타데이터 + 정제 본문 + 요약)     │                               │
│  │               │                    │                               │
│  └───────────────┼────────────────────┘                               │
│                  ▼                                                    │
│  ┌─── Stage 5: 청킹 (Chunk) ────────┐                               │
│  │                                    │                               │
│  │  요약 → 별도 chunk (항상 포함)       │                               │
│  │  본문 → 1,000~2,000자 분절 + overlap│                               │
│  │  is_secondary=true 메타 필수        │                               │
│  │               │                    │                               │
│  │  LanceDB: news_chunks 테이블        │                               │
│  │  (임베딩 벡터 + 메타데이터)           │                               │
│  └───────────────┼────────────────────┘                               │
│                  ▼                                                    │
│  ┌─── Stage 6: 리포트 (Report) ─────┐                               │
│  │                                    │                               │
│  │  수집량, 중복률, 성공/실패, 통계      │                               │
│  │  일일 운영 로그 파일 생성             │                               │
│  └────────────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 핵심 사용자 시나리오

**시나리오 1: 매일 자동 수집**
1. KST 06:00 cron 실행
2. 법률신문 + 네이버에서 전일자 기사 수집
3. 중복 제거 → 본문 정제 → AI 요약 → DB 저장 → 청킹/임베딩
4. 운영 리포트 로그 생성

**시나리오 2: 수동 실행 (백필/재처리)**
```bash
cd backend
uv run python -m scripts.news_pipeline.cli --date 2026-02-25
uv run python -m scripts.news_pipeline.cli --date-range 2026-02-20 2026-02-25
uv run python -m scripts.news_pipeline.cli --source lawtimes --date today
```

**시나리오 3: RAG 검색에서 뉴스 코퍼스 활용 (향후)**
- 기존 `legal_chunks` 테이블과 별도로 `news_chunks` 테이블 검색
- 검색 결과에 `is_secondary=true`, `disclaimer` 포함
- 챗봇 응답에 "참고: 뉴스 기사 요약 기반" 표시

---

## 2. Scope

### 2.1 In Scope

| 영역 | 세부 항목 |
|------|----------|
| **수집** | 법률신문 크롤링 (RSS/HTML), 네이버 뉴스 API 키워드 검색 |
| **정제** | HTML 제거, 광고/네비 필터, 본문 정규화, 길이 검증, PII 마스킹 |
| **요약** | Upstage Solar-Pro2 기반 구조화 요약 (한줄 요지, 쟁점, 법령/판례 언급, 시사점) |
| **저장** | PostgreSQL `news_articles` 테이블, 문서 단위 JSON |
| **청킹** | LanceDB `news_chunks` 테이블, JSONL 출력, 임베딩 생성 |
| **중복 제거** | URL 기준 + 제목+발행일 기준 + 본문 해시 기준 (3단계) |
| **스케줄러** | 로컬 cron/systemd (KST 06:00) |
| **모니터링** | 일일 운영 리포트 로그 (수집량, 중복률, 성공/실패) |
| **CLI** | 수동 실행, 백필, 소스별 실행, 날짜 범위 지정 |
| **크롤링 예절** | robots.txt 준수, Rate Limiting, User-Agent 명시 |

### 2.2 Out of Scope

| 영역 | 이유 |
|------|------|
| 기존 법제처 RAG 검색/응답 로직 변경 | 별도 코퍼스로만 추가 |
| 챗봇 응답 UX/프롬프트 변경 | 향후 별도 기능으로 구현 |
| 실시간 웹검색 (real-time RAG) | 배치 처리만 해당 |
| 유료 DB 계약 | 무료 접근 가능한 소스만 사용 |
| 프론트엔드 UI | 백엔드 파이프라인 + CLI만 구현 |

---

## 3. Technical Architecture

### 3.1 모듈 구조

```
backend/
├── app/
│   ├── models/
│   │   └── news_article.py              # ORM 모델 (news_articles 테이블)
│   ├── tools/
│   │   └── news_pipeline/               # 뉴스 파이프라인 코어
│   │       ├── __init__.py
│   │       ├── sources/                 # 데이터 소스
│   │       │   ├── __init__.py          # BaseNewsSource ABC
│   │       │   ├── lawtimes_source.py   # 법률신문 크롤러
│   │       │   └── naver_news_source.py # 네이버 뉴스 API
│   │       ├── cleaner.py              # 본문 정제/정규화
│   │       ├── pii_filter.py           # PII 마스킹 (인명/전화번호/주소)
│   │       ├── deduplicator.py         # 3단계 중복 제거
│   │       ├── summarizer.py           # Solar-Pro2 요약 생성
│   │       ├── reference_validator.py  # 법령/판례 Cross-Reference 검증
│   │       ├── chunker.py             # RAG용 청킹
│   │       ├── models.py              # 내부 데이터 모델
│   │       ├── config.py              # 파이프라인 설정
│   │       └── exceptions.py          # 예외 클래스
│   └── services/
│       └── service_function/
│           └── news_pipeline_service.py  # 파이프라인 오케스트레이션
├── scripts/
│   └── news_pipeline/                   # CLI & 스케줄러
│       ├── __init__.py
│       ├── __main__.py                  # python -m scripts.news_pipeline
│       ├── cli.py                       # CLI 인터페이스
│       └── cron_setup.sh               # cron 설정 스크립트
├── alembic/
│   └── versions/
│       └── NNN_add_news_articles_table.py  # 마이그레이션
└── data/
    └── news_pipeline/                   # 운영 데이터
        ├── reports/                     # 일일 리포트
        └── output/                      # JSONL 출력
```

### 3.2 데이터 소스 설계

#### A. 법률신문 (lawtimes.co.kr)

| 항목 | 설명 |
|------|------|
| 수집 방식 | RSS 피드 우선, 불가 시 HTML 크롤링 |
| 대상 섹션 | 사회, 법조, 판결, 입법, 해외법조 등 |
| Rate Limit | 최소 2초 간격, 시간당 100 요청 이하 |
| robots.txt | 수집 전 확인 필수 |
| 파싱 | BeautifulSoup4 + 기사 본문 영역 타겟팅 |

#### B. 네이버 뉴스 검색 API

| 항목 | 설명 |
|------|------|
| 수집 방식 | 네이버 검색 API (`/v1/search/news.json`) — 기존 NaverSource 패턴 재활용 |
| 키워드 | 법원, 판결, 검찰, 경찰, 로펌, 변호사, 헌재, 대법원, 민사, 형사, 행정, 노동, 세무, 공정거래 등 |
| API 키 | 기존 `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET` 재사용 |
| Rate Limit | 시간당 25,000 (일간) — 기존 설정 활용 |
| 원문 추출 | 네이버 링크 → 원문 URL → 원문 크롤링 (선택적) |

### 3.3 저장소 설계

#### PostgreSQL: `news_articles` 테이블

```python
# app/models/news_article.py
class NewsArticle(Base):
    __tablename__ = "news_articles"

    id: Mapped[str]                    # SHA256(url) 또는 UUID
    source: Mapped[str]                # "lawtimes" | "naver"
    publisher: Mapped[str]             # 매체명
    title: Mapped[str]
    author: Mapped[str | None]
    published_at: Mapped[datetime]     # ISO8601, KST
    collected_at: Mapped[datetime]     # ISO8601, KST
    url: Mapped[str]                   # UNIQUE
    section: Mapped[str | None]
    tags: Mapped[list[str] | None]     # ARRAY
    cleaned_text: Mapped[str]          # 정제된 본문
    summary_one_liner: Mapped[str]     # 한줄 요지
    summary_issues: Mapped[list[str]]  # 주요 쟁점
    summary_laws: Mapped[list[str]]    # 언급 법령
    summary_cases: Mapped[list[str]]   # 언급 판례
    summary_institutions: Mapped[list[str]]  # 언급 기관
    summary_implications: Mapped[list[str]]  # 시사점
    content_hash: Mapped[str]          # 정제 본문 SHA256
    disclaimer: Mapped[str]            # 고정: "본 문서는 기사 요약이며 법령/판례 원문이 아님"
    schema_version: Mapped[str]        # "1.0"
    is_indexed: Mapped[bool]           # LanceDB 임베딩 완료 여부
```

**인덱스:**
- `idx_news_url`: url UNIQUE
- `idx_news_published_at`: published_at B-tree (날짜 범위 조회)
- `idx_news_source`: source B-tree (소스별 필터)
- `idx_news_content_hash`: content_hash B-tree (중복 검출)
- `idx_news_tags`: tags GIN (태그 검색)

#### LanceDB: `news_chunks` 테이블

```python
# chunk 스키마
{
    "chunk_id": str,          # f"{doc_id}_{chunk_idx}"
    "doc_id": str,            # news_articles.id 참조
    "chunk_text": str,        # 청크 텍스트
    "chunk_type": str,        # "summary" | "body"
    "published_at": str,      # ISO8601
    "source": str,            # "lawtimes" | "naver"
    "publisher": str,
    "title": str,
    "url": str,
    "is_secondary": bool,     # 항상 True
    "data_type": str,         # "news_article"
    "vector": list[float],    # 1024차원 (KURE-v1)
}
```

### 3.4 요약 생성 설계

#### LLM 설정

```python
# Upstage Solar-Pro2 (기존 get_chat_model 활용)
from app.tools.llm import get_chat_model

summarizer_llm = get_chat_model(
    provider="upstage",
    model="solar-pro2",
    temperature=0.3,  # 요약은 낮은 temperature
)
```

#### 요약 프롬프트 템플릿

```
당신은 법률 전문 기자입니다. 아래 기사를 법률 AI 보조 코퍼스용으로 요약해주세요.

## 출력 형식 (JSON)
{
  "one_liner": "핵심 한 문장 요약",
  "issues": ["쟁점1", "쟁점2", ...],  // 3~5개
  "mentions": {
    "laws": ["관련 법령명"],           // 기사에 언급된 것만
    "cases": ["관련 판례"],            // 기사에 언급된 것만
    "institutions": ["관련 기관"]      // 기사에 언급된 것만
  },
  "implications": ["시사점1", ...]     // 1~3개
}

## 규칙
- 기사에 없는 판례/법령 조문 번호를 만들어내지 마세요
- "확정적 법률 결론"을 단정하지 마세요 (기사 요약임)
- 한국어로 작성하세요

## 기사 원문
제목: {title}
매체: {publisher}
발행일: {published_at}

{cleaned_text}
```

### 3.5 스케줄링

```bash
# cron 설정 (KST 06:00 = UTC 21:00 전일)
# /etc/cron.d/legal-news-pipeline
0 6 * * * cd /path/to/project/backend && /path/to/uv run python -m scripts.news_pipeline --date yesterday >> /path/to/data/news_pipeline/reports/cron.log 2>&1
```

---

## 4. Dependencies

### 4.1 기존 재활용 컴포넌트

| 컴포넌트 | 경로 | 재활용 방식 |
|----------|------|------------|
| Naver API 클라이언트 | `app/tools/trend/sources/naver_source.py` | 패턴 참조 (독립 구현) |
| Rate Limiter | `app/tools/trend/rate_limiter.py` | 직접 import 재사용 |
| LLM 클라이언트 | `app/tools/llm/__init__.py` | `get_chat_model(provider="upstage")` |
| 임베딩 모델 | `app/services/rag/embedding.py` | `create_query_embedding()` |
| Config 패턴 | `app/core/config.py` | 환경 변수 추가 |
| DB 연결 | `app/core/database.py` | `async_session_factory` |
| Ingest 패턴 | `scripts/ingest/` | CLI 구조, 배치 처리 패턴 참조 |

### 4.2 신규 의존성

| 패키지 | 용도 | 비고 |
|--------|------|------|
| `beautifulsoup4` | HTML 파싱/정제 | 이미 설치됨 (content_marketing) |
| `feedparser` | RSS 피드 파싱 | 신규 추가 필요 |
| `httpx` | HTTP 클라이언트 | 이미 설치됨 |
| `lxml` | HTML 파서 백엔드 | 이미 설치됨 |

### 4.3 환경 변수 (신규)

```bash
# news_pipeline 설정
NEWS_PIPELINE_ENABLED=true                    # 파이프라인 활성화
NEWS_PIPELINE_SUMMARY_PROVIDER=upstage        # 요약 LLM 프로바이더
NEWS_PIPELINE_SUMMARY_MODEL=solar-pro2        # 요약 LLM 모델
NEWS_PIPELINE_MIN_ARTICLE_LENGTH=500          # 최소 본문 길이 (자)
NEWS_PIPELINE_CHUNK_SIZE=1500                 # 청크 크기 (자)
NEWS_PIPELINE_CHUNK_OVERLAP=200               # 청크 겹침 (자)
NEWS_PIPELINE_MAX_ARTICLES_PER_RUN=200        # 1회 실행 최대 수집 수
NEWS_PIPELINE_RATE_LIMIT_CRAWL=2.0            # 크롤링 요청 간격 (초)
NEWS_PIPELINE_LAWTIMES_ENABLED=true           # 법률신문 수집 활성화
NEWS_PIPELINE_NAVER_ENABLED=true              # 네이버 뉴스 수집 활성화
NEWS_PIPELINE_LANCEDB_TABLE=news_chunks       # LanceDB 테이블명
NEWS_PIPELINE_RETENTION_DAYS=90               # 기사 보관 기간 (일)
NEWS_PIPELINE_LLM_FALLBACK_PROVIDER=openai    # 요약 LLM 장애 시 fallback 프로바이더

# 기존 변수 재사용
# NAVER_CLIENT_ID, NAVER_CLIENT_SECRET (네이버 API)
# UPSTAGE_API_KEY (Solar API)
# DATABASE_URL (PostgreSQL)
# LANCEDB_URI (LanceDB)
```

---

## 5. Data Schema

### 5.1 문서 단위 JSON (저장용)

```json
{
  "doc_id": "sha256_of_url",
  "source": "lawtimes",
  "publisher": "법률신문",
  "title": "대법원, 임대차보증금 반환 판결 기준 변경",
  "author": "김기자",
  "published_at": "2026-02-25T09:00:00+09:00",
  "collected_at": "2026-02-26T06:05:32+09:00",
  "url": "https://www.lawtimes.co.kr/news/...",
  "section": "판결",
  "tags": ["임대차", "보증금", "대법원"],
  "cleaned_text": "정제된 본문 텍스트...",
  "summary": {
    "one_liner": "대법원이 임대차보증금 반환 기준을 기존 판례에서 변경하는 판결을 선고했다",
    "issues": [
      "임대차보증금 반환 범위에 대한 기존 해석 변경",
      "임차인 보호 범위 확대 여부",
      "소급 적용 가능성"
    ],
    "mentions": {
      "laws": ["주택임대차보호법 제3조의2"],
      "cases": ["대법원 2026다12345"],
      "institutions": ["대법원 전원합의체"]
    },
    "implications": [
      "임대인-임차인 간 분쟁에서 보증금 반환 범위가 넓어질 전망",
      "유사 사건 하급심 판결에 즉각적 영향 예상"
    ]
  },
  "disclaimer": "본 문서는 기사 요약이며 법령/판례 원문이 아님",
  "content_hash": "abc123...",
  "language": "ko",
  "schema_version": "1.0"
}
```

### 5.2 청크 단위 JSONL (임베딩용)

```jsonl
{"chunk_id":"sha256_1_0","doc_id":"sha256_1","chunk_text":"[요약] 대법원이 임대차보증금...","chunk_type":"summary","published_at":"2026-02-25T09:00:00+09:00","source":"lawtimes","publisher":"법률신문","title":"대법원, 임대차보증금...","url":"https://...","is_secondary":true,"data_type":"news_article"}
{"chunk_id":"sha256_1_1","doc_id":"sha256_1","chunk_text":"정제된 본문 첫 번째 청크...","chunk_type":"body","published_at":"2026-02-25T09:00:00+09:00","source":"lawtimes","publisher":"법률신문","title":"대법원, 임대차보증금...","url":"https://...","is_secondary":true,"data_type":"news_article"}
```

---

## 6. Implementation Order

### Phase 1: 기반 구조 (1단계)
1. `app/models/news_article.py` — ORM 모델 정의
2. Alembic 마이그레이션 — `news_articles` 테이블 생성
3. `app/core/config.py` — 환경 변수 추가
4. `app/tools/news_pipeline/` — 패키지 스켈레톤 생성
5. `app/tools/news_pipeline/models.py` — 내부 데이터 모델
6. `app/tools/news_pipeline/config.py` — 파이프라인 설정
7. `app/tools/news_pipeline/exceptions.py` — 예외 클래스

### Phase 2: 수집 레이어 (2단계)
1. `app/tools/news_pipeline/sources/__init__.py` — BaseNewsSource ABC
2. `app/tools/news_pipeline/sources/lawtimes_source.py` — 법률신문 크롤러
3. `app/tools/news_pipeline/sources/naver_news_source.py` — 네이버 뉴스 수집
4. `app/tools/news_pipeline/deduplicator.py` — 3단계 중복 제거

### Phase 3: 정제 + PII + 요약 레이어 (3단계)
1. `app/tools/news_pipeline/cleaner.py` — 본문 정제/정규화
2. `app/tools/news_pipeline/pii_filter.py` — PII 마스킹 (정규식 기반)
3. `app/tools/news_pipeline/summarizer.py` — Solar-Pro2 요약 생성 (LLM fallback 포함)
4. `app/tools/news_pipeline/reference_validator.py` — 요약 내 법령/판례 Cross-Reference 검증

### Phase 4: 저장 + 청킹 레이어 (4단계)
1. DB 저장 로직 (PostgreSQL batch insert)
2. `app/tools/news_pipeline/chunker.py` — 청킹 + JSONL 출력
3. LanceDB `news_chunks` 테이블 생성 + 임베딩

### Phase 5: 오케스트레이션 + CLI (5단계)
1. `app/services/service_function/news_pipeline_service.py` — 파이프라인 오케스트레이터
2. `scripts/news_pipeline/cli.py` — CLI 인터페이스
3. `scripts/news_pipeline/cron_setup.sh` — cron 설정
4. 운영 리포트 로그 생성

### Phase 6: 검증 + 문서화 (6단계)
1. 로컬 1회 실행 테스트 → 샘플 데이터 생성
2. 정적 검증 (ruff + mypy)
3. 운영 리포트 검증
4. README 작성

---

## 7. Risk Assessment

| 리스크 | 영향도 | 대응 방안 |
|--------|--------|----------|
| 법률신문 사이트 구조 변경 | 높음 | 크롤러 모듈화, 셀렉터 설정 외부화, 실패 알림 |
| 네이버 API 일일 할당량 초과 | 중간 | Rate Limiter 적용, 키워드 우선순위 설정 |
| Solar-Pro2 API 비용 증가 | 중간 | 기사당 요약 비용 모니터링, 배치 크기 제한 |
| 크롤링 법적 이슈 (저작권) | **Critical** | robots.txt 준수, 이용약관 확인, 출처 항상 표기, 원문 미저장 옵션, 요약만 저장 모드 검토 |
| PII 유출 (개인정보) | 높음 | PII 마스킹 모듈 추가, 인명/전화번호/주소 정규식 필터 |
| 요약 환각 (hallucination) | 중간 | 프롬프트에 명시적 제약, Cross-Reference 검증 (법령/판례 DB 대조) |
| LLM API 장애 | 중간 | LLM provider fallback 전략 (Solar → OpenAI → 건너뛰기) |
| IP 차단 (크롤링) | 중간 | Rate Limit 엄격 적용, 실패 시 즉시 알림, 백오프 강화 |
| DB 저장 공간 증가 | 낮음 | 90일 이상 기사 아카이브 정책 수립 |

---

## 8. Quality Criteria

### 8.1 수집 품질

| 지표 | 목표 |
|------|------|
| 일일 수집 기사 수 | 50~200건 |
| 중복 제거율 | ≤ 15% |
| 수집 성공률 | ≥ 95% (소스별) |
| 평균 본문 길이 | 1,000자 이상 |

### 8.2 요약 품질

| 지표 | 목표 |
|------|------|
| 요약 생성 성공률 | ≥ 98% |
| 환각 법령/판례 비율 | 0% (검출 시 제거) |
| 평균 요약 길이 | 200~500자 |
| 쟁점 개수 | 3~5개 |

### 8.3 시스템 안정성

| 지표 | 목표 |
|------|------|
| 파이프라인 완료 시간 | ≤ 30분 (일일 기준) |
| 부분 실패 허용 | 한 소스 실패 시 다른 소스 계속 진행 |
| 재시도 | 소스별 최대 3회, exponential backoff |

---

## 9. Non-Functional Requirements

### 9.1 크롤링 예절
- robots.txt 확인 및 준수 (`robotparser` 사용)
- `User-Agent: LegalNewsBot/1.0 (legal-president; +contact@example.com)`
- 요청 간격 최소 2초
- 서버 부하 시간대(KST 09:00~18:00) 회피 권장

### 9.2 보안
- API 키는 `.env`에서만 관리 (하드코딩 금지)
- **PII 마스킹 필수**: 인명(판사/검사/변호사/피고인 실명), 전화번호, 주소 패턴 → `[PERSON]`, `[PHONE]`, `[ADDRESS]`로 대체
- 크롤링 결과에 PII 포함 여부 검사 후 마스킹 적용
- 요약 내 법령/판례 번호 Cross-Reference 검증 (존재하지 않는 번호 제거)

### 9.3 저작권 준수 (Red Team Critical 지적)
- robots.txt 확인 및 준수
- 이용약관 검토 후 수집 가능 범위 확인
- **원문 전문 저장은 내부 분석용**, RAG 제공 시 **요약만 사용** 가능 모드 지원
- 모든 출력에 원문 출처(매체/URL/발행일) 필수 포함
- disclaimer: "본 문서는 기사 요약이며 법령/판례 원문이 아님" 항상 포함

### 9.4 확장성
- 새로운 뉴스 소스 추가는 `BaseNewsSource` 상속으로 가능
- 키워드 목록 외부 설정 파일로 관리
- 청킹 전략 파라미터화

---

## 10. Success Criteria

| 항목 | 완료 조건 |
|------|----------|
| 법률신문 수집 | 1회 이상 성공적으로 기사 수집 및 정제 |
| 네이버 뉴스 수집 | 키워드 기반 기사 수집 및 정제 |
| Solar-Pro2 요약 | 구조화 요약 JSON 정상 생성 |
| PostgreSQL 저장 | `news_articles` 테이블에 데이터 적재 |
| LanceDB 청킹 | `news_chunks` 테이블에 임베딩 벡터 저장 |
| 중복 제거 | 3단계 중복 제거 로직 동작 확인 |
| CLI | 수동 실행/백필/소스별 실행 가능 |
| cron 설정 | 스케줄 실행 스크립트 제공 |
| 운영 리포트 | 일일 리포트 로그 생성 확인 |
| 정적 검증 | ruff check + mypy 통과 |

---

## 11. Team Assignment (7명 Agent Team)

| 역할 | 담당 영역 |
|------|----------|
| **PM** | 전체 조율, 우선순위, 사용자 소통 |
| **백엔드 개발자** | ORM 모델, 마이그레이션, DB 서비스, 파이프라인 오케스트레이션 |
| **AI/ML 엔지니어** | 요약 프롬프트 설계, Solar-Pro2 연동, 청킹/임베딩 전략 |
| **프론트엔드 개발자** | 해당 없음 (백엔드 전용 기능) |
| **UI/UX 디자이너** | 해당 없음 (백엔드 전용 기능) |
| **QA 엔지니어** | 수집 품질 검증, 요약 샘플링, 테스트 전략 |
| **데브옵스 엔지니어** | cron 설정, Docker 통합, 모니터링 |

---

## Revision History

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v0.1 | 2026-02-26 | 초안 작성 |
| v0.2 | 2026-02-26 | Red Team 피드백 반영: 저작권 리스크 대응, PII 마스킹, Cross-Reference 검증, LLM fallback 추가 |
| v0.3 | 2026-02-26 | 컨설팅 보고서 반영: 소스별 보관 정책, LLM fallback 프로바이더 설정 추가. 3중 검증 완료 |
