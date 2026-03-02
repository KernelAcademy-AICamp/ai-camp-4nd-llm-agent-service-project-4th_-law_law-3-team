[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/U8e5fcmz)

# 법률 서비스 플랫폼

법률 서비스를 위한 모듈형 플랫폼입니다. 각 기능을 독립적인 모듈로 관리하여 유연하게 추가/삭제할 수 있습니다.

## 기술 스택

- **Backend**: FastAPI (Python)
- **Frontend**: Next.js 14 (React, TypeScript)
- **Database**: PostgreSQL, Neo4j (Graph DB)
- **Vector DB**: LanceDB (RAG, 1문서=1벡터)
- **AI/ML**: Solar (Upstage), LangGraph, LangSmith (트레이싱)
- **Embedding**: KURE-v1 (로컬, 1024차원) / OpenAI (선택)
- **Reranker**: bge-reranker-v2-m3-ko (Cross-encoder, 한국어 특화)

## 프로젝트 구조

```
law-3-team/
├── backend/
│   ├── app/
│   │   ├── api/router/              # 통합 API (채팅 등)
│   │   │   └── chat.py              # /api/chat 엔드포인트
│   │   ├── core/                    # 핵심 인프라
│   │   │   ├── config.py            # 환경 설정
│   │   │   ├── database.py          # DB 연결
│   │   │   ├── errors.py            # 공통 예외
│   │   │   └── registry.py          # 모듈 자동 등록
│   │   ├── multi_agent/             # LangGraph 멀티 에이전트 시스템
│   │   │   ├── graph.py             # StateGraph 빌드/컴파일
│   │   │   ├── nodes.py             # router_node + 에이전트 노드
│   │   │   ├── router.py            # RulesRouter, AgentType
│   │   │   ├── state.py             # ChatState TypedDict
│   │   │   ├── agents/              # 에이전트 구현체
│   │   │   ├── subgraphs/           # 서브그래프 (소액소송, 모의법정)
│   │   │   └── schemas/             # 스키마
│   │   ├── services/                # 비즈니스 로직
│   │   │   ├── rag/                 # RAG 검색 서비스
│   │   │   └── service_function/    # 통합 서비스 함수
│   │   ├── tools/                   # 외부 도구 클라이언트
│   │   │   ├── llm/                 # LLM (Solar)
│   │   │   ├── vectorstore/         # LanceDB
│   │   │   └── graph/               # Neo4j
│   │   ├── modules/                 # 독립 API 모듈 (자동 등록)
│   │   │   ├── lawyer_finder/       # 위치 기반 변호사 추천
│   │   │   ├── lawyer_stats/        # 변호사 통계 대시보드
│   │   │   ├── case_precedent/      # 판례 검색 및 추천
│   │   │   ├── storyboard/          # 타임라인 스토리보드
│   │   │   ├── law_study/           # 로스쿨 학습
│   │   │   └── small_claims/        # 소액 소송 에이전트
│   │   ├── models/                  # ORM 모델
│   │   └── main.py                  # FastAPI 앱 진입점
│   ├── tests/
│   ├── pyproject.toml               # uv 패키지 설정
│   └── .env.example
│
└── frontend/
    ├── src/
    │   ├── app/                     # Next.js App Router 페이지
    │   │   ├── lawyer-finder/
    │   │   ├── lawyer-stats/
    │   │   ├── case-precedent/
    │   │   ├── storyboard/
    │   │   ├── law-study/
    │   │   └── small-claims/
    │   ├── features/                # 기능별 컴포넌트/훅/서비스
    │   │   ├── lawyer-finder/
    │   │   ├── lawyer-stats/
    │   │   ├── case-precedent/
    │   │   ├── storyboard/
    │   │   ├── law-study/
    │   │   └── small-claims/
    │   ├── components/              # 공통 컴포넌트
    │   │   ├── ui/
    │   │   └── shared/
    │   ├── lib/                     # 유틸리티, API 클라이언트
    │   └── styles/
    ├── package.json
    └── tsconfig.json
```

## 모듈 설명

| 모듈 | 설명 | 주요 기능 |
|------|------|----------|
| **lawyer_finder** | 카카오맵 API를 활용한 위치 기반 변호사 검색 | 지도 기반 검색, 반경 설정, 변호사 상세 정보 |
| **lawyer_stats** | 지역별·전문분야별 변호사 분포 및 시장 분석 | 공급(지역별 밀도, 향후 예측), 수요(법원별 사건 수, 부담지수), 히트맵 |
| **case_precedent** | RAG 기반 판례 검색 및 AI 질문 | Split View 검색, 필터(문서유형/법원), AI 판례 분석 |
| **storyboard** | AI 이미지 생성을 활용한 사건 타임라인 시각화 | 타임라인 생성, 이미지 생성 |
| **law_study** | 로스쿨 학생용 학습 자료, 퀴즈 제공 | 학습 자료, 퀴즈, 오답 노트 |
| **small_claims** | 소액 소송 나홀로 소송 지원 | 4단계 위자드, 증거 체크리스트, AI 서류 생성 |

### 변호사 통계 (lawyer_stats) 상세

- **공급 분석**: 시/도 → 시/군/구 드릴다운, 변호사 수 및 인구 대비 밀도
- **향후 예측**: 2030/2035/2040년 추계인구 기반 밀도 변화 예측
- **수요 분석**: 법원별 사건 접수 수 마커 시각화, 부담지수(변호사 1인당 사건 수) 중앙값 비교
- **교차 분석**: 지역×전문분야 히트맵 시각화
- **API 엔드포인트**:
  - `GET /api/lawyer-stats/overview` - 전체 현황 요약
  - `GET /api/lawyer-stats/by-region` - 지역별 변호사 수
  - `GET /api/lawyer-stats/density-by-region?year=current` - 지역별 밀도
  - `GET /api/lawyer-stats/demand?category=민사&year=2024` - 사건 수요 통계
  - `GET /api/lawyer-stats/cross-analysis` - 지역×전문분야 교차 분석

### 판례 검색 (case_precedent) 상세

- **Split View 레이아웃**: 왼쪽 검색 패널 + 오른쪽 상세 패널
- **검색 기능**: 키워드 검색, 문서 유형 필터 (판례/헌재결정), 법원 필터
- **AI 질문**: 선택한 판례에 대해 AI에게 질문하고 답변 받기
- **API 엔드포인트**:
  - `GET /api/case-precedent/precedents` - 판례 검색
  - `GET /api/case-precedent/precedents/{id}` - 판례 상세
  - `POST /api/case-precedent/precedents/{id}/ask` - AI 질문

### 소액소송 도우미 (small_claims) 상세

- **4단계 위자드**: 분쟁유형 선택 → 사건정보 입력 → 증거 체크 → 서류 생성
- **지원 분쟁 유형**:
  - 물품대금 미지급
  - 중고거래 사기
  - 임대차 보증금 미반환
  - 용역대금 미지급
  - 임금 체불
- **증거 체크리스트**: 분쟁 유형별 필수/선택 증거 안내
- **AI 서류 생성**: 내용증명, 지급명령신청서, 소액심판청구서 자동 작성
- **관련 판례**: 분쟁 유형별 유사 판례 사이드바
- **API 엔드포인트**:
  - `GET /api/small-claims/evidence-checklist/{type}` - 증거 체크리스트
  - `GET /api/small-claims/dispute-types` - 분쟁 유형 목록
  - `POST /api/small-claims/generate-document` - 서류 생성
  - `GET /api/small-claims/related-cases/{type}` - 관련 판례

## 데이터

### 변호사 데이터 (lawyer_finder 모듈)

변호사 찾기 기능은 JSON 파일 기반의 변호사 데이터를 사용합니다.

```
data/
├── lawyers_with_coords.json   # 지오코딩된 변호사 데이터 (위경도 포함)
└── geocode_failed.json        # 지오코딩 실패 목록
```

**데이터 구조:**
```json
{
  "metadata": {
    "source": "koreanbar.or.kr",
    "crawled_at": "2026-01-16T12:57:41",
    "total_count": 8506,
    "total_geocoded": 6574
  },
  "lawyers": [
    {
      "name": "홍길동",
      "status": "개업",
      "office_name": "법무법인 예시",
      "address": "서울 강남구 테헤란로 123",
      "phone": "02-1234-5678",
      "specialties": ["민사법", "부동산"],
      "latitude": 37.5059,
      "longitude": 127.0329
    }
  ]
}
```

**데이터 출처:** 대한변호사협회 (koreanbar.or.kr)

### 데이터 생성 방법

#### 1단계: 원본 데이터 준비 (`all_lawyers.json`)

별도 저장소에서 생성된 `all_lawyers.json` 파일을 프로젝트 루트에 배치합니다.
이 파일에는 변호사 기본 정보와 전문분야가 포함되어 있습니다.

```json
{
  "metadata": {
    "source": "koreanbar.or.kr",
    "crawled_at": "2026-01-16T12:57:41"
  },
  "lawyers": [
    {
      "name": "홍길동",
      "status": "개업",
      "office_name": "법무법인 예시",
      "address": "서울 강남구 테헤란로 123",
      "phone": "02-1234-5678",
      "specialties": ["민사법", "부동산"]
    }
  ]
}
```

> **참고:** 변호사 데이터 수집 코드(크롤링, 전문분야)는 별도 저장소에서 관리됩니다. 이 프로젝트에서는 좌표 변환(지오코딩)만 수행합니다.

#### 2단계: 지오코딩 실행 (주소 → 좌표 변환)

```bash
cd backend

# 카카오 REST API 키가 .env에 설정되어 있어야 함
uv run python scripts/geocode_lawyers.py

# 또는 API 키를 직접 전달
uv run python scripts/geocode_lawyers.py --api-key YOUR_KAKAO_REST_API_KEY
```

**입출력 파일:**
| 파일 | 설명 |
|------|------|
| `all_lawyers.json` (입력) | 원본 변호사 데이터 |
| `data/lawyers_with_coords.json` (출력) | 좌표가 추가된 데이터 |
| `data/geocode_failed.json` (출력) | 지오코딩 실패 목록 |

```bash
# 실패 항목 재시도
uv run python scripts/geocode_lawyers.py --retry-failed

# 데이터 상태 확인
uv run python scripts/geocode_lawyers.py --stats

# 입출력 경로 지정
uv run python scripts/geocode_lawyers.py --input path/to/input.json --output path/to/output.json
```

**주의사항:**
- 카카오 REST API 키 필요 (발급: https://developers.kakao.com)
- 약 8,000건 처리 시 API 호출 제한에 주의 (초당 10건)
- 동일 주소는 캐싱하여 중복 호출 방지

> **참고:** 데이터 파일이 없으면 변호사 찾기 기능이 빈 결과를 반환합니다.

### 인구 데이터 (lawyer_stats 모듈)

변호사 통계의 인구 대비 밀도 및 향후 예측 기능에 사용됩니다.

```
data/
└── population.json    # 인구 데이터 (현재 + 추계)
```

**데이터 구조:**
```json
{
  "meta": {
    "source": "KOSIS e지방지표 (https://kosis.kr)",
    "source_current": "주민등록인구(시도/시/군/구)",
    "source_prediction": "추계인구(시/군/구)",
    "current_year": 2025,
    "prediction_years": [2030, 2035, 2040]
  },
  "data": {
    "서울 강남구": {
      "current": 556330,
      "2030": 541234,
      "2035": 528901,
      "2040": 515678
    }
  }
}
```

**데이터 업데이트:**
```bash
cd backend

# 1. KOSIS에서 CSV 다운로드
#    -> e지방지표(주제별) -> 인구
#    -> 주민등록인구(시도/시/군/구) 또는 추계인구(시/군/구)
#    -> 조회 조건 '합계'로 다운로드 (CSV UTF-8)

# 2. CSV 파일을 data/ 폴더에 저장
#    - data/population_YYYYMM.csv (현재 인구)
#    - data/population_pred.csv (추계인구)

# 3. JSON 변환 스크립트 실행
python scripts/update_population.py
```

### 법률 데이터 (판례, 헌재결정례, 행정심판례, 법령해석례)

RAG 기반 검색 및 로스쿨 학습 기능을 위한 법률 문서 데이터입니다.

```
data/law_data/
├── precedents_full.json       # 일반 판례 (분할: 1~5.json)
├── constitutional_full.json   # 헌법재판소 결정례
├── administration_full.json   # 행정심판례
└── legislation_full.json      # 법령해석례
```

**데이터 규모:**
| 유형 | 파일 크기 | 레코드 수 |
|------|----------|----------|
| 판례 (precedent) | ~1GB | 29,120건 |
| 헌재결정례 (constitutional) | 24MB | 36,781건 |
| 행정심판례 (administration) | 444MB | 34,258건 |
| 법령해석례 (legislation) | 80MB | 8,597건 |
| **합계** | ~1.5GB | ~108,756건 |

**저장 구조:**
- **PostgreSQL**: 문서 메타데이터 및 전문 텍스트 + FTS 인덱스 (검색, 필터링용)
- **LanceDB**: 문서 임베딩 벡터 (RAG 유사도 검색용, 1문서=1벡터)

#### 데이터베이스 마이그레이션

```bash
cd backend

# 마이그레이션 실행 (테이블 생성)
uv run alembic upgrade head

# 현재 마이그레이션 상태 확인
uv run alembic current
```

#### 인제스트 파이프라인 (PostgreSQL + FTS + LanceDB)

config-driven 파이프라인으로 19개 데이터 타입별 설정(`scripts/ingest/types/`)을 정의하면 PostgreSQL + FTS + LanceDB를 일괄 처리합니다.
타입별 저장 구조 상세는 `backend/scripts/ingest/ingest.md` 참조.

**`--type` 타입명 목록 (19개, 총 ~423,924건):**

| 타입명 | 데이터 | 건수 |
|--------|--------|------|
| `law` | 법령 | 5,548 |
| `precedent` | 판례 | 92,055 |
| `admin_rule` | 행정규칙 | 5,258 |
| `constitutional` | 헌재결정례 | 31,718 |
| `administration` | 행정심판례 | 34,254 |
| `legislation` | 법령해석례 | 8,597 |
| `treaty` | 조약 | 3,589 |
| `interpretation_ministry` | 부처해석례 (28개 부처) | 37,325 |
| `special_admin_appeal` | 특별행정심판례 (2개 기관) | 148,778 |
| `dec_privacy` ~ `dec_securities` | 위원회 결정문 (10개) | 56,802 |

**1. DB 적재 데이터 소스 위치** — 프로젝트 루트 `data/`:

```
data/
├── law_v1.json                    # 법령
├── precedents_v1.json             # 판례
├── admin_rule_v1.json             # 행정규칙
├── constitutional_v1.json         # 헌재결정례
├── administration_v1.json         # 행정심판례
├── legislation_v1.json            # 법령해석례
├── treaty_v1.json                 # 조약
├── interpretation_ministry/       # 부처해석례 (28개 부처별 JSON)
├── special_admin_appeal/          # 특별행정심판례 (2개 기관별 JSON)
└── decisions_committee/           # 위원회 결정문 (10개 위원회별 JSON)
```

**2. 사전 조건:**
- PostgreSQL 실행: `docker compose up -d postgres`
- Alembic 마이그레이션: `uv run alembic upgrade head`
- MeCab 시스템 패키지: `mecab`, `libmecab-dev`, `mecab-ko-dic` (FTS tsvector용, 미설치 시 에러 발생)
- 환경변수: `DATABASE_URL=postgresql://lawuser:lawpassword@localhost:5432/lawdb` (`backend/.env`)
- 벡터 단계 추가: 임베딩 모델 다운로드 + PyTorch 설치

**3. CLI 사용법:**

```bash
cd backend

# 전체 타입 × 전체 파이프라인 (최초 적재 시)
uv run python -m scripts.ingest.cli --type all --step all --reset

# 특정 타입 전체 파이프라인
uv run python -m scripts.ingest.cli --type precedent --step all --reset

# 단계별 실행
uv run python -m scripts.ingest.cli --type precedent --step db       # PostgreSQL + FTS
uv run python -m scripts.ingest.cli --type precedent --step vector   # LanceDB 벡터
uv run python -m scripts.ingest.cli --type precedent --step fts      # FTS만 재빌드 (토크나이저 변경 후)
uv run python -m scripts.ingest.cli --type precedent --step index    # ANN 인덱스만 재빌드

# 통계 / 검증
uv run python -m scripts.ingest.cli --type all --stats               # 전체 타입
uv run python -m scripts.ingest.cli --type precedent --verify        # 특정 타입
```

> CLI 전체 옵션(`--device`, `--profile`, `--batch-size`, `--source`, `--no-cache` 등) 상세는 `backend/scripts/CLAUDE.md` 인제스트 섹션 참조.

**새 데이터 타입 추가** (7단계):
1. `app/models/ingest/new_type_document.py` 생성 — ORM 테이블 정의 (ai_summary 포함)
2. `app/models/ingest/__init__.py` — import + `__all__` 추가
3. `app/models/__init__.py` — import + `__all__` 추가
4. `alembic/env.py` — import 추가
5. `alembic/versions/NNN_*.py` — 마이그레이션 작성
6. `scripts/ingest/types/_template.py`를 복사하여 `types/new_type.py` 생성 (TODO 주석 따라 수정, 자동 등록)
7. JSON 소스 파일을 `data/` 하위에 배치

#### LanceDB 임베딩 생성

```bash
cd backend

# 임베딩 모델 다운로드 (약 2.3GB, 최초 1회)
uv run python scripts/download_models.py

# 로컬 임베딩 생성 (하드웨어 자동 감지)
uv run --no-sync python scripts/local_lancedb_embeddings.py --type all --reset

# 통계 확인
uv run --no-sync python scripts/local_lancedb_embeddings.py --stats
```

**주의사항:**
- PyTorch 환경별 수동 설치 필요 (`uv pip install torch`)
- `--no-sync` 플래그 필수 (torch 버전 유지)
- GPU VRAM에 따라 batch_size 자동 설정

## 모듈 추가/삭제 방법

### 스크립트로 추가 (권장)

```bash
# 새 모듈 추가 (Backend + Frontend 자동 생성)
python scripts/add_module.py <모듈명> "<설명>"

# 예시
python scripts/add_module.py document_generator "법률 문서 자동 생성"

# 모듈 삭제
python scripts/add_module.py remove <모듈명>
```

스크립트가 자동으로 생성하는 것:
- `backend/app/modules/<모듈명>/` (router, service, schema, model)
- `frontend/src/app/<모듈명>/page.tsx`
- `frontend/src/features/<모듈명>/` (services, components, hooks, types)
- `frontend/src/lib/modules.ts` 업데이트
- `frontend/src/lib/api.ts` endpoints 업데이트

### 수동으로 추가

#### Backend
1. `backend/app/modules/` 아래에 새 폴더 생성
2. `router/__init__.py`에 `router = APIRouter()` 정의
3. 서버 재시작 시 자동 등록됨

#### Frontend
1. `src/app/<모듈명>/page.tsx` 생성
2. `src/features/<모듈명>/services/index.ts` 생성
3. `src/lib/modules.ts`에 모듈 정보 추가
4. `src/lib/api.ts`에 endpoint 추가

### 모듈 비활성화

**Backend** - `.env` 파일:
```env
ENABLED_MODULES=["lawyer_finder","small_claims"]
```
빈 배열(`[]`)이면 모든 모듈 활성화

**Frontend** - `src/lib/modules.ts`:
```typescript
{ id: 'some-module', enabled: false, ... }
```

## 실행 방법

### 방법 1: Docker로 실행 (권장)

Docker를 사용하면 PostgreSQL 설치 없이 빠르게 개발 환경을 구축할 수 있습니다.

#### 1. PostgreSQL 컨테이너 시작
```bash
# PostgreSQL 컨테이너 시작
docker-compose up -d postgres

# 컨테이너 상태 확인
docker-compose ps

# DB 연결 확인
docker-compose exec postgres psql -U lawuser -d lawdb -c "SELECT 1;"
```

#### 2. 환경 변수 설정
```bash
cd backend
cp .env.example .env
# .env 파일 편집하여 DATABASE_URL 확인
# DATABASE_URL=postgresql://lawuser:lawpassword@localhost:5432/lawdb
```

#### 3. Backend 실행
```bash
cd backend
uv sync
uv run alembic upgrade head     # 마이그레이션 실행
uv run uvicorn app.main:app --reload
```

#### 4. Frontend 실행
```bash
cd frontend
npm install
npm run dev
```

#### Docker 명령어 요약
```bash
docker-compose up -d postgres     # PostgreSQL 시작
docker-compose logs -f postgres   # 로그 확인
docker-compose stop postgres      # 중지
docker-compose down               # 중지 및 삭제 (데이터 유지)
docker-compose down -v            # 중지 및 볼륨까지 삭제
```

---

### 방법 2: 로컬 PostgreSQL 설치

> **⚠ 주의**: 이 프로젝트는 **PostgreSQL 17 + pg_textsearch** 확장이 필요합니다.
> pg_textsearch는 BM25 키워드 검색에 사용되며, C 소스 빌드가 필요한 확장입니다.
> 로컬 설치 시 pg_textsearch를 별도로 빌드해야 하므로 **방법 1 (Docker)을 강력히 권장**합니다.
> Docker 이미지(`docker/postgres/Dockerfile`)에는 pg_textsearch가 포함되어 있습니다.

#### PostgreSQL 설치 및 데이터베이스 생성
```bash
# macOS (Homebrew)
brew install postgresql@17
brew services start postgresql@17

# ⚠ pg_textsearch 확장은 별도 소스 빌드 필요
# 상세: docker/postgres/Dockerfile 참조

# 데이터베이스 생성
createdb lawdb

# 또는 psql로 접속하여 생성
psql postgres
CREATE DATABASE lawdb;
CREATE USER lawuser WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE lawdb TO lawuser;
\q
```

#### API 키 발급

| 서비스 | 발급 URL | 용도 |
|--------|----------|------|
| **카카오 개발자** | https://developers.kakao.com | 지도 API (변호사 찾기) |
| **OpenAI** | https://platform.openai.com/api-keys | AI 기능 (판례 분석 등) |

**카카오 API 키 발급 절차:**
1. [카카오 개발자](https://developers.kakao.com) 접속 → 로그인
2. 내 애플리케이션 → 애플리케이션 추가
3. 앱 키 → **JavaScript 키** (`KAKAO_MAP_API_KEY`)
4. 앱 키 → **REST API 키** (`KAKAO_REST_API_KEY`)
5. 플랫폼 → Web → 사이트 도메인에 `http://localhost:3000` 추가

### 2. Backend 실행

```bash
# uv 설치 (아직 없다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd backend
uv sync                   # 의존성 설치 (가상환경 자동 생성)
cp .env.example .env      # 환경변수 파일 생성
# .env 파일을 열어 API 키와 DB 정보 입력

uv run uvicorn app.main:app --reload  # 서버 실행 (localhost:8000)
```

#### 개발 의존성 포함 설치
```bash
uv sync --dev             # pytest, ruff, mypy 포함
```

### 3. Frontend 실행

```bash
cd frontend
npm install
npm run dev               # 개발 서버 (localhost:3000)
```

### 4. 접속 확인

- **Frontend**: http://localhost:3000
- **Backend API 문서**: http://localhost:8000/docs (Swagger UI)

## 환경 변수

### 필수 환경 변수

| 변수 | 설명 | 예시 |
|------|------|------|
| `DATABASE_URL` | PostgreSQL 연결 문자열 | `postgresql://user:password@localhost:5432/lawdb` |
| `OPENAI_API_KEY` | OpenAI API 키 | `sk-...` |

### 선택 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `APP_NAME` | 애플리케이션 이름 | `Law Platform API` |
| `DEBUG` | 디버그 모드 | `true` |
| `ENVIRONMENT` | 환경 (development/docker/production) | `development` |
| `CORS_ORIGINS` | 허용할 CORS 출처 (JSON 배열) | `["http://localhost:3000"]` |
| `KAKAO_MAP_API_KEY` | 카카오맵 JavaScript API 키 (변호사 찾기 기능) | - |
| `KAKAO_REST_API_KEY` | 카카오 REST API 키 (주소 검색 등) | - |
| `ENABLED_MODULES` | 활성화할 모듈 목록 (빈 배열이면 모두 활성화) | `[]` |
| `LANCEDB_URI` | LanceDB 데이터 경로 | `./lancedb_data` |
| `LANCEDB_TABLE_NAME` | LanceDB 테이블명 | `legal_chunks` |
| `USE_LOCAL_EMBEDDING` | 로컬 임베딩 사용 여부 (무료) | `true` |
| `LOCAL_EMBEDDING_MODEL` | 로컬 임베딩 모델 | `nlpai-lab/KURE-v1` |
| `LANGCHAIN_TRACING_V2` | LangSmith 트레이싱 활성화 | `false` |
| `LANGCHAIN_PROJECT` | LangSmith 프로젝트명 | `law-platform` |
| `LANGCHAIN_API_KEY` | LangSmith API 키 | - |

### .env 파일 예시

```env
# 필수
DATABASE_URL=postgresql://lawuser:your_password@localhost:5432/lawdb
OPENAI_API_KEY=sk-your-openai-api-key

# 카카오 API (변호사 찾기 기능 사용 시 필요)
KAKAO_MAP_API_KEY=your_javascript_key
KAKAO_REST_API_KEY=your_rest_api_key

# 선택
DEBUG=true
ENVIRONMENT=development
CORS_ORIGINS=["http://localhost:3000"]
ENABLED_MODULES=[]

# LanceDB (벡터 저장소)
LANCEDB_URI=./lancedb_data
LANCEDB_TABLE_NAME=legal_chunks

# 로컬 임베딩 (무료, 권장)
USE_LOCAL_EMBEDDING=true
LOCAL_EMBEDDING_MODEL=nlpai-lab/KURE-v1

# LangSmith 트레이싱 (선택)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=law-platform
LANGCHAIN_API_KEY=lsv2_pt_your-langsmith-api-key
```

## Docker 프로덕션 배포

### 전체 서비스 빌드 및 실행

```bash
# 프로덕션 환경 변수 설정
export POSTGRES_PASSWORD=secure_password_here
export OPENAI_API_KEY=sk-your-key
export KAKAO_MAP_API_KEY=your-key
export KAKAO_REST_API_KEY=your-key

# 전체 서비스 빌드 및 시작
docker-compose -f docker-compose.prod.yml up -d --build

# 상태 확인
docker-compose -f docker-compose.prod.yml ps

# 로그 확인
docker-compose -f docker-compose.prod.yml logs -f
```

### 개별 서비스 빌드

```bash
# Backend 이미지 빌드
docker build -f docker/backend/Dockerfile.prod -t law-backend:latest ./backend

# Frontend 이미지 빌드
docker build -f docker/frontend/Dockerfile.prod -t law-frontend:latest ./frontend
```

### Docker 파일 구조

```
docker/
├── backend/
│   ├── Dockerfile           # 개발용
│   └── Dockerfile.prod      # 프로덕션용 (multi-stage)
├── frontend/
│   ├── Dockerfile           # 개발용
│   └── Dockerfile.prod      # 프로덕션용 (multi-stage)
└── postgres/
    └── init.sql             # DB 초기화 스크립트
```

## AWS 배포 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                        AWS Cloud                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐ │
│   │  Vercel  │    │ ECS Fargate  │    │     RDS      │ │
│   │ Frontend │───▶│   Backend    │───▶│  PostgreSQL  │ │
│   └──────────┘    └──────────────┘    └──────────────┘ │
│                          │                              │
│                          ▼                              │
│                   ┌──────────────┐                      │
│                   │     EFS      │                      │
│                   │ LanceDB Data │                      │
│                   └──────────────┘                      │
│                                                          │
│   Secrets: AWS Secrets Manager                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```
