[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/U8e5fcmz)

# 법률 서비스 플랫폼

법률 서비스를 위한 모듈형 플랫폼입니다. 각 기능을 독립적인 모듈로 관리하여 유연하게 추가/삭제할 수 있습니다.

## 기술 스택

- **Backend**: FastAPI (Python)
- **Frontend**: Next.js 14 (React, TypeScript)
- **Database**: PostgreSQL
- **AI/ML**: OpenAI, LangChain, ChromaDB (RAG)

## 프로젝트 구조

```
law-3-team/
├── backend/
│   ├── app/
│   │   ├── core/                    # 핵심 설정
│   │   │   ├── config.py            # 환경 설정
│   │   │   └── registry.py          # 모듈 자동 등록
│   │   ├── modules/                 # 기능 모듈 (추가/삭제 용이)
│   │   │   ├── lawyer_finder/       # 위치 기반 변호사 추천
│   │   │   ├── lawyer_stats/         # 변호사 통계 대시보드
│   │   │   ├── case_precedent/      # 판례 검색 및 추천
│   │   │   ├── review_price/        # 후기/가격 비교
│   │   │   ├── storyboard/          # 타임라인 스토리보드
│   │   │   ├── law_study/           # 로스쿨 학습
│   │   │   └── small_claims/        # 소액 소송 에이전트
│   │   ├── common/                  # 공통 유틸리티
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
    │   │   ├── review-price/
    │   │   ├── storyboard/
    │   │   ├── law-study/
    │   │   └── small-claims/
    │   ├── features/                # 기능별 컴포넌트/훅/서비스
    │   │   ├── lawyer-finder/
    │   │   ├── lawyer-stats/
    │   │   ├── case-precedent/
    │   │   ├── review-price/
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
| **lawyer_stats** | 지역별·전문분야별 변호사 분포 및 시장 분석 | 지역별 밀도, 향후 예측(2030/2035/2040), 히트맵 |
| **case_precedent** | RAG 기반 판례 검색 및 AI 질문 | Split View 검색, 필터(문서유형/법원), AI 판례 분석 |
| **review_price** | 상담 후기 및 가격 정보 비교 | 후기 검색, 가격 비교, 필터링 |
| **storyboard** | AI 이미지 생성을 활용한 사건 타임라인 시각화 | 타임라인 생성, 이미지 생성 |
| **law_study** | 로스쿨 학생용 학습 자료, 퀴즈 제공 | 학습 자료, 퀴즈, 오답 노트 |
| **small_claims** | 소액 소송 나홀로 소송 지원 | 4단계 위자드, 증거 체크리스트, AI 서류 생성 |

### 변호사 통계 (lawyer_stats) 상세

- **지역별 현황**: 시/도 → 시/군/구 드릴다운, 변호사 수 및 인구 대비 밀도
- **향후 예측**: 2030/2035/2040년 추계인구 기반 밀도 변화 예측
- **교차 분석**: 지역×전문분야 히트맵 시각화
- **API 엔드포인트**:
  - `GET /api/lawyer-stats/overview` - 전체 현황 요약
  - `GET /api/lawyer-stats/by-region` - 지역별 변호사 수
  - `GET /api/lawyer-stats/density-by-region?year=current` - 지역별 밀도
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
      "latitude": 37.5059,
      "longitude": 127.0329
    }
  ]
}
```

**데이터 출처:** 대한변호사협회 (koreanbar.or.kr)

### 데이터 생성 방법

#### 1단계: 원본 데이터 준비 (`all_lawyers.json`)

대한변호사협회에서 변호사 목록을 크롤링하여 `all_lawyers.json` 파일을 생성합니다.

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
      "phone": "02-1234-5678"
    }
  ]
}
```

> **참고:** 크롤링 스크립트는 별도로 제공되지 않습니다. 직접 크롤러를 작성하거나 수동으로 데이터를 수집해야 합니다.

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
- **PostgreSQL**: 문서 메타데이터 및 전문 텍스트 (검색, 필터링용)
- **ChromaDB**: 문서 임베딩 벡터 (RAG 유사도 검색용)

#### 데이터베이스 마이그레이션

```bash
cd backend

# 마이그레이션 실행 (테이블 생성)
uv run alembic upgrade head

# 현재 마이그레이션 상태 확인
uv run alembic current
```

#### 데이터 로드 및 임베딩 생성

```bash
cd backend

# 1. PostgreSQL에 데이터 로드
uv run python scripts/load_legal_data.py

# 특정 유형만 로드
uv run python scripts/load_legal_data.py --type precedent

# 기존 데이터 삭제 후 재로드
uv run python scripts/load_legal_data.py --reset

# 2. ChromaDB에 임베딩 생성 (OpenAI API 호출)
uv run python scripts/create_embeddings.py

# 특정 유형만 임베딩
uv run python scripts/create_embeddings.py --type constitutional

# 3. 데이터 검증
uv run python scripts/validate_data.py
```

**주의사항:**
- 임베딩 생성 시 OpenAI API 비용 발생 (~$2 for 108K documents)
- 전체 임베딩 생성에 상당한 시간 소요
- `--batch-size` 옵션으로 API 호출 배치 크기 조정 가능

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

#### PostgreSQL 설치 및 데이터베이스 생성
```bash
# macOS (Homebrew)
brew install postgresql@15
brew services start postgresql@15

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
| `CHROMA_PERSIST_DIR` | ChromaDB 저장 경로 | `./data/chroma` |
| `CHROMA_COLLECTION_NAME` | ChromaDB 컬렉션 이름 | `legal_documents` |
| `EMBEDDING_MODEL` | OpenAI 임베딩 모델 | `text-embedding-3-small` |
| `EMBEDDING_BATCH_SIZE` | 임베딩 API 배치 크기 | `100` |
| `USE_LOCAL_EMBEDDING` | 로컬 임베딩 사용 여부 (무료) | `true` |
| `LOCAL_EMBEDDING_MODEL` | 로컬 임베딩 모델 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |

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

# ChromaDB (벡터 저장소)
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=legal_documents

# 임베딩 설정
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BATCH_SIZE=100

# 로컬 임베딩 (무료, 권장)
USE_LOCAL_EMBEDDING=true
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
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
│                   │ ChromaDB Data│                      │
│                   └──────────────┘                      │
│                                                          │
│   Secrets: AWS Secrets Manager                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```
