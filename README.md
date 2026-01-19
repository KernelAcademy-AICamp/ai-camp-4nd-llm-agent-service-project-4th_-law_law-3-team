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
    │   │   ├── case-precedent/
    │   │   ├── review-price/
    │   │   ├── storyboard/
    │   │   ├── law-study/
    │   │   └── small-claims/
    │   ├── features/                # 기능별 컴포넌트/훅/서비스
    │   │   ├── lawyer-finder/
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

| 모듈 | 설명 |
|------|------|
| **lawyer_finder** | 카카오맵 API를 활용한 위치 기반 변호사 검색 |
| **case_precedent** | RAG 기반 상황 분석 및 관련 판례 제공 |
| **review_price** | 상담 후기 및 가격 정보 비교 (치과 비용 비교 앱 컨셉) |
| **storyboard** | AI 이미지 생성을 활용한 사건 타임라인 시각화 |
| **law_study** | 로스쿨 학생용 학습 자료, 퀴즈 제공 |
| **small_claims** | 소액 소송 나홀로 소송 지원 (내용증명, 지급명령, 소액심판) |

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

### 1. 사전 준비

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
| `CORS_ORIGINS` | 허용할 CORS 출처 (JSON 배열) | `["http://localhost:3000"]` |
| `KAKAO_MAP_API_KEY` | 카카오맵 JavaScript API 키 (변호사 찾기 기능) | - |
| `KAKAO_REST_API_KEY` | 카카오 REST API 키 (주소 검색 등) | - |
| `ENABLED_MODULES` | 활성화할 모듈 목록 (빈 배열이면 모두 활성화) | `[]` |

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
CORS_ORIGINS=["http://localhost:3000"]
ENABLED_MODULES=[]
```
