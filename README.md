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

### Backend (uv 사용)

```bash
# uv 설치 (아직 없다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd backend
uv sync                   # 의존성 설치 (가상환경 자동 생성)
cp .env.example .env      # 환경변수 설정
uv run uvicorn app.main:app --reload
```

#### 개발 의존성 포함 설치
```bash
uv sync --dev             # pytest, ruff, mypy 포함
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API 문서

서버 실행 후 `http://localhost:8000/docs`에서 Swagger UI 확인

## 환경 변수

| 변수 | 설명 |
|------|------|
| `DATABASE_URL` | PostgreSQL 연결 문자열 |
| `KAKAO_MAP_API_KEY` | 카카오맵 API 키 |
| `OPENAI_API_KEY` | OpenAI API 키 |
| `ENABLED_MODULES` | 활성화할 모듈 목록 (JSON 배열) |
