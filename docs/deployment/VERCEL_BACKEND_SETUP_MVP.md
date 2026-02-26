# Vercel Frontend + Backend 서버 연결 가이드 (MVP)

## 목표 구조

- Frontend: Vercel (`https://<project>.vercel.app`)
- Backend API: Nginx + FastAPI (`https://api.<free-domain>`)
- Database: PostgreSQL (서버 내부 Docker Compose)
- Vector DB: LanceDB local mode (백엔드 컨테이너 볼륨)

## 1. 로컬 개발 동작 방식 (변경 없음)

`frontend/next.config.js`는 기본값으로 `http://127.0.0.1:8000`를 사용합니다.

- 로컬 백엔드 실행 시 별도 설정 없이 동작
- Vercel 배포 시에만 `BACKEND_URL` 환경변수로 덮어씀

## 2. Vercel 환경변수 설정

Vercel Project Settings -> Environment Variables

- `BACKEND_URL=https://api.<your-free-domain>`
- `NEXT_PUBLIC_KAKAO_MAP_API_KEY=<카카오 JS 키>` (지도 기능 사용 시)

주의:
- `BACKEND_URL`이 `http://...`이면 Vercel(HTTPS)에서 Mixed Content로 차단될 수 있음
- 배포 전에 `api` 도메인에 HTTPS가 먼저 준비되어야 함

## 3. 백엔드 서버 환경변수 설정 (`.env.prod`)

최소 필수:
- `POSTGRES_PASSWORD`
- `OPENAI_API_KEY` (AI 기능 사용 시)
- `CORS_ORIGINS=["https://<project>.vercel.app"]`
- `API_DOMAIN=api.<your-free-domain>`

선택:
- `API_KEY` (브라우저 직접 호출 구조라면 일반적으로 비활성화 권장)
- `LANCEDB_*`, `VECTOR_DB`

## 4. 배포 순서 (권장)

1. 서버에서 백엔드/Nginx/Postgres 배포
2. `http://127.0.0.1/health` 확인 (서버 내부 점검)
3. `api` 도메인 DNS 연결
4. HTTPS(TLS) 적용
5. `https://api.<domain>/health` 확인
6. Vercel env에 `BACKEND_URL` 설정 후 프론트 배포
7. 브라우저에서 CORS/Mixed Content/SSE 동작 점검

## 5. Neo4j -> PostgreSQL 이관 이후 주의점

- 이 compose는 Neo4j를 포함하지 않음
- PostgreSQL 그래프 이관 완료 전에는 일부 기능이 제한될 수 있음
- 이관 완료 후 운영 설정에서 Neo4j 관련 env/문서를 정리하는 것을 권장
