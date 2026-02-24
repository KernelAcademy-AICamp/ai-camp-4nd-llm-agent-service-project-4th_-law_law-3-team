# MVP 배포 런북 (Vercel Frontend + 서버 Backend)

이 문서는 `Vercel(프론트)` + `단일 서버 Docker Compose(백엔드/Nginx/Postgres)` 기준의 MVP 배포 절차를 정리합니다.

## 1. 범위 및 제한

- 기본 스택: `postgres + backend + nginx`
- `lancedb`는 선택 사항 (profile `lancedb`)
- `neo4j`는 이 MVP prod compose에 포함하지 않음
- 그래프 의존 기능은 부분 제한될 수 있음
- 프론트는 Vercel에 배포하므로 이 compose에 포함하지 않음

## 2. 사전 준비

- Linux 서버 1대 (Docker, Docker Compose 설치 완료)
- 무료/유료 도메인(DNS) 준비 (`api.<domain>` 권장)
- 프로젝트 코드 배치
- 필수 비밀값 발급
  - `OPENAI_API_KEY` (AI 기능 사용 시)
  - `KAKAO_MAP_API_KEY`, `KAKAO_REST_API_KEY` (지도/지오코딩 기능 사용 시)
  - 강한 `POSTGRES_PASSWORD`

## 3. 배포 파일 준비

```bash
cp .env.prod.example .env.prod
```

`.env.prod`에서 최소 확인 항목:
- `POSTGRES_PASSWORD`
- `CORS_ORIGINS` (Vercel 프론트 도메인)
- `API_DOMAIN` (예: `api.your-app.duckdns.org`)
- `OPENAI_API_KEY`
- `KAKAO_*` 키 (필요 기능 사용 시)

## 3.1 Vercel 환경변수 준비

Vercel 프로젝트(Environment Variables)에 최소 다음 값 설정:

- `BACKEND_URL=https://api.<your-free-domain>`
- `NEXT_PUBLIC_KAKAO_MAP_API_KEY=<카카오 JS 키>` (지도 UI 사용 시)

로컬 개발은 `frontend/next.config.js`의 기본값(`http://127.0.0.1:8000`)으로 계속 동작합니다.

## 4. 기동 전 검증

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml config
```

옵션 LanceDB 사용 시:

```bash
docker compose --profile lancedb --env-file .env.prod -f docker-compose.prod.yml config
```

## 5. 서비스 기동 (서버)

기본 스택 기동:

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
```

LanceDB 포함 기동:

```bash
docker compose --profile lancedb --env-file .env.prod -f docker-compose.prod.yml up -d --build
```

## 6. 로그 확인 (서버)

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f backend
```

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f nginx
```

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f postgres
```

LanceDB 사용 시:

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f lancedb
```

## 7. 헬스체크 및 스모크 테스트

### 7.1 헬스체크

백엔드 직접 포트는 외부 미노출(Compose 내부) 기준입니다. Nginx 경유로 확인합니다.

```bash
curl http://127.0.0.1/health
```

도메인 연결 후:

```bash
curl https://api.<your-free-domain>/health
```

### 7.2 최소 스모크 테스트 (예시)

- Vercel 프론트 메인 페이지 로드 성공
- Vercel 프론트에서 `/api/*` 요청이 `BACKEND_URL`로 프록시됨 (rewrite 동작)
- `/health` 응답에 `status=healthy` 포함
- 대표 모듈 API 2개 확인 (예: `lawyer-finder`, `small-claims`)
- 필요 시 `/media/*` 경로 접근 확인
- Vercel 배포 환경에서 Mixed Content 오류 없음 (`https` -> `https`)

주의:
- `neo4j` 미포함이므로 그래프 기능 일부는 제한될 수 있음
- 외부 API 키 미설정 시 AI/지도 기능은 정상 동작하지 않을 수 있음
- Vercel 연결 전 `api` 도메인에 HTTPS(TLS) 적용 필수

## 8. 재배포 / 재시작

코드 변경 후 재배포:

```bash
git pull
```

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
```

특정 서비스만 재시작:

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml restart backend nginx
```

## 9. 종료

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml down
```

데이터 볼륨 유지됨 (`postgres_data`, `chroma_data`, `lancedb_data`).

## 10. 최소 롤백 절차

1. 이전 정상 커밋으로 체크아웃
2. 동일 `.env.prod` 유지
3. 재빌드/재기동

```bash
git checkout <previous-good-commit>
```

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
```

## 11. 운영 메모 (MVP)

- `RATE_LIMIT_STORAGE_URI=memory://` 는 단일 인스턴스에서만 일관적
- 운영 도메인 공개 시 Nginx/Caddy로 TLS 종료 권장
- Vercel `BACKEND_URL`은 반드시 `https://api.<domain>` 이어야 함 (HTTP면 Mixed Content로 차단될 수 있음)
- Nginx 템플릿은 `API_DOMAIN` 값을 사용 (`docker/nginx/conf.d/api.conf.template`)
