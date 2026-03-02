---
name: docker-containerization
description: |
  Docker Compose 기반 로컬 개발환경 구성, 프로덕션 배포, 멀티스테이지 빌드 가이드.
  PostgreSQL + LanceDB + FastAPI + Nginx 컨테이너 관리.
  Docker 설정 변경, 새 서비스 추가, Dockerfile 수정, 배포 구성, 컨테이너 디버깅 시 반드시 사용.
  WSL2 환경, 프로덕션 빌드 최적화, 헬스체크, 네트워크 설정 시에도 사용.
---

# Docker & Containerization

법률 서비스 플랫폼의 Docker 기반 인프라 구성 및 운영 가이드.

## 아키텍처 개요

### 개발 환경 (docker-compose.yml)

```
┌─────────────────────────────────────────────┐
│ 로컬 머신                                     │
│  ┌─────────┐  ┌─────────┐  ┌──────────────┐ │
│  │ Backend │  │Frontend │  │ LanceDB svc  │ │
│  │ (로컬)  │  │ (로컬)  │  │ :8100        │ │
│  └────┬────┘  └────┬────┘  └──────────────┘ │
│       │            │                          │
│  ┌────┴────────────┴──────────────────────┐  │
│  │     Docker Compose                      │  │
│  │  ┌──────────┐                          │  │
│  │  │PostgreSQL│                          │  │
│  │  │ :5432    │                          │  │
│  │  └──────────┘                          │  │
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### 프로덕션 환경 (docker-compose.prod.yml)

```
┌─────────────────────────────────────────────────┐
│ Vercel (프론트엔드)                               │
└──────────────┬──────────────────────────────────┘
               │ HTTPS
┌──────────────┴──────────────────────────────────┐
│ 서버 (Docker Compose)                             │
│  ┌────────┐  ┌────────┐  ┌──────┐  ┌─────────┐ │
│  │ Nginx  │→ │Backend │→ │Postgr│  │ LanceDB │ │
│  │  :80   │  │ x4 wkr │  │ :5432│  │ (option)│ │
│  └────────┘  └────────┘  └──────┘  └─────────┘ │
│              law-network (bridge)                  │
└───────────────────────────────────────────────────┘
```

---

## 핵심 파일

| 파일 | 용도 |
|------|------|
| `docker-compose.yml` | 개발 환경 (DB 2종) |
| `docker-compose.prod.yml` | 프로덕션 (Backend + Nginx + DB) |
| `docker/backend/Dockerfile` | 백엔드 개발 이미지 |
| `docker/backend/Dockerfile.prod` | 백엔드 프로덕션 (멀티스테이지) |
| `docker/frontend/Dockerfile` | 프론트엔드 개발 이미지 |
| `docker/frontend/Dockerfile.prod` | 프론트엔드 프로덕션 (멀티스테이지) |
| `services/lancedb/Dockerfile` | LanceDB 마이크로서비스 (MeCab ARM64) |
| `docker/nginx/nginx.conf` | Nginx 기본 설정 |
| `docker/nginx/conf.d/api.conf.template` | API 라우팅 (envsubst) |
| `docker/postgres/init.sql` | PostgreSQL 초기화 (uuid-ossp, pg_trgm) |
| `.dockerignore` | 빌드 컨텍스트 제외 |

---

## 1. 개발 환경 운영

### 시작

```bash
# DB 서비스 시작
docker compose up -d

# 상태 확인
docker compose ps

# 로그 확인
docker compose logs -f postgres
```

### 컨테이너 상세

| 서비스 | 컨테이너명 | 이미지 | 포트 |
|--------|-----------|--------|------|
| PostgreSQL | `law-platform-db` | 커스텀 빌드 (`docker/postgres/Dockerfile`, PG 17 + pg_textsearch) | `127.0.0.1:5432` |
| LanceDB | `lancedb-service` | 커스텀 빌드 | `127.0.0.1:8100` |

### 준비 대기

```bash
# PostgreSQL 준비 대기
for i in $(seq 1 15); do
  docker exec law-platform-db pg_isready -U lawuser -d lawdb 2>&1 && break
  echo "waiting... ($i)"
  sleep 2
done

```

### 환경변수 (.env 루트)

```bash
POSTGRES_USER=lawuser
POSTGRES_PASSWORD=<strong_password>
POSTGRES_DB=lawdb
```

---

## 2. 프로덕션 배포

### 배포 명령

```bash
# 환경변수 설정
cp .env.prod.example .env.prod
# .env.prod 편집 (API 키, 비밀번호 등)

# 빌드 + 배포
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build

# LanceDB 포함 시
docker compose --profile lancedb --env-file .env.prod -f docker-compose.prod.yml up -d --build
```

### 프로덕션 서비스

| 서비스 | 특징 |
|--------|------|
| **Backend** | 멀티스테이지, non-root(appuser), 4 워커, 헬스체크 |
| **Nginx** | 리버스 프록시, 25MB 업로드, keepalive 32, SSE 지원 |
| **PostgreSQL** | Named volume, 포트 비노출 (내부만) |
| **LanceDB** | Optional profile, 외부 volume |

### 멀티스테이지 빌드 구조

```dockerfile
# docker/backend/Dockerfile.prod

# Stage 1: Builder
FROM python:3.11-slim AS builder
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev  # 프로덕션 의존성만

# Stage 2: Runtime
FROM python:3.11-slim
RUN useradd -m appuser          # non-root
COPY --from=builder /app/.venv /app/.venv
COPY --chown=appuser backend/app /app/app
USER appuser
HEALTHCHECK CMD curl -f http://localhost:8000/health
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## 3. 네트워크 & 볼륨

### 네트워크

| 환경 | 네트워크 | 타입 | DNS |
|------|---------|------|-----|
| 개발 | default | bridge | 서비스명 |
| 프로덕션 | `law-network` | bridge | 서비스명 |

프로덕션에서 서비스 간 통신: `postgres:5432`, `backend:8000`, `lancedb:8100`

### 볼륨

| 볼륨 | 용도 | 영속성 |
|------|------|--------|
| `postgres_data` | PostgreSQL 데이터 | Named volume |
| `lancedb_data` | 벡터 임베딩 데이터 | External volume |

### 볼륨 백업

```bash
# PostgreSQL 덤프
docker exec law-platform-db pg_dump -U lawuser lawdb > backup.sql

# 볼륨 직접 백업
docker run --rm -v postgres_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/postgres_data.tar.gz -C /data .
```

---

## 4. 헬스체크

모든 서비스에 헬스체크 구성됨:

| 서비스 | 방법 | 간격 | 타임아웃 | 재시도 |
|--------|------|------|---------|--------|
| PostgreSQL | `pg_isready` | 10s | 5s | 5 |
| LanceDB | `curl http://localhost:8100/health` | 15s | 5s | 3 |
| Backend (prod) | `curl http://localhost:8000/health` | 30s | 10s | 3 |
| Frontend (prod) | `wget http://localhost:3000/` | 30s | 10s | 3 |

### 헬스체크 패턴

```yaml
healthcheck:
  test: ["CMD", "pg_isready", "-U", "lawuser", "-d", "lawdb"]
  interval: 10s
  timeout: 5s
  retries: 5
  start_period: 30s  # 초기 대기 시간
```

---

## 5. 서비스 추가 패턴

새 Docker 서비스 추가 시 절차:

### 1단계: docker-compose.yml에 서비스 정의

```yaml
services:
  new-service:
    image: <image>:<tag>
    container_name: <name>
    ports:
      - "127.0.0.1:<port>:<port>"  # localhost만 바인딩
    environment:
      - KEY=value
    volumes:
      - new_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:<port>/health"]
      interval: 15s
      timeout: 5s
      retries: 3
    restart: unless-stopped

volumes:
  new_data:
```

### 2단계: backend config.py에 연결 설정 추가

```python
# backend/app/core/config.py
NEW_SERVICE_URL: str = "http://localhost:<port>"
```

### 3단계: .env.example 업데이트

```bash
# .env.example
NEW_SERVICE_URL=http://localhost:<port>
```

### 4단계: 프로덕션 구성 추가

docker-compose.prod.yml에 동일 서비스 추가 (포트 비노출, 네트워크 연결).

---

## 6. WSL2 환경 특이사항

> 상세: `.claude/rules/wsl2-docker.md`

| 규칙 | 설명 |
|------|------|
| `docker.exe` 사용 | WSL2에서 `docker` 대신 `docker.exe` |
| 볼륨 마운트 오류 | 마운트 실패 시 볼륨 없이 직접 실행 |
| 이름 충돌 | `docker.exe rm -f <name>` 후 재실행 |

### WSL2 직접 실행 예시

```bash
# 볼륨 마운트 실패 시 대안
docker.exe run -d \
  --name law-platform-db \
  -p 127.0.0.1:5432:5432 \
  -e POSTGRES_USER=lawuser \
  -e POSTGRES_PASSWORD=<password> \
  -e POSTGRES_DB=lawdb \
  postgres:17-alpine
```

---

## 7. LanceDB 마이크로서비스

ARM64 (Apple Silicon, AWS Graviton) 호환 커스텀 빌드:

### 특징

- MeCab 소스 컴파일 (aarch64 지원)
- `config.guess`/`config.sub` 업데이트로 ARM64 호환
- 빌드 도구 제거 후 런타임 최소화
- mecab-ko-dic 포함

### 모드 전환

```bash
# 로컬 모드 (기본)
LANCEDB_MODE=local

# 원격 모드 (마이크로서비스)
LANCEDB_MODE=remote
LANCEDB_SERVICE_URL=http://localhost:8100  # 개발
LANCEDB_SERVICE_URL=http://lancedb:8100    # Docker 내부
```

---

## 8. Nginx 프록시 설정

### 라우팅 규칙 (`docker/nginx/conf.d/api.conf.template`)

| 경로 | 대상 | 특징 |
|------|------|------|
| `/health` | Backend | 헬스체크 |
| `/docs`, `/redoc` | Backend | API 문서 |
| `/api/` | Backend | buffering off, SSE 지원 |
| `/media/` | Backend | 미디어 파일 |
| `/` | 404 | 루트 보호 |

### 타임아웃 설정

```nginx
proxy_read_timeout 3600s;   # 1시간 (LLM 스트리밍)
proxy_send_timeout 3600s;
proxy_connect_timeout 10s;
```

### 환경변수 치환

```nginx
server_name ${API_DOMAIN};  # .env에서 API_DOMAIN 설정
```

---

## 9. 디버깅 & 트러블슈팅

### 로그 확인

```bash
# 특정 서비스 로그
docker compose logs -f --tail=100 postgres

# 모든 서비스 로그
docker compose logs -f

# 프로덕션
docker compose -f docker-compose.prod.yml logs -f backend
```

### 컨테이너 접속

```bash
# PostgreSQL CLI
docker exec -it law-platform-db psql -U lawuser -d lawdb

# Backend 셸
docker exec -it law-platform-backend /bin/bash
```

### 일반적인 문제

| 증상 | 원인 | 해결 |
|------|------|------|
| 포트 충돌 | 기존 컨테이너/프로세스 | `docker rm -f <name>` 또는 `lsof -i :<port>` |
| 볼륨 권한 | non-root 사용자 | `chown` 또는 Dockerfile에서 권한 설정 |
| 네트워크 연결 실패 | 서비스 미시작 | `depends_on` + 헬스체크 대기 |
| 이미지 빌드 실패 | 캐시 오염 | `docker compose build --no-cache` |
| 디스크 부족 | 미사용 이미지/볼륨 | `docker system prune -a` |

---

## 10. 체크리스트

### 새 환경 세팅

- [ ] `.env` 파일 생성 (`.env.example` 복사)
- [ ] 비밀번호 변경 (PostgreSQL)
- [ ] `docker compose up -d`
- [ ] 헬스체크 통과 대기
- [ ] DB 초기 데이터 로드 (스크립트 실행)
- [ ] 벡터 데이터 복원 (Google Drive)
- [ ] 임베딩/리랭커 모델 다운로드

### 프로덕션 배포

- [ ] `.env.prod` 설정 완료
- [ ] API 키, 비밀번호 강력한 값
- [ ] CORS_ORIGINS 프로덕션 도메인
- [ ] Nginx SSL 인증서 (Let's Encrypt)
- [ ] 헬스체크 모니터링 설정
- [ ] 로그 수집 설정
- [ ] 백업 스케줄 설정
