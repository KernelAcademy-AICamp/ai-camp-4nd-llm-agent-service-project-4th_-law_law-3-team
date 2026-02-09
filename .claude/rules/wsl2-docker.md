# WSL2 Docker 실행 규칙

WSL2 환경에서 Docker를 사용할 때 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

## 1. Docker 명령어

WSL2에서는 `docker` 대신 **`docker.exe`** 를 사용합니다.

```bash
# ✅ 올바른 사용법
docker.exe ps
docker.exe compose up -d postgres
docker.exe exec law-platform-db pg_isready -U lawuser

# ❌ WSL2에서 동작하지 않음
docker ps
docker compose up -d
```

## 2. 볼륨 마운트 제한

Docker Desktop이 WSL distro 경로를 마운트하지 못하는 경우가 있습니다.
`docker-compose.yml`의 볼륨 마운트가 실패하면 **볼륨 없이 직접 실행**합니다.

### 에러 패턴

```
Error response from daemon: can't access specified distro mount service:
stat /run/guest-services/distro-services/ubuntu-20.04.sock: no such file or directory
```

### 해결법: docker.exe run으로 직접 실행

```bash
# 기존 컨테이너 제거
docker.exe rm -f law-platform-db

# 볼륨 마운트 없이 실행
docker.exe run -d \
  --name law-platform-db \
  -p 127.0.0.1:5432:5432 \
  -e POSTGRES_USER=lawuser \
  -e POSTGRES_PASSWORD=lawpassword \
  -e POSTGRES_DB=lawdb \
  postgres:15-alpine
```

> **주의**: 이 방식은 `init.sql` 볼륨과 `postgres_data` named volume을 사용하지 않으므로
> 컨테이너 삭제 시 데이터가 유실됩니다. 개발/테스트 용도로만 사용.

### Neo4j도 동일

```bash
docker.exe rm -f neo4j-law-graph

docker.exe run -d \
  --name neo4j-law-graph \
  -p 127.0.0.1:7474:7474 \
  -p 127.0.0.1:7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.15.0
```

## 3. 컨테이너 준비 대기

PostgreSQL은 시작 후 초기화에 수 초가 필요합니다.

```bash
# 준비 대기 루프
for i in $(seq 1 15); do
  docker.exe exec law-platform-db pg_isready -U lawuser -d lawdb 2>&1 && break
  echo "waiting... ($i)"
  sleep 2
done
```

## 4. 컨테이너 관리

```bash
# 상태 확인
docker.exe ps

# 로그 확인
docker.exe logs law-platform-db
docker.exe logs neo4j-law-graph

# 중지
docker.exe stop law-platform-db

# 시작 (기존 컨테이너)
docker.exe start law-platform-db

# 삭제 후 재생성
docker.exe rm -f law-platform-db
```

## 5. 컨테이너 이름 충돌

`docker.exe run`에서 이름 충돌이 발생하면 먼저 제거합니다.

```bash
# 에러: The container name "/law-platform-db" is already in use
docker.exe rm -f law-platform-db && docker.exe run -d ...
```

## 6. 프로젝트 컨테이너 목록

| 컨테이너 | 이미지 | 포트 |
|----------|--------|------|
| `law-platform-db` | `postgres:15-alpine` | 5432 |
| `neo4j-law-graph` | `neo4j:5.15.0` | 7474, 7687 |

---

**중요**: Docker Desktop이 WSL 통합을 정상 지원하면 `docker compose up -d` 사용을 권장합니다.
볼륨 마운트 에러가 발생하는 경우에만 이 규칙의 직접 실행 방식을 사용합니다.
