# WSL2 Docker 상세 가이드

WSL2 환경에서 Docker를 사용할 때의 상세 명령어 및 해결법.

> 핵심 규칙은 `.claude/rules/wsl2-docker.md` 참조

## PostgreSQL 직접 실행 (볼륨 마운트 실패 시)

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
  postgres:17-alpine
```

> **주의**: 이 방식은 `init.sql` 볼륨과 `postgres_data` named volume을 사용하지 않으므로
> 컨테이너 삭제 시 데이터가 유실됩니다. 개발/테스트 용도로만 사용.

## Neo4j 직접 실행

```bash
docker.exe rm -f neo4j-law-graph

docker.exe run -d \
  --name neo4j-law-graph \
  -p 127.0.0.1:7474:7474 \
  -p 127.0.0.1:7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.15.0
```

## 컨테이너 준비 대기

PostgreSQL은 시작 후 초기화에 수 초가 필요합니다.

```bash
for i in $(seq 1 15); do
  docker.exe exec law-platform-db pg_isready -U lawuser -d lawdb 2>&1 && break
  echo "waiting... ($i)"
  sleep 2
done
```

## 컨테이너 관리

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

## 컨테이너 이름 충돌

`docker.exe run`에서 이름 충돌이 발생하면 먼저 제거합니다.

```bash
# 에러: The container name "/law-platform-db" is already in use
docker.exe rm -f law-platform-db && docker.exe run -d ...
```

## 볼륨 마운트 에러 패턴

```
Error response from daemon: can't access specified distro mount service:
stat /run/guest-services/distro-services/ubuntu-20.04.sock: no such file or directory
```

이 에러가 발생하면 위의 "직접 실행" 방식을 사용합니다.
Docker Desktop이 WSL 통합을 정상 지원하면 `docker compose up -d` 사용을 권장합니다.
