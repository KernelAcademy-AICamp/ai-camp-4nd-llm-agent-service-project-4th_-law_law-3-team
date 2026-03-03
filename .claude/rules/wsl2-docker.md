# WSL2 Docker 실행 규칙

WSL2 환경에서 Docker를 사용할 때 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

> **상세 가이드**: `docs/operations/wsl2-docker-guide.md` 참조

## 1. 핵심 규칙

- WSL2에서는 `docker` 대신 **`docker.exe`** 사용
- 볼륨 마운트 에러 시 **볼륨 없이 직접 실행** (상세: 가이드 참조)

## 2. 프로젝트 컨테이너

| 컨테이너 | 이미지 | 포트 | 비고 |
|----------|--------|------|------|
| `law-platform-db` | 커스텀 빌드 (`docker/postgres/Dockerfile`, PG 17 + pg_textsearch) | 5432 | `shm_size: 1gb`, BM25 인덱스용 |

## 3. 준비 대기

```bash
for i in $(seq 1 15); do
  docker.exe exec law-platform-db pg_isready -U lawuser -d lawdb 2>&1 && break
  echo "waiting... ($i)"
  sleep 2
done
```

## 4. 이름 충돌

`docker.exe run`에서 이름 충돌 시: `docker.exe rm -f <name>` 후 재실행

---

Docker Desktop이 WSL 통합을 정상 지원하면 `docker compose up -d` 사용 권장.
