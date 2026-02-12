# AWS 배포 플랜

## 최종 아키텍처

```
Vercel (무료)           AWS EC2 t3.large (8GB)         AWS RDS db.t3.micro
┌──────────┐           ┌─────────────────────┐        ┌────────────────┐
│ Next.js  │──HTTPS──►│ Backend (FastAPI)    │──TCP──►│ PostgreSQL 15  │
│ Frontend │           │ LanceDB Service     │        │ (관계형+그래프) │
└──────────┘           └─────────────────────┘        └────────────────┘
```

**3주 예상 비용: ~$60 (24/7) / ~$35 (12h/일)**

---

## Phase 1: Neo4j → PostgreSQL 마이그레이션 (로컬, 배포 전)

### 1-1. Alembic 마이그레이션 작성

**새 파일: `backend/alembic/versions/009_add_graph_tables.py`**

pg_trgm 확장 + 6개 테이블 생성:

| 테이블 | 역할 | 행 수 |
|--------|------|-------|
| `graph_statutes` | 법령 노드 | 5,572 |
| `graph_aliases` | 비공식 약칭 | 69 |
| `graph_statute_hierarchy` | 시행령→법률 | 3,624 |
| `graph_statute_relations` | 법령↔법령 관련 | 93 |
| `graph_case_statute_citations` | 판례→법령 인용 | 72,414 |
| `graph_case_case_citations` | 판례→판례 인용 | 87,654 |

인덱스: `graph_statutes.name`, `graph_statutes.abbreviation`에 GIN `gin_trgm_ops` (FTS 대체)

### 1-2. ORM 모델

**새 파일들:**
- `backend/app/models/graph_statute.py` — GraphStatute, GraphAlias
- `backend/app/models/graph_relations.py` — GraphStatuteHierarchy, GraphStatuteRelation, GraphCaseStatuteCitation, GraphCaseCaseCitation

**수정:** `backend/app/models/__init__.py`, `backend/alembic/env.py` — import 추가

### 1-3. 데이터 로드 스크립트

**새 파일: `backend/scripts/load_graph_data.py`**

기존 `build_graph.py`와 동일 소스 JSON 파일 사용:
- `data/law_v1.json` → `graph_statutes`
- `data/law_hierarchy.json` → `graph_statute_hierarchy` + `graph_statute_relations`
- `data/precedents_v1.json` → citation 테이블 2개
- `scripts/informal_abbreviations.json` → `graph_aliases`
- `citation_count` 계산 (UPDATE ... SELECT COUNT)

배치 1,000건, `ON CONFLICT DO UPDATE`, `--verify` 플래그

### 1-4. PostgreSQL 그래프 서비스

**새 파일: `backend/app/tools/graph/pg_graph_service.py`**

기존 `GraphService`와 동일 인터페이스, SQLAlchemy async 구현:

| 메서드 | Neo4j → PostgreSQL |
|--------|-------------------|
| `get_cited_statutes()` | `JOIN graph_case_statute_citations → graph_statutes` |
| `get_statute_hierarchy()` | `Recursive CTE on graph_statute_hierarchy` |
| `get_similar_cases()` | `JOIN 2회 (같은 법령 인용 판례)` |
| `get_related_statutes()` | `JOIN graph_statute_relations` |
| `search_statute()` | `pg_trgm similarity() + ILIKE` (정식명/공식약칭/비공식약칭) |
| `enrich_case_context()` | 위 메서드 조합 |

### 1-5. Feature Flag + 라우터 수정

**수정: `backend/app/core/config.py`**
- `USE_PG_GRAPH: bool = True` 추가

**수정: `backend/app/tools/graph/__init__.py`**
- `get_graph_service()` → `USE_PG_GRAPH`에 따라 `PgGraphService` 또는 `GraphService` 반환

**수정: `backend/app/modules/case_precedent/router/__init__.py`**
- 5개 statute 엔드포인트를 `PgGraphService` 사용으로 변경:
  - `GET /statutes/search` — pg_trgm 유사도 검색
  - `GET /statutes/hierarchy/{id}` — Recursive CTE
  - `GET /statutes/{id}/children` — SELECT + JOIN
  - `GET /statutes/graph` — Recursive CTE + 링크 구성
  - `POST /precedents/{id}/ask` — 그래프 enrichment (이미 graceful)

**수정: `backend/app/common/chat_service.py`** — graph_service 호출 부분 (이미 graceful, 변경 최소)

### 1-6. 로컬 검증

```bash
cd backend
uv run alembic upgrade head
uv run python scripts/load_graph_data.py && uv run python scripts/load_graph_data.py --verify
USE_PG_GRAPH=true uv run uvicorn app.main:app --reload
# 5개 엔드포인트 curl 테스트
uv run ruff check backend/app/ && uv run mypy backend/app/
```

---

## Phase 2: 배포용 Docker/Config 수정 (로컬)

### 2-1. 배포용 Docker Compose

**새 파일: `docker-compose.deploy.yml`**

Backend + LanceDB만 포함 (PostgreSQL은 RDS):

```yaml
services:
  backend:
    build:
      context: .
      dockerfile: docker/backend/Dockerfile.prod
    ports: ["0.0.0.0:8000:8000"]
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${RDS_ENDPOINT}:5432/${POSTGRES_DB}
      LANCEDB_MODE: remote
      LANCEDB_SERVICE_URL: http://lancedb:8100
      USE_PG_GRAPH: "true"
      USE_DB_LAWYERS: "true"
      CORS_ORIGINS: '["https://${VERCEL_DOMAIN}"]'
      # ... LLM API keys
    volumes:
      - model_data:/app/data/models
    depends_on:
      lancedb: { condition: service_healthy }

  lancedb:
    build: ./services/lancedb
    volumes:
      - lancedb_data:/app/lancedb_data
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8100/health || exit 1"]

volumes:
  lancedb_data: { external: true, name: law-3-team_lancedb_data }
  model_data: { external: true, name: law-3-team_model_data }
```

### 2-2. Backend Dockerfile.prod 수정

**수정: `docker/backend/Dockerfile.prod`**
- MeCab-ko 빌드 스텝 추가 (services/lancedb/Dockerfile에서 복사)
- workers: 4 → 2 (t3.large 2vCPU 대응)

### 2-3. Frontend next.config.js 수정

**수정: `frontend/next.config.js`**

```javascript
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'
// rewrites destination을 BACKEND_URL 변수로 변경
```

Vercel에서 `BACKEND_URL=http://<EC2_PUBLIC_IP>:8000` 설정

---

## Phase 3: AWS 인프라 구성

### 3-1. 보안 그룹 생성

| 보안 그룹 | 인바운드 규칙 |
|-----------|-------------|
| `law-ec2-sg` | SSH(22) 내 IP만, HTTP(8000) 0.0.0.0/0 |
| `law-rds-sg` | PostgreSQL(5432) EC2 보안그룹에서만 |

### 3-2. RDS 생성

```bash
aws rds create-db-instance \
  --db-instance-identifier law-platform-db \
  --db-instance-class db.t3.micro \
  --engine postgres --engine-version 15 \
  --master-username lawuser --master-user-password <PASSWORD> \
  --allocated-storage 20 --storage-type gp3 \
  --db-name lawdb --no-multi-az --no-publicly-accessible \
  --vpc-security-group-ids <RDS_SG_ID> \
  --backup-retention-period 1
```

RDS 가용 후 `pg_trgm` 확장 활성화

### 3-3. EC2 생성 + 초기 설정

```bash
aws ec2 run-instances \
  --instance-type t3.large \
  --image-id <AMAZON_LINUX_2023_AMI> \
  --key-name law-platform-key \
  --security-group-ids <EC2_SG_ID> \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]'
```

EC2에 Docker + Docker Compose + PostgreSQL client 설치

---

## Phase 4: 데이터 이전 + 서비스 시작

### 4-1. 데이터 전송 (로컬 → EC2)

| 데이터 | 크기 | 전송 방법 |
|--------|------|----------|
| `backend/lancedb_data/` | 4.1GB | scp (tar.gz) 또는 Google Drive |
| `backend/data/models/` | 2.6GB | scp (tar.gz) |
| 소스 코드 | ~50MB | git clone |

### 4-2. Docker 볼륨 준비

```bash
docker volume create law-3-team_lancedb_data
docker volume create law-3-team_model_data
# alpine cp로 데이터 복사
```

### 4-3. RDS 마이그레이션 + 데이터 로드

```bash
cd ~/law-3-team/backend
uv run alembic upgrade head
uv run python scripts/load_lawyers_data.py
uv run python scripts/load_graph_data.py
# --verify로 각각 검증
```

### 4-4. Docker 빌드 + 시작

```bash
docker-compose -f docker-compose.deploy.yml build
docker-compose -f docker-compose.deploy.yml up -d
```

---

## Phase 5: Vercel 배포

### 5-1. Vercel 프로젝트 설정

```bash
cd frontend && vercel
```

환경변수:
- `BACKEND_URL=http://<EC2_PUBLIC_IP>:8000`
- `NEXT_PUBLIC_KAKAO_MAP_API_KEY=<key>`

### 5-2. 배포 + CORS 업데이트

```bash
vercel --prod
# EC2의 CORS_ORIGINS에 Vercel 도메인 추가 후 restart
```

---

## 수정/생성 파일 요약

**새 파일 (7):**
1. `backend/alembic/versions/009_add_graph_tables.py`
2. `backend/app/models/graph_statute.py`
3. `backend/app/models/graph_relations.py`
4. `backend/app/tools/graph/pg_graph_service.py`
5. `backend/scripts/load_graph_data.py`
6. `docker-compose.deploy.yml`
7. EC2 배포용 `.env` (git 외부)

**수정 파일 (8):**
1. `backend/app/models/__init__.py` — 새 모델 import
2. `backend/alembic/env.py` — 새 모델 import
3. `backend/app/core/config.py` — `USE_PG_GRAPH` 추가
4. `backend/app/tools/graph/__init__.py` — feature flag 분기
5. `backend/app/modules/case_precedent/router/__init__.py` — statute 엔드포인트 PostgreSQL 전환
6. `docker/backend/Dockerfile.prod` — MeCab 빌드, workers 조정
7. `frontend/next.config.js` — `BACKEND_URL` 환경변수화
8. `backend/app/common/chat_service.py` — PgGraphService 호환 (최소 변경)

---

## 비용 상세

| 서비스 | 사양 | 시간당 | 3주(504h) |
|--------|------|--------|----------|
| EC2 t3.large | 2vCPU, 8GB | $0.0832 | $41.93 |
| EBS gp3 30GB | 스토리지 | - | $2.40 |
| RDS db.t3.micro | 2vCPU, 1GB | $0.018 | $9.07 |
| RDS Storage 20GB | gp3 | - | $1.60 |
| Data Transfer | ~50GB | - | $4.50 |
| Vercel | Free | $0 | $0 |
| **합계 (24/7)** | | | **~$59.50** |
| **합계 (12h/일)** | | | **~$35** |

---

## 검증 체크리스트

### Phase 1 완료 후 (로컬)
- [ ] `uv run alembic upgrade head` 성공
- [ ] `load_graph_data.py --verify` 통과
- [ ] 5개 statute 엔드포인트 정상 응답
- [ ] 채팅 API 그래프 enrichment 동작
- [ ] `ruff check` + `mypy` 통과

### Phase 4 완료 후 (EC2)
- [ ] `curl http://<EC2_IP>:8000/health` → 200
- [ ] 판례 검색: `/api/case-precedent/precedents?keyword=손해배상`
- [ ] 법령 검색: `/api/case-precedent/statutes/search?query=민법`
- [ ] 변호사 검색: `/api/lawyer-finder/nearby?lat=37.5665&lng=126.978&radius=5`
- [ ] 채팅: `POST /api/chat` 정상 응답

### Phase 5 완료 후 (E2E)
- [ ] Vercel 도메인 접속
- [ ] `/lawyer-finder` — 지도 + 검색
- [ ] `/case-precedent` — 판례 검색 + 법령 계급 트리
- [ ] `/lawyer-stats` — 대시보드 지도
- [ ] 채팅 위젯 — SSE 스트리밍
