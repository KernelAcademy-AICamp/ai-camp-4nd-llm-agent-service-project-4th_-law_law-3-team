# EC2 서버 셋업 가이드

> AWS EC2 ARM(Graviton) + Docker 기반 프로덕션 배포

## 아키텍처

```
┌─────────────────────────────────────────────┐
│  Vercel (프론트엔드)                         │
│  Next.js + BACKEND_URL → EC2               │
└─────────────┬───────────────────────────────┘
              │ HTTPS
┌─────────────▼───────────────────────────────┐
│  EC2 (t4g.xlarge, ARM Graviton)             │
│  ┌─────────────────────────────────────┐    │
│  │ Nginx (80/443, SSL, Rate Limiting)  │    │
│  │   ↓ reverse proxy                  │    │
│  │ Backend (FastAPI, Uvicorn x2)       │    │
│  │   ├── LanceDB (embedded, 7.7GB)    │    │
│  │   ├── MeCab (법률 사전)              │    │
│  │   └── ML Models (KURE-v1, Reranker) │    │
│  │ PostgreSQL 17 (BM25, pg_textsearch) │    │
│  │ Certbot (Let's Encrypt 자동 갱신)    │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  S3 (law-3 버킷) ← 데이터 저장소            │
└─────────────────────────────────────────────┘
```

## 사전 준비

| 항목 | 상태 확인 |
|------|----------|
| AWS 계정 + IAM 사용자 | `aws sts get-caller-identity` |
| S3 버킷 (`law-3`) | `aws s3 ls s3://law-3/deploy/` |
| S3 데이터 업로드 완료 | `bash scripts/deploy/s3-upload.sh` |
| 도메인 (선택) | DNS A 레코드 → EC2 IP |

---

## Phase 1: EC2 인스턴스 생성

### 1-1. AWS 콘솔 > EC2 > Launch Instance

| 설정 | 값 |
|------|-----|
| Name | `law-platform-prod` |
| AMI | Ubuntu 24.04 LTS — **Architecture: 64-bit (Arm)** |
| Instance type | `t4g.xlarge` (4 vCPU, 16GB RAM) |
| Key pair | 새로 생성 또는 기존 사용 (`law-platform-key.pem`) |
| Storage | **50GB gp3** |

### 1-2. Network Settings

```
VPC: 기본 VPC
Subnet: ap-northeast-2a (아무거나)
Auto-assign public IP: Enable (필수)

Security Group: law-platform-sg
┌──────────┬──────┬─────────────┬──────────────────┐
│ Type     │ Port │ Source      │ 용도             │
├──────────┼──────┼─────────────┼──────────────────┤
│ SSH      │ 22   │ My IP       │ SSH 접속         │
│ HTTP     │ 80   │ 0.0.0.0/0  │ Let's Encrypt    │
│ HTTPS    │ 443  │ 0.0.0.0/0  │ API 서비스       │
└──────────┴──────┴─────────────┴──────────────────┘
```

> SSH는 특정 IP만 허용. 0.0.0.0/0으로 열지 않는다.

### 1-3. IAM Role 연결 (EC2 → S3 접근용)

**IAM 콘솔에서 Role 먼저 생성:**

```
IAM > Roles > Create role
- Trusted entity: AWS service → EC2
- Policy (inline):
```

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::law-3",
        "arn:aws:s3:::law-3/*"
      ]
    }
  ]
}
```

```
Role name: law-platform-ec2-s3-role
```

EC2 Launch 화면 > Advanced details > IAM instance profile > `law-platform-ec2-s3-role` 선택

### 1-4. SSH 접속 확인

```bash
chmod 400 law-platform-key.pem
ssh -i law-platform-key.pem ubuntu@<EC2_PUBLIC_IP>
```

---

## Phase 2: Docker 설치

```bash
# 시스템 업데이트
sudo apt-get update && sudo apt-get upgrade -y

# Docker 설치
curl -fsSL https://get.docker.com | sudo sh

# docker 그룹에 추가 (sudo 없이 사용)
sudo usermod -aG docker $USER

# 재접속 (그룹 적용)
exit
```

SSH 재접속 후 확인:

```bash
docker --version         # Docker version 27.x
docker compose version   # Docker Compose version v2.x
```

---

## Phase 3: 프로젝트 클론

```bash
sudo mkdir -p /opt/law-platform
sudo chown $USER:$USER /opt/law-platform

git clone https://github.com/<org>/<repo>.git /opt/law-platform
cd /opt/law-platform
git checkout deploy/aws-arm
```

---

## Phase 4: AWS CLI 설치 + S3 데이터 다운로드

```bash
# AWS CLI 설치 (ARM)
cd /tmp
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
sudo apt-get install -y unzip
unzip -qo awscliv2.zip
sudo ./aws/install
aws --version

# IAM Role 자동 인증 확인 (키 설정 불필요)
aws sts get-caller-identity

# 데이터 다운로드
cd /opt/law-platform
bash scripts/deploy/s3-download.sh
```

### 다운로드 대상

| S3 경로 | 로컬 경로 | 크기 |
|---------|----------|------|
| `s3://law-3/deploy/lancedb_data/` | `backend/lancedb_data/` | 7.7GB |
| `s3://law-3/deploy/data/` | `data/` | ~110MB |
| `s3://law-3/deploy/mecab_userdic/` | `backend/data/mecab_userdic/` | 8.4MB |

---

## Phase 5: 환경변수 설정

```bash
cp .env.prod.example .env.prod
nano .env.prod
```

### 필수 변경 항목

```bash
# 비밀번호/키 생성
openssl rand -hex 16    # → POSTGRES_PASSWORD
openssl rand -hex 32    # → API_KEY
```

```ini
# Database
POSTGRES_PASSWORD=<생성한_값>

# API 보호
API_KEY=<생성한_값>

# 도메인 (도메인 없으면 EC2 Public IP)
API_DOMAIN=<도메인 또는 EC2_PUBLIC_IP>

# 프론트엔드 CORS
CORS_ORIGINS=["https://<vercel-domain>"]

# LLM Provider
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini

# Let's Encrypt (도메인 설정 시)
CERTBOT_EMAIL=your@email.com
```

### 선택 항목

```ini
# 카카오 지도 (변호사 찾기 기능)
KAKAO_MAP_API_KEY=your_kakao_javascript_key
KAKAO_REST_API_KEY=your_kakao_rest_api_key

# Anthropic (LLM_PROVIDER=anthropic 시)
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Google Gemini (스토리보드 일부 기능)
GOOGLE_API_KEY=
GOOGLE_MODEL=gemini-3-flash-preview
```

---

## Phase 6: 빌드 & 기동

```bash
# 볼륨 마운트 디렉토리 사전 생성
mkdir -p backend/data/models backend/data/mecab_userdic

# 빌드 (첫 빌드: 10~15분, MeCab 소스 컴파일 포함)
docker compose --env-file .env.prod -f docker-compose.prod.yml build

# 기동
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d
```

### 컨테이너 상태 확인

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml ps
```

기대 출력:

```
NAME                     STATUS
law-platform-db-prod     Up (healthy)
law-platform-backend     Up (health: starting)
law-platform-nginx       Up
law-platform-certbot     Up
```

---

## Phase 7: DB 마이그레이션

```bash
# PostgreSQL 준비 대기
docker compose --env-file .env.prod -f docker-compose.prod.yml exec -T postgres \
  pg_isready -U lawuser -d lawdb

# Alembic 마이그레이션 실행
docker compose --env-file .env.prod -f docker-compose.prod.yml exec -T backend \
  python -m alembic upgrade head
```

---

## Phase 8: LanceDB 데이터 볼륨 복사

docker-compose.prod.yml의 `lancedb_data`는 named volume이므로, S3에서 다운로드한 데이터를 컨테이너 내부로 복사해야 한다.

```bash
# 호스트 → 컨테이너 볼륨 복사
docker cp backend/lancedb_data/. law-platform-backend:/app/lancedb_data/

# Backend 재시작 (LanceDB 테이블 로드)
docker compose --env-file .env.prod -f docker-compose.prod.yml restart backend
```

---

## Phase 9: 헬스체크

```bash
# ML 모델 자동 다운로드 로그 확인 (2~3분 소요)
docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f backend

# 헬스체크
curl http://localhost/health

# API 동작 확인 (API Key 포함)
curl -H "X-API-Key: <API_KEY>" http://localhost/api/case-precedent/search?query=사기
```

---

## Phase 10: SSL 인증서 (도메인 설정 후)

### 사전 조건

- DNS A 레코드가 EC2 Public IP를 가리킴
- `.env.prod`에 `API_DOMAIN`, `CERTBOT_EMAIL` 설정 완료

```bash
bash scripts/deploy/init-letsencrypt.sh
```

발급 후 확인:

```bash
curl https://<API_DOMAIN>/health
```

> 인증서는 certbot 컨테이너가 12시간마다 자동 갱신한다.

---

## Phase 11: Vercel 프론트엔드 배포

Vercel 프로젝트 환경변수 설정:

| 변수 | 값 |
|------|-----|
| `BACKEND_URL` | `https://<API_DOMAIN>` |
| `NEXT_PUBLIC_KAKAO_MAP_API_KEY` | 카카오 JavaScript 키 |

---

## 운영 명령어

### 로그 확인

```bash
# 전체 로그
docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f

# Backend만
docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f backend

# 최근 100줄
docker compose --env-file .env.prod -f docker-compose.prod.yml logs --tail=100 backend
```

### 서비스 재시작

```bash
# 전체 재시작
docker compose --env-file .env.prod -f docker-compose.prod.yml restart

# Backend만
docker compose --env-file .env.prod -f docker-compose.prod.yml restart backend
```

### 서비스 중지 / 시작

```bash
# 중지 (데이터 유지)
docker compose --env-file .env.prod -f docker-compose.prod.yml down

# 시작
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d
```

### 코드 업데이트 후 재배포

```bash
cd /opt/law-platform
git pull origin deploy/aws-arm
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
docker compose --env-file .env.prod -f docker-compose.prod.yml exec -T backend \
  python -m alembic upgrade head
```

### 디스크 정리

```bash
# 사용하지 않는 Docker 이미지 정리
docker system prune -f

# Docker 전체 용량 확인
docker system df
```

---

## S3 데이터 관리

### 로컬 → S3 업로드

```bash
# 로컬 WSL에서 실행
bash scripts/deploy/s3-upload.sh          # 실제 업로드
bash scripts/deploy/s3-upload.sh --dry-run # 미리보기
```

### S3 → EC2 다운로드

```bash
# EC2 서버에서 실행
bash scripts/deploy/s3-download.sh
```

### S3 내용 확인

```bash
aws s3 ls s3://law-3/deploy/ --region ap-northeast-2
aws s3 ls s3://law-3/deploy/ --region ap-northeast-2 --recursive --summarize | tail -3
```

---

## 트러블슈팅

### Backend가 시작되지 않을 때

```bash
# 로그 확인
docker compose --env-file .env.prod -f docker-compose.prod.yml logs backend

# 일반적 원인:
# 1. POSTGRES_PASSWORD 미설정 → .env.prod 확인
# 2. API_KEY 미설정 → .env.prod 확인
# 3. ML 모델 다운로드 실패 → 네트워크 확인, 수동 재시작
```

### LanceDB 테이블을 찾을 수 없을 때

```bash
# 볼륨 내부 확인
docker exec law-platform-backend ls /app/lancedb_data/

# 데이터 재복사
docker cp backend/lancedb_data/. law-platform-backend:/app/lancedb_data/
docker compose --env-file .env.prod -f docker-compose.prod.yml restart backend
```

### PostgreSQL 연결 실패

```bash
# 컨테이너 상태 확인
docker compose --env-file .env.prod -f docker-compose.prod.yml ps postgres

# 직접 연결 테스트
docker exec law-platform-db-prod pg_isready -U lawuser -d lawdb
```

### SSL 인증서 발급 실패

```bash
# DNS 전파 확인
dig <API_DOMAIN>

# Nginx 80 포트 접근 가능한지 확인
curl http://<API_DOMAIN>/.well-known/acme-challenge/test

# Security Group에서 80 포트가 열려있는지 확인
```

### 디스크 부족

```bash
# 디스크 사용량 확인
df -h

# Docker 이미지 정리
docker system prune -af

# 로그 확인 (Docker json-file 드라이버 자동 회전됨)
du -sh /var/lib/docker/
```

---

## 권장 인스턴스 스펙

| 용도 | 인스턴스 | vCPU | RAM | 디스크 | 월 비용 (서울) |
|------|---------|------|-----|--------|-------------|
| **테스트/데모** | t4g.large | 2 | 8GB | 30GB | ~$60 |
| **프로덕션 (권장)** | t4g.xlarge | 4 | 16GB | 50GB | ~$120 |
| **고트래픽** | m7g.xlarge | 4 | 16GB | 100GB | ~$180 |

> t4g는 크레딧 기반. 지속적 부하 시 m7g 권장.

---

## GitHub Actions CI/CD

`deploy/aws-arm` 브랜치에 push하면 자동 배포된다 (`.github/workflows/deploy.yml`).

### GitHub Secrets 설정 필요

| Secret | 값 |
|--------|-----|
| `EC2_HOST` | EC2 Public IP |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_KEY` | `.pem` 파일 내용 전체 |
| `PROJECT_DIR` | `/opt/law-platform` |
