# 법률 서비스 플랫폼 - 데모 배포 비용 산정

> 기준일: 2026년 2월 3일 | 환율: 1 USD = 1,460 KRW
> 대상 기간: 데모 시연용 (1주~1개월)

---

## 인프라 요구사항 요약

| 구성요소 | 설명 | 리소스 요구 |
|----------|------|------------|
| **Backend (FastAPI)** | 4 workers + 임베딩 모델(2.3GB) | RAM 8~16GB |
| **PostgreSQL** | 법령/판례/사용자 데이터 | 500MB~5GB 스토리지 |
| **Neo4j** | 그래프 DB (70K+ 노드, 163K+ 관계) | 2GB heap (Docker 내장) |
| **LanceDB** | 벡터 DB (임베딩 25만 청크) | 1.6GB 디스크 (백엔드 내장) |
| **Frontend (Next.js)** | 정적 + SSR | Vercel Free 호스팅 |
| **디스크** | 모델 + 데이터 + DB | 최소 15GB |

---

## 데모 비용 산정

> 모든 서비스를 **단일 EC2 인스턴스**에서 Docker로 운영, 프론트엔드는 Vercel Free

### 서버 비용 (EC2 t3.xlarge)

| 항목 | 시간 단가 | 1주 (168h) | 2주 (336h) | 1개월 (720h) |
|------|----------|-----------|-----------|-------------|
| **EC2 t3.xlarge** (4 vCPU / 16GB) | $0.1664/hr | $27.96 | $55.90 | $121.47 |
| **EBS gp3 50GB** | $4.00/mo | $1.00 | $2.00 | $4.00 |
| **서버 소계 (USD)** | | **$28.96** | **$57.90** | **$125.47** |
| **서버 소계 (KRW)** | | **42,280원** | **84,530원** | **183,190원** |

> EBS는 월정액이므로 기간에 비례하여 산정

### OpenAI API 비용 (gpt-4o-mini)

> 건당 평균: Input ~3,500 토큰, Output ~1,000 토큰 | 하루 100건 기준

| 기간 | 채팅 건수 | Input 토큰 | Output 토큰 | 비용 (USD) | 비용 (KRW) |
|------|----------|-----------|------------|-----------|-----------|
| **1주** (7일) | 700건 | ~2.45M | ~0.7M | $0.79 | 1,150원 |
| **2주** (14일) | 1,400건 | ~4.9M | ~1.4M | $1.58 | 2,310원 |
| **1개월** (30일) | 3,000건 | ~10.5M | ~3.0M | $3.50 | 5,110원 |

> 토큰 단가: Input $0.15/1M, Output $0.60/1M

### 무료 항목

| 항목 | 설명 |
|------|------|
| **Upstage Solar API** | 2026.03.02까지 무료 |
| **Kakao Maps API** | 일 30만건 무료 (충분) |
| **Vercel Hobby** | Next.js 호스팅 (비상업 개인용) |
| **Let's Encrypt SSL** | 무료 SSL 인증서 |
| **Neo4j** | Docker 자체 호스팅 (EC2 내 포함) |
| **PostgreSQL** | Docker 자체 호스팅 (EC2 내 포함) |

### 기간별 총 비용 합산 (AWS EC2 기준)

| 항목 | 1주 | 2주 | 1개월 |
|------|-----|-----|-------|
| 서버 (EC2 + EBS) | 42,280원 | 84,530원 | 183,190원 |
| OpenAI API | 1,150원 | 2,310원 | 5,110원 |
| Solar API | 0원 | 0원 | 0원 |
| 프론트엔드 (Vercel) | 0원 | 0원 | 0원 |
| **합계** | **~43,400원** | **~86,800원** | **~188,300원** |

---

## 비용 절감 방안

### 1. Oracle Cloud Always Free (서버 비용 0원)

**가장 파격적인 절감 방안**. Oracle Cloud의 Always Free Tier를 활용하면 서버 비용이 0원.

| 항목 | Always Free 제공량 | 이 프로젝트 요구 |
|------|-------------------|-----------------|
| ARM CPU (Ampere A1) | 4 OCPU (= 4코어) | 충분 |
| RAM | 24GB | 충분 (임베딩 모델 2.3GB + DB + 앱) |
| Block Storage | 200GB | 충분 (데이터 ~15GB) |
| 아웃바운드 트래픽 | 10TB/월 | 충분 |
| 기간 | **영구 무료** (시간 제한 없음) | - |

**주의사항:**
- ARM(aarch64) 아키텍처이므로 Docker 이미지를 ARM용으로 빌드해야 함
- 인기 리전에서는 인스턴스 확보가 어려울 수 있음 (서울 리전 포함)
- sentence-transformers/PyTorch는 ARM에서 정상 동작하나 x86 대비 약간 느릴 수 있음

**Oracle Free 활용 시 기간별 비용:**

| 항목 | 1주 | 2주 | 1개월 |
|------|-----|-----|-------|
| 서버 (OCI Always Free) | 0원 | 0원 | 0원 |
| OpenAI API (100건/일) | 1,150원 | 2,310원 | 5,110원 |
| Vercel Free | 0원 | 0원 | 0원 |
| **합계** | **~1,150원** | **~2,310원** | **~5,110원** |

---

### 2. LLM API 비용 절감

#### Solar Pro 3 무료 기간 최대 활용
- **2026.03.02까지 무료** → 가능한 Solar로 트래픽 분산
- 무료 종료 후에도 gpt-4o-mini와 동일 단가 ($0.15/$0.60)

#### 캐싱 활용 (Cached Input 50% 할인)
- 반복되는 시스템 프롬프트는 캐싱으로 자동 50% 할인

#### 응답 길이 최적화
- max_tokens 제한으로 불필요한 장문 응답 방지
- 시스템 프롬프트 최적화로 Input 토큰 절감

---

### 3. 프론트엔드 무료 호스팅

| 서비스 | 무료 범위 | 적합성 |
|--------|----------|--------|
| **Vercel Hobby** | 100GB 대역폭, 무제한 프로젝트 | 비상업 데모 |
| **Cloudflare Pages** | 무제한 대역폭, 500빌드/월 | 상업용도 가능 |
| **Netlify Free** | 100GB 대역폭, 300빌드분/월 | 대안 |

---

### 4. Neo4j 자체 호스팅 (추가 비용 없음)

AuraDB Pro($65/mo) 대신 같은 EC2에서 Docker로 Neo4j 직접 운영.

- 이미 시나리오에서 Docker로 운영 중이므로 **추가 비용 0원**
- Neo4j AuraDB Free는 50K 노드 제한으로 이 프로젝트(70K+ 노드) 부적합
- 데모 기간 동안 백업/모니터링 부담 최소

---

## OpenAI API 비용 상세 (gpt-4o-mini)

> 토큰 단가: Input $0.15/1M, Output $0.60/1M

| 일일 채팅 건수 | 월간 Input 토큰 | 월간 Output 토큰 | 월 비용 (KRW) |
|--------------|----------------|-----------------|-------------|
| 100건 | ~10.5M | ~3M | 5,110 |
| 500건 | ~52.5M | ~15M | 24,500 |
| 1,000건 | ~105M | ~30M | 49,640 |

> 건당 평균: Input ~3,500 토큰 (시스템 프롬프트 + 대화 기록 + RAG 컨텍스트), Output ~1,000 토큰

---

## 무료 서비스 정리

| 서비스 | 무료 조건 | 비고 |
|--------|----------|------|
| Upstage Solar Pro 3 | 2026.03.02까지 전면 무료 | 이후 $0.15/$0.60 per 1M tokens |
| Kakao Maps API | 일 30만건 | 초과 시 건당 10원 (할인가) |
| Vercel Hobby | 비상업 개인용 무한 | 상업용은 Pro $20/mo 필요 |
| Let's Encrypt SSL | 무제한 무료 | 90일 자동 갱신 |

---

## 결론: 데모 기간별 총 비용 요약

| 기간 | AWS EC2 사용 시 | Oracle Free 활용 시 |
|------|----------------|-------------------|
| **1주** | **~43,400원** | **~1,150원** (API 비용만) |
| **2주** | **~86,800원** | **~2,310원** (API 비용만) |
| **1개월** | **~188,300원** | **~5,110원** (API 비용만) |

> **가장 큰 비용 요인**: 서버(EC2) 비용. Oracle Free 확보 시 API 비용만으로 운영 가능.

---

## 출처

- [OpenAI API Pricing](https://openai.com/api/pricing/)
- [AWS EC2 Pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
- [Neo4j AuraDB Pricing](https://neo4j.com/pricing/)
- [Vercel Pricing](https://vercel.com/pricing)
- [Upstage AI Pricing](https://www.upstage.ai/pricing/api)
- [Kakao Developers 쿼터](https://developers.kakao.com/docs/latest/ko/getting-started/quota)
- [Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/)
- [USD/KRW 환율 (Investing.com)](https://kr.investing.com/currencies/usd-krw)
