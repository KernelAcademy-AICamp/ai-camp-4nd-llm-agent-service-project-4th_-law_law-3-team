# Red Team 검증 보고서 - Content Marketing Keyword Flow

> **Reviewer**: Gemini CLI (Red Team)
> **Date**: 2026-02-25
> **Target**: `docs/01-plan/features/content-marketing-keyword-flow.plan.md` v0.1

---

## 1. 취약점 (Critical/High/Medium/Low)

### [High] In-Memory Cache 기반 IDOR 및 세션 오염
- **위험**: `keyword_id`를 전역 딕셔너리에 저장할 경우, 멀티 테넌트 환경에서 A 사용자가 생성한 `keyword_id`를 B 사용자가 추측하여 뉴스 검색을 트리거하거나 개인화된 페르소나 정보를 탈취할 위험. 서버 재시작 시 모든 세션 끊김.
- **개선**: Redis와 같은 분산 캐시를 사용하고, 캐시 키에 `user_id`를 포함하여 격리.
- **코드**: `cache_key = f"kw-meta:{user_id}:{keyword_id}"`

### [High] Search Query Injection 및 SSRF
- **위험**: LLM이 추출한 키워드가 뉴스 검색 API 쿼리 매개변수로 직접 전달됨. 악의적 문자열(예: `...&admin=true` 또는 내부망 IP)이 키워드로 추출될 경우 2차 공격 경로 가능.
- **개선**: 검색 쿼리 전달 전 엄격한 Sanitize 및 허용 문자열 검증 필터 적용.

### [Medium] API Wallet Drain (DoS)
- **위험**: Step 1(Keyword Collect)은 Tavily와 LLM을 모두 소모하는 고비용 작업. 반복 요청 시 API 비용 급증.
- **개선**: 사용자별/IP별 Rate Limiting(예: 1시간당 5회 수집 제한) 적용.

---

## 2. 아키텍처 개선 제안

### [Stateful → Stateless] 분산 캐시 도입
- `_cache` 딕셔너리 방식은 스케일 아웃 불가. Redis 또는 PostgreSQL JSONB 필드를 활용한 임시 저장소로 전환하여 서버 확장성 확보.

### [Async Task Queue] 비동기 처리
- 5~15초 대기 시간은 HTTP 연결 타임아웃 유발 가능. Celery나 FastAPI Background Tasks를 사용하여 수집을 비동기화하고, 프론트엔드에서 Polling 또는 SSE로 상태 수신.

---

## 3. 고급 기능 추가 제안

### [Legal-Specific Scoring] 판례/법령 기반 가중치
- LLM 판단에만 의존하지 말고, 내부 RAG 엔진(LanceDB)을 이용해 해당 키워드와 유사한 판례 "밀도"를 측정하여 `legal_relevance` 점수에 실질적 근거 부여.

### [Clustering] 중복 이슈 그룹화
- 여러 커뮤니티에서 동일 사건을 다른 단어로 표현할 경우(예: "시청역 역주행", "시청역 사고"), Semantic Similarity로 묶어 하나의 키워드 카드로 병합하는 클러스터링 로직 필요.

---

## 4. 성능 최적화 제안

### [Tavily Optimization] include_domains 전략
- Tavily의 `site:` 쿼리는 검색 엔진 인덱싱 속도에 따라 최신 글 누락 가능성 큼. `search_depth="advanced"` + `topic="news"` 조합하거나, 특정 커뮤니티 전용 RSS/크롤러 병렬 운용이 현실적.

### [Parallel News Search] 소스별 타임아웃 차등 적용
- `asyncio.wait_for`를 사용하여 특정 소스가 전체 응답을 지연시키지 않도록 소스별 개별 타임아웃 설정.

---

## 5. 운영 안정성 제안

### [Observability] 비용 추적 로깅
- 모든 키워드 수집 시 `tavily_tokens`, `llm_input_tokens`, `llm_output_tokens`를 로그에 기록하고 Grafana 대시보드에서 실시간 비용 모니터링.

### [Fallback Quality Audit]
- LLM 실패 시 빈도 기반 폴백 키워드 품질이 매우 낮을 수 있음. 폴백 발생 횟수를 알람(Slack)으로 전송하여 프롬프트 엔지니어링 필요성 즉시 파악.

---

## 6. UX 개선 제안

### [Skeleton UI & Streaming]
- 수집 중 "커뮤니티 글 분석 중...", "키워드 점수 산출 중..." 등 단계별 상태를 실시간 업데이트하여 사용자 체감 대기 시간 감소.

### [Keyword Blacklist]
- 정치적 편향성이 강하거나 혐오 표현이 담긴 키워드는 Content Fitness 점수와 관계없이 필터링되는 블랙리스트 시스템 선행 필요.

---

## 개선 코드 예시 (Backend)

```python
# app/modules/content_marketing/service.py 개선안

async def collect_keywords(user_id: str, request: KeywordCollectRequest):
    # 1. Rate Limiting 검증
    if not await check_rate_limit(user_id, "keyword_collect"):
        raise HTTPException(status_code=429, detail="Too many requests")

    # 2. Redis 캐시 확인 (Idempotency)
    cache_key = f"kw-collect:{request.time_range}:{request.category}"
    if cached_data := await redis_client.get(cache_key):
        return KeywordCollectResponse.parse_raw(cached_data)

    # 3. 비동기 수집 (Tavily + LLM)
    # ... 기존 로직 ...

    # 4. 저장 및 세션 격리
    for kw in scored_keywords:
        kw.id = str(uuid.uuid4())
        await redis_client.setex(
            f"kw-meta:{user_id}:{kw.id}",
            3600,
            kw.json()
        )

    return response
```

---

## 총평

기획의 비즈니스 로직은 우수하나 **상태 관리(State Management)의 분산화**와 **입력값 검증(Sanitization)**이 실제 서비스 배포 전 반드시 해결되어야 할 핵심 과제.
