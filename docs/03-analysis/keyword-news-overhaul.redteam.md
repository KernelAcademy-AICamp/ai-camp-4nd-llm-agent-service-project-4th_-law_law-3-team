# Red Team 검증 보고서 — 키워드/뉴스 수집 시스템 전면 재설계

> **작성자**: Gemini CLI (Red Team)
> **대상**: `docs/01-plan/features/keyword-news-overhaul.plan.md`
> **작성일**: 2026-03-01

---

## 1. 취약점 (Critical/High/Medium/Low)

- **[High] Dynamic Query Injection Risk**: `_build_keyword_query` 등에서 사용자 입력(Category, Persona)이 필터링 없이 쿼리 빌더에 포함될 경우, 검색 API(Google, Naver)의 특수 연산자를 이용한 의도치 않은 데이터 노출이나 할당량 소모 공격이 가능함.
- **[High] API Key/Credential Exposure**: 다수의 외부 소스(Tavily, NewsData, Naver, YouTube 등)를 연동함에 따라 환경 변수 관리 부실 시 대규모 과금 피해 발생 위험. (Secret Rotation 정책 부재)
- **[Medium] Unofficial API Fragility**: `pytrends`는 구글의 공식 API가 아닌 내부 엔드포인트를 사용하므로, 구글의 구조 변경 시 서비스가 즉시 중단되는 단일 장애점(SPOF) 위험이 큼.
- **[Medium] Insecure Content Processing**: 수집된 뉴스/커뮤니티 본문을 LLM에 전달할 때, 프롬프트 인젝션(Prompt Injection)을 유도하는 텍스트가 포함되어 있을 경우 스코어링 로직이 왜곡될 수 있음.
- **[Low] Rate Limit Exhaustion (DoS)**: 다수의 소스를 병렬 수집하므로 특정 시점에 요청이 몰릴 경우 외부 API 측에서 IP 차단이나 Quota 소모로 인해 서비스 거부 상태가 발생할 수 있음.

## 2. 아키텍처 개선 제안

- **분산 태스크 큐(Task Queue) 도입**: 현재 `asyncio.gather` 방식은 수집 대상이 늘어날 경우 서버 가용성을 위협함. **Celery**나 **Temporal**을 도입하여 수집-추출-검색-스코어링 단계를 비동기 워크플로우로 분리하고 재시도(Retry) 전략을 수립해야 함.
- **Vector DB (LanceDB) 최적화**: Semantic Dedup을 위해 매번 임베딩을 계산하는 것은 비효율적임. 수집된 기사들을 LanceDB에 저장 시 **IVF-PQ 인덱싱**을 적용하고, '최근 24시간' 단위의 파티셔닝을 통해 검색 속도를 비약적으로 향상시켜야 함.
- **Circuit Breaker 패턴**: 외부 API(특히 `pytrends`나 `NewsData`) 장애 시 전체 파이프라인이 중단되지 않도록 Fallback 로직(예: 고정 키워드 사용 또는 캐시 데이터 반환)을 강제하는 회로 차단기 구현 필요.

## 3. 고급 기능 추가 제안

- **Entity-Centric 분석**: 단순 키워드가 아닌 **NER(개체명 인식)**을 통해 인물, 사건, 법률명을 추출하고 이들 간의 관계를 **Graph Database(Neo4j 등)**로 매핑하여 '사건의 전개 과정'을 추적하는 기능.
- **Multimodal Viral Detection**: YouTube의 경우 제목/설명뿐만 아니라 **썸네일 이미지의 텍스트와 분위기**를 분석하여 클릭 유도성(Click-bait)과 실제 바이럴 가능성을 분리 측정.
- **Temporal Trend Forecaster**: 현재 수집된 데이터의 기울기(Engagement Velocity)를 기반으로 **'3시간 뒤 가장 뜨거울 키워드'**를 예측하는 시계열 예측 모델 추가.

## 4. 성능 최적화 제안

- **Embedding Batching**: Semantic Dedup 시 기사 하나씩 임베딩하지 않고, **Batch 처리**를 통해 GPU(또는 Inference Server) 활용도를 극대화하여 지연 시간 단축.
- **KR-WordRank 사전 필터링**: 불필용어(Stopwords) 사전을 법률/시사 도메인에 특화하여 구축하고, 추출 전 정규표현식을 통한 노이즈 제거 단계를 강화하여 LLM에 전달되는 토큰 양 최소화.
- **Tiered Caching**: 동일 카테고리/시간 범위의 요청에 대해 **Redis 캐시**를 적용(TTL 5~15분)하여 중복 API 호출 및 연산 비용 절감.

## 5. 운영 안정성 제안

- **Drift Monitoring**: KR-WordRank나 스코어링 로직이 특정 시점에 편향된 결과(예: 광고성 기사 도배)를 내놓는지 모니터링하는 **Data Quality Dashboard** 구축.
- **Structured Logging & Tracing**: 각 소스별 수집 성공률, 지연 시간, 비용을 **OpenTelemetry** 등을 통해 추적하여 병목 지점 실시간 파악.
- **Human-in-the-loop (Optional)**: 랭킹 결과에 대해 관리자가 '부적절' 플래그를 달면 즉시 학습 데이터에 반영되어 이후 필터링 성능을 높이는 피드백 루프 생성.

## 6. 검색 품질 향상 전략 제안

- **Query Expansion (LLM 기반)**: 사용자가 입력한 단순 키워드를 검색 엔진용 **불리언 쿼리(Boolean Query)**로 확장 (예: "음주운전" -> "음주운전 OR (혈중알코올농도 AND 면허취소)").
- **Cross-Source Authority Weighting**: 단순 RRF가 아닌, 해당 카테고리(예: 법률)에서 권위 있는 소스(대법원 판결문, 전문 법률지)에 더 높은 가중치를 부여하는 **Domain-Specific Authority Score** 적용.
- **Contextual Reranking**: 뉴스 검색 결과 리스트를 사용자(변호사/크리에이터)의 페르소나에 맞춰 최종적으로 재정렬하는 **Cross-Encoder 모델** 활용.

## 7. 네이버 뉴스 베스트 능가 전략

- **Hyper-Personalization**: 네이버는 전 국민 대상 '범용' 베스트를 보여주지만, 본 시스템은 **'유튜브 컨텐츠 제작'이라는 특수 목적**에 최적화된 "제작 가치(Production Value)" 점수를 최우선으로 배치.
- **Early Signal Capture**: 네이버 뉴스에 올라오기 전, 커뮤니티(Tavily 수집분)와 Google Trends breakout 데이터를 조합하여 **네이버보다 30분~1시간 빠른 이슈 선점**.
- **Legal Perspective Injection**: 단순 사건 나열이 아닌, 해당 사건이 **어떤 법적 쟁점(Punishment, Precedent)**과 연결되는지를 요약하여 제공함으로써 정보의 깊이 차별화.

---

## 총평

제안된 설계안은 현재의 기술적 부채를 해결할 매우 전략적이고 구체적인 계획입니다. 특히 KR-WordRank를 활용한 비용 절감과 5차원 스코어링은 합리적입니다. 다만, **비동기 아키텍처의 부재와 외부 API 의존성 리스크**가 상용 수준으로 가기 위한 마지막 관문이 될 것입니다. 제안된 개선 사항들을 반영한다면 네이버 뉴스를 상회하는 전문적인 '버티컬 트렌드 플랫폼'으로의 도약이 충분히 가능합니다.
