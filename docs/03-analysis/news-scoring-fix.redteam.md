# Red Team 검증 보고서 - News Article Scoring Fix

## 검증 대상
- `docs/01-plan/features/news-scoring-fix.plan.md`

## 검증 도구
- Gemini CLI (Red Team)

---

## 1. 취약점 (Critical/High/Medium/Low)

### Medium - 인메모리 Rate Limit의 분산 환경 취약성
- 현재 `InMemoryRateLimiter`는 단순 Python `dict` 기반 프로세스 단위 메모리 저장 방식
- **위험**: 멀티 워커/컨테이너 스케일 아웃 환경에서 사용자별 제한이 워커마다 독립 적용
- 유료 API(Tavily, Naver 등) 비용 폭증 및 백엔드 자원 고갈 가능

### Low - 키워드 Sanitize 우회 및 주입 위협
- HTML 태그 제거 + 공백 정규화 추가 시, 뉴스 API snippet의 악의적 스크립트 → XSS 위험
- 키워드 정규화 로직 복잡화 → 검색 엔진 주입(Search Injection) 차단 어려움

### Low - 로그를 통한 민감 정보 노출
- `sanitize_keyword`가 차단 키워드의 앞 30자를 평문 로그 기록 → PII 노출 리스크

---

## 2. 아키텍처 개선 제안

### 랭킹 동질성 문제 (Ranking Homogeneity)
- **문제**: Option A는 모든 기사에 동일한 `related_laws` 일괄 적용 → 기사 간 legal_score 변별력 상실
- **제안**: 기사별 title+snippet과 법령명 간의 텍스트 유사도(Jaccard 등) 가산점 추가

### Rate Limiter 중앙 집중화
- 운영 환경에서 Redis 기반 분산 Rate Limiter로 교체 필요

### 서비스 결합도 완화
- `content_marketing_service.py`가 캐싱, Rate limit, 뉴스 수집, RAG, 재계산을 모두 처리
- scoring 로직을 `article_scorer.py` 내부의 독립 워크플로우로 분리 권장

---

## 3. 고급 기능 추가 제안

### Semantic Relevance Scoring
- 경량 임베딩 모델(MiniLM 등)로 키워드-기사 간 의미적 유사도 산출
- "부동산 사기"↔"전세 사기" 유의어 대응 가능

### LLM 기반 동적 가중치
- 키워드 성격(시사성 vs 법리적)에 따라 가중치 자동 조정

### 사용자 피드백 루프 (Active Learning)
- 클릭/관련없음 데이터 수집 → 가중치 자동 튜닝

---

## 4. 성능 최적화 제안

### RAG 결과 별도 캐싱
- 관련 법령은 고정적 → 키워드별 RAG 결과 별도 캐싱으로 검색 빈도 감소

### 병렬 파이프라인 최적화
- 기사 10건의 스니펫을 배치로 묶어 임베딩 서버에 한 번 요청 방식 권장

### In-place Update 활용
- `model_copy(update=...)` 대신 필드 직접 업데이트로 메모리 오버헤드 감소

---

## 5. 운영 안정성 제안

### 관측 가능성(Observability) 강화
- 키워드별 상위 3개 기사의 평균 total_score/legal_score 분포 모니터링
- 뉴스 소스별 응답 시간 + RAG 검색 시간 구분 기록

### Graceful Degradation 고도화
- Vector DB 장애 시 법률 키워드 사전 활용 최소 legal_score 보장

### 서킷 브레이커 (Circuit Breaker)
- 외부 뉴스 API 및 LanceDB 호출부에 서킷 브레이커 적용
