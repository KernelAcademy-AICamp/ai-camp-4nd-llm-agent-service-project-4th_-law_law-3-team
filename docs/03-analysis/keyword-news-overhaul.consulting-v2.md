# 외부 컨설팅 재검증 보고서 (v1.2) — 키워드/뉴스 수집 시스템 전면 재설계

> **작성자**: Codex CLI (External Consultant, gpt-5.3-codex)
> **대상**: `docs/01-plan/features/keyword-news-overhaul.plan.md` v1.2
> **작성일**: 2026-03-01
> **Iteration**: 2

---

## 1. 변경사항별 업계 수준 대비 평가

| # | 항목 | 등급 | 상세 평가 |
|---|------|------|-----------|
| 1 | pytrends 제거 + 3계층 Provider Abstraction | **적절(상)** | `pytrends` 저장소가 2025-04-17에 아카이브되어 유지보수 리스크가 커졌기 때문에 제거 방향은 업계 기준에 부합. 공식 API 우선 + 상용 스크래핑 API fallback 구조는 실무적으로 타당. |
| 2 | Circuit Breaker (3회 실패, 5분 차단, Half-Open) | **적절(중상)** | 상태머신(Closed/Open/Half-Open) 설계 자체는 표준 패턴과 일치. 다만 "연속 3회" 고정 임계값은 트래픽 레벨별 민감도 튜닝 필요. |
| 3 | LLM Query Expansion | **적절(상)** | Query2doc/HyDE 계열 접근과 방향성이 맞고, 검색 재현율 개선에 효과적. 환각/주제 드리프트 방지 가드레일 필수. |
| 4 | YouTube Shorts viewCount 0.3 계수 | **조건부 적절(중)** | 2025-03-31 이후 Shorts 조회수 정의 변경(재생 시작만으로 카운트)으로 보정 필요성 매우 큼. 고정 0.3은 초기 대응으로 괜찮지만 채널/카테고리별 편차 미반영. |
| 5 | Early Signal Capture (Z>2.0 + 커뮤니티 급증) | **적절(중상)** | 탐지 민감도는 좋으나 노이즈가 많아 오탐 관리가 핵심. |
| 6 | 5차원 스코어링 가중치 | **적절(중상)** | 법률 도메인에서 `relevance + legal`을 높인 것은 타당. 단일 고정 가중치보다 시나리오별 동적 가중치가 업계 상위권 방식. |
| 7 | Semantic Dedup (cosine>0.80, 30건+) | **적절(중)** | 방향은 맞지만 0.80은 과병합 위험. 30건부터 활성화는 늦을 수 있음. |
| 8 | legal_score 3단계 fallback | **적절(상)** | 비용/속도/정확도 균형이 좋음. 단계별 confidence 전달과 최종 신뢰도 캘리브레이션 필요. |

## 2. 파라미터 튜닝 권고

| 파라미터 | 현재값 | 권고값 | 근거 |
|---------|--------|--------|------|
| Circuit Breaker 임계값 | 연속 3회 고정 | `실패율 50% + 최소처리량 20 + 60~120초 윈도우` 병행. 저트래픽 구간만 `연속 3회` 보조 | 트래픽 변동 대응 |
| Break Duration | 고정 5분 | `1분 → 5분 → 15분` 점증(backoff) | 장애 심각도 반영 |
| Half-Open 시험 요청 | 미명시 | `1~3건`만 허용 후 닫힘 전환 | 리스크 최소화 |
| Query Expansion 수 | 미제한 | 최대 3개, 원문 키워드 1개 강제 포함 | 주제 드리프트 방지 |
| Shorts 계수 | 고정 0.3 | 기본 0.3 + `engaged_views / views` 비율 동적 보정 (0.15~0.60) | 카테고리별 편차 반영 |
| Early Signal Z-score | > 2.0 | 초기 2.0, 안정화 후 2.3~2.5 상향. `최소 30건/시간` 볼륨 게이트 추가 | 오탐 감소 |
| 5차원 가중치 | 고정 (0.25/0.25/0.15/0.20/0.15) | 초기값 유지 + 법률 긴급이슈 모드(`legal + recency` 상승 프로파일) 별도 운영 | 시나리오 대응 |
| Semantic Dedup 임계값 | cosine > 0.80, 30건 | `>0.90 자동 병합`, `0.82~0.90 검증 병합`. 활성화 임계 `10~15건`으로 하향 | 과병합 방지 + 조기 활성화 |
| legal_score 최종 합산 | max(keyword, RAG) | `가중 결합(keyword 0.2, RAG 0.4, LLM 0.4)` + 단계별 confidence 보존 | 안정화 |

## 3. 누락 기능 제안

1. **온라인 평가 체계**: Precision@K, 알림 적중률, 오탐비율, 탐지 지연시간 SLA를 실시간 대시보드화
2. **드리프트/품질 감시**: 데이터 소스별 분포 변동, 모델 출력 분산, 임계값 자동 재보정
3. **근거 추적성(Provenance)**: 각 스코어의 증거 문서/타임스탬프/출처 신뢰도 저장 (감사 대응용)
4. **조작 방어**: 봇/어뷰징 트래픽 및 조회수 인플레이션 탐지 계층
5. **실험 프레임워크**: 가중치/임계값 A/B 테스트 및 자동 승격 파이프라인
6. **비용 통제**: LLM 호출 예산, 캐시 히트율, fallback별 단가 KPI 연동
7. **인시던트 운영**: Provider 장애 시 강등 정책(runbook), 재처리(replay) 큐, 배치 백필 전략

## 4. 최종 평가

- **종합 판단**: v1.2는 v1.1 대비 구조적 성숙도가 뚜렷하게 상승. 특히 데이터 소스 안정성/복원력/법률 점수화 측면에서 업계 평균 이상.
- **현재 등급**: `B+ (실전 투입 가능, 단 파라미터/운영계측 보강 필요)`
- **v1.3 우선순위**: `동적 임계값 + 근거추적 + 온라인 평가/AB 테스트`를 최우선으로 권고.

---

## 참고 자료

- pytrends 아카이브: https://github.com/GeneralMills/pytrends
- Circuit Breaker 패턴: https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker
- Martin Fowler Circuit Breaker: https://martinfowler.com/bliki/CircuitBreaker.html
- Query2doc: https://arxiv.org/abs/2303.07678
- HyDE: https://arxiv.org/abs/2212.10496
- YouTube Shorts viewCount 변경: https://support.google.com/youtube/answer/9072033?hl=en
- Naver DataLab API: https://developers.naver.com/docs/serviceapi/datalab/search/search.md
- SerpAPI Google Trends: https://serpapi.com/google-trends-api
