# 외부 컨설팅 보고서 — 키워드/뉴스 수집 시스템 전면 재설계

> **작성자**: Codex CLI (External Consultant, gpt-5.3-codex)
> **대상**: `docs/01-plan/features/keyword-news-overhaul.plan.md`
> **작성일**: 2026-03-01

---

## 1. 격차 분석

현재 기획은 "문제 정의"는 매우 정확하지만, 업계 상위권(vidIQ/TubeBuddy류 + 최신 하이브리드 검색 스택) 대비 아직 5가지 핵심 격차가 있습니다.

1. **채널 맞춤형 난이도/기회 점수 부재**
   상위 도구는 `검색량 x 경쟁도`에 더해 채널별 가중치(Weighted score)를 제공합니다. 현재 안은 글로벌 점수 중심이라 채널 성장단계별 최적 주제가 다르게 추천되지 않습니다.

2. **조회수 중심 편향, 시청지속 신호 부족**
   YouTube는 `views` 외에 `averageViewDuration`, `engagedViews` 등 핵심 지표를 제공합니다. 특히 Shorts `viewCount` 정의가 2025-03-31에 변경되어(재생 시작/재생 포함) 조회수 단독 최적화 리스크가 더 커졌습니다.

3. **트렌드 소스 안정성 리스크** ⚠️ CRITICAL
   `pytrends` 저장소가 **2025-04-17 아카이브(읽기 전용)**되었습니다. 반면 Google Trends 공식 API는 2025-07-24 알파 공개지만 접근 제한형입니다. 즉, 의존성 전략을 재설계해야 합니다.

4. **랭킹 결합 고도화 여지**
   RRF 도입 방향은 맞습니다. 다만 상위권은 BM25/벡터/신뢰도/신선도 가중치를 분리 운영하며, 실험적으로 상수(k), 윈도우, 소스 가중치를 튜닝합니다.

5. **운영 관점(실험·품질·계보) 미흡**
   데이터 검증(quality checkpoint), 계보(lineage), 실험 프레임(A/B)이 명시되지 않아 "정확도 개선"이 지속 가능하게 증명되기 어렵습니다.

## 2. 기술 고도화 제안

1. 트렌드 수집 계층을 **Provider Abstraction**으로 분리
   `Google Trends(공식 API 알파 접근 시)` / `대체 소스` / `장애 시 graceful fallback` 3계층으로 설계.

2. 랭킹을 **"3단 융합"**으로 확장
   `BM25 + Dense Vector + Source Trust`를 1차 결합하고, 최종 순위를 RRF로 재정렬.

3. legal_score를 RAG 단일 의존에서 탈피
   `규칙기반 법률 엔티티 매칭 + 판례/법령 인용 밀도 + LLM 판정` 앙상블로 변경.

4. 실험 체계 내장
   오프라인(NDCG/MRR/Precision@k) + 온라인(클릭률, 평균시청시간, 제작전환율) 이중 평가.

## 3. UX 개선 제안

1. **"왜 이 주제인가" 설명형 카드**: 근거 소스, 급상승 원인, 예상 성과, 리스크(법적 민감도) 한 화면 제공
2. **제작 워크플로우 단축**: 키워드 → 뉴스 클러스터 → 썸네일/제목 후보 → 스크립트 초안까지 원클릭
3. **신뢰 UX**: 뉴스/법령 근거 링크, 수집 시점, 중복 제거 사유 표시

## 4. 데이터 파이프라인 고도화

1. **Bronze/Silver/Gold 레이어**: 원천(raw) / 정규화(clean) / 추천(feature-ready) 분리
2. **데이터 품질 게이트**: Great Expectations 체크포인트 자동화
3. **계보/관측성**: OpenLineage 추적
4. **병렬 수집 최적화**: Airflow Dynamic Task Mapping

## 5. 비즈니스 전략 제안

1. **수익화 3단계**: SaaS 구독 → Pro(법률사무소) → API(B2B)
2. **차별화**: 법률 특화 정확도 + 근거 추적 + 리스크 필터링
3. **확장**: 한국어 PMF → 국가별 법체계 모듈 확장

---

## 참고 자료

- YouTube Data API: https://developers.google.com/youtube/v3/docs/videos
- YouTube Analytics metrics: https://developers.google.com/youtube/analytics/metrics
- Google Trends API alpha: https://developers.google.com/search/apis/trends
- pytrends 아카이브: https://github.com/GeneralMills/pytrends
- RRF: https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
- vidIQ Keyword Research: https://support.vidiq.com/en/articles/9421214-keywords-research
- Great Expectations: https://docs.greatexpectations.io/docs/reference/api/Checkpoint_class
- OpenLineage: https://openlineage.io/
