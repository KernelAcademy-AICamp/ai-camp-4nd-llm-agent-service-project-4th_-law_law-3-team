# 외부 컨설팅 보고서: 스토리보드 간트차트 기획

> 검증 대상: `docs/01-plan/features/storyboard-gantt-chart.plan.md`
> 검증 도구: Codex CLI (External Consultant, gpt-5.3-codex)
> 검증일: 2026-03-01

---

## 외부 컨설팅 보고서

전제: 기획서 원문 전체가 아니라 제공된 컨텍스트 기준으로 외부 벤치마크 검증을 수행했습니다.

### 1. 격차 분석 (현재 vs 목표)

종합평가: 현재는 `기능 구현형(타임라인 시각화)`에 강점이 있고, 업계 최고 수준은 `근거 추적형(검증·협업·업무통합)`까지 포함합니다. 최신 트렌드 반영도는 **부분 반영(약 60%)**으로 판단됩니다.

| 영역 | 현재(추정) | 목표(업계 최고 수준) | 격차 |
|---|---|---|---|
| 사건 타임라인 | 간트차트 + 증거 통합 | Fact 중심 타임라인, 증거-사실 자동 연결, 충돌 탐지 | 큼 |
| AI 결과 신뢰성 | 요약/생성 중심 | 출처 스팬 인용, 근거 클릭 추적, 반증 제시 | 매우 큼 |
| 워크플로우 통합 | 단일 기능 중심 | 리서치-검토-작성-협업 end-to-end | 큼 |
| 멀티모달 | 파일 업로드 중심 | 문서+이메일+음성/증언+이미지 통합 추출 | 중간~큼 |
| 거버넌스/컴플라이언스 | 불명확 | 윤리규정 대응, 감사로그, HITL 승인체계 | 매우 큼 |
| 상용화 구조 | 기능 판매형 | 좌석+사용량+사건 단위 하이브리드 과금 | 큼 |

### 2. 기술 고도화 제안

1. **Evidence Ingestion Layer 별도 모듈화**: PDF/OCR/이메일/음성(ASR) 입력을 공통 이벤트 스키마(주체-행위-시점-근거스팬)로 정규화하면 타임라인 품질이 급상승합니다.

2. **Hybrid Retrieval + Temporal Graph 도입**: 벡터 검색만으로는 법률 키워드 정확도가 떨어질 수 있어 BM25+Dense 하이브리드와 사건 시계열 그래프를 함께 쓰는 구조가 필요합니다.

3. **LangGraph 상태 영속 + Human Checkpoint 중심 재설계**: 핵심 의사결정 노드마다 체크포인트/승인 단계를 두면 고위험 도메인에서 신뢰를 확보할 수 있습니다.

4. **근거 기반 응답 게이트**: 인용 가능한 근거가 부족하면 답변 대신 "근거 부족"으로 반환하는 정책이 필수입니다.

5. **관측성(Observability) 제품 핵심 승격**: 에이전트 트레이스, 실패 유형, 재시도 로그를 표준화하면 운영 난이도가 크게 낮아집니다.

### 3. UX 개선 제안

1. **간트차트 근거 신뢰도 배지**: 각 이벤트마다 신뢰도/근거개수/최근검증일을 표시하면 변호사 의사결정 속도가 빨라집니다.

2. **타임라인-원문 양방향 탐색**: 이벤트 클릭 시 원문 스팬 하이라이트, 원문 선택 시 이벤트 후보 생성.

3. **충돌/누락 인사이트 패널**: 날짜 모순, 동일인물 이명, 증거 간 불일치 자동 알림이 차별점이 됩니다.

4. **협업 UX 강화**: 사건 단위 코멘트, 버전 비교, 승인 워크플로우가 없으면 실제 법무팀 도입 장벽이 큽니다.

5. **모바일 조회 전용 경량화**: 현장 확인/브리핑 중심으로 설계하고, 편집은 데스크톱 중심으로 분리하는 것이 효율적입니다.

### 4. AI/ML 고도화 제안

1. **학습 목표를 "법률 이벤트 추출"로 재정의**: KPI를 `이벤트 F1`, `날짜 정규화 정확도`, `증거-이벤트 링크 정확도`로 관리해야 합니다.

2. **오프라인+온라인 2단 평가체계 구축**: LegalBench/CUAD 같은 공개 벤치마크 + 실제 사건 로그 기반 커스텀 평가셋을 병행하세요.

3. **Continuous Evals 파이프라인 고정**: 모델/프롬프트/에이전트 변경 때마다 자동 회귀평가를 돌려 성능 하락을 차단하세요.

4. **Trace Grading 멀티에이전트 품질 점검**: 최종 답변만 보지 말고 중간 도구호출·추론흐름까지 평가해야 원인분석이 가능합니다.

5. **AI 거버넌스를 제품 기능으로 구현**: PII 마스킹, 모델별 데이터 사용정책, 사용자 역할 기반 권한 제어, 감사로그는 필수입니다.

### 5. 비즈니스 전략 제안

1. **시장 진입은 소송팀/분쟁팀부터**: 타임라인·증거 통합의 ROI가 명확해 초기 전환율이 높습니다.

2. **좌석 + 사용량 + 사건 단위 혼합 과금 모델**: 대형 고객은 보통 예측 가능한 사건 단위 과금을 선호합니다.

3. **상품 구조 3단계 분리**: `Starter(시각화)` / `Pro(에이전트+협업)` / `Enterprise(보안·감사·온프레미스)`.

4. **API/화이트라벨 수익화 확장**: 기존 로펌 시스템·문서관리시스템과 연동하는 B2B2B 전략이 확장성이 큽니다.

5. **12개월 목표 KPI**: `타임라인 생성 시간 50% 단축`, `근거 인용 정확도 95%+`, `주간 활성 사용자 60%+`, `유료 전환율`을 핵심 지표로 운영하세요.

---

### 참고 근거 (2026-03-01 기준)

- [Everlaw Fact Timelines (2025-12)](https://support.everlaw.com/hc/en-us/articles/42469701435803-Storybuilder-Fact-Timelines)
- [Everlaw Evidence/Timeline (2026-02)](https://support.everlaw.com/hc/en-us/articles/360038816812-Story-Timeline)
- [Relativity aiR for Review](https://help.relativity.com/RelativityOne/Content/Relativity/aiR_for_Review/aiR_for_Review.htm)
- [Thomson Reuters CoCounsel Legal (2025-08)](https://www.thomsonreuters.com/en/press-releases/2025/august/thomson-reuters-launches-cocounsel-legal-transforming-legal-work-with-agentic-ai-and-deep-research)
- [Thomson Reuters 2025 GenAI 보고](https://www.thomsonreuters.com/en/press-releases/2025/april/from-incubation-to-integration-generative-ai-adoption-nearly-doubles-as-professional-services-reach-crossroads)
- [LexisNexis Protégé (2025-01)](https://www.lexisnexis.com/community/pressroom/b/news/posts/lexisnexis-introduces-protege-personalized-ai-assistant-with-agentic-ai-making-it-easier-to-power-complex-legal-task-completion)
- [ABA Formal Opinion 512 (2024-07)](https://www.americanbar.org/news/abanews/aba-news-archives/2024/07/aba-issues-first-ethics-guidance-ai-tools/)
- [NIST AI RMF 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)
- [EU AI Act](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
- [OpenAI Evaluation Best Practices](https://platform.openai.com/docs/guides/evaluation-best-practices)
- [LangGraph Checkpoint](https://langchain-ai.github.io/langgraphjs/reference/modules/langgraph-checkpoint.html)
- [LegalBench](https://hazyresearch.stanford.edu/legalbench/)
- [CUAD](https://www.atticusprojectai.org/cuad)
- [Clio 2025 법무 트렌드](https://www.clio.com/about/press/clios-2025-legal-trends-for-mid-sized-law-firm-report/)
