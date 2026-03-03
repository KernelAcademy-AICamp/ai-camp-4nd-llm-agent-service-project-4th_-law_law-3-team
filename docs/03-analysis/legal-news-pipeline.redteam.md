# Red Team 검증 보고서: Legal News Pipeline

> **검증 도구**: Gemini CLI (Red Team)
> **검증 대상**: `docs/01-plan/features/legal-news-pipeline.plan.md` (v0.1)
> **검증 일시**: 2026-02-26

---

## Red Team 검증 보고서

### 1. 취약점 (Vulnerabilities)

| 등급 | 항목 | 설명 및 리스크 |
| :--- | :--- | :--- |
| **Critical** | **저작권 및 법적 리스크** | 법률신문 등 전문지는 데이터 자산 보호에 매우 엄격. 단순 크롤링 및 AI 요약본의 저작권법상 '2차적 저작물' 해당 여부 및 영리 목적 서비스 이용 시 법적 분쟁 가능성. |
| **High** | **개인정보보호(PII) 유출** | 뉴스 본문에 포함된 판사, 검사, 변호사, 피고인 실명 및 민감 정보가 정제 없이 벡터 DB에 저장되어 RAG 응답으로 출력될 경우 개인정보보호법 위반 소지. |
| **Medium** | **프롬프트 인젝션 (Hallucination)** | 기사 내용과 무관한 법령/판례가 요약 과정에서 생성될 경우, 서비스 신뢰도에 타격. |
| **Medium** | **소스 코드 및 API Key 노출** | `.env` 관리만으로는 부족. 크롤러 패턴을 통해 IP 차단(IP Ban) 당할 경우 파이프라인 전체 마비. |
| **Low** | **중복 제거 로직의 한계** | 제목/해시 기반 중복 제거는 동일 사건에 대한 다른 매체의 기사(어뷰징 기사)를 걸러내지 못함. |

### 2. 아키텍처 개선 제안

- **비동기 병렬 처리 도입**: 직렬 구조 → Task Queue (Celery/RabbitMQ or Temporal) 도입하여 수집-정제-요약-임베딩 비동기 파이프라인 분리
- **LLM 추상화 레이어 및 Fallback 전략**: Solar-Pro2 단일 의존성 탈피, Circuit Breaker 패턴으로 장애 시 다른 모델로 스위칭
- **Vector DB 확장성 확보**: LanceDB 로컬 파일 기반 → 다중 노드 서비스 시 동기화 이슈 검토 필요

### 3. 고급 기능 추가 제안

- **PII Masking 모듈 추가**: 정제-요약 사이에 NER 모델 배치, 인명/전화번호/주소 마스킹
- **Cross-Reference 검증**: 요약 내 법령/판례 번호가 실제 DB에 존재하는지 유효성 검증
- **기사 신뢰도 스코어링**: 판결 확정 여부 등 메타데이터 추출, 검색 가중치 반영

### 4. 성능 최적화 제안

- **Incremental Embedding**: content_hash 비교로 변경 문서만 임베딩 갱신
- **Semantic Deduplication**: 임베딩 벡터 유사도 95% 이상 → '동일 이슈' 그룹핑
- **Local LLM 활용**: 단순 정제/분류 작업은 경량 모델 활용 (비용 절감)

### 5. 운영 안정성 제안

- **Dead Letter Queue**: 파싱/요약 실패 기사 별도 관리 + 재처리 CLI
- **통합 알림 시스템**: Critical Fail 시 Slack/Teams 웹훅 알림
- **Data Lineage**: 소스 기사 원문 + 프롬프트 버전 트래킹, Back-testing 지원
- **Health Check Endpoint**: 마지막 가동 시간, 성공률, DB 연결 상태 모니터링
