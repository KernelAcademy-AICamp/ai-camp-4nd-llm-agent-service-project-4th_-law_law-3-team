# Red Team 설계 검증 보고서: 스토리보드 간트차트

> 검증 대상: `docs/02-design/features/storyboard-gantt-chart.design.md`
> 검증 도구: Gemini CLI (Red Team, gemini-2.5-pro)
> 검증일: 2026-03-01

---

## Red Team 설계 검증 보고서

본 보고서는 **Storyboard Gantt Chart (사건 타임라인 간트차트)** 설계안에 대해 시니어 아키텍트 및 Red Team 기술 전문가 관점에서 수행한 보안 및 기술 검증 결과입니다.

---

### 1. 취약점 분석 (Vulnerabilities)

| 등급 | 항목 | 설명 및 위험성 | 대응 방안 제안 |
|:---:|---|---|---|
| **High** | **SSE Connection Leak** | `JobManager`의 `_subscribers` 관리가 `try...finally`로 되어있으나, 클라이언트가 비정상 종료되거나 타임아웃 발생 시 Queue가 잔류할 가능성이 있음. | `JobManager`에 하트비트(Heartbeat) 메커니즘을 추가하고, 일정 시간 응답 없는 구독자를 강제 제거하는 Cleanup 루틴 강화 필요. |
| **High** | **Timeline Merge Conflict Bypass** | `TimelineMerger`에서 LLM 유사도 기반 중복 감지 시, 악의적인 공격자가 정교하게 조작된 텍스트를 주입하여 기존 증거를 덮어쓰거나 무력화할 위험(Data Integrity). | 병합 전 '변경 사항 확인(Diff/Review)' 단계를 UI에 강제하고, 중요 필드 변경 시 감사 로그(Audit Log)를 남겨야 함. |
| **Medium** | **Zip Slip / Path Traversal** | `FileValidationGate`에서 `PK` 매직 넘버(DOCX 등) 처리 시 압축 해제 과정이 명시되지 않음. 압축 해제 시 상위 디렉토리 참조 공격 위험. | Python의 `zipfile` 사용 시 `extractall` 대신 각 파일의 경로를 검증 후 개별 추출하거나, 샌드박스화된 파서 사용 권장. |
| **Medium** | **LLM Token Explosion (DoS)** | 카카오톡 `.txt` 파싱 시 슬라이딩 윈도우를 사용하되, 무한히 긴 단일 메시지나 특수 기호 반복 시 토큰 제한을 초과하여 API 비용 및 응답 지연 초과 위험. | 윈도우 분할 전 텍스트 청크 당 최대 길이(Max length)와 토큰 수 사전 검사(tiktoken 등) 로직을 `batch_analyzer`에 추가. |
| **Low** | **Insecure Direct Object Reference (IDOR)** | `evidence_id`가 UUID v4지만, `GET /evidence/{id}` 호출 시 실제 세션 사용자가 해당 타임라인의 소유자인지 확인하는 로직이 설계상 추상적임. | `EvidenceFile` 모델에 `owner_id` 또는 `session_id`를 명시적으로 연결하고 Router 레벨에서 소유권 검증 로직 강제화. |

---

### 2. 아키텍처 개선 제안

- **JobManager의 영속성(Persistence) 결여**: 현재 `JobManager`는 인메모리(`_jobs: dict`) 기반입니다. 서버 재시작 시 진행 중인 배치 분석 작업이 모두 소실됩니다. 대용량 파일 처리가 포함되므로 **Redis나 DB를 이용한 상태 관리**로 전환하거나, 최소한 작업 결과를 즉시 영속화하는 구조가 필요합니다.
- **Worker 구조 분리**: `asyncio.create_task`는 단기 작업에 적합하나, 대용량 카카오톡 파싱이나 Whisper STT는 CPU/IO 집약적입니다. 트래픽 증가를 대비해 **Celery + Redis/RabbitMQ** 워커 분리 아키텍처를 로드맵에 포함해야 합니다.
- **Dependency Injection 강화**: `BatchAnalyzer`가 `KakaoTalkParser` 등을 직접 참조하지 않고, `Registry` 패턴을 통해 분석기를 동적으로 로드하는 구조를 제안합니다 (새로운 증거 유형 추가 시 OCP 준수 용이).

---

### 3. 누락 항목 및 엣지 케이스 (Edge Cases)

- **증거 파일 삭제 시 참조 무결성**: `EvidenceFile`이 삭제될 때 이를 참조하고 있는 `TimelineItem`의 `evidence_ids` 처리 로직이 누락되었습니다 (Cascading delete vs. Nullify).
- **날짜 파싱 실패 시 처리**: 카카오톡 메시지 중 "날짜 헤더"가 유실되거나 손상된 경우, 이후 모든 메시지의 타임라인이 왜곡될 수 있습니다. `KakaoTalkParser`에 "날짜 불확실성 전파 방지" 로직이 필요합니다.
- **멀티 유저 환경의 간트차트**: 동일한 `job_id`에 대해 여러 브라우저 탭에서 SSE를 구독할 때의 동기화 이슈.

---

### 4. 데이터 모델 검증

- **`TimelineItem`의 하위 호환성**: 기존 `participants: list[str]`와 신규 `participants_detailed: list[Participant]`가 공존합니다. API 응답 시 한쪽이 누락되어도 프론트엔드에서 깨지지 않도록 `Participant` 모델로의 **자동 승격(Promotion) 로직**이 백엔드 시리얼라이저에 필요합니다.
- **`EvidenceFile` 해시 충돌**: SHA-256은 안전하지만, 파일 내용이 1바이트만 달라도 새 파일로 인식합니다. 법률 증거 특성상 "완전 동일 파일" 외에 "유사 파일" 감지를 위한 메타데이터(파일 크기 + 이름 + 생성일 조합) 보완을 제안합니다.

---

### 5. API 설계 검증

- **RESTful 준수**: `POST /analyze-batch`는 리소스를 생성하므로 `201 Created`가 더 적절하며, Location 헤더에 `job_id` 경로를 포함해야 합니다.
- **에러 핸들링**: 422 Unprocessable Entity 발생 시, 단순히 "검증 실패"가 아니라 **어떤 파일이 왜 실패했는지(예: 파일 3번 - 매직넘버 불일치)**에 대한 상세 필드 에러 구조가 응답에 포함되어야 합니다.

---

### 6. 프론트엔드 설계 검증

- **State 관리 복잡도**: `useTimelineState`와 `useGanttChart` 간의 데이터 동기화 이슈가 예상됩니다. 특히 병합(`merge`) 후 데이터 업데이트 시 간트차트의 `DataSet`을 어떻게 효율적으로 Re-render할지(전체 교체 vs 부분 업데이트) 명확한 전략이 필요합니다.
- **UX - 줌 레벨 최적화**: 법률 사건은 수년에 걸친 장기 사건과 수분 내의 폭행 사건이 공존합니다. `zoomMin/zoomMax`를 고정하기보다 **데이터의 시간 범위에 따라 초기 줌 레벨을 동적으로 설정**하는 로직이 누락되었습니다.

---

### 7. 성능 최적화 제안

- **Frontend**: `vis-timeline`에 500개 이상의 이벤트를 렌더링할 때 `stack: true` 옵션은 레이아웃 계산 부하가 큽니다. 항목이 많을 경우 **Virtual Scrolling** 또는 **주제별 그룹 클러스터링** 옵션을 고려해야 합니다.
- **Backend**: `asyncio.gather`의 `Semaphore=3`은 적절하나, Whisper STT 등 GPU/CPU 자원을 많이 쓰는 작업은 `ProcessPoolExecutor`를 사용하여 이벤트 루프 블로킹을 방지해야 합니다.

---

### 8. 고급 기능 추가 제안 (Advanced Features)

1. **증거 사슬(Chain of Custody) 시각화**: 간트차트에서 특정 증거 마커를 클릭하면, 해당 증거가 영향을 미친 모든 타임라인 항목을 하이라이트하는 '증거 영향력 뷰'.
2. **AI 누락 탐지**: 타임라인 사이의 간격이 비정상적으로 길 경우(예: 3개월간 기록 없음), LLM이 "이 기간의 증거가 누락되었을 수 있습니다"라는 인사이트를 간트차트 Y축에 표시.
3. **PII Masking**: 업로드된 카카오톡 대화나 이미지에서 실명, 전화번호, 계좌번호를 자동으로 비식별화 처리하는 보안 옵션.

---

**총평**: 본 설계는 기존 모듈 구조를 존중하면서도 강력한 시각화와 보안 게이트를 도입하여 매우 견고합니다. 위에 제시된 **SSE 연결 관리, 작업 영속성, 그리고 데이터 참조 무결성** 부분만 보완된다면 즉시 구현 가능한 수준의 완성도를 갖추고 있습니다.
