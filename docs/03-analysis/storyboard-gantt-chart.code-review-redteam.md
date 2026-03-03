# Red Team 코드 리뷰 보고서: 스토리보드 간트차트

> 검증 대상: Phase 1-3 구현 코드 (Backend + Frontend)
> 검증 도구: Gemini CLI (Red Team, gemini-2.5-pro)
> 검증일: 2026-03-01

---

## Red Team 코드 리뷰 보고서

### 1. 보안 취약점 (Critical/High/Medium/Low)

| 등급 | 항목 | 위치 | 조치 |
|:---:|---|---|---|
| **Critical** | 인증/인가 부재 | router/__init__.py 전역 | 기존 알려진 이슈 (MVP 범위 외), Phase 0에서 해결 예정 |
| **High** | Prompt Injection | batch_analyzer.py `context` | **수정 완료** - 길이 제한 + 경고 프리픽스 |
| **High** | Zip Bomb (DOCX) | batch_analyzer.py `_extract_text_from_document` | **수정 완료** - 읽기 크기 제한 (10MB) |
| **Medium** | XSS (vis-timeline content) | useGanttChart.ts `timelineItemsToVisItems` | **수정 완료** - HTML 이스케이프 적용 |
| **Medium** | 이벤트 루프 블로킹 | file_validation.py, batch_analyzer.py | **수정 완료** - asyncio.to_thread 적용 |

### 2. 버그 및 논리적 오류

| 항목 | 위치 | 조치 |
|---|---|---|
| 날짜 중복 판단 YYYY-MM 하드코딩 | timeline_merger.py `_normalize_date_prefix` | **수정 완료** - datetime 기반 ±30일 비교 |
| 모바일 카카오톡 날짜 헤더 전 메시지 무시 | kakao_parser.py | **수정 완료** - 날짜 없음 fallback 처리 |
| LLM JSON 파싱 불안정 | timeline_merger.py | **수정 완료** - 다단계 파싱 fallback |
| 배치 실패 파일 추적 불일치 | batch_analyzer.py `_merge_results` | **수정 완료** - 실패 파일 직접 추적 |

### 3. 성능 이슈

| 항목 | 조치 |
|---|---|
| 대용량 파일 메모리 점유 (250MB/배치) | 인지됨, 향후 스풀링 도입 예정 |
| Gantt DataSet clear+add 전체 교체 | 인지됨, DataSet.update() 최적화 향후 적용 |
| SSE 연결 정리 | useEvidenceUpload.ts cleanup 로직 확인 완료 |

### 4. 코드 품질 개선 제안

- BatchAnalyzer Strategy 패턴 분리 → 향후 리팩토링
- Zod 스키마 런타임 검증 도입 → 향후 적용
- 에러 응답 Result 객체 래핑 → 향후 적용

### 5. 총평

SSE 기반 실시간 업데이트 아키텍처는 우수하나, 보안(인증 부재, Prompt Injection)과 안정성(날짜 비교 버그, Zip Bomb) 측면에서 수정이 필요했습니다. 즉시 수정 가능한 6건의 이슈를 모두 해결했습니다.
