# 외부 컨설팅 보고서 - 법률 뉴스 통계 수정 기획서

**검토일**: 2026-02-27
**검토 도구**: Codex CLI (gpt-5.3-codex, External Consultant)
**대상**: `docs/01-plan/features/legal-news-stats-fix.plan.md`
**판정**: **조건부 승인 (Proceed with revisions)**

---

## 1. 모범 사례 대비 격차 (Recharts 라벨/레이아웃)

| 항목 | 판정 | 상세 |
|------|------|------|
| F1 Stacked→Grouped | **적절 (P0)** | 비교 목적 차트에는 grouped가 표준적 |
| F2 Bar LabelList | **적절 (P0)** | 30일 구간에서 라벨 충돌 가능 → `margin.top`, `barSize`, `minTickGap` 동시 조정 필요 |
| F3 도넛 라벨 | **부분 적절 (P0)** | 모바일(260px)에서 긴 한글 카테고리 라벨 겹침 위험. `height 300`, `outerRadius 80` 조정 필수 |
| F6 tooltip formatDate | **적절 (P1)** | `Date` 파싱 대신 문자열 기반 포맷이 더 안전 |
| F7 KST 보간 | **필수 (P1)** | "date-only 문자열은 KST 기준으로 일자 키 생성" 규칙 문서화 권장 |

## 2. UX/접근성 개선 방안

| # | 제안 | 우선순위 |
|---|------|:--------:|
| C1 | 차트 컨테이너에 `role="img"` + `aria-label` 추가 | P2 |
| C2 | 기간 버튼에 `aria-pressed` 적용 (색상 의존 제거) | P2 |
| C3 | 범례에 건수 병기 (색상만으로 구분 방지, 특히 도넛) | P2 |
| C4 | 모바일 도넛: 상위 N개만 라벨 + 나머지 툴팁/범례 전략 | P2 |

## 3. 코드 품질/유지보수성

- **F4 total 쿼리 제거**: 강하게 권장. 불필요한 DB round-trip 제거로 단순/효율 개선.
- **F5 라우트 순서 주석**: 권장. `article_id` 타입 제약(UUID/regex)까지 추가하면 순서 의존 리스크 감소.
- **날짜 포맷 함수 분산**: 차트용 `formatChartDate(dateKey)` 유틸로 통합 권장 (드리프트 방지).
- **카테고리 분류 이중화**: Python/SQL SSOT 정리 과제는 별도 트래킹 필요.

## 4. 추가 개선 제안

| # | 제안 | 우선순위 |
|---|------|:--------:|
| C5 | F2/F3 라벨 충돌 테스트 기준 명시 (320px, 380px, 30일 스냅샷) | P1 |
| C6 | `/stats/daily` 회귀 테스트: total 정확성, KST 경계(00:00~09:00) 케이스 | P1 |
| C7 | `func.date(published_at)` 함수형 인덱스 전략 백로그 등록 | P2 |
| C8 | 통계 API 응답에 `sources` 메타(포함 소스 목록) 추가 | P2 |

## 5. 결론

F1~F4는 즉시 효과가 크며, F3/F7은 레이아웃·타임존 구현 디테일 보완이 필요.
7건 전체 방향은 적절하며, 30일 구간 라벨 충돌과 모바일 도넛 라벨 겹침 보완 후 머지 가능.
