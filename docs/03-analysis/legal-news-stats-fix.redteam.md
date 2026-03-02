# Red Team 검증 보고서 - 법률 뉴스 통계 수정 기획서

**검토일**: 2026-02-27
**검토 도구**: Gemini CLI (Red Team)
**대상**: `docs/01-plan/features/legal-news-stats-fix.plan.md`

---

## 1. 취약점 (Security Analysis)

- **SQL Injection (Low)**: SQLAlchemy Expression Language 사용으로 안전. `article_id`도 파라미터 바인딩 처리됨.
- **XSS (Low)**: Recharts는 SVG 렌더링, React 자동 이스케이프. 데이터가 크롤링/시스템 정의값이므로 위험 극히 낮음.
- **Path Traversal / Route Collision (Medium)**: `/{article_id}` 라우트에 형식 제약(UUID/정규식)이 없어 향후 새 정적 경로와 충돌 가능성 상존.

## 2. 기술적 정확성 검증

- **F1 & F2 (Grouped Bar & Label)**: `stackId` 제거 시 Recharts 자동 Grouped 배치 정확. `LabelList position="top"` 은 각 막대 위에 정확히 표시됨.
- **F4 (Total Count 합산)**: `GROUP BY` 결과의 `cnt` 합산은 전체 `COUNT(*)` 와 수학적으로 동일. 별도 쿼리 제거 정당함.
- **F7 (KST 날짜 보간)**: `toISOString()` UTC 기준 문제를 `+9시간 오프셋`으로 해결하는 방식 적절.

## 3. 누락 사항 및 엣지 케이스

### 3-1. 날짜 경계선 문제 (Sliding Window vs Calendar Day) — **Medium**

`since = datetime.now(kst) - timedelta(days=days)` 사용 시, 현재 시각이 오후 2시면 `days` 전 오후 2시 이후 데이터만 집계됨.
→ 차트 첫 날짜(가장 왼쪽)가 일부 시간대 데이터만 표시되어 **수집량 급감으로 오해** 가능.
→ **해결**: `since` 계산 시 `.replace(hour=0, minute=0, second=0, microsecond=0)` 적용.

### 3-2. 30일 조회 시 데이터 밀집도 — **Medium**

30일 Grouped Bar는 60개 막대(30일 × 2소스) → 막대 너비 2~3px 수준으로 좁아져 **숫자 라벨이 겹치거나 가독성 상실**.
→ `barSize` 최소 너비 설정 또는 30일 기간 시 라벨 숨김 처리 필요.

### 3-3. 도넛 차트 카테고리 폭발

카테고리가 12개(`COLORS` 배열) 초과 시 색상 반복. 카테고리 세분화 시 범례가 차트 영역 침범 가능.

## 4. 성능 영향

- **Backend**: DB 라운드트립 2→1회 감소 (F4). API 응답 미세 향상.
- **Frontend**: `LabelList` + `label` 함수 추가. 데이터셋 최대 30~90건 수준으로 체감 성능 영향 없음.

## 5. 개선 제안

| # | 제안 | 우선순위 |
|---|------|:--------:|
| R1 | Backend `since` 날짜 정규화 (시각 00:00:00) | **P1** |
| R2 | 30일 조회 시 `barSize` 최소 너비 또는 라벨 숨김 | **P1** |
| R3 | `LabelList` value threshold (건수 기반 표시/숨김) | P2 |
| R4 | `/{article_id}` 경로에 `Path(regex=...)` 제약 추가 | P2 |

## 6. 결론

기획서는 전반적으로 기술적 타당성이 높으며, 데이터 정합성(KST)과 성능(Query 최적화)을 고려한 점이 우수함.
**R1(날짜 정규화)과 R2(30일 가독성)** 보완 시 즉시 적용 가능한 수준.
