---
name: summary-quality-audit
description: LLM 요약 필드의 품질을 탐색적으로 감사하는 프로토콜. 길이/공백/중복/환각 패턴 탐지, 샘플링 기반 품질 점검. 요약 품질 검증, 데이터 클리닝 시 사용.
---

# Summary Quality Audit Skill

LLM 요약 필드의 품질을 탐색적으로 감사하는 프로토콜.

## 1. 개요

JSON 데이터의 LLM 생성 요약 필드를 7개 관점에서 자동 감사합니다.

| 도구 | 목적 | 관계 |
|------|------|------|
| `validate_summaries.py` | 사전 정의 규칙 검증 (길이, 프롬프트 누출) | 배포 전 게이트 |
| `clean_summaries.py` | 규칙 기반 클리닝 (마크다운 제거, 불완전 문장 보정) | 후처리 |
| **`audit_summary_quality.py`** | 탐색적(EDA) 품질 감사 | 이슈 발견 → 위 도구로 해결 |

### 커버리지 관계 (겹침과 차이)

세 도구는 역할이 다르지만 일부 탐지 대상이 겹칩니다.

**겹치는 영역 (audit ∩ validate):**
- 프롬프트 누출: 양쪽 동일 정규식 (각각 독립 정의)
- HTML/마크다운: audit이 14개 세분화 패턴, validate는 단일 복합 패턴
- 반복 패턴: validate(`10자+, 2회+`, 민감), audit(`15자+, 5회+, 20%비율`, 엄격)

**audit에만 있는 검사:**
- `(NNN자)` 글자수 표기, `핵심키워드:` 리스트, `[쟁점]/[판단]` 브라켓
- 포맷 일관성 분류, 중복 탐지, 교차 필드 비교

**validate에만 있는 검사 (audit 미커버):**
- 길이 초과 pass/fail 판정 (≤300자/≤600자)
- 불완전 문장 (`...`, 열린 괄호)
- 언어 혼용 (연속 영문 3단어+)

**권장 워크플로우:** `audit` → 이슈 파악 → `clean` → 후처리 → `validate` → 최종 게이트

### 7개 검사 항목

| # | 검사 | 탐지 대상 |
|---|------|----------|
| 1 | 기본 통계 | total, null/empty, 길이 분포 (min/P25/P50/P75/P90/P99/max) |
| 2 | 마크다운 패턴 | `**볼드**`, `# 헤더`, `- 리스트`, `` `코드` ``, `[링크]()`, `\|테이블\|`, `> 인용`, `---` (14개 패턴) |
| 3 | LLM 아티팩트 | `(NNN자)` 글자수, `핵심키워드:` 리스트, `[쟁점]/[판단]` 브라켓, 자기참조, 지시문 누출, HTML |
| 4 | 포맷 일관성 | 구조 패턴 분류 (쟁점+판단, 키워드 포함, 자유형식 등) |
| 5 | 이상치 | 극단적 단문(<80자), 극단적 장문(>1000자), 무한 반복 |
| 6 | 중복 | 완전 동일 요약 텍스트 |
| 7 | 교차 비교 | 요약 vs 비교필드 (동일/포함/독립 비율) |

## 2. 사용 시점

- 새 데이터 수집 후 요약 품질 확인
- LLM 프롬프트 변경 후 품질 검증
- 데이터 인제스트 전 사전 점검
- 사용자가 `/summary-quality-audit` 호출 시

## 3. Step 1: 자동 스크립트 실행

### Mode 1: IngestConfig 등록 타입 (자동 필드 해석)

```bash
cd backend

# 기본 (config 기본 경로)
uv run python scripts/audit_summary_quality.py --type special_admin_appeal

# 데이터 디렉토리 재매핑
uv run python scripts/audit_summary_quality.py --type special_admin_appeal --data-dir ../data

# 교차 비교 + JSON 보고서
uv run python scripts/audit_summary_quality.py --type special_admin_appeal \
    --data-dir ../data \
    --compare-field 재결요지 \
    --output eda_output/audit_special_admin_appeal.json \
    --samples 10
```

### Mode 2: 임의 JSON 파일 (필드 직접 지정)

```bash
cd backend

uv run python scripts/audit_summary_quality.py \
    --file ../data/special_admin_appeal/sadm_case_조세심판원_v2.json \
    --summary-field 심판례요약 \
    --id-field 특별행정심판재결례일련번호 \
    --compare-field 재결요지
```

### 등록된 20개 타입 및 주요 필드 매핑

| 타입 | summary_field | id_field | 주요 compare_field |
|------|---------------|----------|--------------------|
| `law` | 법령 요약 | 법령일련번호 | - |
| `precedent` | 판례요약 | 판례일련번호 | 판결요지 |
| `admin_rule` | 행정규칙요약 | 행정규칙일련번호 | - |
| `constitutional` | 헌재결정요약 | 헌재결정례일련번호 | 결정요지 |
| `administration` | 행정심판요약 | 행정심판례일련번호 | - |
| `legislation` | 법령해석요약 | 법령해석례일련번호 | 회답 |
| `treaty` | 조약요약 | 조약일련번호 | - |
| `interpretation_ministry` | 부처해석요약 | 부처해석례일련번호 | 회답 |
| `special_admin_appeal` | 심판례요약 | 특별행정심판재결례일련번호 | 재결요지 |
| `dec_*` (10개 위원회) | 결정요약 | 결정례일련번호 | 이유 |

## 4. Step 2: 보고서 해석

### 터미널 출력 구조

```
============================================================
  LLM 요약 품질 탐색적 감사 보고서
============================================================
  [1] 기본 통계
  [2] 마크다운 패턴 (영향: N건)
  [3] LLM 아티팩트 (영향: N건)
  [4] 포맷 일관성
  [5] 이상치
  [6] 중복
  [7] 교차 비교 (선택)
  ─── 심각도 요약 ───
  [HIGH]   즉시 조치 필요
  [MEDIUM] 검토 후 조치
  [LOW]    참고 사항
```

### 심각도 판정 기준

| 조건 | 심각도 |
|------|--------|
| null/empty > 5% | HIGH |
| LLM 아티팩트 > 10% | HIGH |
| 무한 반복 발견 | HIGH |
| null/empty 1-5% | MEDIUM |
| LLM 아티팩트 1-10% | MEDIUM |
| 마크다운 > 5% | MEDIUM |
| 나머지 | LOW |

### 정상/주의/문제 판정

| 상태 | 기준 |
|------|------|
| 정상 | HIGH 0건, MEDIUM 0건 |
| 주의 | HIGH 0건, MEDIUM 1건+ |
| 문제 | HIGH 1건+ |

## 5. Step 3: 개선 권고

이슈 유형별 해결 도구 매핑:

| 이슈 유형 | 해결 도구 | 방법 |
|----------|----------|------|
| 마크다운 잔여 (`**볼드**` 등) | `clean_summaries.py` | Phase 1: 마크다운 제거 |
| `핵심키워드:` 리스트 잔류 | `clean_summaries.py` | 정규식 제거 규칙 추가 |
| `(NNN자)` 글자수 표기 | `clean_summaries.py` | 정규식 제거 규칙 추가 |
| 프롬프트 누출 | `validate_summaries.py` | HIGH 패턴으로 검출 → 재생성 |
| 불완전 문장 | `clean_summaries.py` | Phase 2: 문장 보정 |
| 극단적 이상치 (단문/장문) | 수동 확인 | 샘플 검토 후 LLM 재생성 |
| 무한 반복 | 수동 재생성 | 해당 레코드 LLM 재호출 |
| 포맷 불일치 | 프롬프트 개선 | 프롬프트 튜닝 |
| 중복 | 수동 확인 | 원본 데이터 중복 여부 점검 |

## 6. 금지 사항

- 원본 JSON 수정 금지 (read-only 감사)
- 스크립트 미실행 상태로 추측 보고 금지
- 샘플 없이 수치만으로 판단 금지 (항상 샘플 확인)
- 감사 결과를 clean_summaries.py 규칙으로 자동 변환 금지 (사용자 확인 필수)

## 7. 관련 도구

| 파일 | 설명 |
|------|------|
| `backend/scripts/audit_summary_quality.py` | 이 스킬의 자동화 스크립트 |
| `backend/scripts/validate_summaries.py` | 규칙 기반 검증 |
| `backend/scripts/clean_summaries.py` | 규칙 기반 클리닝 |
| `backend/scripts/ingest/config.py` | IngestConfig 정의 |
| `backend/scripts/ingest/types/` | 20개 타입별 설정 |
