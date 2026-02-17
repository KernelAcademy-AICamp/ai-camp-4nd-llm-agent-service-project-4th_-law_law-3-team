---
name: update-docs
description: 코드 변경 사항을 분석하여 관련 문서(CLAUDE.md, README.md, 스킬, 에이전트)를 자동으로 찾아 업데이트. 코드 구조 변경 후 문서 동기화 시 사용.
---

# Update Docs Skill

코드 변경 사항을 분석하여 관련 문서를 자동으로 찾아 업데이트합니다.
`/update-docs` 또는 `/update-docs <대상>` 으로 실행합니다.

## 실행 절차

### 1. 변경 사항 분석

```bash
# 변경된 파일 목록 수집 (staged + unstaged)
git diff --name-only HEAD

# 변경이 없으면 최근 커밋 대비
git diff --name-only HEAD~1
```

### 2. 프로젝트 문서 목록 (Document Registry)

이 프로젝트의 모든 문서 파일과 업데이트 트리거 조건입니다.

#### 2-A. 프로젝트 가이드 문서

| 문서 | 경로 | 업데이트 트리거 |
|------|------|----------------|
| 프로젝트 README | `README.md` | 설치 방법, 실행 방법, 프로젝트 구조, 주요 기능 변경 |
| 루트 CLAUDE.md | `CLAUDE.md` | 아키텍처, 모듈, 환경설정, Key Files, 폴더 구조 변경 |
| AGENTS.md | `AGENTS.md` | CLAUDE.md와 동일 트리거 (CLAUDE.md 변경 시 함께 동기화) |

> **AGENTS.md**: Codex 등 범용 AI 에이전트용 프로젝트 가이드.
> 파일이 없으면 "AGENTS.md가 아직 없습니다. 생성할까요?" 로 사용자에게 확인.
> CLAUDE.md와 내용이 겹치되, AGENTS.md는 빌드/테스트/코드스타일 중심으로 간결하게 유지.

#### 2-B. 서브 프로젝트 문서

| 문서 | 경로 | 업데이트 트리거 |
|------|------|----------------|
| Backend CLAUDE.md | `backend/CLAUDE.md` | 모델, 서비스, 모듈, 마이그레이션, 환경변수, 폴더 구조 변경 |
| Backend Scripts | `backend/scripts/CLAUDE.md` | 스크립트 추가/수정/삭제 |
| Backend Evaluation | `backend/evaluation/CLAUDE.md` | 평가 시스템 변경 |
| Frontend CLAUDE.md | `frontend/CLAUDE.md` | 컴포넌트, 페이지, API 서비스, 라우팅 변경 |

#### 2-C. 아키텍처/설계 문서 (docs/)

| 문서 | 경로 | 업데이트 트리거 |
|------|------|----------------|
| DB 아키텍처 | `docs/architecture/DB_ARCHITECTURE.md` | 테이블 추가/삭제, 스키마 변경, 인덱스 변경 |
| 데이터 카탈로그 | `docs/data/DATA_CATALOG.md` | 새 데이터 소스 추가, 데이터 구조 변경 |
| 벡터DB 설계 | `docs/architecture/vectordb_design.md` | LanceDB 스키마/임베딩 변경 |
| 배포 비용 | `docs/infrastructure/DEPLOYMENT_COST_ESTIMATION.md` | 인프라/서비스 구성 변경 |

> **날짜 접미사 문서** (`*_20260120.md` 등)는 히스토리 기록이므로 업데이트 대상이 아님.
> 새로운 변경이 필요하면 새 날짜로 문서를 생성.

#### 2-D. 에이전트/스킬 문서

| 문서 | 경로 | 업데이트 트리거 |
|------|------|----------------|
| AI 오케스트레이터 | `.claude/agents/ai-orchestrator.md` | CLI 도구, 라우팅 로직 변경 |
| 코드 리팩토러 | `.claude/agents/code-refactorer.md` | 리팩토링 패턴 변경 |
| 각 스킬 | `.claude/skills/*/SKILL.md` | 해당 스킬이 참조하는 코드 경로/패턴 변경 |
| 스킬 카탈로그 | `.claude/skills/CATALOG.md` | 스킬/에이전트/규칙 추가·삭제·이동, description 변경 |

### 3. 변경 → 문서 매핑 규칙

변경된 파일 경로를 아래 패턴으로 매칭하여 업데이트 대상을 결정합니다.

```
변경 경로 패턴                          → 업데이트 대상 문서
─────────────────────────────────────────────────────────────
backend/app/models/*.py                 → backend/CLAUDE.md (모델 파일 위치, 테이블 구조)
                                          CLAUDE.md (관련 섹션 있으면)
                                          docs/architecture/DB_ARCHITECTURE.md
                                          docs/data/DATA_CATALOG.md

backend/alembic/versions/*.py           → backend/CLAUDE.md (마이그레이션 섹션)

backend/app/modules/*/router/           → CLAUDE.md (모듈 매핑, API 엔드포인트)
                                          backend/CLAUDE.md (모듈 섹션)

backend/app/modules/*/schema/           → CLAUDE.md (프론트-백엔드 계약 동기화 테이블)

backend/app/services/                   → backend/CLAUDE.md (Services 섹션)

backend/app/multi_agent/                → CLAUDE.md (에이전트 목록)
                                          backend/CLAUDE.md (에이전트 섹션)

backend/app/core/config.py              → backend/CLAUDE.md (환경 변수 테이블)
                                          CLAUDE.md (환경 변수 언급 있으면)

backend/scripts/*.py                    → backend/scripts/CLAUDE.md

backend/evaluation/                     → backend/evaluation/CLAUDE.md

frontend/src/lib/modules.ts             → CLAUDE.md (모듈 매핑 현황)
frontend/src/lib/api.ts                 → CLAUDE.md (모듈 매핑 현황)
frontend/next.config.*                  → CLAUDE.md (모듈 매핑 현황)
frontend/src/features/*/                → frontend/CLAUDE.md
frontend/src/app/*/page.tsx             → CLAUDE.md (Modules 섹션)

docker-compose.yml                      → CLAUDE.md, backend/CLAUDE.md, README.md
pyproject.toml                          → backend/CLAUDE.md (Commands, 의존성)
package.json                            → frontend/CLAUDE.md (Commands, 의존성)

.claude/skills/*, .claude/agents/*      → 해당 파일 자체 (참조 경로 변경 확인)
                                          .claude/skills/CATALOG.md (추가/삭제/이동 시)

.claude/rules/*                         → .claude/skills/CATALOG.md (규칙 추가/삭제 시)
```

### 4. 모듈 4곳 동기화 확인

모듈 관련 변경이 감지되면 아래 4곳의 일관성을 확인하고, CLAUDE.md의 모듈 매핑 테이블도 업데이트합니다:

1. `frontend/src/lib/modules.ts` — `enabled` 플래그
2. `frontend/src/lib/api.ts` — `endpoints` 객체
3. `frontend/next.config.js` — `rewrites` 프록시 규칙
4. `backend/app/modules/<module>/router/__init__.py` — 라우터 구현

### 5. 스킬/에이전트 참조 검증

코드 경로가 변경된 경우, `.claude/skills/`와 `.claude/agents/` 내에서 이전 경로를 참조하는 파일을 검색합니다:

```bash
# 변경 전 경로가 스킬/에이전트에서 참조되는지 확인
grep -r "변경전_경로" .claude/skills/ .claude/agents/
```

참조가 발견되면 해당 스킬/에이전트 파일도 업데이트합니다.

### 6. CATALOG.md 동기화

스킬/에이전트/규칙이 **추가·삭제·이동**되거나, YAML frontmatter의 `description`이 변경된 경우 `.claude/skills/CATALOG.md`를 업데이트합니다.

#### 트리거 조건

| 변경 유형 | CATALOG.md 업데이트 내용 |
|----------|------------------------|
| 스킬 디렉토리 추가 | 해당 카테고리 테이블에 행 추가 + 크기 분포 갱신 |
| 스킬 디렉토리 삭제 | 해당 카테고리 테이블에서 행 제거 + 크기 분포 갱신 |
| 스킬 description 변경 | 해당 행의 설명 컬럼 업데이트 |
| 에이전트 파일 추가/삭제 | 해당 카테고리 에이전트 테이블 + 카테고리 요약 테이블 갱신 |
| 규칙 파일 추가/삭제 | 규칙 목록 테이블 갱신 |
| 스킬 카테고리 변경 | 이전 카테고리에서 제거 → 새 카테고리에 추가 |

#### 업데이트 절차

1. **현황 수집**: 실제 디렉토리를 스캔하여 최신 목록 확보
   ```bash
   # 스킬 목록 + 줄 수
   for d in .claude/skills/*/; do
     name=$(basename "$d")
     [ -f "$d/SKILL.md" ] && echo "$name $(wc -l < "$d/SKILL.md")"
   done

   # 에이전트 목록
   ls .claude/agents/*.md

   # 규칙 목록
   ls .claude/rules/*.md
   ```

2. **CATALOG.md 읽기**: 현재 카탈로그 내용을 읽는다

3. **차이 비교**: 실제 파일 시스템 vs CATALOG.md 내용을 비교
   - 누락된 스킬/에이전트/규칙 → 추가
   - 삭제된 스킬/에이전트/규칙 → 제거
   - 줄 수 변경 → 크기 분포 섹션 갱신

4. **카테고리 판단** (신규 스킬일 때): SKILL.md의 description을 읽고 기존 9개 카테고리 중 가장 적합한 곳에 배치
   - 워크플로우 / 코드 품질 / 프론트엔드 / RAG·검색 / 데이터·DB / 도메인 지식 / 멀티에이전트 / 외부 CLI / 운영
   - 기존 카테고리에 맞지 않으면 새 카테고리 추가

5. **업데이트 적용**: Edit 도구로 최소 변경
   - 카테고리 요약 테이블의 스킬/에이전트 수
   - 해당 카테고리 섹션의 테이블 행
   - 스킬 크기 분포 섹션
   - 총 줄 수 / 평균
   - `최종 업데이트` 날짜

6. **의존관계 확인**: 새 스킬이 기존 스킬을 참조하면 의존관계 메모도 갱신

### 7. AGENTS.md 동기화

CLAUDE.md가 변경될 때 AGENTS.md도 함께 동기화합니다.

#### AGENTS.md 구조 (CLAUDE.md와의 차이)

| 섹션 | CLAUDE.md | AGENTS.md |
|------|-----------|-----------|
| 프로젝트 개요 | 상세 | 간결 (1~2줄) |
| 빌드/테스트 명령어 | ✅ | ✅ (동일) |
| 아키텍처 | 상세 (폴더 트리, 설명) | 핵심만 (주요 디렉토리) |
| 코드 스타일 | rules/ 참조 | 직접 포함 (린트 도구, 컨벤션) |
| 모듈 시스템 | 상세 | 간략 (추가 방법만) |
| 환경 변수 | 전체 테이블 | 필수 항목만 |
| 보안 | rules/ 참조 | 핵심 원칙만 |

#### AGENTS.md가 없을 때

```
AGENTS.md가 아직 없습니다. Codex 등 범용 AI 에이전트를 위해 생성할까요?
(CLAUDE.md 기반으로 빌드/테스트/코드스타일 중심의 간결한 버전을 생성합니다)
```

사용자가 승인하면 CLAUDE.md를 기반으로 AGENTS.md를 생성합니다.

### 8. 문서 업데이트 실행

각 대상 문서에 대해:

1. **Read**: 현재 문서 내용을 읽는다
2. **Locate**: 변경과 관련된 섹션을 찾는다 (없으면 적절한 위치에 신규 섹션 추가)
3. **Edit**: 최소한의 변경으로 문서를 업데이트한다
4. **Verify**: 마크다운 구조가 깨지지 않았는지 확인한다

#### 업데이트 원칙

- **사실만 기록**: 코드에서 확인 가능한 정보만 문서화
- **기존 스타일 유지**: 해당 문서의 기존 포맷/톤/구조를 따름
- **최소 변경**: 관련 섹션만 수정, 무관한 섹션은 건드리지 않음
- **테이블 행 추가**: 기존 테이블에 행을 추가할 때 정렬 순서 유지
- **트리 구조 추가**: 파일 트리에 새 항목 추가 시 알파벳/논리 순서 유지
- **삭제된 코드**: 문서에서도 해당 항목 제거 (주석 처리 금지)
- **존재하지 않는 문서**: 건너뛰고 결과 보고에 표기

### 9. 결과 보고

```
## 문서 업데이트 결과

| 문서 | 변경 내용 | 상태 |
|------|----------|------|
| CLAUDE.md | [변경 설명] | ✅ 업데이트 |
| backend/CLAUDE.md | [변경 설명] | ✅ 업데이트 |
| AGENTS.md | [변경 설명] | ✅ 업데이트 / ⚠️ 파일 없음 |
| docs/architecture/DB_ARCHITECTURE.md | 변경 사항 없음 | ⏭️ 스킵 |
| ... | ... | ... |
```

### 10. 인자 사용법

| 명령 | 동작 |
|------|------|
| `/update-docs` | 전체 변경 분석 → 모든 관련 문서 업데이트 |
| `/update-docs backend` | `backend/CLAUDE.md`, `backend/scripts/CLAUDE.md`, `backend/evaluation/CLAUDE.md` |
| `/update-docs frontend` | `frontend/CLAUDE.md`, 루트 `CLAUDE.md`의 프론트 관련 섹션 |
| `/update-docs root` | 루트 `CLAUDE.md`, `README.md`, `AGENTS.md` |
| `/update-docs docs` | `docs/` 디렉토리 내 모든 문서 |
| `/update-docs skills` | `.claude/skills/`, `.claude/agents/` 참조 경로 검증 + `CATALOG.md` 동기화 |
| `/update-docs all` | 위 전체 (`/update-docs`와 동일) |
