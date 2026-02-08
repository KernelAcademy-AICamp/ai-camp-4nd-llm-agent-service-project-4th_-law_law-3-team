# Git Convention Rules

Claude는 모든 Git 작업 시 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

## 1. 커밋 메시지 형식

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) 표준 + 구조화 태그를 사용합니다.

```
<타입>(<범위>): <제목>

[context] <왜 이 변경이 필요했는지 1-2문장>
[changes]
- <변경 항목 1>
- <변경 항목 2>
[impact] <이 변경이 영향을 주는 모듈/레이어>
[files] <변경된 핵심 파일 경로 (쉼표 구분)>

<꼬리말>
```

### 구조화 태그 규칙

| 태그 | 필수 여부 | 설명 |
|------|----------|------|
| `[context]` | 필수 | 변경 동기/배경. "왜" 이 작업이 필요한지 |
| `[changes]` | 필수 | 변경 내용 목록. 글머리 기호(`-`)로 나열 |
| `[impact]` | 3개 이상 파일 변경 시 필수 | 영향받는 모듈/레이어 (frontend, backend, DB 등) |
| `[files]` | 3개 이상 파일 변경 시 필수 | 핵심 변경 파일 경로 |

단일 파일 수정 같은 간단한 커밋은 `[context]`와 `[changes]`만 작성합니다.

### 예시

**복잡한 변경 (다중 파일):**
```
feat(auth): 소셜 로그인 기능 추가

[context] 자체 회원가입 전환율이 낮아 소셜 로그인으로 진입 장벽 완화
[changes]
- Google OAuth 2.0 로그인 버튼 컴포넌트 추가
- /api/auth/callback OAuth 콜백 엔드포인트 구현
- JWT 기반 세션 관리 로직 추가
[impact] frontend: 로그인 페이지, backend: auth 모듈, DB: users 테이블 컬럼 추가
[files] frontend/src/features/auth/GoogleLogin.tsx, backend/app/modules/auth/router/callback.py, backend/alembic/versions/006_add_oauth_fields.py

Closes #123
```

**간단한 변경 (단일 파일):**
```
fix(payment): 결제 금액 소수점 오차 수정

[context] 할인율 적용 시 소수점 처리 문제로 1원 단위 오차 발생
[changes]
- calculateDiscount 반환값에 Math.round() 적용
```

## 2. 커밋 타입

| 타입 | 설명 | 분류 기준 | 예시 |
|------|------|----------|------|
| `feat` | 새로운 기능 추가 | 사용자가 인식할 수 있는 새 동작 추가 | `feat(auth): 로그인 기능 추가` |
| `fix` | 버그 수정 | 기존 동작이 의도와 다르게 작동하던 것을 수정 | `fix(api): 응답 파싱 오류 수정` |
| `docs` | 문서 변경 | .md 파일, 주석, docstring만 변경 | `docs: README 업데이트` |
| `style` | 코드 스타일 | 로직 변경 없이 포맷팅만 변경 (공백, 세미콜론, import 정렬) | `style: import 정렬` |
| `refactor` | 코드 개선 | 동작 변경 없이 코드 구조 개선 (함수 분리, 이름 변경, 패턴 적용) | `refactor(db): 쿼리 최적화` |
| `perf` | 성능 개선 | refactor 중 측정 가능한 성능 향상이 목적인 경우 | `perf: 쿼리 N+1 제거` |
| `test` | 테스트 | 테스트 코드만 추가/수정 (프로덕션 코드 변경 없음) | `test: 로그인 단위 테스트 추가` |
| `build` | 빌드/의존성 | pyproject.toml, package.json, Dockerfile 등 빌드 설정 변경 | `build: 패키지 버전 업데이트` |
| `ci` | CI/CD | GitHub Actions, 배포 스크립트 등 CI/CD 파이프라인 변경 | `ci: GitHub Actions 워크플로우 추가` |
| `chore` | 기타 | 위 어디에도 해당하지 않는 변경 (설정 파일, .gitignore, 스킬 등) | `chore: .gitignore 업데이트` |
| `revert` | 되돌리기 | 이전 커밋을 되돌림 | `revert: feat(auth) 커밋 되돌리기` |

### 영향 태그 (꼬리말에 추가)

중요한 변경에는 꼬리말에 영향 태그를 추가합니다:

| 태그 | 의미 | 사용 시점 |
|------|------|----------|
| `BREAKING CHANGE:` | API/인터페이스 호환성 깨짐 | 기존 API 응답 형식, 함수 시그니처 변경 |
| `Migration:` | DB 마이그레이션 필요 | alembic 마이그레이션 포함 시 |
| `API-Change:` | API 계약 변경 | 엔드포인트 추가/수정/삭제 시 |

## 3. 한국어 커밋 메시지 규칙

이 프로젝트는 **한국어 커밋 메시지**를 사용합니다. 타입은 영어로 유지합니다 (도구 호환성).

### 제목 규칙
- **72자 이내** (정보량 우선, 50자에 억지로 맞추지 않음)
- 명령형 또는 서술형 모두 허용. **변경 내용이 명확하게 전달되는 것**이 우선
- 마침표(.) 금지
- 범위(`scope`)에 복수 모듈 가능: `feat(auth,user):` 형식

```
# 명령형 (권장)
feat(user): 프로필 이미지 업로드 기능 추가

# 서술형 (허용)
feat(user): 프로필 이미지 업로드를 위한 S3 연동 및 컴포넌트 구현
```

### 본문 규칙
- 제목과 본문 사이에 **빈 줄** 필수
- 줄바꿈 강제 없음 (72자 규칙 폐지)
- 본문은 **구조화 태그** (`[context]`, `[changes]`, `[impact]`, `[files]`) 형식으로 작성 (Section 1 참조)

### 꼬리말 규칙
- 이슈 연결: `Closes #123`, `Fixes #456`, `Refs #789`
- 영향 태그: `BREAKING CHANGE:`, `Migration:`, `API-Change:` (Section 2 참조)

## 4. 절대 금지 사항

### Force Push 금지
```bash
# ❌ 절대 금지 (main, dev, release 브랜치)
git push --force
git push -f
git push --force-with-lease  # main/dev에서는 이것도 금지

# ✅ 개인 feature 브랜치에서만 허용 (주의해서 사용)
git push --force-with-lease origin feature/my-branch
```

### 민감 정보 커밋 금지
```bash
# ❌ 절대 커밋하면 안 되는 파일들
.env
.env.local
.env.production
*.pem
*.key
credentials.json
secrets.yaml

# 커밋 전 확인
git diff --cached --name-only | grep -E '\.(env|pem|key)$'
```

### 파괴적 명령어 금지 (공유 브랜치)
```bash
# ❌ 공유 브랜치에서 절대 금지
git reset --hard HEAD~n
git rebase -i (이미 push된 커밋)
git commit --amend (이미 push된 커밋)

# ✅ 로컬 작업 중에만 사용
git reset --soft HEAD~1  # 마지막 커밋 취소 (변경사항 유지)
```

## 5. 주의 사항

### .gitignore 확인

커밋 전 `.gitignore` 파일을 확인하여 민감/불필요한 파일이 제외되었는지 점검한다.

### 대용량 파일 금지

```bash
# ❌ 직접 커밋 금지 (100MB 이상)
*.zip
*.tar.gz
*.mp4
*.pth
*.bin
*.model

# ✅ 대용량 파일은 Git LFS 사용 또는 외부 저장소
git lfs track "*.pth"
```

### 빈 폴더 문제

Git은 빈 폴더를 추적하지 않습니다. 빈 폴더가 필요한 경우:

```bash
# 빈 폴더 유지가 필요한 경우 .gitkeep 파일 추가
mkdir -p data/uploads
touch data/uploads/.gitkeep
```

## 6. 브랜치 전략

### 브랜치 명명 규칙

```
main                          # 프로덕션 배포 브랜치
dev                           # 개발 통합 브랜치
feature/<이슈번호>-<설명>     # 새 기능 개발
fix/<이슈번호>-<설명>         # 버그 수정
refactor/<이슈번호>-<설명>    # 리팩토링
docs/<설명>                   # 문서 작업
release/<버전>                # 릴리즈 준비
hotfix/<설명>                 # 긴급 수정
```

이슈번호가 없는 경우 생략 가능 (예: `feature/plan-review-skill`).

### 브랜치 이름 규칙

```bash
# Good - 케밥 케이스, 이슈번호 포함
feature/42-user-profile-image
fix/57-payment-calculation-error
refactor/api-error-handling

# Bad
feature/UserProfileImage     # 카멜 케이스 금지
feature/user_profile_image   # 스네이크 케이스 금지
feature/fix                  # 너무 모호함
Feature/user-profile         # 대문자 금지
```

### 브랜치 워크플로우

```
main ─────────────────────────────────────────►
       │                              ▲
       │                              │
       ▼                              │
dev ──────────────────────────────────┼───────►
       │              ▲               │
       │              │ merge         │
       ▼              │               │
feature/xxx ──────────┘               │
                                      │
hotfix/xxx ───────────────────────────┘
```

## 7. 머지 및 충돌 해결

### 머지 전 체크리스트

```bash
# 1. 최신 코드 동기화
git fetch origin
git rebase origin/dev  # 또는 git pull --rebase

# 2. 충돌 확인
git status

# 3. 테스트 실행
uv run pytest              # Backend
npm run test               # Frontend

# 4. 린트 확인
uv run ruff check .        # Backend
npm run lint               # Frontend
```

### 충돌 해결 가이드

```bash
# 1. 충돌 발생 시
git status  # 충돌 파일 확인

# 2. 충돌 마커 확인 및 해결
<<<<<<< HEAD
현재 브랜치의 코드
=======
병합하려는 브랜치의 코드
>>>>>>> feature/xxx

# 3. 해결 후 스테이징
git add <resolved-file>

# 4. 계속 진행
git rebase --continue  # rebase 중이었다면
git merge --continue   # merge 중이었다면

# 5. 반드시 테스트 실행
uv run pytest
```

### 충돌 해결 원칙

1. **코드 이해 우선**: 양쪽 코드의 의도를 먼저 파악
2. **기능 유지**: 양쪽의 기능이 모두 동작하도록 통합
3. **테스트 필수**: 해결 후 반드시 테스트 실행
4. **의심 시 질문**: 확실하지 않으면 원 작성자에게 확인

## 8. 커밋 체크리스트

커밋 전 자가 점검:

- [ ] 변경사항이 하나의 논리적 단위인가?
- [ ] 커밋 메시지가 규칙을 따르는가?
- [ ] 민감 정보가 포함되지 않았는가?
- [ ] 불필요한 파일이 포함되지 않았는가?
- [ ] 테스트가 통과하는가?
- [ ] 린트 에러가 없는가?
- [ ] 변경된 코드와 관련된 CLAUDE.md가 업데이트되었는가?
- [ ] README.md에 반영할 내용이 있는가?
- [ ] 코드 구조 변경 시 관련 스킬/에이전트가 업데이트되었는가?

## 9. 실수 복구 가이드

### 커밋 메시지 수정 (push 전)

```bash
# 마지막 커밋 메시지만 수정
git commit --amend -m "새로운 메시지"
```

### 커밋 취소 (push 전)

```bash
# 커밋 취소, 변경사항 유지 (staged)
git reset --soft HEAD~1

# 커밋 취소, 변경사항 유지 (unstaged)
git reset HEAD~1

# 커밋 및 변경사항 모두 취소 (주의!)
git reset --hard HEAD~1
```

### 잘못된 파일 커밋 제거 (push 전)

```bash
# 특정 파일만 커밋에서 제거
git reset HEAD~1 -- path/to/file
git commit --amend
```

### 이미 Push한 경우

```bash
# ⚠️ 개인 브랜치에서만 사용
git revert <commit-hash>  # 안전한 방법
git push

# ❌ 공유 브랜치에서는 revert만 사용
```

## 10. 커밋 전 문서 확인 (필수)

코드 변경 시 관련 문서가 최신 상태인지 **반드시** 확인한 후 커밋합니다.

### 확인 대상 문서

| 문서 | 위치 | 확인 시점 |
|------|------|----------|
| `CLAUDE.md` (루트) | 프로젝트 루트 | 아키텍처, 명령어, 모듈, 환경설정 변경 시 |
| `CLAUDE.md` (백엔드) | `backend/CLAUDE.md` | 백엔드 구조, 설정, 스크립트 변경 시 |
| `CLAUDE.md` (프론트) | `frontend/CLAUDE.md` | 프론트엔드 구조, 컴포넌트, API 변경 시 |
| `CLAUDE.md` (스크립트) | `backend/scripts/CLAUDE.md` | 스크립트 추가/수정/삭제 시 |
| `CLAUDE.md` (평가) | `backend/evaluation/CLAUDE.md` | 평가 시스템 변경 시 |
| `README.md` | 프로젝트 루트 | 설치 방법, 데이터 구조, 실행 방법 변경 시 |
| 스킬/에이전트 | `.claude/skills/`, `.claude/agents/` | 코드 경로/구조 변경 시 (coding-style.md Section 10 참조) |

### 규칙
- 코드 변경과 문서 업데이트는 **같은 커밋**에 포함한다
- 문서만 별도로 업데이트하는 경우 `docs:` 타입 사용
- 문서 업데이트를 빠뜨린 경우 별도 커밋으로 즉시 보완한다

---

**중요**: 이 규칙들은 팀 협업을 위해 필수입니다. 규칙을 어길 합당한 이유가 있다면 팀과 먼저 논의하세요.
