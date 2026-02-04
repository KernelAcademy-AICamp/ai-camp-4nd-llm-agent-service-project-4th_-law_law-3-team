# Git Convention Rules

Claude는 모든 Git 작업 시 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

## 1. 커밋 메시지 형식

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) 표준을 따릅니다.

```
<타입>(<범위>): <제목>

<본문>

<꼬리말>
```

### 예시
```
feat(auth): 소셜 로그인 기능 추가

Google OAuth 2.0을 사용한 소셜 로그인 구현
- Google 로그인 버튼 컴포넌트 추가
- OAuth 콜백 처리 로직 구현
- 사용자 세션 관리 로직 추가

Closes #123
```

## 2. 커밋 타입

| 타입 | 설명 | 예시 |
|------|------|------|
| `feat` | 새로운 기능 추가 | `feat(auth): 로그인 기능 추가` |
| `fix` | 버그 수정 | `fix(api): 응답 파싱 오류 수정` |
| `docs` | 문서 변경 | `docs: README 업데이트` |
| `style` | 코드 스타일 (포맷팅, 세미콜론 등) | `style: 불필요한 공백 제거` |
| `refactor` | 기능 변경 없는 코드 개선 | `refactor(db): 쿼리 최적화` |
| `perf` | 성능 개선 | `perf: 이미지 로딩 속도 개선` |
| `test` | 테스트 추가/수정 | `test: 로그인 단위 테스트 추가` |
| `build` | 빌드/의존성 변경 | `build: 패키지 버전 업데이트` |
| `ci` | CI/CD 설정 변경 | `ci: GitHub Actions 워크플로우 추가` |
| `chore` | 기타 변경 (빌드, 도구 등) | `chore: .gitignore 업데이트` |
| `revert` | 이전 커밋 되돌리기 | `revert: feat(auth) 커밋 되돌리기` |

## 3. 한국어 커밋 메시지 규칙

이 프로젝트는 **한국어 커밋 메시지**를 사용합니다. 타입은 영어로 유지합니다 (도구 호환성).

### 제목 작성 규칙

```
# ✅ Good - 명령형 사용
feat(user): 프로필 이미지 업로드 기능 추가
fix(cart): 수량 변경 시 가격 계산 오류 수정
refactor(api): 에러 핸들링 로직 통합

# ❌ Bad - 과거형/진행형 사용 금지
feat(user): 프로필 이미지 업로드 기능 추가함
feat(user): 프로필 이미지 업로드 기능을 추가했습니다
feat(user): 프로필 이미지 업로드 기능 추가하는 중
```

### 제목 규칙
- **50자 이내** 권장 (최대 72자)
- 명령형 동사로 시작 ("추가", "수정", "삭제", "개선", "리팩토링")
- 마침표(.) 금지
- 첫 글자 대문자 불필요 (한국어)

### 본문 규칙
- 제목과 본문 사이에 **빈 줄** 필수
- **72자**마다 줄바꿈
- **무엇을**, **왜** 변경했는지 설명
- "어떻게" 변경했는지는 코드가 설명함

```
# ✅ Good - 무엇을, 왜
fix(payment): 결제 금액 계산 오류 수정

할인율 적용 시 소수점 처리 문제로 인해
실제 결제 금액이 1원 단위로 오차가 발생했음.
Math.round()를 사용하여 반올림 처리.

# ❌ Bad - 어떻게만 설명
fix(payment): 결제 금액 계산 오류 수정

calculateDiscount 함수에서 Math.round()를
호출하도록 변경함.
```

### 꼬리말 규칙
- 이슈 연결: `Closes #123`, `Fixes #456`, `Refs #789`
- Breaking Change: `BREAKING CHANGE: API 응답 형식 변경`

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

### .gitignore 확인 필수

커밋 전 다음 파일/폴더가 제외되었는지 확인:

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/
.venv/
*.egg-info/

# Node.js
node_modules/
.next/
dist/
build/

# IDE
.idea/
.vscode/
*.swp

# 환경 설정
.env*
!.env.example

# 로그
*.log
logs/

# 빈 폴더 문제
# Git은 빈 폴더를 추적하지 않음
# 필요 시 .gitkeep 파일 추가
```

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
main              # 프로덕션 배포 브랜치
dev               # 개발 통합 브랜치
feature/<설명>    # 새 기능 개발
fix/<설명>        # 버그 수정
refactor/<설명>   # 리팩토링
docs/<설명>       # 문서 작업
release/<버전>    # 릴리즈 준비
hotfix/<설명>     # 긴급 수정
```

### 브랜치 이름 규칙

```bash
# ✅ Good - 케밥 케이스, 명확한 설명
feature/user-profile-image
fix/payment-calculation-error
refactor/api-error-handling

# ❌ Bad
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
| 스킬/에이전트 | `.agent/skills/`, `.agent/agents/` | 코드 경로/구조 변경 시 (coding-style.md Section 10 참조) |

### 규칙
- 코드 변경과 문서 업데이트는 **같은 커밋**에 포함한다
- 문서만 별도로 업데이트하는 경우 `docs:` 타입 사용
- 문서 업데이트를 빠뜨린 경우 별도 커밋으로 즉시 보완한다

---

**중요**: 이 규칙들은 팀 협업을 위해 필수입니다. 규칙을 어길 합당한 이유가 있다면 팀과 먼저 논의하세요.
