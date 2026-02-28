# Google Drive Operations Rules

Claude는 Google Drive 관련 작업(백업, 복원, 데이터 동기화) 시 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

> **상세 가이드**: `.claude/skills/google-drive-operations/SKILL.md` 참조

## 1. 사전 검증 (매 작업 전 필수)

| 확인 항목 | 명령어 |
|----------|--------|
| rclone 설치 여부 | `which rclone` |
| rclone.conf 존재 | 프로젝트 루트에 `rclone.conf` 파일 확인 |
| 인증 유효성 | `rclone ls --config rclone.conf gdrive: --max-depth 1` |
| Docker 컨테이너 (백업/복원 시) | `docker ps` 로 `law-platform-db` 확인 |

검증 실패 시 사용자에게 누락된 사전 조건을 안내하고, 직접 해결을 시도하지 않는다.

## 2. copy vs sync 구분 (필수)

| 방향 | 명령 | 이유 |
|------|------|------|
| 다운로드 (Drive → 로컬) | `rclone copy` | 로컬 파일 삭제 방지 |
| 업로드 (로컬 → Drive) | `rclone sync` | 이전 버전 잔류 방지 |

**`sync` 실행 전 반드시 `--dry-run`으로 삭제될 파일 확인** 후 사용자 승인을 받는다.

## 3. 민감 파일 보호

절대 커밋하거나 외부에 노출하지 않는 파일:

- `rclone.conf` — Google Drive 인증 정보 포함
- `secrets/` 디렉토리 — 서비스 계정 키
- `backups/` 디렉토리 — DB 덤프 파일

이 파일들은 `.gitignore`에 포함되어 있으나, `git add -A` 등으로 실수 추가되지 않도록 주의한다.

## 4. DB 백업/복원 시 주의사항

- **백업 전**: Docker 컨테이너 실행 상태 확인
- **복원 시**: 기존 데이터가 덮어씌워짐 → 사용자 확인 후 진행
- **LanceDB 복원 후**: 벡터 인덱스 + FTS 인덱스 재생성 필수 안내

## 5. data/ 동기화 규칙

- `data/`는 `.gitignore`에 포함 → git clone만으로는 받을 수 없음
- 새 환경 세팅 시 DB 복원과 함께 data/ 복원 안내 필수
- 파일명 변경(v2→v3) 후에는 `sync` 사용 (이전 버전 정리)
- `gdrive:data/`는 "현재 작업 세트"이며, 버전 아카이브 용도가 아님

## 6. 스크립트 직접 실행 원칙

백업/복원은 기존 스크립트를 사용한다. rclone 명령을 수동으로 조합하지 않는다:

| 작업 | 스크립트 |
|------|---------|
| DB 백업 | `./scripts/backup_to_gdrive.sh` |
| DB 복원 | `./scripts/restore_from_gdrive.sh <target>` |
| data/ 동기화 | `rclone copy/sync` 직접 사용 (스크립트 없음) |

## 7. 환경변수 기본값 존중

`.env`의 rclone 관련 변수(`RCLONE_CONF`, `RCLONE_REMOTE`)는 기본값이 설정되어 있다.
사용자가 명시적으로 변경을 요청하지 않는 한 기본값을 유지한다.
