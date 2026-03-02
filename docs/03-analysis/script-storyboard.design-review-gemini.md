# 설계 리뷰 — Gemini CLI (시니어 아키텍트)

> **검증 대상**: `docs/02-design/features/script-storyboard.design.md` (v0.1)
> **검증 도구**: Gemini CLI (시니어 소프트웨어 아키텍트)
> **검증일**: 2026-02-28

---

## 총평

설계가 매우 구체적이며 프로젝트의 기존 패턴을 깊이 있게 이해하고 작성됨.
주요 보완 3가지: 1) JobManager 공통화, 2) 페르소나 데이터 동적 연동, 3) 파일 정리 정책

---

## 1. 아키텍처 일관성 및 파일 구조

| 구분 | 내용 |
|------|------|
| [보완] | 서비스 파일 위치 조정 — `app/services/service_function/` 대신 `app/modules/content_marketing/service/webtoon_service.py`로 이동 권장 |
| [보완] | JobManager 중복 제거 — storyboard 모듈의 JobManager와 공통화하여 `app/core/jobs/manager.py`로 분리 |

## 2. 컴포넌트 및 비즈니스 로직

| 구분 | 내용 |
|------|------|
| [보완] | 페르소나 데이터 연동 — `DEFAULT_LAWYER` 하드코딩 → `persona_id` 기반 동적 캐릭터 프로필 주입 |
| [보완] | 이미지 클린업 정책 — `STORYBOARD_CACHE_TTL` 연동 만료 폴더 자동 삭제 백그라운드 태스크 |
| [확인] | 캐릭터 일관성 전략 — 첫 패널 레퍼런스 활용 합리적, 첫 패널 실패 시 세컨드 레퍼런스 로직 추가 권장 |

## 3. 확장성 및 성능

| 구분 | 내용 |
|------|------|
| [대안] | 이미지 저장소 추상화 — `StorageProvider` 인터페이스 도입 (Local → S3 교체 가능) |
| [확인] | 세마포어 적용 — `asyncio.Semaphore(3)` 적절 |

## 4. 보안 및 예외 처리

| 구분 | 내용 |
|------|------|
| [보완] | 인증 통합 — `Depends(get_current_user)` 적용 필요 (stream/status 엔드포인트) |
| [확인] | 입력 검증 및 인젝션 방지 — Pydantic validator + sanitize_input 적절 |
| [보완] | Rate Limit — 사용자당 일일 생성 횟수 제한 도입 권장 |

## 5. 프론트엔드

| 구분 | 내용 |
|------|------|
| [확인] | 스크롤 동기화 및 스켈레톤 UI — 긴 생성 시간 UX 대응 적절 |
| [대안] | 개별 패널 편집 기능 — `scene_description`/`image_prompt` 사용자 미세 조정 후 재생성 |

---

## Agent Team 수용 판단

| 피드백 | 수용 여부 | 사유 |
|--------|----------|------|
| 서비스 파일 위치 조정 | **수용** | 기존 패턴상 `services/service_function/`에 위치하는 것이 프로젝트 규칙. 단, webtoon은 content_marketing의 하위 기능이므로 현 위치 유지가 더 일관적 |
| JobManager 공통화 | **부분 수용** | storyboard 모듈의 JobManager를 그대로 import하여 재사용 (별도 core 이동은 과설계) |
| 페르소나 동적 연동 | **수용** | persona_id가 있을 때 DB에서 조회하여 캐릭터 프로필에 반영하는 로직 추가 |
| 이미지 클린업 정책 | **기록** | 향후 과제로 기록 (현재 MVP에서는 수동 정리) |
| 첫 패널 실패 시 세컨드 레퍼런스 | **수용** | 2번째 성공 패널을 레퍼런스로 사용하는 폴백 로직 추가 |
| StorageProvider 추상화 | **기록** | 향후 과제로 기록 (현재 로컬 스토리지로 충분) |
| 인증 통합 | **기록** | TEMP_USER_ID → 실 인증 전환 시 반영 (현재 MVP) |
| Rate Limit | **기록** | 향후 과제로 기록 |
| 패널 편집 기능 | **기록** | 향후 v2로 기록 |
