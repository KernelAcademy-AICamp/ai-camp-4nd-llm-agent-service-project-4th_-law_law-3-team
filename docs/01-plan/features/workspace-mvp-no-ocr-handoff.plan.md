# 사건 중심 워크스페이스 MVP 구현 계획 (OCR 제외)

## 요약
1. MVP는 `대화 분류/요약/이어가기`와 `사건 중심 워크스페이스`를 우선 구현한다.
2. 보안은 `HttpOnly 세션 토큰` 기반으로 전환하고, `X-Device-Id` 단독 신뢰를 폐기한다.
3. 상태 정본은 `LangGraph Checkpointer`로 단일화한다.
4. 타임라인은 MVP에서 `텍스트/대화 기반 추출`만 지원한다.
5. OCR/문서 파이프라인은 Phase 2로 분리한다.

## MVP 범위
1. 대화 내역 저장, 사건명/고객명 자동 분류, 수동 보정.
2. `/chat-history`에서 사건명/고객명 그룹 조회 및 이어가기.
3. 구조화 요약 기반 재개 컨텍스트(`facts/issues/evidence/open_questions/next_steps`).
4. 사건 워크스페이스 기본 CRUD.
5. 스토리보드/워크스페이스 공통 타임라인 엔진(입력: 수동 텍스트 + 채팅 transcript).
6. 로그인 시 데이터 귀속은 자동 병합이 아닌 `Opt-in 가져오기`.

## MVP 제외 (Phase 2)
1. OCR 추출.
2. 파일 업로드 후 자동 텍스트화/청킹/임베딩.
3. OCR 기반 타임라인 자동 생성.
4. 문서 검색 인덱스 고도화.

## 공개 API/타입 변경
1. `ChatRequest` 확장: `conversation_id`, `case_id`.
2. SSE `done` 확장: `conversation_id`, `case_id`, `thread_id`, `active_agent`.
3. 신규 API:
4. `GET /api/chat/conversations`
5. `GET /api/chat/conversations/{conversation_id}`
6. `PATCH /api/chat/conversations/{conversation_id}`
7. `POST /api/workspace/cases`
8. `GET /api/workspace/cases`
9. `GET /api/workspace/cases/{case_id}`
10. `GET /api/workspace/cases/{case_id}/timeline`
11. `POST /api/workspace/cases/{case_id}/timeline/rebuild` (입력 텍스트/대화 기준)
12. `PATCH /api/workspace/cases/{case_id}/timeline/items/{item_id}`
13. 인증/식별: 서버 발급 세션 쿠키 기반 principal 사용.

## 데이터 모델 (MVP)
1. `workspace_cases`
2. `workspace_case_timeline_items`
3. `chat_conversations` (메타데이터 중심, state 정본 아님)
4. `chat_messages`
5. `workspace_activity_logs`
6. `identity_links` (추후 로그인 연동 대비)
7. 대화 state/cursor는 `chat_conversations`가 아닌 `LangGraph Checkpointer`에서만 관리.

## 보안/프라이버시 결정
1. `X-Device-Id` 단독 식별 금지.
2. 서버 세션 토큰(`HttpOnly/Secure/SameSite`)으로 소유권 검증.
3. 로그인 시 자동 병합 금지, Opt-in 가져오기만 허용.
4. `session_secret` 원문 반환 금지, 필요 시 만료형 capability token 사용.
5. 공용 기기 오귀속 방지를 위한 명시적 확인 UX 추가.

## 상태/이어가기 전략
1. 대화 state/cursor/interrupt는 `LangGraph Checkpointer`만 정본.
2. `chat_conversations`는 `thread_id` 포인터 + 대화 메타데이터만 저장.
3. 이어가기 입력은 `구조화 요약 + 토큰 예산 기반 최근 대화`.
4. 고정 `최근 20턴` 규칙은 사용하지 않는다.

## 구현 순서
1. Alembic 마이그레이션(신규 테이블/인덱스).
2. 세션 기반 principal 유틸/미들웨어 추가.
3. Chat persistence 서비스 구현.
4. `/api/chat/stream` 연계: conversation 생성/재개, 메시지 저장, 요약/분류 갱신.
5. 구조화 요약기 구현 및 수동 보정 보호.
6. `/chat-history` UI 구현(그룹/검색/이어가기/수정).
7. 워크스페이스 기본 페이지(`/workspace`, `/workspace/[caseId]`) 구현.
8. 공통 Timeline Engine 분리(텍스트/대화 입력만).
9. Storyboard를 공통 엔진/정본 타임라인 소비 구조로 연결.
10. WorkspaceAgent 라우팅 추가(조회 vs 재생성 의도 판별).

## 테스트 케이스
1. 소유권 검증 실패 시 대화/사건 접근 차단.
2. 첫 대화에서 `conversation_id` 생성 및 이어가기 성공.
3. 수동 보정값이 자동 분류에 덮어쓰기 되지 않음.
4. interrupt 상태 재개 정상 동작.
5. `/chat-history` 그룹/검색/이어가기 UX 동작.
6. 타임라인 재생성(텍스트/대화 입력) 결과 정본 반영 확인.
7. Storyboard와 Workspace가 동일 타임라인 정본을 조회하는지 검증.

## Phase 2 계획 (OCR 추가)
1. 문서 업로드 API 및 저장소 연동.
2. OCR 추출 파이프라인(비동기 작업큐).
3. semantic split + overlap 기반 장문 처리.
4. OCR 결과를 타임라인 엔진 입력으로 연결.
5. OCR 실패/부분 인식 fallback 및 재시도 정책.
6. 비용/성능 모니터링 지표 추가.

## 가정/기본값
1. MVP는 비회원 중심이며 계정 시스템은 이후 확장.
2. 데이터 복구코드는 MVP 제외.
3. 최소 백업 대안으로 대화/타임라인 내보내기(JSON/TXT)는 MVP에 포함 권장.
4. 요약은 구조화 필드 저장을 기본으로 한다.
