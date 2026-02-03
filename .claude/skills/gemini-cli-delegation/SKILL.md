---
name: gemini-cli-delegation
description: 대규모 코드 분석, 프로젝트 전체 파악, 리팩토링 설계 등 광범위한 작업 시 Gemini CLI를 활용하여 요약 결과를 확보하는 패턴. 많은 파일 분석, 아키텍처 리뷰, 보안 감사, 마이그레이션 분석 시 사용.
---

# Gemini CLI 위임 패턴

## 1. 개요

### Gemini CLI란?
터미널에서 Google Gemini 모델에 접근하는 오픈소스 AI 에이전트. 대규모 컨텍스트 윈도우를 활용하여 많은 파일을 한번에 분석할 수 있다.

### 왜 위임하는가?
- Claude의 컨텍스트를 효율적으로 사용 (요약된 결과만 수신)
- Gemini의 대규모 컨텍스트 윈도우를 활용한 광범위 분석
- 분석 결과를 기반으로 Claude가 정밀한 코드 작성에 집중

## 2. 사전 확인

### 설치 여부 확인
```bash
# Windows
where gemini 2>nul && echo "AVAILABLE" || echo "NOT_AVAILABLE"

# Unix/macOS
which gemini 2>/dev/null && echo "AVAILABLE" || echo "NOT_AVAILABLE"
```

### Fallback 전략
Gemini CLI 미설치 시 Claude가 직접 수행:
- Task(Explore) 에이전트로 코드베이스 탐색
- Glob/Grep 도구로 파일 검색 및 패턴 분석
- 필요 시 파일을 순차적으로 읽어 분석

## 3. 위임 판단 기준

### 사용자 지정 시나리오

| 시나리오 | 트리거 조건 | 예시 |
|---------|-----------|------|
| 많은 파일 동시 파악 | 10개 이상 파일의 패턴/구조 분석 | "모든 API 라우터의 인증 패턴 분석해줘" |
| 프로젝트 전체 분석 | 아키텍처 파악, 의존관계 맵핑 | "프로젝트 아키텍처 전체를 설명해줘" |
| 대규모 리팩토링 설계 | 3개 이상 모듈에 영향, 10+ 파일 변경 예상 | "서비스 레이어를 CQRS 패턴으로 변경" |

### 추가 발굴 시나리오

| 시나리오 | 트리거 조건 | 예시 |
|---------|-----------|------|
| 보안 코드 리뷰 | 인증/권한/입력검증 등 다수 파일 보안 분석 | "API 엔드포인트 전체 보안 취약점 점검" |
| 마이그레이션 영향도 분석 | 라이브러리 업그레이드 시 영향 범위 파악 | "Pydantic v1->v2 마이그레이션 영향 분석" |
| 코드 품질 감사 | 프로젝트 전반의 안티패턴/중복 탐지 | "backend 전체 코드 품질 리포트 작성" |
| 테스트 커버리지 갭 분석 | 테스트 없는 비즈니스 로직 탐지 | "테스트가 없는 핵심 로직 목록 추출" |
| API 호환성 분석 | API 변경 시 클라이언트 영향도 파악 | "v2 API 변경이 프론트엔드에 미치는 영향" |
| 문서-코드 정합성 검증 | 문서와 실제 코드 불일치 탐지 | "CLAUDE.md와 실제 코드 경로 불일치 찾기" |

### 위임하지 않는 경우
- 단일 파일 분석 또는 수정
- 5개 이하 파일에 대한 간단한 분석
- 구체적인 코드 작성 작업
- Gemini CLI 미설치 환경

## 4. 명령어 패턴

### 기본 사용법
```bash
# 기본 (비대화형, 텍스트 출력)
gemini "프롬프트 내용"

# JSON 출력 (파싱 필요 시)
gemini "프롬프트 내용" -o json

# YOLO 모드 (에이전트 동작, 파일 읽기 자동 승인)
gemini "프롬프트 내용" -y

# 샌드박스 모드 (안전한 실행)
gemini "프롬프트 내용" -s

# stdin 파이핑
cat file.py | gemini "이 코드의 보안 취약점을 분석해줘"

# 파일 참조 (@prefix)
gemini "다음 파일들의 아키텍처를 분석해줘 @backend/app/main.py @backend/app/core/registry.py"
```

### 시나리오별 명령어

```bash
# 1. 프로젝트 구조 분석
gemini "이 프로젝트의 전체 아키텍처를 분석해줘. 주요 모듈, 의존관계, 데이터 흐름을 요약해줘." -y -o text

# 2. 다수 파일 패턴 분석
gemini "backend/app/modules/ 하위 모든 router/ 디렉토리의 공통 패턴과 차이점을 분석해줘." -y -o text

# 3. 리팩토링 영향 범위 분석
gemini "backend/app/services/ 를 분석하고, 함수 단위 리팩토링 계획을 제안해줘. 영향받는 파일 목록 포함." -y -o text

# 4. 보안 코드 리뷰
gemini "backend/app/api/ 와 backend/app/modules/ 의 API 엔드포인트 전체를 분석하고 보안 취약점을 보고해줘." -y -o text

# 5. 마이그레이션 분석
gemini "이 프로젝트에서 pydantic v1 패턴을 모두 찾고, v2로 마이그레이션할 때 변경이 필요한 부분을 리스트업해줘." -y -o text

# 6. 코드-문서 정합성 검증
gemini "CLAUDE.md에 기술된 파일 경로와 클래스명이 실제 코드와 일치하는지 검증해줘. 불일치 목록을 작성해줘." -y -o text

# 7. 테스트 갭 분석
gemini "backend/app/services/ 의 비즈니스 로직 중 테스트가 없는 함수 목록을 추출해줘." -y -o text
```

## 5. 출력 활용 방법

### 기본 흐름
1. Gemini CLI로 대규모 분석 실행
2. 출력 결과를 Claude의 작업 컨텍스트로 사용
3. 요약 결과를 기반으로 구체적인 파일 읽기 및 수정 진행

### JSON 출력 파싱
```bash
# jq로 구조화된 데이터 추출
gemini "분석 결과를 JSON으로 출력해줘" -o json | jq '.results[]'
```

### 결과 저장
```bash
# 분석 결과를 파일로 저장하여 참조
gemini "프로젝트 아키텍처를 분석해줘" -y -o text > analysis_result.txt
```

## 6. 주의사항

- **검증 필수**: Gemini 출력은 반드시 Claude가 검증 (잘못된 경로, 존재하지 않는 함수 등 가능)
- **민감 정보 차단**: API 키, 비밀번호가 포함된 파일(`.env`, `credentials.json` 등)은 Gemini에 전달하지 않음
- **에러 시 fallback**: Gemini CLI가 에러를 반환하면 Claude가 직접 분석으로 전환
- **읽기 전용 분석**: `-y` (yolo) 모드는 읽기 전용 분석에만 사용, 파일 수정 작업에는 사용하지 않음
- **네트워크 의존**: Gemini CLI는 네트워크 연결 필요, 오프라인 환경에서는 Claude가 직접 수행
