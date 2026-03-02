# Red Team 검증 보고서 - Keyword Time Range Selection

## 검증 도구: Gemini CLI (Red Team)

## 주요 소견

### 1. 취약점
- [Medium] 30d 범위 반복 요청 시 API 비용 폭증/DoS 위험 → Rate Limiting 강화 필요
- [Low] 캐시 파편화 (user × time_range) → 전역 캐시 레이어 권장
- [Low] Enum 검증 우회 가능성 → FastAPI 자동 검증으로 대응

### 2. 아키텍처 제안
- 전역 트렌드 캐시 도입, 비동기 워커 전환, SourceConfig 추상화

### 3. 고급 기능 제안
- 트렌드 속도(Velocity) 지표, 시맨틱 그룹화, 증분 수집

### 4. 성능 최적화
- Request Collapsing, Pre-fetching, 외부 API 병렬화

### 5. UX 개선
- 시간 범위별 예상 소요시간 안내, localStorage 선택값 유지, 상세 진행률

## 변경 이력
- 2026-02-27: Red Team 검증 수행
