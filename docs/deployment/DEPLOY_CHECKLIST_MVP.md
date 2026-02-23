# MVP 배포 체크리스트 (Vercel Frontend + 서버 Backend)

## 사전 준비

- [ ] 서버에 Docker / Docker Compose 설치 완료
- [ ] 프로젝트 코드 배치 완료
- [ ] `.env.prod.example` 기반으로 `.env.prod` 생성 완료
- [ ] 무료/유료 도메인 준비 완료 (`api.<domain>` 사용 예정)
- [ ] `POSTGRES_PASSWORD` 강한 값으로 설정 완료
- [ ] `OPENAI_API_KEY` 입력 완료 (AI 기능 사용 시)
- [ ] `KAKAO_MAP_API_KEY`, `KAKAO_REST_API_KEY` 입력 완료 (지도 기능 사용 시)
- [ ] `NEXT_PUBLIC_KAKAO_MAP_API_KEY` 입력 완료 (지도 UI 사용 시)
- [ ] `CORS_ORIGINS`를 Vercel 프론트 도메인으로 설정 완료
- [ ] `API_DOMAIN` 설정 완료 (`api.<free-domain>`)
- [ ] (선택) `API_KEY` 설정 완료 (`X-API-Key` 보호)
- [ ] Vercel 프로젝트 env에 `BACKEND_URL=https://api.<domain>` 설정 완료

## 구성 점검

- [ ] `VECTOR_DB`, `LANCEDB_MODE` 설정 확인
- [ ] `neo4j` 미포함에 따른 기능 제한 이해/공지 완료
- [ ] (선택) `lancedb` profile 사용 여부 결정 완료
- [ ] Nginx 설정 파일 존재 확인 (`docker/nginx/nginx.conf`, `docker/nginx/conf.d/api.conf.template`)
- [ ] `api` 도메인 DNS가 서버 IP를 가리키도록 설정 완료
- [ ] HTTPS(TLS) 적용 계획/설정 완료 (Vercel 연동 전 필수)

## 배포 실행

- [ ] `docker compose --env-file .env.prod -f docker-compose.prod.yml config` 성공
- [ ] `docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build` 성공
- [ ] (선택) `--profile lancedb` 포함 기동 성공

## 배포 후 확인 (스모크 테스트)

- [ ] `curl http://127.0.0.1/health` 정상 응답 (Nginx 경유)
- [ ] `https://api.<domain>/health` 정상 응답 (TLS 적용 후)
- [ ] Vercel 프론트에서 `/api/*` 프록시 동작 확인
- [ ] 브라우저 콘솔에 Mixed Content/CORS 오류 없음
- [ ] 대표 API 2개 이상 응답 확인 (`lawyer-finder`, `small-claims` 등)
- [ ] `/media/*` 경로 확인 (데이터 존재 시)
- [ ] backend 로그에 치명적 오류 없음
- [ ] nginx 로그에 치명적 오류 없음

## 운영 준비 (MVP 최소)

- [ ] 로그 확인 명령(runbook) 운영자에게 공유
- [ ] 재배포/재시작 절차 공유
- [ ] 롤백 커밋 기준점 기록
- [ ] 백업 방식(수동/주기) 최소 1개 정의
- [ ] (권장) 도메인 + TLS(reverse proxy) 적용 계획 수립
