-- PostgreSQL 초기화 스크립트
-- docker-entrypoint-initdb.d에서 자동 실행됨

-- 확장 기능 활성화
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- 텍스트 유사도 검색

-- 한국어 전문 검색을 위한 설정 (선택사항)
-- CREATE TEXT SEARCH CONFIGURATION korean (COPY = simple);

-- 기본 권한 설정
GRANT ALL PRIVILEGES ON DATABASE lawdb TO lawuser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO lawuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO lawuser;

-- 기본 테이블은 Alembic 마이그레이션으로 생성됨
-- 이 스크립트는 확장 기능 및 초기 설정만 담당
