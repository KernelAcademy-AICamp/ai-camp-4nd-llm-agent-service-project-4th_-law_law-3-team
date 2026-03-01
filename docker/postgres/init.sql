-- PostgreSQL 초기화 스크립트
-- docker-entrypoint-initdb.d에서 자동 실행됨

-- 확장 기능 활성화
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- 텍스트 유사도 검색

-- 한국어 전문 검색을 위한 설정 (선택사항)
-- CREATE TEXT SEARCH CONFIGURATION korean (COPY = simple);

-- 기본 권한 설정 (최소 권한 원칙)
GRANT CONNECT ON DATABASE lawdb TO lawuser;
GRANT USAGE, CREATE ON SCHEMA public TO lawuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO lawuser;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO lawuser;

-- 향후 Alembic에서 생성되는 테이블/시퀀스에도 자동 권한 적용
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO lawuser;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO lawuser;

-- 기본 테이블은 Alembic 마이그레이션으로 생성됨
-- 이 스크립트는 확장 기능 및 초기 설정만 담당
