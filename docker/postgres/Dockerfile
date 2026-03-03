# PostgreSQL 17 + pg_textsearch (BM25 full-text search)
# Alpine 기반, PGXS 소스 빌드
FROM postgres:17-alpine

# pg_textsearch 빌드에 필요한 도구 설치
RUN apk add --no-cache --virtual .build-deps \
        build-base \
        git \
        clang19 \
        llvm19-dev \
    && cd /tmp \
    && git clone --depth 1 --branch v0.5.1 https://github.com/timescale/pg_textsearch.git \
    && cd pg_textsearch \
    && make \
    && make install \
    && cd / \
    && rm -rf /tmp/pg_textsearch \
    && apk del .build-deps

# init.sql은 docker-compose에서 volume mount
