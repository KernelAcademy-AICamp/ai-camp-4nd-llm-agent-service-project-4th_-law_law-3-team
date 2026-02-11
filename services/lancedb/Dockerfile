FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential autoconf automake libtool curl \
    && rm -rf /var/lib/apt/lists/*

# mecab-ko 소스 빌드 (aarch64 지원을 위해 config.guess/sub 업데이트)
RUN wget -q -O - https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz \
    | tar zxf - \
    && cd mecab-0.996-ko-0.9.2 \
    && cp /usr/share/misc/config.guess . \
    && cp /usr/share/misc/config.sub . \
    && ./configure && make -j$(nproc) && make install && ldconfig \
    && cd / && rm -rf mecab-0.996-ko-0.9.2

# mecab-ko-dic 소스 빌드
RUN wget -q -O - https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz \
    | tar zxf - \
    && cd mecab-ko-dic-2.1.1-20180720 && autoreconf -fi && ./configure && make -j$(nproc) && make install && ldconfig \
    && cd / && rm -rf mecab-ko-dic-2.1.1-20180720

# 빌드 도구 제거 (curl은 healthcheck용으로 유지)
RUN apt-get purge -y build-essential autoconf automake libtool wget \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8100
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
