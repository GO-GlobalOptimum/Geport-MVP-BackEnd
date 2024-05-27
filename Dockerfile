# 베이스 이미지
FROM python:3.9-slim AS builder
WORKDIR /src
COPY . .

# 필요한 패키지 설치 및 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends sqlite3 \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && rm -rf /var/lib/apt/lists/*

# 멀티-스테이지 빌드: 최종 이미지
FROM python:3.9-slim
COPY --from=builder /src /app
WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
