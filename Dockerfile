# 베이스 이미지 설정
FROM python:3.9

# 소스 코드 복사 및 작업 디렉토리 설정
COPY . /src
WORKDIR /src

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    sqlite3 \
    default-libmysqlclient-dev \
    build-essential

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 포트 노출
EXPOSE 8000

# 애플리케이션 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
