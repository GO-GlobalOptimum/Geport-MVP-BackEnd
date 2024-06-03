FROM python:3.9

ARG OPENAI_API_KEY
ARG NAVER_CLOUD_ACCESS_KEY_ID
ARG NAVER_CLOUD_SECRET_KEY
ARG AUTHJWT_SECRET_KEY
ARG GOOGLE_CLIENT_ID
ARG GOOGLE_CLIENT_SECRET
ARG SECRET_KEY
ARG ALGORITHM
ARG ACCESS_TOKEN_EXPIRE_MINUTES

# 환경 변수 설정
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV NAVER_CLOUD_ACCESS_KEY_ID=${NAVER_CLOUD_ACCESS_KEY_ID}
ENV NAVER_CLOUD_SECRET_KEY=${NAVER_CLOUD_SECRET_KEY}
ENV AUTHJWT_SECRET_KEY=${AUTHJWT_SECRET_KEY}
ENV GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID}
ENV GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET}
ENV SECRET_KEY=${SECRET_KEY}
ENV ALGORITHM=${ALGORITHM}
ENV ACCESS_TOKEN_EXPIRE_MINUTES=${ACCESS_TOKEN_EXPIRE_MINUTES}

# 작업 디렉토리 설정
WORKDIR /src    

# 프로젝트 파일 복사
COPY . /src

# 패키지 소스 업데이트 및 시스템 패키지 설치
RUN if [ ! -f /etc/apt/sources.list ]; then \
      echo "deb http://archive.ubuntu.com/ubuntu focal main restricted universe multiverse" > /etc/apt/sources.list; \
    fi && \
    sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/mirrors.edge.kernel.org/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y sqlite3 && \
    apt-get clean

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt  

# 포트 설정
EXPOSE 8000

# 컨테이너 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
