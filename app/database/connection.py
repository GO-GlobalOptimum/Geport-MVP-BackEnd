# DB/connection.py
from pymongo import MongoClient
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

mongodb_url = os.getenv("MONGODB_URL")
client = MongoClient(mongodb_url)
db = client["admin"]

user_baseInfo_collection = db["users_info"]
igeport_user_baseInfo_collection = db['igeport_users_info']
geport_db = db['geport_db']
igeport_db = db['igeport_db']


# 환경 변수에서 데이터베이스 연결 정보 가져오기 URL 수정
DB_URL_READ = os.environ.get('MYSQL_URL_READ')
DB_URL_WRITE = os.environ.get('MYSQL_URL_WRITE')

if not DB_URL_READ or not DB_URL_WRITE:
    raise ValueError("MYSQL_URL_READ 및 MYSQL_URL_WRITE 환경 변수를 설정해야 합니다.")

# 읽기 전용 엔진 및 세션 생성
read_engine = create_engine(DB_URL_READ, pool_recycle=500)
ReadSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=read_engine)

# 쓰기 전용 엔진 및 세션 생성
write_engine = create_engine(DB_URL_WRITE, pool_recycle=500)
WriteSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=write_engine)

# SQLAlchemy 기본 클래스 생성
Base = declarative_base()


# 쓰기 전용 데이터베이스 세션 생성 함수
def get_db():
    db = WriteSessionLocal()
    try:
        yield db
    finally:
        db.close()