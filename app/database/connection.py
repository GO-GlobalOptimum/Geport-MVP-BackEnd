# DB/connection.py
from pymongo import MongoClient
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import os

mongodb_url = os.getenv("MONGODB_URL")
client = MongoClient(mongodb_url)
db = client["admin"]

user_baseInfo_collection = db["users_info"]
igeport_user_baseInfo_collection = db['igeport_users_info']
geport_db = db['geport_db']
igeport_db = db['igeport_db']



# # 환경 변수에서 데이터베이스 연결 정보 가져오기 URL 수정
AZURE_SQL_USERNAME = os.getenv('AZURE_SQL_USERNAME')
AZURE_SQL_PASSWORD = os.getenv('AZURE_SQL_PASSWORD')
AZURE_SQL_SERVER = os.getenv('AZURE_SQL_SERVER')
AZURE_SQL_DATABASE = os.getenv('AZURE_SQL_DATABASE')

#DB_URL = os.environ.get('MYSQL_URL')
DB_URL = f'mysql+pymysql://{AZURE_SQL_USERNAME}:{AZURE_SQL_PASSWORD}@{AZURE_SQL_SERVER}:3306/{AZURE_SQL_DATABASE}'

# SQLAlchemy 엔진 생성
engine = create_engine(DB_URL, pool_recycle=500)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# 데이터베이스 세션 생성 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
