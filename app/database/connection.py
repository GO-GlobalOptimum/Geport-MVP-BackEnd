# # DB/connection.py
# from pymongo import MongoClient
# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, scoped_session
# import os

# # MongoDB 설정
# mongodb_url = os.getenv("MONGODB_URL")
# client = MongoClient(mongodb_url)
# db = client["admin"]

# user_baseInfo_collection = db["users_info"]
# igeport_user_baseInfo_collection = db['igeport_users_info']
# geport_db = db['geport_db']
# igeport_db = db['igeport_db']

# # 환경 변수에서 데이터베이스 연결 정보 가져오기 URL 수정
# # AZURE_SQL_USERNAME = os.getenv('AZURE_SQL_USERNAME')
# # AZURE_SQL_PASSWORD = os.getenv('AZURE_SQL_PASSWORD')
# # AZURE_SQL_SERVER = os.getenv('AZURE_SQL_SERVER')
# # AZURE_SQL_DATABASE = os.getenv('AZURE_SQL_DATABASE')

# DB_URL_READ = os.environ.get('MYSQL_URL_READ')
# DB_URL_WRITE = os.environ.get('MYSQL_URL_WRITE')

# # DB_URL = f'mysql+pymysql://{AZURE_SQL_USERNAME}:{AZURE_SQL_PASSWORD}@{AZURE_SQL_SERVER}:3306/{AZURE_SQL_DATABASE}'

# # 읽기 전용 엔진 및 세션 생성
# read_engine = create_engine(DB_URL_READ, pool_recycle=500)
# ReadSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=read_engine)

# # 쓰기 전용 엔진 및 세션 생성
# write_engine = create_engine(DB_URL_WRITE, pool_recycle=500)
# WriteSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=write_engine)

# # SQLAlchemy 기본 클래스 생성
# Base = declarative_base()

# # 읽기 전용 데이터베이스 세션 생성 함수
# def get_read_db():
#     db = ReadSessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # 쓰기 전용 데이터베이스 세션 생성 함수
# def get_write_db():
#     db = WriteSessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()





# DB/connection.py
from pymongo import MongoClient
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import os

# MongoDB 설정
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

# 읽기 전용 엔진 및 세션 생성
read_engine = create_engine(DB_URL_READ, pool_recycle=500)
ReadSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=read_engine)

# 쓰기 전용 엔진 및 세션 생성
write_engine = create_engine(DB_URL_WRITE, pool_recycle=500)
WriteSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=write_engine)

# SQLAlchemy 기본 클래스 생성
Base = declarative_base()

# 읽기 전용 데이터베이스 세션 생성 함수
def get_read_db():
    db = ReadSessionLocal()
    try:
        yield db
    finally:
        db.close()

# 쓰기 전용 데이터베이스 세션 생성 함수
def get_write_db():
    db = WriteSessionLocal()
    try:
        yield db
    finally:
        db.close()
