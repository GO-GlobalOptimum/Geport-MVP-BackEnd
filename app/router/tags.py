
from fastapi import Depends, HTTPException
import os
from fastapi import APIRouter
from sqlalchemy.orm import Session
from app.database.connection import get_db
from sqlalchemy import text  # `text`를 추가
from app.services.tags.tags import get_post_by_id, generate_tags


router = APIRouter()


# # SQL 데이터베이스 연결 확인 엔드포인트
# @router.get("fastapi/check_sql_connection")
# def check_sql_connection(db: Session = Depends(get_db)):
#     try:
#         # 간단한 쿼리를 실행하여 연결 확인 (text 객체로 감싸기)
#         db.execute(text("SELECT 1"))
#         return {"message": "Successfully connected to the SQL database"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

# # post_id를 통한 게시글 정보 불러오기
# @router.get("fastapi/posts/{post_id}")
# def read_post(post_id: int, db: Session = Depends(get_db)):
#     return get_post_by_id(post_id, db)

# post에 대한 tag 생성.
@router.get("/posts/generate/tags/{post_id}")
def generate_post_tags(post_id :int, db: Session = Depends(get_db)):
    return generate_tags(post_id, db)

