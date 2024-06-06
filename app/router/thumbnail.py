
# from fastapi import Depends, HTTPException, APIRouter
# from sqlalchemy.orm import Session
# from sqlalchemy import text
# from app.database.connection import get_db
# from app.services.thumbnail.thumbnail_text import generate_thumbnailText
# from app.services.thumbnail.thumbnail_image import generate_thumbnailImage
=

# router = APIRouter()


# # # SQL 데이터베이스 연결 확인 엔드포인트
# # @router.get("fastapi/check_sql_connection")
# # def check_sql_connection(db: Session = Depends(get_db)):
# #     try:
# #         # 간단한 쿼리를 실행하여 연결 확인 (text 객체로 감싸기)
# #         db.execute(text("SELECT 1"))
# #         return {"message": "Successfully connected to the SQL database"}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
    

# # post에서 썸내일 글 만들어내기
# @router.get("/posts/generate/thumbnail/text/{post_id}")
# def generate_thumbnail_text(post_id :int,db: Session = Depends(get_db)):
#     return generate_thumbnailText(post_id, db)


# # post에서 썸내일 이미지 만들어내기
# @router.get("/posts/generate/thumbnail/image/{post_id}")
# def generate_thumbnail_image(post_id :int,db: Session = Depends(get_db)):
#     return generate_thumbnailImage(post_id, db)

