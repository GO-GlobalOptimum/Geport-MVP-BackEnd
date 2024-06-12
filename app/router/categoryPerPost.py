# from fastapi import APIRouter, Depends
# from sqlalchemy.orm import Session
# from app.database.connection import get_db
# from app.services.categoryPerPost.categoryPerPost import get_category_post_counts
# from app.services.auth.auth import get_current_user
# from pydantic import BaseModel
# from typing import List, Dict, Any


# router = APIRouter()

# class CategoryPostCount(BaseModel):
#     category_id: int
#     category_name: str
#     post_count: int


# @router.get("/categories/post-counts/", response_model=List[CategoryPostCount])
# async def get_category_post_counts_endpoint(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
#     """
#     카테고리별로 사용자가 작성한 포스트의 개수를 반환하는 API 입니다.

#     Parameters:
#         db (Session): 데이터베이스 세션.
#         current_user (dict): 현재 인증된 사용자 정보.

#     Returns:
#         List[CategoryPostCount]: 카테고리별 포스트 개수를 포함하는 리스트.
#     """
#     result = get_category_post_counts(db, current_user)
#     return result
