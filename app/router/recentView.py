from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
from app.services.recentView.recentView import add_recent_post, get_recent_posts, fetch_post_data, fetch_recent_posts_data
from app.services.auth.auth import get_current_user
from app.database.connection import get_db


router = APIRouter()


""" _summary_ 
    프론트엔드에서 post 를 볼떄 요청 해주면 해당 요청을 사용자가 최근에 본 게시글로 저장해주는 API

Returns:
    _type_: post_id를 db에 저장한다.
"""
@router.get("/posts/{post_id}", tags=["posts"])
async def read_post(post_id: int, request: Request, user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    member_email = user["email"]
    add_recent_post(request, post_id, member_email, db)
    post_data = fetch_post_data(post_id, db)
    return post_data


"""_summary_
    최근에 해당 사용자 (JWT)가 본 게시글들을 불러와준다

Returns:
    _type_: 최근 본 게시글 리스트들
"""
@router.get("/recent-posts", tags=["posts"])
async def get_recent_posts_route(request: Request, user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    recent_posts = get_recent_posts(request)
    recent_posts_data = fetch_recent_posts_data(recent_posts, db)
    return recent_posts_data

