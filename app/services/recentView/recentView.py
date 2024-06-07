from typing import List, Dict
from fastapi import Request, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import logging

MAX_RECENT_POSTS = 10

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_member_id_by_email(email: str, db: Session) -> int:
    query = text("SELECT member_id FROM Member WHERE email = :email")
    result = db.execute(query, {"email": email}).fetchone()
    if result is None:
        raise HTTPException(status_code=404, detail="Member not found")
    logger.info(f"Member ID for email {email}: {result[0]}")
    return result[0]

def get_recent_posts(request: Request) -> List[int]:
    return request.session.get("recent_posts", [])

def add_recent_post(request: Request, post_id: int, member_email: str, db: Session) -> None:
    recent_posts = request.session.get("recent_posts", [])
    
    if post_id not in recent_posts:
        recent_posts.append(post_id)
    
    if len(recent_posts) > MAX_RECENT_POSTS:
        recent_posts.pop(0)
    
    request.session["recent_posts"] = recent_posts

    try:
        member_id = get_member_id_by_email(member_email, db)
        
        query = text("SELECT * FROM View WHERE post_id = :post_id AND member_id = :member_id")
        view = db.execute(query, {"post_id": post_id, "member_id": member_id}).fetchone()

        if view:
            update_query = text("""
                UPDATE View 
                SET view_count = view_count + 1, updated_time = :updated_time
                WHERE post_id = :post_id AND member_id = :member_id
            """)
            db.execute(update_query, {
                "post_id": post_id,
                "member_id": member_id,
                "updated_time": datetime.now()
            })
            logger.info(f"Updated View count for post_id {post_id} and member_id {member_id}")
        else:
            insert_query = text("""
                INSERT INTO View (post_id, member_id, view_count, updated_time)
                VALUES (:post_id, :member_id, :view_count, :updated_time)
            """)
            db.execute(insert_query, {
                "post_id": post_id, 
                "member_id": member_id, 
                "view_count": 1,
                "updated_time": datetime.now()
            })
            logger.info(f"Inserted new view for post_id {post_id} and member_id {member_id}")
        
        db.commit()
        logger.info("Transaction committed")
    except Exception as e:
        db.rollback()
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def fetch_post_data(post_id: int, db: Session) -> dict:
    query = text("SELECT * FROM Post WHERE post_id = :post_id")
    result = db.execute(query, {"post_id": post_id}).fetchone()
    if result is None:
        raise HTTPException(status_code=404, detail="Post not found")
    post = dict(result._mapping)
    logger.info(f"Fetched post data: {post}")
    return post

def fetch_recent_posts_data(recent_posts: List[int], db: Session) -> List[dict]:
    if not recent_posts:
        return []
    
    query = text("SELECT * FROM Post WHERE post_id IN :recent_posts")
    results = db.execute(query, {"recent_posts": tuple(recent_posts)}).fetchall()
    posts = [dict(row._mapping) for row in results]
    logger.info(f"Fetched recent posts data: {posts}")
    return posts