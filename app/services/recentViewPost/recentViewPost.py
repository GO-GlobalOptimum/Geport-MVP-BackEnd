from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi import HTTPException
import logging

def get_recently_viewed_category_post_counts(db: Session, current_user: dict):
    """
    최근 사용자가 본 포스트의 카테고리별 포스트 개수를 반환하는 함수입니다.

    Parameters:
        db (Session): 데이터베이스 세션.
        current_user (dict): 현재 인증된 사용자 정보.

    Returns:
        List[Dict[str, Any]]: 카테고리별 포스트 개수를 포함하는 리스트.
    """
    user_email = current_user["email"]

    # 사용자의 member_id를 조회합니다.
    query_str = text("SELECT member_id FROM Member WHERE email = :email")
    params = {"email": user_email}

    try:
        result = db.execute(query_str, params).fetchone()
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")

    if not result:
        raise HTTPException(status_code=404, detail="User not found")

    member_id = result.member_id

    # 사용자가 최근 본 포스트의 ID를 조회합니다.
    query_str = text("SELECT post_id FROM View WHERE member_id = :member_id ORDER BY updated_time DESC LIMIT 100")
    params = {"member_id": member_id}

    try:
        post_ids = db.execute(query_str, params).fetchall()
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")

    if not post_ids:
        post_ids = []

    post_ids = [row.post_id for row in post_ids]

    # 모든 카테고리를 조회합니다.
    query_str = text("SELECT category_id, name FROM Category")
    try:
        categories = db.execute(query_str).fetchall()
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")

    # 각 카테고리에 대한 포스트 개수를 조회합니다.
    query_str = text("""
    SELECT category_id, COUNT(*) as post_count
    FROM CategoryPost
    WHERE post_id IN :post_ids
    GROUP BY category_id
    """)
    params = {"post_ids": tuple(post_ids)}

    try:
        result = db.execute(query_str, params).fetchall()
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")

    post_counts = {row.category_id: row.post_count for row in result}

    # 모든 카테고리에 대해 포스트 개수를 설정
    category_post_counts = []
    for category in categories:
        category_post_counts.append({
            "category_id": category.category_id,
            "category_name": category.name,
            "post_count": post_counts.get(category.category_id, 0)
        })

    return category_post_counts
