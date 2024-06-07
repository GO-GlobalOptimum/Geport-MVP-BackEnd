from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi import HTTPException
import logging

def get_user_email_from_member_id(db: Session, member_id: int) -> str:
    query_str = text("SELECT email FROM Member WHERE member_id = :member_id")
    params = {"member_id": member_id}

    result = db.execute(query_str, params).fetchone()
    if result:
        return result.email
    raise HTTPException(status_code=404, detail="User email not found for the given member_id")

def get_recently_viewed_person_types(db: Session, current_user: dict):
    """
    사용자가 최근에 본 게시글의 작성자 유형을 조회하여 많은 유형 순으로 정렬하여 반환하는 함수입니다.

    Parameters:
        db (Session): 데이터베이스 세션.
        current_user (dict): 현재 인증된 사용자 정보.

    Returns:
        List[Dict[str, Any]]: 작성자 유형별 개수를 포함하는 리스트.
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
    query_str = text("SELECT post_id FROM View WHERE member_id = :member_id ORDER BY updated_time DESC LIMIT 10")
    params = {"member_id": member_id}

    try:
        post_ids = db.execute(query_str, params).fetchall()
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")

    if not post_ids:
        return []

    post_ids = [row.post_id for row in post_ids]

    # 포스트 ID를 통해 작성자의 member_id를 조회합니다.
    query_str = text("SELECT member_id FROM Post WHERE post_id IN :post_ids")
    params = {"post_ids": tuple(post_ids)}

    try:
        member_ids = db.execute(query_str, params).fetchall()
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")

    member_ids = [row.member_id for row in member_ids]

    # 작성자의 member_id를 통해 person 유형을 조회하고 개수를 셉니다.
    query_str = text("""
    SELECT person, COUNT(*) as count
    FROM Member
    WHERE member_id IN :member_ids
    GROUP BY person
    ORDER BY count DESC
    """)
    params = {"member_ids": tuple(member_ids)}

    try:
        result = db.execute(query_str, params).fetchall()
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")

    return [{"person": row.person, "count": row.count} for row in result]