from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from typing import List, Dict
import logging
from app.services.geport.geport_asyncio import read_list_service, generate_geport
from app.database.connection import get_db
from app.services.auth.auth import get_current_user

class GenerateGeportRequest(BaseModel):
    post_ids: List[int]
    questions: List[str]

class GeportResponse(BaseModel):
    member_id: int
    geport_id: str
    result: Dict[str, str]

router = APIRouter()

@router.post("/geport/generate/", response_model=GeportResponse)
async def generate_geport_endpoint_text(
    request_data: GenerateGeportRequest, 
    read_db: Session = Depends(get_db), 
    current_user: dict = Depends(get_current_user)
):
    """
    Summary: geport를 생성하는 API 입니다.

    Parameters: post_ids, user_questions
    """
    post_ids = request_data.post_ids
    questions = request_data.questions

    logging.info(f"post_ids: {post_ids}")
    logging.info(f"questions: {questions}")

    # post_id 중 하나를 사용하여 member_id 조회
    query_str = f"SELECT member_id FROM Post WHERE post_id = :post_id LIMIT 1"
    query = text(query_str)
    params = {"post_id": post_ids[0]}

    try:
        result = read_db.execute(query, params).fetchone()
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")

    if not result:
        raise HTTPException(status_code=404, detail="Post not found for the given post_id")

    member_id = result.member_id

    logging.info(f"member_id: {member_id}")

    result = await generate_geport(post_ids, questions, read_db)
    return result

@router.get("/geport/database/list")
def get_geport_list():
    result = read_list_service()
    return result
