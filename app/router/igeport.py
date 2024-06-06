from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from app.services.igeport.igeport_asyncio import read_list_service, generate_igeport as generate_igeport_function
from app.database.connection import get_read_db, get_write_db, get_igeport_db
from app.services.auth.auth import get_current_user

class GenerateIgeportRequest(BaseModel):
    post_ids: List[int]
    questions: List[str]

class IgeportResponse(BaseModel):
    member_id: int
    igeport_id: str
    result: Dict[str, Any]


router = APIRouter()

@router.post("/igeport/generate/", response_model=IgeportResponse)
async def generate_igeport_endpoint(
    request_data: GenerateIgeportRequest,
    read_db: Session = Depends(get_read_db),
    write_db: Session = Depends(get_write_db),
    igeport_db = Depends(get_igeport_db),
    current_user: dict = Depends(get_current_user)
):

    """
    Summary: igeport를 생성하는 API입니다.

    Parameters: post_ids, user_questions
    """
    post_ids = request_data.post_ids
    questions = request_data.questions

    logging.info(f"post_ids: {post_ids}")
    logging.info(f"questions: {questions}")


    # post_id 중 하나를 사용하여 member_id 조회
    query_str = "SELECT member_id FROM Post WHERE post_id = :post_id LIMIT 1"
    query = text(query_str)
    params = {"post_id": post_ids[0]}


    try:
        result = read_db.execute(query, params).fetchone()
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")

    if not result:
        raise HTTPException(status_code=404, detail="Post not found for the given post_id")

    # user가 보낸 post_id를 통해서 member_id를 가지고 온다.
    member_id = result.member_id

    logging.info(f"member_id: {member_id}")

    # current_user의 member_id와 조회한 member_id 비교
    # if current_user.get("member_id") != member_id:
    #     raise HTTPException(status_code=403, detail="You are not authorized to generate this report")

    # Igeport 생성 함수를 소환한다( post_id, 질문, SQL 읽기, SQL 쓰기, MongoDB, 현재 사용자 정보 )
    result = await generate_igeport_function(post_ids, questions, read_db, write_db, igeport_db)
    return IgeportResponse(
        member_id=result['member_id'],
        igeport_id=result['igeport_id'],
        result=result['result']
    )

@router.get("/igeport/database/list")
def get_igeport_list(igeport_db = Depends(get_igeport_db)):
    result = read_list_service(igeport_db)

    return result
