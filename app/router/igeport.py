from fastapi import APIRouter, status, Depends
from app.database.connection import igeport_get_read_db, get_read_get_read_db
from app.services.igeport.igeport_asyncio import read_list_service, generate_igeport
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any
from app.services.auth.auth import get_current_user
import logging


router = APIRouter()

class GenerateGeportRequest(BaseModel):
    post_ids: List[int]
    questions: List[str]

class GeportResponse(BaseModel):
    member_id: int
    igeport_id: str
    result: Dict[str, Any]

@router.post("/igeport/generate/", response_model=GeportResponse)
async def generate_geport_endpoint_text(request_data: GenerateGeportRequest, get_read_db: Session = Depends(get_read_get_read_db), current_user: dict = Depends(get_current_user)):
    """
    Summary: igeport 생성하는 API 입니다.

    Parameters: post_ids, user_questions
    """
    post_ids = request_data.post_ids
    questions = request_data.questions

    logging.info(f"post_ids: {post_ids}")
    logging.info(f"questions: {questions}")

    result = await generate_igeport(post_ids, questions, get_read_db, current_user)

    return GeportResponse(
        member_id=result['member_id'],
        igeport_id=result['igeport_id'],
        result=result['result']
    )


@router.get("/igeport/database/list")
def get_geport_list():
    result = read_list_service()
    return result