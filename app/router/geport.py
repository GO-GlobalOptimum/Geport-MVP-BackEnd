from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict
import logging
from app.services.geport.geport_asyncio import generate_geport as generate_geport_function, read_list_service
from app.database.connection import get_read_db, get_write_db, get_geport_db
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
async def generate_geport_endpoint(request_data: GenerateGeportRequest, read_db: Session = Depends(get_read_db), write_db: Session = Depends(get_write_db), mongo_db = Depends(get_geport_db), current_user: dict = Depends(get_current_user)):
    """
    Summary: geport를 생성하는 API 입니다.

    Parameters: post_ids, user_questions
    """

    if mongo_db is None:
        raise ValueError("Mongo DB collection is None")
    post_ids = request_data.post_ids
    questions = request_data.questions

    logging.info(f"post_ids: {post_ids}")
    logging.info(f"questions: {questions}")

    result = await generate_geport_function(post_ids, questions, read_db, write_db, mongo_db, current_user)
    return result

@router.get("/geport/database/list")
def get_geport_list(mongo_db = Depends(get_geport_db)):
    result = read_list_service(mongo_db)
    return result
