
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.services.recentViewPerson.recentViewPerson import get_recently_viewed_person_types
from app.services.auth.auth import get_current_user
from pydantic import BaseModel
from typing import List, Dict, Any


router = APIRouter()

class PersonTypeCount(BaseModel):
    person: str
    count: int


@router.get("/recently-viewed-person-types/", response_model=List[PersonTypeCount])
async def get_recently_viewed_person_types_endpoint(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """
    사용자가 최근에 본 게시글의 작성자 유형을 조회하여 많은 유형 순으로 정렬하여 반환하는 API입니다.


    Parameters:
        db (Session): 데이터베이스 세션.
        current_user (dict): 현재 인증된 사용자 정보.


    Returns:
        List[PersonTypeCount]: 작성자 유형별 개수를 포함하는 리스트.
    """
    result = get_recently_viewed_person_types(db, current_user)
    return result

