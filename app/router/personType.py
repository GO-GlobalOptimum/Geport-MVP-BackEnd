from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.services.personType.personType import generate_personType
from app.database.connection import get_db
from app.services.auth.auth import get_current_user
from app.services.personType.personType import User


router = APIRouter()

@router.get("/generate/PersonType")
def generate_person_type(db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    """
    현재 사용자의 최근 게시글 10개의 AI tag를 이용해서 해당 사람이 어떠한 사람인지 판단한다


    Args:
        db (Session): SQLAlchemy 세션
        user (dict): 현재 사용자 정보


    Returns:
        _type_: JSON
    """
    user_model = User(email=user["email"])
    return generate_personType(db, user_model)

