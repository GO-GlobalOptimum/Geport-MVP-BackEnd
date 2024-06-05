from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.connection import get_read_db, get_write_db
from app.services.tags.tags import generate_tags

router = APIRouter()

@router.get("/posts/generate/tags/{post_id}")
def generate_post_tags(post_id: int, read_db: Session = Depends(get_read_db), write_db: Session = Depends(get_write_db)):
    """ AI로 태그를 생성합니다
        Post write시 같이 호출해주세요.

    Args:
        post_id (int): _description_
        read_db (Session, optional): _description_. Defaults to Depends(get_read_db).
        write_db (Session, optional): _description_. Defaults to Depends(get_write_db).

    Returns:
        _type_: _description_
    """    
    return generate_tags(post_id, read_db, write_db)
