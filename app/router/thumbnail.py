from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.connection import get_read_db, get_write_db
from app.services.thumbnail.thumbnail_text import generate_thumbnailText
from app.services.thumbnail.thumbnail_image import generate_thumbnailImage

router = APIRouter()

# post에서 썸내일 글 만들어내기
@router.get("/posts/generate/thumbnail/text/{post_id}")
def generate_thumbnail_text(post_id: int, read_db: Session = Depends(get_read_db)):
    return generate_thumbnailText(post_id, read_db)


# post에서 썸내일 이미지 만들어내기
@router.get("/posts/generate/thumbnail/image/{post_id}")
def generate_thumbnail_image(post_id: int, read_db: Session = Depends(get_read_db), write_db: Session = Depends(get_write_db)):
    return generate_thumbnailImage(post_id, read_db, write_db)
