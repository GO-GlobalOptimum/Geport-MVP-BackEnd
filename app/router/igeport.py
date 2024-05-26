# todos.py에서 router 정의
from fastapi import APIRouter, status, Depends
from app.database.connection import get_db
from app.database.models import UserData
from app.services.igeport.igeport_asyncio import create_user_service, read_list_service, read_user_service, generate_igeport
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter()

# @router.post("fastapi/igeport/generate/userInfo", status_code=status.HTTP_201_CREATED)
# def create_user(user_data: UserData):
#     return create_user_service(user_data)


# @router.get("fastapi/igeport/read/userInfo/{encrypted_id}")
# def read_user(encrypted_id: str):
#     return read_user_service(encrypted_id)

# @router.get("fastapi/igeport/infoList")
# def read_list():
#     return read_list_service()


# @router.get("fastapi/igeport/generate/Igeport/GPT/Asyncio/{encrypted_id}")
# async def generate_geport_endpoint(encrypted_id:str):
#     # service.py의 generate_geport 함수 호출
#     result = await generate_igeport(encrypted_id)
#     return result


class GenerateGeportRequest(BaseModel):
    member_id: int
    questions: List[str]
    post_ids: List[int]

class GeportResponse(BaseModel):
    member_id: int
    igeport_id: str
    result: Dict[str, Any]

@router.post("/fastapi/igeport/generate/", response_model=GeportResponse)
async def generate_geport_endpoint_text(request_data: GenerateGeportRequest, db: Session = Depends(get_db)):
    """
    Summary: igeport 생성하는 API 입니다.

    Parameters: member_id, post_ids, user_questions
    """
    member_id = request_data.member_id
    post_ids = request_data.post_ids
    questions = request_data.questions

    print(f"member_id: {member_id}")
    print(f"post_ids: {post_ids}")

    result = await generate_igeport(member_id, post_ids, questions, db)

    return GeportResponse(
        member_id=result['member_id'],
        igeport_id=result['igeport_id'],
        result=result['result']
    )


@router.get("/fastapi/igeport/database/list")
def get_geport_list():
    result = read_list_service()
    return result