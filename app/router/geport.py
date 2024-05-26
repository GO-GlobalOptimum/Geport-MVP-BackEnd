# geport.py
from fastapi import APIRouter, HTTPException, status, Request, Depends
from app.database.models import UserData, UserQuestions
from app.services.geport.geport_gpt import create_user_service, read_user_service, read_list_service as read_list_service_MVP, generate_geport as generate_geport_MVP
from app.services.geport.geport_clova import generate_geport as generate_geport_clova
from app.services.geport.geport_asyncio import read_list_service as read_list_service, generate_geport_MVP as generate_geport_MVP, generate_geport as generate_geport
from app.database.connection import get_db
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict


class GenerateGeportRequest(BaseModel):
    member_id: int
    post_ids: List[int]
    questions: List[str]

class GeportResponse(BaseModel):
    member_id: int
    geport_id: str
    result: Dict[str, str]

router = APIRouter()

@router.post("/fastapi/geport/generate/", response_model=GeportResponse)
async def generate_geport_endpoint_text(request_data: GenerateGeportRequest, db: Session = Depends(get_db)):
    """
        Summary: geport를 생성하는 API 입니다.

        Parameters: member_id, post_ids, user_questions
    """
    member_id = request_data.member_id
    post_ids = request_data.post_ids
    questions = request_data.questions

    print(f"member_id: {member_id}")
    print(f"post_ids: {post_ids}")
    print(f"questions: {questions}")

    result = await generate_geport(member_id, post_ids, questions, db)
    
    return result

@router.get("/fastapi/geport/database/list")
def get_geport_list():
    result = read_list_service()
    return result

#__________________________________MVP  __________________________________________

# @router.post("fastapi/geport/generate/userInfo", status_code=status.HTTP_201_CREATED)

# def create_user(user_data: UserData):
#     """
#     _summary_: MVP에서 특정 사용자가 정보를 입력하면 해당 정보를 DB에 저장하는 API 입니다.

#     _params_ : UserData
#     """
#     return create_user_service(user_data)

# @router.get("fastapi/geport/read/userInfo/{encrypted_id}")
# def read_user(encrypted_id: str):
#     """
#     _summary_: MVP에서 사용자 id로 userInfo 테이블에서 특정 사용자의 정보를 가져오는 API 입니다.

#     _params_ : encrypted_id
#     """
#     return read_user_service(encrypted_id)


# @router.get("fastapi/geport/infoList")
# def read_list():
#     """
#     _summary_: MVP에서 userInfo MongoDB 안의 모든 내용을 보여주는 API 입니다.

#     _params_ : 
#     """
#     return read_list_service_MVP()


# @router.get("fastapi/geport/generate/Geport/Clova/{encrypted_id}")  # 클로바를 이용한 지포트 생성
# def generate_geport_endpoint_clova(encrypted_id: str):
#     """
#     _summary_: MVP 애서 geport를 CLOVA로 생성하는 API 입니다.

#     _params_ : encrypted_id
#     """
#     result = generate_geport_clova(encrypted_id)
#     return result


# @router.get("fastapi/geport/generate/Geport/GPT/{encrypted_id}")  
# def generate_geport_endpoint_gpt(encrypted_id: str):
#     """
#     _summary_: MVP 애서 geport를 GPT로 생성하는 API 입니다.

#     _params_ : encrypted_id
#     """
#     result = generate_geport_MVP(encrypted_id)
#     return result


# @router.get("fastapi/geport/generate/Geport/GPT/Asyncio/MVP/{encrypted_id}")
# async def generate_geport_endpoint_asyncio(encrypted_id: str):
#     """
#     _summary_: MVP 애서 geport를 GPT로 병렬적으로 생성하는 API 입니다.

#     _params_ : encrypted_id
#     """
#     result = await generate_geport_MVP(encrypted_id)
#     return result

#__________________________________MVP  __________________________________________

