# geport.py
from fastapi import APIRouter, HTTPException, status, Body
from app.database.models import UserData, UserQuestions
from app.services.geport import create_user_service, read_user_service, read_list_service, generate_geport


router = APIRouter()


@router.post("/geport/create", status_code=status.HTTP_201_CREATED)
def create_user(user_data: UserData):
    return create_user_service(user_data)


@router.get("/geport/read/{encrypted_id}")
def read_user(encrypted_id: str):
    return read_user_service(encrypted_id)


@router.get("/geport/list")
def read_list():
    return read_list_service()


@router.get("/geport/generate-test/{encrypted_id}") #실제 지포트 생성
def generate_geport_endpoint(encrypted_id:str):
    # service.py의 generate_geport 함수 호출
    result = generate_geport(encrypted_id)
    return result


@router.get("/geport/generate-dummy/{encrypted_id}") # 더미 데이터 전송
def dummy_data(encrypted_id: str):
    # 더미 데이터 정의
    dummy_response = {
        "encrypted_id": encrypted_id,
        "result": {
            "저는 이런 사람이 되고싶어요": "더미 응답 데이터입니다.",
            "저의 좌우명은 다음과 같습니다": "더미 응답 데이터입니다.",
            "제 인생의 변곡점은 다음과 같아요": "더미 응답 데이터입니다.",
            "이것이 재 인생 함수입니다": {"content" :"더미 응답 데이터입니다.", 
                               "function" : "y = -x^2"
                               },
            "Geport Solution": "더미 응답 데이터입니다.",
        }
    }
    return dummy_response
    
