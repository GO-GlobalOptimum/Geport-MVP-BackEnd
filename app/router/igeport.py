# todos.py에서 router 정의
from fastapi import APIRouter, status
from app.database.models import UserData
from app.services.igeport import create_user_service, read_list_service, read_user_service, generate_igeport


router = APIRouter()

@router.post("/igeport/create", status_code=status.HTTP_201_CREATED)
def create_user(user_data: UserData):
    return create_user_service(user_data)


@router.get("/igeport/read/{encrypted_id}")
def read_user(encrypted_id: str):
    return read_user_service(encrypted_id)

@router.get("/igeport/list")
def read_list():
    return read_list_service()


@router.get("/igeport/generate-test/{encrypted_id}")
def generate_geport_endpoint(encrypted_id:str):
    # service.py의 generate_geport 함수 호출
    result = generate_igeport(encrypted_id)
    return result

@router.get("/igeport/generate-dummy/{encrypted_id}") # 더미 데이터 전송
def dummy_data(encrypted_id: str):
    # 더미 데이터 정의
    dummy_response = {
        "encrypted_id": encrypted_id,
        "result": {
            "answer1" : {"1일차" : "1일차 내용입니다","2일차" : "2일차 내용입니다","3일차" : "3일차 내용입니다","4일차" : "4일차 내용입니다",},
            "answer2" : {"happy": [10,20,30,40], "joy": [40,30,20,40], "anxious": [20,20,10,10], "depressed": [0,0,10,5], "anger": [5,0,5,10], "sadness": [0,0,0,10]},
            "answer3" : {"anxious": [10,20,30,40], "depressed": [40,20,40,10], "anger": [0,10,20,40], "sadness": [40,10,20,20]},
            "answer4" : {"dog": 80, "chicken": 70, "beef": 75, "taylor swift": 65, "day6": 60},
            "answer5" : {"openness" : 60,
                         "sincerity" : 55,
                         "extroversion" : 41,
                         "friendliness" : 32,
                         "neuroticism" : 60
                        },
            "answer6" : "최종 솔루션입니다",
        }
    }
    return dummy_response
