from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from pydantic import BaseModel
from fastapi import Depends, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class Settings(BaseModel):
    authjwt_secret_key: str = os.getenv("SECRET_KEY")
    authjwt_algorithm: str = os.getenv("ALGORITHM")

@AuthJWT.load_config
def get_config():
    return Settings()

def authjwt_exception_handler(request: Request, exc: AuthJWTException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

def get_current_user(Authorize: AuthJWT = Depends(), request: Request = None):
    try:
        token = request.cookies.get("accessToken")
        print(token)

        Authorize.jwt_required()
        raw_token = Authorize.get_raw_jwt(token)
        user_email = raw_token['email']  # 이메일 정보 추출
        return {"email": user_email}
    
    except KeyError:
        raise HTTPException(status_code=400, detail="Email not found in token")
    except AuthJWTException as e:
        raise HTTPException(status_code=401, detail=str(e))
