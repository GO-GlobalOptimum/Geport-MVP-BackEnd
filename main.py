from typing import Union
from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# 환경 변수 로드
load_dotenv()

# 라우터 임포트
from app.router.geport import router as geport_router
from app.router.igeport import router as igeport_router
from app.router.tags import router as tags_router
from app.router.thumbnail import router as thumbnail_router

class Settings(BaseModel):
    authjwt_secret_key: str = os.getenv("SECRET_KEY")
    authjwt_algorithm: str = os.getenv("ALGORITHM")

@AuthJWT.load_config
def get_config():
    return Settings()

def authjwt_exception_handler(request: Request, exc: AuthJWTException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

def get_current_user(Authorize: AuthJWT = Depends(), credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
    try:
        Authorize.jwt_required()
        raw_token = Authorize.get_raw_jwt()
        user_email = raw_token['email']  # 이메일 정보 추출
        return user_email
    except KeyError:
        raise HTTPException(status_code=400, detail="Email not found in token")
    except AuthJWTException as e:
        raise HTTPException(status_code=401, detail=str(e))

app = FastAPI(root_path="/fastapi")

# 라우터 추가
app.include_router(geport_router)
app.include_router(igeport_router)
app.include_router(tags_router)
app.include_router(thumbnail_router)

# 인증 예외 핸들러 추가
app.add_exception_handler(AuthJWTException, authjwt_exception_handler)

@app.get("/")
def read_root():
    return {"Hello": "World"}

# 보호된 경로 추가
@app.get("/protected")
def protected_route(user: str = Depends(get_current_user)):
    return {"msg": "You are authorized", "user": user}

@app.get("/user-info")
def user_info(email: str = Depends(get_current_user)):
    return {"email": email}
