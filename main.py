from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.sessions import SessionMiddleware
from app.services.auth.auth import get_current_user, authjwt_exception_handler, Settings

# 환경 변수 로드
load_dotenv()

# 라우터 임포트
from app.router.geport import router as geport_router
from app.router.igeport import router as igeport_router
# from app.router.tags import router as tags_router
# from app.router.thumbnail import router as thumbnail_router
# from app.router.recentView import router as recentView_router
# from app.router.personType import router as personType_router
# from app.router.categoryPerPost import router as categoryPerPost_router
# from app.router.recentViewPost import router as recentViewPost_router
# from app.router.recentViewPerson import router as recentViewPerson_router

app = FastAPI(root_path="/fastapi")

# 세션 미들웨어 추가
app.add_middleware(SessionMiddleware, secret_key='your_secret_key')

# 라우터 추가
app.include_router(geport_router)
app.include_router(igeport_router)
# app.include_router(tags_router)
# app.include_router(thumbnail_router)
# app.include_router(recentView_router)
# app.include_router(personType_router)
# app.include_router(categoryPerPost_router)
# app.include_router(recentViewPost_router)
# app.include_router(recentViewPerson_router)


# 인증 예외 핸들러 추가
app.add_exception_handler(AuthJWTException, authjwt_exception_handler)

@app.get("/")
def read_root():
    return {"Hello": "World"}

# 보호된 경로 추가
@app.get("/protected")
def protected_route(user: dict = Depends(get_current_user)):
    return {"msg": "You are authorized", "user": user}

@app.get("/user-info")
def user_info(user: dict = Depends(get_current_user)):
    return {"email": user["email"]}
