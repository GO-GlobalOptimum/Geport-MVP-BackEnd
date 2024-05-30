from typing import Union
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from dotenv import load_dotenv
import logging

# 환경 변수 로드
load_dotenv()

# FastAPI 앱 설정
app = FastAPI(root_path="/fastapi")

from app.router.geport import router as geport_router
from app.router.igeport import router as igeport_router
from app.router.tags import router as tags_router
from app.router.thumbnail import router as thumbnail_router
# 라우터 추가
app.include_router(geport_router)
app.include_router(igeport_router)
app.include_router(tags_router)
app.include_router(thumbnail_router)

# 세션 미들웨어 추가
app.add_middleware(SessionMiddleware, secret_key='!secret')

# OAuth 클라이언트 설정
oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
def read_root():
    return {"Hello": "World"}

# 구글 로그인 엔드포인트
@app.get('/login', tags=["auth"])
async def login(request: Request):
    redirect_uri = 'https://geport.blog/fastapi/auth/callback'  # root_path를 고려한 URI
    return await oauth.google.authorize_redirect(request, redirect_uri)

# 구글 로그인 콜백 엔드포인트
@app.get('/auth/callback', tags=["auth"])
async def auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        logger.info(f'Token received: {token}')

        # access_token을 사용하여 사용자 정보 가져오기
        user_info = await oauth.google.userinfo(token=token)
        logger.info(f'User info: {user_info}')
    except Exception as e:
        logger.error(f'Failed to get user info: {e}')
        raise HTTPException(status_code=400, detail="Failed to get user info") from e

    if not user_info:
        raise HTTPException(status_code=400, detail="Invalid token or user information")

    # 사용자 정보에서 이메일과 이름 추출
    email = user_info.get('email')
    name = user_info.get('name')

    # access_token을 클라이언트에 JSON 형태로 반환 (이메일과 이름 포함)
    return JSONResponse(content={"access_token": token['access_token'], "token_type": token['token_type'], "email": email, "name": name})

# 보호된 엔드포인트
@app.get('/protected', tags=["protected"])
async def protected_route(credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
    token = credentials.credentials

    # 토큰을 사용하여 사용자 정보 가져오기
    try:
        user_info = await oauth.google.userinfo(token={"access_token": token})
        logger.info(f'Protected route user info: {user_info}')
    except Exception as e:
        logger.error(f'Failed to get user info: {e}')
        raise HTTPException(status_code=400, detail="Failed to get user info") from e

    email = user_info.get('email')
    name = user_info.get('name')

    return {"message": "Access granted", "email": email, "name": name}
