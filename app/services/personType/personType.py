from sqlalchemy.orm import Session
from fastapi import HTTPException
from sqlalchemy.sql import text
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import HTTPException, status
import hashlib
from app.database.models import UserData, UserQuestions
from app.database.connection import igeport_user_baseInfo_collection, igeport_db
import os
from typing import List, Dict
import httpx
import json
import requests
# from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
import asyncio
import logging
import time
from sqlalchemy.orm import Session
from sqlalchemy import text 


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, openai_api_key=OPENAI_API_KEY, model_kwargs={"response_format": {'type':"json_object"}}, request_timeout=300)



class User(BaseModel):
    email: str

def get_post_tags_for_recent_posts(db: Session, current_user: User):
    """
    JWT를 통해 사용자 인증: current_user를 통해 현재 사용자의 이메일을 가져옵니다.
    사용자 ID 조회: member 테이블에서 이메일을 사용하여 사용자 ID를 조회합니다.
    최근 게시물 조회: post 테이블에서 해당 사용자의 최근 10개의 post_id를 조회합니다.
    태그 조회: post_tag 테이블에서 최근 10개의 post_id에 해당하는 태그를 조회합니다.
    결과 반환: 조회된 태그 목록을 반환합니다.

    Args:
        db (Session): SQLAlchemy 세션
        current_user (User): 현재 사용자

    Raises:
        HTTPException: 사용자 ID를 찾을 수 없는 경우
        HTTPException: 게시물을 찾을 수 없는 경우
        HTTPException: 기타 데이터베이스 오류

    Returns:
        list: 태그 목록
    """
    try:
        # 사용자 이메일을 통해 사용자 ID를 가져옵니다.
        user_id_query = text("""
            SELECT member_id
            FROM Member
            WHERE email = :email
        """)
        user_id_result = db.execute(user_id_query, {"email": current_user.email}).fetchone()

        if not user_id_result:
            raise HTTPException(status_code=404, detail="User not found")

        user_id = user_id_result.member_id
        print(f"User ID: {user_id}")

        # 해당 사용자의 최근 10개의 post_id를 가져옵니다.
        recent_posts_query = text("""
            SELECT post_id
            FROM Post
            WHERE member_id = :user_id
            ORDER BY createdAt DESC
            LIMIT 10
        """)
        recent_posts = db.execute(recent_posts_query, {"user_id": user_id}).fetchall()

        if not recent_posts:
            raise HTTPException(status_code=404, detail="No recent posts found")

        post_ids = [post.post_id for post in recent_posts]
        print(f"Recent Post IDs: {post_ids}")

        # 최근 10개의 post_id에 대한 태그를 가져옵니다
        tags_query = text("""
            SELECT post_id, post_tag_id, contents
            FROM PostTag
            WHERE post_id IN :post_ids AND is_user = 0
        """)
        tags_result = db.execute(tags_query, {"post_ids": tuple(post_ids)}).fetchall()

        if not tags_result:
            print(f"No tags found for post IDs: {post_ids}")
            raise HTTPException(status_code=404, detail="No tags found for recent posts")

        tags = [dict(tag._mapping) for tag in tags_result]
        print(f"Tags: {tags}")

        return tags

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




def get_concatenated_tags(db: Session, current_user: User) -> str:
    """
    현재 사용자의 최근 게시글 10개의 태그를 하나의 문자열로 묶어서 반환합니다.

    Args:
        db (Session): SQLAlchemy 세션
        current_user (User): 현재 사용자

    Returns:
        str: 하나의 문자열로 묶인 태그 내용
    """
    tags = get_post_tags_for_recent_posts(db, current_user)

    # 태그가 하나라도 있는지 확인
    if not tags:
        raise HTTPException(status_code=404, detail="No tags found for recent posts")

    concatenated_tags = ', '.join(tag['contents'] for tag in tags)
    return concatenated_tags






def create_prompt():
    return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
                """
                    A helper that takes a sentence with a tag string and tells you the subject of the blog post.  
Tag list:
    Technical
    Economic
    Scientific
    Cultural
    Political
    Educational
    Environmental
    Health
    Lifestyle
    Travel
    Food
    Fashion
    Art
    Literature
    Sports
    Entertainment
    Technology
    Finance
    Psychology
    Philosophy
Notify within this content.
The result should always be displayed in Korean and output as JSON.
                """
                ),
            HumanMessagePromptTemplate.from_template(
                """
                Input : CNN, 기술, 과학, 딥러닝, 비전, 포항공대

                ### Example Output :
                {{ "result": "기술적인" }}

                ### Input
                Input: {context}

                ### Output


                """

            )])


def generate_personType(db, current_user):
    """_summary_
        : 사용자의 최근 게시글 10개를 가지고 와서 해당 게시글에 AI가 만든 태그를 통해서
        사용자가 어떠한 사람인지를 만들어주는 함수이다.
        Client에서 원할떄 해당 API를 호출하면 DB에 사용자의 Type이 만들어진다.
    Args:
        db (_type_): _description_
        current_user (_type_): _description_

    Raises:
        HTTPException: _description_

    Returns:
        _type_: _description_
    """    
    tags = get_concatenated_tags(db, current_user)
    

    prompt1 = create_prompt().format_prompt(context=tags).to_messages()

    person_type = llm35.invoke(prompt1)


     # JSON 응답에서 content 값만 추출
    content = person_type.content
    print('content : ', content)
    
    # content 값을 JSON으로 파싱
    person_type = json.loads(content.strip())

    user_id_query = text("""
            SELECT member_id
            FROM Member
            WHERE email = :email
        """)
    user_id_result = db.execute(user_id_query, {"email": current_user.email}).fetchone()

    if not user_id_result:
        raise HTTPException(status_code=404, detail="User not found")

    user_id = user_id_result.member_id
    

    # tags 데이터베이스에 저장
    update_query = text("UPDATE Member SET person = :person WHERE member_id = :member_id")
    db.execute(update_query, {"person": person_type["result"], "member_id": user_id})  
    db.commit()


   

    return person_type
    