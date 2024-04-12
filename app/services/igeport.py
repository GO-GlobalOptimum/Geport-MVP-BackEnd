from fastapi.responses import JSONResponse
from fastapi import HTTPException, status
import hashlib
from app.database.models import UserData, UserQuestions
from app.database.connection import igeport_user_baseInfo_collection
import os
from typing import List, Dict
import httpx
import json
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

env_path = os.path.join(os.path.dirname(__file__), '../../.env')
load_dotenv(dotenv_path=env_path)

# LLM 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, openai_api_key=OPENAI_API_KEY)




def create_encrypted_key(name, phone):
    hasher = hashlib.sha256()
    hasher.update(f"{name}{phone}".encode('utf-8'))
    return hasher.hexdigest()

def create_user_service(user_data: UserData):
    encrypted_key = create_encrypted_key(user_data.name, user_data.phone)

    if igeport_user_baseInfo_collection.find_one({"_id": encrypted_key}):
        # 이미 존재하는 경우, 적절한 에러 메시지를 반환합니다.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this ID already exists"
        )
    
    user_info_data = {
        "_id": encrypted_key,
        **user_data.dict(),
        "encrypted_id": encrypted_key
    }
    
    try:
        igeport_user_baseInfo_collection.insert_one(user_info_data)
        return {"encrypted_id": encrypted_key}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    




def read_user_service(encrypted_id: str):
    user = igeport_user_baseInfo_collection.find_one({"encrypted_id": encrypted_id}, {'_id': False})
    if user:
        return user
    else:
        raise HTTPException(status_code=404, detail="User not found")


def read_list_service():
    users = list(igeport_user_baseInfo_collection.find({}, {'_id': False}))
    return users




def read_user_blog_links(encrypted_id: str) -> list:
    user = igeport_user_baseInfo_collection.find_one({"encrypted_id": encrypted_id}, {'_id': False})
    if user and "blog_links" in user:
        return user["blog_links"]
    else:
        raise HTTPException(status_code=404, detail="User or blog links not found")
    
def read_user_questions(encrypted_id: str) -> list:
    user = igeport_user_baseInfo_collection.find_one({"encrypted_id": encrypted_id}, {'_id': False})
    if user and "questions" in user:
        return user["questions"]
    else:
        raise HTTPException(status_code=404, detail="User or questions not found")
    



def fetch_blog_contents(encrypted_id: str) -> list:
    # 사용자의 블로그 링크를 가져옵니다.
    blog_links = read_user_blog_links(encrypted_id)
    blog_contents = []

    for url in blog_links:
        try:
            # 각 블로그 URL로부터 HTML 내용을 가져옵니다.
            response = requests.get(url)
            response.raise_for_status()  # 에러가 발생하면 예외를 발생시킵니다.

            # BeautifulSoup을 사용하여 HTML 내용을 파싱합니다.
            soup = BeautifulSoup(response.text, 'html.parser')

            # 필요한 텍스트 정보를 추출합니다.
            # 이 예제에서는 전체 <body> 태그의 텍스트를 추출합니다.
            # 실제 사용 사례에 따라 적절한 태그 및 선택자를 사용하여 조정해야 합니다.
            content = soup.body.get_text(separator=' ', strip=True)
            blog_contents.append(content)
        except requests.RequestException as e:
            # 요청 중 에러가 발생하면, 에러 메시지와 함께 계속 진행합니다.
            print(f"Error fetching {url}: {e}")
            continue

    return blog_contents

def read_user_blog_links(encrypted_id: str) -> list:
    # 예시: MongoDB 컬렉션에서 사용자 정보를 조회합니다.
    user = igeport_user_baseInfo_collection.find_one({"encrypted_id": encrypted_id}, {'_id': False})
    if user and "blog_links" in user:
        return user["blog_links"]
    else:
        raise HTTPException(status_code=404, detail="User or blog links not found")


def url_to_text(url):
    loader = WebBaseLoader(url)
    # url을 문서호 한다.
    docs = loader.load()
    return docs


def split_text(docs):
    # 특정 크기로 쪼갠다. 문맥을 위해서 15- 토큰은 overlap 시킨다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    return splits


from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_igeport(encrypted_id: str):
    blogs = read_user_blog_links(encrypted_id)  # 사용자 블로그 링크 읽기
    blog_text = url_to_text(blogs)  # URL에서 텍스트 추출
    docs = split_text(blog_text)  # 텍스트 분할

    answers = read_user_questions(encrypted_id)
    # 병렬 처리를 위한 작업 목록 생성
    tasks = {
        "summary": f"Don't say thank you at the end of a sentence. Please write from the user's perspective. {docs} 블로그 내용들인데, 총 4개의 블로그를 각각 200 ~ 300자 이내로 요약해줘 그리고 각각의 블로그 마다 너가 임의의 제목을 붙혀줘",
        "emotion": f"Don't say thank you at the end of a sentence. Please write from the user's perspective. {docs} 블로그 내용들인데, 각각의 블로그에서 감정을 중점으로 설명해줘, 예를들어 친구들과 불꽃놀이를 봐서 즐거웠지만, 이후 집을 갈때 택시를 잡지 못해 화난 감정을 확인할수 있었습니다, 이런식으로",
        "emotion_sos": f"Don't say thank you at the end of a sentence. Please write from the user's perspective. {docs} 블로그 내용들인데, 블로그 내용에서 슬픔, 화남,우울, 불안이 각각 어느 정도인지 수치적으로 표현해줘, MAX를 10으로 봤을떄, 예를들어 슬픔 : 4, 화남: 2, 우울 :0, 불안 : 1 이런식으로 수치화 해줘, 그리고 왜 그렇게 분석했는지 짧게 요약해줘",
        "happy_keyword": f"You are a useful helper that uses our input to analyze sentiment.You tell us what objects, food, etc. the user was happy with via the input. The format should be JSON that you must key is happy word that string, value is happy intensity that integer i will use your answer for JSON.format. {docs} 블로그 내용들인데, 여기서 행복과 관련이 많은 단어를 찾아줘, 그리고 왜 행복한지도 분석해줘"
        }
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Future 객체를 작업과 매핑
        future_to_task = {executor.submit(llm35.predict, task): name for name, task in tasks.items()}

        # 완료된 작업 처리
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f'{task_name} generated an exception: {exc}')
            else:
                results[task_name] = result


    summary = results['summary']
    emotion = results['emotion']
    emotion_sos = results['emotion_sos']
    happy_keyword = results['happy_keyword']


    answer_1 = summary
    answer_2 = emotion
    answer_3 = emotion_sos
    answer_4 = llm35.predict(f"Don't say thank you at the end of a sentence. Speak like a psychoanalyst. {summary}이게 사용자의 블로그 내용이고, {emotion_sos} 가 사용자의 안좋은 감정들이고, {answers} 이게 본인이 판단한 본인의 모습이야, 이를 통해서 사용자가 행복을 느낄떄 나타나고, 실질적으로 감정은 아니지만, 행복과 관련된 단어가 나올떄 마다 같이 등장하는 단어 10개 정도 추천해줘. 행복, 유쾌함 이런거 말고.")
    answer_5 = llm35.predict(f"Don't say thank you at the end of a sentence. Speak like a psychoanalyst. {answers} 이게 사용자 자신이 자신을 평가 한 건데, {summary}를 보고 사용자의 성격을  개방성, 성실성, 외향성, 우호성, 신경증 측면에서 각각 분석해서 각각 max를 10으로 했을떄 어느정도인지 짧은 설명과 함께 수치적으로 분석해줘")
    answer_6 = llm35.predict(f"Don't say thank you at the end of a sentence. Speak like a psychoanalyst.  {answer_1}, {answer_2}, {answer_3}, {answer_4},{answer_5}를 모두 활용해서 사용자가 앞으로 어떤식으로 살아가야할지에 대해 분석해줘. 예를 들어 불안함 감정이 있다면 어떻게 해결해야하고, 또 어떤 성격은 좋으니깐 계속 유지해야되고 이런식으로")


    result= {
            "result": {
                "블로그 내용 요약": answer_1,
                "감정 물결": answer_2,
                "감정 SOS": answer_3,
                "힐링 키워드": answer_4,
                "성격 유형 검사": answer_5,
                "솔루션" : answer_6
            }
        }
    
    return result