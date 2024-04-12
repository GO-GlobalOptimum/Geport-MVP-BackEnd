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
    if users :
        return users
    else :
        raise HTTPException(status_code=404, detail="Users not found")


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



from concurrent.futures import ThreadPoolExecutor

def create_init_prompt():
    return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
                """
                You are a helper who summarizes the text in 8 sentences, finds various emotions in the text, 
                and finds words related to happiness in the text                
                """
                ),
            HumanMessagePromptTemplate.from_template(
                """
                ### 예시 입력
                Question: 이글을 요약하고, 행복과 관련된 감정, 슬픔과 관련된 감정을 찾아주고, 감정을 나타내는 단어는 아니지만 행복과 관련된 단어를 찾아줘
                Context: 오늘의 시드니 여행은 재미있었다. 오늘은 아침에 모닝빵과 스테이크를 먹었고 그 다음으로 근처 기념품관에서 오르골을 샀다.
                점심에는 바다가 보이는 레스토랑에서 와인과 함께 코스요리를 즐겼다. 저녁은 불꽃 축제와 함께 재미있는 친구들을 사귀며 놀았다.
                이번 여행은 정말 재미있었고 다음에도 다시 왔으면 좋겠다.

                ### 예시 출력
                오르골과 바다 그리고 불꽃축제와 함께 했던 정말 재미있었던 시드니 여행에 대한 이야기 입니다. 
                아침에는 스테이크와 빵을 먹으면서 즐거운 하루를 시작하였습니다.
                기념관에서 오르골을 사서 행복했고, 바다가 보이는 식당에서의 코스 요리를 즐겼습니다.
                친구들과 불꽃놀이를 하고 다양한 친구를 사귀었습니다.
                이 글은 시드니에 놀러가서 다양한 경험을 하고, 다양한 친구들을 사귀며 좋은 추억을 담은 이야기입니다.

                여행에서는 모니빵과 스테이크를 먹어 행복함을 느꼈고, 이번여행은 정말 즐겨웠다 라는 점에서 여행에 대한 행복감을 보이고 있습니다. 
                
                그리고 스테이크, 와인, 해변 의 단어가 행복과 연관되어있습니다.

                ### 입력
                Question: 이글을 요약하고, 행복과 관련된 감정, 슬픔과 관련된 감정을 찾아주고, 감정을 나타내는 단어는 아니지만 행복과 관련된 단어를 찾아줘
                Context: {context}

                ### 출력

                """
            )])


def create_prompt(type):
    # 글 요약 프롬프트
    if type == 1: 
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
                """
                you are a helpful summary assistant that make Great summary using 8 sentence and short answer type
                """
                ),
            HumanMessagePromptTemplate.from_template(
                """
                ### 예시 입력
                Question: 이 블로그 글을 요약해줄 수 있어?
                Context: 오늘의 시드니 여행은 재미있었다. 오늘은 아침에 모닝빵과 스테이크를 먹었고 그 다음으로 근처 기념품관에서 오르골을 샀다.
                점심에는 바다가 보이는 레스토랑에서 와인과 함께 코스요리를 즐겼다. 저녁은 불꽃 축제와 함께 재미있는 친구들을 사귀며 놀았다. 
                이번 여행은 정말 재미있었고 다음에도 다시 왔으면 좋겠다.

                ### 예시 출력
                오르골과 바다 그리고 불꽃축제와 함께 했던 정말 재미있었던 시드니 여행

                ### 입력
                Question: 이 블로그 글을 요약해줘
                Context: {context}

                ### 출력

                """
            )])
    # 감정물결 프롬프트
    elif type == 2:
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
                """
                you are a helpful assistant that analysis emotion using our input and you must give us format is JSON,
                we determine JSON format which each of emotion is key that is string, percentage is value that is integer
                and you must Present an answer in a format that exists as JSON using json.loads()
                """
                ),
            HumanMessagePromptTemplate.from_template(
                """
                ### 예시 입력
                Question: 이 블로그의 텍스트를 분석하여, 작성자가 글을 작성하며 경험했을 것으로 추정되는 주요 감정과 그 강도를 설명해주세요.
                Context: (관련 내용이 있는 블로그 한개)

                ### 예시 출력
                Answer : JSON 형식의 문자열

                ### 입력
                Question: 이 블로그의 내용을 분석하여, 작성자가 경험했을 것으로 추정되는 감정과 그 감정의 강도를 설명해주세요. 
                각 감정은 happy, joy, anxious, depressed, anger, sadness으로 구분해 설명하고, 각각의 감정이 글에서 어떻게 표현되었는지에 대한 예시를 포함해주세요.
                Context: {context}

                ### 출력
                Answer: """
        )])
    # 감정 SOS
    elif type == 3:
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
                """
                you are a helpful assistant that analysis emotion using our input and you must give us format is JSON,
                we determine JSON format which each of bad emotion is key that is string, percentage is value that is integer
                and you must Present an answer in a format that exists as JSON using json.loads()                """
            ),
            HumanMessagePromptTemplate.from_template(
               """
                ### 예시 입력
                Question: 이 블로그의 텍스트를 분석하여, 작성자가 글을 작성하며 경험했을 것으로 추정되는 안좋은 감정과 그 강도를 설명해주세요.
                Context: (관련 내용이 있는 블로그 한개)

                ### 예시 출력
                Answer : JSON 형식의 문자열

                ### 입력
                Question: 이 블로그의 내용을 분석하여, 작성자가 경험했을 것으로 추정되는 안좋은 감정과 그 감정의 강도를 설명해주세요. 
                각 감정은 sadness, anger, depressed, anxious 로 구분해 설명하고, 각각의 감정이 글에서 어떻게 표현되었는지에 대한 예시를 포함해주세요.
                Context: {context}

                ### 출력
                Answer: """
            )])
    # 힐링키워드
    elif type == 4:
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
                """
                You are a useful helper that uses our input to analyze sentiment.
                You tell us what objects, food, etc. the user was happy with via the input. The format should be JSON that you must key is happy word that string, value is happy intensity that integer
                i will use your answer for JSON.format
                """
            ),
            HumanMessagePromptTemplate.from_template(
                """
                ### 예시 입력
                Question: 이 블로그의 텍스트를 분석하여, 작성자가 어떠한 사물, 사람, 음식 등 행복감을 느꼈던 키워드가 무엇인지 단어를 추출해주세요(영어로 출력 바랍니다)
                Context: (관련 내용이 있는 블로그 글들)

                ### 예시 출력
                Answer : 블로그 내용들 중 행복감을 느끼게 했던 key word를 영어로 뽑아내서, 이의 강도를 같이 출력합니다.
                Ex."dog": 80, "chicken": 70, "beef": 75, "taylor swift": 65, "day6": 60

                ### 입력
                Question: 이 블로그의 내용을 분석하여, 작성자가 행복감을 느꼈던 요소 반드시 다섯개만 출력하도록.
                작성자가 행복감을 느꼈던 요소를 분석 할 때, 문맥을 통해 분석해주길 바랍니다. "행복함","즐거움","기쁨" 이런식으로 행복과 연관된 직접적인 단어들이 아닌,
                사물, 사람, 음식 등 "단어"를 추출해주세요. 이 단어들은 영어로 출력해주세요
                그리고 요소로 인해 어느정도 행복했는지 각각의 강도를 백분율로 설명해주세요.
                Context: {context}

                ### 출력
                Answer:  """
            )])
    elif type == 5:
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
                """
                You are a wise psychologist who must give us answers in json format (analyzing the Big 5 personality types),
                we determine JSON format which each of emotion is key that is string, score is value that is integer
                and you must Present an answer in a format that exists as JSON using json.loads()
                """
                ),
            HumanMessagePromptTemplate.from_template(
                """
                ### 예시 입력
                Question: 이건 사용자가 쓴 블로그 글이랑 사용자가 고른 big5 question의 문항 중 하나야. 이걸로 개방성, 성실성, 외향성, 우호성, 신경성에 대해 백분율 점수로 정리해줘
                answer: (user가 답한 대답들)
                Context: (user가 쓴 블로그 글)

                ### 예시 출력
                "openness" : 60,
                "sincerity" : 55,
                "extroversion" : 41,
                "friendliness" : 32,
                "neuroticism" : 60

                ### 입력
                Question: 이건 big5 question의 문항 중 하나야. 이걸로 개방성, 성실성, 외향성, 우호성, 신경성에 대해 백분율 점수로 정리해줘
                answer: {user}
                Context: {context}

                ### 출력
                """
            )])
    # 최종 솔루션
    elif type == 6:
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
                """
                you are a helpful summary assistant that make Great summary, Through emotion, big5 information, happiness keywords, and blog posts,
                you help summarize and organize user information professionally. and you must speak korean. don't answer need more data
                """
                ),
                HumanMessagePromptTemplate.from_template(
                """
                ### 예시 입력
                Question: 다음과 같은 정보들로 사용자를 요약해줘
                emotion : ("happy" : 81, "sadness" : 21, "anger : 11, "joy" : 91, "depressed" : 1, "anxious" : 23)
                big : ("openness" : 60, "sincerity" : 55, "extroversion" : 41, "friendliness" : 76, "neuroticism" : 30 )
                word : ("dog": 80, "chicken": 70, "beef": 75, "taylor swift": 65, "day6": 60)
                Context: (user가 쓴 블로그 글)

                ### 예시 출력
                Answer: 종합적으로 다양한 감정 중 기쁨 감정이 상대적으로 높고 개나 치킨과 같은 단어가 행복함을 잘 표현하는 것으로 보아
                사용자는 개나 치킨 등의 귀엽고 맛있는 것을 좋아하는 것으로 보이며 블로그 글 등에서 확인 할 수 있었습니다.
                그리고 글의 작성 요령이나 big5 결과를 보았을 때 딱히 크게 예민하지 않고 외향적인 사람으로써 바캉스를 매우 잘 즐길 수 있는 성격이라고 
                보여집니다. 따라서 다양한 액티비티한 활동과 여행이 사용자의 행복감을 증가시켜줄 것이며 부정적인 감정들을 없애 줄 것입니다.

                ### 입력
                Question: 다음과 같은 정보들로 사용자를 요약해줘
                emotion : {emotion}
                big : {big5}
                word : {word}
                Context: {context}

                ### 출력
                Answer: 
                """
            )])



def get_init4(split_docs):
    #analy_1 = create_prompt(1).format_prompt(context=split_docs[0]).to_messages()

    analyze_1 = create_init_prompt().format_prompt(context=split_docs[0]).to_messages()
    analyze_2 = create_init_prompt().format_prompt(context=split_docs[1]).to_messages()
    analyze_3 = create_init_prompt().format_prompt(context=split_docs[2]).to_messages()
    analyze_4 = create_init_prompt().format_prompt(context=split_docs[3]).to_messages()

    analyze_1 = llm35(analyze_1)
    analyze_2 = llm35(analyze_2)
    analyze_3 = llm35(analyze_3)
    analyze_4 = llm35(analyze_4)

    result = {
    "1st": analyze_1.content,
    "2nd": analyze_2.content,
    "3rd": analyze_3.content,
    "4th": analyze_4.content
}

    return result



def generate_igeport(encrypted_id: str):
    blog_urls = read_user_blog_links(encrypted_id)
    blog_docs = url_to_text(blog_urls)
    split_docs = split_text(blog_docs)
    #print(split_docs)

    answers = read_user_questions(encrypted_id)

    # 각 블로그에서 요약, 감정분석, 힐링키워드를 모두 담아둔다.
    inital_4 = get_init4(split_docs)


    return inital_4['1st']