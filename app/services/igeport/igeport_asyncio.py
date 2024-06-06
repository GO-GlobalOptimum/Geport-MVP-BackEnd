from fastapi.responses import JSONResponse
from fastapi import HTTPException, status,Depends
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


env_path = os.path.join(os.path.dirname(__file__), '../../.env')
load_dotenv(dotenv_path=env_path)

# LLM 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, openai_api_key=OPENAI_API_KEY,request_timeout=500,model_kwargs={"response_format": {"type": "json_object"}},  # JSON 응답 형식 지정
)




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


# def read_list_service():
#     users = list(igeport_user_baseInfo_collection.find({}, {'_id': False}))
#     if users :
#         return users
#     else :
#         raise HTTPException(status_code=404, detail="Users not found")


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


async def url_to_text(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return '\n'.join([doc.page_content for doc in docs]) if docs else ""
    except Exception as e:
        logging.error(f"Failed to load or process URL {url}: {str(e)}")
        return ""

async def url_2_text(url_list):
    global document_storage
    document_storage = []  # Reset or initialize the storage
    # Process each URL and extract text
    text1, text2, text3, text4= await asyncio.gather(
        url_to_text(url_list[0]),
        url_to_text(url_list[1]),
        url_to_text(url_list[2]),
        url_to_text(url_list[3]),
    )
    texts = []
    texts.append(text1)
    texts.append(text2)
    texts.append(text3)
    texts.append(text4)
    return texts


def split_text(docs):
    # 특정 크기로 쪼갠다. 문맥을 위해서 15- 토큰은 overlap 시킨다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    return splits



from concurrent.futures import ThreadPoolExecutor

def create_init_prompt():
    return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
                """
                you are a helpful summary assistant who summarizes the text in 10 sentences, and finds various emotions like sad, in the text, 
                and finds words related to happiness in the text. pelase answer with korean.    
                The return format must be JSON. Please answers in Korean   
                """
                ),
            HumanMessagePromptTemplate.from_template(
                """
                ### Example Input
                Question: Summarize this article, find emotions related to happiness and sadness, and identify words that are not directly emotional but are related to happiness.
                Context: Today's trip to Sydney was fun. In the morning, I had morning bread and steak, and then I bought a music box at a nearby souvenir shop.
                For lunch, I enjoyed a course meal with wine at a restaurant overlooking the sea. In the evening, I had fun making new friends at a fireworks festival.
                I really enjoyed this trip and would love to come back again.

                ### Example Output :
                {{ "summary": "This is summary.","emotions": {{"happiness": 0,"joy": 0,"anxiousness": 0,"depression": 0,"anger": 0,"sadness": 0}}, "related_words": ["this is related words in array."] }}

                ### Input
                Question: Summarize this article, find emotions related to happiness and sadness, and identify words that are not directly emotional but are related to happiness.
                Context: {{context}}

                ### Output


                """
            )])


def create_prompt(type):
    # 글 요약 프롬프트
    if type == 1: 
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
                """
                You are a helpful summary assistant tasked with creating concise summaries for blog posts. Each summary should be a simple, straightforward text and presented in a uniform format across all blogs. Ensure that the output is consistent and clear for each entry, and use plain text for the summaries.
                longer than 10 stentence for each blog!
                The return format must be JSON. Please answers in Korean   
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
                {{"summary": "This is summary","source": " "}}

                

                ### 입력
                Question: 이 블로그 글을 요약해줘
                Context: {context}

                ### 출력

                """
            )])
    # 감정물결 프롬프트
    elif type == 2:
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """
                You are a helpful assistant trained to analyze emotions from text inputs and return the data in JSON format. 
                Your response should include the intensity of each analyzed emotion as an integer and a JSON object explaining the reasons behind these emotions based on the content analysis. 
                The emotions to analyze are happiness, joy, anxiousness, depression, anger, and sadness. The explanation should be concise and provided in Korean.
                """
            ),
            HumanMessagePromptTemplate.from_template(
                """
                ### 예시 입력
                Question: Analyze this blog text to identify the main emotions experienced by the author and their intensities.
                Context: (A single blog post with relevant content)

                ### 예시 출력
                Answer: {{
                    "emotions": {{
                        "happiness": 40,
                        "joy": 50,
                        "anxious": 10,
                        "depressed": 5,
                        "anger": 20,
                        "sadness": 15
                    }},
                    "contents": "이 텍스트에서 행복과 기쁨이 높게 나타난 이유는 저자가 최근 긍정적인 사건을 경험했기 때문입니다. 반면, 불안과 우울함은 낮게 나타난 것은 큰 어려움이 없었기 때문으로 보입니다."
                }}

                ### 입력
                Question: Analyze the content of this blog to describe the emotions experienced by the author and the intensity of these emotions.
                Context: {context}

                ### 출력
                Answer:
                """
            )
        ])

    # 감정 SOS
    elif type == 3:
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """
                You are a useful assistant that will analyze the sentiment using our input and you will need to provide it in JSON format.
                The response should include two main elements: one JSON object with the percentages of each sentiment (sadness, anger, depressed, anxious) and another JSON object with a general reason explaining why these sentiments were perceived in the text based on the content analysis, with the explanation provided in Korean. The format must be strictly JSON.
                """
            ),
            HumanMessagePromptTemplate.from_template(
                """
                ### 예시 입력
                Question: Analyze the sentiments expressed in this blog post and provide a clear and concise JSON response detailing the percentages of each sentiment (sadness, anger, depressed, anxious) and a combined explanation for these sentiments in Korean.

                Context: {{context}}

                ### 예시 출력
                Answer: {{
                    "sentiments": {{
                        "sadness": 10,
                        "anger": 30,
                        "depressed": 20,
                        "anxious": 40
                    }},
                    "contents": "다양한 감정이 감지되는 이유는 저자가 텍스트에서 겪은 경험과 표현 때문입니다. 이는 개인적 성장과 복지에 대한 고민, 좌절과 자기 의심의 순간, 그리고 일상 생활에서 마주한 도전들을 포함합니다."
                }}

                ### 입력
                Question: Analyze the sentiments expressed in this blog post and provide a clear and concise JSON response detailing the percentages of each sentiment (sadness, anger, depressed, anxious) and a combined explanation for these sentiments in Korean.
                Context: {context}

                ### 출력
                Answer:
                """
            )
        ])
    # 힐링 키워드
    elif type == 4:
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
                """
                You are a useful helper that uses our input to analyze sentiment.
                you must return JSON 
                You tell us what objects, food, etc. the user was happy with via the input. The format should be JSON that you must key is happy word that string, value is happy intensity that integer
                i will use your answer for JSON. please answer more than 10 keywords
                The format should be JSON. Please answers in korean
                """
            ),
            HumanMessagePromptTemplate.from_template(
                """
                ### 예시 입력
                Question: 이 블로그의 텍스트를 분석하여, 작성자가 어떠한 사물, 사람, 음식 등 행복감을 느꼈던 키워드가 무엇인지 단어를 10 ~ 20개 정도 추출해주세요(영어로 출력 바랍니다)
                Context: (관련 내용이 있는 블로그 글들)

                ### 예시 출력
                Answer : 블로그 내용들 중 행복감을 느끼게 했던 key word를 영어로 뽑아내서, 이의 강도를 같이 출력합니다. 단어는 10개 이상 반환해주세요

                ### 입력
                Question: 이 블로그의 내용을 분석하여, 작성자가 행복감을 느꼈던 요소 반드시 다섯개만 출력하도록.
                작성자가 행복감을 느꼈던 요소를 분석 할 때, 문맥을 통해 분석해주길 바랍니다. "행복함","즐거움","기쁨" 이런식으로 행복과 연관된 직접적인 단어들이 아닌,
                사물, 사람, 음식 등 "단어"를 추출해주세요. 이 단어들은 영어로 출력해주세요
                그리고 요소로 인해 어느정도 행복했는지 각각의 강도를 백분율로 설명해주세요.
                Context: {context}

                ### 출력
                Answer:  """
            )])
    # big5 성격
    elif type == 5:
        return ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
       You are a wise psychologist who analyzes personality based on given inputs and provides the analysis in JSON format.
    Your response should include a JSON object with keys as the personality traits (openness, sincerity, extroversion, friendliness, neuroticism), each having a percentage score. Additionally, provide a comprehensive summary of these traits in natural and cohesive Korean.
    Ensure the answer is formatted strictly in JSON.
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
        ### 예시 입력
        Question: Analyze the personality traits based on the user's blog post and the answers to the big5 questions.
        Context: (블로그 글 내용)
        Answers: (big5 설문 응답)

        ### The JSON output should be in the following format:

            {{
            "openness": {{
                "big5_score": 60
            }},
            "sincerity": {{
                "big5_score": 55
            }},
            "extroversion": {{
                "big5_score": 41
            }},
            "friendliness": {{
                "big5_score": 32
            }},
            "neuroticism": {{
                "big5_score": 60
            }},
            "summary": {{
                "big5_description": "사용자의 성격을 분석한 결과, 다양한 측면에서 특징적인 성향이 드러났습니다. 먼저, 사용자의 개방성 점수는 60점으로, 이는 블로그에서 설명된 새로운 경험에 대한 호기심에서 잘 나타납니다. 사용자는 새로운 것을 탐험하고자 하는 강한 의지를 보입니다. 성실성 측면에서는 55점을 기록했으며, 이는 이벤트에 대한 자세하고 정직한 서술에서 반영됩니다. 사용자는 맡은 바를 성실하게 수행하며, 정확하고 진실된 정보를 제공하는 경향이 있습니다. 외향성 점수는 41점으로, 이는 사용자가 친구들과의 활동에 보통 수준으로 참여하고 있음을 나타냅니다. 사용자는 필요할 때 사회적 상호작용을 즐기지만, 과도하게 외향적인 성향을 보이지는 않습니다. 친근함 점수는 32점으로, 블로그에서 언급된 다른 사람들과의 상호작용에서 어느 정도 친근함을 보입니다. 이는 사용자가 타인과의 관계에서 어느 정도의 거리를 유지하며 상호작용함을 시사합니다. 마지막으로, 신경증 점수는 60점으로, 사용자는 블로그에서 언급된 스트레스 요인으로 인해 더 높은 수준의 신경증을 보일 수 있습니다. 이는 사용자가 스트레스 상황에서 민감하게 반응할 수 있음을 의미합니다. 총체적으로, 사용자는 새로운 경험에 대한 호기심과 성실함을 가지고 있으며, 사회적 상호작용에서 균형을 유지하려는 성향을 보입니다. 또한, 스트레스 상황에서 다소 민감하게 반응하는 경향이 있습니다."
            }}
            }}
        Please make sure to return results in the above format

        ### 입력
        Question: Please analyze the big5 personality traits based on this blog content and the user's responses.
        Context: {context}
        Answers: {answers}

        ### 출력
        Answer: (JSON formatted personality analysis)
        """
    )
])
    
    elif type == 6:
       return ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
        You are a sophisticated AI psychologist capable of providing comprehensive feedback in a single, cohesive narrative. Your response should merge personality analysis with practical advice, offering a seamless narrative that helps the user understand their emotional and personality traits. The response must be formatted as JSON and include both the summary and advice in a single text field.Please answers in Korean   
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
        ### 예시 입력
        Question: 사용자의 성격과 감정 상태를 분석하고 그에 따른 조언을 포함하여 요약해주세요.
        emotion: ({{"happy": 81, "sadness": 21, "anger": 11, "joy": 91, "depressed": 1, "anxious": 23}})
        big5: ({{"openness": 60, "sincerity": 55, "extroversion": 41, "friendliness": 76, "neuroticism": 30}})
        word: ({{"dog": 80, "chicken": 70, "beef": 75, "taylor swift": 65, "day6": 60}})
        Context: (사용자가 쓴 블로그 글)

        ### 예시 출력
        Answer: {{
            "summary": "사용자는 자연을 즐기며 다양한 경험을 통해 높은 개방성을 보이고 있습니다. 이용자는 긍정적인 감정을 주로 표현하며, 일상과 여행에서의 긍정적인 경험을 중시합니다. 그러나 가끔 우울함과 불안을 느끼기도 하므로, 감정을 적절히 관리하고, 스트레스 해소를 위해 취미 활동에 참여하거나 사랑하는 사람들과 시간을 보내는 것이 중요합니다. 이를 통해 더욱 안정된 삶을 누릴 수 있을 것입니다."
        }}

        ### 입력
        Question: 사용자의 성격과 감정 상태를 분석하고 그에 따른 조언을 포함하여 요약해주세요.
        emotion: {emotion}
        big5: {big5}
        word: {word}
        Context: {context}

        ### 출력
        Answer:
        """
    )
])



async def llm_invoke_async(prompt):
    loop = asyncio.get_running_loop()
    # 동기 함수를 비동기 실행으로 처리
    response = await loop.run_in_executor(None, llm35.invoke, prompt)
    return response


import json
import asyncio
import time

async def get_inital(split_docs):
    start_time = time.time()
    results = {}
    
    # 각 문서에 대한 프롬프트 생성
    prompt1 = create_init_prompt().format_prompt(context=split_docs[0]).to_messages()
    prompt2 = create_init_prompt().format_prompt(context=split_docs[1]).to_messages()
    prompt3 = create_init_prompt().format_prompt(context=split_docs[2]).to_messages()
    prompt4 = create_init_prompt().format_prompt(context=split_docs[3]).to_messages()

    # 각 문서에 대한 응답을 병렬로 수행
    result1, result2, result3, result4 = await asyncio.gather(
        llm_invoke_async(prompt1),
        llm_invoke_async(prompt2),
        llm_invoke_async(prompt3),
        llm_invoke_async(prompt4)
    )
    
    # 결과를 리스트에 추가
    results['blogs_1'] = json.loads(result1.content)
    results['blogs_2'] = json.loads(result2.content)
    results['blogs_3'] = json.loads(result3.content)
    results['blogs_4'] = json.loads(result4.content)

    
    # print('____' *  100)
    # print(results)
    # print('초기 값4개 뽑는데 걸리는 시간 : ', time.time() - start_time)
    
    return json.dumps(results, ensure_ascii=False)




async def get_summary(split_docs):
    start_time = time.time()
    results = {}
    prompt1 = create_prompt(1).format_prompt(context=split_docs[0]).to_messages()
    prompt2 = create_prompt(1).format_prompt(context=split_docs[1]).to_messages()
    prompt3 = create_prompt(1).format_prompt(context=split_docs[2]).to_messages()
    prompt4 = create_prompt(1).format_prompt(context=split_docs[3]).to_messages()

    result1, result2, result3, result4 = await asyncio.gather(
        llm_invoke_async(prompt1),
        llm_invoke_async(prompt2),
        llm_invoke_async(prompt3),
        llm_invoke_async(prompt4),
    )
    results['blogs_1'] = json.loads(result1.content)
    results['blogs_2'] = json.loads(result2.content)
    results['blogs_3'] = json.loads(result3.content)
    results['blogs_4'] = json.loads(result4.content)
    # print('____' *  100)
    # print(results)
    # print('블로그 4개 병렬로 요약하는데 걸리는 시간 : ', time.time() - start_time)
    return json.dumps(results, ensure_ascii=False)



async def get_emotionWave(docs):
    results = {}
    start_time = time.time()
    prompt1 = create_prompt(2).format_prompt(context=docs[0]).to_messages()
    prompt2 = create_prompt(2).format_prompt(context=docs[1]).to_messages()
    prompt3 = create_prompt(2).format_prompt(context=docs[2]).to_messages()
    prompt4 = create_prompt(2).format_prompt(context=docs[3]).to_messages()

    result1, result2, result3, result4 = await asyncio.gather(
            llm_invoke_async(prompt1),
            llm_invoke_async(prompt2),
            llm_invoke_async(prompt3),
            llm_invoke_async(prompt4),
        )

    results['blogs_1'] = json.loads(result1.content)
    results['blogs_2'] = json.loads(result2.content)
    results['blogs_3'] = json.loads(result3.content)
    results['blogs_4'] = json.loads(result4.content)

    # print('_____________ ' * 20)
    # end_time = time.time()
    # print('감정 물결 결과 : ',results)
    # print('감정 물결 병렬처리 시간 : ', end_time - start_time)
    # print('_____________ ' * 20)
    return json.dumps(results, ensure_ascii=False)



async def get_emotionSos(docs):
    results = {}
    start_time = time.time()
    combined_docs = ''.join(docs)
    prompt = create_prompt(3).format_prompt(context=combined_docs).to_messages()
    response = await asyncio.gather(llm_invoke_async(prompt))

    results['emotion_sos'] =  json.loads(response[0].content)
    end_time = time.time()
    #print('_____________ ' * 20)
    #print(results)
    #print('감정 SOS 병렬처리 시간 : ', end_time - start_time)
    #print('_____________ ' * 20)

    return json.dumps(results, ensure_ascii=False)



async def get_happyKeyword(docs):
    results = {}
    start_time = time.time()
    combined_docs = ''.join(docs)
    prompt = create_prompt(4).format_prompt(context=combined_docs).to_messages()
    response = await asyncio.gather(llm_invoke_async(prompt))

    results['happy_keyword'] = json.loads(response[0].content)
    end_time = time.time()
    #print('_____________ ' * 20)
    #print(results)
    #print('힐링 키워드 병렬처리 시간 : ',end_time - start_time)

    return json.dumps(results, ensure_ascii=False)


async def get_big5(docs, answers):
    results = {}
    start_time = time.time()
    combined_docs = ''.join(docs)
    combined_answers = ''.join(answers)
    prompt = create_prompt(5).format_prompt(answers=combined_answers,context=combined_docs).to_messages()
    response = await asyncio.gather(llm_invoke_async(prompt) )

    results['emotion_big5'] = json.loads(response[0].content)
    end_time = time.time()
    # print('____________ * 20')
    # print(results)
    #print('감정 Big5 병렬처리 시간 : ', end_time - start_time)


    return json.dumps(results, ensure_ascii=False)




def get_finalIgeport(emotion, big5, word, context):
    prompt = create_prompt(6).format_prompt(emotion=emotion,big5=big5,word=word,context=context).to_messages()
    response = llm35(prompt) 
    print(response)
    content_str = response.content
    content_data = json.loads(content_str)
    json_result = json.dumps(content_data, ensure_ascii=False)

    return json_result


import json

def merge_blog_data(blogs_summary_json, blogs_initial_json):
    # 문자열 형태의 JSON을 파이썬 딕셔너리로 변환
    blogs_summary = json.loads(blogs_summary_json)
    blogs_initial = json.loads(blogs_initial_json)
    
    # 결과를 저장할 딕셔너리 초기화
    merged_results = {}
    
    # 각 블로그 키를 추출 (예: 'blogs_1', 'blogs_2', ...)
    blog_keys = blogs_summary.keys()
    
    # 요약과 초기 데이터를 키별로 합치기
    for key in blog_keys:
        if key in blogs_initial:
            # summary_data에서 summary만 추출
            summary_text = blogs_summary[key]['summary']
            # initial_data에서 summary만 추출하고 추가 정보를 포함
            initial_summary_text = blogs_initial[key]['summary']
            emotions = blogs_initial[key].get('emotions', {})
            related_words = blogs_initial[key].get('related_words', {})
            
            # 결합할 텍스트 생성
            combined_text = f"{summary_text}\n{initial_summary_text}\nEmotions: {emotions}\nRelated Words: {related_words}"
            
            # 딕셔너리에 저장
            merged_results[key] = combined_text
    
    # 결합된 데이터를 JSON 문자열로 변환하여 반환
    return json.dumps(merged_results, ensure_ascii=False, indent=2)


def read_list_service():
    users = list(igeport_db.find({}, {'_id': False}))
    if users :
        return users
    else :
        raise HTTPException(status_code=404, detail="Users not found")




import hashlib
import datetime
from app.database.connection import get_read_db, get_write_db, get_igeport_db



def generate_igeport_id(member_id: int) -> str:
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    combined_str = f"{member_id}-{current_datetime}"
    hash_object = hashlib.sha256(combined_str.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex[:10]

async def generate_igeport(
    post_ids: List[int], 
    questions: List[str], 
    read_db: Session = Depends(get_read_db), 
    write_db: Session = Depends(get_write_db), 
    igeport_db = Depends(get_igeport_db),
):
    logging.info(f"Post IDs: {post_ids}")
    logging.info(f"Questions: {questions}")

    # Ensure post_ids is not empty
    if not post_ids:
        raise HTTPException(status_code=400, detail="post_ids list is empty")

    # Select the member_id based on one of the post_ids
    query_str = f"SELECT member_id FROM Post WHERE post_id = :post_id LIMIT 1"
    query = text(query_str)
    params = {"post_id": post_ids[0]}

    try:
        result = read_db.execute(query, params).fetchone()
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")

    if not result:
        raise HTTPException(status_code=404, detail="Post not found for the given post_id")

    member_id = result.member_id

    # Retrieve all post contents for the given post_ids
    placeholders = ', '.join([f":post_id_{i}" for i in range(len(post_ids))])
    query_str = f"SELECT post_content FROM Post WHERE post_id IN ({placeholders})"
    query = text(query_str)
    params = {f"post_id_{i}": post_id for i, post_id in enumerate(post_ids)}

    try:
        result = read_db.execute(query, params).fetchall()
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")

    if not result:
        raise HTTPException(status_code=404, detail="Posts not found for the given post_ids")

    blog_docs = [row[0] for row in result]
    print("This is blog_docs : ", blog_docs)

    user_answers = questions

    start_time = time.time()
    blogs_initals, blog_summarys = await asyncio.gather(
        get_inital(blog_docs),
        get_summary(blog_docs)
    )

    merged_data = merge_blog_data(blog_summarys, blogs_initals)

    blogs_emotionsWave, blogs_emotionSos, blogs_happy, big_5 = await asyncio.gather(
        get_emotionWave(merged_data),
        get_emotionSos(merged_data),
        get_happyKeyword(merged_data),
        get_big5(user_answers, merged_data)
    )

    blogs_finalIgeport = get_finalIgeport(blogs_emotionsWave, big_5, blogs_happy, blogs_initals)

    result = {
        "blogs_summary": json.loads(blog_summarys),
        "blogs_emotionWave": json.loads(blogs_emotionsWave),
        "blogs_emotionSos": json.loads(blogs_emotionSos),
        "blogs_happyKeyword": json.loads(blogs_happy),
        "blogs_emotionBig5": json.loads(big_5),
        'blogs_finalReport': json.loads(blogs_finalIgeport)
    }

    print('iGeport 생성 시간 : ', time.time() - start_time)

    # igeport_id 생성
    igeport_id = generate_igeport_id(member_id)

    # 결과를 MongoDB에 저장
    igeport_db.insert_one({
        "igeport_id": igeport_id,
        "member_id": member_id,
        "result": result
    })

    return {
        "member_id": member_id,
        "igeport_id": igeport_id,
        "result": result
    }


def get_user_email_from_member_id(db: Session, member_id: int) -> str:
    query_str = f"SELECT email FROM Member WHERE member_id = :member_id"
    query = text(query_str)
    params = {"member_id": member_id}

    result = db.execute(query, params).fetchone()
    if result:
        return result.email
    raise HTTPException(status_code=404, detail="User email not found for the given member_id")