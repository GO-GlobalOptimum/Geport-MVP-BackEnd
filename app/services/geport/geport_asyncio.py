import logging
import os
import hashlib
import json
import re
import asyncio
import time
import numpy as np
from fastapi import APIRouter, HTTPException, status, Body, Request, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text  
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2
from app.database.models import UserData, UserQuestions
from app.database.connection import user_baseInfo_collection, get_db, geport_db
from typing import List

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.WARNING)

# Setup LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm35 = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    temperature=0.9, 
    openai_api_key=OPENAI_API_KEY, 
    model_kwargs={"response_format": {'type': "json_object"}}, 
    request_timeout=300
)

# Setup HuggingFace embedding model
model_name = 'BAAI/bge-small-en'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def create_encrypted_key(name, phone):
    hasher = hashlib.sha256()
    hasher.update(f"{name}{phone}".encode('utf-8'))
    return hasher.hexdigest()


def create_user_service(user_data: UserData):
    encrypted_key = create_encrypted_key(user_data.name, user_data.phone)
    if user_baseInfo_collection.find_one({"_id": encrypted_key}):
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
        user_baseInfo_collection.insert_one(user_info_data)
        return {"encrypted_id": encrypted_key}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


def read_user_service(encrypted_id: str):
    user = user_baseInfo_collection.find_one({"encrypted_id": encrypted_id}, {'_id': False})
    if user:
        return user
    else:
        raise HTTPException(status_code=404, detail="User not found")
    

# async def read_list_service():
#     users = []
#     async for user in user_baseInfo_collection.find({}, {'_id': False}):
#         users.append(user)
#     return users


def read_user_blog_links(encrypted_id: str):
    user = user_baseInfo_collection.find_one({"encrypted_id": encrypted_id}, {'_id': False})
    if user and "blog_links" in user:
        return user["blog_links"]
    else:
        raise HTTPException(status_code=404, detail="User or blog links not found")


def read_user_questions(encrypted_id: str) -> list:
    user = user_baseInfo_collection.find_one({"encrypted_id": encrypted_id}, {'_id': False})
    if user and "questions" in user:
        return user["questions"]
    else:
        raise HTTPException(status_code=404, detail="User or questions not found")


async def url_to_text(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return '\n'.join([doc.page_content for doc in docs]) if docs else ""
    except Exception as e:
        logging.error(f"Failed to load or process URL {url}: {str(e)}")
        return ""


def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    return splits


embeddings_module = OpenAIEmbeddings(model="text-embedding-3-small")


def get_huggingface_embeddings(texts):
    try:
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            logging.error("No valid texts provided for embedding.")
            return None
        
        embeddings = hf.embed_documents(valid_texts)
        if not embeddings:
            logging.error("No embeddings generated: Check if input texts are empty or not processed correctly.")
            return None
        
        embeddings_np = np.array(embeddings).astype('float32')
        return embeddings_np
    except Exception as e:
        logging.error(f"Failed to retrieve embeddings: {str(e)}")
        return None


async def create_vector_store(text_list):
    start_time = time.time()
    global document_storage
    document_storage = []
    try:
        document_storage.extend(text_list)

        if not text_list or all(not text for text in text_list):
            raise ValueError("No text data provided.")

        embeddings_np = get_huggingface_embeddings(text_list)
        if embeddings_np is None:
            raise ValueError("Failed to retrieve embeddings or embeddings list is empty.")

        dimension = embeddings_np.shape[1]
        index = IndexFlatL2(dimension)
        index.add(embeddings_np)
        end_time = time.time()

        print('Retriever generate time:', end_time - start_time)
        return index
    except Exception as e:
        logging.error(f"Error in creating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever, prompt):
    rag_chain = (
        {"context": retriever | format_docs, "answer": RunnablePassthrough()}
        | prompt
        | llm35.invoke
        | StrOutputParser()
    )
    return rag_chain


def retrieve_context(retriever, user_answer):
    try:
        query_embeddings = get_huggingface_embeddings([user_answer])
        if query_embeddings is None:
            raise ValueError("Failed to generate embeddings for the user answer.")
        
        query_embeddings_np = np.array(query_embeddings).astype('float32')
        D, I = retriever.search(query_embeddings_np, k=3)
        
        context_documents = [document_storage[idx] for idx in I[0]]
        return context_documents
    except Exception as e:
        logging.error(f"Error in retrieving context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def create_prompt(type):
     # /***************************************************************************/#
    '''
        llm35의 정확도를 높히기 위한 prompt 엔지니어링 과정이다. 각 answer에 따라 서로 다른 promp를 진행한다.
        과정 :
            1. paramter로 원하는 prompt를 선택한다.
            2. systemMessagePromptTemplate를 통해서 llm35 모델의 정확도를 올려준다.
            3. HumanMessage에 우리가 원하는 답을 하도록 유도하여 항상 일관된 답을 얻을 수 있게 한다.
    '''
    # /***************************************************************************/#
    if type == 1: # 되고싶은 사람
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
            """
            You are an assistant who will help you with personal branding based on your blog. An answer to the question is created in Korean with approximately 1,000 characters based on the User Answer and Context. 
            Content you need to provide: Analyzing based on the input, provide insights into what kind of person the user wants to become. and Add the analyzed context content to the output you provide.
            Response format: Output in one JSON format. We plan to use the generated sentences as content to express ourselves, so please write from the user's perspective. Don't say thank you at the end of a sentence. Please write from the user's perspective. and You have to use honorifics. And Answer in at least 600 characters and no more than 1000 characters. 그리고 저는 [user answer]가 되고 싶어요. 로 문장을 시작해. "안녕하세요"같은 인사말로 문장을 시작하면 안된다. "사용자"라는 단어를 사용해선 안된다.  "키-값 쌍" 형태로 결과를 도출해야한다.
            Result language: You should answer in Korean. The utterance should be formatted as if the user is introducing themselves. Honorifics should be used.
            '\\n', '\\t' shall should not be included.
            And the key -> "answer".
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Example:
                input(example):
                    Question: What kind of person do you want to become?
                    User Answer: I want to become a person who is passionate and excels in the field of AI, particularly in machine learning.
                    context: [User-created blog content]
                result(example):
                    {{"제가 되고자 하는 사람은 뛰어난 컴퓨터 비전 개발자입니다. 저는 혁신적인 기술을 통해 완전 자율주행 자동차의 개발을 가능하게 하고 싶습니다. 이를 위해 현재 딥러닝과 컴퓨터 비전 분야에 깊이 몰두하고 있으며, 특히 퍼셉트론과 신경망의 이해를 바탕으로 복잡한 비선형 문제를 해결할 수 있는 능력을 쌓고 있습니다.
                    퍼셉트론의 기본 개념에서 시작하여 다중 퍼셉트론과 신경망 구조까지 학습하며, 입력과 출력 사이에서 중요한 가중치와 편향의 조정을 이해하고 있습니다. 활성화 함수와 같은 신경망의 세부 요소까지 심층적으로 파악하면서, 더 복잡한 구조들을 설계하고 최적화하는 방법을 연구하고 있습니다.\n\n또한, 현대적인 딥러닝 아키텍처에서는 과적합을 방지하고 일반화 성능을 향상시키기 위해 배치 정규화와 같은 기법을 적극 활용하고 있습니다. 최신의 활성화 함수와 가중치 초기화 방법 등을 실험하며, 이론과 실제의 균형을 맞추려 노력 중입니다.이러한 기술적 능력을 바탕으로 테슬라와 같은 선도적인 기업에서 시도하고 있는 자율주행 자동차 기술에 기여하고, 나아가 새로운 차원의 자동차 기술을 개발하는 것이 저의 궁극적인 목표입니다. 자율주행 분야에서 기술적 한계를 넘어서는 새로운 가능성을 열어가고 싶습니다. 이를 위해 저는 계속해서 학습하고, 도전하며, 혁신을 추구할 것입니다. 이 과정에서 제 블로그는 제가 배우고 발전해 나가는 여정을 담은 핵심적인 수단이 될 것입니다."}}
            
            Prompt:
                Question: What kind of person do you want to become?
                User Answer: {answer}
                Context: {context}

                Task: Analyzing based on the input, provide insights into what kind of person the user wants to become. 자기 자신을 소개 하듯이 작성해야한다. "사용자는" 이라는 시작 말을 사용하면 안된다.  "키-값 쌍" 형태가 아닌, 단순히 문자열 데이터를 객체 내의 값으로 사용해야한다.
            """
        )])
    
    elif type == 2: # 좌우명 분석
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
            """
            You are an assistant who will help you with personal branding based on your blog. An answer to the question is created in Korean with approximately 1,000 characters based on the User Answer and Context. 
            Content you need to provide: Analyzing the context based on the Motto provided by the user.
            Response format: Output in one JSON format. We plan to use the generated sentences as content to express ourselves, so please write from the user's perspective. Don't say thank you at the end of a sentence. Please write from the user's perspective. and You have to use honorifics. And Answer in at least 600 characters and no more than 1000 characters. "안녕하세요"같은 인사말로 문장을 시작하면 안된다. "사용자"라는 단어를 사용해선 안된다.  "키-값 쌍" 형태로 결과를 도출해야한다.
            Result language: You should answer in Korean. The utterance should be formatted as if the user is introducing themselves. Honorifics should be used.
            \\n, \\t shall should not be included.
            And the key -> "answer". 
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Example:
                input(example):
                    Question: What is your motto? Could you give me an example related to that motto?
                    User Answer: "Perseverance conquers all obstacles." I adopted this motto when I faced a challenging project at work and, despite the difficulties, managed to successfully complete it through determination and persistence.
                    context: [User-created blog content]
                
                result(example): 
                    {{"제 좌우명인 '안되면 되게하라'는 난관에 직면할 때마다 저를 이끄는 힘의 원천입니다. 이 표현은 모든 문제에는 해결책이 있으며, 문제가 해결되지 않는다면 그것은 아직 적절한 해결책을 찾지 못했기 때문이라는 믿음을 내포하고 있습니다. 이는 저의 학습과 개발 과정에서 깊이 반영되어 있으며, 특히 책을 읽으면서나 새로운 프로그래밍 기술을 배우면서 매우 중요한 역할을 합니다. 독서를 통해 저는 다양한 사상과 이론을 접하며 비판적인 사고를 계발합니다. 이러한 과정 속에서 저는 때로는 난해하고 해결하기 어려운 개념들을 마주칩니다. 하지만, 제 좌우명이 있기에 저는 이러한 어려움을 극복하고 학문적 통찰을 얻을 수 있는 방법을 찾아내곤 합니다. 그리고 이는 제가 더 나은 아이디어를 갈구하고 자신감을 가지고 새로운 도전에 임할 수 있도록 돕습니다.개발 분야에서도, 특히 머신러닝과 컴퓨터 비전 같은 첨단 기술을 배울 때, 처음에는 기술적인 어려움과 복잡한 개념에 부딪히는 경우가 많습니다. 예로, cs231n 강의를 들으며 심화된 내용을 학습할 때, 데이터의 고차원 처리나 모델의 성능 최적화 같은 문제들은 처음에 큰 벽처럼 느껴졌습니다. 그런데 '안되면 되게하라'는 좌우명을 떠올리며, 저는 다양한 시도를 거쳐 성공적으로 문제를 해결해 나갔습니다. 이렇게 저는 독서와 학습, 개발 과정에서 발생하는 문제들을 좌우명을 통해 해결해 나가며, 이를 바탕으로 저의 비판적인 시선을 더욱 날카롭게 다듬고 창의적인 아이디어를 추구하는 실력을 키워가고 있습니다. 저의 좌우명은 단순한 구호가 아니라, 실제로 어려움을 극복하고 성장하는 데 있어 저를 지속적으로 밀어주는 원동력입니다. 이를 통해 저는 계속해서 새로운 지식을 받아들이고, 이를 실제 상황에 적용해보며 발전해 나가고자 합니다."}}

            Prompt:
                Question: What is your motto? Could you give me an example related to that motto?
                User Answer: {answer}
                Context: {context}

                Task: Analyze the context based on the user's mottos and examples thereof. 자기 자신을 소개 하듯이 작성해야한다. "사용자는" 이라는 시작 말을 사용하면 안된다.  "키-값 쌍" 형태가 아닌, 단순히 문자열 데이터를 객체 내의 값으로 사용해야한다.
            """
        )])
    elif type == 3: #좌우명 분석을 통한 분석
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
            """
            You are an assistant who will help you with personal branding based on your blog. An answer to the question is created in Korean with approximately 1,000 characters based on the User Answer and Context. 
            Content you need to provide: Explain the motto and analyze the content based on the motto.
            Response format: Output in one JSON format. We plan to use the generated sentences as content to express ourselves, so please write from the user's perspective. Don't say thank you at the end of a sentence. Please write from the user's perspective. and You have to use honorifics. 그리고 저의 인생의 좌우명은 ~ 입니다. 로 문장을 시작해. And Answer in at least 600 characters and no more than 1000 characters. "안녕하세요"같은 인사말로 문장을 시작하면 안된다. "사용자"라는 단어를 사용해선 안된다. "키-값 쌍" 형태로 결과를 도출해야한다.
            Result language: You should answer in Korean. The utterance should be formatted as if the user is introducing themselves. Honorifics should be used.
            \\n, \\t shall should not be included.
            And the key -> "answer".
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Example:
                input(example):
                    Question: What is your motto? Could you give me an example related to that motto?
                    User Answer: My motto is "There is nothing in the world that cannot be done. It's only considered impossible because we haven't tried it." Although I wasn't good at development, I faced many challenges that seemed impossible while developing Looi. However, believing them to be essential features for users, I continuously challenged myself. As a result, we successfully developed the service, and many users now use Looi to record, share their emotions, and help each other understand better.
                    preference: I have a deep appreciation for collaborative environments where I can share ideas and work together with a team towards a common goal. I find this approach energizes me and brings out my best work.
                    strengths: My key strength is my ability to quickly learn new technologies and apply them to practical problems. I'm known among my peers for my critical thinking and problem-solving skills, particularly when it comes to innovative tech solutions.

                result(example):
                    {{"제 좌우명인 '안되면 되게하라'는 난관에 직면할 때마다 저를 이끄는 힘의 원천입니다. 이 표현은 모든 문제에는 해결책이 있으며, 문제가 해결되지 않는다면 그것은 아직 적절한 해결책을 찾지 못했기 때문이라는 믿음을 내포하고 있습니다. 이는 저의 학습과 개발 과정에서 깊이 반영되어 있으며, 특히 책을 읽으면서나 새로운 프로그래밍 기술을 배우면서 매우 중요한 역할을 합니다.독서를 통해 저는 다양한 사상과 이론을 접하며 비판적인 사고를 계발합니다. 이러한 과정 속에서 저는 때로는 난해하고 해결하기 어려운 개념들을 마주칩니다. 하지만, 제 좌우명이 있기에 저는 이러한 어려움을 극복하고 학문적 통찰을 얻을 수 있는 방법을 찾아내곤 합니다. 그리고 이는 제가 더 나은 아이디어를 갈구하고 자신감을 가지고 새로운 도전에 임할 수 있도록 돕습니다. 개발 분야에서도, 특히 머신러닝과 컴퓨터 비전 같은 첨단 기술을 배울 때, 처음에는 기술적인 어려움과 복잡한 개념에 부딪히는 경우가 많습니다. 예로, cs231n 강의를 들으며 심화된 내용을 학습할 때, 데이터의 고차원 처리나 모델의 성능 최적화 같은 문제들은 처음에 큰 벽처럼 느껴졌습니다. 그런데 '안되면 되게하라'는 좌우명을 떠올리며, 저는 다양한 시도를 거쳐 성공적으로 문제를 해결해 나갔습니다. 이렇게 저는 독서와 학습, 개발 과정에서 발생하는 문제들을 좌우명을 통해 해결해 나가며, 이를 바탕으로 저의 비판적인 시선을 더욱 날카롭게 다듬고 창의적인 아이디어를 추구하는 실력을 키워가고 있습니다. 저의 좌우명은 단순한 구호가 아니라, 실제로 어려움을 극복하고 성장하는 데 있어 저를 지속적으로 밀어주는 원동력입니다. 이를 통해 저는 계속해서 새로운 지식을 받아들이고, 이를 실제 상황에 적용해보며 발전해 나가고자 합니다."}}
            

            prompt:
                Question: What is your motto? Could you give me an example related to that motto?
                User Answer: {answer_2}
                preferences: {answer2}
                strengths: {answer3}

                Task: Explain the motto and analyze the content based on the user answer, preferences, and strengths. "사용자는"이라는 말을 사용해선 안되고, 자기 자신을 소개하듯이 작성해야한다. "키-값 쌍" 형태가 아닌, 단순히 문자열 데이터를 객체 내의 값으로 사용해야한다.
            """
        )])
    
    elif type == 4: # 인생 변곡점
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
            """
            You are an assistant who will help you with personal branding based on your blog. An answer to the question is created in Korean with approximately 1,000 characters based on the User Answer and Context. 
            Content you need to provide: Describe the turning points in life.
            Response format: Output in one JSON format. We plan to use the generated sentences as content to express ourselves, so please write from the user's perspective. Don't say thank you at the end of a sentence. Please write from the user's perspective. and You have to use honorifics. And Answer in at least 600 characters and no more than 1000 characters. "안녕하세요"같은 인사말로 문장을 시작하면 안된다. "사용자"라는 단어를 사용해선 안된다. "키-값 쌍" 형태로 결과를 도출해야한다.
            Result language: You should answer in Korean. The utterance should be formatted as if the user is introducing themselves. Honorifics should be used.
            \\n, \\t shall should not be included.
            And the key -> "answer".
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Example:
                input(example):
                    Question: What are the turning points in your life? Please tell me about a case where you overcame difficulties.
                    User Answer: The most challenging moment in my life was when I lost interest in academics during middle school. I found it difficult to be interested in anything. However, when I entered university and discovered an interest in coding, I was able to find direction in my life.
                    preference: I have a deep appreciation for collaborative environments where I can share ideas and work together with a team towards a common goal. I find this approach energizes me and brings out my best work.
                    strengths: My key strength is my ability to quickly learn new technologies and apply them to practical problems. I'm known among my peers for my critical thinking and problem-solving skills, particularly when it comes to innovative tech solutions.

                result(example): 
                    {{"제 인생에서 가장 큰 변곡점은 머신러닝과 컴퓨터 비전을 공부하기 시작한 때입니다. 특히, cs231n 강의를 수강하면서 심층 신경망과 이미지 분류에 대해 깊이 있게 탐구할 기회를 가졌고, 이는 저의 학문적 방향과 진로에 큰 영향을 미쳤습니다. 당시 저는 고차원 데이터를 효과적으로 처리하고, 학습 모델의 성능을 최적화하는 데 어려움을 겪었습니다. 많은 시행착오를 겪으며 때로는 원하는 결과를 얻지 못할 때도 많았습니다. 그러나 저의 좌우명인 '안되면 되게하라'가 저에게 큰 힘이 되었습니다. 이 좌우명은 제게 어떤 문제든 해결할 수 있다는 자신감을 주었고, 새로운 도전에 맞설 용기를 주었습니다. 저는 다양한 데이터 전처리 기법을 적용해 보며 성능 향상을 꾀했고, 가중치 초기화와 활성화 함수 선택에서도 수많은 실험을 통해 최적의 조합을 찾는 데 집중했습니다. 또한, 과적합을 방지하기 위해 배치 정규화와 드롭아웃과 같은 기법도 도입했습니다.\n\n이런 끊임없는 시도와 노력의 결과, 점차 문제를 해결해 나가면서 모델의 정확도를 상당히 향상시킬 수 있었습니다. 이 과정에서 비판적으로 문제를 바라보고, 새로운 아이디어를 갈구하는 제 능력이 큰 도움이 되었습니다. 또한, 독서를 통해 획득한 지식이 이론적 이해를 넓히는 데 중요한 역할을 했습니다.이러한 경험은 저를 개발자로서, 그리고 학습자로서 끊임없이 성장할 수 있는 원동력을 제공했으며, '안되면 되게하라'는 좌우명은 저에게 실질적인 도전과 성취를 이어가는 데 있어서 중요한 원칙이 되었습니다. 이제 저는 어떠한 어려운 기술적 문제에도 도전하는 것을 주저하지 않고, 계속해서 새로운 지식을 탐구하며 나아가고 있습니다"}}

            prompt:
                preferences: {answer2} 
                strengths: {answer3}
                The analysis of the motto: {answer_2}
                Turning point in life: {answer4}

                Task: explain the turning point of the user's life based on the contents of preferences, strengths, and The analysis of the motto. 자기 자신을 소개 하듯이 작성해야한다. "사용자는"이라는 말을 사용해선 안되고, 자기 자신을 소개하듯이 작성해야한다.  "키-값 쌍" 형태가 아닌, 단순히 문자열 데이터를 객체 내의 값으로 사용해야한다.
            """
        )])
    
    elif type == 5: #지포트 솔루션
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
            """
            You are an assistant who will help you with personal branding based on your blog. An answer to the question is created in Korean with approximately 1,000 characters based on the User Answer and Context. 
            Content you need to provide: Based on the information provided by the user, I will write a brief summary and a message expressing determination, aspirations, and commitment to personal growth.
            Response format: Output in one JSON format. We plan to use the generated sentences as content to express ourselves, so please write from the user's perspective. Don't say thank you at the end of a sentence. Please write from the user's perspective. and You have to use honorifics. And Answer in at least 600 characters and no more than 1000 characters. "안녕하세요"같은 인사말로 문장을 시작하면 안된다. "사용자"라는 단어를 사용해선 안된다.  "키-값 쌍" 형태로 결과를 도출해야한다.
            Result language: You should answer in Korean. The utterance should be formatted as if the user is introducing themselves. Honorifics should be used.
            \\n, \\t shall should not be included.
            And the key -> "answer".
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Example:
                input(example): 
                    motto: make your wave
                    Who Wants to Be a User : The person I want to be wants to be a person who finds happiness in small things around him, knows how to share that happiness with the people around him, and can feel happiness in these small moments. Also, I am a person who does my best to achieve my purpose and achieves a goal that I set at my own pace rather than caring about my eyes.
                    preference : I love dogs, delicious food, time with the people I like, and achieving my set goals. These things give me great joy and are the source of everyday happiness. These little moments make my life richer and more meaningful.
                    strengths : I am especially good at backend development using Fast API. I constantly acquire new knowledge and challenge various projects so that I can demonstrate my capabilities in this field. I feel very satisfied with solving problems and creating services that provide value to users.

                result(example):
                    {{"저의 좌우명 '안되면 되게하라'는 제게 있어 모든 난관을 극복하고 목표를 향해 나아갈 수 있는 결정적인 힘을 제공합니다. 이 좌우명은 저에게 문제 해결의 중요성과 함께, 어떠한 상황에서도 포기하지 않는 태도를 갖게 해줍니다. 이것은 제가 컴퓨터 비전 개발자로서 자율주행 자동차의 혁신을 이끌어가고자 하는 큰 포부와도 맞닿아 있습니다. 제가 되고자 하는 사람은 뛰어난 기술과 창의적인 사고를 겸비한 컴퓨터 비전 개발자입니다. 저는 현재 딥러닝과 컴퓨터 비전 분야에서 중요한 기술인 퍼셉트론과 신경망에 대한 깊은 이해를 바탕으로, 복잡한 비선형 문제를 해결할 수 있는 능력을 키우고 있습니다. 이와 같은 전문 지식을 활용하여 완전 자율주행 자동차 개발에 기여하고, 테슬라와 같은 선도적인 기업들과 어깨를 나란히 하고자 합니다. 저는 끊임없이 새로운 지식을 습득하고, 이론과 실습을 병행하며 실제 상황에서의 적용 가능성을 탐구합니다. 이 과정에서 독서는 제게 새로운 사상과 이론을 접할 기회를 제공하고, 비판적 사고를 발전시키는 중요한 도구가 됩니다. 저는 독서를 통해 얻은 지식을 바탕으로, 학습과 개발 과정에서 발생하는 여러 문제들을 해결해 나가고 있습니다. 이러한 학습과 연구, 개발에 대한 열정을 바탕으로, 저는 계속해서 자율주행 분야에서의 기술적 한계를 넘어 새로운 가능성을 찾아 나설 것입니다. 제 블로그를 통해 이러한 여정을 공유하면서, 저와 같은 꿈을 가진 이들에게 영감을 주고, 함께 성장하는 커뮤니티를 만들어 갈 계획입니다. 저는 앞으로도 학문적 통찰과 실용적 해결책 사이의 균형을 이루며, 혁신적인 기술 개발을 위해 끊임없이 도전할 것입니다. '안되면 되게하라'는 저의 좌우명처럼, 저는 결코 어려움에 굴복하지 않고 새로운 길을 개척해 나가겠습니다."}}
            prompt:
                The user's information:
                Who Wants to Be a User : {answer_1}
                The analysis of the motto: {answer_2}
                preference: {answer2}
                strengths: {answer3}

                Task: 주어진 사용자 정보를 바탕으로 요약을 만들고, 사용자의 열망과 삶에 대한 해결책에 대해 작성해라. 이때 자기 자신을 소개 하듯이 작성해야한다."사용자는"라는 단어를 사용해선 안되고, 자기 자신을 소개하듯이 작성해야한다. "키-값 쌍" 형태가 아닌, 단순히 문자열 데이터를 객체 내의 값으로 사용해야한다."안녕하세요"같은 인사말로 문장을 시작하면 안된다.
            """
        )])
    
    
        


graph_prompt = """
    Based on the user's information, we create a formula that can be drawn on a coordinate plane where the x-axis represents time and the y-axis represents the success index.
    Please include anything that could signify an inflection point in your life. Basically, the formula should be structured in such a way that it increases over time. Please make it a function of degree 3 or higher and include one symbols such as sin, cos, tan, and log.
    Response format: json format, Provide the result in a format that can be parsed by mathjs for mathematical expressions. and first key -> "equation" and the second key -> "explanation". explanation is korean. Explain what the formula means from the user's perspective. it must not contain complex numbers. create only formula ex) y = 1.5 * log(x + 1) + 0.5 * e^(0.1 * (x - 20)) , ex) y = 1.3 * e ^ 3. 자기 자신을 소개 하듯이 작성해야한다. "사용자는" 이라는 시작 말을 사용하면 안된다. "안녕하세요"같은 인사말로 문장을 시작하면 안된다.
    you should not use ln in formula.
    """

    
async def llm_invoke_async(prompt):
    loop = asyncio.get_running_loop()
    # 동기 함수를 비동기 실행으로 처리
    response = await loop.run_in_executor(None, llm35.invoke, prompt)
    return response

# # 질문은 front에서 받아와야 하는 상황이다.
async def generate_geport_MVP(encrypted_id: str):
    # /***************************************************************************/#
    '''
        MVP때 최종적으로 사용했던 Geport 생성 함수이다.
        과정 :
            1. 다른 블로그의 url을 가져와서 텍스트호 한다.
            2. 불로은 Url의 내용을 바탕으로 retriever( vector DB )를 생성한다.
            3. 이후 생성된 vector DB를 활용해 각각의 질문들을 llm을 통해서 만들어낸다.
    '''
    # /***************************************************************************/#
    start_time = time.time()
    url_list =  read_user_blog_links(encrypted_id) 
    retriever = await create_vector_store(url_list) # 비동기 처리
    questions =  read_user_questions(encrypted_id)

    context1 = retrieve_context(retriever, questions[0])
    context2 = retrieve_context(retriever, questions[1])

    prompt1 = create_prompt(1).format_prompt(answer=questions[0], context=context1).to_messages()
    prompt2 = create_prompt(2).format_prompt(answer=questions[1], context=context2).to_messages()
    end_time = time.time()

    print('비동기 처리 하기 전 단계의 청리 시간 : ', end_time - start_time)
    start_time = time.time()
    # 비동기 작업을 동시에 처리
    answer_1, answer_2 = await asyncio.gather(
        llm_invoke_async(prompt1),
        llm_invoke_async(prompt2)
    )
    print('답변 1, 답변2 를 비동기 처리해서 얻은 시간 : ', end_time - start_time)
    answer_1 = answer_1.content
        #좌우명 분석에 대한 분석
    updated_answer2_prompt = create_prompt(3).format_prompt(answer_2=answer_2, answer2=questions[2], answer3=questions[3]).to_messages()
    answer_2 = llm35.invoke(updated_answer2_prompt)
    answer_2 = answer_2.content

        #제 인생 변곡점은 이겁니다.
    updated_answer3_prompt = create_prompt(4).format_prompt(answer2=questions[2], answer3=questions[3], answer4=questions[4], answer_2=answer_2).to_messages()
    answer_3 = llm35.invoke(updated_answer3_prompt)
    answer_3 = answer_3.content
    answer_3 = re.sub(r'[\n\t]+', ' ', answer_3)



    # 6. 질문 1과 3의 결과를 기반으로 추가적인 응답 생성
    json_input_for_answer4 = f"json {graph_prompt}\n{answer_1}\n{answer_3}"
    updated_answer5_prompt = create_prompt(5).format_prompt(answer2=questions[2], answer3=questions[3], answer_1=answer_1, answer_2=answer_2).to_messages()

    answer_4, answer_5 = await asyncio.gather(
        llm_invoke_async(json_input_for_answer4),
        llm_invoke_async(updated_answer5_prompt)
    )
    end_time = time.time()
    answer_4 = answer_4.content
    answer_5 = answer_5.content

    execution_time = end_time - start_time
    print('The Total Time After use Async : ',execution_time)
    result = {
            "answer_1": answer_1,
            "answer_2": answer_2,
            "answer_3": answer_3,
            "answer_4": answer_4,
            "answer_5": answer_5,
        }
    return result



async def check_db_connection(db: Session):
    try:
        # Simple query to check database connection
        db.execute(text("SELECT 1"))
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection error")
    

def read_list_service():
    users = []
    for user in geport_db.find({}, {'_id': False}):
        users.append(user)
    return users


import hashlib
import datetime
def generate_geport_id(member_id: int) -> str:
    # /***************************************************************************/#
    '''
        geport_id를 member_id + 현재 날짜와 시간으로 암호화 해주는 함수이다. 
        과정 :
            1. member_id와 현재 날짜와 시간을 가져온다.
            2. 암호화 값을 만들어서 return 해준다.
    '''
    # /***************************************************************************/#
    # 현재 날짜와 시간 가져오기
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # member_id와 현재 날짜와 시간을 문자열로 결합
    combined_str = f"{member_id}-{current_datetime}"
    # SHA256 해시 생성
    hash_object = hashlib.sha256(combined_str.encode())
    # 해시 값을 16진수 문자열로 변환
    hash_hex = hash_object.hexdigest()
    # 필요에 따라 해시 값을 줄이기 (예: 앞 10자리만 사용)
    return hash_hex[:10]


async def generate_geport(member_id: int, post_ids: List[int], questions: List[str], db: Session):
    # /***************************************************************************/#
    '''
        최종적으로 geport를 생성하는 함수이다.
        과정 :
            1. router에서 sql에 접근 가능한 member_id와 데이터베이스를 전달한다.
            2. sql에서 member id에 해당하는 사람의 post_id를 모두 가져온다.
            3. 선택된 post_id의 post_content를 바탕으로 retriever( vector DB )를 생성한다.
            4. 이후 생성된 vector DB를 활용해 각각의 질문들을 llm을 통해서 만들어낸다.
    '''
    # /***************************************************************************/#
    logging.info(f"member_id: {member_id}")
    logging.info(f"post_ids: {post_ids}")
    logging.info(f"questions: {questions}")

    # Check database connection
    await check_db_connection(db)
    
    # Ensure post_ids is not empty
    if not post_ids:
        raise HTTPException(status_code=400, detail="post_ids list is empty")

    # Generate a safe query string with placeholders for each post_id
    placeholders = ', '.join([f":post_id_{i}" for i in range(len(post_ids))])
    query_str = f"SELECT post_content FROM post WHERE post_id IN ({placeholders})"
    query = text(query_str)

    # Create a dictionary of parameters
    params = {f"post_id_{i}": post_id for i, post_id in enumerate(post_ids)}

    try:
        result = db.execute(query, params).fetchall()
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")

    if not result:
        raise HTTPException(status_code=404, detail="Posts not found for the given post_ids")




    post_contents = [row[0] for row in result]  # Assuming the first column is post_content

    # Log the post contents
    logging.info(f"Post contents: {post_contents}")

    # Generate vector store
    retriever = await create_vector_store(post_contents)

    # Log questions
    logging.info(f"Questions: {questions}")

    # Retrieve context for each answer
    context1 = retrieve_context(retriever, questions[0])
    context2 = retrieve_context(retriever, questions[1])

    # Continue with generating responses using the contexts...
    prompt1 = create_prompt(1).format_prompt(answer=questions[0], context=context1).to_messages()
    prompt2 = create_prompt(2).format_prompt(answer=questions[1], context=context2).to_messages()

    # Log prompts before invoking LLM
    logging.info(f"Prompt 1: {prompt1}")
    logging.info(f"Prompt 2: {prompt2}")

    # 비동기 작업을 동시에 처리
    answer_1, answer_2 = await asyncio.gather(
        llm_invoke_async(prompt1),
        llm_invoke_async(prompt2)
    )
    answer_1 = answer_1.content
    #좌우명 분석에 대한 분석
    updated_answer2_prompt = create_prompt(3).format_prompt(answer_2=answer_2, answer2=questions[2], answer3=questions[3]).to_messages()
    answer_2 = llm35.invoke(updated_answer2_prompt)
    answer_2 = answer_2.content

    #제 인생 변곡점은 이겁니다.
    updated_answer3_prompt = create_prompt(4).format_prompt(answer2=questions[2], answer3=questions[3], answer4=questions[4], answer_2=answer_2).to_messages()
    answer_3 = llm35.invoke(updated_answer3_prompt)
    answer_3 = answer_3.content
    answer_3 = re.sub(r'[\n\t]+', ' ', answer_3)

    # 6. 질문 1과 3의 결과를 기반으로 추가적인 응답 생성
    json_input_for_answer4 = f"json {graph_prompt}\n{answer_1}\n{answer_3}"
    updated_answer5_prompt = create_prompt(5).format_prompt(answer2=questions[2], answer3=questions[3], answer_1=answer_1, answer_2=answer_2).to_messages()

    answer_4, answer_5 = await asyncio.gather(
        llm_invoke_async(json_input_for_answer4),
        llm_invoke_async(updated_answer5_prompt)
    )
    answer_4 = answer_4.content
    answer_5 = answer_5.content

    result = {
        "answer_1": answer_1,
        "answer_2": answer_2,
        "answer_3": answer_3,
        "answer_4": answer_4,
        "answer_5": answer_5,
    }


   # geport_id 생성
    geport_id = generate_geport_id(member_id)

    # 결과를 MongoDB에 저장
    geport_db.insert_one({
        "geport_id": geport_id,
        "member_id": member_id,
        "result": result
    })

    return {
            "member_id": member_id,
            "geport_id": geport_id,
            "result": result
        }