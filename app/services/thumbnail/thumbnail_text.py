from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi import HTTPException
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import logging
import json

logging.basicConfig(level=logging.WARNING)
env_path = os.path.join(os.path.dirname(__file__), '../../.env')
load_dotenv(dotenv_path=env_path)

# LLM 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, openai_api_key=OPENAI_API_KEY, model_kwargs={"response_format": {'type':"json_object"}}, request_timeout=300)

def create_prompt():
    return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
        """
        You are the assistant that creates thumbnail posts with the key content from the post blog. 
        Thumbnail posts should be 3 to 4 sentences long, and you can make them relevant. 
        Make the thumbnail something that people will want to read. The result should always be in Korean and in JSON format.
        It's okay to use emojis
        ex) {{ "content" : "thumbnail Content " }}
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
       ### Example Input
        Title : Sydeny trip
        Context: Today's trip to Sydney was fun. In the morning, I had morning bread and steak, and then I bought a music box at a nearby souvenir shop.
        For lunch, I enjoyed a course meal with wine at a restaurant overlooking the sea. In the evening, I had fun making new friends at a fireworks festival.
        I really enjoyed this trip and would love to come back again.

        ### Example Output :
        {{
            "content": "시드니에서의 하루, 아침 스테이크와 음악 상자 쇼핑부터 바다 전망 레스토랑에서의 코스 요리와 불꽃놀이 축제까지! 잊지 못할 추억과 새로운 친구들이 기다리고 있습니다! 🌟🦘🌊"
        }}
        ### Input
        Title : {title}
        Context: {context}

        ### Output
        """
        )])

def get_post_by_id(post_id: int, db: Session):
    try:
        # 특정 필드만 선택하는 SQL 쿼리
        query = text("SELECT title, post_id, post_content FROM post WHERE post_id = :post_id")
        result = db.execute(query, {"post_id": post_id}).fetchone()
        if result is None:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # 결과를 딕셔너리로 변환
        post = dict(result._mapping)
        return post
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 썸내일 내용을 만들어준다.
def generate_thumbnailText(post_id: int, db: Session):
    result = get_post_by_id(post_id, db)
    title = result['title']
    content = result['post_content']
    prompt1 = create_prompt().format_prompt(title=title, context=content).to_messages()

    generate_thumbnail_txt = llm35.invoke(prompt1)

    # JSON 응답에서 content 값만 추출
    generate_thumbnail_txt = generate_thumbnail_txt.content
    
    # content 값을 JSON으로 파싱
    thumbnail_json = json.loads(generate_thumbnail_txt.strip())

    # 썸네일 텍스트를 데이터베이스에 업데이트
    try:
        update_query = text("UPDATE post SET thumbnail_text = :thumbnail_text WHERE post_id = :post_id")
        db.execute(update_query, {"thumbnail_text": thumbnail_json['content'], "post_id": post_id})
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    return thumbnail_json
