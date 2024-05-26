import os
from dotenv import load_dotenv
import boto3
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import logging
import json
import requests
import uuid
from io import BytesIO
from PIL import Image


logging.basicConfig(level=logging.WARNING)
env_path = os.path.join(os.path.dirname(__file__), '../../.env')
load_dotenv(dotenv_path=env_path)
# LLM 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, openai_api_key=OPENAI_API_KEY, model_kwargs={"response_format": {'type':"json_object"}}, request_timeout=300)

# 네이버 클라우드 S3 설정
service_name = 's3'
endpoint_url = 'https://kr.object.ncloudstorage.com'
region_name = 'kr-standard'
access_key = os.getenv("NAVER_CLOUD_ACCESS_KEY_ID")
secret_key = os.getenv("NAVER_CLOUD_SECRET_KEY")

if not access_key or not secret_key:
    raise ValueError("Missing access key or secret key for Naver Cloud S3.")

print("Access Key:", access_key)  # 디버깅용 출력
print("Secret Key:", secret_key)  # 디버깅용 출력

# Initialize the boto3 client
s3_client = boto3.client(
    service_name,
    endpoint_url=endpoint_url,
    region_name=region_name,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)

def create_image_prompt():
    return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
        """
        You are the assistant that creates image prompts for DALL-E. 
        The prompts should be detailed and suitable for generating a thumbnail image based on the post content. 
        And make sure the image is not lopsided
        Don't include text in your thumbnail image, and keep it as concise and descriptive as possible.
        The result should always be in English and in JSON format.
        ex) {{ "prompt" : "A vivid digital illustration of a beautiful sunset over the ocean, with vibrant colors and a calm, serene mood." }}
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
        ### Example Input
        Title : Sydney trip
        Context: Today's trip to Sydney was fun. In the morning, I had morning bread and steak, and then I bought a music box at a nearby souvenir shop.
        For lunch, I enjoyed a course meal with wine at a restaurant overlooking the sea. In the evening, I had fun making new friends at a fireworks festival.
        I really enjoyed this trip and would love to come back again.

        ### Example Output :
        {{
            "prompt": "A vibrant digital illustration of a day in Sydney, starting with a morning breakfast of bread and steak, buying a music box at a souvenir shop, having a seaside lunch with wine, and ending with a fireworks festival in the evening."
        }}
        ### Input
        Title : {title}
        Context: {context}

        ### Output
        """
        )])

def get_post_by_id(post_id: int, db: Session):
    try:
        query = text("SELECT title, post_id, post_content FROM post WHERE post_id = :post_id")
        result = db.execute(query, {"post_id": post_id}).fetchone()
        if result is None:
            raise HTTPException(status_code=404, detail="Post not found")
        
        post = dict(result._mapping)
        return post
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 썸네일 이미지 만들기
def generate_thumbnailImage(post_id: int, db: Session):
    result = get_post_by_id(post_id, db)
    title = result['title']
    content = result['post_content']
    prompt1 = create_image_prompt().format_prompt(title=title, context=content).to_messages()

    generate_thumbnail_img_prompt = llm35.invoke(prompt1)

    # JSON 응답에서 prompt 값만 추출
    generate_thumbnail_img_prompt = generate_thumbnail_img_prompt.content
    
    # prompt 값을 JSON으로 파싱
    prompt_json = json.loads(generate_thumbnail_img_prompt.strip())

    # DALL-E API 호출하여 이미지 생성
    try:
        dall_e_response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"prompt": prompt_json['prompt'], "n": 1, "size": "1024x1024"}
        )
        
        if dall_e_response.status_code != 200:
            raise HTTPException(status_code=dall_e_response.status_code, detail="Failed to generate image")

        image_url = dall_e_response.json().get("data")[0].get("url")

        # 이미지를 S3에 업로드
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((512, 512), Image.LANCZOS)

        unique_id = uuid.uuid4()
        destination_blob_name = f"{post_id}.png"
        bucket_name = "geport"

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        s3_client.put_object(
            Bucket=bucket_name,
            Key=destination_blob_name,
            Body=buffer,
            ACL='public-read'
        )

        public_url = f"{endpoint_url}/{bucket_name}/{destination_blob_name}"

        # 생성된 이미지 URL을 post 테이블에 업데이트
        try:
            update_query = text("UPDATE post SET thumbnail_image = :thumbnail_image WHERE post_id = :post_id")
            db.execute(update_query, {"thumbnail_image": public_url, "post_id": post_id})
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))

        return {"thumbnail_image_url": public_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))