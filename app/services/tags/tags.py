# from sqlalchemy.orm import Session
# from sqlalchemy import text
# from fastapi import HTTPException
# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain.prompts import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate
# )
# import logging
# import json

# logging.basicConfig(level=logging.WARNING)
# env_path = os.path.join(os.path.dirname(__file__), '../../.env')
# load_dotenv(dotenv_path=env_path)

# # LLM 설정
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, openai_api_key=OPENAI_API_KEY, model_kwargs={"response_format": {'type':"json_object"}}, request_timeout=300)

# def create_prompt():
#     return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(
#         """
#         You are an assistant who looks at a user's blog and picks out important tags. Tags can be words or short sentences.
#         You pick one tag from the title and four from the body, for a total of five tags.
#         The tags must be in Korean, always provide 5 tags, and use the result in JSON format {{ "tags" : ["tag1", "tag2", "tag3", "tag4", "tag5" ] }} like this
#         """
#     ),
#     HumanMessagePromptTemplate.from_template(
#         """
#        ### Example Input
#         Title : Sydeny trip
#         Context: Today's trip to Sydney was fun. In the morning, I had morning bread and steak, and then I bought a music box at a nearby souvenir shop.
#         For lunch, I enjoyed a course meal with wine at a restaurant overlooking the sea. In the evening, I had fun making new friends at a fireworks festival.
#         I really enjoyed this trip and would love to come back again.

#         ### Example Output :
#         {{
#             "tags": ["sydney", "bread and steak", "restaurant", "sea", "I really enjoyed this trip"]
#         }}
#         ### Input
#         Title : {title}
#         Context: {context}

#         ### Output
#         """
#         )])

# def get_post_by_id(post_id: int, db: Session):
#     try:
#         # 특정 필드만 선택하는 SQL 쿼리
#         query = text("SELECT title, post_id, member_id, post_content FROM Post WHERE post_id = :post_id")
#         result = db.execute(query, {"post_id": post_id}).fetchone()
#         if result is None:
#             raise HTTPException(status_code=404, detail="Post not found")
        
#         # 결과를 딕셔너리로 변환
#         post = dict(result._mapping)
#         return post
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# def generate_tags(post_id: int, db: Session):
#     result = get_post_by_id(post_id, db)
#     title = result['title']
#     content = result['post_content']
#     post_id = result['post_id']
#     member_id = result['member_id']
#     prompt1 = create_prompt().format_prompt(title=title, context=content).to_messages()

#     generated_tags = llm35.invoke(prompt1)

#     # JSON 응답에서 content 값만 추출
#     content = generated_tags.content
    
#     # content 값을 JSON으로 파싱
#     tags_json = json.loads(content.strip())

#     # 태그들을 하나의 문자열로 결합
#     tags_string = ",".join(tags_json['tags'])

#     # tags 데이터베이스에 저장
#     insert_query = text("INSERT INTO Post_tag (post_id, contents, is_user) VALUES (:post_id, :name, :is_user)")
#     db.execute(insert_query, {"post_id": post_id, "name": tags_string, "is_user": False})
    
#     db.commit()

#     return tags_json

