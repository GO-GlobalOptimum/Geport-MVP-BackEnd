# from fastapi.responses import JSONResponse
# from fastapi import HTTPException, status
# import hashlib
# from app.database.models import UserData
# from app.database.connection import user_baseInfo_collection
# import os
# import json
# from dotenv import load_dotenv
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import FAISS
# import logging
# import requests


# logging.basicConfig(level=logging.WARNING)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



# def create_encrypted_key(name, phone):
#     hasher = hashlib.sha256()
#     hasher.update(f"{name}{phone}".encode('utf-8'))
#     return hasher.hexdigest()

# def create_user_service(user_data: UserData):
#     encrypted_key = create_encrypted_key(user_data.name, user_data.phone)

#     if user_baseInfo_collection.find_one({"_id": encrypted_key}):
#         # 이미 존재하는 경우, 적절한 에러 메시지를 반환합니다.
#         raise HTTPException(
#             status_code=status.HTTP_409_CONFLICT,
#             detail="User with this ID already exists"
#         )
    
#     user_info_data = {
#         "_id": encrypted_key,
#         **user_data.dict(),
#         "encrypted_id": encrypted_key
#     }
    
#     try:
#         user_baseInfo_collection.insert_one(user_info_data)
#         return {"encrypted_id": encrypted_key}
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# def read_user_service(encrypted_id: str):
#     user = user_baseInfo_collection.find_one({"encrypted_id": encrypted_id}, {'_id': False})
#     if user:
#         return user
#     else:
#         raise HTTPException(status_code=404, detail="User not found")
    

# def read_list_service():
#     users = list(user_baseInfo_collection.find({}, {'_id': False}))
#     return users


# def read_user_blog_links(encrypted_id: str) -> list:
#     user = user_baseInfo_collection.find_one({"encrypted_id": encrypted_id}, {'_id': False})
#     if user and "blog_links" in user:
#         return user["blog_links"]
#     else:
#         raise HTTPException(status_code=404, detail="User or blog links not found")
    

    
# def read_user_questions(encrypted_id: str) -> list:
#     user = user_baseInfo_collection.find_one({"encrypted_id": encrypted_id}, {'_id': False})
#     if user and "questions" in user:
#         return user["questions"]
#     else:
#         raise HTTPException(status_code=404, detail="User or questions not found")
    

# def url_to_text(urls):
#     # url 리스트를 받아 각각에 대해 문서를 로드한다.
#     results = []
#     for url in urls:
#         loader = WebBaseLoader(url)
#         docs = loader.load()
#         results.append(docs)
#     return results


# # Prompt templates for system and human messages
# class SystemMessagePromptTemplate:
#     @staticmethod
#     def from_template(template):
#         return template

# class HumanMessagePromptTemplate:
#     @staticmethod
#     def from_template(template):
#         return template

# class ChatPromptTemplate:
#     @staticmethod
#     def from_messages(templates):
#         return templates


# def create_prompt(type, model = None, user_answer = None, user_context = None, motto = None, tobe = None):
#     if type == 1 and model == 'CLOVA':  #  블로그 글 요약
#         return {
#                     "system": 
#                         """
#                         해당 게시글 500토큰 이내로 요약해줘. 흐름에 맞게 주요 내용만 정리해줘. ~습니다. 로 끝나도록 해주세요.
#                         """
#                     ,
#                     "human": 
#                         f"""
#                         내용 : {user_context} 
#                         """
#         }
#     if type == 2 and model == 'CLOVA':
#         return {
#                     "system": 
#                         """
#                         해당 게시글과 사용자의 대답을 통해서 사용자가 되고싶은 사람을 블로그 내용을 참고해서 저는 ~ 합니다 이런 말투로 작성해주세요.
#                         그리고 내용에는 저는 xxx 사람이 되고싶습니다. 라는 내용으로 시작해줘.
#                         그리고 블로그 내용을 참고해서 예를들어, 저는 이런사람이 되고 싶어 이런 공부룰 하고 있습니다. 
#                         그리고 앞으로 이런이런 연구를 하고 싶고, 이런 쪽으로 더 넓은 학문 지식을 갖고싶습니다. 이런식으로 블로그 내용을 2개 이상 참고해주세요.
#                         작성자를 평가하지말고, 본인이 작성자라고 생각하고, 자기소개글을 쓴다는 생각으로 작성해주세요.
#                         반드시, 저는 ~ 입니다. 이런식으로 작성자 입장에서 작성해주세요.
#                         결과는 JSON으로 항상 일정하게 {"result" : { "content" : ~~~ }  } 형식을 지켜주세요
#                         """
#                     ,
#                     "human": 
#                         f"""
#                         게시물 내용 : {user_context}, 사용자 답변 : {user_answer} 
#                         """
#         }
#     if type == 3 and model == 'CLOVA':
#         return {
#                     "system":
#                     '''
#                     당신은 좌우명을 분석해주는 도우미입니다. 사용자의 답변과 context를 활용해서 사용자의 좌우명을 분석해주세요.
#                     당신의 좌우명은 000입니다.로 시작해주세요. 그리고 모든 문장은 평서문으로 해주세요.
#                     좌우명을 선택한 이유에 대해서 나의 좌우명을 블로그의 내용과 엮어서 설명해주세요. 꼭 블로그에 있는 내용이 아니더라도 비슷한 내용으로 분석해주세요.
#                     작성자를 평가하지말고, 본인이 작성자라고 생각하고, 자기소개글을 쓴다는 생각으로 작성해주세요.
#                     반드시, 저는 ~ 입니다. 이런식으로 작성자 입장에서 작성해주세요.

#                     결과는 항상 JSON 형식으로 {"result" : { "content" : ~~~ }  } 형식을 지켜세요.
#                     '''
#                     ,
#                     'human':
#                     f''' 
#                         사용자 답변: {user_answer[1]}, {user_answer[2]}, {user_answer[3]}, 
#                         블로그 내용: {user_context}
#                     '''
#         }
    
#     if type == 4 and model == 'CLOVA':
#         return {
#                     "system":
#                     '''
#                     당신은 사용자의 대답을 통해서 더욱 더 멋있는 말을 만들어주는 도우미입니다.
#                     사용자가 자기소개에서 자신의 인생터닝 포인트를 설명할 수 있도록 글을써주세요.
#                     존댓말로 끝내게 하고 모든 문장은 평서문으로 끝나다로고 해주세요.
#                     작성자의 인생변곡점, 되고싶은 사람, 좌우명에 대한 대답을 최대한 엮어서 작성자의 인생변환 시점을 잘 설명해주세요.
#                     반드시, 저는 ~ 입니다. 이런식으로 작성자 입장에서 작성해주세요.
#                     결과는 항상 JSON 형식으로 {"result" : { "content" : ~~~ }  } 형식을 지켜세요.
#                     '''
#                     ,
#                     'human':
#                     f''' 
#                         인생 변곡점 : {user_answer[-1]},  
#                         되고 싶은 사람 : {user_answer[0]}
#                         좌우명 : {user_answer[1]}
#                     '''
#         }
    
#     if type == 6 and model == 'CLOVA':
#         return {
#                     "system":
#                     '''
#                     작성자가 되어서 자신의 최종 목표를 인생모토, 되고싶은 사람, 잘하는것, 좋아하는것을 엮어서 작성해주세요. 
#                     저의 목표는 000 이고, 이를 위해 어떠한 노력을 하고 있습니다, 항상 이런사람이 되기 위해서 이런식으로 행동합니다. 이런식으로 작성해주세요.
#                     반드시, 저는 ~ 입니다. 이런식으로 작성자 입장에서 작성해주세요.
#                     결과는 항상 JSON 형식으로 {"result" : { "content" : ~~~ }  } 형식을 지켜세요.
#                     '''
#                     ,
#                     'human':
#                     f''' 
#                         인생모토 : {motto}, 
#                         되고싶은 사람 : {tobe}, 
#                         잘하는것 : {user_answer[3]}, 
#                         좋아하는것 :{user_answer[2]},
#                     '''
#         }
    

# ##  앞으로 Clova랑 api 동신을 하기 위해 필요한 변수들 지정.
# class CompletionExecutor:
#     def __init__(self, host, api_key, api_key_primary_val):
#         self._host = host
#         self._api_key = api_key
#         self._api_key_primary_val = api_key_primary_val

#     def execute(self, completion_request):
#         headers = {
#             'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
#             "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
#             'Content-Type': 'application/json; charset=utf-8',
#         }
#         return requests.post(self._host + '/testapp/v1/chat-completions/HCX-003', headers=headers, json=completion_request).text


# # 해당 키에 API를 넣고 모델을 사용할수 있도록 설정한다.
# def get_api_credentials():
#     load_dotenv(dotenv_path='/Users/jeongseungmin/Desktop/Geport-MVP-BackEnd/.env')
#     naver_api_key = os.environ.get("NAVER_API_KEY")
#     naver_gateway_key = os.environ.get("NAVER_GATEWAY_KEY")

#     if not naver_api_key or not naver_gateway_key:
#         raise ValueError("NAVER API KEY or Gateway key is empty!")

#     return naver_api_key, naver_gateway_key


# # Executor를 생성한다.
# def create_completion_executor():
#     api_key, gateway_key = get_api_credentials()
#     completion_executor = CompletionExecutor(
#         host='https://clovastudio.stream.ntruss.com',
#         api_key=api_key,
#         api_key_primary_val=gateway_key
#     )
#     return completion_executor


# # 블로그 요약.
# def Summary(model, type, contents):
#     results = []  # 결과를 저장할 리스트
#     for content in contents:
#         if model == 'CLOVA':
#             completion_executor = create_completion_executor()
#             prompt = create_prompt(type = type , model = model , user_context = content)
            
#             messages = [
#                 {"role": "system", "content":  prompt["system"]},
#                 {"role": "user", "content":  prompt["human"]}
#             ]
#             answer = completion_executor.execute({
#                 'messages': messages,
#                 'topP': 0.9,
#                 'topK': 0,
#                 'maxTokens': 500,  
#                 'temperature': 0.8,
#                 'repeatPenalty': 5.0,
#                 'stopBefore': [],
#                 'includeAiFilters': False,
#                 'seed': 0
#             })
#             answer = json.loads(answer)
#             results.append(answer)
#             contents = [item['result']['message']['content'] for item in results]

#             print('_____' * 100)
#             print('SummarySummarySummarySummarySummarySummary',contents)
#             print('_____' * 100)


#     return contents
    

# # 되고싶은 사람 
# def Tobe(model, type, blog_docs, answer):
#     if model == 'CLOVA' :
#         completion_executor = create_completion_executor()
#         prompt = create_prompt(type = type , model = model , user_context = blog_docs,user_answer = answer)
#         messages = [
#             {"role": "system", "content":  prompt["system"]},
#             {"role": "user", "content":  prompt["human"]}
#         ]
#         answer = completion_executor.execute({
#             'messages': messages,
#             'topP': 0.9,
#             'topK': 0,
#             'maxTokens': 500,  
#             'temperature': 0.8,
#             'repeatPenalty': 5.0,
#             'stopBefore': [],
#             'includeAiFilters': False,
#             'seed': 0
#         })
#         answer = json.loads(answer)
#         result = answer['result']['message']['content']

#         print('_____' * 100)
#         print('ToBEToBEToBEToBEToBEToBEToBE' ,result)
#         print('_____' * 100)
        
#     return result

# def Motto(model, type, blogs, answers):
#     if model == 'CLOVA':
#         completion_executor = create_completion_executor()
#         prompt = create_prompt(type = type , model = model , user_context = blogs ,user_answer = answers)
#         messages = [
#             {"role": "system", "content":  prompt["system"]},
#             {"role": "user", "content":  prompt["human"]}
#         ]
#         answer = completion_executor.execute({
#             'messages': messages,
#             'topP': 0.9,
#             'topK': 0,
#             'maxTokens': 500,  
#             'temperature': 0.8,
#             'repeatPenalty': 5.0,
#             'stopBefore': [],
#             'includeAiFilters': False,
#             'seed': 0
#         })
#         answer = json.loads(answer)
#         result = answer['result']['message']['content']

#         print('_____' * 100)
#         print('MottoMottoMottoMottoMottoMottoMottoMottoMotto' ,result)
#         print('_____' * 100)
        
#     return result

# def TunningPoint(model, type, answers):
#     if model == 'CLOVA':
#         completion_executor = create_completion_executor()
#         prompt = create_prompt(type = type , model = model ,user_answer = answers)
#         messages = [
#             {"role": "system", "content":  prompt["system"]},
#             {"role": "user", "content":  prompt["human"]}
#         ]
#         answer = completion_executor.execute({
#             'messages': messages,
#             'topP': 0.9,
#             'topK': 0,
#             'maxTokens': 1000,  
#             'temperature': 0.9,
#             'repeatPenalty': 5.0,
#             'stopBefore': [],
#             'includeAiFilters': False,
#             'seed': 0
#         })
#         answer = json.loads(answer)
#         result = answer['result']['message']['content']

#         print('_____' * 100)
#         print('TunningTunningTunningTunningTunningTunningTunningTunningTunning' ,result)
#         print('_____' * 100)
#         return result

# def FinalSolution(model, type, motto, tobe, answers):
#     if model == 'CLOVA':
#         completion_executor = create_completion_executor()
#         prompt = create_prompt(type = type , model = model ,user_answer = answers, motto=motto, tobe=tobe)
#         messages = [
#             {"role": "system", "content":  prompt["system"]},
#             {"role": "user", "content":  prompt["human"]}
#         ]
#         answer = completion_executor.execute({
#             'messages': messages,
#             'topP': 0.9,
#             'topK': 0,
#             'maxTokens': 500,  
#             'temperature': 0.9,
#             'repeatPenalty': 5.0,
#             'stopBefore': [],
#             'includeAiFilters': False,
#             'seed': 0
#         })
#         answer = json.loads(answer)
#         result = answer['result']['message']['content']

#         print('_____' * 100)
#         print('TunningTunningTunningTunningTunningTunningTunningTunningTunning' ,result)
#         print('_____' * 100)

#         return result
    

# import time


# def generate_geport(encrypted_id: str):
#     start_time = time.time()  # 함수 실행 시작 시간을 기록
#     blog_urls = read_user_blog_links(encrypted_id)
#     blog_docs = url_to_text(blog_urls)
#     answers = read_user_questions(encrypted_id)

#     summary_clova = Summary('CLOVA', 1, blog_docs)
#     tobo_clova = Tobe('CLOVA', 2, summary_clova, answers[0])
#     motto_clova = Motto('CLOVA', 3, summary_clova, answers)
#     tunningPoint_clova = TunningPoint('CLOVA', 4, answers)
#     final_solution = FinalSolution('CLOVA', 6, motto_clova, tobo_clova, answers)


#     end_time = time.time()  # 함수 실행 완료 시간을 기록
#     execution_time = end_time - start_time  # 실행 시간을 계산

#     result ={
#         'answers' : answers,
#         'summary' : summary_clova,
#         'tobe' : tobo_clova,
#         'motto' :motto_clova,
#         'tunningPoint' : tunningPoint_clova,
#         'final_solution' : final_solution,
#         'Time : ' : execution_time
#     }

    


#     return result






