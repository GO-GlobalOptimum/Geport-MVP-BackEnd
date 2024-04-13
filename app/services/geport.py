from fastapi.responses import JSONResponse
from fastapi import HTTPException, status
import hashlib
from app.database.models import UserData, UserQuestions
from app.database.connection import user_baseInfo_collection
import os
import json
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
llm4_json = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.9, openai_api_key=OPENAI_API_KEY, model_kwargs={"response_format": {'type':"json_object"}}, request_timeout=300)
llm4 = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.9, openai_api_key=OPENAI_API_KEY, request_timeout=300)


def create_encrypted_key(name, phone):
    hasher = hashlib.sha256()
    hasher.update(f"{name}{phone}".encode('utf-8'))
    return hasher.hexdigest()

def create_user_service(user_data: UserData):
    encrypted_key = create_encrypted_key(user_data.name, user_data.phone)

    if user_baseInfo_collection.find_one({"_id": encrypted_key}):
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
    

def read_list_service():
    users = list(user_baseInfo_collection.find({}, {'_id': False}))
    return users


def create_rag_prompt(type):
    if type == 1:
        return ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(
    """
    Task Description:
        You are an assistant who will help you with personal branding based on your blog. An answer to the question is created in Korean with approximately 1,000 characters based on the User Answer and Context. 
        Content you need to provide: Analyzing based on the input, provide insights into what kind of person the user wants to become. and Add the analyzed context content to the output you provide.
        Response format: We plan to use the generated sentences as content to express ourselves, so please write from the user's perspective. Don't say thank you at the end of a sentence. Please write from the user's perspective. and You have to use honorifics.
        Weight considered during analysis: Set the importance at 60% for extracted content from context and 40% for answer to questions.

    Example
        input:
            Question: 당신은 어떤 사람이 되고싶나요?
            User Answer: 나만의 기록 친구, Looi라는 앱을 개발하면서, 많은 사람들이 기록을 통해서 심리적 불안감을 해소하고 있다는 것을 느꼈습니다. 이러한 경험을 통해서 더 많은 사람들이 자신의 감정을 기록하고, 공유하며 내면이 건강해지는 세상을 만들고 싶습니다.
            Context: 관련 내용들이 담긴 문서

        output:
            Answer: 저의 스타트업 프로젝트인 Looi 앱을 통해, 많은 분들이 자신의 일상과 감정을 기록함으로써 심리적 안정감을 찾고 계시다는 것을 직접 목격하였습니다. 이러한 경험은 저에게 매우 큰 영감을 주었으며, 이를 통해 더 많은 분들이 자신의 감정을 솔직하게 기록하고, 이를 통해 자기 자신과 타인과의 깊은 연결을 경험할 수 있는 세상을 만드는 것이 저의 큰 소망이 되었습니다. 저는 기술이 단순히 생활을 편리하게 하는 도구를 넘어, 우리 각자의 내면을 들여다보고 성찰할 수 있는 강력한 수단이 될 수 있다고 믿습니다. 이러한 비전을 바탕으로, 저는 앞으로도 사용자들이 자신의 내면을 건강하게 다스리고, 서로의 경험을 공유하며 서로를 이해하는 데 도움을 줄 수 있는 다양한 프로젝트를 개발해 나갈 계획입니다. 이 과정에서 저는 기술과 인간의 감정이 서로 조화롭게 어우러질 수 있는 방법을 모색하며, 사람들이 자신의 감정을 건강하게 표현하고 관리할 수 있는 더 많은 기회를 제공하기 위해 노력할 것입니다.

    prompt         
        input:
            Question: 당신은 어떤 사람이 되고싶나요?
            User Answer: {answer}
            Context: {context}

        Analyzing based on the input, provide insights into what kind of person the user wants to become. 

     """)])


    elif type == 2:
         return ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(
    """
    Task Description:
        You are an assistant who will help you with personal branding based on your blog. An answer to the question is created in Korean with approximately 1,000 characters based on the User Answer and Context. 
        Content you need to provide: Analyzing the context based on the Motto provided by the user.
        Response format: We plan to use the generated sentences as content to express ourselves, so please write from the user's perspective. Don't say thank you at the end of a sentence. Please write from the user's perspective. and You have to use honorifics.

    Example:
        input
            Question: 당신은 좌우명은 무엇인가요? 그 좌우명에 관련된 사례를 하나 알려주세요
            User Answer: "세상에 안 되는 일은 없다. 안 된다고 생각해서 시도하지 않았기 때문에 안 되는 것일 뿐이다." 라는 좌우명을 가지고 있습니다. 개발을 잘 하지 못했지만 Looi를 개발하면서, 불가능해 보이는 많은 일들을 마주했습니다. 하지만 사용자에게 꼭 필요한 기능이라고 생각하면서 끊임없이 도전했고 그 결과, 성공적으로 서비스를 제작했으며 많은 사용자들이 Looi를 통해 자신의 감정을 기록하고, 공유하며 서로를 이해하는 데 도움을 받고 있습니다.
            Context: 관련 내용들이 담긴 문서

        output
            Answer: 세상에 안 되는 일은 없다"는 좌우명을 마음속 깊이 새기고, 저는 Looi 앱 개발이라는 새로운 여정을 시작하였습니다. 초기 개발 과정에서 많은 어려움에 부딪혔음에도 불구하고, 저는 사용자분들에게 꼭 필요한 기능을 제공하고자 하는 일념 하에 끊임없이 노력하였습니다. 그 결과, Looi는 성공적으로 출시되었으며, 많은 분들이 이를 통해 자신의 감정을 기록하고 공유함으로써 서로를 더 잘 이해하게 되었다는 소식을 접하게 되었습니다. 이는 저에게 큰 만족감과 보람을 안겨주었습니다.\n\n이러한 경험을 통해, 저는 도전을 두려워하지 않는 마음가짐의 중요성을 깨달았습니다. 앞으로도 저는 계속해서 새로운 도전에 맞서며, 이를 통해 배우고 성장해 나가고자 합니다. 또한, 저는 더 많은 분들이 자신의 내면을 탐색하고 서로를 이해하는 데 도움이 될 수 있는 방법을 모색할 계획입니다.

    prompt
        Question: 당신은 좌우명은 무엇인가요? 그 좌우명에 관련된 사례를 하나 알려주세요
        User Answer: {answer}
        Context: {context}

        
        Analyze the context based on the user's mottos and examples thereof.

    """)])


graph_prompt = """
Based on the user's information, we create a formula that can be drawn on a coordinate plane where the x-axis represents time and the y-axis represents the success index.
Please include anything that could signify an inflection point in your life. Basically, the formula should be structured in such a way that it increases over time. Please make it a function of degree 3 or higher and include one symbols such as sin, cos, tan, and ln.
Response format: json format, and first key -> "equation" and the second key -> "explanation". explanation is korean. 사용자의 입장에서 수식에 대한 의미를 설명하는 것입니다. The more complex the formula, the better. but, it must not contain complex numbers. create only formula ex) y = 1.5 * ln(x + 1) + 0.5 * e**(0.1 * (x - 20))
"""

answer2_prompt = """
Task Description:
    You are an assistant who will help you with personal branding based on your blog. An answer to the question is created in Korean with approximately 1,000 characters based on the User Answer and Context. 
    Content you need to provide: Explain the motto and analyze the content based on the motto.
    Response format: We plan to use the generated sentences as content to express ourselves, so please write from the user's perspective. Don't say thank you at the end of a sentence. Please write from the user's perspective. and You have to use honorifics. 그리고 저의 인생의 좌우명은 ~ 입니다. 로 문장을 시작해.

Example:
    input
        Question: 당신은 좌우명은 무엇인가요? 그 좌우명에 관련된 사례를 하나 알려주세요
        User Answer: "세상에 안 되는 일은 없다. 안 된다고 생각해서 시도하지 않았기 때문에 안 되는 것일 뿐이다." 라는 좌우명을 가지고 있습니다. 개발을 잘 하지 못했지만 Looi를 개발하면서, 불가능해 보이는 많은 일들을 마주했습니다. 하지만 사용자에게 꼭 필요한 기능이라고 생각하면서 끊임없이 도전했고 그 결과, 성공적으로 서비스를 제작했으며 많은 사용자들이 Looi를 통해 자신의 감정을 기록하고, 공유하며 서로를 이해하는 데 도움을 받고 있습니다.

    output
        세상에 안 되는 일은 없다"는 좌우명을 마음속 깊이 새기고, 저는 Looi 앱 개발이라는 새로운 여정을 시작하였습니다. 초기 개발 과정에서 많은 어려움에 부딪혔음에도 불구하고, 저는 사용자분들에게 꼭 필요한 기능을 제공하고자 하는 일념 하에 끊임없이 노력하였습니다. 그 결과, Looi는 성공적으로 출시되었으며, 많은 분들이 이를 통해 자신의 감정을 기록하고 공유함으로써 서로를 더 잘 이해하게 되었다는 소식을 접하게 되었습니다. 이는 저에게 큰 만족감과 보람을 안겨주었습니다.\n\n이러한 경험을 통해, 저는 도전을 두려워하지 않는 마음가짐의 중요성을 깨달았습니다. 앞으로도 저는 계속해서 새로운 도전에 맞서며, 이를 통해 배우고 성장해 나가고자 합니다. 또한, 저는 더 많은 분들이 자신의 내면을 탐색하고 서로를 이해하는 데 도움이 될 수 있는 방법을 모색할 계획입니다.

prompt:
    Explain the motto and analyze the content based on the motto , preferences, and strengths.
    
    사용자의 좋아하는 것은 {answers[2]}\n사용자가 잘하는 것은 {answers[3]}\n\n{answer_2}.
"""

answer3_prompt = """
Task Description:
    You are an assistant who will help you with personal branding based on your blog. An answer to the question is created in Korean with approximately 1,000 characters based on the User Answer and Context. 
    Content you need to provide: Describe the turning points in life.
    Response format: We plan to use the generated sentences as content to express ourselves, so please write from the user's perspective. Don't say thank you at the end of a sentence. Please write from the user's perspective. and You have to use honorifics.

Example:
    input
        Question: 인생의 변곡점은 무엇인가요? 힘들었지만 극복했었던 사례를 알려주세요.
        User Answer: 중학생때 학업에 대한 흥미를 잃었을 때가 가장 힘들었습니다. 어떤 것에도 흥미를 가지기 어려웠어요. 그렇지만 대학교에 들어와서 코딩에 대한 흥미를 찾게 되면서 인생의 방향을 찾을 수 있었습니다.

    output
        저의 인생의 변곡점은 중학생 때 학업에 대한 흥미를 잃었던 시기입니다. 그 당시에는 어떤 것에도 큰 관심을 가지기 어려웠고, 저 자신도 방향을 잃은 채로 헤매고 있었습니다. 하지만 대학에 진학하여 코딩을 처음 접하게 되면서, 제 인생에 새로운 전환점을 맞이하게 되었습니다.
        코딩을 통해 저는 문제를 해결하는 과정에서 오는 성취감과 기쁨을 발검할 수 있었고, 점차 학업에 대한 열정을 되찾게 되었습니다. 
        또한, 코딩은 저에게 끊임없이 새로운 것을 배우고 탐구하며 성장할 수 있는 기회를 제공해주었습니다. 이를 통해 저는 목표를 설정하고, 그 목표를 향해 꾸준히 노력하는 중요성을 깨닫게 되었습니다.
        \"Make your wave!\"라는 좌우명을 통해 새로운 도전을 두렵지 않게 받아들이고, 끊임없이 발전할 수 있었습니다.

prompt:
    Describe the turning points in life.

    사용자의 정보는 다음과 같습니다.
    사용자의 좋아하는 것은 {answers[2]}\n 사용자가 잘하는 것은 {answers[3]}\n\n{answer_2}이다.
    그리고 사용자의 좌우명과 그에 대한 분석은 다음과 같다. {answer_2}

"""
#다음 내용을 참고하여 새로 작성해주세요.\n\n{answer_2}\n{answers[4]}

answer5_prompt = """
Task Description:
    You are an assistant who will help you with personal branding based on your blog. An answer to the question is created in Korean with approximately 1,000 characters based on the User Answer and Context. 
    Content you need to provide: Based on the information provided by the user, I will write a brief summary and a message expressing determination, aspirations, and commitment to personal growth.
    Response format: We plan to use the generated sentences as content to express ourselves, so please write from the user's perspective. Don't say thank you at the end of a sentence. Please write from the user's perspective. and You have to use honorifics.

Example:
    input: 
        좌우명 -> make your wave
        사용자가 되고 싶은 사람 -> 제가 되고 싶은 사람은 주변의 작은 것에서 행복을 찾고, 그 행복을 주변 사람들과 나눌 줄 알고 이런 사소한 순간들에서 행복을 느낄 수 있는 사람이 되고 싶습니다. 또한 자신의 목적을 이루기 위해 최선을 다하고 주변의 시선을 신경쓰기보다 자신만의 속도로 설정한 목표를 이루는 사람입니다.
        사용자가 좋아하는 것 -> 저는 강아지, 맛있는 음식, 좋아하는 사람들과의 시간, 그리고 설정한 목표를 이루는 것을 좋아합니다. 이런 것들은 저에게 큰 기쁨을 주며, 일상 생활에서의 행복을 찾는 원천이 됩니다. 이런 작은 순간들이 제 삶을 더욱 풍부하고 의미 있게 만들어줍니다.
        사용자가 잘하는 것 -> 저는 특히 Fast API를 이용한 백엔드 개발에 능숙합니다. 이 분야에서 제가 가진 역량을 발휘할 수 있도록, 끊임없이 새로운 지식을 습득하고, 다양한 프로젝트에 도전합니다. 저는 이를 통해 문제를 해결하고, 사용자에게 가치를 제공하는 서비스를 만드는 것에 큰 만족감을 느낍니다.

    output:
        제 좌우명 "Make your wave"는 저만의 길을 만들어나가며, 주변의 작은 것에서 큰 행복을 찾고 이를 나누는 삶을 살겠다는 저의 꿈과 목표를 담고 있습니다. 강아지의 순수한 기쁨, 맛있는 음식과의 소중한 순간, 그리고 사랑하는 사람들과 보내는 시간은 저의 일상을 풍요롭게 합니다. 또한, 제가 설정한 목표를 향해 나아가는 과정에서 발견하는 성취감과 만족은 저를 더욱 단단하게 만듭니다.
        저는 Fast API를 활용한 백엔드 개발 분야에서 끊임없이 성장하고자 합니다. 이 기술을 통해 사용자에게 실질적인 가치를 제공하며, 저의 전문성을 더욱 발전시키고 싶습니다. 새로운 지식을 습득하고 다양한 도전을 하는 과정에서 저는 저만의 파도를 만들어 나가고자 합니다.
        되고 싶은 사람, 즉 주변의 작은 것에서 행복을 찾고 그 행복을 주변 사람들과 나눌 줄 아는 사람이 되기 위해, 저는 매 순간 최선을 다할 것입니다. 저의 목적을 이루기 위해 주변의 시선보다는 자신만의 속도로 목표를 향해 나아갈 것입니다. 이를 위해 저는 끊임없이 자기계발에 힘쓰며, 저만의 길을 걸어갈 것입니다.
        저의 삶은 "Make your wave"라는 좌우명 아래에서, 저만의 속도로, 저만의 방식으로 행복을 찾고 나누며, 저만의 파도를 만들어나가는 여정입니다. 저는 이 여정을 통해 제가 진정으로 원하는 삶을 살아가고자 합니다. 앞으로도 저는 저의 열정과 끈기를 바탕으로 더욱 발전하고 성장하는 모습을 보여드리겠습니다.
        

prompt:
    한줄 요약과 발전하기 위한 각오, 포부 등을 담은 글을 존댓말로 작성해주세요.

    사용자의 정보는 다음과 같습니다.
    사용자가 되고싶은 사람 -> {answer_1}
    좌우명 -> {answer_2}
    사용자가 좋아하는 것 -> {answers[2]}
    사용자가 잘하는 것 -> {answers[3]}
    
"""

def read_user_blog_links(encrypted_id: str) -> list:
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


def create_vector_store(url_list):
    flag = False
    for url in url_list:
        if not flag:
            docs = url_to_text(url)
            flag = True
        else:
            docs += url_to_text(url)

    splits = split_text(docs)

    vector_store = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vector_store.as_retriever()
    return retriever

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


# 여기서 return 하는 rag_chain은 이 모든 과정을 하게 해주는것!
# 그니깐 retriver, prompt 만 입력해서 넣으면 다음고 같은 과정을 수행하는 rag_chain을 얻는다.
# 그래서 나중에 .invoke()를 사용해서 질문에 대한 답을 얻을 수 있다.
def create_rag_chain(retriever, prompt):
    rag_chain = (
        {"context": retriever | format_docs, "answer": RunnablePassthrough()}
        | prompt
        | llm4
        | StrOutputParser()
    )
    return rag_chain


# 질문은 front에서 받아와야 하는 상황이다.
def generate_geport(encrypted_id: str):
    url_list = read_user_blog_links(encrypted_id)
    # url_list의 문서들을 가져와 문자로 바꾸고
    # 해당 문자를 분할해서 특정 조각으로 쪼개고
    # 그 쪼갠 조각들을 embedding 하여 vector_store에저장하고
    # vectore_store.as_retriver()를 사용하여 겁색가능 한 retriver를 만들었다.
    # 즉, url_list들을 벡터디비에 넣고 해당 디비에서 검색 기능을 구현하는 retriever를 만든것이다.
    retriever = create_vector_store(url_list)

    answers = read_user_questions(encrypted_id)


    # 2. RAG chain 생성 (1), (2)
    # retreiver 즉, 블로그 내용들을 검색하는데, prompo에 맞게 검색을 하는 rag_chain을 생성한다.
    # 그래서 총 2개에 대해서 각각 블로그와 prompt를 활용한다.
    rag_chain_1 = create_rag_chain(retriever, create_rag_prompt(1))
    rag_chain_2 = create_rag_chain(retriever, create_rag_prompt(2))

    # 3. RAG chain 실행 Q1 -> (1), Q2 -> (2)
    # anwer를 사용해 검색을 한다.
    answer_1 = rag_chain_1.invoke(answers[0])
    answer_2 = rag_chain_2.invoke(answers[1])

    # 4. Q3, Q4, (2)를 활용하여 (2)를 다시 생성
    answer_2 = llm4.predict(answer2_prompt)

    # 5. Q5를 활용하여 (3)을 생성
    answer_3 = llm4.predict(answer3_prompt)

    # 6. (1), (3)을 활용하여 (4)를 생성
    answer_4 = llm4_json.predict(f"json {graph_prompt}\n{answer_1}\n{answer_3}")

    # 7. (2), (4)를 활용하여 (5)를 생성
    answer_5 = llm4.predict(f"Don't say thank you at the end of a sentence. Please write from the user's perspective. 한줄 요약과 발전하기 위한 각오, 포부 등을 담은 글을 존댓말로 작성해주세요.(1000자)\n\n사용자의 정보는 다음과 같습니다.\n\n{answer_1}\n\n{answer_2}\n\n{json.loads(answer_4)['explanation']}")
    
    result= {
            "result": {
                "저는 이런 사람이 되고싶어요": answer_1,
                "저의 좌우명은 다음과 같습니다 ": answer_2,
                "제 인생의 변곡점은 다음과 같아요": answer_3,
                "이것이 재 인생 함수입니다": answer_4,
                "Geport Solution": answer_5,
            }
        }
    return result
    ## 데이터 베이스 저장
    # try:
    #     # 해당 encrypted_key를 가진 문서를 찾아, geport 필드를 빈 배열로 설정합니다.
    #     update_result = user_baseInfo_collection.update_one(
    #         {"_id": encrypted_id},
    #         {"$set": {"geport": result}}
    #     )

    #     # 문서 업데이트가 성공적으로 이루어졌는지 확인합니다.
    #     if update_result.modified_count == 0:
    #         # 아무 문서도 업데이트되지 않았다면, 문서가 없는 것일 수 있습니다.
    #         return {"message": "No document found with the given encrypted_key, or the geport field is already set to an empty array."}
    #     else:
    #         # 문서 업데이트 성공
    #         return {"message": "The geport field has been successfully added or updated to an empty array."}
    # except Exception as e:
    #     # 에러 처리
    #     raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

