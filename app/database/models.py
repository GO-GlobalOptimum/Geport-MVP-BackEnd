from pydantic import BaseModel, HttpUrl
from typing import List

class UserData(BaseModel):
    name: str
    bio: str
    phone: str
    mbti: str
    age: int
    gender: str
    blog_links: List[HttpUrl]
    questions: List[str]
    

class UserQuestions(BaseModel):
    question_1: str
    question_2: str
    question_3: str
    question_4: str
    question_5: str