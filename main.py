from typing import Union
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from pydantic import BaseModel
import os
import logging

from app.router.geport import router as geport_router
from app.router.igeport import router as igeport_router


app = FastAPI(root_path="/api")
app.include_router(geport_router)
app.include_router(igeport_router)

@app.get("/")
def read_root():
    return {"Hello": "World"}
print('hello')




# @app.get("/generate_branding_report/{encrypted_id}", response_model=dict)
# async def generate_branding_report(encrypted_id: str):
#     # MongoDB에서 사용자 정보 조회
#     user_data = read_user_service(encrypted_id)
#     if not user_data:
#         raise HTTPException(status_code=404, detail="User not found")

#     # ChatGPT를 사용하여 개인 브랜딩 보고서 생성
#     response = ask_chatgpt(user_data)
#     return {"ChatGPT Response": response}




# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

# # MongoDB에 아이템 추가
# @app.post("/items/")
# def create_item(item: Item):
#     item_dict = item.dict()
#     if item.tax:
#         price_with_tax = item.price + item.tax
#         item_dict.update({"price_with_tax": price_with_tax})
#     db.items.insert_one(item_dict)
#     return item_dict

# # MongoDB에서 아이템 조회
# @app.get("/items/")
# def read_items():
#     items = list(db.items.find())
#     for item in items:
#         item["_id"] = str(item["_id"])
#     return items