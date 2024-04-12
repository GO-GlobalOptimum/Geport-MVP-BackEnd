# DB/connection.py
from pymongo import MongoClient
import os

mongodb_url = os.getenv("MONGODB_URL", "mongodb://admin:1234@localhost:27017/admin")
client = MongoClient(mongodb_url)
db = client["admin"]

user_baseInfo_collection = db["users_info"]
igeport_user_baseInfo_collection = db['igeport_users_info']

