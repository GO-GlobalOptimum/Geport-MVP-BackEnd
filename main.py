from typing import Union
from fastapi import FastAPI, HTTPException
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


