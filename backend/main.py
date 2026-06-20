from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from rag import answer_query

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def read_root():
    return {"message": "BizBot backend is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = answer_query(request.question)
    return ChatResponse(answer=answer)