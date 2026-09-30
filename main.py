from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from query import search, generate_answer


app = FastAPI(
    title="Chris Lies RAG API",
    description="RAG API for professional case studies",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://chrislies.com",
        "https://www.chrislies.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
def root():
    return {
        "message": "RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):

    results = search(request.question)

    answer = generate_answer(
        request.question,
        results
    )

    return QuestionResponse(
        question=request.question,
        answer=answer
    )