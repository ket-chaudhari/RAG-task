from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from collections import defaultdict
import time
import os

from preprocessing import extract_text, chunk_text
from embedding import (
    create_embeddings,
    store_embeddings,
    search_similar
)

from openai import OpenAI


app = FastAPI(
    title="RAG API",
    description="PDF/TXT document question answering API",
    version="1.0.0"
)


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def home():
    return {
        "message": "RAG API is running",
        "docs": "/docs",
        "health": "/health"
    }


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "document_loaded": len(chunks) > 0
    }


# =====================================================
# OPENAI
# =====================================================

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) if api_key else None


# =====================================================
# STORAGE
# =====================================================

chunks = []

request_times = defaultdict(list)


# =====================================================
# RATE LIMIT
# =====================================================

@app.middleware("http")
async def rate_limit(request: Request, call_next):

    ip = request.client.host
    now = time.time()

    request_times[ip] = [
        t for t in request_times[ip]
        if now - t < 60
    ]

    if len(request_times[ip]) >= 5:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too many requests. Try again later."
            }
        )

    request_times[ip].append(now)

    return await call_next(request)


# =====================================================
# BACKGROUND PROCESSING
# =====================================================

def process_document(text: str):

    global chunks

    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    store_embeddings(
        embeddings,
        chunks
    )


# =====================================================
# QUESTION MODEL
# =====================================================

class Question(BaseModel):
    question: str


# =====================================================
# UPLOAD
# =====================================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):

    content = await file.read()

    filename = file.filename.lower()

    if filename.endswith(".pdf"):

        text = extract_text(
            content,
            file_type="pdf"
        )

    elif filename.endswith(".txt"):

        text = content.decode("utf-8")

    else:

        return {
            "error": "Only PDF and TXT files are allowed"
        }

    if not text.strip():

        return {
            "error": "No text could be extracted"
        }

    background_tasks.add_task(
        process_document,
        text
    )

    return {
        "message": "Upload successful. Processing started.",
        "filename": file.filename
    }


# =====================================================
# GENERATE ANSWER
# =====================================================

def generate_answer(question: str, context: str):

    if client is None:

        return (
            "OpenAI API key is not configured.\n\n"
            "Context-based answer:\n"
            + context[:800]
        )

    prompt = f"""
You are a helpful AI assistant.

Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"OpenAI error: {str(e)}"


# =====================================================
# ASK
# =====================================================

@app.post("/ask")
def ask_question(req: Question):

    if not req.question.strip():

        return {
            "error": "Question cannot be empty"
        }

    if not chunks:

        return {
            "error": (
                "Document not processed yet. "
                "Please upload a document first."
            )
        }

    try:

        top_chunks = search_similar(
            req.question,
            top_k=3
        )

        if not top_chunks:

            return {
                "error": "No relevant chunks found"
            }

        context = "\n".join(top_chunks)

        answer = generate_answer(
            req.question,
            context
        )

        return {
            "question": req.question,
            "answer": answer,
            "context_used": top_chunks
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# =====================================================
# STATUS
# =====================================================

@app.get("/status")
def status():

    return {
        "api": "RAG API",
        "status": "running",
        "document_loaded": len(chunks) > 0,
        "chunks": len(chunks)
    }
