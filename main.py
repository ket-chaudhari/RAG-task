from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from collections import defaultdict
import time
import os

from preprocessing import extract_text, chunk_text
from embedding import create_embeddings, store_embeddings, search_similar

from openai import OpenAI


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI()


# =====================================================
# HOME / HEALTH CHECK
# =====================================================

@app.get("/")
def home():
    return {
        "message": "RAG API is running"
    }


# =====================================================
# OPENAI CLIENT
# =====================================================

# Set your API key as an environment variable:
# Windows CMD:
# set OPENAI_API_KEY=your_new_api_key

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) if api_key else None


# =====================================================
# STORAGE
# =====================================================

chunks = []

request_times = defaultdict(list)


# =====================================================
# RATE LIMITING
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
                "error": "Too many requests. Try later."
            }
        )

    request_times[ip].append(now)

    return await call_next(request)


# =====================================================
# BACKGROUND DOCUMENT PROCESSING
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
# REQUEST MODEL
# =====================================================

class Question(BaseModel):
    question: str


# =====================================================
# UPLOAD API
# PDF + TXT
# =====================================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):

    content = await file.read()

    filename = file.filename.lower()

    # -----------------------------
    # PDF
    # -----------------------------

    if filename.endswith(".pdf"):

        text = extract_text(
            content,
            file_type="pdf"
        )

    # -----------------------------
    # TXT
    # -----------------------------

    elif filename.endswith(".txt"):

        text = content.decode("utf-8")

    # -----------------------------
    # INVALID FILE
    # -----------------------------

    else:

        return {
            "error": "Only PDF and TXT files are allowed"
        }

    # -----------------------------
    # PROCESS DOCUMENT
    # -----------------------------

    background_tasks.add_task(
        process_document,
        text
    )

    return {
        "message": "Upload successful. Processing started."
    }


# =====================================================
# LLM FUNCTION
# =====================================================

def generate_answer(question, context):

    # ---------------------------------
    # If OpenAI key is not configured
    # ---------------------------------

    if client is None:

        return (
            "OpenAI API key is not configured.\n\n"
            "Context-based answer:\n"
            + context[:500]
        )

    # ---------------------------------
    # Prompt
    # ---------------------------------

    prompt = f"""
You are a helpful AI assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    # ---------------------------------
    # OpenAI request
    # ---------------------------------

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


# =====================================================
# ASK API
# RAG CORE
# =====================================================

@app.post("/ask")
def ask_question(req: Question):

    # ---------------------------------
    # STEP 1: Check document
    # ---------------------------------

    if not chunks:

        return {
            "error": (
                "Document not processed yet. "
                "Please wait after upload."
            )
        }

    try:

        # ---------------------------------
        # STEP 2: Retrieval
        # ---------------------------------

        top_chunks = search_similar(
            req.question,
            top_k=3
        )

        if not top_chunks:

            return {
                "error": "No relevant chunks found in document"
            }

        # ---------------------------------
        # STEP 3: Create context
        # ---------------------------------

        context = "\n".join(top_chunks)

        # ---------------------------------
        # STEP 4: Generate answer
        # ---------------------------------

        answer = generate_answer(
            req.question,
            context
        )

        # ---------------------------------
        # STEP 5: Return response
        # ---------------------------------

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
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "document_loaded": bool(chunks)
    }
