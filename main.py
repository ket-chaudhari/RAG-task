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


# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="RAG Document Question Answering API",
    description="Upload a PDF/TXT document and ask questions from it.",
    version="1.0.0"
)


# =====================================================
# ROOT ROUTE
# =====================================================

@app.get("/")
def home():
    return {
        "message": "RAG API is running",
        "docs": "/docs",
        "health": "/health"
    }


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "document_loaded": len(chunks) > 0
    }


# =====================================================
# OPENAI CONFIGURATION
# =====================================================

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None


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
        t
        for t in request_times[ip]
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

    response = await call_next(request)

    return response


# =====================================================
# BACKGROUND DOCUMENT PROCESSING
# =====================================================

def process_document(text: str):

    global chunks

    try:

        chunks = chunk_text(text)

        embeddings = create_embeddings(chunks)

        store_embeddings(
            embeddings,
            chunks
        )

    except Exception as e:

        print("Document processing error:", e)


# =====================================================
# QUESTION MODEL
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

    if not file.filename:
        return {
            "error": "No file selected"
        }

    filename = file.filename.lower()

    content = await file.read()

    # ---------------------------------------------
    # PDF
    # ---------------------------------------------

    if filename.endswith(".pdf"):

        text = extract_text(
            content,
            file_type="pdf"
        )

    # ---------------------------------------------
    # TXT
    # ---------------------------------------------

    elif filename.endswith(".txt"):

        try:

            text = content.decode("utf-8")

        except UnicodeDecodeError:

            return {
                "error": "Unable to read TXT file as UTF-8"
            }

    # ---------------------------------------------
    # INVALID FILE
    # ---------------------------------------------

    else:

        return {
            "error": "Only PDF and TXT files are allowed"
        }

    # ---------------------------------------------
    # CHECK EXTRACTED TEXT
    # ---------------------------------------------

    if not text or not text.strip():

        return {
            "error": "No text could be extracted from the file"
        }

    # ---------------------------------------------
    # BACKGROUND PROCESSING
    # ---------------------------------------------

    background_tasks.add_task(
        process_document,
        text
    )

    return {
        "message": "Upload successful. Processing started.",
        "filename": file.filename
    }


# =====================================================
# LLM ANSWER GENERATION
# =====================================================

def generate_answer(question: str, context: str):

    # ---------------------------------------------
    # No OpenAI API key
    # ---------------------------------------------

    if client is None:

        return (
            "OpenAI API key is not configured.\n\n"
            "Context from document:\n"
            + context[:800]
        )

    # ---------------------------------------------
    # Prompt
    # ---------------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the
information provided in the context.

If the answer cannot be found in the context,
say that the information is not available
in the uploaded document.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    # ---------------------------------------------
    # OpenAI API
    # ---------------------------------------------

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

        return f"LLM error: {str(e)}"


# =====================================================
# ASK API
# =====================================================

@app.post("/ask")
def ask_question(req: Question):

    # ---------------------------------------------
    # Check question
    # ---------------------------------------------

    if not req.question.strip():

        return {
            "error": "Question cannot be empty"
        }

    # ---------------------------------------------
    # Check document
    # ---------------------------------------------

    if not chunks:

        return {
            "error": (
                "Document not processed yet. "
                "Please upload a PDF/TXT document first "
                "and wait for processing to complete."
            )
        }

    try:

        # -----------------------------------------
        # Semantic search
        # -----------------------------------------

        top_chunks = search_similar(
            req.question,
            top_k=3
        )

        if not top_chunks:

            return {
                "error": "No relevant chunks found in document"
            }

        # -----------------------------------------
        # Create context
        # -----------------------------------------

        context = "\n\n".join(top_chunks)

        # -----------------------------------------
        # Generate answer
        # -----------------------------------------

        answer = generate_answer(
            req.question,
            context
        )

        # -----------------------------------------
        # Response
        # -----------------------------------------

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
# SERVER INFORMATION
# =====================================================

@app.get("/status")
def status():

    return {
        "api": "RAG API",
        "status": "running",
        "document_loaded": len(chunks) > 0,
        "chunks": len(chunks)
    }
