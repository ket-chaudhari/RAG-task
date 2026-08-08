from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from collections import defaultdict
from dotenv import load_dotenv

import os
import time
import pickle
import faiss

from preprocessing import extract_text, chunk_text
from embedding import (
    create_embeddings,
    store_embeddings,
    search_similar
)

from openai import OpenAI


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found.")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="RAG AI Assistant",
    description="PDF/TXT based Retrieval Augmented Generation API",
    version="1.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# STORAGE
# =========================================================

chunks = []
request_times = defaultdict(list)


# =========================================================
# LOAD EXISTING FAISS DATA
# =========================================================

def load_existing_data():

    global chunks

    try:

        if os.path.exists("chunks.pkl") and os.path.exists("vector.index"):

            with open("chunks.pkl", "rb") as f:
                chunks = pickle.load(f)

            print(f"Loaded {len(chunks)} chunks from existing database.")

        else:

            print("No existing vector database found.")

    except Exception as e:

        print("Error loading existing data:", e)


load_existing_data()


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "RAG AI Assistant API is running",
        "status": "success"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "documents_loaded": len(chunks)
    }


# =========================================================
# RATE LIMITING
# =========================================================

@app.middleware("http")
async def rate_limit(request: Request, call_next):

    ip = request.client.host
    now = time.time()

    request_times[ip] = [
        t for t in request_times[ip]
        if now - t < 60
    ]

    if len(request_times[ip]) >= 30:

        return JSONResponse(
            status_code=429,
            content={
                "error": "Too many requests. Try again later."
            }
        )

    request_times[ip].append(now)

    return await call_next(request)


# =========================================================
# DOCUMENT PROCESSING
# =========================================================

def process_document(text):

    global chunks

    try:

        new_chunks = chunk_text(text)

        if not new_chunks:

            print("No text chunks created.")

            return

        embeddings = create_embeddings(new_chunks)

        store_embeddings(
            embeddings,
            new_chunks
        )

        chunks = new_chunks

        print(
            f"Document processed successfully: {len(chunks)} chunks"
        )

    except Exception as e:

        print("Document processing error:", e)


# =========================================================
# UPLOAD API
# =========================================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):

    filename = file.filename.lower()

    if not (
        filename.endswith(".pdf")
        or filename.endswith(".txt")
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "Only PDF and TXT files are allowed."
            }
        )

    try:

        content = await file.read()

        if filename.endswith(".pdf"):

            text = extract_text(
                content,
                file_type="pdf"
            )

        else:

            text = content.decode(
                "utf-8",
                errors="ignore"
            )

        if not text.strip():

            return JSONResponse(
                status_code=400,
                content={
                    "error": "No readable text found in document."
                }
            )

        background_tasks.add_task(
            process_document,
            text
        )

        return {
            "message": "File uploaded successfully.",
            "filename": file.filename,
            "status": "processing"
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


# =========================================================
# REQUEST MODEL
# =========================================================

class Question(BaseModel):

    question: str


# =========================================================
# GENERATE AI ANSWER
# =========================================================

def generate_answer(question, context):

    # If OpenAI key is not available,
    # return retrieved context instead.

    if client is None:

        return (
            "OpenAI API key is not configured.\n\n"
            "Relevant information from the document:\n\n"
            + context[:1500]
        )

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided document context.

If the answer is not present in the context, say:
"I could not find this information in the uploaded document."

Keep the answer clear and concise.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    try:

        response = client.chat.completions.create(

            model="gpt-4o-mini",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            "AI generation failed.\n\n"
            f"Error: {str(e)}\n\n"
            "Relevant document context:\n"
            + context[:1500]
        )


# =========================================================
# ASK API
# =========================================================

@app.post("/ask")
def ask_question(req: Question):

    if not req.question.strip():

        return JSONResponse(
            status_code=400,
            content={
                "error": "Please enter a question."
            }
        )

    if not chunks:

        return JSONResponse(
            status_code=400,
            content={
                "error": "No document uploaded yet. Please upload a PDF or TXT file first."
            }
        )

    try:

        top_chunks = search_similar(
            req.question,
            top_k=3
        )

        if not top_chunks:

            return {
                "question": req.question,
                "answer": "No relevant information found.",
                "context_used": []
            }

        context = "\n\n".join(top_chunks)

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

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


# =========================================================
# TEST ROUTE
# =========================================================

@app.get("/test")
def test():

    return {
        "message": "Test route working"
    }
