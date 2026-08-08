from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict
import time
import os

from preprocessing import extract_text, chunk_text
from embedding import create_embeddings, store_embeddings, search_similar

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="Document based Question Answering API"
)

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    # Don't rate-limit OPTIONS requests
    if request.method != "OPTIONS":
        if len(request_times[ip]) >= 20:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Try again later."}
            )

        request_times[ip].append(now)

    return await call_next(request)


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():
    return {
        "message": "RAG API is running",
        "docs": "/docs"
    }


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "document_loaded": len(chunks) > 0,
        "chunks": len(chunks)
    }


# =====================================================
# PROCESS DOCUMENT
# =====================================================

def process_document(text: str):
    global chunks

    new_chunks = chunk_text(text)

    if not new_chunks:
        return

    embeddings = create_embeddings(new_chunks)
    store_embeddings(embeddings, new_chunks)

    chunks = new_chunks


# =====================================================
# REQUEST MODEL
# =====================================================

class Question(BaseModel):
    question: str


# =====================================================
# UPLOAD DOCUMENT
# =====================================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):

    filename = file.filename.lower()

    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        return {
            "success": False,
            "error": "Only PDF and TXT files are supported."
        }

    try:
        content = await file.read()

        if filename.endswith(".pdf"):
            text = extract_text(content, file_type="pdf")
        else:
            text = content.decode("utf-8")

        if not text or not text.strip():
            return {
                "success": False,
                "error": "Could not extract text from the document."
            }

        background_tasks.add_task(
            process_document,
            text
        )

        return {
            "success": True,
            "message": "Document uploaded successfully.",
            "filename": file.filename,
            "status": "Processing document..."
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# =====================================================
# ASK QUESTION
# =====================================================

@app.post("/ask")
def ask_question(req: Question):

    question = req.question.strip()

    if not question:
        return {
            "success": False,
            "error": "Please enter a question."
        }

    if not chunks:
        return {
            "success": False,
            "error": "No document is loaded. Please upload a document first."
        }

    try:

        top_chunks = search_similar(
            question,
            top_k=3
        )

        if not top_chunks:
            return {
                "success": False,
                "error": "No relevant information found."
            }

        context = "\n\n".join(top_chunks)

        # Simple document-based answer
        answer = (
            "Based on the uploaded document:\n\n"
            + context[:1500]
        )

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "context_used": top_chunks
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }
