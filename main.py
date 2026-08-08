from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocessing import extract_text, chunk_text
from embedding import (
    create_embeddings,
    store_embeddings,
    search_similar
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="Document based Question Answering API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL DOCUMENT STORAGE
# ============================================================

chunks = []


# ============================================================
# QUESTION MODEL
# ============================================================

class Question(BaseModel):
    question: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "RAG API is running",
        "status": "online"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# TEST
# ============================================================

@app.get("/test")
def test():
    return {
        "message": "Test route working"
    }


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_document(text: str):

    global chunks

    if not text or not text.strip():
        raise ValueError(
            "No readable text found in document."
        )

    # Create text chunks
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(
            "Could not create chunks from document."
        )

    print(
        f"Created {len(chunks)} text chunks."
    )

    # Create embeddings
    embeddings = create_embeddings(
        chunks
    )

    # Store embeddings in FAISS
    store_embeddings(
        embeddings,
        chunks
    )

    print(
        "Document processing completed successfully."
    )


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...)
):

    global chunks

    if not file.filename:

        return {
            "success": False,
            "error": "No file selected."
        }

    filename = file.filename.lower()

    # Check file type
    if not (
        filename.endswith(".pdf")
        or filename.endswith(".txt")
    ):

        return {
            "success": False,
            "error": "Only PDF and TXT files are allowed."
        }

    try:

        # Read file
        content = await file.read()

        # Extract text
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

        if not text or not text.strip():

            return {
                "success": False,
                "error": "Could not extract readable text from the document."
            }

        # Process document immediately
        # This ensures embeddings are ready
        # before user asks a question.
        process_document(text)

        return {
            "success": True,
            "message": "Document uploaded and processed successfully.",
            "filename": file.filename,
            "chunks": len(chunks),
            "status": "ready"
        }

    except Exception as e:

        print(
            "Upload error:",
            str(e)
        )

        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
def ask_question(
    req: Question
):

    global chunks

    question = req.question.strip()

    # Empty question
    if not question:

        return {
            "success": False,
            "error": "Please enter a question."
        }

    # No document
    if not chunks:

        return {
            "success": False,
            "error": "Please upload and process a document first."
        }

    try:

        # Search relevant chunks
        top_chunks = search_similar(
            question,
            top_k=3
        )

        if not top_chunks:

            return {
                "success": False,
                "error": "No relevant information found in the document."
            }

        # Combine retrieved chunks
        context = "\n\n".join(
            top_chunks
        )

        # Current RAG answer
        # This returns the relevant document content.
        answer = (
            "Based on your document:\n\n"
            + context[:2000]
        )

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "context_used": top_chunks
        }

    except Exception as e:

        print(
            "Question error:",
            str(e)
        )

        return {
            "success": False,
            "error": str(e)
        }