from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocessing import extract_text, chunk_text
from embedding import create_embeddings, store_embeddings, search_similar

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="Document based Question Answering API"
)

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# STORAGE
# =========================

chunks = []


# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {
        "message": "RAG API is running"
    }


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================
# TEST
# =========================

@app.get("/test")
def test():
    return {
        "message": "Test route working"
    }


# =========================
# DOCUMENT PROCESSING
# =========================

def process_document(text: str):
    global chunks

    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    store_embeddings(
        embeddings,
        chunks
    )


# =========================
# UPLOAD
# =========================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):

    filename = file.filename.lower()

    if not filename.endswith((".pdf", ".txt")):
        return {
            "success": False,
            "error": "Only PDF and TXT files are allowed."
        }

    content = await file.read()

    try:

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
            return {
                "success": False,
                "error": "No readable text found in document."
            }

        background_tasks.add_task(
            process_document,
            text
        )

        return {
            "success": True,
            "message": "Document uploaded successfully.",
            "status": "Processing started."
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# =========================
# QUESTION MODEL
# =========================

class Question(BaseModel):
    question: str


# =========================
# ASK QUESTION
# =========================

@app.post("/ask")
def ask_question(req: Question):

    global chunks

    question = req.question.strip()

    if not question:
        return {
            "success": False,
            "error": "Please enter a question."
        }

    if not chunks:
        return {
            "success": False,
            "error": "No document is ready. Please upload a document first and wait for processing."
        }

    try:

        top_chunks = search_similar(
            question,
            top_k=3
        )

        if not top_chunks:

            return {
                "success": False,
                "error": "No relevant information found in the document."
            }

        context = "\n\n".join(top_chunks)

        # Simple answer using retrieved document context
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
