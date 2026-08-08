from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from pydantic import BaseModel
from collections import defaultdict
import time

from preprocessing import extract_text, chunk_text
from embedding import create_embeddings, store_embeddings, search_similar

# OPTIONAL LLM (OpenAI)
# pip install openai
from openai import OpenAI

app = FastAPI()

# 👉 ADD YOUR API KEY HERE (or leave if not using LLM)
client = OpenAI(api_key="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")

# ---------------- STORAGE ----------------
chunks = []
request_times = defaultdict(list)


# =====================================================
# 📌 RATE LIMITING (basic)
# =====================================================
@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip = request.client.host
    now = time.time()

    request_times[ip] = [t for t in request_times[ip] if now - t < 60]

    if len(request_times[ip]) >= 5:
        return {"error": "Too many requests. Try later."}

    request_times[ip].append(now)
    return await call_next(request)


# =====================================================
# 📌 BACKGROUND PROCESSING
# =====================================================
def process_document(text: str):
    global chunks

    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks)
    store_embeddings(embeddings, chunks)


# =====================================================
# 📌 REQUEST MODEL
# =====================================================
class Question(BaseModel):
    question: str


# =====================================================
# 📌 UPLOAD API (PDF + TXT)
# =====================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):

    content = await file.read()

    if file.filename.endswith(".pdf"):
        text = extract_text(content, file_type="pdf")
    elif file.filename.endswith(".txt"):
        text = content.decode("utf-8")
    else:
        return {"error": "Only PDF and TXT allowed"}

    background_tasks.add_task(process_document, text)

    return {"message": "Upload successful. Processing started."}


# =====================================================
# 📌 LLM FUNCTION (RAG + AI ANSWER)
# =====================================================
def generate_answer(question, context):
    return f"Answer based on document:\n\n{context[:800]}"

    # If no API key → fallback mode
    if "YOUR_API_KEY_HERE" in client.api_key:
        return f"Context-based answer:\n{context[:500]}"

    prompt = f"""
You are a helpful AI assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# =====================================================
# 📌 ASK API (RAG CORE)
# =====================================================
@app.post("/ask")
def ask_question(req: Question):

    # 🛑 STEP 1: check if data ready
    if not chunks or len(chunks) == 0:
        return {"error": "Document not processed yet. Please wait after upload."}

    try:
        # 🔍 STEP 2: retrieval
        top_chunks = search_similar(req.question, top_k=3)

        if not top_chunks:
            return {"error": "No relevant chunks found in document"}

        context = "\n".join(top_chunks)

        # 🤖 STEP 3: LLM answer
        answer = generate_answer(req.question, context)

        return {
            "question": req.question,
            "answer": answer,
            "context_used": top_chunks
        }

    except Exception as e:
        return {
            "error": str(e)
        }
