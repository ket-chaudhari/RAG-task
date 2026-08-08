from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {"message": "RAG API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/test")
def test():
    return {"message": "Test route working"}


@app.post("/upload")
def upload():
    return {"message": "Upload route working"}


@app.post("/ask")
def ask():
    return {"message": "Ask route working"}
