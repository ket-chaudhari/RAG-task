# RAG-Based Question Answering System

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) based Question Answering system built using FastAPI. It allows users to upload documents (PDF/TXT) and ask questions based on the uploaded content. The system retrieves relevant information using embeddings and returns context-aware answers.

---

## Features
- Upload PDF and TXT documents
- Text preprocessing and chunking
- Embedding generation
- Vector storage using FAISS
- Semantic similarity search
- Background document processing
- REST API using FastAPI
- Basic rate limiting
- Context-based question answering

---

## System Architecture
Upload Document → Preprocessing → Chunking → Embedding Generation → FAISS Vector Store → Similarity Search → Context Retrieval → Answer Generation

---

## Installation

### Clone repository
git clone https://github.com/your-username/rag-project.git
cd rag-project

### Create virtual environment
python -m venv venv
venv\Scripts\activate

### Install dependencies
pip install fastapi uvicorn openai numpy faiss-cpu pydantic

---

## Run Project
python -m uvicorn main:app --reload

Open in browser:
http://127.0.0.1:8000/docs

---

## API Endpoints

### Upload Document
POST /upload

Input: PDF or TXT file  
Output: Processing started message

---

### Ask Question
POST /ask

Example:
{
  "question": "What is networking?"
}

---

## Metrics
- Latency: Measures response time from query to answer generation
- Similarity Score: Used in FAISS to retrieve relevant chunks

---

## Chunking Strategy
Chunk size of 200–500 words is used to balance semantic meaning and retrieval accuracy. Smaller chunks lose context while larger chunks reduce precision.

---

## Retrieval Failure Case
Sometimes irrelevant or partially related chunks may be retrieved due to semantic similarity overlap in embeddings.

---

## Tech Stack
- FastAPI
- Python
- FAISS
- OpenAI (optional)
- NumPy

---

## Author
Internship Project - RAG Based Question Answering System