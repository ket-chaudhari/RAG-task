from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

index = None
stored_chunks = []


def create_embeddings(chunks):
    return model.encode(chunks)


def store_embeddings(embeddings, chunks):
    global index, stored_chunks

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    stored_chunks = chunks

    faiss.write_index(index, "vector.index")

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def search_similar(question, top_k=3):
    global index, stored_chunks

    q_emb = model.encode([question])

    D, I = index.search(np.array(q_emb), top_k)

    results = []
    for i in I[0]:
        if i < len(stored_chunks):
            results.append(stored_chunks[i])

    return results