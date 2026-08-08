from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os


# ============================================================
# EMBEDDING MODEL
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# GLOBAL STORAGE
# ============================================================

index = None
stored_chunks = []


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):

    if not chunks:
        return np.array([])

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings.astype("float32")


# ============================================================
# STORE EMBEDDINGS
# ============================================================

def store_embeddings(embeddings, chunks):

    global index
    global stored_chunks

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings were created.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.asarray(embeddings).astype("float32")
    )

    stored_chunks = list(chunks)

    # Save FAISS index
    faiss.write_index(
        index,
        "vector.index"
    )

    # Save chunks
    with open("chunks.pkl", "wb") as f:
        pickle.dump(
            stored_chunks,
            f
        )

    print(
        f"Stored {len(stored_chunks)} chunks in FAISS."
    )


# ============================================================
# SEARCH SIMILAR
# ============================================================

def search_similar(question, top_k=3):

    global index
    global stored_chunks

    # Check whether FAISS index exists
    if index is None:

        # Try loading saved index
        if os.path.exists("vector.index"):

            index = faiss.read_index(
                "vector.index"
            )

        else:

            raise ValueError(
                "FAISS index is not ready. Please upload and process a document first."
            )

    # Check chunks
    if not stored_chunks:

        if os.path.exists("chunks.pkl"):

            with open(
                "chunks.pkl",
                "rb"
            ) as f:

                stored_chunks = pickle.load(f)

        else:

            raise ValueError(
                "No document chunks found. Please upload a document first."
            )

    # Create question embedding
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = (
        question_embedding.astype("float32")
    )

    # Don't search more items than available
    k = min(
        top_k,
        index.ntotal
    )

    if k == 0:

        raise ValueError(
            "FAISS index is empty. Please upload a document again."
        )

    distances, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for i in indices[0]:

        if (
            i >= 0
            and i < len(stored_chunks)
        ):

            results.append(
                stored_chunks[i]
            )

    return results


# ============================================================
# LOAD EXISTING INDEX
# ============================================================

def load_existing_index():

    global index
    global stored_chunks

    if (
        os.path.exists("vector.index")
        and os.path.exists("chunks.pkl")
    ):

        try:

            index = faiss.read_index(
                "vector.index"
            )

            with open(
                "chunks.pkl",
                "rb"
            ) as f:

                stored_chunks = pickle.load(f)

            print(
                f"Loaded existing index with {len(stored_chunks)} chunks."
            )

        except Exception as e:

            print(
                "Could not load existing index:",
                e
            )

            index = None
            stored_chunks = []


# Load existing data when application starts
load_existing_index()