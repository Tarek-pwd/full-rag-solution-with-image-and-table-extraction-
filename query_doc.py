import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
FAISS_INDEX_PATH = "document_faiss.index"
CHUNKS_PKL_PATH = "document_chunks.pkl"
TOP_K = 15

# =========================
# LOAD MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# LOAD FAISS INDEX
# =========================


# =========================
# LOAD CHUNKS
# =========================



# =========================
# USER QUERY
# =========================
index = None
chunks = None
def RAG_query(query):
    global index,chunks
    if index is None or chunks is None: 
        print("LOADING FAISS AND CHUNKS ... ")
        index = faiss.read_index(FAISS_INDEX_PATH)
        print("FAISS index loaded")
        print("Total vectors in index:", index.ntotal)
        with open(CHUNKS_PKL_PATH, "rb") as f:
            chunks = pickle.load(f)
        print("Chunks loaded:", len(chunks))

    # SAFETY CHECK
        assert index.ntotal == len(chunks), "❌ index and chunks count mismatch!"
    # =========================
    # EMBED QUERY
    # =========================
    query_embedding = model.encode(query, convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.reshape(1, -1)

    # =========================
    # SEARCH
    # =========================
    scores, indices = index.search(query_embedding, TOP_K)

    # =========================
    # SHOW RESULTS
    # =========================
    print("\n" + "=" * 90)
    print("RETRIEVED CHUNKS")
    print("=" * 90)

    chunks_arr = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        print(f"\nRANK #{rank}")
        print(f"FAISS IDX : {idx}")
        print(f"SCORE     : {score:.4f}")
        print("-" * 90)
        print(chunks[idx])
        print("-" * 90)
        chunks_arr.append(chunks[idx])

    print("\nDONE.")
    return chunks_arr