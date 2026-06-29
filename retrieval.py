import os
import pickle
from functools import lru_cache
from huggingface_hub import hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "vector_db")
BM25_PATH = os.path.join(BASE_DIR, "bm25.pkl")

# =========================
# DOWNLOAD FROM HF
# =========================
def download_indexes():
    if not os.path.exists(FAISS_DIR):
        print("Downloading FAISS index...")
        os.makedirs(FAISS_DIR, exist_ok=True)
        for filename in ["index.faiss", "index.pkl"]:
            hf_hub_download(
                repo_id="vanshhh-18/rag-agentic",
                filename=filename,
                repo_type="dataset",
                local_dir=FAISS_DIR
            )
        print("✅ FAISS ready.")

    if not os.path.exists(BM25_PATH):
        print("Downloading BM25 index...")
        hf_hub_download(
            repo_id="vanshhh-18/rag-agentic",
            filename="bm25.pkl",
            repo_type="dataset",
            local_dir=BASE_DIR
        )
        print("✅ BM25 ready.")

download_indexes()

# =========================
# EMBEDDINGS
# =========================
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# =========================
# FAISS LOAD
# =========================
@lru_cache(maxsize=1)
def load_faiss():
    return FAISS.load_local(
        FAISS_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

# =========================
# BM25 LOAD
# =========================
with open(BM25_PATH, "rb") as f:
    bm25, docs = pickle.load(f)

# =========================
# RETRIEVERS
# =========================
def faiss_retriever():
    return load_faiss().as_retriever(search_kwargs={"k": 3})

def hybrid_search(query):
    faiss_docs = faiss_retriever().invoke(query)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    if isinstance(docs[0], str):
        bm25_docs = [Document(page_content=docs[i]) for i in top_idx]
    else:
        bm25_docs = [docs[i] for i in top_idx]

    seen = set()
    final_docs = []
    for d in faiss_docs + bm25_docs:
        if d.page_content not in seen:
            final_docs.append(d)
            seen.add(d.page_content)
    return final_docs[:3]
