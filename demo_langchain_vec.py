#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
# Run before import HF transformers.
os.environ["HF_HOME"] = "hf_cache"
import asyncio
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from qdrant_client.models import Distance, VectorParams, Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL = "http://qdrant:6333"

# In[13]:


app = FastAPI(title="langchain vec backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allowed origins
    allow_credentials=True,         # Allow cookies/auth headers
    allow_methods=["*"],            # Allow all HTTP methods
    allow_headers=["*"],            # Allow all HTTP headers
)

embed_model: HuggingFaceEmbeddings
vector_store: QdrantVectorStore


# In[4]:


@app.post("/upload")
async def upload(file: UploadFile):
    global vector_store
    texts = chunk_pdf(file.filename)
    _ = vector_store.add_texts(texts)

@app.get("/search")
async def search(sentence: Optional[str] = None):
    global vector_store
    if sentence:
        results: list[Document] = await vector_store.asimilarity_search(sentence)
        sentences = list(map(lambda d: d.page_content, results))
        return {"sentences": sentences}
    else:
        return {"sentences": []}

@app.on_event("startup")
def init():
    load_embedding_model()
    init_vector_store()

# In[6]:


def load_embedding_model():
    global embed_model
    print("Loading embedding model:", EMBED_MODEL_NAME, flush=True)
    embed_model = HuggingFaceEmbeddings(model=EMBED_MODEL_NAME)

def chunk_pdf(filename):
    doc = pymupdf.open(filename)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=3,
        length_function=len
    )

    texts = ""
    for page in doc:
        texts += " " + page.get_text()

    texts = text_splitter.split_text(texts)
    return texts

def init_vector_store():
    global embed_model, vector_store, QDRANT_URL
    qdrant = QdrantClient(QDRANT_URL)
    if not qdrant.collection_exists("test"):
        qdrant.create_collection(
            collection_name="test",
            vectors_config=VectorParams(
                size=len(embed_model.embed_query('')),
                distance=Distance.COSINE)
        )
    vector_store = QdrantVectorStore(
        client=qdrant,
        collection_name="test",
        embedding=embed_model,
    )

