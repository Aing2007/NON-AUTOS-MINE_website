from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader  # ใช้ TextLoader ตรงๆ เสถียรกว่า

import os
import time

ollama_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")

def md_rag(
    source_directory: str,
    db_path: str,
    collection_name: str,
    model_name: str = "bge-m3",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    start = time.time()
    src = Path(source_directory).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    # 1) หาไฟล์ .md ทุกระดับ (รวมซับโฟลเดอร์)
    md_files = sorted(src.rglob("*.md"))
    print(f"[MD] Found {len(md_files)} files under {src}")
    if not md_files:
        raise FileNotFoundError("No .md files found.")

    # 2) โหลดเป็นข้อความตรงๆ (UTF-8) → ไม่พึ่ง unstructured
    docs = []
    for p in md_files:
        docs.extend(TextLoader(str(p), encoding="utf-8").load())
    print(f"[MD] Loaded {len(docs)} raw docs")

    # 3) แบ่งชิ้น
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"[MD] Split into {len(chunks)} chunks")

    # 4) Embedding (ต้องมี ollama serve + ollama pull bge-m3)
    embeddings = OllamaEmbeddings(model=model_name, base_url=ollama_url)

    # 5) สร้าง/อัปเดต Chroma
    if not os.path.exists(db_path):
        print(f"[DB] Creating new DB at {db_path}")
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,            # บางเวอร์ชันอาจชื่อ embedding_function
            persist_directory=db_path,
            collection_name=collection_name,
        )
    else:
        print(f"[DB] Opening DB at {db_path} and upserting chunks")
        vs = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=embeddings,   # ให้เข้ากับ open path
        )
        if chunks:
            vs.add_documents(chunks)

    # 6) นับจำนวนที่ถูกต้อง
    try:
        count = vs._collection.count()
    except Exception:
        count = len(vs.get()["ids"])
    print(f"[DB] Total document chunks in store: {count}")
    print(f"[OK] Processed in {time.time() - start:.2f}s")

    return vs.as_retriever(search_kwargs={"k": 5})
