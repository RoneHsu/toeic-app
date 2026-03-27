"""
RAG 模組：將講義文件向量化，供出題時檢索相關段落。
支援 PDF、DOCX、TXT、Markdown 格式。

模式一（立即可用）：Simple PDF — 用 pypdf 直接讀取全文，存成 .txt 快取
模式二（完整 RAG）：LlamaIndex + ChromaDB 向量檢索（需安裝額外套件）
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "toeic_notes"
SIMPLE_CACHE_PATH = "./simple_cache"   # 模式一的快取目錄


# ── 模式一：Simple PDF（不需要 RAG 套件）────────────────────────────────────

def _extract_pdf_text(file_path: Path) -> str:
    """用 pypdf 擷取 PDF 全文。"""
    from pypdf import PdfReader
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def _simple_ingest(file_path: Path) -> dict:
    """
    簡易模式：讀取 PDF/TXT/MD 並存成純文字快取。
    不需要 ChromaDB，馬上可用。
    """
    os.makedirs(SIMPLE_CACHE_PATH, exist_ok=True)
    cache_file = Path(SIMPLE_CACHE_PATH) / f"{file_path.stem}.txt"

    try:
        if file_path.suffix.lower() == ".pdf":
            text = _extract_pdf_text(file_path)
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        if not text.strip():
            return {"success": False, "chunks": 0, "message": "PDF 內容為空（可能是掃描圖片 PDF）"}

        # 按段落分塊，存成 JSON
        chunks = _split_text(text, chunk_size=800, overlap=100)
        cache_data = {
            "source": file_path.name,
            "total_chars": len(text),
            "chunks": chunks,
        }
        cache_file.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")

        logger.info(f"[Simple] 已快取 {file_path.name}，共 {len(chunks)} 個段落")
        return {
            "success": True,
            "chunks": len(chunks),
            "message": f"✓ 成功讀取 {file_path.name}（{len(chunks)} 個段落，共 {len(text)} 字元）",
        }
    except Exception as e:
        logger.error(f"Simple ingest 失敗: {e}")
        return {"success": False, "chunks": 0, "message": str(e)}


def _split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """將長文字切成重疊的段落塊。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _simple_retrieve(query: str, top_k: int = 3) -> Optional[str]:
    """
    簡易模式：用關鍵字比對找最相關的段落（不需要向量資料庫）。
    """
    cache_dir = Path(SIMPLE_CACHE_PATH)
    if not cache_dir.exists():
        return None

    all_chunks = []
    for cache_file in cache_dir.glob("*.txt"):
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            for chunk in data.get("chunks", []):
                all_chunks.append((chunk, data["source"]))
        except Exception:
            continue

    if not all_chunks:
        return None

    # 關鍵字評分
    query_words = set(query.lower().split())
    scored = []
    for chunk, source in all_chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for w in query_words if w in chunk_lower)
        if score > 0:
            scored.append((score, chunk, source))

    if not scored:
        # 無關鍵字命中時，回傳前幾個段落作為通用背景
        top = all_chunks[:top_k]
    else:
        scored.sort(reverse=True)
        top = [(chunk, src) for _, chunk, src in scored[:top_k]]

    contexts = [f"[講義段落]\n{chunk}" for chunk, _ in top]
    return "\n\n".join(contexts)


def _simple_stats() -> dict:
    cache_dir = Path(SIMPLE_CACHE_PATH)
    if not cache_dir.exists():
        return {"available": False, "count": 0, "mode": "simple"}

    total = 0
    for f in cache_dir.glob("*.txt"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            total += len(data.get("chunks", []))
        except Exception:
            pass
    return {"available": total > 0, "count": total, "mode": "simple"}


# ── 模式二：完整 RAG（LlamaIndex + ChromaDB）────────────────────────────────

def _get_vector_storage():
    """延遲載入 RAG 相關套件。"""
    try:
        import chromadb
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import Settings
        from llama_index.core.embeddings import resolve_embed_model

        Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
        Settings.llm = None

        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return {
            "VectorStoreIndex": VectorStoreIndex,
            "SimpleDirectoryReader": SimpleDirectoryReader,
            "storage_context": storage_context,
            "collection": collection,
        }
    except ImportError:
        return None


# ── 公開介面 ─────────────────────────────────────────────────────────────────

def ingest_document(file_path: str) -> dict:
    """
    將講義匯入系統。
    優先嘗試完整 RAG；若套件未安裝則自動降級為 Simple 模式。
    """
    path = Path(file_path)
    if not path.exists():
        return {"success": False, "chunks": 0, "message": f"找不到檔案: {file_path}"}

    # 嘗試完整 RAG
    storage = _get_vector_storage()
    if storage is not None:
        try:
            docs = storage["SimpleDirectoryReader"](input_files=[str(path)]).load_data()
            storage["VectorStoreIndex"].from_documents(
                docs, storage_context=storage["storage_context"], show_progress=True
            )
            return {
                "success": True,
                "chunks": len(docs),
                "message": f"✓ [RAG 模式] 成功索引 {path.name}，共 {len(docs)} 個段落",
            }
        except Exception as e:
            logger.warning(f"RAG 索引失敗，降級為 Simple 模式: {e}")

    # 降級：Simple 模式
    return _simple_ingest(path)


def retrieve_context(query: str, top_k: int = 3) -> Optional[str]:
    """
    檢索與查詢最相關的講義段落。
    優先使用向量 RAG；若不可用則使用關鍵字比對。
    """
    # 嘗試完整 RAG
    storage = _get_vector_storage()
    if storage is not None and storage["collection"].count() > 0:
        try:
            index = storage["VectorStoreIndex"].from_vector_store(
                storage["storage_context"].vector_store
            )
            nodes = index.as_retriever(similarity_top_k=top_k).retrieve(query)
            if nodes:
                return "\n\n".join(
                    f"[段落 {i+1}]\n{n.text.strip()}" for i, n in enumerate(nodes)
                )
        except Exception as e:
            logger.warning(f"RAG 檢索失敗，降級為 Simple: {e}")

    # 降級：Simple 模式
    return _simple_retrieve(query, top_k)


def get_collection_stats() -> dict:
    """回傳目前資料庫狀態。"""
    storage = _get_vector_storage()
    if storage is not None:
        try:
            count = storage["collection"].count()
            if count > 0:
                return {"available": True, "count": count, "mode": "vector_rag"}
        except Exception:
            pass

    return _simple_stats()
