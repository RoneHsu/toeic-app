"""
RAG 模組：將講義文件向量化，供出題時檢索相關段落。
支援 PDF、DOCX、TXT、Markdown 格式。

模式一（立即可用）：Simple PDF — 用 pypdf 直接讀取全文，存成 .txt 快取，TF-IDF 關鍵字檢索
模式二（完整 RAG）：LlamaIndex + ChromaDB 向量檢索（需安裝額外套件）
"""

import os
import json
import math
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "toeic_notes"
SIMPLE_CACHE_PATH = "./simple_cache"


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


def _split_paragraphs(text: str, max_chunk: int = 700, overlap: int = 1) -> list[str]:
    """
    段落感知分塊：優先按雙換行分段，將短段落合併至 max_chunk 字元，
    並保留 overlap 個段落作為上下文銜接（比固定字元切割更貼合文章結構）。
    """
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 20]
    if not paras:
        # fallback: 固定長度分塊
        return [
            text[i:i + max_chunk].strip()
            for i in range(0, len(text), max_chunk - 100)
            if text[i:i + max_chunk].strip()
        ]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paras:
        if current_len + len(para) > max_chunk and current:
            chunks.append('\n\n'.join(current))
            current = current[-overlap:]
            current_len = sum(len(p) for p in current)

        # Paragraph alone exceeds max_chunk — sub-split it with fixed-size fallback
        if len(para) > max_chunk:
            if current:
                chunks.append('\n\n'.join(current))
                current = []
                current_len = 0
            for i in range(0, len(para), max_chunk - 100):
                sub = para[i:i + max_chunk].strip()
                if sub:
                    chunks.append(sub)
            continue

        current.append(para)
        current_len += len(para)

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


def _simple_ingest(file_path: Path) -> dict:
    """簡易模式：讀取 PDF/TXT/MD 並存成純文字快取。"""
    os.makedirs(SIMPLE_CACHE_PATH, exist_ok=True)
    cache_file = Path(SIMPLE_CACHE_PATH) / f"{file_path.stem}.txt"

    try:
        if file_path.suffix.lower() == ".pdf":
            text = _extract_pdf_text(file_path)
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        if not text.strip():
            return {"success": False, "chunks": 0, "message": "PDF 內容為空（可能是掃描圖片 PDF）"}

        chunks = _split_paragraphs(text)
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


def _tfidf_score(
    query_terms: list[str],
    chunk: str,
    n_docs: int,
    doc_freqs: dict[str, int],
) -> float:
    """TF-IDF 評分：讓稀有但關鍵的詞彙有更高權重，過濾掉 'TOEIC' 等無鑑別力的常見詞。"""
    chunk_lower = chunk.lower()
    word_count = max(len(chunk_lower.split()), 1)
    score = 0.0
    for term in query_terms:
        tf = chunk_lower.count(term) / word_count
        idf = math.log((n_docs + 1) / (doc_freqs.get(term, 0) + 1)) + 1
        score += tf * idf
    return score


def _simple_retrieve(query: str, top_k: int = 3) -> Optional[str]:
    """
    簡易模式：TF-IDF 評分找最相關段落（不需要向量資料庫）。
    """
    cache_dir = Path(SIMPLE_CACHE_PATH)
    if not cache_dir.exists():
        return None

    all_chunks: list[tuple[str, str]] = []
    for cache_file in cache_dir.glob("*.txt"):
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            for chunk in data.get("chunks", []):
                all_chunks.append((chunk, data["source"]))
        except Exception:
            continue

    if not all_chunks:
        return None

    query_terms = [w for w in query.lower().split() if len(w) > 1]
    if not query_terms:
        return "\n\n".join(f"[講義段落]\n{chunk}" for chunk, _ in all_chunks[:top_k])

    # 預計算每個 query term 的 document frequency
    n_docs = len(all_chunks)
    doc_freqs: dict[str, int] = {
        term: sum(1 for chunk, _ in all_chunks if term in chunk.lower())
        for term in query_terms
    }

    scored = sorted(
        ((_tfidf_score(query_terms, chunk, n_docs, doc_freqs), chunk, source)
         for chunk, source in all_chunks),
        reverse=True,
    )

    # 回傳有得分的結果；全部為 0 時退回前幾個段落
    top = [(chunk, src) for score, chunk, src in scored[:top_k] if score > 0]
    if not top:
        top = [(chunk, src) for _, chunk, src in scored[:top_k]]

    return "\n\n".join(f"[講義段落]\n{chunk}" for chunk, _ in top)


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

    return _simple_ingest(path)


def retrieve_context(query: str, top_k: int = 3) -> Optional[str]:
    """
    檢索與查詢最相關的講義段落。
    優先使用向量 RAG；若不可用則使用 TF-IDF 關鍵字檢索。
    """
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
