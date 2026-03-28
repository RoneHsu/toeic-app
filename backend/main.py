"""
TOEIC 練習 App — FastAPI 後端
啟動：uvicorn main:app --reload --port 8000
"""

import logging
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from models import GenerateRequest, GenerateResponse, IngestRequest, IngestResponse
from quiz_generator import generate_questions
from rag import ingest_document, get_collection_stats

app = FastAPI(
    title="TOEIC Practice App API",
    description="基於 Claude AI + RAG 的 TOEIC 自動出題系統",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 健康檢查 ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    stats = get_collection_stats()
    return {
        "status": "ok",
        "rag_available": stats["available"],
        "indexed_chunks": stats["count"],
    }


# ── 出題 API ──────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    自動產生 TOEIC 題目。

    - **question_type**: grammar / vocabulary / reading / incomplete_sentence / text_completion
    - **difficulty**: easy / medium / hard
    - **toeic_part**: 5、6 或 7
    - **count**: 產生題數（建議 1-10）
    - **topic**: 指定主題（選填），如 "商務信件"、"會議"
    - **use_rag**: 是否使用講義內容出題（預設 true）
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="請設定 GEMINI_API_KEY 環境變數")

    if req.count < 1 or req.count > 30:
        raise HTTPException(status_code=400, detail="count 必須介於 1 到 30 之間")

    stats = get_collection_stats()
    questions = generate_questions(req)

    if not questions:
        raise HTTPException(status_code=500, detail="題目生成失敗，請稍後再試")

    return GenerateResponse(
        questions=questions,
        total=len(questions),
        source_used=req.use_rag and stats["count"] > 0,
    )


# ── 講義上傳 API ──────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """
    將講義文件（PDF / DOCX / TXT / MD）匯入向量資料庫。
    匯入後，出題時會自動參考講義內容。
    """
    result = ingest_document(req.file_path)
    return IngestResponse(
        success=result["success"],
        message=result["message"],
        chunks_indexed=result["chunks"],
    )


@app.get("/rag/stats")
def rag_stats():
    """查詢向量資料庫狀態。"""
    return get_collection_stats()


# ── 快速測試（不需要 API Key）────────────────────────────────────────────────

@app.get("/sample")
def sample():
    """回傳範例題目（用於前端開發測試）。"""
    return {
        "questions": [
            {
                "id": "sample-001",
                "question_type": "grammar",
                "difficulty": "medium",
                "toeic_part": 5,
                "question": "The new software _____ by the IT department last week.",
                "choices": [
                    {"label": "A", "text": "installed"},
                    {"label": "B", "text": "was installed"},
                    {"label": "C", "text": "has installed"},
                    {"label": "D", "text": "is installing"},
                ],
                "correct_answer": "B",
                "explanation": (
                    "【正確答案：B】\n"
                    "句中有時間副詞 'last week'，表示過去發生的動作。"
                    "主詞 'software' 是被安裝的對象（受詞），因此需使用被動語態。"
                    "過去式被動語態結構為：was/were + 過去分詞。\n\n"
                    "【錯誤選項分析】\n"
                    "A. installed：主動語態，但軟體本身不會「安裝」別的東西\n"
                    "C. has installed：現在完成式主動，時態不符\n"
                    "D. is installing：現在進行式，與 last week 矛盾\n\n"
                    "【文法重點】過去被動語態 (Past Passive Voice)"
                ),
                "grammar_point": "過去被動語態",
            }
        ],
        "total": 1,
        "source_used": False,
    }


# 服務前端靜態檔案（部署用）
_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
