"""
圖片 OCR 匯入腳本
用 Groq Vision 模型提取圖片中的 TOEIC 題目文字，存入 RAG simple_cache。

使用方式：
    python ingest_images.py "C:/Users/user/Desktop/模測2/模測2_圖片"

分塊策略（rag-architect 建議）：
  - 識別 [TOEIC Part X ...] 標記或 "Questions N-N refer to" 邊界
  - 每個題組（Part 6 一篇文章＋4題、Part 7 一篇文章＋N題）保持在同一 chunk
  - fallback：段落感知切塊，避免 Part 5 題目被截斷
"""

import os
import sys
import json
import re
import base64
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SIMPLE_CACHE_PATH = Path("./simple_cache")

# 結構化 OCR prompt：要求 Vision 模型在輸出中加入可解析的題組標記
OCR_PROMPT = """\
Extract all TOEIC exam content from this image using the section markers below.

MARKERS TO USE (copy exactly):
  [TOEIC Part 5 Grammar]      — for word-form questions (same root, 4 parts of speech)
  [TOEIC Part 5 Vocabulary]   — for meaning/collocation questions (4 different words)
  [TOEIC Part 6 Text Completion] Q131-134  — include the question range
  [TOEIC Part 7 Single Reading] Q147-148   — include the question range
  [TOEIC Part 7 Double Passage] Q181-185   — include the question range
  [TOEIC Part 7 Triple Passage] Q191-195   — include the question range

RULES:
1. Start each section with the appropriate marker on its own line
2. For Part 5: group grammar questions together, vocabulary questions together
3. For Part 6: include the FULL passage text first, then all 4 questions with options
4. For Part 7: include the FULL article/passage first, then all questions with options
5. Preserve question numbers (101., 135., etc.) and option labels (A) (B) (C) (D)
6. Output only the structured content — no explanations

If content type is unclear, use the closest matching marker."""

# 識別 [TOEIC Part X ...] 標記行
_MARKER_RE = re.compile(
    r'^\[TOEIC\s+Part\s+[567][^\]]*\]',
    re.IGNORECASE | re.MULTILINE,
)

# fallback：識別 "Questions NNN-NNN refer to" 邊界
_QSET_RE = re.compile(
    r'(?:^Questions?\s+\d+[\s\-–]+\d+\s+refer|^Questions?\s+\d+\s+and\s+\d+)',
    re.IGNORECASE | re.MULTILINE,
)


def image_to_base64(image_path: Path) -> str:
    """壓縮圖片後轉 base64，避免超過 API 大小限制。"""
    from PIL import Image
    import io
    img = Image.open(image_path).convert("RGB")
    max_size = 1600
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ocr_image(client: OpenAI, image_path: Path) -> str:
    """用 Groq Vision 提取圖片中的結構化 TOEIC 題目文字。"""
    b64 = image_to_base64(image_path)
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }],
    )
    return response.choices[0].message.content.strip()


def chunk_by_question_set(text: str) -> list[str]:
    """
    按題組邊界切塊（rag-architect 建議：document-aware chunking）。

    優先使用 [TOEIC Part X] 標記；若模型未輸出標記，
    退而使用 "Questions N-N refer to" 邊界；最後 fallback 段落感知切塊。
    每個 chunk 保留完整題組（文章 + 對應題目），避免文章和題目被分離。
    """
    boundaries: set[int] = set()

    for m in _MARKER_RE.finditer(text):
        boundaries.add(m.start())
    for m in _QSET_RE.finditer(text):
        boundaries.add(m.start())

    if boundaries:
        sorted_bounds = sorted(boundaries) + [len(text)]
        chunks = []
        for i in range(len(sorted_bounds) - 1):
            chunk = text[sorted_bounds[i]:sorted_bounds[i + 1]].strip()
            if len(chunk) > 50:
                chunks.append(chunk)
        return chunks

    # Fallback：段落感知切塊，每組 ~1200 字，保留 1 段重疊
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 20]
    if not paras:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paras:
        if current_len + len(para) > 1200 and current:
            chunks.append('\n\n'.join(current))
            current = current[-1:]
            current_len = len(current[0])
        current.append(para)
        current_len += len(para)
    if current:
        chunks.append('\n\n'.join(current))
    return chunks


def ingest_folder(client: OpenAI, folder: Path) -> None:
    """處理一個資料夾內所有圖片，存成 simple_cache。"""
    images = (
        sorted(folder.glob("*.png"))
        + sorted(folder.glob("*.jpg"))
        + sorted(folder.glob("*.jpeg"))
    )
    if not images:
        logger.warning(f"{folder.name}: 沒有找到圖片")
        return

    logger.info(f"開始處理 {folder.name}（{len(images)} 張圖片）")
    all_text_parts: list[str] = []

    for i, img_path in enumerate(images):
        logger.info(f"  [{i+1}/{len(images)}] OCR: {img_path.name}")
        try:
            text = ocr_image(client, img_path)
            if text:
                all_text_parts.append(text)
        except Exception as e:
            logger.warning(f"  跳過 {img_path.name}: {e}")

    full_text = "\n\n".join(all_text_parts)
    if not full_text.strip():
        logger.warning(f"{folder.name}: 提取結果為空")
        return

    chunks = chunk_by_question_set(full_text)
    cache_file = SIMPLE_CACHE_PATH / f"{folder.name}.txt"
    cache_data = {
        "source": folder.name,
        "total_chars": len(full_text),
        "chunks": chunks,
    }
    SIMPLE_CACHE_PATH.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")
    logger.info(f"✓ {folder.name} 完成，共 {len(chunks)} 個段落存入 RAG")


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("請設定 GROQ_API_KEY 環境變數")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Users/user/Desktop/模測2/模測2_圖片")
    if not root.exists():
        logger.error(f"找不到資料夾: {root}")
        sys.exit(1)

    subfolders = sorted([f for f in root.iterdir() if f.is_dir()])
    if not subfolders:
        subfolders = [root]

    logger.info(f"找到 {len(subfolders)} 個資料夾")
    for folder in subfolders:
        ingest_folder(client, folder)

    logger.info("=== 全部完成！RAG 已更新 ===")


if __name__ == "__main__":
    main()
