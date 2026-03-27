"""一次匯入 docs/ 目錄下的所有 PDF 講義。"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from rag import ingest_document

DOCS_DIR = Path(__file__).parent.parent / "docs"

pdfs = sorted(DOCS_DIR.glob("*.pdf"))
if not pdfs:
    print("找不到任何 PDF，請確認放在 docs/ 目錄下")
    sys.exit(1)

print(f"找到 {len(pdfs)} 份講義，開始匯入...\n")

total_chunks = 0
for i, pdf in enumerate(pdfs, 1):
    print(f"[{i}/{len(pdfs)}] 處理：{pdf.name}")
    result = ingest_document(str(pdf))
    status = "OK" if result["success"] else "FAIL"
    msg = result["message"].encode("ascii", "ignore").decode() or result["message"]
    print(f"  [{status}] chunks={result['chunks']}\n")
    if result["success"]:
        total_chunks += result["chunks"]

print(f"Done! Total chunks indexed: {total_chunks}")
