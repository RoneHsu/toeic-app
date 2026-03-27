# 放置講義的目錄

將你的 TOEIC 講義放在這裡，支援格式：
- PDF (.pdf)
- Word (.docx)
- 純文字 (.txt)
- Markdown (.md)

## 如何匯入講義

啟動後端後，呼叫以下 API：

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./docs/my_notes.pdf"}'
```
