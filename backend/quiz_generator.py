"""
TOEIC 題目生成器
使用 Google Gemini + RAG 自動產生貼近真實 TOEIC 的題目與詳解。
"""

import os
import json
import uuid
import logging
import re
from typing import Optional

from openai import OpenAI
from models import (
    QuizQuestion, GenerateRequest, QuestionType, Difficulty, Choice
)
from rag import retrieve_context

logger = logging.getLogger(__name__)

_openrouter_client = None

def _get_client():
    global _openrouter_client
    if _openrouter_client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        _openrouter_client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_client

SYSTEM_PROMPT = """你是一位專業的 TOEIC 閱讀測驗命題專家，出題風格完全對標《多益閱讀模測解密》系列，符合真實 ETS TOEIC 考試規格。

【TOEIC 閱讀測驗結構】
- Part 5（Q101–130）：不完整句子，30 題，測試文法與單字
- Part 6（Q131–146）：短文填空，16 題（4 篇文章 × 4 題），測試段落脈絡理解
- Part 7（Q147–200）：閱讀理解，54 題，包含單篇（29題）、雙篇（10題）、三篇（15題）

【出題規則】
Part 5：
- 商務情境的完整句子，空格位置測試：詞性（名詞/動詞/形容詞/副詞）、時態、語態、連接詞、介係詞、代名詞
- 四個選項通常為同一字根的不同詞性（如：analysis / analyze / analytical / analytically）
- 干擾選項需具高度迷惑性
- passage 欄位放含空格（_____）的完整句子，空格用五個底線 _____ 標示，passages 為 null
- question 欄位只寫「請選出最適合填入空格的答案。」

Part 6：
- 先寫一篇完整商務文章（e-mail、公告、通知、廣告等，約 150–200 字）
- 文章中嵌入 4 個空格，標示為 ______（1）、______（2）、______（3）、______（4）
- 第 4 題考「句子插入」（選哪個完整句子最適合填入）
- 輸出 4 道題目，每題的 passage 欄位放【相同的完整文章原文】，question 只寫「請選出最適合填入空格（N）的答案。」
- passages 為 null，part7_subtype 為 null

Part 7 單篇（single）：
- 先寫一篇完整商務文件，根據主題選擇下列格式之一，約 200–300 字：
  ① 信件／電子郵件（Letter / E-mail）
  ② 廣告／促銷通知（Advertisement）
  ③ 公告／備忘錄（Notice / Memo）
  ④ 新聞稿／文章（Article / Press release）
  ⑤ 時刻表／價目表（Schedule / Price list）
  ⑥ 表格（Form）：如報名表、問卷、回饋表，包含姓名欄、選項欄、備註欄等結構
  ⑦ 產品評論（Product review）：含規格、評分項目、總評
  ⑧ 線上聊天室（Online chat discussion）：多位參與者、含時間戳記，格式完全仿照真實 TOEIC，如：
     Marcel Deprez  [11:01 A.M.]
     Hi Babette, have you had a chance to proof the Japanese translation of the manual?

     Babette Mars  [11:04 A.M.]
     Not yet. I've been busy arranging the details of the Tokyo launch event.

     Marcel Deprez  [11:07 A.M.]
     I'm afraid that's not going to work. I think I made a mistake when I told you the deadline.
  （聊天室至少 4 則訊息，2–3 位參與者，內容需有情境轉折）
  ⑨ 手機簡訊鏈（Text-message chain）：兩人往返的手機訊息，含時間戳記，格式如：
     Bill Visconti  11:50 A.M.
     Hi. Where's the meeting? I'm outside the conference room, but no one else is here.

     Terri Patel  11:54 A.M.
     Hi. It's been pushed back until 3:00. I e-mailed everyone about it this morning.

     Bill Visconti  11:57 A.M.
     I didn't get anything. Are you sure you included me on the list of recipients?

     Terri Patel  11:59 A.M.
     Fairly sure. I'll double check as soon as I get back to my desk.
  （簡訊鏈至少 4 則訊息，2 人對話，內容需有情境轉折）
- 根據文章隨機出 2 或 3 道問題（每次隨機，不固定），題型包含：主旨、細節、推論、同義字、NOT/EXCEPT、說話者意圖（What does X mean when he/she writes "..."?）
- 每題 passage 欄位放【相同的完整文章原文】，passages 為 null，part7_subtype 填 "single"

Part 7 雙篇（double）：
- 先寫兩份相關但不同的商務文件，從以下組合中選擇一種：
  · 求職信（200字）+ 推薦信（250字）：信件含日期、地址、稱謂、4–5段正文、結語、簽名
  · 投訴信（200字）+ 官方回覆信（200字）：各含事件描述、爭議細節、解決方案
  · 宣傳手冊/廣告（150字）+ 詢問信（200字）：手冊含活動說明、費用、報名方式
  · Email（200字）+ 預算/估價表格：表格含 Manufacturer/Item/Unit Price/Quantity/Total 等欄位，數字真實具體
  · 行程公告信（200字，含項目日期與說明）+ 報名 Email（150字）
  · 組織網頁介紹（200字）+ 活動行程表（含 Date/Time/Place/Description 欄位，至少 4 個活動）
  · 行銷提案（含 Background/Concept/Deadlines 小標）+ 通知備忘錄（含截止日期清單）
  · 會議記錄（含 Present/各議題小標/決議）+ 感謝 Email（150字）
  · 客戶滿意度問卷（含姓名、分支、5道選擇題、手寫意見）+ 分析報告 Email（200字）
- 每份文件長度：200–260 個英文單字（表格/問卷除外，但欄位需完整且數字真實）
- 正式信件必須包含：日期、寄件人地址（可選）、收件人姓名/職稱、稱謂（Dear ...）、4–5段完整正文、結語（Sincerely/Best regards）、署名
- Email 必須包含：From/To/Subject/Date 標頭欄（表格格式）、稱謂、3–4段正文、結語、署名
- 出 5 道題目：前 2 題各自對應一份文件，後 3 題需交叉比對兩份文件才能回答
- 每題 passage 留空字串，passages 欄位放 ["文件1完整原文", "文件2完整原文"]，part7_subtype 填 "double"
- 同一組 5 題的 passages 陣列內容必須完全相同

Part 7 三篇（triple）：
- 先寫三份相關的商務文件，從以下組合中選擇一種：
  · 廣告/手冊（150字）+ 客戶 Q&A 論壇（4–5則留言，各含 username 與日期）+ 文章/新聞稿（200字）
  · 頒獎典禮議程（含時間/活動/講者欄位，至少 6 個項目）+ Email 感謝信（200字）+ 活動網頁（150字）
  · 邀請函（200字）+ 研討會時刻表（含 Session/Speaker/Time/Room 欄位，至少 5 場）+ 確認 Email（180字）
  · 新聞稿（200字）+ 訂閱價格表格（含 Plan/Monthly/Yearly 欄位，至少 3 方案）+ Email（150字）
  · 培訓手冊（200字）+ 報名表（含姓名/公司/票種選擇表/付款方式）+ 正式邀請信（200字）
  · 行銷提案（200字）+ 截止日期備忘錄（清單格式）+ 確認 Email（150字）
- 每份文件長度：180–250 個英文單字（表格/問卷需有真實完整的欄位與數字）
- 出 5 道題目：5 題全部都基於這三份文件（第1-2題考單一文件細節，第3-4題需比對兩份文件，第5題需交叉比對三份文件），題型含細節、推論、同義字、NOT題
- 每題 passage 留空字串，passages 欄位放 ["文件1完整原文", "文件2完整原文", "文件3完整原文"]，part7_subtype 填 "triple"
- 同一組 5 題的 passages 陣列內容必須完全相同

【表格格式規則】
凡文件中出現表格資料（價格表、時刻表、預算表、報名表等），一律使用 Markdown 管道表格：
| 欄位1 | 欄位2 | 欄位3 |
|---|---|---|
| 資料 | 資料 | 資料 |
禁止使用空格或 --- 作為欄位分隔，也不可用純文字列出。

【解析格式】
① 正確答案理由（文法/語意分析）
② 錯誤選項逐一說明
③ 關鍵文法點或單字用法

回應格式：嚴格使用以下 JSON 陣列，不得包含任何其他文字：
[
  {
    "part7_subtype": null,
    "passage": "文章原文（Part 5 填含空格_____的句子；Part 6/Part 7 單篇填完整文章；Part 7 雙篇/三篇留空字串）",
    "passages": null,
    "question": "題目",
    "choices": [
      {"label": "A", "text": "選項內容"},
      {"label": "B", "text": "選項內容"},
      {"label": "C", "text": "選項內容"},
      {"label": "D", "text": "選項內容"}
    ],
    "correct_answer": "A",
    "explanation": "詳細解析（中文）",
    "grammar_point": "文法重點標籤"
  }
]"""


def _build_user_prompt(req: GenerateRequest, context: Optional[str]) -> str:
    type_desc = {
        QuestionType.GRAMMAR: "文法題（測試時態、語態、詞性等文法概念）",
        QuestionType.VOCABULARY: "單字題（測試商用英文詞彙與片語）",
        QuestionType.READING: "閱讀理解題（根據短文回答問題）",
        QuestionType.INCOMPLETE_SENTENCE: "不完整句子題（TOEIC Part 5 格式）",
        QuestionType.TEXT_COMPLETION: "短文填空題（TOEIC Part 6 格式）",
    }

    difficulty_desc = {
        Difficulty.EASY: "簡單（TOEIC 500分以下程度）",
        Difficulty.MEDIUM: "中等（TOEIC 600-750分程度）",
        Difficulty.HARD: "困難（TOEIC 800分以上程度）",
    }

    subtype_desc = {
        "single": "單篇閱讀（一篇文章配 2–4 題）",
        "double": "雙篇閱讀（兩份相關文件配 5 題，passages 欄位填兩份文件）",
        "triple": "三篇閱讀（三份相關文件配 5 題，passages 欄位填三份文件）",
    }

    # Part 7 單篇：依題數算出篇數（每篇 2–3 題）
    if req.part7_subtype == "single" and req.toeic_part == 7:
        import math
        passage_count = max(1, round(req.count / 2.5))
        prompt_parts = [
            f"請出 {passage_count} 篇不同格式的 TOEIC Part 7 單篇閱讀文章，每篇配 2 或 3 道題目（每篇自行隨機決定），規格如下：",
            f"- 難度：{difficulty_desc.get(req.difficulty, req.difficulty)}",
            f"- 每篇文件格式請從以下隨機選擇，{passage_count} 篇盡量不重複：信件、Email、廣告、公告／備忘錄、新聞稿、時刻表、表格／問卷、產品評論、線上聊天室、手機簡訊鏈",
            f"- passage 欄位放完整文章原文，passages 為 null，part7_subtype 填 \"single\"",
        ]
    else:
        prompt_parts = [
            f"請出 {req.count + 1} 道 TOEIC 題目，規格如下：",
            f"- 題型：{type_desc.get(req.question_type, req.question_type)}",
            f"- 難度：{difficulty_desc.get(req.difficulty, req.difficulty)}",
            f"- Part：TOEIC Part {req.toeic_part}",
        ]

    if req.part7_subtype and req.toeic_part == 7 and req.part7_subtype != "single":
        prompt_parts.append(f"- 文章篇數：{subtype_desc.get(req.part7_subtype, req.part7_subtype)}")
        if req.part7_subtype == "double":
            prompt_parts.append("- 注意：必須輸出 5 題，passages 欄位放包含兩份文件的陣列，passage 留空字串，part7_subtype 填 \"double\"")
            prompt_parts.append("- 每份文件至少 200 個英文單字，正式信件需含日期、稱謂、4–5 段正文、結語、署名，Email 需含 From/To/Subject/Date 標頭")
        elif req.part7_subtype == "triple":
            prompt_parts.append("- 注意：必須輸出 5 題，passages 欄位放包含三份文件的陣列，passage 留空字串，part7_subtype 填 \"triple\"")
            prompt_parts.append("- 每份文件至少 180 個英文單字，表格類需有完整欄位與真實數字，不可省略")

    if req.topic:
        prompt_parts.append(f"- 主題：{req.topic}（商務情境）")

    if context:
        prompt_parts.append(
            f"\n以下是來自學員講義的相關內容，請優先根據此內容出題：\n\n"
            f"```\n{context}\n```"
        )
    else:
        prompt_parts.append(
            "\n請根據常見 TOEIC 商務情境出題（會議、信件、差旅、人事、財務等）。"
        )

    prompt_parts.append("\n請嚴格按照 JSON 格式回應，不要包含任何說明文字。")
    return "\n".join(prompt_parts)


def generate_questions(req: GenerateRequest) -> list[QuizQuestion]:
    import time
    _t0 = time.time()
    # 1. RAG 檢索
    context = None
    if req.use_rag:
        rag_query = f"TOEIC {req.question_type.value} {req.topic or ''}"
        context = retrieve_context(rag_query, top_k=3)
        if context:
            logger.info(f"RAG 找到相關段落，長度: {len(context)} 字元")

    # 2. 組合 Prompt
    user_prompt = _build_user_prompt(req, context)

    # 3. 呼叫 OpenRouter（依序嘗試，遇到限流自動換下一個）
    # 快速模型優先，大模型作備援
    _MODELS = [
        "stepfun/step-3.5-flash:free",
        "google/gemma-3-27b-it:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
    ]
    # 動態計算所需 token（每題約 400 token，加 buffer）
    _max_tokens = min(req.count * 500 + 1000, 8192)
    last_err = None
    full_text = None
    for model in _MODELS:
        try:
            response = _get_client().chat.completions.create(
                model=model,
                max_tokens=_max_tokens,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            full_text = response.choices[0].message.content
            logger.info(f"使用模型: {model}")
            break
        except Exception as e:
            logger.warning(f"模型 {model} 失敗: {e}")
            last_err = e
    if full_text is None:
        raise last_err

    # 4. 解析 JSON
    questions_raw = _parse_json_response(full_text)

    # 4.5 去重複 passages（同一組雙篇/三篇只有第 1 題需要傳 passages，節省 Token）
    prev_passages = None
    for q in questions_raw:
        raw_p = q.get("passages")
        if isinstance(raw_p, list) and len(raw_p) > 0:
            if raw_p == prev_passages:
                q["passages"] = None   # 重複，清空
            else:
                prev_passages = raw_p  # 記住這組 passages
        else:
            q["passages"] = None

    # 5. 轉換為 Pydantic 模型
    result = []
    for i, q in enumerate(questions_raw):
        try:
            raw_passages = q.get("passages")
            passages = raw_passages if isinstance(raw_passages, list) and len(raw_passages) > 0 else None
            subtype = q.get("part7_subtype") or req.part7_subtype or None
            question = QuizQuestion(
                id=str(uuid.uuid4()),
                question_type=req.question_type,
                difficulty=req.difficulty,
                toeic_part=req.toeic_part,
                part7_subtype=subtype,
                passage=q.get("passage") or None,
                passages=passages,
                question=q["question"],
                choices=[Choice(**c) for c in q["choices"]],
                correct_answer=q["correct_answer"],
                explanation=q["explanation"],
                grammar_point=q.get("grammar_point"),
                source_context=context[:200] if context else None,
            )
            result.append(question)
        except Exception as e:
            logger.warning(f"跳過第 {i+1} 題（解析失敗）: {e}")

    # 裁切到請求數量（buffer 多一題以防 LLM 少生成）
    result = result[:req.count]
    logger.info(f"成功生成 {len(result)} 道題目（耗時 {time.time() - _t0:.1f}s）")
    return result


def _fix_json_strings(text: str) -> str:
    """將 JSON 字串值內的原始換行符轉義，修復 LLM 常見的 JSON 格式錯誤。"""
    result = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\\' and in_string:
            result.append(c)
            i += 1
            if i < len(text):
                result.append(text[i])
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
        elif in_string and c == '\n':
            result.append('\\n')
        elif in_string and c == '\r':
            result.append('\\r')
        elif in_string and c == '\t':
            result.append('\\t')
        else:
            result.append(c)
        i += 1
    return ''.join(result)


def _parse_json_response(text: str) -> list[dict]:
    text = text.strip()

    def try_parse(s: str):
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        try:
            data = json.loads(_fix_json_strings(s))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return None

    # 直接解析
    result = try_parse(text)
    if result is not None:
        return result

    # 從 markdown code block 提取（greedy，取最後一個 ``` 避免 passages 內容截斷）
    json_match = re.search(r"```(?:json)?\s*([\s\S]*)```", text)
    if json_match:
        result = try_parse(json_match.group(1).strip())
        if result is not None:
            return result

    # 找最外層 [ ... ]
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        result = try_parse(text[start:end])
        if result is not None:
            return result

    logger.error(f"無法解析 JSON 回應:\n{text[:500]}")
    return []
