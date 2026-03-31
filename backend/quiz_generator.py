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

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    return _client

SYSTEM_PROMPT = """你是 TOEIC 閱讀測驗命題專家，出題完全對標 ETS TOEIC 規格，英文程度、句型複雜度、選項設計均須符合真實考試水準。

【TOEIC 閱讀結構】
- Part 5（Q101–130）：不完整句子 30 題，測文法與單字
- Part 6（Q131–146）：短文填空 16 題（4 篇 × 4 題）
- Part 7（Q147–200）：閱讀理解 54 題（單篇 29、雙篇 10、三篇 15）

【Part 5 規則】
- 完整商務句子含一個空格 ------（六個破折號）
- 四選項通常為同字根不同詞性（如 analysis/analyze/analytical/analytically），干擾選項高度迷惑
- 測試：詞性、時態、語態、連接詞、介係詞、代名詞
- passage 欄位放含空格句子，question 只寫「請選出最適合填入空格的答案。」

【Part 6 規則】
- 完整商務文章（email/公告/通知/廣告，200–250 字），含 4 個空格標示為 ------(1)~------(4)
- 第 4 題必須考「整句插入」，四選項各為一個完整英文句子
- 輸出 4 題，每題 passage 放【相同完整文章原文】，question 只寫「請選出最適合填入空格（N）的答案。」
- passages 為 null，part7_subtype 為 null

【Part 7 單篇（single）規則】
- 一篇商務文件 300–450 字，格式選一：信件/Email/廣告/公告/備忘錄/新聞稿/時刻表/表格/線上聊天室/手機簡訊鏈
- 線上聊天室格式：「姓名  [時間]\n內容」；手機簡訊：「姓名  時間\n內容」；至少 4 則，有情境轉折
- 出 2–3 道題，題型：主旨/細節/推論/同義字/NOT/說話者意圖
- 每題 passage 放完整文章，passages 為 null，part7_subtype 填 "single"

【Part 7 雙篇（double）規則】
- 兩份相關商務文件（求職+推薦 / 投訴+回覆 / Email+表格 / 宣傳+詢問 等組合）
- 每份 280–380 字；正式信件含日期/地址/稱謂/4–5段/結語/署名；Email 含 From/To/Subject/Date
- 5 道題：Q1-2 各對應一份文件，Q3-5 需交叉比對兩份
- passages 放 ["文件1完整原文", "文件2完整原文"]，passage 留空，part7_subtype 填 "double"

【Part 7 三篇（triple）規則】
- 三份文件圍繞同一情境，至少一位具名人物出現在 2 份以上文件
- 組合選一：公告+活動網頁+新聞報導 / 徵才廣告+應徵Email+主管備忘錄 / 新聞稿+方案比較表+客戶Email / 研討會邀請+議程表+感謝Email
- 每份 250–350 字（表格除外），金額/日期/地點跨文件須一致
- 5 道題：Q1 只需 Doc1、Q2 只需 Doc2、Q3 比對 Doc1+2、Q4 比對 Doc2+3、Q5 交叉三份
- passages 放 ["文件1", "文件2", "文件3"]，passage 留空，part7_subtype 填 "triple"

【表格格式】使用 Markdown 管道格式：| 欄位1 | 欄位2 |\n|---|---|\n| 資料 | 資料 |

【解析格式】① 正確答案理由（文法/語意）② 錯誤選項逐一說明 ③ 關鍵文法點

回應格式：嚴格 JSON 陣列，不含任何其他文字：
[{"part7_subtype":null,"passage":"...","passages":null,"question":"...","choices":[{"label":"A","text":"..."},{"label":"B","text":"..."},{"label":"C","text":"..."},{"label":"D","text":"..."}],"correct_answer":"A","explanation":"...","grammar_point":"..."}]"""


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
            prompt_parts.append("- 每份文件至少 280 個英文單字，正式信件需含日期、稱謂、4–5 段正文、結語、署名，Email 需含 From/To/Subject/Date 標頭")
        elif req.part7_subtype == "triple":
            prompt_parts.append("- 注意：必須輸出 5 題，passages 欄位放包含三份文件的陣列，passage 留空字串，part7_subtype 填 \"triple\"")
            prompt_parts.append("- 每份文件至少 250 個英文單字，表格類需有完整欄位與真實數字，不可省略")

    if req.topic:
        prompt_parts.append(f"- 主題：{req.topic}（商務情境）")

    if context:
        prompt_parts.append(
            f"\n以下是來自學員講義的參考資料，請從中擷取出題主題與考點，"
            f"但所有文章內容必須自行創作為全英文商務文件，禁止將講義內容直接作為文章使用：\n\n"
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
        context = retrieve_context(rag_query, top_k=1)
        if context:
            logger.info(f"RAG 找到相關段落，長度: {len(context)} 字元")

    # 2. 組合 Prompt
    user_prompt = _build_user_prompt(req, context)

    # 3. 呼叫 Gemini API
    _max_tokens = min(req.count * 1200 + 1500, 9000)
    full_text = None
    try:
        response = _get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=_max_tokens,
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        full_text = response.choices[0].message.content
        logger.info("使用模型: llama-3.3-70b-versatile (Groq)")
    except Exception as e:
        raise RuntimeError(f"Gemini API 呼叫失敗: {e}") from e

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
