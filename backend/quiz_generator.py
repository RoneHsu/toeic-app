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

from groq import Groq
from models import (
    QuizQuestion, GenerateRequest, QuestionType, Difficulty, Choice
)
from rag import retrieve_context

logger = logging.getLogger(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """你是一位專業的 TOEIC 閱讀測驗命題專家，出題風格完全對標《多益閱讀模測解密》系列，符合真實 ETS TOEIC 考試規格。

【TOEIC 閱讀測驗結構】
- Part 5（Q101–130）：不完整句子，30 題，測試文法與單字
- Part 6（Q131–146）：短文填空，16 題（4 篇文章 × 4 題），測試段落脈絡理解
- Part 7（Q147–200）：閱讀理解，54 題，包含單篇、雙篇、三篇文章

【出題規則】
Part 5：
- 商務情境的完整句子，空格位置測試：詞性（名詞/動詞/形容詞/副詞）、時態、語態、連接詞、介係詞、代名詞
- 四個選項通常為同一字根的不同詞性（如：analysis / analyze / analytical / analytically）
- 干擾選項需具高度迷惑性

Part 6：
- 提供一篇完整的商務文章（e-mail、公告、通知、廣告等，約 150–200 字）
- 文章中有 4 個空格（編號 131–134、135–138 等）
- 第 4 題通常考「句子插入」（選哪個完整句子最適合填入）

Part 7：
- 提供真實商務文件（信件、廣告、時刻表、表格、新聞稿等）
- 問題測試：主旨、細節、推論、同義字、句子插入位置
- 雙篇/三篇文章需有跨篇整合題

【解析格式】
① 正確答案理由（文法/語意分析）
② 錯誤選項逐一說明
③ 關鍵文法點或單字用法

回應格式：嚴格使用以下 JSON 陣列，不得包含任何其他文字：
[
  {
    "question": "完整題目（Part 6/7 需包含文章原文，用\\n分段）",
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

    prompt_parts = [
        f"請出 {req.count} 道 TOEIC 題目，規格如下：",
        f"- 題型：{type_desc.get(req.question_type, req.question_type)}",
        f"- 難度：{difficulty_desc.get(req.difficulty, req.difficulty)}",
        f"- Part：TOEIC Part {req.toeic_part}",
    ]

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
    # 1. RAG 檢索
    context = None
    if req.use_rag:
        rag_query = f"TOEIC {req.question_type.value} {req.topic or ''}"
        context = retrieve_context(rag_query, top_k=3)
        if context:
            logger.info(f"RAG 找到相關段落，長度: {len(context)} 字元")

    # 2. 組合 Prompt
    user_prompt = _build_user_prompt(req, context)

    # 3. 呼叫 Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=8192,
        temperature=0.7,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    full_text = response.choices[0].message.content

    # 4. 解析 JSON
    questions_raw = _parse_json_response(full_text)

    # 5. 轉換為 Pydantic 模型
    result = []
    for i, q in enumerate(questions_raw):
        try:
            question = QuizQuestion(
                id=str(uuid.uuid4()),
                question_type=req.question_type,
                difficulty=req.difficulty,
                toeic_part=req.toeic_part,
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

    logger.info(f"成功生成 {len(result)} 道題目")
    return result


def _parse_json_response(text: str) -> list[dict]:
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        try:
            data = json.loads(json_match.group(1).strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    logger.error(f"無法解析 JSON 回應:\n{text[:500]}")
    return []
