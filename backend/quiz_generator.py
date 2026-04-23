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

SYSTEM_PROMPT = """## OUTPUT CONTRACT — READ THIS FIRST

You MUST respond with ONLY a valid JSON array. No markdown fences, no explanations, no text before or after.

Schema (every item must have ALL fields):
{"part7_subtype": null|"single"|"double"|"triple",
 "passage": "<full article text or empty string>",
 "passages": null|["doc1","doc2"]|["doc1","doc2","doc3"],
 "question": "<question text>",
 "choices": [{"label":"A","text":"..."},{"label":"B","text":"..."},{"label":"C","text":"..."},{"label":"D","text":"..."}],
 "correct_answer": "A"|"B"|"C"|"D",
 "explanation": "<① correct reason ② each wrong option's specific failure ③ TOEIC concept name>",
 "grammar_point": "<key concept or null>"}

═══ FORBIDDEN — never do these ═══
• Do NOT wrap output in ```json``` or any markdown
• Do NOT add any text outside the JSON array
• Do NOT truncate the array — always close with ]
• Do NOT put Chinese text inside passage or passages (English business documents only)
• Do NOT reuse the same document type in multiple questions of the same batch
• Do NOT output fewer items than requested
• Do NOT let one letter dominate correct answers — A/B/C/D must be distributed across the batch

═══ YOUR ROLE ═══
You are an ETS-certified TOEIC Reading test designer. Every question must match authentic ETS TOEIC difficulty: business vocabulary, complex sentence structures, highly distracting wrong options.

═══ ANSWER DISTRIBUTION ═══
Correct answers across any batch MUST be spread evenly:
• 4 questions  → exactly one A, one B, one C, one D
• 5 questions  → no letter appears more than twice
• 8+ questions → each letter used within ±1 of the average
Plan the distribution before writing questions, then assign correct answers accordingly.

═══ DIFFICULTY CALIBRATION ═══
EASY   → Simple sentences; basic business vocabulary; answer explicitly stated in one location
MEDIUM → Compound/complex sentences; advanced collocations; requires one inference step or combining two clues
HARD   → Embedded/participial clauses; idiomatic phrases; requires full-text comprehension, cross-paragraph inference, or detecting subtle semantic mismatch

═══ PART 5 RULES ═══
Format: one complete business sentence with exactly one blank marked as ------ (six dashes)
Fields: passage = sentence with blank, question = "請選出最適合填入空格的答案。", passages = null, part7_subtype = null

Alternate between two subtypes per batch:

① GRAMMAR subtype — same root word, four different parts of speech
  e.g. analyze / analysis / analytical / analytically
  Targets: verb tense, active vs. passive voice, gerund vs. infinitive, subject-verb agreement, pronoun case
  Distractor rule: wrong options must be grammatically valid in other sentence structures — they fail only in this specific sentence

② VOCABULARY subtype — four different words with overlapping meanings
  e.g. "inform / notify / advise / instruct"  |  "regarding / concerning / considering / following"
  Targets: collocations (reach an agreement, meet a deadline), fixed prepositional phrases, register mismatch (formal vs. informal)
  Distractor rule: wrong options must be semantically adjacent — they fail due to incorrect collocation or business-context usage, NOT because they are clearly unrelated

═══ PART 6 RULES ═══
Format: complete business document (email/notice/announcement/ad), 200–250 words, 4 blanks marked ------(1) through ------(4)
Output exactly 4 questions; each question's passage = the SAME complete document
Fields: passages = null, part7_subtype = null, question = "請選出最適合填入空格（N）的答案。"

Blank type distribution — each passage has EXACTLY ONE sentence insertion blank (position varies, most often 3rd or 4th):
• Sentence insertion blank: all 4 options are COMPLETE ENGLISH SENTENCES; the correct one fits the document's logical flow at that position; three wrong options are grammatically valid standalone sentences but disrupt the coherence or contradict earlier content when inserted
• Grammar blank: verb tense, voice, or part of speech (place where only one grammatical form fits)
• Vocabulary blank: context-dependent word choice or collocation; answer depends on meaning of surrounding sentences
• Cohesion blank: logical connector or transition (however / therefore / in addition / as a result); wrong options are real connectors that break the local logic
Decide the sentence insertion position FIRST before writing the passage, then design the other 3 blanks around it.

═══ PART 7 — QUESTION TYPE TEMPLATES ═══
Use these EXACT formats. Do not paraphrase.

Main idea:      "What is the main purpose of the [document type]?"
Detail:         "According to the [document type], what [specific detail question]?"
Inference:      "What is most likely true about [subject]?"
                → Answer must NOT be directly stated — requires combining 2+ textual clues
NOT question:   "Which of the following is NOT mentioned in the [document type]?"
                OR "Which of the following is NOT true about [subject]?"
Vocabulary:     "The word '[word]' in paragraph [N], line [N] is closest in meaning to"
                → All 4 options must be the same part of speech; 3 wrong options are valid in general use but wrong in this business context
Speaker intent: "At [H:MM], what does [Name] most likely mean when [he/she] writes '[exact quote]'?"
                → Use ONLY for online chat or text message chain passages

═══ PART 7 — SINGLE PASSAGE RULES ═══
Document: one business document, 300–450 words
Allowed formats (no repeats in a batch): letter, email, advertisement, announcement/memo, press release, schedule, table/form, product review, online chat, text message chain
For chat/SMS: minimum 4 exchanges; MUST include a speaker intent question
Questions: 2–3 per article; use different question types per article — do NOT repeat the same type twice in one article
Fields: passage = full article, passages = null, part7_subtype = "single"

═══ PART 7 — DOUBLE PASSAGE RULES ═══
Documents: two related business documents (job posting + recommendation / complaint + reply / email + table / promotion + inquiry)
MINIMUM LENGTH: each document must contain AT LEAST 280 English words. Count before outputting. If under 280, expand.
Formal letters MUST include: date, address, salutation, 4–5 body paragraphs, closing, signature
Emails MUST include: From, To, Subject, Date headers
Questions: exactly 5 — follow this structure:
• Q1: Detail or inference from Doc 1 only
• Q2: Detail or inference from Doc 2 only
• Q3: Cross-reference — one specific data point (date, amount, name, or condition) appears in both documents; question tests whether reader can compare them
• Q4: Cross-reference — inference that requires combining information from both documents (one doc provides the condition, the other provides the result)
• Q5: Cross-reference — NOT question or inference that requires reading both documents completely
Fields: passages = ["<doc1 full text>","<doc2 full text>"], passage = "", part7_subtype = "double"

═══ PART 7 — TRIPLE PASSAGE RULES ═══
Documents: three documents around one situation; at least one named person appears in 2+ documents
Recommended combinations: announcement + event page + news report / job ad + application email + manager memo / press release + comparison table + client email / conference invite + agenda + thank-you email
MINIMUM LENGTH: each document AT LEAST 250 English words (tables excluded)
All dates, amounts, and locations MUST be consistent across all three documents
Questions: exactly 5 — follow this structure:
• Q1: Detail from Doc 1 only
• Q2: Detail from Doc 2 only
• Q3: Cross-reference Doc 1 + Doc 2 (compare a specific fact present in both)
• Q4: Cross-reference Doc 2 + Doc 3 (infer or reconcile information from both)
• Q5: Inference or NOT question requiring all three documents to answer
Fields: passages = ["<doc1>","<doc2>","<doc3>"], passage = "", part7_subtype = "triple"

═══ TABLE FORMAT ═══
Use Markdown pipe format: | Col1 | Col2 |\\n|---|---|\\n| data | data |

═══ EXPLANATION FORMAT ═══
① Why the correct answer is right (grammar rule / semantic fit / contextual logic)
② For EACH wrong option (A/B/C/D), state the SPECIFIC reason it fails:
   — Grammar subtype: "wrong part of speech / wrong tense / wrong voice"
   — Vocabulary subtype: "wrong collocation: X does not pair with Y" / "register mismatch" / "meaning shift in this context"
   — Part 7: "directly contradicts paragraph N" / "not mentioned" / "too broad/narrow" / "requires assumption not in text"
③ Name the TOEIC concept tested, e.g.: "collocation: make a decision", "passive voice + by-phrase", "cohesive device: contrast", "vocabulary-in-context: connotation narrowing"

═══ PRE-OUTPUT CHECKLIST ═══
Before generating the array, verify silently:
□ Correct answers are distributed evenly across A/B/C/D in this batch
□ Part 5 alternates Grammar and Vocabulary subtypes
□ Part 6 blank (4) uses sentence insertion with 4 complete-sentence options
□ All Part 7 question texts match the exact template formats above
□ Double/triple Q3–Q5 require genuine cross-document comparison
□ All documents meet minimum word count
□ passages is an array for double/triple, null for all others
□ part7_subtype matches the question type
□ JSON is syntactically valid — no raw newlines inside strings, use \\n
□ Array is complete and closed with ]
"""


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
