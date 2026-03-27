from pydantic import BaseModel
from typing import Optional
from enum import Enum


class QuestionType(str, Enum):
    GRAMMAR = "grammar"          # 文法題
    VOCABULARY = "vocabulary"    # 單字題
    READING = "reading"          # 閱讀題
    INCOMPLETE_SENTENCE = "incomplete_sentence"  # 不完整句子（Part 5）
    TEXT_COMPLETION = "text_completion"          # 短文填空（Part 6）


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Choice(BaseModel):
    label: str       # A, B, C, D
    text: str        # 選項內容


class QuizQuestion(BaseModel):
    id: str
    question_type: QuestionType
    difficulty: Difficulty
    toeic_part: int                  # TOEIC Part 5 / 6 / 7
    passage: Optional[str] = None    # 文章原文（Part 6/7 專用，同組題目共用）
    question: str                    # 題目
    choices: list[Choice]            # 四個選項
    correct_answer: str              # 正確答案 (A/B/C/D)
    explanation: str                 # 詳細解析
    grammar_point: Optional[str] = None   # 文法重點（如：現在完成式）
    source_context: Optional[str] = None  # 來源講義段落


class GenerateRequest(BaseModel):
    question_type: QuestionType = QuestionType.GRAMMAR
    difficulty: Difficulty = Difficulty.MEDIUM
    toeic_part: int = 5
    count: int = 5                   # 一次產生幾題
    topic: Optional[str] = None      # 指定主題（如：商務信件、會議）
    use_rag: bool = True             # 是否使用講義 RAG


class GenerateResponse(BaseModel):
    questions: list[QuizQuestion]
    total: int
    source_used: bool                # 是否使用了講義內容


class IngestRequest(BaseModel):
    file_path: str                   # 講義檔案路徑


class IngestResponse(BaseModel):
    success: bool
    message: str
    chunks_indexed: int
