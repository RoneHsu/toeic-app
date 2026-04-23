"""Unit tests for quiz_generator.py — JSON parsing, prompt building, and API integration."""

import json
from unittest.mock import MagicMock, patch

import pytest

from models import GenerateRequest, QuestionType, Difficulty
from quiz_generator import (
    _fix_json_strings,
    _parse_json_response,
    _build_user_prompt,
    generate_questions,
)

SAMPLE_QUESTION = {
    "part7_subtype": None,
    "passage": "The company ------ its annual report last quarter.",
    "passages": None,
    "question": "請選出最適合填入空格的答案。",
    "choices": [
        {"label": "A", "text": "published"},
        {"label": "B", "text": "publishes"},
        {"label": "C", "text": "publication"},
        {"label": "D", "text": "publicly"},
    ],
    "correct_answer": "A",
    "explanation": "① Past tense needed ② B is present tense ③ C is noun ④ D is adverb",
    "grammar_point": "verb tense",
}

SAMPLE_JSON = json.dumps([SAMPLE_QUESTION])


def make_req(**kwargs) -> GenerateRequest:
    defaults = dict(
        question_type=QuestionType.INCOMPLETE_SENTENCE,
        difficulty=Difficulty.MEDIUM,
        toeic_part=5,
        count=3,
        use_rag=False,
        topic=None,
        part7_subtype=None,
    )
    defaults.update(kwargs)
    return GenerateRequest(**defaults)


class TestFixJsonStrings:
    def test_escapes_raw_newline_in_string(self):
        raw = '{"key": "line1\nline2"}'
        fixed = _fix_json_strings(raw)
        parsed = json.loads(fixed)
        assert parsed["key"] == "line1\nline2"

    def test_leaves_valid_json_unchanged(self):
        raw = '{"key": "value", "num": 42}'
        assert _fix_json_strings(raw) == raw

    def test_escapes_tab_in_string(self):
        raw = '{"key": "col1\tcol2"}'
        fixed = _fix_json_strings(raw)
        parsed = json.loads(fixed)
        assert parsed["key"] == "col1\tcol2"

    def test_preserves_already_escaped_backslash(self):
        raw = '{"key": "path\\\\file"}'
        result = _fix_json_strings(raw)
        assert json.loads(result)["key"] == "path\\file"

    def test_does_not_double_escape_existing_escape(self):
        raw = '{"key": "line1\\nline2"}'
        fixed = _fix_json_strings(raw)
        parsed = json.loads(fixed)
        assert parsed["key"] == "line1\nline2"


class TestParseJsonResponse:
    def test_parses_plain_json_array(self):
        result = _parse_json_response(SAMPLE_JSON)
        assert len(result) == 1
        assert result[0]["correct_answer"] == "A"

    def test_parses_markdown_code_block(self):
        wrapped = f"```json\n{SAMPLE_JSON}\n```"
        result = _parse_json_response(wrapped)
        assert len(result) == 1

    def test_parses_json_embedded_in_text(self):
        text = f"Here are the questions:\n{SAMPLE_JSON}\nEnd."
        result = _parse_json_response(text)
        assert len(result) == 1

    def test_returns_empty_list_for_invalid_input(self):
        result = _parse_json_response("This is definitely not JSON.")
        assert result == []

    def test_handles_raw_newlines_inside_strings(self):
        bad_json = '[{"key": "line1\nline2", "correct_answer": "A"}]'
        result = _parse_json_response(bad_json)
        assert len(result) == 1

    def test_strips_surrounding_whitespace(self):
        result = _parse_json_response(f"  \n{SAMPLE_JSON}\n  ")
        assert len(result) == 1

    def test_returns_list_of_dicts(self):
        result = _parse_json_response(SAMPLE_JSON)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)


class TestBuildUserPrompt:
    def test_part5_prompt_includes_count_plus_one(self):
        req = make_req(toeic_part=5, count=3)
        prompt = _build_user_prompt(req, context=None)
        assert "4" in prompt  # count + 1

    def test_includes_topic_when_provided(self):
        req = make_req(topic="finance")
        prompt = _build_user_prompt(req, context=None)
        assert "finance" in prompt

    def test_includes_rag_context_when_provided(self):
        req = make_req()
        prompt = _build_user_prompt(req, context="TOEIC grammar: present perfect tense")
        assert "present perfect tense" in prompt

    def test_part7_single_calculates_passage_count(self):
        req = make_req(toeic_part=7, part7_subtype="single",
                       question_type=QuestionType.READING, count=5)
        prompt = _build_user_prompt(req, context=None)
        assert "2" in prompt  # round(5 / 2.5) == 2 passages

    def test_part7_double_mentions_five_questions(self):
        req = make_req(toeic_part=7, part7_subtype="double",
                       question_type=QuestionType.READING, count=5)
        prompt = _build_user_prompt(req, context=None)
        assert "5 題" in prompt

    def test_part7_triple_mentions_five_questions(self):
        req = make_req(toeic_part=7, part7_subtype="triple",
                       question_type=QuestionType.READING, count=5)
        prompt = _build_user_prompt(req, context=None)
        assert "5 題" in prompt

    def test_no_context_adds_default_scenario_hint(self):
        req = make_req()
        prompt = _build_user_prompt(req, context=None)
        assert "商務情境" in prompt


class TestGenerateQuestions:
    def _mock_api(self, content: str):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = content
        return mock_response

    def test_returns_quiz_questions(self):
        with patch("quiz_generator._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = self._mock_api(SAMPLE_JSON)
            result = generate_questions(make_req(count=1))

        assert len(result) == 1
        assert result[0].correct_answer == "A"
        assert result[0].toeic_part == 5

    def test_truncates_to_requested_count(self):
        two_items = json.dumps([SAMPLE_QUESTION, SAMPLE_QUESTION])
        with patch("quiz_generator._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = self._mock_api(two_items)
            result = generate_questions(make_req(count=1))

        assert len(result) == 1

    def test_raises_runtime_error_on_api_failure(self):
        with patch("quiz_generator._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = Exception("timeout")
            with pytest.raises(RuntimeError, match="API 呼叫失敗"):
                generate_questions(make_req(count=1))

    def test_skips_malformed_questions_gracefully(self):
        bad_items = json.dumps([
            SAMPLE_QUESTION,
            {"broken": "no required fields"},
        ])
        with patch("quiz_generator._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = self._mock_api(bad_items)
            result = generate_questions(make_req(count=2))

        assert len(result) == 1  # only the valid one

    def test_deduplicates_passages_for_multi_doc(self):
        passages = ["Document 1 full text.", "Document 2 full text."]
        q_with_passages = {**SAMPLE_QUESTION, "passages": passages, "part7_subtype": "double"}
        q_duplicate = {**SAMPLE_QUESTION, "passages": passages, "part7_subtype": "double"}
        two_items = json.dumps([q_with_passages, q_duplicate])

        with patch("quiz_generator._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = self._mock_api(two_items)
            result = generate_questions(make_req(count=2))

        assert result[0].passages == passages
        assert result[1].passages is None  # duplicate cleared
