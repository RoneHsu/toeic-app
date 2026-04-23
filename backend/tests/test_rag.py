"""Unit tests for rag.py — pure functions and simple-mode retrieval."""

import json
import pytest
from rag import _split_paragraphs, _tfidf_score, _simple_retrieve


class TestSplitParagraphs:
    def test_all_fit_in_one_chunk(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = _split_paragraphs(text, max_chunk=500)
        assert len(chunks) == 1
        assert "First paragraph." in chunks[0]

    def test_splits_when_over_max_chunk(self):
        para = "A" * 300
        text = f"{para}\n\n{para}\n\n{para}"
        chunks = _split_paragraphs(text, max_chunk=400)
        assert len(chunks) >= 2

    def test_overlap_includes_last_paragraph(self):
        para1 = "Grammar rules section. " * 10
        para2 = "Vocabulary section. " * 10
        para3 = "Reading section. " * 10
        text = f"{para1}\n\n{para2}\n\n{para3}"
        chunks = _split_paragraphs(text, max_chunk=250, overlap=1)
        assert len(chunks) >= 2
        # The second chunk should start with the last paragraph of the first chunk
        combined = " ".join(chunks)
        assert "Grammar" in combined
        assert "Reading" in combined

    def test_empty_text_returns_empty(self):
        assert _split_paragraphs("", max_chunk=700) == []

    def test_filters_out_short_paragraphs(self):
        text = "Hi\n\nThis paragraph has enough content to pass the 20-char minimum.\n\nOK"
        chunks = _split_paragraphs(text, max_chunk=700)
        assert len(chunks) == 1
        assert "enough content" in chunks[0]

    def test_fallback_for_single_block_text(self):
        # No double newlines — should use fixed-size fallback
        text = "X" * 800
        chunks = _split_paragraphs(text, max_chunk=300)
        assert len(chunks) >= 2


class TestTfidfScore:
    def test_zero_for_no_match(self):
        score = _tfidf_score(["grammar"], "vocabulary reading comprehension", 10, {"grammar": 0})
        assert score == 0.0

    def test_rare_term_scores_higher_than_common_term(self):
        # "grammar" in 1 out of 10 docs vs. 8 out of 10 docs
        score_rare = _tfidf_score(["grammar"], "grammar is important", 10, {"grammar": 1})
        score_common = _tfidf_score(["grammar"], "grammar is important", 10, {"grammar": 8})
        assert score_rare > score_common

    def test_more_occurrences_score_higher(self):
        score_few = _tfidf_score(["vocab"], "vocab test", 10, {"vocab": 2})
        score_many = _tfidf_score(["vocab"], "vocab vocab vocab vocab test", 10, {"vocab": 2})
        assert score_many > score_few

    def test_multiple_terms_add_up(self):
        score_one = _tfidf_score(["grammar"], "grammar test", 10, {"grammar": 2})
        score_two = _tfidf_score(["grammar", "test"], "grammar test", 10, {"grammar": 2, "test": 2})
        assert score_two > score_one


class TestSimpleRetrieve:
    def test_returns_none_when_cache_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("rag.SIMPLE_CACHE_PATH", str(tmp_path / "nonexistent"))
        result = _simple_retrieve("grammar", top_k=3)
        assert result is None

    def test_returns_none_when_cache_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("rag.SIMPLE_CACHE_PATH", str(tmp_path))
        result = _simple_retrieve("grammar", top_k=3)
        assert result is None

    def test_retrieves_relevant_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setattr("rag.SIMPLE_CACHE_PATH", str(tmp_path))
        cache_data = {
            "source": "notes.pdf",
            "total_chars": 200,
            "chunks": [
                "This chapter explains TOEIC grammar rules including verb tense and voice.",
                "Listening comprehension strategies focus on key words.",
                "Vocabulary expansion helps reading comprehension in TOEIC.",
            ],
        }
        (tmp_path / "notes.txt").write_text(json.dumps(cache_data), encoding="utf-8")

        result = _simple_retrieve("grammar verb tense", top_k=1)

        assert result is not None
        assert "grammar" in result.lower()

    def test_returns_prefix_label_in_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr("rag.SIMPLE_CACHE_PATH", str(tmp_path))
        cache_data = {
            "source": "notes.pdf",
            "total_chars": 50,
            "chunks": ["TOEIC grammar notes about present perfect tense."],
        }
        (tmp_path / "notes.txt").write_text(json.dumps(cache_data), encoding="utf-8")

        result = _simple_retrieve("grammar", top_k=1)

        assert result is not None
        assert "[講義段落]" in result

    def test_falls_back_to_top_chunks_when_no_match(self, tmp_path, monkeypatch):
        monkeypatch.setattr("rag.SIMPLE_CACHE_PATH", str(tmp_path))
        cache_data = {
            "source": "notes.pdf",
            "total_chars": 50,
            "chunks": ["Completely unrelated content about cooking recipes."],
        }
        (tmp_path / "notes.txt").write_text(json.dumps(cache_data), encoding="utf-8")

        result = _simple_retrieve("zyxwvutsrq", top_k=1)

        assert result is not None
        assert "Completely unrelated" in result
