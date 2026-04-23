"""Unit tests for ingest_images.py — chunking strategy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest_images import chunk_by_question_set


PART5_TEXT = """\
[TOEIC Part 5 Grammar]
101. The new business class seats offered by Dorcal Air have been reviewed ------ in several publications.
(A) positive (B) positively (C) positivity (D) positives

102. The most ------ studies confirmed that staff turnover has been lowest since establishment.
(A) many (B) soon (C) often (D) recent

[TOEIC Part 5 Vocabulary]
103. Production supervisors will meet ------ review the safety regulations currently being used at the plant.
(A) for (B) at (C) to (D) with
"""

PART6_TEXT = """\
[TOEIC Part 6 Text Completion] Q135-138
Frimp's Department Store ------ a renovation project in order to create a more modern environment.
The construction work is due to start on December 1. [136]. The first stage involves an overhaul
of our renowned basement food hall, which is to be closed for two weeks. Patrons can shop [137]
the online store. Local delivery fees usually charged by the store will be waived until this [138] is completed.

135. (A) has begun (B) is begun (C) will begin (D) was beginning
136. (A) The work will only affect the building's exterior.
     (B) Bids for the project are invited from all contractors.
     (C) To keep disruption to a minimum, the work will be done on one floor at a time.
     (D) All services offered by Frimp's will be suspended throughout the project.
137. (A) through (B) along (C) behind (D) across
138. (A) analysis (B) relocation (C) order (D) phase
"""

PART7_TEXT = """\
[TOEIC Part 7 Single Reading] Q154-155
BELLEVUE (March 28) — The Bellevue Historical Society (BHS) has unveiled plans to renovate
the historic Main Street courthouse. A statement by BHS President Darren Shepherd reads:
"We are very pleased to announce that repairs to the Main Street courthouse will soon be underway."

Built 150 years ago, the Main Street courthouse is a Bellevue historical treasure.
The BHS, in conjunction with the Bellevue City Council, is planning to reopen the courthouse as a tourist site later this year.

154. Why was the article written?
(A) To report on a community project (B) To promote a newly constructed tourist attraction
(C) To solicit volunteers for a fundraising event (D) To explain a road closure

155. What is indicated in the article?
(A) The courthouse was designed by an award-winning architect.
(B) Mr. Shepherd established a construction company.
(C) A history museum will be added to the building.
(D) Two organizations are working together.
"""

DOUBLE_TEXT = """\
[TOEIC Part 7 Double Passage] Q181-185
From: Josh Laporte <jlaporte@rapidmail.com>
To: Tom Mueller <tmueller@argotractors.com>
Subject: My search for work
Date: February 12

Dear Mr. Mueller,
I have just successfully completed my studies in San Jose, and I'm now looking for work again.

---

Argo Tractors
42 Arboreal Avenue, Bismarck, ND 58507
February 29

Dear Mr. Preston,
This letter is to confirm that Josh Laporte was employed as a mechanic at Argo Tractors
from October 2010 until four years ago.

181. Why did Josh Laporte write to Mr. Mueller?
(A) To request a meeting (B) To ask for a job referral
(C) To inquire about a position (D) To thank him for past support
"""

NO_MARKER_TEXT = """\
Some introductory text about the TOEIC test format.

This is a paragraph about reading comprehension and business English vocabulary.
Students should practice regularly to improve their scores.

Another paragraph discussing exam strategies and time management techniques
for the TOEIC Reading section which includes Parts 5, 6, and 7.
"""

MIXED_TEXT = PART5_TEXT + "\n\n" + PART6_TEXT + "\n\n" + PART7_TEXT


class TestChunkByQuestionSet:
    def test_splits_on_toeic_part5_marker(self):
        chunks = chunk_by_question_set(PART5_TEXT)
        assert len(chunks) == 2  # grammar group + vocabulary group
        assert any("[TOEIC Part 5 Grammar]" in c for c in chunks)
        assert any("[TOEIC Part 5 Vocabulary]" in c for c in chunks)

    def test_part6_kept_as_single_chunk(self):
        chunks = chunk_by_question_set(PART6_TEXT)
        assert len(chunks) == 1
        chunk = chunks[0]
        # passage and all 4 questions must be in the same chunk
        assert "Frimp's Department Store" in chunk
        assert "135." in chunk
        assert "138." in chunk

    def test_part7_passage_and_questions_together(self):
        chunks = chunk_by_question_set(PART7_TEXT)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "BELLEVUE" in chunk       # passage text
        assert "154." in chunk           # first question
        assert "155." in chunk           # last question

    def test_double_passage_single_chunk(self):
        chunks = chunk_by_question_set(DOUBLE_TEXT)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "Josh Laporte" in chunk
        assert "Argo Tractors" in chunk
        assert "181." in chunk

    def test_mixed_content_splits_correctly(self):
        chunks = chunk_by_question_set(MIXED_TEXT)
        # 2 Part5 groups + 1 Part6 + 1 Part7 = 4 chunks
        assert len(chunks) == 4

    def test_fallback_for_no_markers(self):
        chunks = chunk_by_question_set(NO_MARKER_TEXT)
        assert len(chunks) >= 1
        combined = " ".join(chunks)
        assert "TOEIC" in combined

    def test_empty_text_returns_empty(self):
        assert chunk_by_question_set("") == []

    def test_tiny_fragments_filtered_out(self):
        text = "[TOEIC Part 5 Grammar]\nOK\n\n[TOEIC Part 5 Vocabulary]\n" + "X" * 200
        chunks = chunk_by_question_set(text)
        # "OK" alone is < 50 chars so filtered; only the vocabulary chunk survives
        assert all(len(c) > 50 for c in chunks)

    def test_fallback_groups_short_paragraphs(self):
        # Text with no markers but multiple paragraphs — should be grouped into chunks
        long_para = "This is a detailed paragraph about TOEIC grammar. " * 15
        text = f"{long_para}\n\n{long_para}\n\n{long_para}"
        chunks = chunk_by_question_set(text)
        assert len(chunks) >= 1
        # No chunk should exceed fallback limit significantly
        for chunk in chunks:
            assert len(chunk) < 2400  # at most 2× the 1200 limit

    def test_qset_boundary_fallback(self):
        # Text uses "Questions N-N refer to" pattern (no [TOEIC] marker)
        text = (
            "Questions 154-155 refer to the following article.\n\n"
            "BELLEVUE — The courthouse will be renovated soon.\n\n"
            "154. Why was the article written?\n(A) To report (B) To promote\n\n"
            "Questions 156-158 refer to the following email.\n\n"
            "Dear Mr. Smith, Thank you for your inquiry.\n\n"
            "156. What is the email about?\n(A) A job (B) A product"
        )
        chunks = chunk_by_question_set(text)
        assert len(chunks) == 2
        assert "154." in chunks[0]
        assert "156." in chunks[1]
