"""
tests/test_matching.py
Unit tests for core matching functions.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest

from src.step3_matching import (
    jaccard_similarity,
    compute_match,
    compute_ats_score,
    detect_seniority_mismatch,
    get_benchmark_percentile,
)
from src.step1_utils import clean_text


# ---------------------------------------------------------------------------
# jaccard_similarity
# ---------------------------------------------------------------------------

class TestJaccardSimilarity:
    def test_identical_sets(self):
        a = {"python", "sql", "pandas"}
        assert jaccard_similarity(a, a) == 1.0

    def test_no_overlap(self):
        assert jaccard_similarity({"python"}, {"java"}) == 0.0

    def test_partial_overlap(self):
        a = {"python", "sql", "pandas"}
        b = {"python", "sql", "java"}
        # intersection=2, union=4 → 0.5
        assert jaccard_similarity(a, b) == pytest.approx(0.5)

    def test_empty_sets(self):
        assert jaccard_similarity(set(), {"python"}) == 0.0
        assert jaccard_similarity({"python"}, set()) == 0.0
        assert jaccard_similarity(set(), set()) == 0.0


# ---------------------------------------------------------------------------
# compute_match
# ---------------------------------------------------------------------------

class TestComputeMatch:
    def _make_embedding(self, size=384):
        vec = np.random.rand(size).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def test_keys_present(self):
        emb = self._make_embedding()
        result = compute_match(
            "resume text", {"python", "sql"}, emb,
            "job text", {"python", "java"}, emb,
        )
        assert "final_score" in result
        assert "jaccard" in result
        assert "semantic" in result
        assert "matched_skills" in result
        assert "missing_skills" in result

    def test_score_in_range(self):
        emb = self._make_embedding()
        result = compute_match(
            "resume text", {"python"}, emb,
            "job text", {"python"}, emb,
        )
        assert 0.0 <= result["final_score"] <= 1.0

    def test_identical_embeddings_high_semantic(self):
        emb = self._make_embedding()
        result = compute_match(
            "text", {"python"}, emb,
            "text", {"python"}, emb,
        )
        # Identical embeddings → semantic ≈ 1.0
        assert result["semantic"] > 0.99

    def test_matched_and_missing_skills(self):
        emb = self._make_embedding()
        result = compute_match(
            "text", {"python", "sql"}, emb,
            "text", {"python", "java"}, emb,
        )
        assert "python" in result["matched_skills"]
        assert "java" in result["missing_skills"]
        assert "sql" not in result["missing_skills"]

    def test_keyword_weight_zero_ignores_jaccard(self):
        emb = self._make_embedding()
        # With keyword_weight=0, only semantic matters
        result = compute_match(
            "text", {"python"}, emb,
            "text", {"java"}, emb,
            keyword_weight=0.0,
        )
        assert result["final_score"] == pytest.approx(result["semantic"], abs=0.01)


# ---------------------------------------------------------------------------
# compute_ats_score
# ---------------------------------------------------------------------------

class TestATSScore:
    def test_score_in_range(self):
        result = compute_ats_score(
            "John Smith\njohn@email.com\n+1 555 1234567\nExperience Education Skills",
            "experience education skills project python sql",
            {"python", "sql"},
            {"python"},
        )
        assert 0 <= result["score"] <= 100

    def test_email_detected(self):
        result = compute_ats_score("test@example.com", "", set(), set())
        email_check = next(c for c, p in result["checks"] if "Email" in c)
        assert email_check  # should be True

    def test_no_email(self):
        result = compute_ats_score("no email here", "", set(), set())
        _, passed = result["checks"][0]  # first check is email
        assert not passed

    def test_checks_list_length(self):
        result = compute_ats_score("", "", set(), set())
        assert len(result["checks"]) == 6


# ---------------------------------------------------------------------------
# detect_seniority_mismatch
# ---------------------------------------------------------------------------

class TestSeniorityMismatch:
    def test_senior_jd_junior_resume(self):
        result = detect_seniority_mismatch(
            "worked as data analyst for 2 years",
            "senior data scientist 7+ years of experience required",
            "Senior Data Scientist",
        )
        assert result is not None
        assert result["type"] == "underqualified"

    def test_junior_jd_senior_resume(self):
        result = detect_seniority_mismatch(
            "senior lead data scientist manager",
            "entry level junior analyst 0-2 years",
            "Junior Data Analyst",
        )
        assert result is not None
        assert result["type"] == "overqualified"

    def test_no_mismatch(self):
        result = detect_seniority_mismatch(
            "data analyst 3 years experience",
            "data analyst 2-4 years experience",
            "Data Analyst",
        )
        assert result is None


# ---------------------------------------------------------------------------
# get_benchmark_percentile
# ---------------------------------------------------------------------------

class TestBenchmarkPercentile:
    def test_high_score(self):
        assert get_benchmark_percentile(0.85) >= 95

    def test_low_score(self):
        assert get_benchmark_percentile(0.1) <= 5

    def test_mid_score(self):
        pct = get_benchmark_percentile(0.55)
        assert 30 <= pct <= 80

    def test_boundary_values(self):
        assert get_benchmark_percentile(1.0) >= 99
        assert get_benchmark_percentile(0.0) >= 1


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_lowercases(self):
        assert clean_text("Python SQL") == "python sql"

    def test_html_removed(self):
        assert "<b>" not in clean_text("<b>Python</b>")

    def test_empty_input(self):
        assert clean_text("") == ""
        assert clean_text("   ") == ""
        assert clean_text(None) == ""

    def test_camel_case_split(self):
        result = clean_text("dataScience")
        assert "data" in result and "science" in result

    def test_special_chars_stripped(self):
        result = clean_text("Python! SQL@##")
        assert "!" not in result
        assert "@" not in result
