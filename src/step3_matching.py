"""
src/matching.py  |  STEP 3 — Matching Engine (Core Logic)
----------------------------------------------------------
The heart of the project. Takes cleaned text + extracted skills from
Steps 1 & 2, and computes how well resume matches the job.
Contains:
  - jaccard_similarity()       : keyword overlap score (set intersection/union)
  - compute_match()            : combines Jaccard + semantic cosine similarity
  - compute_ats_score()        : heuristic ATS compatibility checker
  - detect_seniority_mismatch(): warns if seniority level doesn't match
  - get_benchmark_percentile() : estimates how score compares vs other applicants
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Core match computation
# ---------------------------------------------------------------------------

def jaccard_similarity(skills_a: set, skills_b: set) -> float:
    if not skills_a or not skills_b:
        return 0.0
    intersection = len(skills_a & skills_b)
    union = len(skills_a | skills_b)
    return intersection / union if union > 0 else 0.0


def compute_match(
    resume_text_clean: str,
    resume_skills: set,
    resume_embedding: np.ndarray,
    job_text_clean: str,
    job_skills: set,
    job_embedding: np.ndarray,
    keyword_weight: float = 0.4,
) -> dict:
    jaccard = jaccard_similarity(resume_skills, job_skills)
    semantic = float(
        cosine_similarity(
            resume_embedding.reshape(1, -1), job_embedding.reshape(1, -1)
        )[0][0]
    )
    final = keyword_weight * jaccard + (1 - keyword_weight) * semantic

    return {
        "jaccard": round(jaccard, 3),
        "semantic": round(semantic, 3),
        "final_score": round(final, 3),
        "matched_skills": sorted(resume_skills & job_skills),
        "missing_skills": sorted(job_skills - resume_skills),
    }


# ---------------------------------------------------------------------------
# ATS score
# ---------------------------------------------------------------------------

def compute_ats_score(
    resume_text_raw: str,
    resume_clean: str,
    job_skills: set,
    resume_skills: set,
) -> dict:
    """
    Heuristic ATS (Applicant Tracking System) compatibility score.
    Mirrors the kinds of checks real ATS parsers and ATS-scan tools perform:
    structure, contact info, keyword coverage, and readability signals.
    """
    checks = []
    points = 0
    max_points = 0

    # 1. Contact info — email
    max_points += 15
    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", resume_text_raw))
    if has_email:
        points += 15
    checks.append(("Email address found", has_email))

    # 2. Phone number
    max_points += 10
    has_phone = bool(re.search(r"(\+?\d[\d\-\s]{8,}\d)", resume_text_raw))
    if has_phone:
        points += 10
    checks.append(("Phone number found", has_phone))

    # 3. Standard section headers
    max_points += 20
    section_keywords = ["experience", "education", "skills", "project"]
    sections_found = sum(1 for kw in section_keywords if kw in resume_clean)
    section_score = int((sections_found / len(section_keywords)) * 20)
    points += section_score
    checks.append((
        f"Standard resume sections found ({sections_found}/{len(section_keywords)})",
        sections_found >= 3,
    ))

    # 4. Keyword / skill coverage vs this specific job
    max_points += 35
    if job_skills:
        coverage = len(resume_skills & job_skills) / len(job_skills)
    else:
        coverage = 0
    keyword_score = int(coverage * 35)
    points += keyword_score
    checks.append((
        f"Job keyword coverage ({int(coverage * 100)}%)",
        coverage >= 0.4,
    ))

    # 5. Resume length (too short = incomplete, too long = ATS may truncate)
    max_points += 10
    word_count = len(resume_clean.split())
    length_ok = 200 <= word_count <= 1200
    if length_ok:
        points += 10
    checks.append((f"Resume length reasonable ({word_count} words)", length_ok))

    # 6. Quantifiable achievements
    max_points += 10
    has_numbers = bool(re.search(r"\d+%|\d+\+|\b\d{2,}\b", resume_text_raw))
    if has_numbers:
        points += 10
    checks.append(("Quantifiable achievements found (numbers/%)", has_numbers))

    final_pct = int((points / max_points) * 100) if max_points > 0 else 0
    return {"score": final_pct, "checks": checks}


# ---------------------------------------------------------------------------
# Seniority mismatch detection
# ---------------------------------------------------------------------------

SENIOR_SIGNALS_JD = [
    "senior", "lead", "principal", "staff", "head of", "manager",
    "5+ years", "7+ years", "8+ years", "10+ years",
]
JUNIOR_SIGNALS_JD = [
    "junior", "entry level", "entry-level", "0-2 years", "1-2 years",
    "fresher", "graduate", "intern",
]
SENIOR_SIGNALS_RESUME = [
    "senior", "lead", "principal", "staff", "manager", "director",
    "head of", "vp ", "chief",
]


def detect_seniority_mismatch(
    resume_text_clean: str, job_text_clean: str, job_title: str
) -> dict | None:
    """
    Returns a warning dict if there's a likely seniority mismatch,
    or None if everything looks fine.
    """
    combined_jd = (job_text_clean + " " + (job_title or "")).lower()

    jd_is_senior = any(s in combined_jd for s in SENIOR_SIGNALS_JD)
    jd_is_junior = any(s in combined_jd for s in JUNIOR_SIGNALS_JD)

    resume_is_senior = any(s in resume_text_clean for s in SENIOR_SIGNALS_RESUME)

    if jd_is_senior and not resume_is_senior:
        return {
            "type": "underqualified",
            "message": (
                "⚠️ This job posting uses **senior-level language** (e.g. 'Senior', 'Lead', '5+ years'). "
                "Your resume doesn't show obvious seniority signals. "
                "Consider emphasising leadership, mentoring, or ownership of projects."
            ),
        }
    if jd_is_junior and resume_is_senior:
        return {
            "type": "overqualified",
            "message": (
                "ℹ️ This role appears to be **entry/junior level** but your resume shows senior experience. "
                "You may be overqualified — consider whether this is intentional."
            ),
        }
    return None


# ---------------------------------------------------------------------------
# Benchmark scoring (heuristic distribution based on typical match scores)
# ---------------------------------------------------------------------------

# Approximate percentile table built from empirical resume-job match distributions.
# Values represent: "a final_score of X beats approximately Y% of applicants"
_BENCHMARK_TABLE = [
    (0.80, 99), (0.75, 95), (0.70, 90), (0.65, 82), (0.60, 72),
    (0.55, 60), (0.50, 48), (0.45, 36), (0.40, 25), (0.35, 15),
    (0.30, 8),  (0.25, 4),  (0.00, 1),
]


def get_benchmark_percentile(final_score: float) -> int:
    """Return estimated percentile (1–99) for a given final match score."""
    for threshold, pct in _BENCHMARK_TABLE:
        if final_score >= threshold:
            return pct
    return 1
