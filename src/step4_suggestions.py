"""
src/suggestions.py  |  STEP 4 — Gap Analysis & Suggestions
-----------------------------------------------------------
Takes missing skills from Step 3 and turns them into actionable advice.
Contains:
  - rank_missing_skills()         : sorts missing skills by how important they are
  - generate_template_suggestions(): rule-based resume improvement tips
  - generate_ai_suggestions()     : Gemini API powered copy-paste bullet rewrites
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------

def score_missing_skill_priority(
    skill: str,
    job_title: str,
    job_text_clean: str,
    extractor,
) -> int:
    """
    Priority score for a missing skill based on how prominently it's
    mentioned in the job posting. Higher = more important to add.
    """
    score = 0
    title_clean = job_title.lower() if job_title else ""

    surface_forms = [
        form
        for form, canonical in extractor.surface_to_canonical.items()
        if canonical == skill
    ]
    if not surface_forms:
        surface_forms = [skill]

    for form in surface_forms:
        if form in title_clean:
            score += 5
        score += job_text_clean.count(form) * 2
        if form in job_text_clean[:200]:
            score += 3

    return score


def rank_missing_skills(
    missing_skills: list,
    job_title: str,
    job_text_clean: str,
    extractor,
) -> list:
    """Return missing skills sorted by priority, each as (skill, priority_label, score)."""
    scored = [
        (skill, score_missing_skill_priority(skill, job_title, job_text_clean, extractor))
        for skill in missing_skills
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    ranked = []
    for skill, score in scored:
        if score >= 6:
            label = "High"
        elif score >= 2:
            label = "Medium"
        else:
            label = "Low"
        ranked.append((skill, label, score))
    return ranked


# ---------------------------------------------------------------------------
# Template-based suggestions (no API key needed)
# ---------------------------------------------------------------------------

SUGGESTION_TEMPLATES = {
    "technical": (
        "Add a bullet describing a project or task where you applied **{skill}** — "
        "name the specific outcome (e.g. \"Used {skill} to reduce processing time by X%\")."
    ),
    "tool": (
        "Mention **{skill}** explicitly by name if you've used it, even briefly — "
        "recruiters and ATS systems often filter on exact tool names."
    ),
    "soft": (
        "Demonstrate **{skill}** through a concrete example rather than listing it — "
        "e.g. describe a situation where you exercised {skill} and the result."
    ),
}


def generate_template_suggestions(ranked_missing_skills: list, extractor, top_n: int = 5) -> list:
    """Generate phrasing suggestions for the top N priority missing skills."""
    taxonomy_lookup = (
        extractor.taxonomy
        .set_index(extractor.taxonomy["skill_name"].str.strip().str.lower())["category"]
        .to_dict()
    )

    suggestions = []
    for skill, label, score in ranked_missing_skills[:top_n]:
        category = taxonomy_lookup.get(skill, "technical")
        template = SUGGESTION_TEMPLATES.get(category, SUGGESTION_TEMPLATES["technical"])
        suggestions.append({
            "skill": skill,
            "priority": label,
            "suggestion": template.format(skill=skill),
            "ai_powered": False,
        })
    return suggestions


# ---------------------------------------------------------------------------
# AI-powered bullet rewrites (Gemini API — optional, free tier)
# ---------------------------------------------------------------------------

def generate_ai_suggestions(
    ranked_missing_skills: list,
    job_text: str,
    resume_text: str,
    gemini_api_key: str,
    top_n: int = 5,
) -> list:
    """
    Use Gemini to generate specific, copy-paste-ready resume bullet rewrites
    for missing skills.

    Falls back to template suggestions if the API call fails.

    Requires: pip install google-genai
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=gemini_api_key)

        skills_list = [skill for skill, _, _ in ranked_missing_skills[:top_n]]
        skills_str = ", ".join(skills_list)

        prompt = f"""You are an expert resume writer helping a job applicant improve their resume.

The applicant is missing these skills that the job description requires: {skills_str}

Job description (excerpt):
{job_text[:1500]}

Resume (excerpt):
{resume_text[:1500]}

For each missing skill, write ONE specific, ready-to-use resume bullet point that:
1. Naturally incorporates the skill
2. Includes a plausible quantified outcome (use X%, $Y, or N units as placeholder)
3. Sounds professional but not generic
4. Is 1-2 sentences max

Format your response as a numbered list matching the order of skills above.
Only output the bullet points, nothing else."""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=800,
            ),
        )

        raw_text = response.text.strip()
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

        # Parse numbered lines
        bullet_lines = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned:
                bullet_lines.append(cleaned)

        suggestions = []
        for i, (skill, label, score) in enumerate(ranked_missing_skills[:top_n]):
            bullet = bullet_lines[i] if i < len(bullet_lines) else None
            if bullet:
                suggestions.append({
                    "skill": skill,
                    "priority": label,
                    "suggestion": bullet,
                    "ai_powered": True,
                })
            else:
                # Fall back to template
                suggestions.extend(
                    generate_template_suggestions([(skill, label, score)], None, top_n=1)
                )

        return suggestions

    except Exception as e:
        # Graceful fallback — return empty list so caller can fall back to templates
        return []


# Needed for regex inside generate_ai_suggestions
import re
