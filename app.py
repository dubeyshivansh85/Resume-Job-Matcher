"""
app.py
Resume <-> Job Matching Tool - Streamlit App

Run with: streamlit run app.py
"""

import os
import re
import ast
import pandas as pd
import streamlit as st
import spacy
import pdfplumber
import plotly.graph_objects as go
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TAXONOMY_PATH = os.path.join(BASE_DIR, "data", "skills_taxonomy.csv")
SAMPLE_RESUMES_PATH = os.path.join(BASE_DIR, "data", "ds_resumes_with_skills.csv")
KEYWORD_WEIGHT = 0.4

st.set_page_config(page_title="Resume <-> Job Matcher", page_icon="🧩", layout="wide")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def segment_concatenated_skills(text: str, skill_extractor) -> str:
    """
    Some job sites (e.g. Naukri 'Key Skills' sections) list skills back-to-back
    with no space between them, e.g. "pythondata analysisdata analyticssql".
    This inserts a space around every known skill phrase found in the text,
    even mid-word, so the PhraseMatcher can pick them up individually.
    """
    forms = sorted(set(skill_extractor.surface_to_canonical.keys()), key=len, reverse=True)
    for form in forms:
        if len(form) < 3:
            continue
        text = re.sub(re.escape(form), f" {form} ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF file object."""
    text_parts = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Skill extractor
# ---------------------------------------------------------------------------
class SkillExtractor:
    def __init__(self, taxonomy_path: str, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.taxonomy = pd.read_csv(taxonomy_path)
        self._build_matcher()

    def _build_matcher(self):
        self.surface_to_canonical = {}
        for _, row in self.taxonomy.iterrows():
            canonical = str(row["skill_name"]).strip().lower()
            surface_forms = [canonical]
            if pd.notna(row.get("synonyms")) and str(row["synonyms"]).strip():
                syns = [s.strip().lower() for s in str(row["synonyms"]).split(",") if s.strip()]
                surface_forms.extend(syns)
            for form in surface_forms:
                self.surface_to_canonical[form] = canonical

        patterns = [self.nlp.make_doc(form) for form in self.surface_to_canonical.keys()]
        self.matcher.add("SKILL", patterns)

    def extract(self, text: str) -> set:
        if not isinstance(text, str) or not text.strip():
            return set()
        doc = self.nlp(text)
        matches = self.matcher(doc)
        found = set()
        for match_id, start, end in matches:
            span_text = doc[start:end].text.lower()
            canonical = self.surface_to_canonical.get(span_text, span_text)
            found.add(canonical)
        return found


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------
def jaccard_similarity(skills_a: set, skills_b: set) -> float:
    if not skills_a or not skills_b:
        return 0.0
    intersection = len(skills_a & skills_b)
    union = len(skills_a | skills_b)
    return intersection / union if union > 0 else 0.0


def compute_match(resume_text_clean, resume_skills, resume_embedding,
                   job_text_clean, job_skills, job_embedding,
                   keyword_weight: float = KEYWORD_WEIGHT) -> dict:
    jaccard = jaccard_similarity(resume_skills, job_skills)
    semantic = cosine_similarity(
        resume_embedding.reshape(1, -1), job_embedding.reshape(1, -1)
    )[0][0]
    final = keyword_weight * jaccard + (1 - keyword_weight) * semantic

    return {
        "jaccard": round(jaccard, 3),
        "semantic": round(float(semantic), 3),
        "final_score": round(float(final), 3),
        "matched_skills": sorted(resume_skills & job_skills),
        "missing_skills": sorted(job_skills - resume_skills),
    }


def score_missing_skill_priority(skill: str, job_title: str, job_text_clean: str,
                                  extractor: "SkillExtractor") -> int:
    """
    Priority score for a missing skill based on how prominently it's
    mentioned in the job posting. Higher = more important to add.
    """
    score = 0
    title_clean = clean_text(job_title) if job_title else ""

    # Get all surface forms (synonyms) for this skill to check mentions properly
    surface_forms = [
        form for form, canonical in extractor.surface_to_canonical.items()
        if canonical == skill
    ]
    if not surface_forms:
        surface_forms = [skill]

    for form in surface_forms:
        # Mentioned in the job title -> strong signal
        if form in title_clean:
            score += 5
        # Count occurrences in the full description -> repeated emphasis
        score += job_text_clean.count(form) * 2
        # Mentioned in the first 200 characters -> usually requirements section
        if form in job_text_clean[:200]:
            score += 3

    return score


def rank_missing_skills(missing_skills: list, job_title: str, job_text_clean: str,
                         extractor: "SkillExtractor") -> list:
    """Return missing skills sorted by priority, each as (skill, priority_label)."""
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


SUGGESTION_TEMPLATES = {
    "technical": "Add a bullet describing a project or task where you applied {skill} — name the specific outcome (e.g. \"Used {skill} to reduce processing time by X%\").",
    "tool": "Mention {skill} explicitly by name if you've used it, even briefly — recruiters and ATS systems often filter on exact tool names.",
    "soft": "Demonstrate {skill} through a concrete example rather than listing it — e.g. describe a situation where you exercised {skill} and the result.",
}


def generate_suggestions(ranked_missing_skills: list, extractor: "SkillExtractor", top_n: int = 5) -> list:
    """Generate phrasing suggestions for the top N priority missing skills."""
    taxonomy_lookup = extractor.taxonomy.set_index(
        extractor.taxonomy["skill_name"].str.strip().str.lower()
    )["category"].to_dict()

    suggestions = []
    for skill, label, score in ranked_missing_skills[:top_n]:
        category = taxonomy_lookup.get(skill, "technical")
        template = SUGGESTION_TEMPLATES.get(category, SUGGESTION_TEMPLATES["technical"])
        suggestions.append({
            "skill": skill,
            "priority": label,
            "suggestion": template.format(skill=skill),
        })
    return suggestions


def compute_ats_score(resume_text_raw: str, resume_clean: str, job_skills: set, resume_skills: set) -> dict:
    """
    Heuristic ATS (Applicant Tracking System) compatibility score.
    Mirrors the kinds of checks real ATS parsers and ATS-scan tools perform:
    structure, contact info, keyword coverage, and readability signals.
    """
    checks = []
    points = 0
    max_points = 0

    # 1. Contact info present (email)
    max_points += 15
    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", resume_text_raw))
    if has_email:
        points += 15
    checks.append(("Email address found", has_email))

    # 2. Phone number present
    max_points += 10
    has_phone = bool(re.search(r"(\+?\d[\d\-\s]{8,}\d)", resume_text_raw))
    if has_phone:
        points += 10
    checks.append(("Phone number found", has_phone))

    # 3. Standard section headers present
    max_points += 20
    section_keywords = ["experience", "education", "skills", "project"]
    sections_found = sum(1 for kw in section_keywords if kw in resume_clean)
    section_score = int((sections_found / len(section_keywords)) * 20)
    points += section_score
    checks.append((f"Standard resume sections found ({sections_found}/{len(section_keywords)})", sections_found >= 3))

    # 4. Keyword / skill coverage vs this specific job
    max_points += 35
    if job_skills:
        coverage = len(resume_skills & job_skills) / len(job_skills)
    else:
        coverage = 0
    keyword_score = int(coverage * 35)
    points += keyword_score
    checks.append((f"Job keyword coverage ({int(coverage*100)}%)", coverage >= 0.4))

    # 5. Resume length (too short = incomplete, too long = ATS may truncate)
    max_points += 10
    word_count = len(resume_clean.split())
    length_ok = 200 <= word_count <= 1200
    if length_ok:
        points += 10
    checks.append((f"Resume length reasonable ({word_count} words)", length_ok))

    # 6. Quantifiable achievements (numbers/percentages present)
    max_points += 10
    has_numbers = bool(re.search(r"\d+%|\d+\+|\b\d{2,}\b", resume_text_raw))
    if has_numbers:
        points += 10
    checks.append(("Quantifiable achievements found (numbers/%)", has_numbers))

    final_pct = int((points / max_points) * 100) if max_points > 0 else 0

    return {"score": final_pct, "checks": checks}


def build_donut_chart(matched_count: int, missing_count: int) -> go.Figure:
    """Hollow donut chart showing matched vs missing skill counts."""
    labels = ["Matched", "Missing"]
    values = [matched_count, missing_count]
    colors = ["#22c55e", "#ef4444"]  # green / red

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#0e1117", width=2)),
        textinfo="label+value",
        textfont=dict(size=13, color="white"),
        hoverinfo="label+percent",
    )])

    total = matched_count + missing_count
    coverage_pct = int((matched_count / total) * 100) if total > 0 else 0

    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b>{coverage_pct}%</b><br><span style='font-size:11px'>coverage</span>",
            x=0.5, y=0.5, font=dict(size=22, color="white"), showarrow=False
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# Cached resources (loaded once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_skill_extractor():
    return SkillExtractor(taxonomy_path=TAXONOMY_PATH)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def load_sample_resumes():
    if not os.path.exists(SAMPLE_RESUMES_PATH):
        return pd.DataFrame(columns=["Category", "Resume_str", "Resume_clean", "skills"])
    df = pd.read_csv(SAMPLE_RESUMES_PATH)
    df["skills"] = df["skills"].apply(ast.literal_eval)
    return df


extractor = load_skill_extractor()
model = load_embedding_model()
sample_resumes = load_sample_resumes()
has_sample_resumes = len(sample_resumes) > 0


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🧩 Resume ↔ Job Matching Tool")
st.caption(
    "Pick a sample resume, paste a job description, and see how well they match — "
    "combining keyword overlap and semantic (AI) similarity."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Choose a resume")
    source_options = ["Upload a resume (PDF)"]
    if has_sample_resumes:
        source_options.append("Use a sample resume")

    resume_source = st.radio(
        "Resume source",
        source_options,
        horizontal=True,
    )

    uploaded_resume_text = None
    selected_resume = None

    if resume_source == "Upload a resume (PDF)":
        uploaded_file = st.file_uploader(
            "Upload your resume or a friend's resume (PDF)", type=["pdf"]
        )
        if uploaded_file is not None:
            with st.spinner("Reading PDF..."):
                uploaded_resume_text = extract_text_from_pdf(uploaded_file)
            if uploaded_resume_text.strip():
                st.success(f"Extracted {len(uploaded_resume_text)} characters from the PDF.")
                with st.expander("Preview extracted text"):
                    st.write(uploaded_resume_text[:1500] + "...")
            else:
                st.error(
                    "Couldn't extract text from this PDF — it may be a scanned image "
                    "rather than selectable text. Try a different file."
                )
    else:
        resume_options = sample_resumes.apply(
            lambda r: f"{r['Category']} — Resume #{r.name}", axis=1
        ).tolist()
        selected_label = st.selectbox("Sample resume", resume_options)
        selected_idx = resume_options.index(selected_label)
        selected_resume = sample_resumes.iloc[selected_idx]

        with st.expander("Preview resume text"):
            st.write(selected_resume["Resume_str"][:1500] + "...")

with col2:
    st.subheader("2. Paste a job description")
    job_title_input = st.text_input("Job title (optional, improves skill priority ranking)", placeholder="e.g. Senior Data Engineer")
    job_text = st.text_area(
        "Job description",
        height=220,
        placeholder="Paste the full job description here...",
    )

analyze_clicked = st.button("🔍 Analyze Match", type="primary", use_container_width=True)

st.divider()

if analyze_clicked:
    have_resume = (uploaded_resume_text and uploaded_resume_text.strip()) or (selected_resume is not None)

    if not job_text.strip():
        st.warning("Please paste a job description first.")
    elif not have_resume:
        st.warning("Please upload a resume or select a sample resume first.")
    else:
        with st.spinner("Analyzing..."):
            if uploaded_resume_text and uploaded_resume_text.strip():
                # Resume side (computed live from uploaded PDF)
                resume_text_raw = uploaded_resume_text
                resume_clean = clean_text(uploaded_resume_text)
                resume_skills = extractor.extract(segment_concatenated_skills(resume_clean, extractor))
            else:
                # Resume side (already precomputed for sample resumes)
                resume_text_raw = selected_resume["Resume_str"]
                resume_clean = selected_resume["Resume_clean"] if "Resume_clean" in selected_resume else clean_text(selected_resume["Resume_str"])
                resume_skills = selected_resume["skills"]

            resume_embedding = model.encode(resume_clean)

            # Job side (computed live from pasted text)
            job_clean = clean_text(job_text)
            job_skills = extractor.extract(segment_concatenated_skills(job_clean, extractor))
            job_embedding = model.encode(job_clean)

            result = compute_match(
                resume_clean, resume_skills, resume_embedding,
                job_clean, job_skills, job_embedding
            )

            ats_result = compute_ats_score(resume_text_raw, resume_clean, job_skills, resume_skills)

        # --- Display results ---
        score_pct = int(result["final_score"] * 100)

        st.subheader("Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Match", f"{score_pct}%")
        m2.metric("Keyword Overlap", f"{int(result['jaccard'] * 100)}%")
        m3.metric("Semantic Similarity", f"{int(result['semantic'] * 100)}%")

        st.progress(result["final_score"])

        st.markdown("### 🤖 ATS Compatibility Score")
        ats_col1, ats_col2 = st.columns([1, 2])
        with ats_col1:
            ats_score = ats_result["score"]
            if ats_score >= 75:
                st.success(f"{ats_score}/100 — Good")
            elif ats_score >= 50:
                st.warning(f"{ats_score}/100 — Needs work")
            else:
                st.error(f"{ats_score}/100 — Weak")
        with ats_col2:
            for check_label, passed in ats_result["checks"]:
                icon = "✅" if passed else "⚠️"
                st.write(f"{icon} {check_label}")
        st.caption(
            "This estimates how well an Applicant Tracking System might parse and rank this resume "
            "for this specific job — based on structure, contact info, keyword coverage, and readability. "
            "Not a guarantee of any real ATS's exact scoring."
        )

        st.divider()

        st.markdown("### 📊 Skill Coverage")
        chart_col, legend_col = st.columns([1, 2])
        with chart_col:
            matched_count = len(result["matched_skills"])
            missing_count = len(result["missing_skills"])
            fig = build_donut_chart(matched_count, missing_count)
            st.plotly_chart(fig, use_container_width=True)
        with legend_col:
            st.markdown(f"🟢 **Matched skills:** {matched_count}")
            st.markdown(f"🔴 **Missing skills:** {missing_count}")
            st.caption(
                "Coverage = share of the job's required skills your resume already mentions. "
                "This is based on exact skill-name matches, not overall fit."
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### ✅ Matched Skills")
            if result["matched_skills"]:
                st.write(" ".join([f"`{s}`" for s in result["matched_skills"]]))
            else:
                st.write("No exact skill overlap found.")

        with col_b:
            st.markdown("### ❌ Missing Skills")
            if result["missing_skills"]:
                ranked_missing = rank_missing_skills(
                    result["missing_skills"], job_title_input, job_clean, extractor
                )
                priority_colors = {"High": "🔴", "Medium": "🟡", "Low": "⚪"}
                for skill, label, score in ranked_missing:
                    st.write(f"{priority_colors[label]} `{skill}` — **{label} priority**")
                st.caption("Priority is based on how often the skill is mentioned in the job posting, and whether it appears in the title.")
            else:
                st.write("No missing skills detected — strong match!")
                ranked_missing = []

        if result["missing_skills"]:
            st.markdown("### 💡 Suggestions to Improve Your Match")
            suggestions = generate_suggestions(ranked_missing, extractor, top_n=5)
            for s in suggestions:
                st.markdown(f"**{s['skill']}** ({s['priority']} priority) — {s['suggestion']}")

        with st.expander("How this score is calculated"):
            st.markdown(
                f"""
                - **Keyword overlap (Jaccard)**: measures exact skill-word overlap between resume and job. Weight: {int(KEYWORD_WEIGHT*100)}%
                - **Semantic similarity**: uses AI embeddings to compare overall meaning, catching related skills phrased differently. Weight: {int((1-KEYWORD_WEIGHT)*100)}%
                - **Final score** = weighted average of both
                """
            )
else:
    st.info("Select a resume, paste a job description, and click **Analyze Match** to see results.")
