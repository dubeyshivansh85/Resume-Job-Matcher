"""
app.py
Resume ↔ Job Matching Tool — Streamlit App (v2)

Run with: streamlit run app.py
"""

# Load .env for local development (ignored on Streamlit Cloud)
from dotenv import load_dotenv
load_dotenv()

import ast
import json
import os
import sys

import pandas as pd
import streamlit as st

# Make src/ importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.step1_utils import clean_text, extract_resume_text, segment_concatenated_skills
from src.step2_extractor import SkillExtractor, get_available_industries, TAXONOMY_FILES
from src.step3_matching import compute_match, compute_ats_score, detect_seniority_mismatch, get_benchmark_percentile
from src.step4_suggestions import rank_missing_skills, generate_template_suggestions, generate_ai_suggestions
from src.step5_charts import build_donut_chart, build_score_gauge, build_comparison_bar_chart
from src.step6_scraper import fetch_job_from_url

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RESUMES_PATH = os.path.join(BASE_DIR, "data", "ds_resumes_with_skills.csv")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume ↔ Job Matcher",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Card-like containers */
    div[data-testid="stExpander"] { border: 1px solid #1f2937; border-radius: 8px; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1a1d2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
    }

    /* Priority tags */
    .tag-high   { background:#7f1d1d; color:#fca5a5; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:600; }
    .tag-medium { background:#78350f; color:#fcd34d; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:600; }
    .tag-low    { background:#1f2937; color:#9ca3af; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:600; }

    /* AI badge */
    .badge-ai   { background:#3730a3; color:#a5b4fc; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600; }
    .badge-tmpl { background:#1f2937; color:#6b7280; padding:2px 8px; border-radius:12px; font-size:11px; }

    /* Skill chips */
    code { background:#1e293b !important; color:#7dd3fc !important; border-radius:6px !important; }

    /* Divider */
    hr { border-color: #1f2937 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    industry = st.selectbox(
        "🏭 Industry preset",
        get_available_industries(),
        help="Select the industry to load the matching skills taxonomy for.",
    )

    keyword_weight = st.slider(
        "🔑 Keyword vs Semantic weight",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help=(
            "Higher = more weight on exact skill keyword matching. "
            "Lower = more weight on AI semantic similarity."
        ),
    )

    model_choice = st.selectbox(
        "🤖 Embedding model",
        ["Fast — all-MiniLM-L6-v2 (~80 MB)", "Accurate — all-mpnet-base-v2 (~420 MB)"],
        help="Accurate model gives better semantic matching but is slower and larger to download.",
    )
    model_name = (
        "all-MiniLM-L6-v2"
        if "MiniLM" in model_choice
        else "all-mpnet-base-v2"
    )

    st.divider()
    st.markdown("## 🔑 AI Rewrites (Optional)")

    # Load key silently — checks Streamlit Cloud secrets first, then local .env
    # Never expose the key value in the UI
    try:
        _env_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        _env_key = ""
    if not _env_key:
        _env_key = os.getenv("GEMINI_API_KEY", "")

    if _env_key:
        # Key found in environment — show badge only, never display the key
        gemini_api_key = _env_key
        st.success("✅ AI rewrites enabled")
        st.caption("🔒 API key loaded securely from environment.")
    else:
        # No env key — let the user paste their own (empty field, no pre-fill)
        st.caption(
            "Enter your free Gemini API key to get AI-generated, copy-paste-ready "
            "resume bullet rewrites instead of generic suggestions.\n\n"
            "[Get a free key →](https://aistudio.google.com/app/apikey)"
        )
        gemini_api_key = st.text_input(
            "Gemini API key",
            value="",
            type="password",
            placeholder="AIza...",
            help="Your key is only used for this session and never stored.",
        )
        if gemini_api_key:
            st.success("✅ AI rewrites enabled")

    st.divider()
    st.markdown(
        "**How scoring works**\n\n"
        f"Final score = {int(keyword_weight*100)}% keyword overlap + "
        f"{int((1-keyword_weight)*100)}% semantic similarity"
    )
    st.caption("v2.0 · Made with ❤️ using spaCy + Sentence Transformers")


# ---------------------------------------------------------------------------
# Cached resources — reload when industry or model changes
# ---------------------------------------------------------------------------
@st.cache_resource
def load_skill_extractor(industry_key: str):
    taxonomy_path = TAXONOMY_FILES[industry_key]
    return SkillExtractor(taxonomy_path=taxonomy_path)


@st.cache_resource
def load_embedding_model(model_id: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id)


@st.cache_data
def load_sample_resumes():
    if not os.path.exists(SAMPLE_RESUMES_PATH):
        return pd.DataFrame(columns=["Category", "Resume_str", "Resume_clean", "skills"])
    df = pd.read_csv(SAMPLE_RESUMES_PATH)
    df["skills"] = df["skills"].apply(ast.literal_eval)
    return df


extractor = load_skill_extractor(industry)
embed_model = load_embedding_model(model_name)
sample_resumes = load_sample_resumes()
has_sample_resumes = len(sample_resumes) > 0


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "resume_text_raw": None,
        "resume_clean": None,
        "resume_skills": None,
        "resume_embedding": None,
        "resume_source_label": None,
        "job_bank": [],           # list of {"label", "title", "text_raw", "text_clean"}
        "compare_results": [],    # list of result dicts
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ---------------------------------------------------------------------------
# Helper: run full analysis for one job
# ---------------------------------------------------------------------------
def run_analysis(job_title: str, job_text_raw: str) -> dict:
    job_clean = clean_text(job_text_raw)
    job_skills = extractor.extract(segment_concatenated_skills(job_clean, extractor))
    job_embedding = embed_model.encode(job_clean)

    result = compute_match(
        st.session_state.resume_clean,
        st.session_state.resume_skills,
        st.session_state.resume_embedding,
        job_clean,
        job_skills,
        job_embedding,
        keyword_weight=keyword_weight,
    )
    result["ats"] = compute_ats_score(
        st.session_state.resume_text_raw,
        st.session_state.resume_clean,
        job_skills,
        st.session_state.resume_skills,
    )
    result["seniority"] = detect_seniority_mismatch(
        st.session_state.resume_clean, job_clean, job_title
    )
    result["benchmark"] = get_benchmark_percentile(result["final_score"])
    result["job_title"] = job_title
    result["job_clean"] = job_clean
    result["job_text_raw"] = job_text_raw
    result["job_skills"] = job_skills
    return result


# ---------------------------------------------------------------------------
# Helper: render full result panel
# ---------------------------------------------------------------------------
def render_result(result: dict, expanded: bool = True):
    score_pct = int(result["final_score"] * 100)
    ats_score = result["ats"]["score"]
    benchmark = result["benchmark"]

    # Score banner
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a1d2e,#0f1117);
                border:1px solid #2d3748;border-radius:16px;padding:20px 24px;margin-bottom:16px;'>
        <div style='display:flex;align-items:center;gap:16px;flex-wrap:wrap;'>
            <div>
                <div style='font-size:48px;font-weight:800;color:{"#22c55e" if score_pct>=70 else "#f59e0b" if score_pct>=50 else "#ef4444"};line-height:1;'>
                    {score_pct}%
                </div>
                <div style='color:#9ca3af;font-size:14px;margin-top:4px;'>Overall Match</div>
            </div>
            <div style='flex:1;min-width:200px;'>
                <div style='color:#d1d5db;font-size:14px;margin-bottom:8px;'>
                    🏆 Your resume scores better than ~<b>{benchmark}%</b> of applicants (estimated)
                </div>
                <div style='background:#1f2937;border-radius:8px;height:8px;'>
                    <div style='background:{"#22c55e" if score_pct>=70 else "#f59e0b" if score_pct>=50 else "#ef4444"};
                                width:{score_pct}%;height:8px;border-radius:8px;'></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔑 Keyword Overlap", f"{int(result['jaccard'] * 100)}%")
    m2.metric("🧠 Semantic Similarity", f"{int(result['semantic'] * 100)}%")
    m3.metric("📋 ATS Score", f"{ats_score}/100",
              delta="Good" if ats_score >= 75 else ("Needs work" if ats_score >= 50 else "Weak"),
              delta_color="normal" if ats_score >= 75 else "inverse")
    m4.metric("🎯 Skills Matched", f"{len(result['matched_skills'])}/{len(result['matched_skills'])+len(result['missing_skills'])}")

    # Seniority warning
    if result.get("seniority"):
        if result["seniority"]["type"] == "underqualified":
            st.warning(result["seniority"]["message"])
        else:
            st.info(result["seniority"]["message"])

    st.divider()

    # ATS checks
    with st.expander("📋 ATS Compatibility Details", expanded=False):
        for check_label, passed in result["ats"]["checks"]:
            icon = "✅" if passed else "⚠️"
            st.write(f"{icon} {check_label}")
        st.caption(
            "Heuristic estimate based on structure, contact info, keyword coverage, "
            "and readability — not a simulation of any real ATS product."
        )

    # Skill coverage
    st.markdown("### 📊 Skill Coverage")
    chart_col, legend_col = st.columns([1, 2])
    with chart_col:
        fig = build_donut_chart(len(result["matched_skills"]), len(result["missing_skills"]))
        st.plotly_chart(fig, use_container_width=True)
    with legend_col:
        st.markdown(f"🟢 **Matched:** {len(result['matched_skills'])} skills")
        st.markdown(f"🔴 **Missing:** {len(result['missing_skills'])} skills")
        st.caption(
            "Coverage = share of the job's required skills your resume already mentions. "
            "Based on exact skill-name matches from the taxonomy."
        )

    # Matched / missing skills side by side
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
                result["missing_skills"],
                result.get("job_title", ""),
                result["job_clean"],
                extractor,
            )
            priority_colors = {"High": "🔴", "Medium": "🟡", "Low": "⚪"}
            for skill, label, score in ranked_missing:
                st.markdown(
                    f"{priority_colors[label]} `{skill}` — "
                    f"<span class='tag-{label.lower()}'>{label} priority</span>",
                    unsafe_allow_html=True,
                )
            st.caption("Priority = how often the skill appears in the job posting + title.")
        else:
            ranked_missing = []
            st.success("No missing skills detected — strong match! 🎉")

    # Suggestions
    if result["missing_skills"]:
        st.markdown("### 💡 How to Improve Your Resume")

        if gemini_api_key:
            with st.spinner("🤖 Generating AI-powered bullet rewrites..."):
                suggestions = generate_ai_suggestions(
                    ranked_missing,
                    result["job_text_raw"],
                    st.session_state.resume_text_raw,
                    gemini_api_key,
                    top_n=5,
                )
            if not suggestions:
                st.caption("AI generation failed — falling back to templates.")
                suggestions = generate_template_suggestions(ranked_missing, extractor, top_n=5)
        else:
            suggestions = generate_template_suggestions(ranked_missing, extractor, top_n=5)

        for s in suggestions:
            badge = (
                "<span class='badge-ai'>✨ AI</span>"
                if s.get("ai_powered")
                else "<span class='badge-tmpl'>template</span>"
            )
            priority_tag = f"<span class='tag-{s['priority'].lower()}'>{s['priority']}</span>"
            st.markdown(
                f"**{s['skill']}** {priority_tag} {badge}<br>"
                f"{s['suggestion']}",
                unsafe_allow_html=True,
            )
            st.write("")

    # Download report
    st.divider()
    report = {
        "job_title": result.get("job_title", ""),
        "overall_match_pct": int(result["final_score"] * 100),
        "keyword_overlap_pct": int(result["jaccard"] * 100),
        "semantic_similarity_pct": int(result["semantic"] * 100),
        "ats_score": result["ats"]["score"],
        "benchmark_beats_pct": result["benchmark"],
        "matched_skills": result["matched_skills"],
        "missing_skills": result["missing_skills"],
        "seniority_flag": result.get("seniority"),
    }
    st.download_button(
        "⬇️ Download Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"match_report_{result.get('job_title','job').replace(' ','_')[:30]}.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("ℹ️ How this score is calculated"):
        st.markdown(f"""
        - **Keyword overlap (Jaccard):** Measures exact skill-word overlap between resume and job. Weight: **{int(keyword_weight*100)}%**
        - **Semantic similarity:** Uses AI embeddings (`{model_name}`) to compare overall meaning, catching related skills phrased differently. Weight: **{int((1-keyword_weight)*100)}%**
        - **Final score** = weighted average of both
        - **ATS score** = heuristic based on structure, contact info, keyword coverage, resume length, and quantifiable achievements
        - **Benchmark** = estimated percentile vs. a typical distribution of applicant scores
        """)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<h1 style='text-align:center;background:linear-gradient(90deg,#6366f1,#8b5cf6,#ec4899);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           font-size:2.5rem;margin-bottom:4px;'>
    🧩 Resume ↔ Job Matcher
</h1>
<p style='text-align:center;color:#9ca3af;font-size:1rem;margin-bottom:24px;'>
    Upload your resume once · Compare multiple jobs · Get AI-powered improvement tips
</p>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Step 1 — Resume (persists in session)
# ---------------------------------------------------------------------------
st.markdown("## Step 1 — Load Your Resume")

resume_loaded = st.session_state.resume_text_raw is not None

if resume_loaded:
    st.success(
        f"✅ Resume loaded: **{st.session_state.resume_source_label}** "
        f"({len(st.session_state.resume_text_raw.split())} words, "
        f"{len(st.session_state.resume_skills)} skills extracted)"
    )
    if st.button("🔄 Load a different resume", use_container_width=False):
        for k in ["resume_text_raw", "resume_clean", "resume_skills",
                  "resume_embedding", "resume_source_label"]:
            st.session_state[k] = None
        st.rerun()
else:
    source_options = ["Upload a resume (PDF or DOCX)"]
    if has_sample_resumes:
        source_options.append("Use a sample resume")

    resume_source = st.radio("Resume source", source_options, horizontal=True)

    if resume_source == "Upload a resume (PDF or DOCX)":
        uploaded_file = st.file_uploader(
            "Upload your resume", type=["pdf", "docx"]
        )
        if uploaded_file:
            with st.spinner("Reading file..."):
                text, err = extract_resume_text(uploaded_file)
            if err:
                st.error(err)
            elif text:
                st.success(f"Extracted {len(text.split())} words from **{uploaded_file.name}**")
                with st.expander("Preview extracted text"):
                    st.write(text[:1500] + ("..." if len(text) > 1500 else ""))
                if st.button("✅ Use this resume", type="primary"):
                    with st.spinner("Processing resume..."):
                        clean = clean_text(text)
                        skills = extractor.extract(segment_concatenated_skills(clean, extractor))
                        embedding = embed_model.encode(clean)
                    st.session_state.resume_text_raw = text
                    st.session_state.resume_clean = clean
                    st.session_state.resume_skills = skills
                    st.session_state.resume_embedding = embedding
                    st.session_state.resume_source_label = uploaded_file.name
                    st.rerun()
    else:
        resume_options = sample_resumes.apply(
            lambda r: f"{r['Category']} — Resume #{r.name}", axis=1
        ).tolist()
        selected_label = st.selectbox("Sample resume", resume_options)
        selected_idx = resume_options.index(selected_label)
        selected_resume = sample_resumes.iloc[selected_idx]

        with st.expander("Preview resume text"):
            st.write(str(selected_resume["Resume_str"])[:1500] + "...")

        if st.button("✅ Use this resume", type="primary"):
            with st.spinner("Processing sample resume..."):
                raw = selected_resume["Resume_str"]
                clean = (
                    selected_resume["Resume_clean"]
                    if "Resume_clean" in selected_resume
                    else clean_text(raw)
                )
                skills = set(selected_resume["skills"]) if isinstance(selected_resume["skills"], list) else selected_resume["skills"]
                embedding = embed_model.encode(clean)
            st.session_state.resume_text_raw = raw
            st.session_state.resume_clean = clean
            st.session_state.resume_skills = skills
            st.session_state.resume_embedding = embedding
            st.session_state.resume_source_label = selected_label
            st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Tabs — only show after resume is loaded
# ---------------------------------------------------------------------------
if not resume_loaded:
    st.info("⬆️ Load your resume above to get started.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "🔍 Analyze Single Job",
    "📊 Compare Multiple Jobs",
    "✏️ Resume Editor (Live Score)",
])


# ===========================================================================
# TAB 1 — Single Job Analysis
# ===========================================================================
with tab1:
    st.markdown("## Step 2 — Enter a Job Description")

    job_title_input = st.text_input(
        "Job title (optional — improves skill priority ranking)",
        placeholder="e.g. Senior Data Engineer",
        key="tab1_title",
    )

    job_input_method = st.radio(
        "How do you want to provide the job?",
        ["📋 Paste job description", "🔗 Paste job URL (auto-fetch)"],
        horizontal=True,
        key="tab1_method",
    )

    job_text_raw = ""

    if job_input_method == "📋 Paste job description":
        job_text_raw = st.text_area(
            "Job description",
            height=200,
            placeholder="Paste the full job description here...",
            key="tab1_paste",
        )
    else:
        job_url = st.text_input(
            "Job posting URL",
            placeholder="https://indeed.com/viewjob?jk=...",
            key="tab1_url",
        )
        if job_url:
            if st.button("🌐 Fetch job description", key="tab1_fetch"):
                with st.spinner("Fetching job description..."):
                    fetched_text, fetch_err = fetch_job_from_url(job_url)
                if fetch_err:
                    st.error(fetch_err)
                elif fetched_text:
                    st.session_state["tab1_fetched"] = fetched_text
                    st.success(f"Fetched {len(fetched_text.split())} words from URL.")

        if "tab1_fetched" in st.session_state and st.session_state["tab1_fetched"]:
            job_text_raw = st.session_state["tab1_fetched"]
            with st.expander("Preview fetched text"):
                st.write(job_text_raw[:1000] + "...")

    if st.button("🔍 Analyze Match", type="primary", use_container_width=True, key="tab1_analyze"):
        if not job_text_raw.strip():
            st.warning("Please provide a job description first.")
        else:
            with st.spinner("Analyzing..."):
                result = run_analysis(job_title_input, job_text_raw)
            render_result(result)


# ===========================================================================
# TAB 2 — Multi-Job Comparison
# ===========================================================================
with tab2:
    st.markdown("## Compare Your Resume Against Multiple Jobs")
    st.caption("Add up to 5 jobs and see them ranked by match score.")

    with st.expander("➕ Add a Job", expanded=len(st.session_state.job_bank) == 0):
        cj_title = st.text_input("Job title", placeholder="e.g. Data Scientist at Google", key="cj_title")
        cj_method = st.radio(
            "Input method",
            ["📋 Paste text", "🔗 Paste URL"],
            horizontal=True,
            key="cj_method",
        )

        cj_text = ""
        if cj_method == "📋 Paste text":
            cj_text = st.text_area("Job description", height=150, key="cj_paste")
        else:
            cj_url = st.text_input("Job URL", placeholder="https://...", key="cj_url")
            if cj_url and st.button("🌐 Fetch", key="cj_fetch"):
                with st.spinner("Fetching..."):
                    cj_fetched, cj_err = fetch_job_from_url(cj_url)
                if cj_err:
                    st.error(cj_err)
                else:
                    st.session_state["cj_fetched"] = cj_fetched
                    st.success(f"Fetched {len(cj_fetched.split())} words.")
            if "cj_fetched" in st.session_state:
                cj_text = st.session_state.get("cj_fetched", "")
                st.caption(f"Using fetched text ({len(cj_text.split())} words)")

        if st.button("➕ Add to comparison", key="cj_add", type="primary"):
            if not cj_text.strip():
                st.warning("Please provide a job description.")
            elif len(st.session_state.job_bank) >= 5:
                st.warning("Maximum 5 jobs. Remove one first.")
            else:
                label = cj_title.strip() or f"Job {len(st.session_state.job_bank) + 1}"
                st.session_state.job_bank.append({
                    "label": label,
                    "title": cj_title.strip(),
                    "text_raw": cj_text,
                })
                st.session_state.pop("cj_fetched", None)
                st.success(f"Added: **{label}**")
                st.rerun()

    if st.session_state.job_bank:
        st.markdown(f"### {len(st.session_state.job_bank)} Job(s) in comparison")

        # Remove buttons
        for i, job in enumerate(st.session_state.job_bank):
            col_l, col_r = st.columns([4, 1])
            col_l.markdown(f"**{i+1}.** {job['label']}")
            if col_r.button("🗑️ Remove", key=f"remove_{i}"):
                st.session_state.job_bank.pop(i)
                st.rerun()

        if st.button("⚡ Run Comparison", type="primary", use_container_width=True, key="cj_run"):
            compare_results = []
            progress = st.progress(0)
            for i, job in enumerate(st.session_state.job_bank):
                with st.spinner(f"Analyzing: {job['label']}..."):
                    r = run_analysis(job["title"], job["text_raw"])
                    r["label"] = job["label"]
                    compare_results.append(r)
                progress.progress((i + 1) / len(st.session_state.job_bank))
            st.session_state.compare_results = compare_results

        if st.session_state.compare_results:
            results = sorted(st.session_state.compare_results, key=lambda x: x["final_score"], reverse=True)

            # Summary table
            st.markdown("### 📊 Ranked Results")
            table_data = []
            for rank, r in enumerate(results, 1):
                score_pct = int(r["final_score"] * 100)
                emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                table_data.append({
                    "Rank": emoji,
                    "Job": r["label"],
                    "Overall Match": f"{score_pct}%",
                    "Keywords": f"{int(r['jaccard']*100)}%",
                    "Semantic": f"{int(r['semantic']*100)}%",
                    "ATS": f"{r['ats']['score']}/100",
                    "Matched Skills": len(r["matched_skills"]),
                    "Missing Skills": len(r["missing_skills"]),
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            # Bar chart
            st.plotly_chart(
                build_comparison_bar_chart(results),
                use_container_width=True,
            )

            # Expandable details per job
            st.markdown("### 📂 Detailed Results per Job")
            for r in results:
                with st.expander(f"{r['label']} — {int(r['final_score']*100)}% match"):
                    render_result(r, expanded=False)


# ===========================================================================
# TAB 3 — Live Resume Editor
# ===========================================================================
with tab3:
    st.markdown("## ✏️ Resume Editor — See Your Score Update Live")
    st.caption(
        "Edit your resume text below and pick a job to score against. "
        "The score updates every time you click **Re-score**."
    )

    editor_col, job_col = st.columns([3, 2])

    with editor_col:
        st.markdown("#### Your Resume (editable)")
        edited_resume = st.text_area(
            "Resume text",
            value=st.session_state.resume_text_raw or "",
            height=400,
            key="editor_resume",
            label_visibility="collapsed",
        )

    with job_col:
        st.markdown("#### Job to score against")
        editor_job_title = st.text_input("Job title", placeholder="e.g. Data Analyst", key="editor_title")
        editor_job_text = st.text_area(
            "Job description",
            height=280,
            placeholder="Paste a job description...",
            key="editor_job",
        )

    if st.button("🔄 Re-score", type="primary", use_container_width=True, key="editor_score"):
        if not edited_resume.strip():
            st.warning("Resume text is empty.")
        elif not editor_job_text.strip():
            st.warning("Please paste a job description to score against.")
        else:
            with st.spinner("Scoring..."):
                # Temporarily use the edited resume for scoring
                edited_clean = clean_text(edited_resume)
                edited_skills = extractor.extract(segment_concatenated_skills(edited_clean, extractor))
                edited_embedding = embed_model.encode(edited_clean)

                job_clean = clean_text(editor_job_text)
                job_skills = extractor.extract(segment_concatenated_skills(job_clean, extractor))
                job_embedding = embed_model.encode(job_clean)

                result = compute_match(
                    edited_clean, edited_skills, edited_embedding,
                    job_clean, job_skills, job_embedding,
                    keyword_weight=keyword_weight,
                )

            score_pct = int(result["final_score"] * 100)

            # Before/after delta if session resume exists
            original_skills = st.session_state.resume_skills or set()
            original_embedding = st.session_state.resume_embedding
            if original_embedding is not None:
                from src.matching import compute_match as _cm
                orig_result = _cm(
                    st.session_state.resume_clean,
                    original_skills,
                    original_embedding,
                    job_clean,
                    job_skills,
                    job_embedding,
                    keyword_weight=keyword_weight,
                )
                orig_pct = int(orig_result["final_score"] * 100)
                delta = score_pct - orig_pct
                delta_str = f"+{delta}%" if delta >= 0 else f"{delta}%"
                st.metric(
                    "📈 Match Score (edited resume)",
                    f"{score_pct}%",
                    delta=delta_str,
                    delta_color="normal" if delta >= 0 else "inverse",
                )
                if delta > 0:
                    st.success(f"Your edits improved the match by **{delta} percentage points**! 🎉")
                elif delta < 0:
                    st.warning(f"Your edits lowered the match by {abs(delta)} points.")
                else:
                    st.info("Score unchanged.")
            else:
                st.metric("Match Score", f"{score_pct}%")

            col_x, col_y = st.columns(2)
            with col_x:
                st.markdown("**✅ Matched Skills**")
                st.write(" ".join([f"`{s}`" for s in sorted(result["matched_skills"])]) or "None")
            with col_y:
                st.markdown("**❌ Missing Skills**")
                st.write(" ".join([f"`{s}`" for s in sorted(result["missing_skills"])]) or "None")

            if st.button("💾 Save edited resume to session", key="save_edited"):
                st.session_state.resume_text_raw = edited_resume
                st.session_state.resume_clean = edited_clean
                st.session_state.resume_skills = edited_skills
                st.session_state.resume_embedding = edited_embedding
                st.session_state.resume_source_label = "Edited resume"
                st.success("Saved! Your edited resume is now active across all tabs.")
