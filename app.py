"""
app.py - Resume <-> Job Matching & ATS Analytics
Concise, self-contained Streamlit application.
Run with: streamlit run app.py
"""

from dotenv import load_dotenv
load_dotenv()

import io, os, re, json, warnings
from typing import Optional, Tuple, Dict, List, Set, Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Compact Taxonomy Knowledge Base
# ---------------------------------------------------------------------------
TAXONOMY_CONFIG = {
    "data_science": {
        "technical": (
            "python,sql,r,java,scala,c++,javascript,machine learning,deep learning,"
            "natural language processing,computer vision,data visualization,data analysis,"
            "data engineering,data modeling,statistics,a/b testing,hypothesis testing,"
            "regression,classification,clustering,neural networks,etl,ci/cd,linux,bash,"
            "rest api,graphql,nosql,data warehousing,big data,distributed systems,"
            "time series analysis,recommendation systems,llm,generative ai,prompt engineering,"
            "mlops,model deployment,feature engineering,data cleaning,data pipeline,"
            "spreadsheet modeling,financial modeling,business intelligence,kpi tracking,"
            "dashboarding,agile,c#,c,node.js,react,html,css,data governance,data quality,"
            "data privacy,gdpr,model monitoring,a/b test design,experimentation,"
            "causal inference,bayesian statistics,optimization,linear programming,"
            "supply chain analytics,fraud detection,anomaly detection,churn prediction,"
            "customer segmentation,market basket analysis,sentiment analysis,topic modeling,"
            "transformers,bert,gpt,image processing,speech recognition,reinforcement learning,"
            "random forest,gradient boosting"
        ),
        "tool": (
            "excel,tableau,power bi,looker,tensorflow,pytorch,keras,scikit-learn,pandas,"
            "numpy,matplotlib,seaborn,spark,hadoop,kafka,airflow,dbt,snowflake,redshift,"
            "bigquery,aws,azure,gcp,docker,kubernetes,git,github,mongodb,postgresql,mysql,"
            "huggingface,langchain,matlab,sas,spss,apache flink,apache beam,terraform,"
            "jenkins,jira,confluence,salesforce,looker studio,d3.js,flask,django,fastapi,"
            "opencv,xgboost,lightgbm"
        ),
        "soft": (
            "communication,teamwork,problem solving,leadership,project management,"
            "stakeholder management,critical thinking,attention to detail,presentation skills,mentoring"
        ),
        "synonyms": {
            "sql": ["structured query language"], "c++": ["cpp"], "javascript": ["js"], "excel": ["microsoft excel"],
            "power bi": ["powerbi"], "machine learning": ["ml"], "deep learning": ["dl"], "natural language processing": ["nlp"],
            "computer vision": ["cv"], "statistics": ["statistical analysis"], "a/b testing": ["ab testing"], "scikit-learn": ["sklearn"],
            "spark": ["apache spark"], "kafka": ["apache kafka"], "airflow": ["apache airflow"], "etl": ["extract transform load"],
            "redshift": ["amazon redshift"], "bigquery": ["google bigquery"], "nosql": ["non relational database"],
            "llm": ["large language model", "llms"], "generative ai": ["genai", "gen ai"], "ci/cd": ["cicd", "continuous integration"],
            "mlops": ["machine learning operations"], "time series analysis": ["time series"],
            "neural networks": ["ann", "artificial neural networks"], "pytorch": ["torch"], "aws": ["amazon web services"],
            "gcp": ["google cloud platform"], "docker": ["containerization"], "kubernetes": ["k8s"], "mongodb": ["mongo"],
            "postgresql": ["postgres"], "a/b test design": ["ab test design"], "transformers": ["transformer models"],
            "bert": ["bidirectional encoder representations"], "gpt": ["generative pre-trained transformer"],
            "random forest": ["rf"], "gradient boosting": ["gbm", "gbdt"], "xgboost": ["xgb"]
        }
    },
    "software_engineering": {
        "technical": (
            "python,java,javascript,typescript,c++,c#,go,rust,ruby,kotlin,swift,php,html,css,"
            "sql,nosql,rest api,graphql,grpc,microservices,distributed systems,system design,"
            "data structures,algorithms,object-oriented programming,functional programming,"
            "database design,event-driven architecture,concurrency,multithreading,ci/cd,linux,"
            "bash,networking,security,oauth,jwt,test-driven development,unit testing,"
            "integration testing,e2e testing,code review,design patterns,serverless,"
            "web development,backend development,frontend development,full stack development,"
            "cloud computing,agile,scrum,devops"
        ),
        "tool": (
            "docker,kubernetes,terraform,ansible,jenkins,github actions,git,github,gitlab,"
            "aws,gcp,azure,react,vue,angular,next.js,node.js,express,fastapi,django,flask"
        ),
        "soft": (
            "communication,teamwork,problem solving,leadership,project management,"
            "attention to detail,mentoring"
        ),
        "synonyms": {
            "javascript": ["js"], "typescript": ["ts"], "c++": ["cpp"], "c#": ["csharp"], "go": ["golang"],
            "html": ["html5"], "css": ["css3"], "rest api": ["restful api", "rest"], "microservices": ["microservice architecture"],
            "system design": ["systems design"], "object-oriented programming": ["oop"], "ci/cd": ["cicd", "continuous integration"],
            "test-driven development": ["tdd"], "aws": ["amazon web services"], "gcp": ["google cloud"],
            "docker": ["containers"], "kubernetes": ["k8s"], "github actions": ["gha"], "node.js": ["nodejs", "node"],
            "react": ["reactjs", "react.js"], "vue": ["vuejs", "vue.js"], "angular": ["angularjs"], "next.js": ["nextjs"],
            "fastapi": ["fast api"], "full stack development": ["fullstack"], "cloud computing": ["cloud"]
        }
    }
}

def load_taxonomy_records(industry_key: str = "data_science") -> List[Dict[str, Any]]:
    cfg = TAXONOMY_CONFIG.get(industry_key, TAXONOMY_CONFIG["data_science"])
    syn_map = cfg.get("synonyms", {})
    records = []
    for cat in ["technical", "tool", "soft"]:
        for name in cfg[cat].split(","):
            name = name.strip()
            if name:
                records.append({"skill_name": name, "category": cat, "synonyms": syn_map.get(name, [])})
    return records

TAXONOMY_OPTIONS = {"Data Science / ML / AI": "data_science", "Software Engineering": "software_engineering"}

SAMPLE_RESUME = """Alex Chen
Email: alex.chen@example.com | Phone: +1 (555) 382-9912 | San Francisco, CA
GitHub: github.com/alexchen-ml | LinkedIn: linkedin.com/in/alexchen-ai

PROFESSIONAL SUMMARY
Data Scientist with 3+ years experience building predictive machine learning models, statistical inference pipelines, and scalable predictive systems. Proven track record of leveraging Python, SQL, and deep learning frameworks to drive business outcomes.

TECHNICAL SKILLS
Languages: Python, SQL, R, Bash
Machine Learning: PyTorch, TensorFlow, Scikit-Learn, XGBoost, Regression, Classification, Clustering
Data & Cloud: Pandas, NumPy, Spark, AWS, Docker, PostgreSQL, CI/CD, A/B Testing

EXPERIENCE
Data Scientist | Nexa Analytics | 2022 - Present
- Deployed customer churn prediction model in PyTorch and XGBoost, reducing churn by 18% ($1.2M annual savings).
- Built automated feature engineering pipeline in SQL and Spark processing 50M+ daily events.
- Executed 40+ production algorithm experiments using rigorous hypothesis testing and statistical analysis.

Junior ML Engineer | DataCore Solutions | 2020 - 2022
- Developed NLP sentiment classification models using HuggingFace and PyTorch with 92% F1 score.
- Packaged Scikit-Learn pipelines into Docker containers deployed on AWS.

EDUCATION
B.S. in Computer Science | UC Berkeley | 2020"""

SAMPLE_JOBS = {
    "Senior Machine Learning Engineer (Apex AI)": """Job Title: Senior Machine Learning Engineer
Company: Apex AI
Location: Remote
Seeking a Senior ML Engineer to architect, build, and deploy production machine learning and deep learning systems.
Requirements:
- 5+ years experience in Python, SQL, PyTorch, TensorFlow, and Scikit-Learn.
- Experience with MLOps, Docker, Kubernetes, Spark, and CI/CD pipelines.
- Strong background in deep learning, regression, classification, clustering, neural networks, and NLP.
- Demonstrated technical leadership and mentoring.""",
    "Lead Data Analyst & BI (Metro Financial)": """Job Title: Lead Data Analyst
Company: Metro Financial
Location: New York, NY
Looking for a Lead Data Analyst to transform complex financial data into executive intelligence models.
Requirements:
- 4+ years experience with SQL, Tableau, Power BI, Excel, and statistical modeling.
- Experience with Python or R for data analysis, A/B testing, and hypothesis testing.
- Strong communication, data modeling, business intelligence, and stakeholder management.""",
    "Full Stack Software Engineer (CloudVenture)": """Job Title: Full Stack Software Engineer
Company: CloudVenture
Location: Remote
Seeking a Full Stack Engineer to build responsive web applications and cloud microservices.
Requirements:
- 3+ years experience with TypeScript, JavaScript, React, Next.js, and Node.js.
- Strong knowledge of REST APIs, GraphQL, Docker, PostgreSQL, AWS, and Git.
- Experience with CI/CD, microservices architecture, and agile software development."""
}

# ---------------------------------------------------------------------------
# NLP & Matching Engine
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"&\w+;|<[^>]+>", " ", text).lower()
    text = re.sub(r"[^a-z0-9\s\+\#\.\-\/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_resume_text(uploaded_file) -> Tuple[str, Optional[str]]:
    if isinstance(uploaded_file, str):
        return uploaded_file.strip(), None
    fname = getattr(uploaded_file, "name", "").lower()
    if fname.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(uploaded_file) as pdf:
                txt = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
            return txt.strip(), None if txt.strip() else ("Warning: No selectable text in PDF.",)
        except Exception as e:
            return "", f"Failed to read PDF: {e}"
    if fname.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(uploaded_file.read()))
            txt = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return txt.strip(), None if txt.strip() else ("Warning: Empty DOCX file.",)
        except Exception as e:
            return "", f"Failed to read DOCX: {e}"
    return uploaded_file.read().decode("utf-8", errors="ignore").strip(), None

class SkillExtractor:
    def __init__(self, industry_key: str = "data_science"):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = spacy.blank("en")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.records = load_taxonomy_records(industry_key)
        self.surface_to_canonical = {}
        self.skill_categories = {}
        for r in self.records:
            c, cat = r["skill_name"], r["category"]
            self.skill_categories[c] = cat
            for s in [c] + r["synonyms"]:
                self.surface_to_canonical[s] = c
        self.matcher.add("SKILL", [self.nlp.make_doc(s) for s in self.surface_to_canonical])

    def extract(self, text: str) -> Set[str]:
        if not text or not isinstance(text, str):
            return set()
        doc = self.nlp(text)
        return {self.surface_to_canonical.get(doc[start:end].text.lower(), doc[start:end].text.lower()) for _, start, end in self.matcher(doc)}

def segment_skills(text: str, ext: SkillExtractor) -> str:
    for form in sorted(ext.surface_to_canonical, key=len, reverse=True):
        if len(form) >= 3:
            text = re.sub(re.escape(form), f" {form} ", text)
    return re.sub(r"\s+", " ", text).strip()

def group_skills(s_list: Any, ext: SkillExtractor) -> Dict[str, List[str]]:
    g = {"technical": [], "tool": [], "soft": []}
    for s in s_list:
        cat = ext.skill_categories.get(s, "technical")
        g[cat if cat in g else "technical"].append(s)
    return g

def jaccard(s_a: Set[str], s_b: Set[str]) -> float:
    return float(len(s_a & s_b) / len(s_a | s_b)) if s_a | s_b else 0.0

def compute_match(r_clean, r_skills, r_emb, j_clean, j_skills, j_emb, kw_weight=0.4):
    jac = jaccard(r_skills, j_skills)
    cos = float(cosine_similarity(r_emb.reshape(1, -1), j_emb.reshape(1, -1))[0][0])
    final = kw_weight * jac + (1.0 - kw_weight) * cos
    return {"jaccard": round(jac, 4), "semantic": round(cos, 4), "final_score": round(final, 4),
            "matched_skills": sorted(r_skills & j_skills), "missing_skills": sorted(j_skills - r_skills)}

def compute_ats_score(r_raw, r_clean, j_skills, r_skills):
    checks = []
    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", r_raw))
    checks.append(("Contact email address detected", has_email, 15))
    has_phone = bool(re.search(r"\b\+?[\d\-\s\(\)]{7,}\d\b", r_raw))
    checks.append(("Phone number detected", has_phone, 10))
    sec_count = sum(1 for kw in ["experience", "education", "skills", "project"] if kw in r_clean)
    checks.append((f"Standard sections found ({sec_count}/4)", sec_count >= 3, int(sec_count * 5)))
    cov = len(r_skills & j_skills) / len(j_skills) if j_skills else 0.0
    checks.append((f"Job keyword coverage ({int(cov*100)}%)", cov >= 0.4, int(cov * 25)))
    wc = len(r_clean.split())
    checks.append((f"Word count optimal ({wc} words)", 200 <= wc <= 1200, 10 if 200 <= wc <= 1200 else 0))
    metrics = bool(re.search(r"\d+%|\$\d+|\b\d{2,}\b", r_raw))
    checks.append(("Quantifiable impact metrics (%, $, numbers)", metrics, 10 if metrics else 0))
    action_verbs = any(re.search(rf"\b{v}\b", r_clean) for v in ["led", "developed", "architected", "engineered", "built", "deployed", "optimized", "designed", "analyzed", "automated"])
    checks.append(("Impact action verbs detected (architected, deployed, etc.)", action_verbs, 10 if action_verbs else 0))
    score = min(sum(pts for _, ok, pts in checks if ok), 100)
    return {"score": score, "checks": [(lbl, ok) for lbl, ok, _ in checks]}

SENIOR_SIGNALS = ["senior", "lead", "principal", "staff", "head of", "director", "manager", "5+ years", "7+ years", "10+ years"]
JUNIOR_SIGNALS = ["junior", "entry level", "entry-level", "associate", "0-2 years", "graduate", "intern"]

def detect_seniority(r_clean, j_clean, title):
    jd = (j_clean + " " + (title or "")).lower()
    if any(s in jd for s in SENIOR_SIGNALS) and not any(s in r_clean for s in SENIOR_SIGNALS):
        return {"type": "underqualified", "message": "⚠️ Seniority Notice: Role expects Senior/Lead signals; consider emphasizing leadership."}
    if any(s in jd for s in JUNIOR_SIGNALS) and any(s in r_clean for s in SENIOR_SIGNALS):
        return {"type": "overqualified", "message": "ℹ️ Note: Position is junior/entry level, but resume emphasizes senior leadership."}
    return None

BENCHMARK_TABLE = [(0.8, 99), (0.75, 95), (0.7, 90), (0.65, 82), (0.6, 72), (0.55, 60), (0.5, 48), (0.4, 25), (0.3, 8), (0.0, 1)]
def get_benchmark(score):
    return next((pct for th, pct in BENCHMARK_TABLE if score >= th), 1)

def rank_missing(missing, title, j_clean, ext):
    t_clean = (title or "").lower()
    scored = []
    for skill in missing:
        forms = [f for f, c in ext.surface_to_canonical.items() if c == skill] or [skill]
        score = sum(5 if f in t_clean else 0 for f in forms) + sum(j_clean.count(f) * 2 for f in forms)
        tier = "High" if score >= 6 else ("Medium" if score >= 2 else "Low")
        scored.append((skill, tier, score))
    return sorted(scored, key=lambda x: x[2], reverse=True)

SUGG_TEMPLATES = {
    "technical": "Add a project accomplishment where you utilized **{skill}** with measurable outcomes (e.g., 'Implemented {skill} to improve throughput by 25%').",
    "tool": "Explicitly feature **{skill}** in your core skills and tools section for automated keyword indexing.",
    "soft": "Illustrate **{skill}** through cross-functional team leadership or measurable project delivery.",
}

def generate_suggestions(ranked_missing, j_raw, r_raw, ext, api_key=None, top_n=5):
    key = api_key or os.getenv("GEMINI_API_KEY")
    if key and key.strip() != "your_gemini_api_key_here":
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=key)
            skills_str = ", ".join(s for s, _, _ in ranked_missing[:top_n])
            prompt = f"Resume writer task: For each missing skill [{skills_str}], write ONE concise, quantified resume bullet point. JD: {j_raw[:1000]}. Resume: {r_raw[:1000]}. Numbered list only."
            res = client.models.generate_content(model="gemini-3.8-flash", contents=prompt, config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=600))
            lines = [re.sub(r"^\d+[\.\)]\s*", "", l).strip() for l in res.text.strip().split("\n") if l.strip()]
            return [{"skill": s, "priority": t, "suggestion": lines[i] if i < len(lines) else SUGG_TEMPLATES[ext.skill_categories.get(s, "technical")].format(skill=s), "ai": i < len(lines)}
                    for i, (s, t, _) in enumerate(ranked_missing[:top_n])]
        except Exception:
            pass
    return [{"skill": s, "priority": t, "suggestion": SUGG_TEMPLATES[ext.skill_categories.get(s, "technical")].format(skill=s), "ai": False} for s, t, _ in ranked_missing[:top_n]]


def fetch_url(url: str):
    if not url.strip():
        return "", "Invalid URL."
    url = "https://" + url.strip() if not url.strip().startswith(("http://", "https://")) else url.strip()
    if any(d in url for d in ["linkedin.com", "lnkd.in"]):
        return "", "LinkedIn blocks automated scraping. Please paste text directly."
    try:
        r = requests.get("https://r.jina.ai/" + url, headers={"Accept": "text/plain"}, timeout=15)
        txt = re.sub(r"!\[.*?\]\(.*?\)|\[([^\]]+)\]\([^)]+\)", r"\2", r.text.strip())
        return (txt, None) if len(txt) >= 100 else ("", "Page content too short.")
    except Exception as e:
        return "", str(e)

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_donut(matched: int, missing: int) -> go.Figure:
    tot = matched + missing
    pct = int((matched / tot) * 100) if tot else 0
    fig = go.Figure(go.Pie(labels=["Matched", "Missing"], values=[matched, missing], hole=0.68,
                           marker=dict(colors=["#22c55e", "#ef4444"], line=dict(color="#0f172a", width=2)),
                           textinfo="label+value", hoverinfo="label+percent"))
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=240, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                      annotations=[dict(text=f"<b>{pct}%</b><br><span style='font-size:11px;color:#94a3b8'>Coverage</span>", x=0.5, y=0.5, font=dict(size=20, color="white"), showarrow=False)])
    return fig

def plot_bar(results: List[Dict[str, Any]]) -> go.Figure:
    labels = [r["label"] for r in results]
    fig = go.Figure([
        go.Bar(name="Overall Match", y=labels, x=[int(r["final_score"]*100) for r in results], orientation="h", marker_color="#6366f1", text=[f"{int(r['final_score']*100)}%" for r in results], textposition="outside"),
        go.Bar(name="Keywords", y=labels, x=[int(r["jaccard"]*100) for r in results], orientation="h", marker_color="#22c55e"),
        go.Bar(name="Semantic", y=labels, x=[int(r["semantic"]*100) for r in results], orientation="h", marker_color="#f59e0b"),
    ])
    fig.update_layout(barmode="group", height=max(250, len(labels)*60+80), margin=dict(t=20, b=20, l=10, r=60), paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                      font=dict(color="white"), xaxis=dict(range=[0, 115], showgrid=False, ticksuffix="%", color="#94a3b8"), yaxis=dict(color="#cbd5e1"))
    return fig

# ---------------------------------------------------------------------------
# Streamlit Application Interface
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Resume ↔ Job Matcher", page_icon="🧩", layout="wide")
st.markdown("<style>.stApp { background:#0f1117; } div[data-testid='metric-container'] { background:#1a1d2e; border:1px solid #2d3748; border-radius:12px; padding:16px; } .tag-high { background:#7f1d1d; color:#fca5a5; padding:2px 8px; border-radius:8px; font-size:12px; font-weight:600; } .tag-medium { background:#78350f; color:#fcd34d; padding:2px 8px; border-radius:8px; font-size:12px; font-weight:600; } .tag-low { background:#1f2937; color:#9ca3af; padding:2px 8px; border-radius:8px; font-size:12px; font-weight:600; }</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    ind_key = TAXONOMY_OPTIONS[st.selectbox("Industry Preset", list(TAXONOMY_OPTIONS))]
    kw_w = st.slider("Keyword vs Semantic Weight", 0.0, 1.0, 0.4, 0.05)
    model_name = "all-MiniLM-L6-v2" if "MiniLM" in st.selectbox("Embedding Model", ["Fast — all-MiniLM-L6-v2 (~80 MB)", "all-mpnet-base-v2 (~420 MB)"]) else "all-mpnet-base-v2"
    st.divider()
    gemini_key = st.text_input("Gemini API Key (Optional)", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    if gemini_key:
        st.success("✅ AI Rewrites Active")

@st.cache_resource
def get_ext(k): return SkillExtractor(k)
@st.cache_resource
def get_emb(m): return SentenceTransformer(m)
ext, embed_model = get_ext(ind_key), get_emb(model_name)

if "resume_clean" not in st.session_state:
    st.session_state.update({"resume_raw": "", "resume_clean": "", "resume_skills": set(), "resume_emb": None, "label": "", "job_bank": []})

def run_analysis(title, text_raw):
    j_clean = clean_text(text_raw)
    j_skills = ext.extract(segment_skills(j_clean, ext))
    j_emb = embed_model.encode(j_clean)
    res = compute_match(st.session_state.resume_clean, st.session_state.resume_skills, st.session_state.resume_emb, j_clean, j_skills, j_emb, kw_w)
    res.update({"title": title, "raw": text_raw, "clean": j_clean, "skills": j_skills, "ats": compute_ats_score(st.session_state.resume_raw, st.session_state.resume_clean, j_skills, st.session_state.resume_skills), "seniority": detect_seniority(st.session_state.resume_clean, j_clean, title), "bench": get_benchmark(res["final_score"])})
    return res

def render_result(r):
    score = int(r["final_score"] * 100)
    col = "#22c55e" if score >= 70 else ("#f59e0b" if score >= 50 else "#ef4444")
    st.markdown(f"<div style='background:#1a1d2e;border:1px solid #2d3748;border-radius:12px;padding:16px;margin:12px 0;'><h2 style='color:{col};margin:0;'>{score}% Overall Match</h2><p style='color:#94a3b8;margin:4px 0 0;'>Estimated to beat ~<b>{r['bench']}%</b> of applicants</p></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Keywords", f"{int(r['jaccard']*100)}%")
    c2.metric("Semantic", f"{int(r['semantic']*100)}%")
    c3.metric("ATS Score", f"{r['ats']['score']}/100")
    c4.metric("Skills Matched", f"{len(r['matched_skills'])}/{len(r['matched_skills']) + len(r['missing_skills'])}")
    if r["seniority"]:
        st.warning(r["seniority"]["message"])
    with st.expander("📋 ATS Checklist"):
        for lbl, ok in r["ats"]["checks"]:
            st.write(f"{'✅' if ok else '⚠️'} {lbl}")
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        st.plotly_chart(plot_donut(len(r["matched_skills"]), len(r["missing_skills"])), use_container_width=True)
    with sc2:
        m_grp = group_skills(r["matched_skills"], ext)
        ms_grp = group_skills(r["missing_skills"], ext)
        def cat_tag(cat_name, m_items, ms_items):
            tot = len(m_items) + len(ms_items)
            pct = int((len(m_items) / tot) * 100) if tot else 100
            return f"{cat_name}: **{pct}%** ({len(m_items)}/{tot})" if tot else None
        c_tags = [t for t in [cat_tag("Core Tech", m_grp["technical"], ms_grp["technical"]),
                              cat_tag("Tools & Cloud", m_grp["tool"], ms_grp["tool"]),
                              cat_tag("Soft Skills", m_grp["soft"], ms_grp["soft"])] if t]
        if c_tags:
            st.markdown("📊 **Skill Breakdown:** " + " | ".join(c_tags))
        st.markdown(f"**🟢 Matched ({len(r['matched_skills'])}):** " + (", ".join(f"`{s}`" for s in r["matched_skills"]) or "None"))
        if m_grp["tool"]: st.caption("Tools/Platforms: " + ", ".join(f"`{s}`" for s in m_grp["tool"]))
        st.markdown(f"**🔴 Missing ({len(r['missing_skills'])}):** " + (", ".join(f"`{s}`" for s in r["missing_skills"]) or "None"))
        if ms_grp["tool"]: st.caption("Tools/Platforms Missing: " + ", ".join(f"`{s}`" for s in ms_grp["tool"]))
    if r["missing_skills"]:
        st.markdown("### 💡 Recommended Bullet Improvements")
        ranked = rank_missing(r["missing_skills"], r["title"], r["clean"], ext)
        suggs = generate_suggestions(ranked, r["raw"], st.session_state.resume_raw, ext, gemini_key)
        st.session_state["last_suggestions"] = suggs
        for s in suggs:
            badge = "✨ AI" if s["ai"] else "Template"
            st.markdown(f"**{s['skill']}** <span class='tag-{s['priority'].lower()}'>{s['priority']} Priority</span> [{badge}]<br>{s['suggestion']}", unsafe_allow_html=True)

    with st.expander("📊 Full Skill Gap & ATS Coverage Matrix"):
        all_skills = sorted(list(r["matched_skills"]) + list(r["missing_skills"]))
        matrix_data = []
        for s in all_skills:
            is_matched = s in r["matched_skills"]
            cat = ext.skill_categories.get(s, "technical").capitalize()
            matrix_data.append({"Skill": s, "Category": cat, "In Resume": "✅ Present" if is_matched else "❌ Missing", "ATS Status": "Matched" if is_matched else "Critical Gap"})
        st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)

    # 1-Click Action Report Download
    report_md = f"""# Resume Match & ATS Audit Report
**Target Role:** {r['title'] or 'Target Role'}
**Overall Match Score:** {score}%
**Estimated Applicant Percentile:** Beats ~{r['bench']}% of applicants
**ATS Compatibility Score:** {r['ats']['score']}/100
**Keyword Overlap:** {int(r['jaccard']*100)}%
**Semantic Similarity:** {int(r['semantic']*100)}%

## Matched Skills ({len(r['matched_skills'])})
{', '.join(r['matched_skills']) if r['matched_skills'] else 'None'}

## Missing Skills ({len(r['missing_skills'])})
{', '.join(r['missing_skills']) if r['missing_skills'] else 'None'}

## ATS Diagnostic Breakdown
""" + "\n".join(f"- [{'x' if ok else ' '}] {lbl}" for lbl, ok in r['ats']['checks'])
    st.download_button("📥 Download Analysis Report (Markdown)", data=report_md, file_name=f"Match_Report_{re.sub(r'[^a-zA-Z0-9_]', '_', r['title'] or 'job')}.md", mime="text/markdown", use_container_width=True)

# Main UI
st.markdown("<h1 style='text-align:center;color:#6366f1;'>🧩 Resume ↔ Job Matcher & ATS Analytics</h1>", unsafe_allow_html=True)
st.markdown("## Step 1 — Load Resume")
if st.session_state.resume_clean:
    st.success(f"✅ Loaded: **{st.session_state.label}** ({len(st.session_state.resume_skills)} skills extracted)")
    with st.expander(f"🔎 View Detected Skills in Resume ({len(st.session_state.resume_skills)} skills)"):
        rg = group_skills(st.session_state.resume_skills, ext)
        if rg["technical"]: st.markdown("**Technical Core:** " + ", ".join(f"`{s}`" for s in sorted(rg["technical"])))
        if rg["tool"]: st.markdown("**Tools & Cloud:** " + ", ".join(f"`{s}`" for s in sorted(rg["tool"])))
        if rg["soft"]: st.markdown("**Soft Skills:** " + ", ".join(f"`{s}`" for s in sorted(rg["soft"])))
    if st.button("🔄 Change Resume"):
        st.session_state.update({"resume_raw": "", "resume_clean": "", "resume_skills": set(), "resume_emb": None, "label": "", "last_analysis": None})
        st.rerun()
else:
    mode = st.radio("Resume Source", ["Use Preloaded Sample Resume", "Upload File (PDF / DOCX)"], horizontal=True)
    if mode == "Use Preloaded Sample Resume":
        if st.button("✅ Load Alex Chen (Data Scientist Sample)", type="primary"):
            c = clean_text(SAMPLE_RESUME)
            st.session_state.update({"resume_raw": SAMPLE_RESUME, "resume_clean": c, "resume_skills": ext.extract(segment_skills(c, ext)), "resume_emb": embed_model.encode(c), "label": "Alex Chen (Sample)", "last_analysis": None})
            st.rerun()
    else:
        f = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
        if f and st.button("✅ Analyze Uploaded File", type="primary"):
            txt, err = extract_resume_text(f)
            if err: st.error(err)
            else:
                c = clean_text(txt)
                st.session_state.update({"resume_raw": txt, "resume_clean": c, "resume_skills": ext.extract(segment_skills(c, ext)), "resume_emb": embed_model.encode(c), "label": f.name, "last_analysis": None})
                st.rerun()

st.divider()
if not st.session_state.resume_clean:
    st.info("⬆️ Load a resume above to proceed.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔍 Analyze Single Job", "📊 Compare Multiple Jobs", "✏️ Live Resume Editor"])

with tab1:
    inp_mode = st.radio("Input Format", ["Quick Load Sample Job", "Paste Job Text", "Fetch URL"], horizontal=True)
    title = ""
    j_text = ""
    if inp_mode == "Quick Load Sample Job":
        chosen_j = st.selectbox("Select Target Job", list(SAMPLE_JOBS.keys()))
        title = chosen_j.split(" (")[0]
        j_text = SAMPLE_JOBS[chosen_j]
        with st.expander("Preview Job Description"):
            st.text(j_text)
    elif inp_mode == "Paste Job Text":
        title = st.text_input("Job Title", placeholder="e.g. Senior Machine Learning Engineer")
        j_text = st.text_area("Job Description", height=200)
    else:
        title = st.text_input("Job Title", placeholder="e.g. Senior Machine Learning Engineer")
        url = st.text_input("Posting URL")
        if url and st.button("Fetch"):
            txt, err = fetch_url(url)
            if err: st.error(err)
            else: st.session_state["f_text"] = txt; st.success("Fetched!")
        j_text = st.session_state.get("f_text", "")
    if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
        if j_text.strip():
            st.session_state["last_analysis"] = run_analysis(title, j_text)
            st.session_state["last_job_text"] = j_text
            st.session_state["last_job_title"] = title
        else:
            st.warning("Please provide job text.")
    if st.session_state.get("last_analysis"):
        render_result(st.session_state["last_analysis"])

with tab2:
    st.markdown("### Compare Multiple Roles")
    tb_c1, tb_c2 = st.columns([1, 1])
    with tb_c1:
        if st.button("⚡ Load 3 Sample Roles into Bank"):
            st.session_state.job_bank = [{"title": k, "text": v} for k, v in SAMPLE_JOBS.items()]
            st.rerun()
    with tb_c2:
        if st.session_state.job_bank and st.button("🗑️ Clear Bank"):
            st.session_state.job_bank = []
            st.rerun()

    ctitle = st.text_input("Role Title", key="c_t")
    ctext = st.text_area("Role Description", height=120, key="c_d")
    if st.button("➕ Add to Comparison Bank") and ctext.strip():
        st.session_state.job_bank.append({"title": ctitle or f"Job {len(st.session_state.job_bank)+1}", "text": ctext})
        st.rerun()
    if st.session_state.job_bank:
        for i, j in enumerate(st.session_state.job_bank):
            st.write(f"**{i+1}.** {j['title']}")
        if st.button("⚡ Run Comparison", type="primary", use_container_width=True):
            res_list = [{**run_analysis(j['title'], j['text']), "label": j['title']} for j in st.session_state.job_bank]
            st.dataframe(pd.DataFrame([{
                "Job": r["label"], "Overall Match": f"{int(r['final_score']*100)}%",
                "Keywords": f"{int(r['jaccard']*100)}%", "Semantic": f"{int(r['semantic']*100)}%",
                "ATS": f"{r['ats']['score']}/100", "Matched Skills": len(r["matched_skills"]),
                "Missing Skills": len(r["missing_skills"])
            } for r in sorted(res_list, key=lambda x: x["final_score"], reverse=True)]), use_container_width=True)
            st.plotly_chart(plot_bar(res_list), use_container_width=True)

with tab3:
    st.markdown("### ✏️ Real-Time Resume Editor & Optimizer")
    st.caption("Edit your resume content to see real-time match score delta and export your optimized version.")
    if st.session_state.get("last_suggestions"):
        if st.button("⚡ Auto-Append Recommended Missing Skill Bullets into Resume"):
            appended = "\n\nKEY ACHIEVEMENTS & SKILLS:\n" + "\n".join(f"- {s['suggestion']}" for s in st.session_state["last_suggestions"])
            st.session_state.resume_raw = st.session_state.resume_raw + appended
            st.rerun()
    c_e1, c_e2 = st.columns(2)
    with c_e1:
        ed_res = st.text_area("Edit Resume", value=st.session_state.resume_raw, height=350, key="ed_res_box")
    with c_e2:
        default_job = st.session_state.get("last_job_text", "")
        ed_job = st.text_area("Target Job Text", value=default_job, height=350, key="ed_job_box")
    col_ed1, col_ed2 = st.columns([1, 1])
    with col_ed1:
        if st.button("🔄 Recalculate Delta", type="primary", use_container_width=True) and ed_res and ed_job:
            c1, c2 = clean_text(ed_res), clean_text(ed_job)
            s1, s2 = ext.extract(segment_skills(c1, ext)), ext.extract(segment_skills(c2, ext))
            e1, e2 = embed_model.encode(c1), embed_model.encode(c2)
            new_res = compute_match(c1, s1, e1, c2, s2, e2, kw_w)
            orig_res = compute_match(st.session_state.resume_clean, st.session_state.resume_skills, st.session_state.resume_emb, c2, s2, e2, kw_w)
            delta = int(new_res["final_score"]*100) - int(orig_res["final_score"]*100)
            st.metric("New Score", f"{int(new_res['final_score']*100)}%", delta=f"{delta:+d}%")
    with col_ed2:
        st.download_button("📥 Download Revised Resume (.txt)", data=ed_res, file_name="tailored_resume.txt", mime="text/plain", use_container_width=True)
