import html
import json
import re
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillMatch Jobs", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
    .project-title {
        font-size: 16px;
        color: #8A8F98;
        font-style: italic;
        margin-bottom: 6px;
    }
    .sub-header {
        font-size: 17px;
        color: #B0B4BB;
        margin-bottom: 10px;
    }
    .job-card {
        background: linear-gradient(145deg, #1A1C24, #15161C);
        padding: 20px 22px;
        border-radius: 14px;
        margin-bottom: 18px;
        border: 1px solid #2A2D37;
        border-left: 5px solid #4D7FFF;
    }
    .job-title {
        font-size: 18px;
        font-weight: 700;
        color: #FFFFFF;
    }
    .job-company {
        font-size: 14px;
        color: #8A8F98;
        margin-bottom: 10px;
    }
    .badge-strong { background-color: #1F3D2B; color: #4ADE80; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .badge-moderate { background-color: #3D351F; color: #FACC15; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .badge-low { background-color: #3D1F1F; color: #F87171; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .skill-line {
        margin-top: 12px;
        margin-bottom: 4px;
        font-size: 13px;
        color: #B0B4BB;
    }
    .apply-btn {
        display: inline-block;
        margin-top: 10px;
        background-color: #4D7FFF;
        color: white !important;
        padding: 6px 16px;
        border-radius: 8px;
        text-decoration: none !important;
        font-size: 13px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Big title: drawn inside its own iframe so Streamlit's styles cannot shrink it
components.html(
    """
    <div style="font-family: 'Source Sans Pro', 'Segoe UI', Arial, sans-serif;
                font-size: 80px; font-weight: 800; color: #4D7FFF;
                line-height: 1.1; letter-spacing: -1px; margin: 0; padding: 0;">
        🎯 SkillMatch Jobs
    </div>
    """,
    height=110,
)
st.markdown('<p class="project-title">IT Job Recommendation System for Freshers</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Skill-based job matching with ATS-style compatibility scoring</p>', unsafe_allow_html=True)
st.divider()

with open("saved_jobs.json") as f:
    all_data = json.load(f)

# Skills the system looks for (developer, data analyst and QA roles)
common_skills = [
    "python", "java", "javascript", "c++", "sql", "mysql", "mongodb", "html", "css",
    "react", "angular", "node.js", "php", "django", "flask", "spring", "api", "rest",
    "git", "linux", "aws", "azure", "cloud", "docker", "kubernetes",
    "machine learning", "pandas", "numpy", "excel", "power bi", "tableau", "statistics",
    "selenium", "testing", "manual testing", "automation testing",
    "communication", "problem solving",
]

SENIOR_TITLE_KEYWORDS = ["senior", "sr.", "lead", "manager", "principal", "head of",
                         "director", "architect", "staff", "vp", "chief", "sr "]

def is_fresher_friendly(title):
    title_lower = title.lower()
    return not any(word in title_lower for word in SENIOR_TITLE_KEYWORDS)

def has_skill(text, skill):
    # Whole-word match, so "excel" does not match "excellent" and "java" does not match "javascript"
    pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None

def skill_gap(description, resume_text):
    desc = description.lower()
    resume = resume_text.lower()
    required = [s for s in common_skills if has_skill(desc, s)]
    missing = [s for s in required if not has_skill(resume, s)]
    return required, missing

def company_and_location(job):
    # Some postings come without an employer name (agencies / job boards hide it)
    company_info = job.get("company") or {}
    company_name = company_info.get("display_name") or company_info.get("canonical_name")
    if not company_name:
        company_name = "Company not disclosed"
    location_info = job.get("location") or {}
    location = location_info.get("display_name", "")
    line = html.escape(company_name)
    if location:
        line += " · 📍 " + html.escape(location)
    return line

def ats_badge(percent):
    if percent >= 70:
        return "Strong Match", "badge-strong", "🟢"
    elif percent >= 40:
        return "Moderate Match", "badge-moderate", "🟡"
    else:
        return "Low Match", "badge-low", "🔴"

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown("#### Your Profile")
    resume_input = st.text_area("Paste your resume / skills", 
                                 "Python, pandas, SQL, basic AWS knowledge, good communication skills",
                                 height=160)
    find_clicked = st.button("🔍 Find My Matches", type="primary", use_container_width=True)

with right_col:
    if find_clicked:
        data = all_data
        fresher_jobs = [job for job in data["results"] if is_fresher_friendly(job.get("title", ""))]
        if not fresher_jobs:
            fresher_jobs = data["results"]

        if not fresher_jobs:
            st.warning("No jobs found in saved_jobs.json. Run fetch_jobs.py once to load jobs.")
            st.stop()

        job_descriptions = [job.get("description", "") for job in fresher_jobs]
        all_texts = [resume_input] + job_descriptions

        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(all_texts)

        resume_vector = vectors[0]
        job_vectors = vectors[1:]
        scores = cosine_similarity(resume_vector, job_vectors)[0]
        max_score = max(scores) if max(scores) > 0 else 1
        scaled_scores = [(s / max_score) * 100 for s in scores]

        ranked = sorted(zip(fresher_jobs, scores, scaled_scores), key=lambda x: x[1], reverse=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Jobs Scanned", len(fresher_jobs))
        m2.metric("Top Match", f"{round(max(scaled_scores))}%")
        strong_count = sum(1 for s in scaled_scores if s >= 70)
        m3.metric("Strong Matches", strong_count)

        st.markdown("#### Top Job Matches")
        for job, raw_score, ats_percent in ranked[:10]:
            title = html.escape(job.get("title", "No title"))
            company_line = company_and_location(job)
            description = job.get("description", "")
            apply_link = html.escape(job.get("redirect_url", ""), quote=True)
            required, missing = skill_gap(description, resume_input)
            label, badge_class, emoji = ats_badge(ats_percent)

            if not required:
                gap_html = ("<div class='skill-line'>ℹ️ The preview doesn't list skills. "
                            "Click Apply Now for the full requirements.</div>")
            elif not missing:
                gap_html = ("<div class='skill-line'>✅ You have every skill this posting mentions: <b>"
                            + ", ".join(required) + "</b></div>")
            else:
                gap_html = ("<div class='skill-line'>Skills mentioned: " + ", ".join(required) + "</div>"
                            "<div class='skill-line'>Missing keywords: <b>" + ", ".join(missing) + "</b></div>")

            apply_html = f'<a href="{apply_link}" target="_blank" class="apply-btn">Apply Now →</a>' if apply_link else ""

            st.markdown(f"""
                <div class="job-card">
                    <div class="job-title">{title}</div>
                    <div class="job-company">{company_line}</div>
                    <span class="{badge_class}">{emoji} {round(ats_percent)}% · {label}</span>
                    {gap_html}
                    {apply_html}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Paste your skills on the left and click **Find My Matches** to see your results here.")