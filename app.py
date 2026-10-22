import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillMatch Jobs", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
    .main-header {
        font-size: 90px;
        font-weight: 800;
        color: #4D7FFF;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    .project-title {
        font-size: 14px;
        color: #8A8F98;
        font-style: italic;
        margin-bottom: 6px;
    }
    .sub-header {
        font-size: 16px;
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

st.markdown('<p class="main-header">🎯 SkillMatch Jobs</p>', unsafe_allow_html=True)
st.markdown('<p class="project-title">IT Job Recommendation System for Freshers</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Skill-based job matching with ATS-style compatibility scoring</p>', unsafe_allow_html=True)
st.divider()

with open("saved_jobs.json") as f:
    all_data = json.load(f)

common_skills = ["python", "sql", "aws", "java", "docker", "react", "pandas",
                  "machine learning", "excel", "communication", "cloud", "linux"]

SENIOR_TITLE_KEYWORDS = ["senior", "sr.", "lead", "manager", "principal", "head of",
                         "director", "architect", "staff", "vp", "chief", "sr "]

def is_fresher_friendly(title):
    title_lower = title.lower()
    return not any(word in title_lower for word in SENIOR_TITLE_KEYWORDS)

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

        resume_skills = set(s.strip().lower() for s in resume_input.split(","))
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

        def missing_skills(description, resume_skills):
            job_text = description.lower()
            required = [s for s in common_skills if s in job_text]
            return [s for s in required if s not in resume_skills]

        m1, m2, m3 = st.columns(3)
        m1.metric("Jobs Scanned", len(fresher_jobs))
        m2.metric("Top Match", f"{round(scaled_scores[0] if len(scaled_scores) else 0)}%")
        strong_count = sum(1 for s in scaled_scores if s >= 70)
        m3.metric("Strong Matches", strong_count)

        st.markdown("#### Top Job Matches")
        for job, raw_score, ats_percent in ranked[:10]:
            title = job.get("title", "No title")
            company = job.get("company", {}).get("display_name", "Unknown company")
            description = job.get("description", "")
            apply_link = job.get("redirect_url", "")
            gaps = missing_skills(description, resume_skills)
            gap_text = ", ".join(gaps) if gaps else "None identified"
            label, badge_class, emoji = ats_badge(ats_percent)

            apply_html = f'<a href="{apply_link}" target="_blank" class="apply-btn">Apply Now →</a>' if apply_link else ""

            st.markdown(f"""
                <div class="job-card">
                    <div class="job-title">{title}</div>
                    <div class="job-company">{company}</div>
                    <span class="{badge_class}">{emoji} {round(ats_percent)}% · {label}</span>
                    <p style="margin-top:12px; margin-bottom:4px; font-size:13px; color:#B0B4BB;">
                        Missing keywords: <b>{gap_text}</b>
                    </p>
                    {apply_html}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Paste your skills on the left and click **Find My Matches** to see your results here.")