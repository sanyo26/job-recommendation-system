import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillMatch Jobs", page_icon="🎯", layout="centered")

st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: 800;
        color: #2E5BFF;
        margin-bottom: 0px;
    }
    .project-title {
        font-size: 15px;
        color: #AAAAAA;
        font-style: italic;
        margin-bottom: 8px;
    }
    .sub-header {
        font-size: 16px;
        color: #888888;
        margin-bottom: 25px;
    }
    .job-card {
        background-color: #1E2029;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 4px solid #2E5BFF;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎯 SkillMatch Jobs</p>', unsafe_allow_html=True)
st.markdown('<p class="project-title">IT Job Recommendation System for Freshers</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Skill-based job matching with ATS-style compatibility scoring, built for freshers</p>', unsafe_allow_html=True)

resume_input = st.text_area("Paste your resume / skills (comma-separated)", 
                             "Python, pandas, SQL, basic AWS knowledge, good communication skills",
                             height=120)

common_skills = ["python", "sql", "aws", "java", "docker", "react", "pandas",
                  "machine learning", "excel", "communication", "cloud", "linux"]

SENIOR_TITLE_KEYWORDS = ["senior", "sr.", "lead", "manager", "principal", "head of",
                         "director", "architect", "staff", "vp", "chief", "sr "]

def is_fresher_friendly(title):
    title_lower = title.lower()
    return not any(word in title_lower for word in SENIOR_TITLE_KEYWORDS)

def ats_label(percent):
    if percent >= 70:
        return "Strong Match", "🟢"
    elif percent >= 40:
        return "Moderate Match", "🟡"
    else:
        return "Low Match", "🔴"

if st.button("🔍 Find Matches", type="primary"):
    with open("saved_jobs.json") as f:
        data = json.load(f)

    fresher_jobs = [job for job in data["results"] 
                     if is_fresher_friendly(job.get("title", ""))]

    if not fresher_jobs:
        st.info("No titles were excluded by the fresher filter — showing all available matches.")
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

    st.subheader(f"Top {min(10, len(ranked))} Job Matches")
    for job, raw_score, ats_percent in ranked[:10]:
        title = job.get("title", "No title")
        company = job.get("company", {}).get("display_name", "Unknown company")
        description = job.get("description", "")
        apply_link = job.get("redirect_url", "")
        gaps = missing_skills(description, resume_skills)
        gap_text = ", ".join(gaps) if gaps else "None identified"
        label, emoji = ats_label(ats_percent)

        with st.container():
            st.markdown(f"""
                <div class="job-card">
                <b>{title}</b> — {company}<br>
                {emoji} <b>ATS Match: {round(ats_percent)}%</b> ({label})
                </div>
            """, unsafe_allow_html=True)
            st.progress(min(int(ats_percent), 100))
            st.write(f"Missing keywords: {gap_text}")
            if apply_link:
                st.markdown(f"[Apply Here →]({apply_link})")
            st.write("")