import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("IT Job Recommendation System for Freshers")
st.write("Paste your resume or skill list below to find your best-matching jobs, with an ATS-style compatibility score for each.")

resume_input = st.text_area("Your resume / skills (comma-separated)", 
                             "Python, pandas, SQL, basic AWS knowledge, good communication skills")

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

if st.button("Find Matches"):
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

    # Scale raw cosine similarity (small numbers) into a 0-100 ATS-style percentage
    max_score = max(scores) if max(scores) > 0 else 1
    scaled_scores = [(s / max_score) * 100 for s in scores]

    ranked = sorted(zip(fresher_jobs, scores, scaled_scores), key=lambda x: x[1], reverse=True)

    def missing_skills(description, resume_skills):
        job_text = description.lower()
        required = [s for s in common_skills if s in job_text]
        return [s for s in required if s not in resume_skills]

    st.subheader("Top 5 Job Matches — ATS Compatibility Score")
    for job, raw_score, ats_percent in ranked[:5]:
        title = job.get("title", "No title")
        company = job.get("company", {}).get("display_name", "Unknown company")
        description = job.get("description", "")
        apply_link = job.get("redirect_url", "")
        gaps = missing_skills(description, resume_skills)
        gap_text = ", ".join(gaps) if gaps else "None identified"
        label, emoji = ats_label(ats_percent)

        st.markdown(f"**{title}** — {company}")
        st.write(f"{emoji} ATS Match: {round(ats_percent)}% ({label})")
        st.progress(min(int(ats_percent), 100))
        st.write(f"Missing keywords: {gap_text}")
        if apply_link:
            st.markdown(f"[Apply Here]({apply_link})")
        st.divider()