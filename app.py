import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("IT Job Recommendation System for Freshers")
st.write("Paste your resume or skill list below to find your best-matching jobs.")

resume_input = st.text_area("Your resume / skills (comma-separated)", 
                             "Python, pandas, SQL, basic AWS knowledge, good communication skills")

common_skills = ["python", "sql", "aws", "java", "docker", "react", "pandas",
                  "machine learning", "excel", "communication", "cloud", "linux"]

if st.button("Find Matches"):
    with open("saved_jobs.json") as f:
        data = json.load(f)

    resume_skills = set(s.strip().lower() for s in resume_input.split(","))
    job_descriptions = [job.get("description", "") for job in data["results"]]
    all_texts = [resume_input] + job_descriptions

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(all_texts)

    resume_vector = vectors[0]
    job_vectors = vectors[1:]
    scores = cosine_similarity(resume_vector, job_vectors)[0]

    ranked = sorted(zip(data["results"], scores), key=lambda x: x[1], reverse=True)

    def missing_skills(description, resume_skills):
        job_text = description.lower()
        required = [s for s in common_skills if s in job_text]
        return [s for s in required if s not in resume_skills]

    st.subheader("Top 5 Job Matches")
    for job, score in ranked[:5]:
        title = job.get("title", "No title")
        company = job.get("company", {}).get("display_name", "Unknown company")
        description = job.get("description", "")
        gaps = missing_skills(description, resume_skills)
        gap_text = ", ".join(gaps) if gaps else "None identified"

        st.markdown(f"**{title}** — {company}")
        st.write(f"Match Score: {round(score, 3)}")
        st.write(f"Missing skills: {gap_text}")
        st.divider()