import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample resume text (you can change this to test different skill sets)
resume_text = "Python, pandas, SQL, basic AWS knowledge, good communication skills, quick learner"
resume_skills = set(s.strip().lower() for s in resume_text.split(","))

# A fixed list of common skills to check for in job descriptions
common_skills = ["python", "sql", "aws", "java", "docker", "react", "pandas",
                  "machine learning", "excel", "communication", "cloud", "linux"]

# Load the job data we saved earlier
with open("saved_jobs.json") as f:
    data = json.load(f)

print(f"Loaded {len(data['results'])} jobs from saved_jobs.json")

# Pull out just the job descriptions as plain text
job_descriptions = [job.get("description", "") for job in data["results"]]

# Combine resume + all job descriptions into one list
all_texts = [resume_text] + job_descriptions

# Convert all of this text into numeric "fingerprints" using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(all_texts)

# The first vector is the resume; the rest are job descriptions
resume_vector = vectors[0]
job_vectors = vectors[1:]

# Compare the resume against every job, get a similarity score for each
scores = cosine_similarity(resume_vector, job_vectors)[0]

# Pair each job with its score, then sort highest-first
ranked = sorted(zip(data["results"], scores), key=lambda x: x[1], reverse=True)

def missing_skills(job_description, resume_skills):
    job_text = job_description.lower()
    required = [s for s in common_skills if s in job_text]
    missing = [s for s in required if s not in resume_skills]
    return missing

print("\nTop 5 Job Matches with Skill Gaps:\n")
for job, score in ranked[:5]:
    title = job.get("title", "No title")
    company = job.get("company", {}).get("display_name", "Unknown company")
    description = job.get("description", "")
    gaps = missing_skills(description, resume_skills)
    gap_text = ", ".join(gaps) if gaps else "None identified"
    print(f"{title} - {company} | Match Score: {round(score, 3)}")
    print(f"   Missing skills: {gap_text}\n")