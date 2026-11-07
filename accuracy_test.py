import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SENIOR_TITLE_KEYWORDS = ["senior", "sr.", "lead", "manager", "principal", "head of",
                         "director", "architect", "staff", "vp", "chief", "sr "]

# Write each test resume and decide the expected role BEFORE you run this script.
# "expected" = words that should appear in the title of a correct job.
TESTS = [
    ("Python fresher",
     "BCA graduate. Skills: Python, Django, Flask, SQL, Git, REST API. Projects: library management "
     "system in Django with MySQL, expense tracker in Flask. Internship: python developer intern. "
     "Education: BCA, 8.5 CGPA.",
     ["python"]),
    ("Java fresher",
     "BCA graduate. Skills: Java, Spring, Hibernate, MySQL, Git. Projects: online bookstore using "
     "Java and Spring, student record system with JDBC. Internship: java developer trainee. "
     "Education: BCA, 8.2 CGPA.",
     ["java"]),
    ("Data analyst fresher",
     "BCA graduate. Skills: Excel, SQL, Power BI, Tableau, Python, pandas, statistics. Projects: sales "
     "dashboard in Power BI, customer data analysis with pandas. Internship: data analyst intern. "
     "Education: BCA, 8.0 CGPA.",
     ["data", "analyst", "analytics"]),
    ("QA / testing fresher",
     "BCA graduate. Skills: manual testing, Selenium, test cases, bug reporting, SQL, Git. Projects: "
     "test plan and test cases for an e-commerce site, Selenium automation of a login page. "
     "Internship: QA trainee. Education: BCA, 8.1 CGPA.",
     ["qa", "tester", "testing", "quality"]),
    ("Web / full stack fresher",
     "BCA graduate. Skills: HTML, CSS, JavaScript, React, Node.js, MongoDB, Git. Projects: portfolio "
     "website in React, MERN task manager. Internship: web developer intern. Education: BCA, 8.3 CGPA.",
     ["web", "full stack", "frontend", "front end", "react", "mern"]),
]

def whole_word(text, word):
    return re.search(r"(?<![a-z0-9])" + re.escape(word), text.lower()) is not None

def title_matches(title, expected):
    return any(whole_word(title, w) for w in expected)

with open("saved_jobs.json") as f:
    jobs = json.load(f).get("results", [])
pool = [j for j in jobs if not any(w in j.get("title", "").lower() for w in SENIOR_TITLE_KEYWORDS)] or jobs
print(f"Jobs in pool after fresher filter: {len(pool)}\n")

scored, hits_top3, hits_top1 = 0, 0, 0
for name, resume, expected in TESTS:
    in_pool = sum(1 for j in pool if title_matches(j.get("title", ""), expected))
    vectors = TfidfVectorizer(stop_words="english").fit_transform([resume] + [j.get("description", "") for j in pool])
    scores = cosine_similarity(vectors[0], vectors[1:])[0]
    order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)[:3]
    print(f"TEST: {name}  (expected title words: {', '.join(expected)})")
    print(f"  Jobs of this type in the pool: {in_pool}")
    top_titles = [pool[i].get("title", "") for i in order]
    for rank, i in enumerate(order, start=1):
        mark = "<-- match" if title_matches(pool[i].get("title", ""), expected) else ""
        print(f"  {rank}. {pool[i].get('title', '')}  (score {scores[i]:.3f}) {mark}")
    if in_pool == 0:
        print("  RESULT: SKIPPED (no jobs of this type in the pool, so the test is not fair)\n")
        continue
    scored += 1
    top3 = any(title_matches(t, expected) for t in top_titles)
    top1 = title_matches(top_titles[0], expected)
    hits_top3 += top3
    hits_top1 += top1
    print(f"  RESULT: {'HIT' if top3 else 'MISS'} in top 3 | {'HIT' if top1 else 'MISS'} at rank 1\n")

print("=" * 50)
print(f"Tests scored: {scored} of {len(TESTS)}")
print(f"Top-3 hit rate: {hits_top3} / {scored}")
print(f"Top-1 hit rate: {hits_top1} / {scored}")