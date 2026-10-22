import requests
import json

APP_ID = "013eea14"
APP_KEY = "89df4df4720f9bf678b089a0b448d2f4"

url = "https://api.adzuna.com/v1/api/jobs/in/search/1"

search_terms = [
    "software developer fresher",
    "data analyst fresher",
    "QA tester fresher"
]

all_results = []

for term in search_terms:
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "what": term,
        "results_per_page": 15
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        print(f"'{term}': HTTP status {response.status_code}")
        data = response.json()
        results = data.get("results", [])
    except Exception as e:
        print(f"'{term}': request failed ({e})")
        results = []
    print(f"'{term}': {len(results)} jobs found")
    all_results.extend(results)

# Remove duplicate postings (the same job can appear under two searches)
seen_ids = set()
unique_results = []
for job in all_results:
    job_id = job.get("id")
    if job_id not in seen_ids:
        seen_ids.add(job_id)
        unique_results.append(job)

print(f"\nTotal unique jobs collected: {len(unique_results)}")

# Safety check: only overwrite saved_jobs.json if the fetch clearly worked
MIN_JOBS = 5
if len(unique_results) < MIN_JOBS:
    print("\nNOT SAVED: too few jobs came back, so your existing saved_jobs.json was left untouched.")
else:
    with open("saved_jobs.json", "w") as f:
        json.dump({"results": unique_results}, f)
    print("\nSaved combined job data to saved_jobs.json")