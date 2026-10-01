import requests
import json

APP_ID = "013eea14"
APP_KEY = "89df4df4720f9bf678b089a0b448d2f4"

url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
params = {
    "app_id": APP_ID,
    "app_key": APP_KEY,
    "what": "software developer",
    "results_per_page": 20
}

response = requests.get(url, params=params)
data = response.json()

for job in data["results"]:
    title = job.get("title", "No title")
    company = job.get("company", {}).get("display_name", "Unknown company")
    print(title, "-", company)

with open("saved_jobs.json", "w") as f:
    json.dump(data, f)

print("\nSaved job data to saved_jobs.json")