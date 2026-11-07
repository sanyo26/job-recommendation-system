import html
import io
import json
import re
from collections import Counter
from datetime import datetime
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional libraries for reading resume files; the app still runs without them
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    import docx
except ImportError:
    docx = None

st.set_page_config(page_title="SkillMatch Jobs", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 12% 0%, #14203d 0%, #0B0F19 42%); }
#MainMenu, footer, .stDeployButton { visibility: hidden; display: none; }
.block-container { padding-top: 1.2rem; max-width: 1400px; }
.project-title { font-size: 15px; color: #8FA0C0; font-style: italic; margin: 0 0 4px 0; }
.sub-header { font-size: 17px; color: #B9C4DC; margin-bottom: 8px; }
.panel-title { font-size: 20px; font-weight: 700; color: #FFFFFF; margin: 4px 0 8px 0; }

/* tabs */
div[data-testid="stTabs"] button { font-size: 15px; font-weight: 600; padding: 8px 16px; border-radius: 10px 10px 0 0; transition: all .2s ease; }
div[data-testid="stTabs"] button:hover { background: rgba(77,127,255,.14); }

/* metric boxes */
div[data-testid="stMetric"] { background: linear-gradient(145deg, #172033, #121827); border: 1px solid #26324D;
  border-radius: 14px; padding: 14px 18px; transition: transform .2s ease, border-color .2s ease; }
div[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: #4D7FFF; }

/* buttons */
div.stButton > button[kind="primary"] { background: linear-gradient(90deg, #4D7FFF, #7A5CFF); border: none; color: #fff;
  font-weight: 700; border-radius: 12px; padding: .6rem 1rem; transition: transform .15s ease, box-shadow .15s ease; }
div.stButton > button[kind="primary"]:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(77,127,255,.35); }
div[data-testid="stDownloadButton"] button { border-radius: 12px; border: 1px solid #4D7FFF; transition: all .2s ease; }
div[data-testid="stDownloadButton"] button:hover { background: rgba(77,127,255,.15); }

/* job cards */
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
.job-card { background: linear-gradient(145deg, #1A2033, #141827); padding: 20px 24px; border-radius: 16px; margin-bottom: 18px;
  border: 1px solid #27304A; border-left: 5px solid #4D7FFF; animation: rise .5s ease both;
  transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease; }
.job-card:hover { transform: translateY(-4px); box-shadow: 0 14px 30px rgba(0,0,0,.45); border-color: #4D7FFF; }
.job-top { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.job-title { font-size: 19px; font-weight: 700; color: #FFFFFF; }
.rank { font-size: 22px; min-width: 34px; text-align: right; color: #8FA0C0; font-weight: 700; }
.job-company { font-size: 14px; color: #8FA0C0; margin: 2px 0 12px 0; }
.bar { height: 8px; background: #232B42; border-radius: 8px; overflow: hidden; margin: 12px 0 6px 0; }
.fill { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #4D7FFF, #7A5CFF); }
.badge-strong { background: #1F3D2B; color: #4ADE80; padding: 3px 11px; border-radius: 20px; font-size: 12px; font-weight: 700; }
.badge-moderate { background: #3D351F; color: #FACC15; padding: 3px 11px; border-radius: 20px; font-size: 12px; font-weight: 700; }
.badge-low { background: #3D1F1F; color: #F87171; padding: 3px 11px; border-radius: 20px; font-size: 12px; font-weight: 700; }
.role-chip { background: #1E2A44; color: #8FB0FF; padding: 3px 11px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-left: 8px; }
.row-label { margin-top: 12px; font-size: 12px; color: #8FA0C0; text-transform: uppercase; letter-spacing: .6px; }
.chip { display: inline-block; padding: 3px 10px; border-radius: 14px; font-size: 12px; font-weight: 600; margin: 4px 6px 0 0; }
.chip-have { background: #16352A; color: #4ADE80; }
.chip-miss { background: #3A2020; color: #FF8A7A; }
.chip-why { background: #1B2745; color: #9DB7FF; }
.note-line { margin-top: 12px; font-size: 13px; color: #B9C4DC; }
.apply-btn { display: inline-block; margin-top: 14px; background: linear-gradient(90deg, #4D7FFF, #7A5CFF); color: white !important;
  padding: 7px 18px; border-radius: 10px; text-decoration: none !important; font-size: 13px; font-weight: 700; transition: transform .15s ease; }
.apply-btn:hover { transform: translateX(3px); }
</style>
""", unsafe_allow_html=True)

with open("saved_jobs.json") as f:
    all_data = json.load(f)

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

# Role groups are decided from the job title; the FIRST group that matches wins
ROLE_RULES = [
    ("QA / Testing", ["qa", "tester", "testing", "quality", "sdet", "selenium"]),
    ("Data Science / ML", ["data scientist", "machine learning", "ml", "ai", "artificial intelligence"]),
    ("Data / Analytics", ["data", "analyst", "analytics", "business intelligence", "power bi", "tableau", "bi"]),
    ("Cloud / DevOps", ["cloud", "devops", "aws", "azure", "sre", "site reliability", "infrastructure"]),
    ("Support / IT Ops", ["support", "helpdesk", "help desk", "system administrator", "sysadmin", "network", "noc"]),
    ("Full Stack / Web", ["full stack", "fullstack", "frontend", "front end", "backend", "back end",
                          "web", "react", "angular", "node", "php", "mern", "mean", "ui"]),
    ("Java Developer", ["java"]),
    ("Python Developer", ["python"]),
    (".NET Developer", [".net", "dotnet", "c#", "asp.net"]),
    ("Software Developer", ["developer", "engineer", "programmer", "software", "trainee"]),
]

def has_skill(text, skill):
    # Whole-word match, so "excel" does not match "excellent"
    return re.search(r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])", text) is not None

def is_fresher_friendly(title):
    return not any(word in title.lower() for word in SENIOR_TITLE_KEYWORDS)

def role_family(title):
    t = title.lower()
    for family, words in ROLE_RULES:
        if any(has_skill(t, w) for w in words):
            return family
    return "Other"

def read_resume_file(uploaded):
    """Returns (text, error_message). Supports .txt, .pdf and .docx."""
    name = uploaded.name.lower()
    try:
        if name.endswith(".txt"):
            text = uploaded.getvalue().decode("utf-8", errors="ignore")
            return (text, None) if text.strip() else ("", "This text file is empty.")
        if name.endswith(".pdf"):
            if PdfReader is None:
                return "", "PDF reading is not installed on this machine (pypdf is missing)."
            reader = PdfReader(io.BytesIO(uploaded.getvalue()))
            if reader.is_encrypted:
                return "", "This PDF is password-protected. Remove the password and upload it again."
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            if not text.strip():
                return "", ("No text found. This looks like a scanned or image-based PDF. "
                            "Upload a text-based PDF or a DOCX, or paste your text instead.")
            return text, None
        if name.endswith(".docx"):
            if docx is None:
                return "", "DOCX reading is not installed on this machine (python-docx is missing)."
            document = docx.Document(io.BytesIO(uploaded.getvalue()))
            parts = [p.text for p in document.paragraphs]
            for table in document.tables:
                for row in table.rows:
                    for cell in row.cells:
                        parts.append(cell.text)
            text = "\n".join(parts)
            return (text, None) if text.strip() else ("", "No text found in this document.")
        return "", "Unsupported file type. Please upload a .txt, .pdf or .docx file."
    except Exception:
        return "", "Could not read this file. Try another file, or paste your text instead."

def skill_gap(description, resume_text):
    desc, resume = description.lower(), resume_text.lower()
    required = [s for s in common_skills if has_skill(desc, s)]
    missing = [s for s in required if not has_skill(resume, s)]
    return required, missing

def company_parts(job):
    info = job.get("company") or {}
    name = info.get("display_name") or info.get("canonical_name") or "Company not disclosed"
    location = (job.get("location") or {}).get("display_name", "") or ""
    return name, location

def ats_badge(percent):
    if percent >= 70:
        return "Strong Match", "badge-strong", "🟢"
    if percent >= 40:
        return "Moderate Match", "badge-moderate", "🟡"
    return "Low Match", "badge-low", "🔴"

def draw_bar(labels, values, value_name, horizontal):
    """Bars sorted highest first, in the direction chosen by the user."""
    df = pd.DataFrame({"Skill": labels, value_name: values})
    try:
        st.bar_chart(df, x="Skill", y=value_name, sort="-" + value_name, horizontal=horizontal)
    except TypeError:
        # Older Streamlit versions do not support sort/horizontal; fall back to a plain chart
        st.bar_chart(df.set_index("Skill"))

def job_pool():
    all_jobs = all_data.get("results", [])
    return [j for j in all_jobs if is_fresher_friendly(j.get("title", ""))] or all_jobs

def run_search(resume_text, family):
    """Scores the fresher-friendly jobs (optionally one role group) against the resume."""
    jobs = job_pool()
    if family != "All roles":
        jobs = [j for j in jobs if role_family(j.get("title", "")) == family]
    if not jobs:
        return None
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text] + [j.get("description", "") for j in jobs])
    names = vectorizer.get_feature_names_out()
    scores = cosine_similarity(vectors[0], vectors[1:])[0]
    top_score = max(scores) if max(scores) > 0 else 1
    scaled = [(s / top_score) * 100 for s in scores]
    order = sorted(range(len(jobs)), key=lambda i: scores[i], reverse=True)

    rows = []
    for i in order[:10]:
        job = jobs[i]
        required, missing = skill_gap(job.get("description", ""), resume_text)
        # "Why this match": words that the resume and this job share, strongest first
        overlap = vectors[0].multiply(vectors[i + 1]).toarray()[0]
        why = [names[k] for k in overlap.argsort()[::-1][:5] if overlap[k] > 0]
        name, location = company_parts(job)
        label, badge_class, emoji = ats_badge(scaled[i])
        rows.append({
            "title": job.get("title", "No title"), "company": name, "location": location,
            "family": role_family(job.get("title", "")),
            "percent": scaled[i], "label": label, "badge_class": badge_class, "emoji": emoji,
            "required": required, "missing": missing, "why": why, "link": job.get("redirect_url", ""),
        })
    return {"rows": rows, "scanned": len(jobs), "top": round(max(scaled)),
            "strong": sum(1 for s in scaled if s >= 70), "family": family}

def results_to_csv(rows):
    table = pd.DataFrame([{
        "Rank": i + 1, "Job title": r["title"], "Role type": r["family"], "Company": r["company"],
        "Location": r["location"], "Match %": round(r["percent"]), "Match level": r["label"],
        "Matched words": ", ".join(r["why"]), "Skills mentioned": ", ".join(r["required"]),
        "Missing keywords": ", ".join(r["missing"]), "Apply link": r["link"],
    } for i, r in enumerate(rows)])
    return table.to_csv(index=False).encode("utf-8-sig")

def market_counts():
    """How many postings mention each tracked skill (does not depend on any resume)."""
    jobs = job_pool()
    counts = Counter()
    for job in jobs:
        desc = job.get("description", "").lower()
        for s in common_skills:
            if has_skill(desc, s):
                counts[s] += 1
    return counts, len(jobs)

def health_check(text):
    """Simple rule-based check of the resume itself. Returns (score out of 100, checks, skills found)."""
    t = text.lower()
    words = len(text.split())
    found = [s for s in common_skills if has_skill(t, s)]
    phones = [m for m in re.findall(r"\+?\d[\d\s\-()]{8,}\d", text) if len(re.sub(r"\D", "", m)) >= 10]
    checks = []

    def add(name, ok, max_points, good, tip):
        checks.append({"name": name, "points": max_points if ok else 0, "max": max_points,
                       "note": good if ok else tip})

    add("Email address", bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)), 10,
        "Email found", "Add a professional email address")
    add("Phone number", bool(phones), 10,
        "Phone number found", "Add a phone number recruiters can call")
    add("Education section",
        any(w in t for w in ["education", "bachelor", "bca", "b.tech", "btech", "b.sc", "mca", "degree",
                             "cgpa", "university", "college"]), 15,
        "Education found", "Add your degree, college and year")
    add("Skills section", bool(re.search(r"\bskills?\b", t)), 15,
        "Skills section found", "Add a clear Skills section")
    add("Projects", bool(re.search(r"\bprojects?\b", t)), 15,
        "Projects found", "Add 1 to 3 projects with what you built and the tools you used")
    add("Experience / internship", bool(re.search(r"\b(internship|intern|experience)\b", t)), 10,
        "Experience or internship found", "Add internships, training or any work experience")
    add("Length", 150 <= words <= 700, 10,
        f"{words} words, a good length", f"{words} words. Aim for about 150 to 700 words (1 to 2 pages)")
    n = len(found)
    points = 15 if n >= 8 else 10 if n >= 4 else 5 if n >= 1 else 0
    checks.append({"name": "Tracked IT skills", "points": points, "max": 15,
                   "note": f"{n} tracked skills found" + ("" if n >= 8 else ". Add the relevant skills you really have")})
    return sum(c["points"] for c in checks), checks, found

def chips(items, css):
    return "".join(f"<span class='chip {css}'>{html.escape(i)}</span>" for i in items)

def job_card_html(r, rank):
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
    company_line = html.escape(r["company"]) + ((" · 📍 " + html.escape(r["location"])) if r["location"] else "")
    link = html.escape(r["link"], quote=True)
    parts = [f"<div class='job-card' style='animation-delay:{(rank - 1) * 0.07:.2f}s'>",
             f"<div class='job-top'><div class='job-title'>{html.escape(r['title'])}</div><div class='rank'>{medal}</div></div>",
             f"<div class='job-company'>{company_line}</div>",
             f"<span class='{r['badge_class']}'>{r['emoji']} {round(r['percent'])}% · {r['label']}</span>",
             f"<span class='role-chip'>{html.escape(r['family'])}</span>",
             f"<div class='bar'><div class='fill' style='width:{max(2, round(r['percent']))}%'></div></div>"]
    if r["why"]:
        parts.append("<div class='row-label'>🔎 Why this match</div>" + chips(r["why"], "chip-why"))
    if not r["required"]:
        parts.append("<div class='note-line'>ℹ️ The preview doesn't list skills. Click Apply Now for the full requirements.</div>")
    else:
        have = [s for s in r["required"] if s not in r["missing"]]
        if have:
            parts.append("<div class='row-label'>✅ Skills you have</div>" + chips(have, "chip-have"))
        if r["missing"]:
            parts.append("<div class='row-label'>⚠️ Missing keywords</div>" + chips(r["missing"], "chip-miss"))
        else:
            parts.append("<div class='note-line'>You have every skill this posting mentions.</div>")
    if link:
        parts.append(f"<a href='{link}' target='_blank' class='apply-btn'>Apply Now →</a>")
    parts.append("</div>")
    return "".join(parts)

# ---------- Title banner (drawn in its own frame so Streamlit styles cannot shrink it) ----------
HERO = """
<div style="font-family:'Source Sans Pro','Segoe UI',Arial,sans-serif;padding:4px 2px;display:flex;align-items:center;gap:14px;">
  <span style="font-size:64px;line-height:1.1;">🎯</span>
  <span style="font-size:78px;font-weight:800;line-height:1.1;letter-spacing:-1px;
               background:linear-gradient(90deg,#4D7FFF,#8B6BFF,#FF7A59);-webkit-background-clip:text;
               background-clip:text;color:transparent;">SkillMatch Jobs</span>
</div>
"""
components.html(HERO, height=120)
st.markdown('<p class="project-title">IT Job Recommendation System for Freshers</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Skill-based job matching with ATS-style compatibility scoring</p>', unsafe_allow_html=True)
st.divider()

if "history" not in st.session_state:
    st.session_state["history"] = []

# Role-type options come from the loaded jobs, so every option has at least one job
family_counts = Counter(role_family(j.get("title", "")) for j in job_pool())
role_options = {"All roles": "All roles"}
for fam, count in sorted(family_counts.items(), key=lambda x: (-x[1], x[0])):
    role_options[f"{fam} ({count})"] = fam

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<div class="panel-title">👤 Your Profile</div>', unsafe_allow_html=True)
    mode = st.radio("How do you want to add your resume?", ["Paste text", "Upload file"], horizontal=True)
    resume_input = ""
    source_label = "Pasted text"
    if mode == "Paste text":
        resume_input = st.text_area("Paste your resume / skills",
                                     "Python, pandas, SQL, basic AWS knowledge, good communication skills",
                                     height=160)
    else:
        uploaded = st.file_uploader("Upload your resume (.txt, .pdf or .docx)", type=["txt", "pdf", "docx"])
        if uploaded is not None:
            resume_input, read_error = read_resume_file(uploaded)
            source_label = uploaded.name
            if read_error:
                st.error(read_error)
            else:
                st.success(f"Read {len(resume_input.split())} words from {uploaded.name}")
                with st.expander("See the text we read from your file"):
                    st.write(resume_input[:1500] + ("..." if len(resume_input) > 1500 else ""))
        st.caption("Tip: your file is read in memory and is not saved by this app.")

    role_label = st.selectbox("🧭 Role type", list(role_options.keys()))
    selected_family = role_options[role_label]
    find_clicked = st.button("🔍 Find My Matches", type="primary", use_container_width=True)

    history = st.session_state["history"]
    if history:
        with st.expander(f"🕘 Recent searches ({len(history)})"):
            for h in history:
                st.caption(f"{h['time']} · {h['source']} → {h['top_job']} ({h['top_percent']}%)")
            if st.button("Clear history"):
                st.session_state["history"] = []
                st.rerun()

if find_clicked:
    if not resume_input.strip():
        with right_col:
            st.warning("Please paste your skills or upload a resume file first.")
    else:
        result = run_search(resume_input, selected_family)
        if result is None:
            with right_col:
                st.warning("No jobs found for this choice. Pick 'All roles' or run fetch_jobs.py once to load jobs.")
        else:
            st.session_state["results"] = result
            st.session_state["history"].insert(0, {
                "time": datetime.now().strftime("%d %b, %H:%M"), "source": source_label,
                "top_job": result["rows"][0]["title"], "top_percent": round(result["rows"][0]["percent"]),
            })
            st.session_state["history"] = st.session_state["history"][:5]
            st.rerun()

with right_col:
    result = st.session_state.get("results")
    chart_style = st.radio("Chart style", ["Horizontal", "Vertical"], horizontal=True)
    horizontal = chart_style == "Horizontal"
    tab_matches, tab_gap, tab_market, tab_health = st.tabs(
        ["🎯 Matches", "📚 Skill gap summary", "📊 Market insights", "🩺 Resume health"])

    with tab_matches:
        if result:
            if result["family"] != "All roles":
                st.caption(f"Showing {result['family']} roles only. Scores are relative within these jobs.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Jobs Scanned", result["scanned"])
            m2.metric("Top Match", f"{result['top']}%")
            m3.metric("Strong Matches", result["strong"])

            head_col, btn_col = st.columns([2, 1])
            head_col.markdown('<div class="panel-title">Top Job Matches</div>', unsafe_allow_html=True)
            btn_col.download_button("⬇️ Download results (CSV)", data=results_to_csv(result["rows"]),
                                    file_name="skillmatch_results.csv", mime="text/csv",
                                    use_container_width=True)
            for rank, r in enumerate(result["rows"], start=1):
                st.markdown(job_card_html(r, rank), unsafe_allow_html=True)
        else:
            st.info("👈 Add your resume on the left and click **Find My Matches** to see your results here.")

    with tab_gap:
        if result:
            considered = [r for r in result["rows"] if r["required"]]
            missing_counts = Counter(s for r in considered for s in r["missing"])
            st.markdown('<div class="panel-title">What to learn next</div>', unsafe_allow_html=True)
            if missing_counts:
                top_missing = missing_counts.most_common(8)
                st.caption(f"Based on your top {len(result['rows'])} matches. Only {len(considered)} of them list skills in their preview.")
                draw_bar([s for s, _ in top_missing], [c for _, c in top_missing], "Jobs", horizontal)
                st.write("**Learn next:** " + ", ".join(f"{s} ({c} of {len(considered)} jobs)" for s, c in top_missing[:3]))
            else:
                st.success("No missing tracked skills in your top matches.")
        else:
            st.info("Run a search first. This tab shows which skills you are missing most often across your top matches.")

    with tab_market:
        counts, total = market_counts()
        st.markdown('<div class="panel-title">Skills employers ask for</div>', unsafe_allow_html=True)
        if counts:
            top = counts.most_common(10)
            st.caption(f"Based on {total} fresher postings. Only the tracked skill list is counted, and Adzuna shows a description excerpt.")
            draw_bar([s for s, _ in top], [c for _, c in top], "Postings", horizontal)
            st.write(f"**Most requested:** {top[0][0]} appears in {top[0][1]} of {total} postings.")
        else:
            st.info("No tracked skills were found in the current job descriptions.")

    with tab_health:
        st.markdown('<div class="panel-title">Resume health check</div>', unsafe_allow_html=True)
        if len(resume_input.split()) >= 60:
            score, checks, found = health_check(resume_input)
            level = "Strong" if score >= 80 else "Fair" if score >= 50 else "Needs work"
            h1, h2 = st.columns([1, 2])
            h1.metric("Resume health", f"{score} / 100")
            h2.markdown(f"**{level}**")
            h2.progress(int(score))
            for c in checks:
                icon = "✅" if c["points"] == c["max"] else "⚠️"
                st.write(f"{icon} **{c['name']}** ({c['points']}/{c['max']}): {c['note']}")
            if found:
                st.write("**Tracked skills found:** " + ", ".join(found))
            st.caption("A simple rule-based check of the resume itself, not a real ATS.")
        elif resume_input.strip():
            st.info("This looks like a short skill list, so a health score would not be meaningful. "
                    "Paste your full resume text or upload your resume file to get a score.")
        else:
            st.info("Paste your resume or upload a file on the left to see its health check.")