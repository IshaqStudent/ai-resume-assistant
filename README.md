# ai-resume-assistant
# 📄 Resume ATS Score Checker

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get:

- An **ATS score out of 100** (shown at the top of the results)
- A **score breakdown** (keywords, impact, formatting, structure, clarity)
- **Prioritised improvements**, missing keywords and example rewrites

Built with [Streamlit](https://streamlit.io) and Google's Gemini Flash model.

> Scores are AI estimates, not the output of a real ATS. Use them as guidance.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export GEMINI_API_KEY="your-key"  # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

Get a free API key at <https://aistudio.google.com/apikey>.
If no key is configured, the app shows a password field in the sidebar.

## Configuration

| Name | Required | Default | Purpose |
|------|----------|---------|---------|
| `GEMINI_API_KEY` | Yes | - | Your Gemini API key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model name to use |

Set them as environment variables locally, or in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key"
```

## Push to GitHub

1. Create an empty repository on GitHub (no README, no .gitignore).
2. Create a `.gitignore` so secrets never get committed:
   ```
   .venv/
   __pycache__/
   .streamlit/secrets.toml
   ```
3. In your project folder run:
   ```bash
   git init
   git add app.py README.md requirements.txt .gitignore
   git commit -m "Initial commit: resume ATS checker"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** and choose your repo, branch `main`, and main file `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. Click **Deploy**. Every `git push` to `main` redeploys automatically.

## Notes

- Scanned/image-only PDFs have no extractable text; upload a text-based PDF or DOCX.
- Max upload size is 5 MB; only the first ~15,000 characters are analysed.
- Resumes are sent to the Gemini API for analysis and are not stored by this app.
