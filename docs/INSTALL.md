# Installation Guide

This guide walks you through setting up and running the **Helper Quote** chatbot on your machine, step by step. No prior experience needed.

## 1. Install Python

You need **Python 3.10 or newer**.

Check if you already have it:

```bash
python --version
```

If you don't have Python, download it from [python.org/downloads](https://www.python.org/downloads/) and install it (make sure to check "Add Python to PATH" during setup on Windows).

## 2. Create a Virtual Environment

A virtual environment keeps this project's packages separate from the rest of your system.

### Windows (PowerShell or Command Prompt)

```bash
python -m venv .venv
```

### Linux / macOS

```bash
python3 -m venv .venv
```

## 3. Activate the Virtual Environment

### Windows (PowerShell)

```bash
.venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)

```bash
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
source .venv/bin/activate
```

You'll know it worked because your terminal prompt will now start with `(.venv)`.

## 4. Install Requirements

```bash
pip install -r requirements.txt
```

This installs Streamlit, LangChain, LangGraph, Pandas, Matplotlib, and everything else the app needs.

## 5. Create Your `.env` File

Copy the example file:

**Windows:**
```bash
copy .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

## 6. Add Your Groq API Key

Get a free API key at [console.groq.com/keys](https://console.groq.com/keys).

Open the `.env` file in any text editor and paste your key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

Save the file. **Never share this file or commit it to git.**

## 7. Run the App

```bash
streamlit run main.py
```

Your browser should automatically open to `http://localhost:8501`. If it doesn't, copy that URL into your browser manually.

## 8. Try It Out

1. Make sure the HCC price workbook path in `config/settings.json` points to your file.
2. Type in the chat, e.g. *"I'm John, 407-555-0199. Quote 2 W2136 in white shaker, not assembled, pickup."*
3. Open **Thinking & steps** to watch the agent read the sheet and decide.
4. The quote appears as a card with a **Download quote** button (printable HTML).
5. Optional: set `STAFF_PIN` in `.env`, unlock **Staff mode** in the sidebar and teach a rule:
   *"Remember: espresso color goes to JSI."*

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `GROQ_API_KEY is missing` error | Make sure `.env` exists in the project root and contains a valid key, then restart Streamlit. |
| `streamlit: command not found` | Make sure your virtual environment is activated before running the command. |
| PowerShell blocks `Activate.ps1` | Run PowerShell as Administrator and execute: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Replies are very slow / "rate limit" | Groq's free tier allows ~8K tokens/minute for gpt-oss-120b; an agent needs more. The app waits and retries, but use Groq's Dev tier for real use. |
| Price sheet shows an error in the sidebar | Check the `path` in `config/settings.json` and that the workbook was saved by Excel (cached values are needed). |
| Bot forgot everything | Memory lives in `storage/`. Don't delete that folder (it is git-ignored on purpose). |
