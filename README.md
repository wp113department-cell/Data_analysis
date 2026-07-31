# AI Data Analyst Agent

[![Try it live](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://analysiscsv.streamlit.app/)

**🔗 Try it live: [analysiscsv.streamlit.app](https://analysiscsv.streamlit.app/)** - upload your own CSV or use one from `data/` and ask it anything.

A simple, working demo of an AI agent that analyzes CSV data end-to-end: upload a file, ask a question in plain English, and get statistics, charts, AI-written insights, and a downloadable markdown report - all running locally with **Streamlit**, **LangChain**, **LangGraph**, and the **Groq** API.

This project is built as a clean demonstration of how **LangChain** (LLM calls) and **LangGraph** (agent orchestration) work together in a real multi-agent pipeline.

## Features

- 📤 Upload any CSV and preview it instantly
- 💬 Ask questions about your data in plain English and get a direct, accurate **Response** grounded in the real data (never guesses - if it's not in the CSV, it says so)
- 📈 Automatic statistics: rows, columns, missing values, mean/median/min/max, grouped summaries, exact distinct-value counts
- 📊 Automatic, question-relevant charts: bar, histogram, pie, scatter
- 💡 AI-generated insights: trends, anomalies, business recommendations
- 📄 A final downloadable markdown report combining everything
- ⚡ Powered by Groq's `llama-3.3-70b-versatile` for fast responses
- 🖥️ 100% local - no Docker, no database, nothing to deploy

## Architecture

```
User Question + CSV
        |
        +-------------------------------------------+
        |                                            |
        v                                            v
 [ Query/Response Agent ]                    [ LangGraph Pipeline ]      <- runs ONCE, first question only
 runs on EVERY question                              |
 (LangChain: writes + executes a                  planner  (LangChain: structured output)
  real pandas query, then phrases              /    \
  the real result - never guesses)      statistics  visualization    <- parallel
        |                                      \    /
        v                                     insight   (LangChain: LLM + prompt)
  Streamlit "Response" section                    |
  (always open, always up to date)              report   (LangChain: LLM + prompt)
                                                    |
                                                    v
                                     Statistics / Charts / Insights / Report
                                          (collapsed, generated once)
```

- **LangChain** ([docs/LANGCHAIN_USAGE.md](docs/LANGCHAIN_USAGE.md)) handles all LLM communication: prompts, structured output, parsing.
- **LangGraph** ([docs/LANGGRAPH_USAGE.md](docs/LANGGRAPH_USAGE.md)) handles the agent pipeline: state, nodes, edges, and parallel execution.
- **Streamlit** ([main.py](main.py)) handles the UI.

## Folder Structure

```
project/
├── main.py                     # Streamlit UI
├── agent/
│   ├── agents.py                # LangChain logic (LLM, prompts, stats, charts, query agent)
│   └── graph.py                  # LangGraph orchestration (state, nodes, edges)
├── data/
│   └── sample_data.csv            # Example dataset to try the app with
├── docs/
│   ├── INSTALL.md                  # Step-by-step setup guide
│   ├── LANGCHAIN_USAGE.md           # Where/why LangChain is used
│   └── LANGGRAPH_USAGE.md            # Where/why LangGraph is used
├── requirements.txt              # Python dependencies
├── .env.example                    # Template for your Groq API key
└── README.md                        # This file
```

## Screenshots

*(Add screenshots here after running the app locally - e.g. `screenshots/upload.png`, `screenshots/report.png`)*

## How to Run

See [docs/INSTALL.md](docs/INSTALL.md) for full step-by-step instructions. Quick version:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY inside
streamlit run main.py
```

Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys).

## Example Questions

Try these with `data/sample_data.csv`:

- Which product generated the highest revenue?
- Which product sold the most?
- Show monthly revenue.
- What is the average sales?
- Which customer spent the most?
- Which city generated the highest revenue?
- Show the distribution of sales.
- Are there any missing values?
- What are the top five products?
- Summarize this dataset.
- Generate business insights.

## Technologies

| Technology | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Web UI |
| [LangChain](https://python.langchain.com/) | LLM calls, prompts, structured output |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Multi-agent orchestration |
| [Groq](https://groq.com/) (`llama-3.3-70b-versatile`) | LLM inference |
| [Pandas](https://pandas.pydata.org/) | Data analysis |
| [Matplotlib](https://matplotlib.org/) | Charts |

## Future Improvements

- Support Excel (`.xlsx`) and JSON uploads in addition to CSV
- Let users ask follow-up questions with conversation memory
- Add more chart types (line charts for time series, box plots)
- Cache LLM responses for repeated questions to save API calls
- Export the final report as PDF in addition to markdown

---

Built as a client demo to showcase LangChain + LangGraph working together in a real, runnable application.
