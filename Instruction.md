# FlightOps Copilot – Your Personal Guide (do not upload this file)

## 1. Get it running on your computer (about 30 minutes)

1. Install Python 3.12 (python.org) and Git (git-scm.com). Install VS Code.
2. Unzip `flightops-copilot.zip`. Open the folder in VS Code. Open the terminal (Ctrl + `).
3. Run these commands one by one:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python scripts/generate_data.py
   pytest -q
   ```
   You should see "19 passed".
4. Get an API key at console.anthropic.com (add $5 credit – enough for hundreds of questions).
   Copy `.env.example` to `.env` and paste your key.
5. Run `streamlit run app/streamlit_app.py`. Your browser opens the app. Try the example questions.
6. Run `python eval/run_eval.py`. Put the agent numbers into the "Results" table in README.md.

If something fails, copy the full error message and paste it to me.

## 2. Publish it on GitHub

1. Create a new public repo on github.com named `flightops-copilot` (no README, it already has one).
2. In the terminal:
   ```
   git init
   git add .
   git commit -m "FlightOps Copilot: AI agent for airline catering operations"
   git branch -M main
   git remote add origin https://github.com/<your-username>/flightops-copilot.git
   git push -u origin main
   ```
3. Check that `.env` was NOT uploaded (it is in .gitignore – your API key must stay private).
4. Replace "Your Name" and the LinkedIn link at the bottom of README.md.
5. Record a 60–90 second screen video (Loom or Windows Game Bar): ask 2–3 questions, open the
   "How I got this" panel to show the SQL. Add the link at the top of the README. Recruiters watch videos more than they read code.
6. Optional (makes it much stronger): deploy free on Streamlit Community Cloud
   (share.streamlit.io) and put the live link in the README and on your resume.

## 3. Resume – Projects section (pick 3–4 bullets)

**FlightOps Copilot – AI Agent for Airline Operations Analytics** | Python, Claude API, RAG, SQL, Streamlit, FastAPI, Docker | github.com/<you>/flightops-copilot
- Built an LLM agent that answers operations questions by calling 4 tools (text-to-SQL, policy search, forecasting, resource planning) on a 620K-row flight, labor and safety database.
- Designed SQL guardrails (SELECT-only validation, read-only connection, timeout, row limit) so the agent can query data safely, and returned tool errors to the model for self-correction.
- Implemented hybrid retrieval (BM25 built from scratch + vector embeddings fused with Reciprocal Rank Fusion) over SOP and KPI documents; reached 0.93 hit@3 on a labeled evaluation set.
- Built an evaluation harness measuring numeric accuracy against SQL ground truth, tool-selection accuracy, latency and token cost; added 19 unit tests, Docker and GitHub Actions CI.
- Developed back-tested forecasting models (2–3% MAPE) and a resource-planning tool that converts forecast flights into FTE and truck requirements.

(After you run the eval, add: "achieving XX% numeric accuracy on 13 benchmark questions".)

## 4. Interview – how to explain it (practice saying these out loud)

**"Tell me about this project." (60 seconds)**
"In my job, managers ask me data questions every day, like hours per flight or driver attrition.
I built an AI agent that can answer these questions itself. It has four tools: it can write and
run SQL on the database, search our policy documents, run a forecast, and calculate staffing needs.
The important part is that the AI never invents numbers – every number comes from a query, and the
app shows every query it ran. I also built an evaluation set to measure accuracy, because with AI
you need to test, not only demo."

**"Why didn't you use LangChain?"**
"The agent loop is only about 50 lines. Writing it myself means I understand every step and can
debug it easily. For a bigger system with many agents I would consider LangGraph."

**"How do you stop the AI from damaging the database?"**
"Three layers: I only allow one SELECT statement, I block dangerous keywords, and the database
connection itself is read-only. There is also a timeout and a row limit."

**"How do you know the answers are correct?"**
"I have a golden set of questions. For each question I wrote the correct SQL, so the evaluation
script calculates the true answer and checks whether the agent's answer contains it. I also check
if the agent chose the right tool, and I track latency and cost."

**"What would you improve?"**
"Deploy it on Azure, move the data to PostgreSQL, add LLM-as-judge scoring, and turn the tools
into an MCP server so other AI apps can use them."

**"What was difficult?"**
"My first hours-per-flight forecast was too high – 17.2 when the real value was 15.7. With only one
year of data, the model confused the month pattern with the trend. I changed it to use the last 180
days without month effects, and the back-test error dropped to about 3%."
(This is true – it happened while building the project. Real problems make great interview stories.)

## 5. Connect it to your real experience

This project mirrors your real work, so you can say: "The KPIs, the attrition analysis and the
planning rules come from my experience at Newrest – for example, I created the hours-per-flight KPI,
and my analysis supported the +$2 driver raise and safety incentive that reduced driver attrition
from about 5–6% to 1–2%. The data in the project is synthetic to protect company information."

## 6. Next projects (for the 2–3 month plan)

- Data Engineer: build a pipeline (Airflow or Azure Data Factory → PostgreSQL/Snowflake → dbt) that loads the same flight and labor data daily.
- AI Engineer #2: turn this project's tools into an MCP server, or add a document-upload RAG feature.
