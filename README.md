# MGC Lead Scoring

A small full-stack application for:

1. **Lead scoring** — estimate which new CRM leads are more likely to convert.
2. **Document assistance** — answer sales questions using the supplied project documents with sources.

The project runs locally with three processes:

| Process | Purpose | Address |
|---|---|---|
| Scoring API | Scores new sales leads | `http://127.0.0.1:8000` |
| Document Assistant API | Answers grounded questions from supplied documents | `http://127.0.0.1:8001` |
| Web App | Browser interface for both features | `http://localhost:3000` |

The browser communicates with the Next.js web app, which forwards requests to the two Python APIs.

---

## Prerequisites

Install:

- **Python 3.12+**
- **Node.js 20.9+**
- **npm**
- **PowerShell** on Windows

Check your versions:

```powershell
python --version
node --version
npm --version
```



## 1. Configure Environment Variables

Create a `.env` file in the repository root:

```env
PYTHON_API_URL=http://127.0.0.1:8000
DOCUMENT_ASSISTANT_API_URL=http://127.0.0.1:8001
```

### Optional LLM configuration

The supplied pricing, transfer-fee, rental-yield, and anchor-tenant test cases work without an LLM key.

To enable broader free-form document answers, add one supported provider.

OpenAI example:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

OpenRouter example:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

> Never commit `.env` or expose real API keys. The file should remain ignored by Git.

---

## 2. Create the Python Environment

Run once from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .\services\scoring
python -m pip install -r .\services\document-assistant\requirements.txt
```

If PowerShell blocks virtual-environment activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This execution-policy change applies only to the current PowerShell session.

---

## 3. Train the Lead-Scoring Model

A trained model is already included. Retrain only if this is a fresh setup or if `data/leads.csv` changes.

```powershell
.\.venv\Scripts\python.exe -m mgc_lead_scoring.train --data .\data\leads.csv
```

Successful training creates or updates:

```text
services\scoring\artifacts\model.joblib
services\scoring\artifacts\metadata.json
```

The training pipeline compares fixed candidate models and keeps the best validated model.

---

## 4. Start the Application

Open **three PowerShell terminals** in the repository root.

Keep all three running while testing the application.

### Terminal 1 — Scoring API

```powershell
.\.venv\Scripts\python.exe -m uvicorn mgc_lead_scoring.api:app --reload --port 8000
```

Expected address:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

Swagger API docs:

```text
http://127.0.0.1:8000/docs
```

### Terminal 2 — Document Assistant API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --app-dir .\services\document-assistant --reload --port 8001
```

Expected address:

```text
http://127.0.0.1:8001
```

Health check:

```text
http://127.0.0.1:8001/health
```

The document index is already included.

Rebuild it only when files in `data\documents\` change:

```powershell
.\.venv\Scripts\python.exe .\services\document-assistant\ingest.py
```

> Use `python.exe -m uvicorn` instead of `.venv\Scripts\uvicorn.exe`. The direct launcher can retain an old path if the project folder has been moved.

### Terminal 3 — Next.js Web App

```powershell
Set-Location .\apps\web
npm.cmd install
npm.cmd run dev
```

Open:

```text
http://localhost:3000
```

---

# Testing

## Document Assistant

Open the **Ask documents** section and test:

1. `What is the base price of a 2-bed in Block B?`
2. `What is the total price for a Margalla-facing corner unit, floor 15, 2-bed Block B?`
3. `What's the transfer fee?`
4. `What is the rental yield on a 1-bed?`
5. `Who is the anchor tenant?`

### Expected behaviour

- Every supported answer shows at least one source.
- 2-bed Block B base price: **PKR 22,425,000**.
- The floor-15 Margalla-facing corner-unit question shows the calculation.
- The transfer-fee question reports the **2% vs 2.5% conflict** between source documents.
- Rental yield is reported as unavailable instead of being invented.
- Anchor tenant is reported as unconfirmed instead of being invented.

Run the document test script directly:

```powershell
.\.venv\Scripts\python.exe .\services\document-assistant\test_questions.py
```

---

## Lead Scoring

Open **Score a lead** and try:

| Field | Example value |
|---|---|
| Lead source | Referral |
| City | Islamabad |
| Area | B-17 |
| Property type | Apartment |
| Budget | 220 PKR lac |
| Bedrooms | 2 |
| Referred by existing client | Yes |
| Purchase timeframe | 0–30 days |
| Budget matches inventory | Yes |
| Initial intent level | High |

Click **Score lead**.

A successful response shows:

- estimated conversion likelihood;
- selected model name;
- prioritisation note.

Example:

```text
Estimated conversion likelihood: 80%
Model: logistic regression
Use this estimate to prioritize sales follow-up.
```

The percentage is a **lead-priority estimate**, not a guaranteed sale.

---

## Health Checks

From PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8001/health
```

Both services should return:

```text
status : ok
```

---

## Frontend Verification

From `apps\web`:

```powershell
npm.cmd run lint
npm.cmd run build
```

A successful build confirms the frontend is ready for production packaging.

---

# Lead-Scoring Approach

The scoring system uses only information intended to be known at or near lead intake.

## Data cleaning decisions

- Remove duplicate CRM records using `crm_record_hash`.
- Normalize inconsistent city names.
- Keep useful missing values instead of deleting leads.
- Median-impute numeric values for scikit-learn models.
- Use an explicit missing category for categorical fields.
- Drop identifiers such as `lead_id` and `crm_record_hash`.
- Drop post-contact activity fields that could introduce target leakage.
- Use `has_financing_approved` only if it is genuinely available at intake.
- Use `created_at` only to derive time-related features.
- Split data chronologically so the newest leads remain the final test set.

## Leakage fields removed

Examples include:

```text
first_response_minutes
calls_made
total_call_seconds
whatsapp_replies
site_visits
token_amount_received_pkr
agent_experience_years
```

These fields occur after sales work has started and would make a "who should we call first?" model misleading.

---

## Models Compared

The training pipeline compares four fixed candidates:

| Model | Validation Average Precision |
|---|---:|
| Logistic Regression | **0.1404** |
| CatBoost | 0.1313 |
| Gradient Boosting | 0.1296 |
| XGBoost | 0.1235 |

**Selected model: Logistic Regression**

A more complex model is not automatically better. The selected model is the one that ranked converting leads best on the chronological validation set.

---

## Model Evaluation

The deduplicated dataset is split chronologically:

- **65%** — initial training
- **15%** — model-selection validation
- **20%** — untouched final test

The winning model is retrained on the oldest 80% and evaluated once on the newest 20%.

### Reported metric

**Final test Average Precision: `0.2075`**

Average Precision is used because conversions are rare.

The final test set contains only about **7.9% converted leads**, so plain accuracy could look misleadingly high by predicting most leads as non-converting.

Average Precision is better suited to the actual business problem: ranking true converters near the top of the sales callback list.

---

# Document Assistant Approach

The document assistant is designed around three principles:

### Grounding

Answers should come from the supplied project files and show their sources.

### Refusal

If information is not present in the documents, the system should say so instead of inventing an answer.

### Conflict handling

If two documents disagree, both values are surfaced rather than silently choosing one.

For example, the transfer-fee sources contain conflicting values:

```text
Price list:      2%
Booking policy:  2.5%
```

The assistant reports the conflict instead of pretending one is definitely correct.

---

# Database

The project includes database support for properly storing lead records.

The schema focuses on:

- appropriate primary keys;
- useful field types;
- deduplication;
- preserving lead-intake information separately from later activity.

Duplicate prevention is based on the CRM record hash so the same underlying lead is not counted multiple times.

---

# Application Flow

## Lead scoring

```text
Browser
   |
   v
Next.js Web App
   |
   v
/api/score
   |
   v
FastAPI Scoring Service
   |
   v
model.joblib
   |
   v
Conversion likelihood
```

## Document assistant

```text
Browser
   |
   v
Next.js Web App
   |
   v
Document API proxy
   |
   v
FastAPI Document Assistant
   |
   v
Persisted document index
   |
   v
Grounded answer + source
```

---

# Project Structure

```text
MGC Task/
|
|-- apps/
|   `-- web/
|       `-- Next.js frontend and API proxies
|
|-- services/
|   |-- scoring/
|   |   |-- src/mgc_lead_scoring/
|   |   |-- artifacts/
|   |   `-- pyproject.toml
|   |
|   `-- document-assistant/
|       |-- api.py
|       |-- ingest.py
|       |-- test_questions.py
|       `-- requirements.txt
|
|-- data/
|   |-- leads.csv
|   `-- documents/
|
|-- database/
|   `-- schema/import/diagnostic files
|
|-- .env
`-- README.md
```

---

# Troubleshooting

## `Fatal error in launcher` or an old project path appears

The virtual environment was probably created before the repository was moved.

Recreate it:

```powershell
deactivate

Remove-Item -LiteralPath .\.venv -Recurse -Force

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .\services\scoring
python -m pip install -r .\services\document-assistant\requirements.txt
```

Then restart the APIs using the documented `python.exe -m uvicorn` commands.

---

## Scoring service unavailable

Check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

If the API reports a missing model, retrain:

```powershell
.\.venv\Scripts\python.exe -m mgc_lead_scoring.train --data .\data\leads.csv
```

---

## Document assistant unavailable

Check:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

If the source documents changed, rebuild the index:

```powershell
.\.venv\Scripts\python.exe .\services\document-assistant\ingest.py
```

Then restart the document API.

---

## Port already in use

Check which process owns a port:

```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, OwningProcess
```

Repeat with ports `8001` or `3000` as needed.

Stop the old process or change the port consistently in both the `.env` file and the relevant start command.

---

# Production Improvements

Before production use, the next steps would be:

- verify exactly when every CRM field becomes available;
- evaluate precision and recall at the sales team's real daily call capacity;
- calibrate probability outputs if percentages are shown directly to staff;
- inspect performance across lead sources and time periods;
- monitor data and model drift;
- add authentication and authorization;
- add structured logging and monitoring;
- persist lead-score history;
- add automated tests and CI/CD;
- deploy the frontend and Python services behind production infrastructure.

---

# Important Notes

The lead score is intended to **prioritize sales follow-up**. It is not a booking decision and does not guarantee conversion.

Document-assistant answers are intended to remain grounded in the supplied project documents and show their sources.
