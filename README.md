# NLP2Query — Natural Language to Data Filtering & Visualization

A Streamlit application that lets you upload a CSV or Excel file and query your data using **plain English** or **Hinglish** (Hindi-English). The app converts your question into safe pandas filter code or chart specifications, runs them on your data, and shows the results.

No SQL. No manual coding. Just type what you want in your own words.

---

## Table of contents

1. [What it does](#what-it-does)
2. [How it works](#how-it-works)
3. [Project structure](#project-structure)
4. [Prerequisites & setup](#prerequisites--setup)
5. [Running the app](#running-the-app)
6. [Using the UI](#using-the-ui)
7. [How to write queries](#how-to-write-queries)
8. [Query reference — examples](#query-reference--examples)
9. [Action modes](#action-modes)
10. [Safety model](#safety-model)
11. [Troubleshooting](#troubleshooting)
12. [Tech stack](#tech-stack)

---

## What it does

| Step | Description |
|------|-------------|
| **Upload** | Load a `.csv`, `.xlsx`, or `.xls` file |
| **Preview** | Inspect the first rows and column names |
| **Ask** | Type a natural language question about your data |
| **Filter** | Get matching rows (e.g. "station code is swv") |
| **Visualize** | Get charts (e.g. "show age distribution") |

The system understands casual phrasing, including Hinglish patterns like *"records dikho jaha station code swv hai"*.

---

## How it works

End-to-end flow when you click **Run**:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Upload CSV  │────▶│ Preprocess   │────▶│ Build column    │
│ or Excel    │     │ & clean data │     │ reference       │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                  │
┌─────────────┐     ┌──────────────┐              ▼
│ Show result │◀────│ Validate &   │     ┌─────────────────┐
│ (table or   │     │ execute      │◀────│ LLM generates   │
│  chart)     │     │ safely       │     │ pandas code or  │
└─────────────┘     └──────────────┘     │ chart spec      │
                           ▲             └────────▲────────┘
                           │                        │
                    ┌──────┴───────┐     ┌─────────┴────────┐
                    │ Auto-retry   │     │ Route intent:    │
                    │ on errors    │     │ filter vs chart  │
                    └──────────────┘     └────────▲────────┘
                                                  │
                                         ┌────────┴────────┐
                                         │ User query +    │
                                         │ column hints    │
                                         └─────────────────┘
```

### 1. Data loading (`data_loader.py`)

- Reads uploaded CSV or Excel into a pandas DataFrame named `df`
- Tries multiple text encodings for CSV files (UTF-8, CP1252, etc.)

### 2. Preprocessing (`preprocessor.py`)

- Cleans column names and data types
- Detects numeric columns stored as strings and can auto-convert them
- Builds a **column profile** (numeric, categorical, datetime) used later by the LLM

### 3. Query normalization (`utils.py`)

Your raw query is normalized and interpreted before it reaches the model:

- Strips filler words: `show`, `give`, `records`, `dikho`, `jahan`, `jisme`, `hai`
- Maps phrases: `greater than` → `>`, `less than` → `<`, `is none` → missing value
- Matches multi-word column names (e.g. `station code` → `Station Code`)
- Extracts filter values (e.g. `swv` is a **value**, not a column)
- Builds **interpretation hints** sent to the LLM (null checks, equality, column ranking)

### 4. Column reference for the LLM (`utils.py`)

The model receives rich context about your actual data:

```
- `Station Code` | type: categorical | sample values: "SWV", "CNA", "BCT"
- `age` | type: numeric | sample values: "25", "30", "40"
```

This helps the model use **exact column names** from your file instead of guessing.

### 5. Intent routing (`llm_engine.py` — Auto detect mode only)

When mode is **Auto detect**, a lightweight LLM call classifies the query:

- **filter** — user wants rows from the table
- **visualization** — user wants a chart

Filter code is **not** generated at this step. Routing only decides the path.

### 6. Code generation (`llm_engine.py`)

**Filtering:** The dedicated filter prompt asks the Hugging Face model (`Qwen/Qwen2.5-Coder-7B-Instruct`) to output a single pandas expression starting with `df[`, for example:

```python
df[(df["Station Code"] == "swv")]
df[(df["age"].isna())]
df[df["station name"].str.contains("karmali", case=False)]
```

**Visualization:** The model returns JSON with chart `type` and `columns`. The app renders charts using fixed matplotlib templates — no arbitrary code execution.

### 7. Validation (`validator.py`)

Before any filter expression runs, it is checked:

- Must start with `df[`
- Must be a single line
- Column names must exist in your uploaded file
- AST allowlist: only safe operations (comparisons, `pd.to_numeric`, `.str.contains`, `.isna()`, `.isin()`, etc.)
- No imports, `df.query()`, loops, or arbitrary function calls

### 8. Execution (`executor.py`)

Valid expressions run in a restricted environment:

- `eval()` with `__builtins__` disabled
- Only `df` and `pd` available in scope

### 9. Auto-correction

If validation or execution fails, the app asks the LLM to fix the expression once and retries.

---

## Project structure

```
nlp2query/
├── main.py              # Streamlit UI and orchestration
├── llm_engine.py        # Hugging Face LLM calls (filter, route, charts)
├── utils.py             # Query normalization, column matching, hints
├── validator.py         # AST safety checks for filter expressions
├── executor.py          # Safe execution of validated expressions
├── preprocessor.py      # Data cleaning and type detection
├── data_loader.py       # CSV/Excel loading
├── visualization.py     # Chart generation from specs
├── requirements.txt     # Python dependencies
├── .env.example         # API key template
└── logs/                # Application logs (nlp2filter.log)
```

---

## Prerequisites & setup

### Requirements

- **Python 3.10+** recommended
- A **Hugging Face API token** with access to the inference router

### 1. Clone or download the project

```powershell
cd d:\projects\nlp2query
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure API key

Create a `.env` file in the project root (copy from `.env.example`):

```env
HF_API_KEY=your_huggingface_api_key_here
```

Get a token from: https://huggingface.co/settings/tokens

> **Important:** Never commit `.env` to git. It contains your secret API key.

---

## Running the app

From the project directory:

```powershell
streamlit run main.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## Using the UI

1. **Upload** a CSV or Excel file
2. **Preview** your data in the expander (check exact column names)
3. Open **"How to write queries"** for tips and clickable examples
4. **Type your query** in the text box (or click an example to fill it)
5. Choose an **action mode** (Auto detect, Data Filtering, or Data Visualization)
6. Click **Run**

### Graph settings (visualization)

When creating charts, expand **Graph settings** to adjust width, height, bins, colors, and opacity.

---

## How to write queries

### Basic pattern

```
[action] + [column name] + [condition] + [value]
```

**Examples:**

| Intent | Query |
|--------|-------|
| Text match | `show records where station name is karmali` |
| Hinglish | `records dikho jaha station code swv hai` |
| Number | `age greater than 30` |
| Missing data | `give records where age is none` |
| Chart | `show age distribution` |

### Words the system understands

| Category | Words |
|----------|-------|
| **Actions** | `show`, `give`, `get`, `filter`, `dikho`, `batao`, `nikalo` |
| **Where** | `where`, `jahan`, `jaha`, `jisme`, `jinme` |
| **End of sentence** | `hai`, `hain`, `ho` |
| **Missing / empty** | `none`, `null`, `empty`, `missing`, `khali`, `na` |
| **Comparisons** | `greater than`, `less than`, `at least`, `at most`, `equals`, `is`, `high`, `low` |
| **Charts** | `plot`, `chart`, `histogram`, `distribution`, `scatter`, `bar`, `pie` |

### Tips for accurate results

1. **Use column names from your file** — check the data preview for exact spelling and spaces (e.g. `Station Code` vs `station_code`)
2. **Separate column from value** — in `station code swv hai`, `station code` is the column and `swv` is the value
3. **Casual language is fine** — you do not need perfect grammar
4. **Use Data Filtering mode** if Auto detect misclassifies your query
5. **Missing values** — use `none`, `null`, `missing`, or `khali`; the system maps these to `df["col"].isna()`, not string comparison

---

## Query reference — examples

Replace column names and values with those from **your** uploaded file.

### Text / string filters (English)

```
show records where station name is karmali
station name equals Chennai
filter rows where city contains mum
name is John
records where department == Sales
```

### Text / string filters (Hinglish)

```
records dikho jaha station name karmali hai
jisme city mumbai ho
station code swv wale records
jahan station name chennai hai dikhao
```

### Code / ID filters

```
records dikho jaha station code swv hai
show rows where employee id is E102
filter where product code ABC123
```

### Numeric comparisons

```
age greater than 30
salary less than 50000
age at least 18
price at most 1000
marks >= 75
experience > 5
```

### High / low (relative thresholds)

```
high salary
low age
employees with high experience
```

> For `high` / `low`, the system uses the 75th / 25th percentile of the matched numeric column.

### Missing / empty values

```
give records where age is none
age is missing
age is null
jisme age khali hai
records where salary is empty
show rows where station name is na
```

### Not missing (has a value)

```
age is not empty
records where age is not missing
salary is not null
```

### Combined-style queries

```
show records where age greater than 30
give data where station code swv and city mumbai
records where salary high
```

### Visualization queries

```
show age distribution
plot salary histogram
age distribution chart
bar chart of city
scatter plot between age and salary
plot age vs salary
show count of department
pie chart of category
box plot of salary
line chart of sales over time
```

---

## Action modes

| Mode | When to use |
|------|-------------|
| **Auto detect** | Default. The app decides filter vs chart from your query. |
| **Data Filtering** | Force row filtering only. Best when you only want table results. |
| **Data Visualization** | Force chart generation only. Best when you only want graphs. |

---

## Safety model

The app **never** runs arbitrary LLM-generated Python. Filter expressions must pass all checks:

| Check | Rule |
|-------|------|
| Format | Single line, starts with `df[` |
| Columns | Only names that exist in your uploaded file |
| AST allowlist | No imports, assignments, loops, `df.query()`, or unknown function calls |
| Allowed operations | Comparisons, `pd.to_numeric`, `.str.contains`, `.isna()`, `.notna()`, `.isin()` |
| Execution | Restricted `eval()` with no builtins |

If unsafe or invalid code is generated, the app shows an error and may retry once with an LLM correction.

**Visualization** uses predefined chart templates only — the LLM returns a JSON spec (`type` + `columns`), not executable matplotlib code.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Missing HF_API_KEY` | No API key in environment | Create `.env` with `HF_API_KEY=...` and restart Streamlit |
| `LLM routing failed` | Auto detect could not classify query | Use **Data Filtering** or **Data Visualization** mode |
| `Column not found` | Query references a column not in your file | Check column names in the data preview |
| `Unsafe code generated` | Model returned disallowed code | Rephrase query; use simpler wording |
| `LLM generation failed` | API timeout or bad response | Retry; check Hugging Face token and network |
| No / wrong results | Value mismatch (case, spelling) | Try `contains` phrasing or check sample values in preview |

### Logs

Application logs are written to:

```
logs/nlp2filter.log
```

---

## Tech stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| Data | pandas, openpyxl, numpy |
| LLM | Hugging Face Router API |
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct:nscale` |
| Charts | matplotlib |
| Config | python-dotenv |
| HTTP | requests |

---

## License & notes

- Upload only data you are allowed to process.
- LLM responses depend on model quality and your column naming — use the **How to write queries** panel in the app for live examples based on your file.
- Restart Streamlit after changing `.env`.
