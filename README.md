# Natural Language to Data Filtering & visualization System

End-to-end Streamlit app that:
1. Lets users upload a CSV/Excel file.
2. Loads it into a pandas `df`.
3. Uses Groq LLM to convert a natural language query into a **safe** pandas filtering expression.
4. Validates + executes the expression and displays results.
5. Routes queries to either **filtering** or **visualization** (charts).
6. Optionally runs an ML training pipeline (sklearn) when a target column is provided.

## Prerequisites

Python 3.10+ recommended.

## Setup

1. Install dependencies:
   - `pip install -r requirements.txt`

2. Provide Groq credentials via environment variable:
   - Create a `.env` file based on `.env.example` (do **not** commit your real API key).

## Run

From the project directory:

```powershell
streamlit run main.py
```

Open the displayed local URL in your browser.

## Notes on Safety

The system never executes arbitrary LLM output. It validates that the generated expression:
- Starts with `df[`
- References only allowed column names
- Uses a restricted AST allowlist (no imports/calls/attributes)
- Executes via a restricted `eval` environment with no builtins

If unsafe or invalid code is generated, the app returns a clear error instead of executing it.

## Visualization

For visualization queries (e.g., "plot age distribution", "city-wise user count"), the LLM returns a JSON spec:
- chart `type`: bar/line/hist/pie/scatter/box/count
- `columns`: list of referenced columns

The app renders charts using safe, predefined templates (no raw code execution).

