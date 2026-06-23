from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import requests


API_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct:nscale"


class GroqLLMError(Exception):
    pass


def _get_api_key() -> str:
    api_key = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise GroqLLMError(
            "Missing HF_API_KEY. Set it in your environment or create a .env file."
        )
    return api_key


def _chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> str:
    headers = {"Authorization": f"Bearer {_get_api_key()}"}
    payload: dict = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as e:
        raise GroqLLMError(f"Hugging Face API request failed: {e}") from e

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise GroqLLMError(f"Unexpected Hugging Face API response: {response.text}") from e


def _extract_first_df_expression(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer a single-line expression. The system prompt asks the model to output
    # ONLY a single pandas filtering expression, so we can safely take the first
    # line that starts with `df[`.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("df["):
            return stripped

    # Fallback: best-effort scan. Take substring from first `df[` until the next
    # newline; this avoids premature termination inside `df["col"]`.
    start = text.find("df[")
    if start == -1:
        return None
    end = text.find("\n", start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


SYSTEM_PROMPT = """You are a strict code generator for a pandas filtering system.

The user may write in casual English, Hindi-English (Hinglish), or informal grammar.
Understand their intent, not exact wording.

Rules:
1. Output ONLY a single pandas filtering expression as plain text. No markdown. No explanations.
2. The DataFrame variable name MUST be exactly `df`.
3. The expression MUST start with `df[`.
4. Only use the provided column names when you write `df["<col>"]`.
5. For boolean conditions, use parentheses around each condition and combine with `&` / `|`.
6. If a column is numeric but stored as string, use `pd.to_numeric(df["<col>"], errors="coerce")` for numeric comparisons such as >, <, >=, <=.
7. Comparisons against converted numeric columns must remain null-safe; `errors="coerce"` is required so invalid values become NaN and are ignored by the comparison.
8. Do NOT write imports, function definitions, assignments, loops, df.query(), apply(), or any code outside the pandas filtering expression.
9. The final output must be a valid Python expression that starts with `df[`.
10. For exact text matches, prefer: df[(df["column"] == "value")]
11. For partial text matches you may use: df[df["column"].str.contains("value", case=False)]
12. For NULL / missing / empty / none / NA values, ALWAYS use: df[(df["column"].isna())]
13. For NOT missing / has value, use: df[(df["column"].notna())]
14. NEVER use == None, == "none", == "null", or == "" when the user means missing data.
15. Match column names exactly as provided in the allowed columns list.

Examples:
- "give records where age is none" -> df[(df["age"].isna())]
- "station name is karmali" -> df[(df["station name"] == "karmali")]
- "records dikho jaha station code swv hai" -> df[(df["station code"] == "swv")]
- "salary greater than 50000" -> df[(pd.to_numeric(df["salary"], errors="coerce") > 50000)]
- "age is not empty" -> df[(df["age"].notna())]
"""


def _column_section(
    column_reference: Optional[str],
    df_columns: list[str],
    column_context: Dict[str, str],
) -> str:
    if column_reference:
        return column_reference
    cols_str = ", ".join([f"`{c}`" for c in df_columns])
    type_hint = "; ".join([f"`{k}`: {v}" for k, v in sorted(column_context.items())])
    return f"Allowed columns: {cols_str}\nColumn types: {type_hint}"


ROUTER_SYSTEM_PROMPT = """You route user queries for a pandas DataFrame system.

Task:
- Classify the user's request as either "filter" or "visualization" only.

If intent is "filter":
- Return JSON: {"intent": "filter"}
- Filtering means the user wants to see/select rows from the table.

If intent is "visualization":
- Return JSON with keys: intent, type, columns
- type MUST be one of: bar, line, hist, pie, scatter, box, count
- columns MUST be a list of 1 (or 2 for scatter) column names from the allowed columns
- Visualization means charts/plots/histograms/distributions/scatter graphs.

Output rules:
- Output ONLY valid JSON. No markdown. No explanations. No pandas code for filter intent.
"""

VISUALIZATION_SYSTEM_PROMPT = """You generate visualization specs for a pandas DataFrame.

Return ONLY JSON with this exact structure:
{
  "type": "bar|line|hist|pie|scatter|box|count",
  "columns": ["col1"]  // or ["col1","col2"] for scatter
}

Rules:
- Use only allowed columns.
- For scatter, return exactly two columns.
- For all other chart types, return exactly one column.
- Do NOT return code or explanations.
"""


@dataclass(frozen=True)
class RoutedAction:
    intent: str  # "filter" | "visualization"
    expression: Optional[str] = None
    chart_type: Optional[str] = None
    columns: Optional[list[str]] = None


def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1].strip()


def _heuristic_route_intent(query: str) -> str:
    q = query.lower()
    viz_keywords = (
        "plot",
        "chart",
        "graph",
        "histogram",
        "distribution",
        "scatter",
        "bar chart",
        "pie chart",
        "visualiz",
        "vs ",
    )
    if any(keyword in q for keyword in viz_keywords):
        return "visualization"
    return "filter"


def route_query(
    query: str,
    df_columns: list[str],
    column_context: Dict[str, str],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 1,
) -> RoutedAction:
    """
    Use the LLM to decide if a query is for filtering or visualization.
    Filter code is generated separately; routing only classifies intent.
    """
    cols_str = ", ".join([f"`{c}`" for c in df_columns])
    type_hint = "; ".join([f"`{k}`: {v}" for k, v in sorted(column_context.items())])

    prompt = f"""User query:
{query}

Allowed columns:
{cols_str}

Column types (hint):
{type_hint}
"""

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            content = _chat_completion(
                [
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=temperature,
            )
            json_text = _extract_json_object(content)
            if not json_text:
                raise GroqLLMError("LLM did not return JSON")
            obj = json.loads(json_text)
            intent = str(obj.get("intent", "")).strip().lower()
            if intent not in {"filter", "visualization"}:
                raise GroqLLMError("Invalid intent in JSON")

            if intent == "filter":
                return RoutedAction(intent="filter")

            chart_type = str(obj.get("type", "")).strip().lower()
            columns = obj.get("columns")
            if chart_type not in {"bar", "line", "hist", "pie", "scatter", "box", "count"}:
                raise GroqLLMError("Invalid chart type in JSON")
            if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
                raise GroqLLMError("Invalid columns in JSON")
            return RoutedAction(
                intent="visualization",
                chart_type=chart_type,
                columns=[c.strip() for c in columns],
            )
        except Exception as e:
            last_err = e
            time.sleep(0.5 + attempt * 0.5)

    # Fallback: keyword-based intent when the router model returns bad JSON.
    fallback_intent = _heuristic_route_intent(query)
    if fallback_intent == "filter":
        return RoutedAction(intent="filter")
    raise GroqLLMError(f"LLM routing failed: {last_err}")


def generate_visualization_spec(
    query: str,
    df_columns: list[str],
    column_context: Dict[str, str],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 1,
) -> RoutedAction:
    """
    Force visualization intent and return chart type + columns.
    """
    cols_str = ", ".join([f"`{c}`" for c in df_columns])
    type_hint = "; ".join([f"`{k}`: {v}" for k, v in sorted(column_context.items())])
    prompt = f"""User query:
{query}

Allowed columns:
{cols_str}

Column types (hint):
{type_hint}
"""

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            content = _chat_completion(
                [
                    {"role": "system", "content": VISUALIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=temperature,
            )
            json_text = _extract_json_object(content)
            if not json_text:
                raise GroqLLMError("LLM did not return visualization JSON")
            obj = json.loads(json_text)
            chart_type = str(obj.get("type", "")).strip().lower()
            columns = obj.get("columns")
            if chart_type not in {"bar", "line", "hist", "pie", "scatter", "box", "count"}:
                raise GroqLLMError("Invalid chart type in JSON")
            if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
                raise GroqLLMError("Invalid columns in JSON")
            return RoutedAction(
                intent="visualization",
                chart_type=chart_type,
                columns=[c.strip() for c in columns],
            )
        except Exception as e:
            last_err = e
            time.sleep(0.5 + attempt * 0.5)
    raise GroqLLMError(f"Visualization spec generation failed: {last_err}")


def generate_pandas_filter_expression(
    query: str,
    df_columns: list[str],
    column_context: Dict[str, str],
    heuristic_constraints_text: Optional[str] = None,
    *,
    normalized_query: Optional[str] = None,
    column_reference: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 1,
) -> str:
    """
    Returns an expression starting with `df[` that filters rows.
    """
    column_section = _column_section(column_reference, df_columns, column_context)
    heuristic_hint = ""
    if heuristic_constraints_text:
        heuristic_hint = f"\nInterpretation hints (must follow):\n{heuristic_constraints_text}\n"

    normalized = normalized_query or query
    user_prompt = f"""Original user query (any language/style):
{query}

Normalized query:
{normalized}

DataFrame column reference (use these EXACT column names in df["..."]):
{column_section}

Output format:
Return ONLY a single line pandas expression starting with df[ ... ]
"""

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            content = _chat_completion(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt + heuristic_hint},
                ],
                model=model,
                temperature=temperature,
            )
            expr = _extract_first_df_expression(content)
            if not expr or not expr.strip().startswith("df["):
                raise GroqLLMError("LLM did not return a df[...] expression")
            return expr.strip()
        except Exception as e:
            last_err = e
            time.sleep(0.5 + attempt * 0.5)
    raise GroqLLMError(f"LLM generation failed: {last_err}")


def fix_pandas_filter_expression(
    query: str,
    df_columns: list[str],
    column_context: Dict[str, str],
    bad_expression: str,
    error_message: str,
    *,
    column_reference: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> str:
    """
    Ask the LLM to correct the expression based on a validation/execution error.
    """
    column_section = _column_section(column_reference, df_columns, column_context)

    prompt = f"""We generated this pandas filtering expression:
{bad_expression}

It failed with this error:
{error_message}

Natural language query:
{query}

DataFrame column reference (use these EXACT column names in df["..."]):
{column_section}

Fix requirements:
- Return ONLY a single pandas filtering expression starting with df[
- Do not add explanations
- Keep using only allowed columns from the column reference above
- Keep the expression restricted to a boolean filtering selection
- For numeric comparisons on string-backed numeric columns, use `pd.to_numeric(df["<col>"], errors="coerce")`
"""

    content = _chat_completion(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
    )
    expr = _extract_first_df_expression(content)
    if not expr or not expr.strip().startswith("df["):
        raise GroqLLMError("LLM did not return a corrected df[...] expression")
    return expr.strip()


def clarify_ambiguity_question(
    query: str,
    df_columns: list[str],
    column_context: Dict[str, str],
) -> str:
    """
    Returns a short clarifying question (not code) when heuristics cannot resolve ambiguity.
    """
    cols_str = ", ".join([f"`{c}`" for c in df_columns])
    type_hint = "; ".join([f"`{k}`: {v}" for k, v in sorted(column_context.items())])

    sys = """You are a helpful assistant.
Return ONE short clarifying question to resolve ambiguity for a pandas filtering task.
Do not return code.
"""
    prompt = f"""Natural language query:
{query}

Allowed columns:
{cols_str}

Column types (hint):
{type_hint}

Return ONE clarifying question.
"""
    content = _chat_completion(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return content.strip()

