import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple

import pandas as pd

from preprocessor import ColumnProfile


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("nlp2filter")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    log_path = os.path.join(log_dir, "nlp2filter.log")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def normalize_query_for_operators(query: str) -> str:
    """
    Normalize casual / Hinglish phrasing so the LLM can interpret intent reliably.
    """
    q = query.strip()
    q = re.sub(
        r"\b(give|show|get|fetch|display|dikhao|dikho|batao|nikalo)\s+(me\s+)?(the\s+)?(all\s+)?(records?|rows?|data|entries?)\s+(where|jahan|jaha|jisme|jinme|jin|in which)?\s*",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(
        r"\b(records?|rows?|data)\s+(dikhao|dikho|batao|nikalo)\s+(jahan|jaha|jisme|where)?\s*",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\b(records?|rows?|data)\s+(where|jahan|jaha|jisme)\s+", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(where|jahan|jaha|jisme|jinme|jin|in which)\s+", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(dikhao|dikho|batao|nikalo)\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(hai|hain|ho|tha|the|thi)\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(older than|greater than|more than|above|over|se zyada|zyada hai)\b", ">", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(younger than|less than|below|under|se kam|kam hai)\b", "<", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(equal to|equals|is equal to|same as)\b", "==", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(at least|minimum|min)\b", ">=", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(at most|maximum|max)\b", "<=", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(is not|are not|not equal to|!=)\b", "!=", q, flags=re.IGNORECASE)
    q = re.sub(
        r"\b(is|are|equals?|equal to|hai|hain|ho|mein|me|ka|ki|ke)\s+(none|null|empty|blank|missing|na|n/?a|khali)\b",
        " is missing ",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(
        r"\b(none|null|empty|blank|missing|na|n/?a|khali)\s+(value|values|hai|hain|ho)?\b",
        " missing ",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(
        r"\b(has|have)\s+no\s+(value|data)\b",
        " is missing ",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(
        r"\b(is not|are not|not)\s+(none|null|empty|blank|missing|na|n/?a|khali)\b",
        " is not missing ",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\s+", " ", q)
    return q.strip()


_QUERY_STOP_WORDS = frozenset(
    {
        "give",
        "show",
        "get",
        "fetch",
        "records",
        "record",
        "rows",
        "row",
        "data",
        "where",
        "jahan",
        "jisme",
        "jinme",
        "jin",
        "the",
        "me",
        "all",
        "with",
        "is",
        "are",
        "was",
        "were",
        "hai",
        "hain",
        "ho",
        "ka",
        "ki",
        "ke",
        "mein",
        "me",
        "none",
        "null",
        "empty",
        "blank",
        "missing",
        "na",
        "n/a",
        "khali",
        "value",
        "values",
        "dikhao",
        "dikho",
        "batao",
        "nikalo",
        "jaha",
        "please",
        "a",
        "an",
        "of",
        "for",
        "in",
        "on",
        "to",
        "and",
        "or",
    }
)


def merge_hint_texts(*hints: Optional[str]) -> Optional[str]:
    parts = [h.strip() for h in hints if h and h.strip()]
    return "\n".join(parts) if parts else None


def build_query_interpretation_hints(query: str, df: pd.DataFrame) -> Optional[str]:
    """
    Turn casual / Hinglish phrasing into explicit pandas intent for the LLM.
    """
    q_l = query.lower()
    hints: list[str] = []

    null_patterns = (
        r"\bis\s+(none|null|empty|blank|missing|na|n/?a|khali)\b",
        r"\bare\s+(none|null|empty|blank|missing|na|n/?a|khali)\b",
        r"\b(none|null|empty|blank|missing|na|n/?a|khali)\s+(value|values|hai|hain|ho)?\b",
        r"\bhas\s+no\s+(value|data)\b",
        r"\bis\s+missing\b",
        r"\bno\s+(value|data)\b",
    )
    not_null_patterns = (
        r"\bis\s+not\s+(none|null|empty|blank|missing|na|n/?a|khali)\b",
        r"\bare\s+not\s+(none|null|empty|blank|missing|na|n/?a|khali)\b",
        r"\bnot\s+(none|null|empty|blank|missing|na|n/?a|khali)\b",
        r"\bis\s+not\s+missing\b",
        r"\bhas\s+(a\s+)?(value|data)\b",
    )

    referenced_col = _best_column_match(query, list(df.columns))
    ranked_cols = rank_columns_for_query(query, list(df.columns))
    filter_value = _extract_filter_value(query, referenced_col)
    wants_null = any(re.search(p, q_l) for p in null_patterns)
    wants_not_null = any(re.search(p, q_l) for p in not_null_patterns)

    if wants_null and not wants_not_null:
        if referenced_col:
            hints.append(
                f'User means rows where `{referenced_col}` is NULL / missing / empty / NA. '
                f'Use: df[(df["{referenced_col}"].isna())]. '
                f'Do NOT compare to the string "none".'
            )
        else:
            hints.append(
                "User means NULL / missing / empty / NA values. "
                'Use df[(df["<column>"].isna())] for the relevant column. '
                'Do NOT compare to the string "none".'
            )

    if wants_not_null and not wants_null:
        if referenced_col:
            hints.append(
                f'User means rows where `{referenced_col}` has a value (is not missing). '
                f'Use: df[(df["{referenced_col}"].notna())]'
            )
        else:
            hints.append(
                "User means rows with a present value (not missing). "
                'Use df[(df["<column>"].notna())] for the relevant column.'
            )

    if referenced_col and filter_value and not wants_null and not wants_not_null:
        hints.append(
            f'User wants rows where `{referenced_col}` matches "{filter_value}". '
            f'Use exact match: df[(df["{referenced_col}"] == "{filter_value}")] '
            f'or partial match: df[df["{referenced_col}"].str.contains("{filter_value}", case=False)]. '
            f'"{filter_value}" is a VALUE, not a column name.'
        )

    if ranked_cols:
        top_cols = ", ".join(f"`{col}`" for col, _ in ranked_cols[:3])
        hints.append(f"Best matching columns from the query: {top_cols}.")
    elif referenced_col:
        hints.append(f'Most likely column from the query: `{referenced_col}`.')

    return "\n".join(hints) if hints else None


def _normalize_column_key(name: str) -> str:
    return re.sub(r"[\s_]+", "", name.lower())


def rank_columns_for_query(query: str, columns: list[str]) -> list[tuple[str, float]]:
    """
    Rank dataframe columns by how likely the user referenced them in the query.
    Supports multi-word names like "station code".
    """
    q_clean = re.sub(r"[^a-zA-Z0-9_ ]+", " ", query.lower())
    q_compact = re.sub(r"\s+", "", q_clean)
    tokens = [t for t in q_clean.split() if t and t not in _QUERY_STOP_WORDS and not t.isdigit()]

    scored: list[tuple[float, str]] = []
    for col in columns:
        col_l = col.lower()
        col_spaced = re.sub(r"[_]+", " ", col_l).strip()
        col_compact = _normalize_column_key(col)
        col_tokens = [t for t in re.split(r"[\s_]+", col_l) if t]

        score = 0.0
        if col_spaced and col_spaced in q_clean:
            score += 4.0
        if col_compact and col_compact in q_compact:
            score += 3.5
        if col_tokens and all(t in tokens for t in col_tokens):
            score += 2.5
        score += sum(0.6 for t in col_tokens if t in tokens)
        score += SequenceMatcher(None, q_clean, col_spaced).ratio()
        if col_l in tokens or col_compact in tokens:
            score += 1.0
        scored.append((score, col))

    scored.sort(reverse=True)
    return [(col, score) for score, col in scored if score > 0.4]


def _extract_filter_value(query: str, column: Optional[str]) -> Optional[str]:
    if not column:
        return None

    q_work = re.sub(r"[^a-zA-Z0-9_ ]+", " ", query.lower())
    for token in re.split(r"[\s_]+", column.lower()):
        if token:
            q_work = re.sub(rf"\b{re.escape(token)}\b", " ", q_work)

    tokens = [t for t in q_work.split() if t and t not in _QUERY_STOP_WORDS and not t.isdigit()]
    if not tokens:
        return None
    return tokens[-1]


def _best_column_match(query: str, columns: list[str]) -> Optional[str]:
    ranked = rank_columns_for_query(query, columns)
    if ranked:
        return ranked[0][0]
    return None


@dataclass(frozen=True)
class AmbiguityResult:
    is_ambiguous: bool
    clarifier_question: Optional[str] = None
    heuristic_constraints_text: Optional[str] = None
    updated_query: Optional[str] = None


def detect_and_apply_heuristics_for_high_low(
    query: str, df: pd.DataFrame
) -> AmbiguityResult:
    """
    Example ambiguity: "high salary" without an explicit number.
    Heuristic: map "high" to q75 and "low" to q25 for the referenced numeric column.
    """
    q_l = query.lower()
    if ("high" not in q_l and "low" not in q_l) or re.search(r"\d", q_l):
        return AmbiguityResult(is_ambiguous=False)

    referenced_col = _best_column_match(query, list(df.columns))
    if not referenced_col:
        return AmbiguityResult(
            is_ambiguous=True,
            clarifier_question="Which column should “high/low” apply to (e.g., salary, amount, age)?",
        )

    # Only apply this heuristic to numeric columns.
    if not pd.api.types.is_numeric_dtype(df[referenced_col]):
        return AmbiguityResult(
            is_ambiguous=True,
            clarifier_question=f'“high/low” needs a numeric column; is `{referenced_col}` numeric?',
        )

    high_is_requested = "high" in q_l
    low_is_requested = "low" in q_l

    constraints: list[str] = []
    if high_is_requested:
        thr = float(df[referenced_col].quantile(0.75))
        constraints.append(f'For `{referenced_col}`, interpret "high" as `>= {thr}`.')
    if low_is_requested:
        thr = float(df[referenced_col].quantile(0.25))
        constraints.append(f'For `{referenced_col}`, interpret "low" as `<= {thr}`.')

    heuristic_text = "\n".join(constraints)
    return AmbiguityResult(
        is_ambiguous=False,
        heuristic_constraints_text=heuristic_text,
        updated_query=query.strip(),
    )


def build_column_context(
    df: pd.DataFrame, column_profiles: Optional[Dict[str, ColumnProfile]] = None
) -> Dict[str, str]:
    """
    Provide small context for LLM: numeric vs categorical based on dtypes and optional
    preprocessing metadata.
    """
    ctx: Dict[str, str] = {}
    for col in df.columns:
        if column_profiles and col in column_profiles:
            profile = column_profiles[col]
            if profile.converted:
                ctx[col] = "numeric (auto-converted from string with pd.to_numeric)"
            else:
                ctx[col] = profile.detected_type
        elif pd.api.types.is_numeric_dtype(df[col]):
            ctx[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            ctx[col] = "datetime"
        else:
            ctx[col] = "categorical"
    return ctx


def build_column_reference_for_llm(
    df: pd.DataFrame,
    column_profiles: Optional[Dict[str, ColumnProfile]] = None,
    *,
    max_samples: int = 8,
) -> str:
    """
    Rich column reference for the LLM: exact names, types, and sample values.
    """
    column_context = build_column_context(df, column_profiles)
    lines = [
        "Use ONLY these exact column names in df[\"...\"]. "
        "Users may say column names in a different case or with spaces/underscores."
    ]

    for col in df.columns:
        dtype = column_context.get(col, "unknown")
        non_null = df[col].dropna()
        if non_null.empty:
            sample_text = "(all missing)"
        else:
            unique_vals = non_null.astype(str).unique()[:max_samples]
            samples = ", ".join(f'"{v}"' for v in unique_vals)
            extra = ""
            if non_null.astype(str).nunique() > max_samples:
                extra = ", ..."
            sample_text = f"{samples}{extra}"

        lines.append(f'- `{col}` | type: {dtype} | sample values: {sample_text}')

    return "\n".join(lines)


def get_query_guide_examples(df: Optional[pd.DataFrame] = None) -> dict[str, list[str]]:
    """
    Example queries shown in the UI. Uses real column names when a dataframe is loaded.
    """
    text_col: Optional[str] = None
    numeric_col: Optional[str] = None
    text_sample = "karmali"
    code_col: Optional[str] = None
    code_sample = "swv"

    if df is not None and not df.empty:
        for col in df.columns:
            series = df[col]
            if text_col is None and not pd.api.types.is_numeric_dtype(series):
                text_col = col
                non_null = series.dropna().astype(str)
                if not non_null.empty:
                    text_sample = non_null.iloc[0]
            if numeric_col is None and pd.api.types.is_numeric_dtype(series):
                numeric_col = col
            col_l = str(col).lower()
            if code_col is None and ("code" in col_l or "id" in col_l):
                code_col = col
                non_null = series.dropna().astype(str)
                if not non_null.empty:
                    code_sample = non_null.iloc[0]

    text_col = text_col or "station name"
    numeric_col = numeric_col or "age"
    code_col = code_col or "station code"

    return {
        "Text match (English)": [
            f'show records where {text_col} is {text_sample}',
            f'{text_col} equals {text_sample}',
            f'filter rows with {text_col} containing {text_sample[:3]}',
        ],
        "Text match (Hinglish)": [
            f'records dikho jaha {text_col} {text_sample} hai',
            f'jisme {text_col} {text_sample} ho',
            f'{text_col} {text_sample} wale records',
        ],
        "Code / ID match": [
            f'records dikho jaha {code_col} {code_sample} hai',
            f'show rows where {code_col} is {code_sample}',
        ],
        "Numbers": [
            f'{numeric_col} greater than 30',
            f'{numeric_col} less than 50',
            f'{numeric_col} at least 25',
            f'high {numeric_col}',
            f'low {numeric_col}',
        ],
        "Missing values": [
            f'give records where {numeric_col} is none',
            f'{numeric_col} is missing',
            f'jisme {numeric_col} khali hai',
            f'{numeric_col} is not empty',
        ],
        "Visualization": [
            f'show {numeric_col} distribution',
            f'plot {numeric_col} histogram',
            f'scatter plot between {numeric_col} and {text_col}',
            f'bar chart of {text_col}',
        ],
    }


def format_query_guide_markdown(df: Optional[pd.DataFrame] = None) -> str:
    """Short markdown guide for how to write queries."""
    lines = [
        "Write queries in **plain English** or **Hinglish**. Mention the **column name** and the **value** you want.",
        "",
        "**Tips**",
        "- Use column names from your uploaded file (see preview above).",
        "- Words like `show`, `give`, `records`, `dikho`, `jahan`, `jisme`, `hai` are understood.",
        "- For missing/empty data use: `none`, `null`, `missing`, `khali`.",
        "- For charts, use: `show`, `plot`, `distribution`, `histogram`, `scatter`.",
        "",
        "**Pattern**",
        "- Filter: `[action] [column] [condition] [value]`",
        "- Example: `records dikho jaha station code swv hai`",
        "",
    ]
    if df is not None:
        lines.append("**Your columns**")
        for col in df.columns:
            lines.append(f"- `{col}`")
        lines.append("")
    return "\n".join(lines)

