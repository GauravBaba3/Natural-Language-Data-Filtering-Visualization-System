import logging

import pandas as pd

from validator import UnsafeCodeError, ValidationError, validate_pandas_filter_expression

logger = logging.getLogger("nlp2filter")


class ExecutionError(Exception):
    pass


def safe_execute_filter(expression: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely execute an LLM-generated pandas filtering expression.

    Safety gates:
    - Expression must be validated by `validator`.
    - AST allowlist forbids calls/attributes/imports/etc.
    - eval runs with `__builtins__` disabled.
    """
    try:
        validate_pandas_filter_expression(expression, list(df.columns))
        # Convert boolean mask result into df[...] selection (pandas handles it).
        safe_globals = {"__builtins__": {}}
        safe_locals = {"df": df, "pd": pd}
        result = eval(compile(expression, "<nlp2filter_expr>", "eval"), safe_globals, safe_locals)
        if not isinstance(result, pd.DataFrame):
            raise ExecutionError("Expression did not produce a DataFrame")
        return result
    except ValidationError:
        raise
    except UnsafeCodeError:
        raise
    except SyntaxError as e:
        # Provide a controlled error for retry handling upstream.
        raise ExecutionError(f"SyntaxError: {e}") from e
    except TypeError as e:
        raise ExecutionError(
            "Type mismatch during filtering. Numeric comparisons require numeric data."
        ) from e
    except Exception as e:
        raise ExecutionError(str(e)) from e

