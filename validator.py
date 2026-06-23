import ast
from typing import List, Set


class ValidationError(Exception):
    def __init__(self, message: str, *, invalid_columns: List[str] | None = None):
        super().__init__(message)
        self.invalid_columns = invalid_columns or []


class UnsafeCodeError(Exception):
    pass


ALLOWED_STR_METHODS = frozenset(
    {"contains", "lower", "upper", "strip", "startswith", "endswith", "match"}
)
ALLOWED_SERIES_METHODS = frozenset({"isna", "notna", "isin"})
ALLOWED_PD_METHODS = frozenset({"isna", "notna", "to_numeric"})
ALLOWED_CALL_KEYWORDS = frozenset({"case", "na", "regex", "errors"})


def extract_referenced_columns(expression: str) -> List[str]:
    """
    Extract df["<col>"] references from a pandas filtering expression.
    """
    parsed = ast.parse(expression, mode="eval")
    cols: List[str] = []
    for node in ast.walk(parsed):
        if isinstance(node, ast.Subscript):
            # We only accept df["col"].
            if not isinstance(node.value, ast.Name) or node.value.id != "df":
                continue
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                cols.append(sl.value)
    # Preserve order while de-duping.
    seen: Set[str] = set()
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _is_allowed_to_numeric_call(node: ast.Call) -> bool:
    """
    Allow exactly: pd.to_numeric(df["col"], errors="coerce")
    """
    if not isinstance(node.func, ast.Attribute):
        return False
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "pd":
        return False
    if node.func.attr != "to_numeric":
        return False
    if len(node.args) != 1:
        return False

    series_arg = node.args[0]
    if not isinstance(series_arg, ast.Subscript):
        return False
    if not isinstance(series_arg.value, ast.Name) or series_arg.value.id != "df":
        return False
    if not isinstance(series_arg.slice, ast.Constant) or not isinstance(series_arg.slice.value, str):
        return False

    if len(node.keywords) != 1:
        return False
    kw = node.keywords[0]
    if kw.arg != "errors":
        return False
    return isinstance(kw.value, ast.Constant) and kw.value.value == "coerce"


def _is_df_column_subscript(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    if not isinstance(node.value, ast.Name) or node.value.id != "df":
        return False
    sl = node.slice
    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
        return True
    if isinstance(sl, ast.Index) and isinstance(sl.value, ast.Constant):
        return isinstance(sl.value.value, str)
    return False


def _is_df_str_accessor(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "str"
        and _is_df_column_subscript(node.value)
    )


def _is_df_str_method_accessor(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr in ALLOWED_STR_METHODS
        and _is_df_str_accessor(node.value)
    )


def _keyword_args_are_safe(keywords: list[ast.keyword]) -> bool:
    for kw in keywords:
        if kw.arg not in ALLOWED_CALL_KEYWORDS:
            return False
        if not isinstance(kw.value, ast.Constant):
            return False
    return True


def _is_allowed_isin_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "isin":
        return False
    if not _is_df_column_subscript(node.func.value):
        return False
    if len(node.args) != 1:
        return False
    values = node.args[0]
    if not isinstance(values, (ast.List, ast.Tuple)):
        return False
    return all(isinstance(elt, ast.Constant) for elt in values.elts)


def _is_allowed_str_method_call(node: ast.Call) -> bool:
    if not _is_df_str_method_accessor(node.func):
        return False
    if node.func.attr in {"lower", "upper", "strip"}:
        return len(node.args) == 0 and not node.keywords
    if node.func.attr in {"contains", "startswith", "endswith", "match"}:
        if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
            return False
        return _keyword_args_are_safe(node.keywords)
    return False


def _is_allowed_series_null_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in {"isna", "notna"}:
        return False
    return _is_df_column_subscript(node.func.value) and not node.args and not node.keywords


def _is_allowed_pd_null_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "pd":
        return False
    if node.func.attr not in {"isna", "notna"}:
        return False
    if len(node.args) != 1 or node.keywords:
        return False
    return _is_df_column_subscript(node.args[0])


def _is_allowed_call(node: ast.Call) -> bool:
    if _is_allowed_to_numeric_call(node):
        return True
    if _is_allowed_isin_call(node):
        return True
    if _is_allowed_str_method_call(node):
        return True
    if _is_allowed_series_null_call(node):
        return True
    if _is_allowed_pd_null_call(node):
        return True
    return False


def _is_allowed_attribute(node: ast.Attribute) -> bool:
    if isinstance(node.value, ast.Name) and node.value.id == "pd" and node.attr in ALLOWED_PD_METHODS:
        return True
    if _is_df_str_accessor(node):
        return True
    if _is_df_str_method_accessor(node):
        return True
    if node.attr in ALLOWED_SERIES_METHODS and _is_df_column_subscript(node.value):
        return True
    return False


def _validate_literal_collection(node: ast.AST) -> None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise UnsafeCodeError("Collection literals must be list/tuple")
    for elt in node.elts:
        if not isinstance(elt, ast.Constant):
            raise UnsafeCodeError("Collection literals may only contain constants")


def _ast_is_safe(parsed: ast.AST) -> None:
    """
    Enforce a strict allowlist:
    - No calls, no attributes, no comprehensions
    - Only expressions that index `df[...]` and compare against literals/columns
    - Only `df` name is allowed
    """
    allowed_node_types = (
        ast.Expression,
        ast.Subscript,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Compare,
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.Eq,
        ast.NotEq,
        ast.BoolOp,  # allowed but discouraged; we will still restrict to allowed operators
        ast.And,
        ast.Or,
        ast.BinOp,
        ast.BitAnd,
        ast.BitOr,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.Tuple,
        ast.List,
        ast.Index,  # py<3.9 compatibility in AST
        ast.Slice,
        ast.Call,
        ast.Attribute,
        ast.keyword,
    )

    banned = (
        ast.Lambda,
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.ClassDef,
        ast.For,
        ast.While,
        ast.If,
        ast.With,
        ast.Try,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Dict,
        ast.Set,
        ast.DictComp,
        ast.ListComp,
        ast.SetComp,
        ast.GeneratorExp,
        ast.ListComp,
        ast.comprehension,
        ast.NamedExpr,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
    )

    for node in ast.walk(parsed):
        if isinstance(node, banned):
            raise UnsafeCodeError(f"Unsafe code node: {type(node).__name__}")
        if not isinstance(node, allowed_node_types):
            raise UnsafeCodeError(f"Disallowed node type: {type(node).__name__}")

        if isinstance(node, ast.Name) and node.id not in {"df", "pd"}:
            raise UnsafeCodeError(f"Only `df` name is allowed, got `{node.id}`")

        if isinstance(node, ast.BoolOp):
            # Allow `and/or` only if parentheses exist; we cannot fully enforce
            # parentheses, but we can at least ensure only And/Or is used.
            if not isinstance(node.op, (ast.And, ast.Or)):
                raise UnsafeCodeError("Unsafe boolean operator")

        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.BitAnd, ast.BitOr)):
                raise UnsafeCodeError("Unsafe binary boolean operator")

        if isinstance(node, ast.Attribute) and not _is_allowed_attribute(node):
            raise UnsafeCodeError("Disallowed attribute access")

        if isinstance(node, ast.Call) and not _is_allowed_call(node):
            raise UnsafeCodeError(
                "Only safe pandas filtering calls are allowed "
                "(to_numeric, isna, notna, isin, str.*)"
            )

        if isinstance(node, (ast.List, ast.Tuple)):
            _validate_literal_collection(node)


def validate_pandas_filter_expression(expression: str, df_columns: List[str]) -> List[str]:
    """
    Validate that `expression`:
    - Starts with `df[`
    - Is parseable as an `eval`-mode expression
    - Uses only safe AST nodes
    - References only allowed column names

    Returns the referenced columns.
    """
    if not isinstance(expression, str):
        raise ValidationError("Expression must be a string")

    expr = expression.strip()
    if "\n" in expr or "\r" in expr:
        raise ValidationError("Expression must be a single line")
    if not expr.startswith("df["):
        raise ValidationError("Expression must start with `df[`")
    if ("&" in expr or "|" in expr) and "(" not in expr:
        # Enforces the prompt requirement that boolean conditions should be parenthesized.
        raise UnsafeCodeError("Boolean conditions must be parenthesized")

    parsed = ast.parse(expr, mode="eval")

    # Ensure the top-level is `df[...]` indexing.
    if not isinstance(parsed.body, ast.Subscript):
        raise UnsafeCodeError("Top-level must be `df[...]`")
    if not isinstance(parsed.body.value, ast.Name) or parsed.body.value.id != "df":
        raise UnsafeCodeError("Top-level must index `df`")

    _ast_is_safe(parsed)

    referenced = extract_referenced_columns(expr)
    invalid = [c for c in referenced if c not in set(df_columns)]
    if invalid:
        # Requirement: user-friendly error message should be "Column not found".
        raise ValidationError("Column not found", invalid_columns=invalid)

    return referenced

