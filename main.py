from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import data_loader
import executor
import llm_engine
import preprocessor
import utils
import validator
import visualization


LOGGER = utils.setup_logging()
load_dotenv()

st.set_page_config(page_title="NLP2Query Filtering & Visualization", layout="wide")
st.title("Natural Language to Data Filtering & Visualization System")

st.caption("Uploads a CSV/Excel file, converts natural language to safe pandas filtering, and displays results.")

if "df" not in st.session_state:
    st.session_state.df = None
if "df_name" not in st.session_state:
    st.session_state.df_name = None
if "column_profiles" not in st.session_state:
    st.session_state.column_profiles = None


uploaded = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
)

if uploaded is not None:
    try:
        raw_df, df_name = data_loader.load_dataframe(uploaded)
        df, column_profiles = preprocessor.clean_dataframe(raw_df)
        st.session_state.df = df
        st.session_state.df_name = df_name
        st.session_state.column_profiles = column_profiles
        st.success(f"Loaded `{df_name}` with {len(df)} rows and {len(df.columns)} columns.")
        converted_columns = [
            col for col, profile in column_profiles.items() if profile.converted
        ]
        if converted_columns:
            st.info(
                "Auto-converted numeric-like columns: "
                + ", ".join(f"`{col}`" for col in converted_columns)
            )
    except preprocessor.PreprocessingError as e:
        st.error(f"Data preprocessing failed: {e}")
    except Exception as e:
        st.error(f"Failed to load file: {e}")

df: pd.DataFrame | None = st.session_state.df
column_profiles = st.session_state.column_profiles
if df is not None:
    with st.expander("Preview data (head)"):
        st.dataframe(df.head(50), use_container_width=True)


st.divider()

with st.expander("Graph settings", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        graph_width = st.slider("Width", min_value=3.0, max_value=16.0, value=8.0, step=0.5)
        graph_bins = st.slider("Histogram bins", min_value=5, max_value=100, value=30, step=1)
        graph_color = st.color_picker("Primary color", value="#1f77b4")
    with c2:
        graph_height = st.slider("Height", min_value=2.0, max_value=10.0, value=3.5, step=0.5)
        graph_top_n = st.slider("Top categories (bar/pie/count)", min_value=3, max_value=50, value=20, step=1)
        graph_alpha = st.slider("Opacity", min_value=0.2, max_value=1.0, value=0.85, step=0.05)
    graph_grid = st.toggle("Show grid", value=True)

with st.expander("How to write queries", expanded=df is None):
    st.markdown(utils.format_query_guide_markdown(df))
    example_groups = utils.get_query_guide_examples(df)
    for group_name, examples in example_groups.items():
        st.markdown(f"**{group_name}**")
        ex_cols = st.columns(2)
        for idx, example in enumerate(examples):
            with ex_cols[idx % 2]:
                if st.button(example, key=f"query_example_{group_name}_{idx}", use_container_width=True):
                    st.session_state.query_text = example

query = st.text_input(
    "Enter your natural language query",
    key="query_text",
    placeholder='e.g. "records dikho jaha station code swv hai" or "age greater than 30"',
)
mode = st.radio(
    "Choose action",
    options=["Auto detect", "Data Filtering", "Data Visualization"],
    horizontal=True,
)
run = st.button("Run", type="primary")

if run:
    if df is None:
        st.error("Please upload a CSV/Excel file first.")
    elif not query.strip():
        st.error("Please enter a query.")
    else:
        try:
            LOGGER.info("Received query: %s", query)
            original_query = query.strip()
            normalized_query = utils.normalize_query_for_operators(original_query)
            column_context = utils.build_column_context(df, column_profiles)
            column_reference = utils.build_column_reference_for_llm(df, column_profiles)
            interpretation_hints = utils.build_query_interpretation_hints(original_query, df)
            heuristic_constraints_text = interpretation_hints

            # 1) Determine mode.
            if mode == "Data Filtering":
                action = llm_engine.RoutedAction(intent="filter")
            elif mode == "Data Visualization":
                action = llm_engine.generate_visualization_spec(
                    original_query,
                    list(df.columns),
                    column_context,
                )
            else:
                # Auto detect from the query.
                action = llm_engine.route_query(
                    normalized_query,
                    list(df.columns),
                    column_context,
                )

            if action.intent == "filter":
                # Heuristic handling for common ambiguity in filter mode.
                heur = utils.detect_and_apply_heuristics_for_high_low(original_query, df)
                heuristic_constraints_text = utils.merge_hint_texts(
                    interpretation_hints,
                    heur.heuristic_constraints_text,
                )
                if heur.is_ambiguous:
                    # If heuristics can't resolve, ask LLM for a clarifier question.
                    col_ctx = utils.build_column_context(df, column_profiles)
                    question = llm_engine.clarify_ambiguity_question(
                        original_query, list(df.columns), col_ctx
                    )
                    st.warning(question or "Please clarify your query (e.g., specify the column and threshold).")
                    st.stop()

            if action.intent == "visualization":
                try:
                    spec = visualization.ChartSpec(
                        type=action.chart_type or "",
                        columns=action.columns or [],
                    )
                    options = visualization.ChartOptions(
                        width=graph_width,
                        height=graph_height,
                        bins=graph_bins,
                        top_n=graph_top_n,
                        color=graph_color,
                        alpha=graph_alpha,
                        grid=graph_grid,
                    )
                    fig = visualization.generate_chart(df, normalized_query, spec, options=options)
                    st.success("Visualization generated.")
                    st.pyplot(fig, use_container_width=True)
                except visualization.VisualizationError as ve:
                    st.error(str(ve))
                st.stop()

            # Always generate filter code via the dedicated LLM prompt.
            expression = llm_engine.generate_pandas_filter_expression(
                original_query,
                list(df.columns),
                column_context,
                heuristic_constraints_text=heuristic_constraints_text,
                normalized_query=normalized_query,
                column_reference=column_reference,
            )

            LOGGER.info("LLM filter expression: %s", expression)

            # 2) Validate + execute safely (filter intent).
            current_expr = expression
            result_df: pd.DataFrame | None = None
            last_exec_err: Exception | None = None

            for attempt_idx in range(2):  # initial + 1 correction
                try:
                    result_df = executor.safe_execute_filter(current_expr, df)
                    last_exec_err = None
                    break
                except validator.ValidationError as ve:
                    if ve.invalid_columns:
                        st.error("Column not found")
                        st.stop()
                    if attempt_idx == 0:
                        LOGGER.warning("Validation error. Will try LLM correction: %s", ve)
                        current_expr = llm_engine.fix_pandas_filter_expression(
                            original_query,
                            list(df.columns),
                            column_context,
                            current_expr,
                            str(ve),
                            column_reference=column_reference,
                        )
                        continue
                    st.error(str(ve))
                    st.stop()
                except validator.UnsafeCodeError as ue:
                    if attempt_idx == 0:
                        LOGGER.warning("Unsafe code. Will try LLM correction: %s", ue)
                        current_expr = llm_engine.fix_pandas_filter_expression(
                            original_query,
                            list(df.columns),
                            column_context,
                            current_expr,
                            str(ue),
                            column_reference=column_reference,
                        )
                        continue
                    st.error("Unsafe code generated. Please try a different query.")
                    st.stop()
                except executor.ExecutionError as ee:
                    last_exec_err = ee
                    err_str = str(ee)
                    # Requirement: only auto-correct on syntax errors.
                    if attempt_idx == 0 and err_str.startswith("SyntaxError"):
                        LOGGER.warning("Syntax error. Will try LLM correction: %s", err_str)
                        current_expr = llm_engine.fix_pandas_filter_expression(
                            original_query,
                            list(df.columns),
                            column_context,
                            current_expr,
                            err_str,
                            column_reference=column_reference,
                        )
                        continue
                    break

            if result_df is None:
                st.error(f"Failed to execute filter: {last_exec_err}")
                st.stop()

            st.success(f"Filtered rows: {len(result_df)}")
            st.dataframe(result_df.head(500), use_container_width=True)
            if len(result_df) > 500:
                st.caption("Showing first 500 rows. Refine your query to narrow results.")

        except llm_engine.GroqLLMError as e:
            st.error(f"LLM error: {e}")
        except validator.ValidationError:
            st.error("Column not found")
        except validator.UnsafeCodeError:
            st.error("Unsafe code generated. Please try a different query.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


#
# ML training UI removed per request.
#
