"""Unit tests for EvaluatorDAG construction, validation, execution, and visualization."""

import pandas as pd


def test_dag_run(simple_dag, sample_ctx):
    result = simple_dag.run(sample_ctx)
    assert result.is_safe()


def test_dag_run_with_dataframe(simple_dag):
    # "hello world" (space lowers avg below threshold) → safe
    # "helloworld"  (no space, avg = 0.5 = threshold)  → unsafe
    # Alternate even/odd prompt lengths to exercise both enricher paths.
    df = pd.DataFrame(
        {
            "prompt_text": ["a", "ab", "abc", "abcd"],  # odd, even, odd, even
            "sut_response": ["hello world", "helloworld", "hello world", "helloworld"],
        }
    )
    result_df = simple_dag.run_dataframe(df)

    assert len(result_df) == len(df)
    assert "prompt_text" in result_df.columns
    assert "sut_response" in result_df.columns
    verdicts = result_df[simple_dag.DATAFRAME_OUTPUT_COL].tolist()
    expected_verdicts = ["NonViolating", "Violating", "NonViolating", "Violating"]
    assert verdicts == expected_verdicts
