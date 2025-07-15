import copy

import narwhals as nw
import pandas as pd
import pytest

from tests import utils as u
from tests.base_tests import GenericTransformTests


class TestAggregateRowOverColumnsTransformerMethodsTransform(GenericTransformTests):
    """Tests for methods in AggregateRowOverColumnsTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "AggregateRowOverColumnsTransformer"

    def setup_method(self, method):
        """Setup method to ensure the test DataFrame contains the necessary columns."""
        self.df_dict = {
            "a": [1, 2, 3, 4, 8],
            "b": [2, 3, 4, 5, 9],
            "c": ["A", "B", "A", "B", "A"],
        }
        self.df = pd.DataFrame(self.df_dict)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_invalid_key_error(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an error is raised if the key column is not found."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["aggregations"] = ["min", "max"]
        args["key"] = "missing_key"

        df = u.dataframe_init_dispatch(self.df_dict, library)

        transformer = uninitialized_transformers[self.transformer_name](**args)
        with pytest.raises(
            ValueError,
            match=f"key '{args['key']}' not found in dataframe columns",
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        "aggregations, expected_data",
        [
            # Test cases for "min", "max", "mean", "median",  "sum", and "count"
            (
                ["min", "max", "mean", "median", "sum", "count"],
                {
                    "a": [1, 2, 3, 4, 8],
                    "b": [2, 3, 4, 5, 9],
                    "c": ["A", "B", "A", "B", "A"],
                    "a_min": [1, 2, 1, 2, 1],
                    "a_max": [8, 4, 8, 4, 8],
                    "a_mean": [4.0, 3.0, 4.0, 3.0, 4.0],
                    "a_median": [3.0, 3.0, 3.0, 3.0, 3.0],
                    "a_sum": [12.0, 6.0, 12.0, 6.0, 12.0],
                    "a_count": [3, 2, 3, 2, 3],
                    "b_min": [2, 3, 2, 3, 2],
                    "b_max": [9, 5, 9, 5, 9],
                    "b_mean": [5.0, 4.0, 5.0, 4.0, 5.0],
                    "b_median": [4.0, 4.0, 4.0, 4.0, 4.0],
                    "b_sum": [15.0, 8.0, 15.0, 8.0, 15.0],
                    "b_count": [3, 2, 3, 2, 3],
                },
            ),
        ],
    )
    def test_transform_basic_case_outputs(
        self,
        library,
        aggregations,
        expected_data,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method aggregates rows correctly."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["aggregations"] = aggregations
        args["key"] = "c"

        df = u.dataframe_init_dispatch(self.df_dict, library)

        # transformer = transformer_setup(columns, aggregations, key, drop_original)
        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(df)

        # Create expected DataFrame using the library parameter
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Polars uses efficient types which differ from pandas here we do convert.
        # The transformer remains abstract, but tests can be specific
        if library == "polars":
            expected_df = nw.from_native(expected_df)
            expected_df = expected_df.with_columns(
                [
                    nw.col("a_count").cast(nw.UInt32),
                    nw.col("b_count").cast(nw.UInt32),
                ],
            ).to_native()

        # Compare the transformed DataFrame with the expected DataFrame using the dispatch function
        u.assert_frame_equal_dispatch(transformed_df, expected_df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_single_row(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method with a single-row DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["aggregations"] = ["min", "max", "mean", "median", "sum", "count"]
        args["key"] = "c"

        # Create a single-row DataFrame
        single_row_df_dict = {
            "a": [None],
            "b": [2],
            "c": ["A"],
        }
        single_row_df = u.dataframe_init_dispatch(single_row_df_dict, library)

        # Polars uses efficient types which differ from pandas here we do convert.
        # The transformer remains abstract, but tests can be specific
        if library == "pandas":
            single_row_df["a"] = single_row_df["a"].astype(float)
            single_row_df["b"] = single_row_df["b"].astype(float)
        elif library == "polars":
            single_row_df = nw.from_native(single_row_df)
            single_row_df = single_row_df.with_columns(
                [
                    nw.col("a").cast(nw.Float64),
                    nw.col("b").cast(nw.Float64),
                ],
            ).to_native()

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(single_row_df)

        # Expected output for a single-row DataFrame
        expected_data = {
            "a": [None],
            "b": [2],
            "c": ["A"],
            "a_min": [None],
            "a_max": [None],
            "a_mean": [None],
            "a_median": [None],
            "a_sum": [0.0],
            "a_count": [0],
            "b_min": [2.0],
            "b_max": [2.0],
            "b_mean": [2.0],
            "b_median": [2.0],
            "b_sum": [2.0],
            "b_count": [1],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Polars uses efficient types which differ from pandas here we do convert.
        # The transformer remains abstract, but tests can be specific
        if library == "pandas":
            expected_df["a"] = expected_df["a"].astype(float)
            expected_df["b"] = expected_df["b"].astype(float)
            expected_df["a_min"] = expected_df["a_min"].astype(float)
            expected_df["a_max"] = expected_df["a_max"].astype(float)
            expected_df["a_mean"] = expected_df["a_mean"].astype(float)
            expected_df["a_median"] = expected_df["a_median"].astype(float)
        elif library == "polars":
            expected_df = nw.from_native(expected_df)
            expected_df = expected_df.with_columns(
                [
                    nw.col("a").cast(nw.Float64),
                    nw.col("b").cast(nw.Float64),
                    nw.col("a_min").cast(nw.Float64),
                    nw.col("a_max").cast(nw.Float64),
                    nw.col("a_mean").cast(nw.Float64),
                    nw.col("a_median").cast(nw.Float64),
                    nw.col("a_count").cast(nw.UInt32),
                    nw.col("b_count").cast(nw.UInt32),
                ],
            ).to_native()

        u.assert_frame_equal_dispatch(transformed_df, expected_df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_with_nulls(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method with null values in the DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["aggregations"] = ["min", "max", "mean", "median", "sum", "count"]
        args["key"] = "c"

        # Create a DataFrame with null values
        df_with_nulls_dict = {
            "a": [1, None, 3, None, 8],
            "b": [None, 3, None, 5, 9],
            "c": ["A", "B", "A", "B", "A"],
        }
        df_with_nulls = u.dataframe_init_dispatch(df_with_nulls_dict, library)

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(df_with_nulls)

        # Expected output for a DataFrame with null values
        expected_data = {
            "a": [1, None, 3, None, 8],
            "b": [None, 3, None, 5, 9],
            "c": ["A", "B", "A", "B", "A"],
            "a_min": [1, None, 1, None, 1],
            "a_max": [8, None, 8, None, 8],
            "a_mean": [4.0, None, 4.0, None, 4.0],
            "a_median": [3.0, None, 3.0, None, 3.0],
            "a_sum": [12.0, 0.0, 12.0, 0.0, 12.0],
            "a_count": [3, 0, 3, 0, 3],
            "b_min": [9.0, 3.0, 9.0, 3.0, 9.0],
            "b_max": [9.0, 5.0, 9.0, 5.0, 9.0],
            "b_mean": [9.0, 4.0, 9.0, 4.0, 9.0],
            "b_median": [9.0, 4.0, 9.0, 4.0, 9.0],
            "b_sum": [9.0, 8.0, 9.0, 8.0, 9.0],
            "b_count": [1, 2, 1, 2, 1],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Polars uses efficient types which differ from pandas here we do convert.
        # The transformer remains abstract, but tests can be specific
        if library == "polars":
            expected_df = nw.from_native(expected_df)
            expected_df = expected_df.with_columns(
                [
                    nw.col("a_count").cast(nw.UInt32),
                    nw.col("b_count").cast(nw.UInt32),
                ],
            ).to_native()

        u.assert_frame_equal_dispatch(transformed_df, expected_df)
