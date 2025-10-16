import copy

import polars as pl
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests import utils as u
from tests.numeric.test_BaseNumericTransformer import (
    BaseNumericTransformerInitTests,
    BaseNumericTransformerTransformTests,
)
from tests.test_data import (
    create_ratio_test_df,  # Ensure this function generates the correct test data
)


class TestRatioTransformerInit(BaseNumericTransformerInitTests):
    """Tests for init method in RatioTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "RatioTransformer"

    @pytest.mark.parametrize("columns", (["a"], ["a", "b", "c"], None, "a"))
    def test_columns_arg_errors(
        self,
        columns,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = columns
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("return_dtype", ["InvalidType", None])
    def test_return_dtype_arg_errors(
        self,
        return_dtype,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["return_dtype"] = return_dtype
        with pytest.raises(BeartypeCallHintParamViolation):
            uninitialized_transformers[self.transformer_name](**args)


class TestRatioTransformerTransform(BaseNumericTransformerTransformTests):
    """Tests for transform method in RatioTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "RatioTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("return_dtype", ["Float32", "Float64"])
    def test_transform_basic_case_outputs(
        self,
        library,
        return_dtype,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method performs division correctly."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["return_dtype"] = return_dtype

        df = create_ratio_test_df(library=library)  # Use the correct test data function

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(df)

        # Expected output for basic division
        expected_data = {
            "a": [100, 200, 300],
            "b": [80, 150, 200],
            "a_divided_by_b": [1.25, 1.333333, 1.5],  # Ensure consistent precision
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Cast the expected column to match the return_dtype
        if library == "pandas":
            expected_df["a_divided_by_b"] = expected_df["a_divided_by_b"].astype(
                return_dtype.lower(),
            )
            transformed_df = transformed_df[
                expected_df.columns
            ]  # Reorder columns using indexing
        else:  # For polars
            expected_df = expected_df.with_columns(
                expected_df["a_divided_by_b"].cast(getattr(pl, return_dtype)),
            )
            transformed_df = transformed_df.select(expected_df.columns)

        u.assert_frame_equal_dispatch(transformed_df, expected_df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("return_dtype", ["Float32", "Float64"])
    def test_single_row(
        self,
        library,
        return_dtype,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method with a single-row DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["return_dtype"] = return_dtype

        # Create a single-row DataFrame
        single_row_df_dict = {
            "a": [100],
            "b": [80],
        }
        single_row_df = u.dataframe_init_dispatch(single_row_df_dict, library)

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(single_row_df)

        # Expected output for a single-row DataFrame
        expected_data = {
            "a": [100],
            "b": [80],
            "a_divided_by_b": [1.25],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Cast the expected column to match the return_dtype
        if library == "pandas":
            expected_df["a_divided_by_b"] = expected_df["a_divided_by_b"].astype(
                return_dtype.lower(),
            )
            transformed_df = transformed_df[
                expected_df.columns
            ]  # Reorder columns using indexing
        else:  # For polars
            expected_df = expected_df.with_columns(
                expected_df["a_divided_by_b"].cast(getattr(pl, return_dtype)),
            )
            transformed_df = transformed_df.select(expected_df.columns)

        u.assert_frame_equal_dispatch(transformed_df, expected_df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("return_dtype", ["Float32", "Float64"])
    def test_with_nulls(
        self,
        library,
        return_dtype,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method with null values in the DataFrame."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]
        args["return_dtype"] = return_dtype

        # Create a DataFrame with null values
        df_with_nulls_dict = {
            "a": [100, None, 300],
            "b": [80, 150, None],
        }
        df_with_nulls = u.dataframe_init_dispatch(df_with_nulls_dict, library)

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(df_with_nulls)

        # Expected output for a DataFrame with null values
        expected_data = {
            "a": [100, None, 300],
            "b": [80, 150, None],
            "a_divided_by_b": [1.25, None, None],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Cast the expected column to match the return_dtype
        if library == "pandas":
            expected_df["a_divided_by_b"] = expected_df["a_divided_by_b"].astype(
                return_dtype.lower(),
            )
            transformed_df = transformed_df[
                expected_df.columns
            ]  # Reorder columns using indexing
        else:  # For polars
            expected_df = expected_df.with_columns(
                expected_df["a_divided_by_b"].cast(getattr(pl, return_dtype)),
            )
            transformed_df = transformed_df.select(expected_df.columns)

        u.assert_frame_equal_dispatch(transformed_df, expected_df)
