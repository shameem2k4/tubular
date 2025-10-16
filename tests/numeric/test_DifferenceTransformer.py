import copy

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from tests import utils as u
from tests.numeric.test_BaseNumericTransformer import (
    BaseNumericTransformerInitTests,
    BaseNumericTransformerTransformTests,
)
from tests.test_data import create_difference_test_df


class TestDifferenceTransformerInit(BaseNumericTransformerInitTests):
    """Tests for init method in DifferenceTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DifferenceTransformer"

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


class TestDifferenceTransformerTransform(BaseNumericTransformerTransformTests):
    """Tests for transform method in DifferenceTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DifferenceTransformer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_invalid_columns_error(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that an error is raised if the specified columns are not found."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["missing_col1", "missing_col2"]

        df = create_difference_test_df(library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)
        with pytest.raises(
            ValueError,
            match=f"DifferenceTransformer: variables {set(args['columns'])} not in X",
        ):
            transformer.transform(df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_transform_basic_case_outputs(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test transform method performs subtraction correctly."""
        args = copy.deepcopy(minimal_attribute_dict[self.transformer_name])
        args["columns"] = ["a", "b"]

        df = create_difference_test_df(library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(df)

        # Expected output for basic subtraction
        expected_data = {
            "a": [100, 200, 300],
            "b": [80, 150, 200],
            "a_minus_b": [20, 50, 100],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

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
            "a_minus_b": [20],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

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
            "a_minus_b": [20, None, None],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        u.assert_frame_equal_dispatch(transformed_df, expected_df)
