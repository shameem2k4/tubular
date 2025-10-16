import copy

import narwhals as nw
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

# from tests.utils import _handle_from_json, assert_frame_equal_dispatch, dataframe_init_dispatch


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

        df = create_ratio_test_df(library=library)

        transformer = uninitialized_transformers[self.transformer_name](**args)
        transformed_df = transformer.transform(df)

        # Expected output for basic division
        expected_data = {
            "a": [100, 200, 300],
            "b": [80, 150, 200],
            "a_divided_by_b": [1.25, 1.333333, 1.5],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Use Narwhals for casting and column selection
        expected_df = nw.from_native(expected_df)
        transformed_df = nw.from_native(transformed_df)
        expected_df = expected_df.with_columns(
            nw.col("a_divided_by_b").cast(getattr(nw, return_dtype)),
        )
        transformed_df = transformed_df.select(expected_df.columns)
        expected_df = expected_df.to_native()
        transformed_df = transformed_df.to_native()

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

        # Use Narwhals for casting and column selection
        expected_df = nw.from_native(expected_df)
        transformed_df = nw.from_native(transformed_df)
        expected_df = expected_df.with_columns(
            nw.col("a_divided_by_b").cast(getattr(nw, return_dtype)),
        )
        transformed_df = transformed_df.select(expected_df.columns)
        expected_df = expected_df.to_native()
        transformed_df = transformed_df.to_native()

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

        # Use Narwhals for casting and column selection
        expected_df = nw.from_native(expected_df)
        transformed_df = nw.from_native(transformed_df)
        expected_df = expected_df.with_columns(
            nw.col("a_divided_by_b").cast(getattr(nw, return_dtype)),
        )
        transformed_df = transformed_df.select(expected_df.columns)
        expected_df = expected_df.to_native()
        transformed_df = transformed_df.to_native()

        u.assert_frame_equal_dispatch(transformed_df, expected_df)


class TestRatioTransformerSerialization:
    """Tests for RatioTransformer serialization and deserialization."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "RatioTransformer"

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_ratio_transformer_to_from_json(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
        from_json,
    ):
        """Test that RatioTransformer can be serialized to JSON and reconstructed correctly."""
        # Create a sample DataFrame using the library parameter
        df_dict = {
            "a": [100, 200, 300],
            "b": [80, 150, 200],
        }
        df = u.dataframe_init_dispatch(df_dict, library)

        # Initialize the transformer
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = ["a", "b"]
        args["return_dtype"] = "Float32"
        transformer = uninitialized_transformers[self.transformer_name](**args)

        # Handle serialization and deserialization
        transformer = u._handle_from_json(transformer, from_json)

        # Transform the DataFrame
        transformed_df = transformer.transform(df)

        # Expected output for basic division
        expected_data = {
            "a": [100, 200, 300],
            "b": [80, 150, 200],
            "a_divided_by_b": [1.25, 1.333333, 1.5],
        }
        expected_df = u.dataframe_init_dispatch(expected_data, library)

        # Use Narwhals for casting and column selection
        expected_df = nw.from_native(expected_df)
        transformed_df = nw.from_native(transformed_df)
        expected_df = expected_df.with_columns(
            nw.col("a_divided_by_b").cast(nw.Float32),
        )
        transformed_df = transformed_df.select(expected_df.columns)
        expected_df = expected_df.to_native()
        transformed_df = transformed_df.to_native()

        u.assert_frame_equal_dispatch(transformed_df, expected_df)
