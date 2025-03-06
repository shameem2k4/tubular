import narwhals as nw
import pandas as pd
import pytest

import tests.utils as u
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.imputers.test_BaseImputer import GenericImputerTransformTests
from tubular.imputers import ArbitraryImputer


# Dataframe used exclusively in this testing script
def create_downcast_df(library):
    """Create a dataframe with mixed dtypes to use in downcasting tests."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
    )

    return u.dataframe_init_dispatch(df, library)


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"

    def test_impute_value_type_error(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
    ):
        """Test that an exception is raised if impute_value is not an int, float or str."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["impute_value"] = [1, 2]

        with pytest.raises(
            ValueError,
            match="ArbitraryImputer: impute_value should be a single value .*",
        ):
            uninitialized_transformers[self.transformer_name](**args)


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"


class TestTransform(GenericImputerTransformTests, GenericTransformTests):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_impute_value_preserve_dtype(self, library):
        """Testing downcast dtypes of columns are preserved after imputation using the create_downcast_df dataframe.

        Explicitly setting the dtype of "a" to int8 and "b" to float32 and check if the dtype of the columns are preserved after imputation.
        """
        df = create_downcast_df(library)
        df = nw.from_native(df)
        df = df.with_columns(
            df["a"].cast(nw.Int8),
            df["b"].cast(nw.Float32),
        )

        # Imputing the dataframe
        x = ArbitraryImputer(impute_value=1, columns=["a", "b"])

        df_transformed = x.transform(df.to_native())

        # Convert both DataFrames to a common format using Narwhals
        df_transformed_common = nw.from_native(df_transformed)

        # Check if the dtype of "a" and "b" are preserved after imputation
        assert df_transformed_common["a"].dtype == df["a"].dtype
        assert df_transformed_common["b"].dtype == df["b"].dtype


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"
