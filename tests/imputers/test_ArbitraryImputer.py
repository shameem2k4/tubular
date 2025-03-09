import narwhals as nw
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
@pytest.fixture(params=["pandas", "polars"])
def downcast_df(request):
    """
    Fixture that returns a DataFrame with columns suitable for downcasting
    for both pandas and polars.
    """
    data = {
        "a": [1, 2, 3, 4, 5],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
    library = request.param

    return u.dataframe_init_dispatch(data, library)


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

    def test_impute_value_preserve_dtype(self, downcast_df):
        """Test that downcast dtypes are preserved after imputation."""

        df_nw = nw.from_native(downcast_df)

        df_nw = df_nw.with_columns(
            df_nw["a"].cast(nw.Int8),
            df_nw["b"].cast(nw.Float32),
        )

        x = ArbitraryImputer(impute_value=1, columns=["a", "b"])
        df_transformed_native = x.transform(df_nw.to_native())

        df_transformed_nw = nw.from_native(df_transformed_native)

        assert df_transformed_nw["a"].dtype == df_nw["a"].dtype
        assert df_transformed_nw["b"].dtype == df_nw["b"].dtype


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"
