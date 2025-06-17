import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from pandas.api.types import CategoricalDtype
from pandas.testing import assert_series_equal as assert_series_equal_pandas
from polars.testing import assert_series_equal as assert_series_equal_polars

from tubular._utils import (
    new_narwhals_series_with_best_pandas_types,  # noqa: PLC2701, importing from private module to test
)


class TestNewNarwhalsSeriesWithBestPandasTypes:
    @pytest.mark.parametrize(
        ("values", "dtype"),
        [
            ([1, 0, 0], "Int64"),
            ([1, 0, None], "Float64"),
            (["a", "b", None], "String"),
            ((["a", None, "c"], "Categorical")),
            ([True, False, True], "Boolean"),
            ([True, False, None], "Boolean"),
        ],
    )
    def test_polars_unaffected(self, values, dtype):
        "test that polars Series are initialised as usual"
        name = "a"
        output = new_narwhals_series_with_best_pandas_types(
            name=name,
            values=values,
            backend="polars",
            dtype=getattr(nw, dtype),
        )

        expected = pl.Series(
            name=name,
            values=values,
            dtype=getattr(pl, dtype),
        )

        assert_series_equal_polars(expected, output.to_native())

    @pytest.mark.parametrize(
        ("values", "polars_dtype", "expected_pandas_dtype"),
        [
            ([1, 0, 0], "Int64", "Int64"),
            ([1, 0, None], "Float64", "Float64"),
            ([True, False, True], "Boolean", "boolean"),
            ([True, False, None], "Boolean", "boolean"),
        ],
    )
    def test_pandas_output(self, values, polars_dtype, expected_pandas_dtype):
        "test that polars Series are initialised as usual"
        name = "a"
        output = new_narwhals_series_with_best_pandas_types(
            name=name,
            values=values,
            backend="pandas",
            dtype=getattr(nw, polars_dtype),
        )

        expected = pd.Series(
            name=name,
            data=values,
            dtype=expected_pandas_dtype,
        )

        assert_series_equal_pandas(expected, output.to_native())

    def test_pandas_output_category(self):
        """test that polars Series are initialised as usual
        for category type which requires special handling"""
        name = "a"
        categories = ["a", "b", "c"]
        cat_type = CategoricalDtype(
            pd.Series(data=categories, dtype="string"),
            ordered=False,
        )
        output = new_narwhals_series_with_best_pandas_types(
            name=name,
            values=categories,
            backend="pandas",
            dtype=nw.Categorical,
        )

        expected = pd.Series(
            name=name,
            data=categories,
            dtype=cat_type,
        )

        assert_series_equal_pandas(expected, output.to_native())

    def test_pandas_output_string(self):
        """test that polars Series are initialised as usual
        for string type which requires special handling"""
        name = "a"
        values = ["a", None, "c"]
        output = new_narwhals_series_with_best_pandas_types(
            name=name,
            values=values,
            backend="pandas",
            dtype=nw.String,
        )

        expected = pd.Series(
            name=name,
            data=values,
            dtype="string",
        ).fillna(pd.NA)

        print(expected)
        print(output.to_native())
        assert_series_equal_pandas(expected, output.to_native())
