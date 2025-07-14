from enum import Enum

import narwhals as nw
import narwhals.selectors as ncs
from beartype import beartype
from beartype.typing import Annotated, List
from beartype.vale import Is
from narwhals.typing import FrameT

from tubular.base import BaseTransformer
from tubular.mixins import DropOriginalMixin


class AggregationOptions(str, Enum):
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    SUM = "sum"
    COUNT = "count"


ListOfAggregations = Annotated[
    List,
    Is[
        lambda list_value: all(
            entry in AggregationOptions._value2member_map_ for entry in list_value
        )
    ],
]


class BaseAggregationTransformer(BaseTransformer, DropOriginalMixin):
    """Base class for aggregation transformers.

    This class provides the foundation for aggregation-based transformations,
    handling common setup tasks such as validating aggregation methods and
    managing column specifications.

    Parameters
    ----------
    columns : list[str]
        List of column names to apply the aggregation transformations to.
    aggregations : list[str]
        List of aggregation methods to apply. Valid methods include 'min', 'max',
        'mean', 'median', 'mode', 'sum', and 'count'.
    drop_original : bool, optional
        Whether to drop the original columns after transformation. Default is False.
    verbose : bool, optional
        If True, enables verbose output for debugging purposes. Default is False.

    Attributes
    ----------
    columns : list[str]
        Columns to apply the transformations to.
    aggregations : list[str]
        Aggregation methods to apply.
    drop_original : bool
        Indicator for dropping original columns.
    verbose : bool
        Indicator for verbose output.
    """

    @beartype
    def __init__(
        self,
        columns: list[str],
        aggregations: ListOfAggregations,
        drop_original: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(columns=columns, verbose=verbose)

        self.aggregations = aggregations

        self.set_drop_original_column(drop_original)
        self.polars_compatible = True

    @beartype
    def create_new_col_names(self, prefix: str) -> list[str]:
        """Automatically generate new column names based on the aggregation type.

        Parameters
        ----------
        prefix : str
            Prefix to use for generating new column names.

        Returns
        -------
        list[str]
            List of new column names with the specified prefix and aggregation type.
        """
        return [f"{prefix}_{agg}" for agg in self.aggregations]


class CategoricalAggregationOptions(str, Enum):
    MODE = "mode"
    COUNT = "count"


def mode_aggregation(df, column, key):
    # Group by the key and calculate mode for the specified column
    mode_series = (
        df.to_native()
        .groupby(key)[column]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )
    mode_series.name = f"{column}_mode"
    return nw.from_native(mode_series.reset_index())


class AggregateRowOverColumnsTransformer(BaseAggregationTransformer):
    """Transformer that aggregates rows over specified columns.

    This transformer aggregates data within specified columns, grouping by a key column,
    and can optionally drop the original columns post-transformation.

    Parameters
    ----------
    columns : list[str]
        List of column names to apply the aggregation transformations to.
    aggregations : list[str]
        List of aggregation methods to apply.
    key : str
        Column name to group by for aggregation.
    drop_original : bool, optional
        Whether to drop the original columns after transformation. Default is False.

    Attributes
    ----------
    key : str
        Column used for grouping during aggregation.
    """

    @beartype
    def __init__(
        self,
        columns: list[str],
        aggregations: ListOfAggregations,
        key: str,
        drop_original: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            columns=columns,
            aggregations=aggregations,
            drop_original=drop_original,
            verbose=verbose,
        )
        self.key = key
        self.polars_compatible = True

    @nw.narwhalify
    def transform(
        self,
        df: FrameT,
    ) -> FrameT:
        """Transforms the dataframe by aggregating rows over specified columns.

        Parameters
        ----------
        df : pd.DataFrame or pl.DataFrame
            DataFrame to transform by aggregating specified columns.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Transformed DataFrame with aggregated columns.

        Raises
        ------
        ValueError
            If the key column is not found in the DataFrame.
        """

        df = super().transform(df)
        df = nw.from_native(df)

        if self.key not in df.columns:
            msg = f"key '{self.key}' not found in dataframe columns"
            raise ValueError(msg)

        # Identify column types using ncs selectors
        numeric_columns_in_df = df.select(ncs.numeric()).columns
        non_numeric_columns = list(
            set(self.columns).difference(set(numeric_columns_in_df)),
        )

        cat_columns = set(self.columns).intersection(
            df.select(ncs.categorical()).columns,
        )
        bool_columns = set(self.columns).intersection(df.select(ncs.boolean()).columns)
        str_columns = set(self.columns).intersection(df.select(ncs.string()).columns)
        numeric_columns = df.select(ncs.numeric()).columns

        non_categorical_agg_friendly_columns = (
            set(self.columns)
            .difference(cat_columns)
            .difference(bool_columns)
            .difference(str_columns)
            .difference(numeric_columns)
        )

        # Check for incompatible aggregation methods
        numeric_aggs_requested = any(
            agg not in CategoricalAggregationOptions._value2member_map_
            for agg in self.aggregations
        )
        non_categorical_agg_friendly_columns = (
            set(self.columns)
            .difference(cat_columns)
            .difference(bool_columns)
            .difference(str_columns)
            .difference(numeric_columns)
        )

        # print("Numeric columns in DataFrame:", numeric_columns_in_df)
        # print("Non-numeric columns identified:", non_numeric_columns)
        # print("Numeric aggregations requested:", numeric_aggs_requested)

        if len(non_numeric_columns) > 0 and numeric_aggs_requested:
            raise TypeError(
                "Numeric aggregation methods requested for non-numeric columns.",
            )

        if not numeric_aggs_requested and len(non_categorical_agg_friendly_columns) > 0:
            raise TypeError(
                "Categorical aggregation methods requested for non-categorical columns.",
            )

        aggregation_dict = {col: self.aggregations for col in self.columns}

        grouped_df = df.group_by(self.key).agg(
            *(
                getattr(nw.col(col_name), agg)().alias(f"{col_name}_{agg}")
                for col_name, aggs in aggregation_dict.items()
                for agg in aggs
                if agg != "mode"
            ),
        )

        for col_name in self.columns:
            if "mode" in aggregation_dict[col_name]:
                mode_df = (
                    df.filter(~nw.col(col_name).is_null())
                    .group_by(self.key)
                    .agg(
                        nw.col(col_name).mode().alias(f"{col_name}_mode"),
                    )
                )
                grouped_df = grouped_df.join(mode_df, on=self.key, how="left")

        # Replace None with 0 for sum columns in the aggregated DataFrame
        for col_name in self.columns:
            if "sum" in aggregation_dict[col_name]:
                sum_col_name = f"{col_name}_sum"
                grouped_df = grouped_df.with_columns(
                    nw.when(nw.col(sum_col_name).is_null())
                    .then(0)
                    .otherwise(nw.col(sum_col_name))
                    .alias(sum_col_name),
                )
                grouped_df = grouped_df.with_columns(
                    nw.col(sum_col_name).cast(nw.Float64),
                )

        # Merge the aggregated results back with the original DataFrame
        df = df.join(grouped_df, on=self.key, how="left")

        # Use mixin method to drop original columns
        return self.drop_original_column(
            df,
            self.drop_original,
            self.columns,
        )
