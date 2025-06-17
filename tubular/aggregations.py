import pandas as pd
from beartype import beartype
from mixins import DropOriginalMixin


class BaseAggregationTransformer:
    """Base class for aggregation transformers."""

    @beartype
    def __init__(
        self,
        columns: list[str],
        aggregations: list[str],
        drop_original: bool = False,
        level: str = "row",
    ) -> None:  # Added return type annotation
        valid_aggregations = {"min", "max", "mean", "median", "mode", "sum", "count"}

        if not all(agg in valid_aggregations for agg in aggregations):
            msg = f"aggregations must be a list containing any of {valid_aggregations}"  # Assign message to variable
            raise ValueError(msg)

        if level not in {"row", "column"}:
            msg = "level must be 'row' or 'column'"  # Assign message to variable
            raise ValueError(msg)

        self.columns = columns
        self.aggregations = aggregations
        self.drop_original = drop_original
        self.level = level

    @beartype
    def create_new_col_names(self, prefix: str) -> list[str]:
        """Automatically generate new column names based on the aggregation type."""
        return [f"{prefix}_{agg}" for agg in self.aggregations]


class AggregateRowOverColumnsTransformer(DropOriginalMixin, BaseAggregationTransformer):
    """Transformer that aggregates rows over specified columns."""

    @beartype
    def __init__(
        self,
        columns: list[str],
        aggregations: list[str],
        key: str,
        drop_original: bool = False,
    ) -> None:  # Added return type annotation
        super().__init__(
            columns=columns,
            aggregations=aggregations,
            drop_original=drop_original,
            level="row",
        )
        self.key = key
        self.set_drop_original_column(drop_original)

    @beartype
    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:  # Added return type annotation
        """Transforms the dataframe by aggregating rows over specified columns."""
        if self.key not in df.columns:
            msg = f"key '{self.key}' not found in dataframe columns"  # Assign message to variable
            raise ValueError(msg)

        aggregation_dict = {col: self.aggregations for col in self.columns}
        grouped_df = df.groupby(self.key).agg(aggregation_dict)

        # Flatten MultiIndex columns
        grouped_df.columns = [
            "_".join(col).strip() for col in grouped_df.columns.to_numpy()
        ]  # Use .to_numpy() instead of .values

        # Reset index to merge with original DataFrame
        grouped_df = grouped_df.reset_index()

        # Merge the aggregated results back with the original DataFrame
        df = df.merge(grouped_df, on=self.key, how="left")

        # Use mixin method to drop original columns
        return self.drop_original_column(
            df,
            self.drop_original,
            self.columns,
        )  # Directly return without unnecessary assignment
