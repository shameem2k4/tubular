from typing import Union

import narwhals as nw
import pandas as pd
import polars as pl

DataFrame = Union[
    pd.DataFrame,
    pl.DataFrame,
    pl.LazyFrame,
    nw.DataFrame,
    nw.LazyFrame,
]

Series = Union[
    pd.Series,
    pl.Series,
    nw.Series,
]

NumericTypes = [
    nw.Int8,
    nw.Int16,
    nw.Int32,
    nw.Int64,
    nw.Float64,
    nw.Float32,
    nw.UInt8,
    nw.UInt16,
    nw.UInt32,
    nw.UInt64,
    nw.UInt128,
]
