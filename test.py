import narwhals as nw
import polars as pl

df = pl.DataFrame({"a": [None, "a", "b", "d"]})
df = nw.from_native(df)
df = df.with_columns(nw.col("a").cast(nw.Categorical))
print(df["a"].to_native())
df = df.with_columns(nw.col("a").cast(nw.Enum(["a", "b", "c"])))
