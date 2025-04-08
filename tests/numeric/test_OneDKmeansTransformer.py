import pytest

from tests.numeric.test_BaseNumericTransformer import (
    BaseNumericTransformerFitTests,
    BaseNumericTransformerInitTests,
)
from tubular.numeric import OneDKmeansTransformer


class TestInit(BaseNumericTransformerInitTests):
    """Tests for OneDKmeansTransformer.init()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneDKmeansTransformer"

    def test_new_column_name_type_error(self):
        """Test that an exception is raised if new_column_name is not a str."""
        with pytest.raises(
            TypeError,
            match=r"""OneDKmeansTransformer: new_column_name should be a str but got type \<class 'int'\>""",
        ):
            OneDKmeansTransformer(column="b", new_column_name=1)

    def test_column_type_error(self):
        """Test that an exception is raised if column is not a str."""
        with pytest.raises(
            TypeError,
            match="OneDKmeansTransformer: column arg should be a single str giving the column to group.",
        ):
            OneDKmeansTransformer(column=1, new_column_name="a")

    def test_n_clusters_type_error(self):
        """Test that an exception is raised if n_clusters is not an int."""
        with pytest.raises(
            TypeError,
            match=r"""OneDKmeansTransformer: n_clusters should be a int but got type \<class 'str'\>""",
        ):
            OneDKmeansTransformer(column="a", new_column_name="b", n_clusters="c")

    def test_n_init_type_error(self):
        """Test that an exception is raised if n_init is not an int or 'auto'."""
        with pytest.raises(
            TypeError,
            match=r"""OneDKmeansTransformer: n_init should be 'auto' or int but got type \<class 'str'\>""",
        ):
            OneDKmeansTransformer(column="a", new_column_name="b", n_init="c")

    def test_kmeans_kwargs_type_error(self):
        """Test that an exception is raised if kmeans_kwargs is not a dict."""
        with pytest.raises(
            TypeError,
            match=r"""OneDKmeansTransformer: kmeans_kwargs should be a dict but got type \<class 'int'\>""",
        ):
            OneDKmeansTransformer(column="b", new_column_name="a", kmeans_kwargs=1)

    def test_kmeans_kwargs_key_type_error(self):
        """Test that an exception is raised if kmeans_kwargs has keys which are not str."""
        with pytest.raises(
            TypeError,
            match=r"""OneDKmeansTransformer: unexpected type \(\<class 'int'\>\) for kmeans_kwargs key in position 1, must be str""",
        ):
            OneDKmeansTransformer(
                new_column_name="a",
                column="b",
                kmeans_kwargs={"a": 1, 2: "b"},
            )


class TestFit(BaseNumericTransformerFitTests):
    """Tests for OneDKmeansTransformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneDKmeansTransformer"
