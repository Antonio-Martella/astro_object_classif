import pandas as pd
import pytest

from src.utils.validate_type import validate_type


@pytest.fixture
def dummy_dataframes():
    """Fixture per oggetti di tipo Data Structure complessi."""
    return {
        "df": pd.DataFrame({"a": [1, 2]}),
        "series": pd.Series([1, 2], name="s"),
    }


@pytest.mark.parametrize(
    "param_name, val, expected_type",
    [
        ("integer_param", 42, int),
        ("float_param", 3.14, float),
        ("string_param", "astro_model", str),
        ("list_param", [1, 2, 3], list),
        ("dict_param", {"key": "value"}, dict),
        ("union_type_param_1", 10, (int, float)),
        ("union_type_param_2", 10.5, (int, float)),
    ],
)
def test_validate_type_success_basic_types(param_name, val, expected_type):
    """Testa che la validazione passi senza errori per i tipi primitivi standard."""
    validate_type(**{param_name: (val, expected_type)})


def test_validate_type_success_pandas_types(dummy_dataframes):
    df = dummy_dataframes["df"]
    series = dummy_dataframes["series"]

    validate_type(
        features=(df, pd.DataFrame),
        target=(series, pd.Series),
    )


def test_validate_type_handles_none_correctly():
    """Testa che la validazione passi senza errori per i tipi primitivi standard."""
    with pytest.raises(TypeError) as exc_info:
        validate_type(missing_val=(None, int))

    assert "Parameter 'missing_val'" in str(exc_info.value)
    assert "expected to be of type int" in str(exc_info.value)


def test_validate_type_multiple_params_mixed():
    """Testa la validazione corretta per oggetti Pandas (Data Structure)."""
    validate_type(a=(1, int), b=(3.14, float), c=("hi", str), d=([1, 2], list), e=({"key": "value"}, dict))


@pytest.mark.parametrize(
    "param_name, val, expected_type, expected_error_msg",
    [
        (
            "integer_",
            3.14,
            int,
            "Parameter 'integer_' expected to be of type int, received float",
        ),
        (
            "series_pandas",
            pd.DataFrame(),
            pd.Series,
            "Parameter 'series_pandas' expected to be of type Series, received DataFrame",
        ),
        (
            "str_",
            123,
            str,
            "Parameter 'str_' expected to be of type str, received int",
        ),
    ],
)
def test_validate_type_raises_type_error(param_name, val, expected_type, expected_error_msg):
    """
    Verifica che ogni tipo errato sollevi un TypeError distinto
    e restituisca un messaggio d'errore esplicito e utile per il log di produzione.
    """
    with pytest.raises(TypeError) as exc_info:
        validate_type(**{param_name: (val, expected_type)})

    assert expected_error_msg in str(exc_info.value)
