def validate_type(**kwargs_with_expected_types) -> None:
    """
    Validate that each value has one of the expected types.

    Args:
        kwargs_with_expected_types:
            Mapping of parameter names to tuples containing the value
            and expected type(s).
    """
    for name, (value, expected_type) in kwargs_with_expected_types.items():
        if not isinstance(value, expected_type):
            if isinstance(expected_type, tuple):
                expected_type_name = " or ".join(expected.__name__ for expected in expected_type)
            else:
                expected_type_name = expected_type.__name__

            raise TypeError(
                f"Parameter '{name}' expected to be of type " f"{expected_type_name}, received {type(value).__name__}"
            )
