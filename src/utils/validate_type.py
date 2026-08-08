def validate_type(**kwargs_with_expected_types) -> None:
    """
    kwargs_with_expected_types: {value_name: (value, expected_type)}.
    """
    for name, (value, expected_type) in kwargs_with_expected_types.items():
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Parameter '{name}' expected to be of type {expected_type.__name__}, "
                f"received {type(value).__name__}"
            )
