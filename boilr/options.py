"""Package-wide options."""

from copy import deepcopy

# Define options and set default values
_options = {
    'model_print_depth': 3,
    'train_summarizer_ma_length': 1000,
    'show_progress_bar': True,
}


def set_options(**kwargs):
    """Sets specified package-wide options.

    Args:
        **kwargs: keyword arguments defining options that have to be updated.
            Options that are not specified are left unchanged.
    """

    # Check arguments
    invalid = []
    for k in kwargs:
        if k not in _options:
            invalid.append(k)
    if len(invalid) > 0:
        msg = "Invalid option names: "
        invalid = [repr(k) for k in invalid]
        msg = msg + ", ".join(invalid)
        raise ValueError(msg)

    for k, v in kwargs.items():
        if not _check_option(k, v):
            msg = "Invalid value {} for option {}".format(repr(v), repr(k))
            raise ValueError(msg)

    # Update options
    _options.update(**kwargs)


def get_option(name):
    """Returns the current value of the specified option.

    Args:
        name (str): Option name

    Returns:
        value: Option value
    """
    return _get_options(name)


def get_options():
    """Returns the current value of all options.

    Returns:
        options (dict): all packagewide options
    """
    return _get_options()


def _get_options(name=None):
    """Returns the current value of either one or all options.

    Args:
        name (str, optional): Name of the required option

    Returns:
        If name is specified, the value of the option with the specified name.
        Otherwise, dictionary containing all options.
    """
    if name is None:
        return deepcopy(_options)
    return deepcopy(_options[name])


def _check_option(name, value):
    """Check that the value is valid for the option with the given name.

    Args:
        name (str): Option name
        value: Option value

    Returns:
        True if ``value`` is a valid value for option ``name``, False
        otherwise.

    Raises:
        ValueError: Unknown option name.
    """
    if name == 'model_print_depth':
        if value is None:
            return True
        return isinstance(value, int) and value > 0
    elif name == 'train_summarizer_ma_length':
        return isinstance(value, int) and value > 0
    elif name == 'show_progress_bar':
        return isinstance(value, bool)
    else:
        raise ValueError("Unknown option name {}".format(repr(name)))
