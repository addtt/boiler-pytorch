from copy import deepcopy


_options = {
    'model_print_depth': 3,
    'train_summarizer_ma_length': 1000,
}


def set_options(**kwargs):
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
    return _get_options(name)


def get_options():
    return _get_options()


def _get_options(name=None):
    if name is None:
        return deepcopy(_options)
    return deepcopy(_options[name])


def _check_option(name, value):
    if name == 'model_print_depth':
        if value is None:
            return True
        return isinstance(value, int) and value > 0
    elif name == 'train_summarizer_ma_length':
        return isinstance(value, int) and value > 0
    else:
        raise ValueError("Unknown option name {}".format(repr(name)))
