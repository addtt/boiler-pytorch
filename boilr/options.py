from copy import deepcopy

_print_options = {
    'model_print_depth': 3,
}

def set_options(**kwargs):
    # Check arguments
    invalid = []
    for k in kwargs:
        if k not in _print_options:
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
    _print_options.update(**kwargs)

def get_options(name=None):
    if name is None:
        return deepcopy(_print_options)
    return deepcopy(_print_options[name])

def _check_option(name, value):
    if name == 'model_print_depth':
        if value is None:
            return True
        if not isinstance(value, int):
            return False
        if value < 1:
            return False
        return True
    raise ValueError("Unknown option name {}".format(repr(name)))
